#!/usr/bin/env python3
"""Generate the known-answer vectors for pith-inflate.

Every expected plaintext in the generated file comes from Python's zlib, which
is an implementation this workspace does not contain. Round-tripping the crate
against itself would prove nothing; round-tripping against zlib proves the
decoder agrees with an independent one.

Regenerate with:

    python gen_vectors.py

and commit the result together with the regeneration. The output is
deterministic: the only randomness is a seeded PRNG.

The generator also records, for each vector, which DEFLATE block type it
actually exercises - read from the stream itself, not assumed from the
compression level - so the suite can assert that all three block types are
covered instead of hoping they are.
"""

import gzip
import zlib
import random
import struct
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "tests" / "vectors.rs"

# Block type is the low 3 bits of the first byte of a raw DEFLATE stream.
BLOCK_NAMES = {0: "stored", 1: "fixed", 2: "dynamic"}


def raw_block_type(compressed: bytes) -> int:
    """The first block's type, read from the stream's first bits.

    DEFLATE packs bits least-significant first: BFINAL is bit 0, BTYPE is
    bits 1 and 2. Reading bits 0..2 as one field would fold BFINAL into the
    type and misreport every final fixed block as the reserved type.
    """
    if not compressed:
        raise ValueError("empty deflate stream")
    btype = (compressed[0] >> 1) & 0b11
    if btype == 3:
        raise ValueError("first block is the reserved type")
    return btype

def zlib_header(compressed: bytes) -> bytes:
    cmf, flg = compressed[0], compressed[1]
    assert cmf & 0x0F == 8, "not deflate"
    assert (cmf << 8 | flg) % 31 == 0, "bad header check bits"
    flevel = flg >> 6
    fdict = bool(flg & 0x20)
    return f"cmf={cmf:#04x} flg={flg:#04x} flevel={flevel} fdict={fdict}"


# --- the plaintexts -------------------------------------------------------
# Chosen so that each one stresses a different decoder path.
rnd = random.Random(20261002)

PLAINS = []


def add(name: str, data: bytes, why: str) -> None:
    PLAINS.append((name, data, why))


add("empty", b"", "the empty payload; a stored block with LEN 0")
add("one_byte", b"A", "the shortest non-empty payload")
add("short_run", b"a" * 10, "a single back-reference whose distance exceeds the match")
add("overlap_run", b"ab" * 50, "distance 2 with length 100: the copy must overlap itself")
add("two_symbols", b"ab", "a distance tree with only one usable code")
add("text", (b"the quick brown fox jumps over the lazy dog. " * 12), "repetitive prose: long matches, one dynamic block")
add("incompressible", bytes(rnd.randrange(256) for _ in range(300)), "random bytes: forces stored blocks even at level 9")
add("mixed", bytes(rnd.randrange(256) for _ in range(200)) + b"z" * 400 + bytes(rnd.randrange(256) for _ in range(200)),
    "random bookends around a long run: stored block then a large match")
add("long_repeat", (b"the same forty two bytes over and over!!\n" * 400), "one match repeated past 32 KiB of output")
add("large", bytes((i * 7919) % 256 for i in range(70000)), "70 KB of structured bytes spanning many blocks")

# --- the malformed streams ------------------------------------------------
# Hand-edited from known-good streams so each has exactly one defect.

BAD = []


def add_bad(name: str, data: bytes, kind: str, why: str) -> None:
    BAD.append((name, data, kind, why))


_good_stored = zlib.compress(b"stored payload" * 8, 0)
add_bad("stored_nlen_mismatch", _good_stored[:4] + bytes([0xFF, 0xFF]) + _good_stored[6:], "BadValue",
        "NLEN is not the ones' complement of LEN")

# BFINAL=1, BTYPE=3. BTYPE occupies bits 1-2, so the low three bits must be
# 0b111; 0b101 sets BTYPE=2 and would exercise a truncated dynamic header
# instead. The raw_block_type assert further down is what would catch a
# repeat of that mistake.
_reserved = bytes([0b00000111, 0b00000000])
assert ((_reserved[0] >> 1) & 0b11) == 3, "reserved vector does not actually carry BTYPE=3"
add_bad("reserved_block_type", _reserved, "BadValue", "BTYPE 3 is reserved and must be refused")

_trunc = zlib.compress(b"x" * 500, 9)
add_bad("truncated_midstream", _trunc[: len(_trunc) // 2], "Truncated", "stream ends mid-block")

_badcrc = bytearray(zlib.compress(b"checksum me" * 40, 6))
_badcrc[-1] ^= 0xFF
add_bad("bad_adler32", bytes(_badcrc), "BadValue", "the Adler-32 trailer does not match the output")

# 0x7801 is 31 * 991, so it is a perfectly valid header - zlib emits it at
# level 0. 0x7802 leaves remainder 1 and is what the name claims.
assert 0x7801 % 31 == 0, "0x7801 is a valid zlib header after all"
add_bad("zlib_bad_header_checkbits", bytes([0x78, 0x02]), "InvalidMagic",
        "CMF/FLG fails the modulo-31 check, so this is not a zlib stream")
add_bad("zlib_bad_method", bytes([0x79, 0xBB]), "InvalidMagic", "compression method is 9, not 8")

# FDICT is set on a header whose check bits are already valid, so this vector
# has exactly one defect. 0x78bc would fail modulo 31 as well, which would make
# the vector test two things at once and leave FDICT-vs-FCHECK ordering
# untestable.
assert 0x78BB % 31 == 0
_fdict = bytearray(zlib.compress(b"preset dict" * 20, 6))
_fdict[1] = 0xBB
add_bad("zlib_preset_dictionary", bytes(_fdict), "Unsupported", "FDICT set: preset dictionaries are refused")

add_bad("empty_input", b"", "Truncated", "no bytes at all")

# A genuine back-reference reaching before the start of the output, packed bit
# by bit rather than guessed at: BFINAL=1, BTYPE=fixed, one literal 'A', then
# length 3 at distance 2 - but only one byte exists, so the copy underruns.
#
# The previous version of this vector was five hand-written bytes that turned
# out to be a stored block with a bad NLEN, so it silently duplicated the
# first vector while claiming to test something else.
class _Bits:
    """DEFLATE packs bits least-significant first; Huffman codes go in MSB first."""

    def __init__(self) -> None:
        self.bits: list[int] = []

    def put(self, code: int, length: int) -> "_Bits":
        for i in range(length - 1, -1, -1):
            self.bits.append((code >> i) & 1)
        return self

    def put_lsb(self, value: int, length: int) -> "_Bits":
        """Write a non-Huffman field, which the stream reads least-bit first.

        Only Huffman codes go in most-significant-bit first. BFINAL and BTYPE
        are plain fields, so writing 1 as `01` MSB-first makes the reader see
        BTYPE 2 - a dynamic block, and a stream that then fails for a reason
        unrelated to what it was built to test.
        """
        for i in range(length):
            self.bits.append((value >> i) & 1)
        return self

    def bytes(self) -> bytes:
        out = bytearray()
        for i in range(0, len(self.bits), 8):
            byte = 0
            for j, bit in enumerate(self.bits[i:i + 8]):
                byte |= bit << j
            out.append(byte)
        return bytes(out)


def _fixed_literal(sym: int) -> tuple[int, int]:
    """(code, bit length) for a literal under the fixed Huffman table."""
    if 0 <= sym <= 143:
        return 0b00110000 + sym, 8
    if 144 <= sym <= 255:
        return 0b110010000 + sym - 144, 9
    if 256 <= sym <= 279:
        return sym - 256, 7
    return 0b11000000 + sym - 280, 8


_w = _Bits().put_lsb(1, 1).put_lsb(0b01, 2)      # BFINAL=1, BTYPE=fixed
_lit, _litlen = _fixed_literal(ord("A"))
_w.put(_lit, _litlen)
_lc, _lclen = _fixed_literal(257)                # length code 257 -> length 3
_w.put(_lc, _lclen)
_w.put(0b00001, 5)                               # distance code 1 -> distance 2
_hand = _w.bytes()
add_bad("distance_before_start", _hand, "BadValue",
        "a back-reference at distance 2 when only one byte has been produced")

# --- gzip containers (RFC 1952) -------------------------------------------
# Valid members are the committed fixture files in tests/fixtures/gzip/
# (see their PROVENANCE.md); this script reads the bytes and inlines them
# as hex, so the corpus stays self-contained while the fixture files stay
# canonical. Malformed members are one-defect mutations of a crafted
# good member, built here so each names exactly the check it defeats.

GZIP_FIXTURES = HERE / "tests" / "fixtures" / "gzip"

GZ_FTEXT, GZ_FHCRC, GZ_FEXTRA, GZ_FNAME, GZ_FCOMMENT = 0x01, 0x02, 0x04, 0x08, 0x10

GZIP = []


def add_gzip(name: str, fixture: str, why: str) -> None:
    GZIP.append((name, fixture, why))


for _stem, _why in [
    ("hello", "a plain gzip(1) member: fixed header, no optional fields"),
    ("empty", "the empty payload through a full gzip member (empty DEFLATE, zero CRC-32, zero ISIZE)"),
    ("two-members", "two members concatenated: output is the concatenation, each member validated alone"),
    ("binary", "incompressible member: stored blocks inside the gzip container"),
    ("fextra", "FEXTRA: the XLEN-prefixed extra field is skipped byte-exactly"),
    ("fname", "FNAME: the NUL-terminated original name is skipped"),
    ("fcomment", "FCOMMENT: the NUL-terminated comment is skipped"),
    ("fhcrc", "FHCRC: the header CRC16 is validated through pith-digest's crc32"),
    ("all-flags", "FTEXT|FEXTRA|FNAME|FCOMMENT|FHCRC at once: every optional field in one header"),
]:
    add_gzip(f"gzip/{_stem}", _stem + ".gz", _why)

GZIP_BAD = []


def add_gzip_bad(name: str, data: bytes, kind: str, why: str) -> None:
    GZIP_BAD.append((name, data, kind, why))


def _gzip_member(payload: bytes, flg: int = 0, extra: bytes = b"", name: bytes = b"",
                 comment: bytes = b"", fhcrc_value: int | None = None) -> bytes:
    """The crafted-member recipe of tests/fixtures/gzip/make_fixtures.py.

    MTIME 0, XFL 2, OS 3, raw DEFLATE body. `fhcrc_value` overrides the
    header CRC16 for the corrupt-header mutation only.
    """
    head = bytearray(b"\x1f\x8b\x08")
    head.append(flg)
    head += (0).to_bytes(4, "little")
    head.append(2)
    head.append(3)
    if flg & GZ_FEXTRA:
        head += len(extra).to_bytes(2, "little")
        head += extra
    if flg & GZ_FNAME:
        head += name + b"\x00"
    if flg & GZ_FCOMMENT:
        head += comment + b"\x00"
    if flg & GZ_FHCRC:
        crc16 = (zlib.crc32(bytes(head)) & 0xFFFF) if fhcrc_value is None else fhcrc_value
        head += crc16.to_bytes(2, "little")
    body = zlib.compressobj(9, zlib.DEFLATED, -15)
    body_c = body.compress(payload) + body.flush()
    trailer = (zlib.crc32(payload) & 0xFFFFFFFF).to_bytes(4, "little")
    trailer += (len(payload) & 0xFFFFFFFF).to_bytes(4, "little")
    return bytes(head) + body_c + trailer


_gz_good = _gzip_member(b"the gzip bad-vector payload")

_flip = bytearray(_gz_good)
_flip[1] = 0x8C  # ID2, part of the 1f 8b magic
add_gzip_bad("gzip_bad_magic", bytes(_flip), "InvalidMagic",
             "ID2 is not 0x8b, so this is not a gzip stream")

_flip = bytearray(_gz_good)
_flip[2] = 0x09  # CM: compression method 9
add_gzip_bad("gzip_bad_method", bytes(_flip), "InvalidMagic",
             "compression method is 9, not 8")

_flip = bytearray(_gz_good)
_flip[3] |= 0x20  # FLG bit 5 is reserved and must be zero
add_gzip_bad("gzip_reserved_flags", bytes(_flip), "InvalidMagic",
             "a reserved FLG bit is set")

_crc16 = bytearray(
    _gzip_member(b"FHCRC payload", flg=GZ_FHCRC, fhcrc_value=0x0000)
)
assert _crc16[10:12] == b"\x00\x00", "the FHCRC override did not land where expected"
add_gzip_bad("gzip_bad_fhcrc", bytes(_crc16), "BadValue",
             "the FHCRC check does not match CRC-32 of the header bytes")

add_gzip_bad("gzip_truncated_header", _gz_good[:5], "Truncated",
             "the stream ends inside the ten-byte fixed header")

_fname = bytearray(_gzip_member(b"FNAME payload", flg=GZ_FNAME, name=b"long-name.txt"))
add_gzip_bad("gzip_truncated_fname", bytes(_fname[:10 + 8]), "Truncated",
             "the stream ends inside the NUL-terminated FNAME")

_extra = bytearray(_gzip_member(
    b"FEXTRA payload", flg=GZ_FEXTRA, extra=b"\x70\x68\x04\x00\x01\x02\x03\x04"
))
add_gzip_bad("gzip_truncated_fextra", bytes(_extra[:12 + 2]), "Truncated",
             "the stream ends inside the XLEN extra-field bytes")

_crcflip = bytearray(_gz_good)
_crcflip[-5] ^= 0xFF  # CRC-32 trailer, one byte
add_gzip_bad("gzip_bad_crc32", bytes(_crcflip), "BadValue",
             "the CRC-32 trailer does not match the output")

_isize = bytearray(_gz_good)
_isize[-1] ^= 0x01  # ISIZE trailer, one bit
add_gzip_bad("gzip_bad_isize", bytes(_isize), "BadValue",
             "the ISIZE trailer does not match the output length")

add_gzip_bad("gzip_truncated_trailer", _gz_good[:-2], "Truncated",
             "the stream ends inside the CRC-32/ISIZE trailer")

add_gzip_bad("gzip_empty_input", b"", "Truncated", "no bytes at all")

# --- emit -----------------------------------------------------------------

def hexlit(data: bytes) -> str:
    return '"' + data.hex() + '"'


def rust_str(s: str) -> str:
    """A quoted, escaped Rust string literal.

    Escaping without the surrounding quotes emitted `name: empty` instead of
    `name: "empty"`, which is a syntax error rather than a bad value, so it is
    worth stating in the docstring that the quotes are part of the contract.
    """
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


lines = [
    "//! Known-answer vectors for `pith-inflate`.",
    "//!",
    "//! Generated - see `gen_vectors.py` in the crate root for the regenerating",
    "//! command and the provenance of every expected value. Every plaintext here",
    "//! came from Python's zlib, an implementation outside this workspace.",
    "//!",
    "//! The workspace denies `missing_docs` and that lint reaches test targets,",
    "//! so every item below, including each struct field, carries a doc comment.",
    "//! A plain `//` banner is not a crate doc and does not satisfy it.",
    "//!",
    "//! This is a data module, not a test. The harness includes it through",
    "//! `#[path = \"vectors.rs\"]`, but Cargo also compiles the file as its own",
    "//! test target, where nothing references these items, so `dead_code` fires",
    "//! under the workspace's `-D warnings`. The allow below covers that build",
    "//! and is scoped to this file rather than to the crate.",
    "#![allow(dead_code)]",
    "",
    "/// A known-answer vector: `compressed` must inflate to `plain`.",
    "pub struct Vector {",
    "    /// The vector's name, which is also the test's name.",
    "    pub name: &'static str,",
    "    /// What this vector is here to exercise.",
    "    pub why: &'static str,",
    "    /// The compressed bytes, hex encoded.",
    "    pub compressed: &'static str,",
    "    /// The expected output, hex encoded.",
    "    pub plain: &'static str,",
    "    /// The DEFLATE block type the stream actually uses: `stored`,",
    "    /// `fixed` or `dynamic`, read out of the stream rather than assumed",
    "    /// from the compression level.",
    "    pub block: &'static str,",
    "}",
    "",
    "/// A stream with exactly one defect, and the error it must produce.",
    "pub struct BadVector {",
    "    /// The vector's name, which is also the test's name.",
    "    pub name: &'static str,",
    "    /// What is wrong with this stream.",
    "    pub why: &'static str,",
    "    /// The malformed bytes, hex encoded.",
    "    pub input: &'static str,",
    "    /// The `Error` variant the decoder must return, by name.",
    "    pub kind: &'static str,",
    "}",
    "",
]

lines.append("/// Every valid vector, in both raw DEFLATE and zlib framing.")
lines.append("pub const VECTORS: &[Vector] = &[")
seen_blocks = set()
for name, data, why in PLAINS:
    raw = zlib.compressobj(9, zlib.DEFLATED, -15)
    raw_c = raw.compress(data) + raw.flush()
    z = zlib.compress(data, 9)
    bt = raw_block_type(raw_c)
    seen_blocks.add(bt)
    hdr = zlib_header(z)
    lines.append("    Vector {")
    lines.append(f"        name: {rust_str(name)},")
    lines.append(f"        why: {rust_str(why)},")
    lines.append(f"        compressed: {hexlit(raw_c)},")
    lines.append(f"        plain: {hexlit(data)},")
    lines.append(f"        block: {rust_str(BLOCK_NAMES[bt])},")
    lines.append("    },")
    lines.append("    Vector {")
    lines.append(f"        name: {rust_str(name + '/zlib')},")
    lines.append(f"        why: {rust_str(why)},")
    lines.append(f"        compressed: {hexlit(z)},")
    lines.append(f"        plain: {hexlit(data)},")
    lines.append(f"        block: {rust_str(BLOCK_NAMES[bt])},")
    lines.append("    },")
    print(f"  {name:16s} raw={len(raw_c):6d}B out={len(data):6d}B block={BLOCK_NAMES[bt]:8s} zlib={hdr}")

lines.append("];")
lines.append("")
lines.append("/// Every malformed stream.")
lines.append("pub const BAD_VECTORS: &[BadVector] = &[")
for name, data, kind, why in BAD:
    lines.append("    BadVector {")
    lines.append(f"        name: {rust_str(name)},")
    lines.append(f"        why: {rust_str(why)},")
    lines.append(f"        input: {hexlit(data)},")
    lines.append(f"        kind: {rust_str(kind)},")
    lines.append("    },")
    print(f"  BAD {name:24s} {len(data):4d}B -> {kind}")
lines.append("];")
lines.append("")
lines.append("/// A gzip (RFC 1952) container vector: `input` is the full member")
lines.append("/// stream - exactly the bytes of the committed fixture file named")
lines.append("/// by `file` - and must decompress through [`crate::inflate_gzip`]")
lines.append("/// to `plain`.")
lines.append("pub struct GzipVector {")
lines.append("    /// The vector's name, which is also the test's name.")
lines.append("    pub name: &'static str,")
lines.append("    /// The committed fixture file these bytes are, relative to")
lines.append("    /// `tests/fixtures/gzip/` (see its PROVENANCE.md).")
lines.append("    pub file: &'static str,")
lines.append("    /// What this vector is here to exercise.")
lines.append("    pub why: &'static str,")
lines.append("    /// The complete gzip stream, hex encoded.")
lines.append("    pub input: &'static str,")
lines.append("    /// The expected output, hex encoded.")
lines.append("    pub plain: &'static str,")
lines.append("}")
lines.append("")
lines.append("/// A gzip container with exactly one defect, and the error it must")
lines.append("/// produce.")
lines.append("pub struct GzipBadVector {")
lines.append("    /// The vector's name, which is also the test's name.")
lines.append("    pub name: &'static str,")
lines.append("    /// What is wrong with this stream.")
lines.append("    pub why: &'static str,")
lines.append("    /// The malformed bytes, hex encoded.")
lines.append("    pub input: &'static str,")
lines.append("    /// The `Error` variant the decoder must return, by name.")
lines.append("    pub kind: &'static str,")
lines.append("}")
lines.append("")
lines.append("/// Every valid gzip container, byte-identical to its fixture file.")
lines.append("pub const GZIP_VECTORS: &[GzipVector] = &[")
for name, fixture, why in GZIP:
    data = (GZIP_FIXTURES / fixture).read_bytes()
    plain = gzip.decompress(data)  # multi-member aware, unlike zlib.decompress
    lines.append("    GzipVector {")
    lines.append(f"        name: {rust_str(name)},")
    lines.append(f"        file: {rust_str(fixture)},")
    lines.append(f"        why: {rust_str(why)},")
    lines.append(f"        input: {hexlit(data)},")
    lines.append(f"        plain: {hexlit(plain)},")
    lines.append("    },")
    print(f"  GZIP {name:20s} {len(data):6d}B -> {len(plain):6d}B ({fixture})")
lines.append("];")
lines.append("")
lines.append("/// Every malformed gzip container.")
lines.append("pub const GZIP_BAD_VECTORS: &[GzipBadVector] = &[")
for name, data, kind, why in GZIP_BAD:
    lines.append("    GzipBadVector {")
    lines.append(f"        name: {rust_str(name)},")
    lines.append(f"        why: {rust_str(why)},")
    lines.append(f"        input: {hexlit(data)},")
    lines.append(f"        kind: {rust_str(kind)},")
    lines.append("    },")
    print(f"  BAD {name:24s} {len(data):4d}B -> {kind}")
lines.append("];")
lines.append("")

seen_names = {BLOCK_NAMES[b] for b in seen_blocks}
missing = set(BLOCK_NAMES.values()) - seen_names
if missing:
    raise SystemExit(f"vector set does not exercise block types: {sorted(missing)}")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(lines), encoding="utf-8", newline="\n")
print(f"\nblock types covered: {sorted(seen_names)}")
print(f"valid vectors: {len(PLAINS) * 2}   malformed: {len(BAD)}")
print(f"gzip vectors: {len(GZIP)}   gzip malformed: {len(GZIP_BAD)}")
print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")