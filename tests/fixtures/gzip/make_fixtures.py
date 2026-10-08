#!/usr/bin/env python3
"""Regenerates every `.gz` fixture beside this script.

The committed bytes are canonical; this script exists so they stay
reproducible. Two provenance classes:

* gzip(1)-generated (deterministic through `-n`: no MTIME, no FNAME):
  the script shells out to the real `gzip` CLI, exactly the commands
  PROVENANCE.md records.
* crafted (RFC 1952 optional fields the CLI cannot emit: FEXTRA,
  FHCRC, and byte-exact FLG combinations): packed here from Python's
  zlib, raw DEFLATE body (method 8, window -15), MTIME 0, XFL 2, OS 3.

Run from this directory: `python make_fixtures.py` (gzip(1) must be on
PATH). The script rewrites every `.gz` it owns; `PROVENANCE.md` lists
the expected SHA-256 of each committed file.
"""

from __future__ import annotations

import hashlib
import pathlib
import subprocess
import sys
import zlib

HERE = pathlib.Path(__file__).resolve().parent

FTEXT, FHCRC, FEXTRA, FNAME, FCOMMENT = 0x01, 0x02, 0x04, 0x08, 0x10


def raw_deflate(payload: bytes) -> bytes:
    """A raw RFC 1951 stream at level 9, the same body gzip(1) carries."""
    compressor = zlib.compressobj(9, zlib.DEFLATED, -15)
    return compressor.compress(payload) + compressor.flush()


def gzip_member(
    payload: bytes,
    flg: int = 0,
    extra: bytes = b"",
    name: bytes = b"",
    comment: bytes = b"",
    fhcrc_value: int | None = None,
) -> bytes:
    """One RFC 1952 member: fixed header, optional fields, DEFLATE, trailer.

    MTIME is 0, XFL 2 (level 9), OS 3 (Unix). `fhcrc_value` overrides the
    header CRC16 - only for building corrupt-header bad vectors, never a
    committed fixture.
    """
    head = bytearray(b"\x1f\x8b\x08")
    head.append(flg)
    head += (0).to_bytes(4, "little")
    head.append(2)
    head.append(3)
    if flg & FEXTRA:
        head += len(extra).to_bytes(2, "little")
        head += extra
    if flg & FNAME:
        head += name + b"\x00"
    if flg & FCOMMENT:
        head += comment + b"\x00"
    if flg & FHCRC:
        crc16 = zlib.crc32(bytes(head)) & 0xFFFF if fhcrc_value is None else fhcrc_value
        head += crc16.to_bytes(2, "little")
    trailer = (zlib.crc32(payload) & 0xFFFFFFFF).to_bytes(4, "little")
    trailer += (len(payload) & 0xFFFFFFFF).to_bytes(4, "little")
    return bytes(head) + raw_deflate(payload) + trailer


def run_gzip(args: list[str], data: bytes) -> bytes:
    """Feeds `data` through the real gzip CLI (`gzip -n ...`) and returns the bytes."""
    proc = subprocess.run(
        ["gzip", *args],
        input=data,
        capture_output=True,
        check=True,
    )
    return proc.stdout


def deterministic_binary(count: int) -> bytes:
    """The corpus' structured-bytes recipe: `(i * 7919) % 256`, seed-free."""
    return bytes((i * 7919) % 256 for i in range(count))


HELLO = b"Hello, gzip! The quick brown fox jumps over the lazy dog.\n"
MEMBER_ONE = b"the first member's payload, repeated. " * 8
MEMBER_TWO = b"second member\n"


def main() -> int:
    fixtures: dict[str, bytes] = {}

    # --- gzip(1)-generated (-n: no MTIME, no FNAME -> deterministic) ---
    fixtures["hello.gz"] = run_gzip(["-9", "-n"], HELLO)
    fixtures["empty.gz"] = run_gzip(["-9", "-n"], b"")
    fixtures["two-members.gz"] = (
        run_gzip(["-6", "-n"], MEMBER_ONE) + run_gzip(["-1", "-n"], MEMBER_TWO)
    )
    fixtures["binary.gz"] = run_gzip(["-9", "-n"], deterministic_binary(4096))

    # --- crafted (RFC 1952 optional fields) ---
    fixtures["fextra.gz"] = gzip_member(
        b"FEXTRA payload",
        flg=FEXTRA,
        extra=bytes([0x70, 0x68, 0x04, 0x00, 0x01, 0x02, 0x03, 0x04]),
    )
    fixtures["fname.gz"] = gzip_member(b"FNAME payload", flg=FNAME, name=b"payload.txt")
    fixtures["fcomment.gz"] = gzip_member(
        b"FCOMMENT payload", flg=FCOMMENT, comment=b"crafted for the pith-inflate corpus"
    )
    fixtures["fhcrc.gz"] = gzip_member(b"FHCRC payload", flg=FHCRC)
    fixtures["all-flags.gz"] = gzip_member(
        b"all flags payload - FTEXT FEXTRA FNAME FCOMMENT FHCRC\n",
        flg=FTEXT | FEXTRA | FNAME | FCOMMENT | FHCRC,
        extra=b"\x70\x68\x04\x00\xde\xad\xbe\xef",
        name=b"all-flags-payload.txt",
        comment=b"every optional header field at once",
    )

    for filename, data in fixtures.items():
        path = HERE / filename
        path.write_bytes(data)
        print(f"{filename:20s} {len(data):6d}B  sha256={hashlib.sha256(data).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
