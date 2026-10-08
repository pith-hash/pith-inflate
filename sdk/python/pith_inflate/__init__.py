# SPDX-License-Identifier: MIT
# Copyright (c) 2026 pith-hash
"""pith-inflate SDK: DEFLATE and zlib decompression through ctypes.

Every function binds the Rust cdylib built by ``cargo build
--release``; the library is located through the suite's discovery
chain (``PITH_CDYLIB``, ``PITH_CDYLIB_DIR``, the packaged wheel
location, then ``<repo root>/target/release``). All three decompress
operations run with the core's conservative default limits (64 MiB of
input, 64 MiB of output); a stream the decoder refuses raises
:class:`FfiError` with ``status == STATUS_REJECTED``.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

__all__ = [
    "STATUS_OK",
    "STATUS_INVALID",
    "STATUS_REJECTED",
    "CDYLIB_NAMES",
    "LibraryNotFoundError",
    "FfiError",
    "find_cdylib",
    "inflate_raw",
    "inflate_zlib",
    "inflate_auto",
    "inflate_gzip",
    "StreamingInflater",
    "adler32",
]

#: Status: success.
STATUS_OK = 0
#: Status: a caller argument is invalid (a null pointer).
STATUS_INVALID = -1
#: Status: the core decoder refused the input (malformed, truncated or
#: over-limit stream, or a framing the crate refuses).
STATUS_REJECTED = -2

#: Every cdylib file name cargo may drop into the build directory, per
#: platform (windows / linux / macOS).
CDYLIB_NAMES = ("pith_inflate.dll", "libpith_inflate.so", "libpith_inflate.dylib")


class LibraryNotFoundError(OSError):
    """No cdylib was found through the discovery chain."""


class FfiError(Exception):
    """A non-zero status code came back from the cdylib."""

    def __init__(self, op: str, status: int) -> None:
        kind = {
            STATUS_INVALID: "invalid argument",
            STATUS_REJECTED: "input rejected",
        }.get(status, "unknown failure")
        super().__init__(f"{op} failed: {kind} (status {status})")
        #: The raw status code the FFI returned.
        self.status = status


def find_cdylib() -> Path:
    """Locates the cdylib through the suite's discovery chain."""
    explicit = os.environ.get("PITH_CDYLIB")
    if explicit:
        p = Path(explicit)
        if p.is_file():
            return p
    env_dir = os.environ.get("PITH_CDYLIB_DIR")
    candidates: list[Path] = []
    if env_dir:
        env_dir_path = Path(env_dir)
        candidates.append(env_dir_path)
        if not env_dir_path.is_absolute():
            # CD and local runs invoke tools from the repository root or
            # from sdk/<lang>; resolve the env value against both.
            candidates.append(Path.cwd() / env_dir_path)
            candidates.append(Path(__file__).resolve().parents[3] / env_dir_path)
    candidates.append(Path(__file__).resolve().parent)  # packaged wheel
    candidates.append(Path(__file__).resolve().parents[3] / "target" / "release")
    for directory in candidates:
        for name in CDYLIB_NAMES:
            p = directory / name
            if p.is_file():
                return p
    raise LibraryNotFoundError(
        "no pith-inflate cdylib found (searched PITH_CDYLIB, PITH_CDYLIB_DIR, "
        "the package directory and <repo>/target/release); "
        "run `cargo build --release` first"
    )


_lib: ctypes.CDLL | None = None


def _load() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        lib = ctypes.CDLL(str(find_cdylib()))
        for op in (
            "pith_inflate_inflate_raw",
            "pith_inflate_inflate_zlib",
            "pith_inflate_inflate_auto",
            "pith_inflate_inflate_gzip",
        ):
            fn = getattr(lib, op)
            fn.argtypes = [
                ctypes.c_void_p,  # data
                ctypes.c_size_t,  # len
                ctypes.POINTER(ctypes.c_void_p),  # out buffer
                ctypes.POINTER(ctypes.c_size_t),  # out length
            ]
            fn.restype = ctypes.c_int32
        lib.pith_inflate_adler32.argtypes = [
            ctypes.c_void_p,  # data
            ctypes.c_size_t,  # len
            ctypes.POINTER(ctypes.c_uint32),  # out checksum
        ]
        lib.pith_inflate_adler32.restype = ctypes.c_int32
        lib.pith_inflate_free.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        lib.pith_inflate_free.restype = None
        _lib = lib
    return _lib


def _decompress(op: str, data: bytes) -> bytes:
    """Runs one decompress entry point and copies the handed-out buffer."""
    out = ctypes.c_void_p()
    out_len = ctypes.c_size_t()
    status = getattr(_load(), op)(data, len(data), ctypes.byref(out), ctypes.byref(out_len))
    if status != STATUS_OK:
        raise FfiError(op, status)
    try:
        return ctypes.string_at(out, out_len.value)
    finally:
        _load().pith_inflate_free(out, out_len.value)


def inflate_raw(data: bytes) -> bytes:
    """Decompresses a raw RFC 1951 DEFLATE stream (trailing bytes after
    the final block are ignored).

    Raises :class:`FfiError` with ``status == STATUS_REJECTED`` for any
    malformed, truncated or over-limit input — the decoder never panics
    through this boundary. A successfully decoded empty stream returns
    ``b""``.
    """
    return _decompress("pith_inflate_inflate_raw", data)


def inflate_zlib(data: bytes) -> bytes:
    """Decompresses a zlib (RFC 1950) stream: two-byte header, DEFLATE
    payload, big-endian Adler-32 trailer.

    Raises :class:`FfiError` with ``status == STATUS_REJECTED`` for any
    malformed input, including a trailer that does not match the
    Adler-32 of the produced output.
    """
    return _decompress("pith_inflate_inflate_zlib", data)


def inflate_auto(data: bytes) -> bytes:
    """Decompresses either framing, sniffing the two-byte zlib header.

    gzip is recognised and refused (``STATUS_REJECTED``), never
    mis-decoded.
    """
    return _decompress("pith_inflate_inflate_auto", data)


def inflate_gzip(data: bytes) -> bytes:
    """Decompresses a gzip (RFC 1952) container: the ten-byte header
    (``FEXTRA``/``FNAME``/``FCOMMENT``/``FHCRC`` optional fields
    included), the DEFLATE payload, and the CRC-32 + ISIZE trailer.

    A multi-member stream (concatenated ``.gz`` files) decodes to the
    concatenation of every member's payload.

    Raises :class:`FfiError` with ``status == STATUS_REJECTED`` for any
    malformed input, including a trailer whose CRC-32 or ISIZE does not
    match the produced output.
    """
    return _decompress("pith_inflate_inflate_gzip", data)


class StreamingInflater:
    """Incremental-feed decode canonicalized through the one-shot FFI.

    The binding holds no decoder state across the C ABI - the suite's
    FFI is deliberately stateless - so this helper implements the
    feed/finish shape by re-decoding the bytes buffered so far through
    the framing's one-shot export on every call. That is O(n^2) across
    a stream in the worst case, correct by construction, and
    byte-identical to the Rust ``StreamingDecoder`` by the chunked-feed
    equivalence property the Rust test suite pins. Throughput paths
    should call the one-shot functions directly.

    A feed reports whether the bytes fed so far already decode
    completely - for a multi-member stream that is true at every
    member boundary, so keep feeding and let :meth:`finish` give the
    final verdict. :meth:`output` tracks the longest decoded prefix.
    An incomplete - or so far invalid - stream reports ``False``: the
    stateless FFI cannot tell "needs more input" from "broken input".
    :meth:`finish` gives the verdict, raising :class:`FfiError` for a
    stream that never decoded.
    """

    def __init__(self, framing: str = "gzip") -> None:
        ops = {"raw": inflate_raw, "zlib": inflate_zlib, "gzip": inflate_gzip}
        if framing not in ops:
            valid = ", ".join(sorted(ops))
            raise ValueError(f"framing must be one of {valid}, got {framing!r}")
        #: The one-shot decode this canonicalization runs.
        self._op = ops[framing]
        #: Every byte fed so far.
        self._buffer = bytearray()
        #: The longest decoded prefix, else ``None``.
        self._output: bytes | None = None

    def feed(self, data: bytes) -> bool:
        """Feeds the next chunk; ``True`` when the bytes fed so far
        decode completely, ``False`` while more (or a verdict) is
        needed."""
        self._buffer += data
        try:
            self._output = self._op(bytes(self._buffer))
        except FfiError:
            return False
        return True

    def output(self) -> bytes:
        """The decoded bytes; empty until the stream decoded."""
        return self._output if self._output is not None else b""

    def finish(self) -> bytes:
        """Ends the stream: the decoded bytes, or the decode's refusal
        as :class:`FfiError` (malformed, truncated or over-limit)."""
        if self._output is None:
            self._output = self._op(bytes(self._buffer))
        return self._output


def adler32(data: bytes) -> int:
    """Computes the Adler-32 checksum (RFC 1950 section 9) of ``data``.

    The checksum of an empty input is 1, so ``adler32(b"") == 1``.
    """
    out = ctypes.c_uint32()
    status = _load().pith_inflate_adler32(data, len(data), ctypes.byref(out))
    if status != STATUS_OK:
        raise FfiError("pith_inflate_adler32", status)
    return out.value
