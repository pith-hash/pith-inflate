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
        for op in ("pith_inflate_inflate_raw", "pith_inflate_inflate_zlib", "pith_inflate_inflate_auto"):
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


def adler32(data: bytes) -> int:
    """Computes the Adler-32 checksum (RFC 1950 section 9) of ``data``.

    The checksum of an empty input is 1, so ``adler32(b"") == 1``.
    """
    out = ctypes.c_uint32()
    status = _load().pith_inflate_adler32(data, len(data), ctypes.byref(out))
    if status != STATUS_OK:
        raise FfiError("pith_inflate_adler32", status)
    return out.value
