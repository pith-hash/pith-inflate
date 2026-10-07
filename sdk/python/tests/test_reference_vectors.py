# SPDX-License-Identifier: MIT
# Copyright (c) 2026 pith-hash
"""Hex-exact conformance: the committed reference vectors through ctypes.

Every vector in the repository-root ``reference.json`` is replayed
through the cdylib and compared byte-exact — the decompressed output
against the recorded ``plain`` hex, and the FFI's own Adler-32 of that
output against the recorded digest. Malformed vectors must come back
as an :class:`FfiError` with ``status == -2``, never a crash. The same
vectors the Rust ``gen-reference verify`` gate and the Node/Go SDKs
check.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pith_inflate import FfiError, adler32, find_cdylib, inflate_auto, inflate_raw, inflate_zlib

REPO_ROOT = Path(__file__).resolve().parents[3]

REFERENCE = json.loads((REPO_ROOT / "reference.json").read_text(encoding="utf-8"))


def looks_like_zlib(b: bytes) -> bool:
    """The reference contract's structural sniff for bad vectors."""
    return (
        len(b) >= 2
        and (b[0] & 0x0F) == 8
        and (b[0] >> 4) <= 7
        and ((b[0] << 8) | b[1]) % 31 == 0
    )


def test_cdylib_is_discoverable() -> None:
    path = find_cdylib()
    assert path.is_file(), path


@pytest.mark.parametrize("vector", REFERENCE["vectors"], ids=lambda v: v["name"])
def test_reference_vector_is_reproduced_hex_exact(vector: dict) -> None:
    compressed = bytes.fromhex(vector["compressed"])
    op = inflate_zlib if vector["name"].endswith("/zlib") else inflate_raw

    out = op(compressed)
    assert out == bytes.fromhex(vector["plain"]), vector["name"]
    assert f"{adler32(out):08x}" == vector["adler32"], vector["name"]


@pytest.mark.parametrize("vector", REFERENCE["bad_vectors"], ids=lambda v: v["name"])
def test_bad_vector_is_refused_not_crashing(vector: dict) -> None:
    data = bytes.fromhex(vector["input"])
    op = inflate_zlib if (vector["name"].startswith("zlib_") or looks_like_zlib(data)) else inflate_raw

    with pytest.raises(FfiError) as err:
        op(data)
    assert err.value.status == -2, vector["name"]


def test_truncated_stream_is_refused_on_every_routing() -> None:
    # A zlib header with no DEFLATE blocks behind it: refused by the
    # core through raw, zlib and auto framing alike, never a crash.
    for op in (inflate_raw, inflate_zlib, inflate_auto):
        with pytest.raises(FfiError) as err:
            op(b"\x78\xda")
        assert err.value.status == -2


def test_garbage_is_refused_not_crashing() -> None:
    with pytest.raises(FfiError) as err:
        inflate_auto(b"\x00" * 16)
    assert err.value.status == -2


def test_empty_input_is_refused() -> None:
    with pytest.raises(FfiError) as err:
        inflate_raw(b"")
    assert err.value.status == -2


def test_empty_output_round_trips_through_free() -> None:
    # The `empty/zlib` vector decoded to a zero-length buffer, and that
    # buffer is released through the same free path as any other.
    out = inflate_zlib(bytes.fromhex("78da030000000001"))
    assert out == b""
    assert adler32(out) == 1


def test_full_output_matches_a_rust_pinned_vector() -> None:
    # The `text` vector, pinned literally in the Rust unit tests and
    # re-derived here from the committed values; this test fails loudly
    # even if reference.json were regenerated wrongly.
    compressed = bytes.fromhex(
        "2bc94855282ccd4cce56482aca2fcf5348cbaf50c82acd2d2856c82f4b2d5228014ae72456552aa4e4a7eb8179a38a478c6200"
    )
    out = inflate_auto(compressed)
    assert out.hex() == (
        "74686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c617a7920646f672e"
        "2074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c617a7920646f67"
        "2e2074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c617a7920646f"
        "672e2074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c617a792064"
        "6f672e2074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c617a7920"
        "646f672e2074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c617a79"
        "20646f672e2074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c617a"
        "7920646f672e2074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c61"
        "7a7920646f672e2074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c"
        "617a7920646f672e2074686520717569636b2062726f776e20666f78206a756d7073206f7665722074686520"
        "6c617a7920646f672e2074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865"
        "206c617a7920646f672e2074686520717569636b2062726f776e20666f78206a756d7073206f766572207468"
        "65206c617a7920646f672e20"
    )
    assert f"{adler32(out):08x}" == "f724c355"
