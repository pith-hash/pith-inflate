// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pith-hash
"use strict";

// Hex-exact conformance: the committed reference vectors through koffi.
// Every vector in the repository-root reference.json is replayed through
// the cdylib and compared byte-exact — the decompressed output against
// the recorded plain hex, and the FFI's own Adler-32 of that output
// against the recorded digest. Malformed vectors must come back as an
// FfiError with status -2, never a crash. The same vectors the Rust
// gen-reference verify gate and the Python/Go SDKs check.

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const {
  FfiError,
  adler32,
  findCdylib,
  inflateAuto,
  inflateRaw,
  inflateZlib,
} = require("../index.js");

const REPO_ROOT = path.resolve(__dirname, "..", "..", "..");

const REFERENCE = JSON.parse(fs.readFileSync(path.join(REPO_ROOT, "reference.json"), "utf8"));

/** The reference contract's structural sniff for bad vectors. */
function looksLikeZlib(b) {
  return (
    b.length >= 2 &&
    (b[0] & 0x0f) === 8 &&
    b[0] >> 4 <= 7 &&
    ((b[0] << 8) | b[1]) % 31 === 0
  );
}

test("cdylib is discoverable", () => {
  assert.ok(fs.statSync(findCdylib()).isFile());
});

for (const vector of REFERENCE.vectors) {
  test(`reference vector ${vector.name} is reproduced hex-exact`, () => {
    const compressed = Buffer.from(vector.compressed, "hex");
    const op = vector.name.endsWith("/zlib") ? inflateZlib : inflateRaw;

    const out = op(compressed);
    assert.deepEqual(out, Buffer.from(vector.plain, "hex"), vector.name);
    assert.equal(adler32(out).toString(16).padStart(8, "0"), vector.adler32, vector.name);
  });
}

for (const vector of REFERENCE.bad_vectors) {
  test(`bad vector ${vector.name} is refused, not crashing`, () => {
    const data = Buffer.from(vector.input, "hex");
    const op =
      vector.name.startsWith("zlib_") || looksLikeZlib(data) ? inflateZlib : inflateRaw;

    assert.throws(() => op(data), (err) => {
      assert.ok(err instanceof FfiError);
      assert.equal(err.status, -2, vector.name);
      return true;
    });
  });
}

test("truncated stream is refused on every routing", () => {
  // A zlib header with no DEFLATE blocks behind it: refused by the
  // core through raw, zlib and auto framing alike, never a crash.
  const truncated = Buffer.from([0x78, 0xda]);
  for (const op of [inflateRaw, inflateZlib, inflateAuto]) {
    assert.throws(() => op(truncated), (err) => err instanceof FfiError && err.status === -2);
  }
});

test("garbage is refused, not crashing", () => {
  assert.throws(() => inflateAuto(Buffer.alloc(16)), (err) => err instanceof FfiError && err.status === -2);
});

test("empty input is refused", () => {
  assert.throws(() => inflateRaw(Buffer.alloc(0)), (err) => err instanceof FfiError && err.status === -2);
});

test("empty output round-trips through free", () => {
  // The empty/zlib vector decoded to a zero-length buffer, and that
  // buffer is released through the same free path as any other.
  const out = inflateZlib(Buffer.from("78da030000000001", "hex"));
  assert.equal(out.length, 0);
  assert.equal(adler32(out), 1);
});

test("full output matches a rust-pinned vector", () => {
  // The text vector, pinned literally in the Rust unit tests and
  // re-derived here from the committed values; this test fails loudly
  // even if reference.json were regenerated wrongly.
  const compressed = Buffer.from(
    "2bc94855282ccd4cce56482aca2fcf5348cbaf50c82acd2d2856c82f4b2d5228014ae72456552aa4e4a7eb8179a38a478c6200",
    "hex",
  );
  const out = inflateAuto(compressed);
  assert.deepEqual(
    out,
    Buffer.from(
      "74686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c617a7920646f672e" +
        "2074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c617a7920646f67" +
        "2e2074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c617a7920646f" +
        "672e2074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c617a792064" +
        "6f672e2074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c617a7920" +
        "646f672e2074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c617a79" +
        "20646f672e2074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c617a" +
        "7920646f672e2074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c61" +
        "7a7920646f672e2074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c" +
        "617a7920646f672e2074686520717569636b2062726f776e20666f78206a756d7073206f7665722074686520" +
        "6c617a7920646f672e2074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865" +
        "206c617a7920646f672e2074686520717569636b2062726f776e20666f78206a756d7073206f766572207468" +
        "65206c617a7920646f672e20",
      "hex",
    ),
  );
  assert.equal(adler32(out).toString(16).padStart(8, "0"), "f724c355");
});
