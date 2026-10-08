// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pith-hash
"use strict";

/**
 * pith-inflate SDK: DEFLATE and zlib decompression through koffi.
 *
 * Every function binds the Rust cdylib built by `cargo build
 * --release`; the library is located through the suite's discovery
 * chain (PITH_CDYLIB, PITH_CDYLIB_DIR, the packaged prebuilds/
 * location, then <repo root>/target/release). All three decompress
 * operations run with the core's conservative default limits (64 MiB
 * of input, 64 MiB of output); a stream the decoder refuses throws
 * an FfiError with `status === STATUS_REJECTED`.
 */

const koffi = require("koffi");
const fs = require("node:fs");
const path = require("node:path");

const STATUS_OK = 0;
const STATUS_INVALID = -1;
const STATUS_REJECTED = -2;

/** Every cdylib file name cargo may drop into the build directory, per platform. */
const CDYLIB_NAMES = ["pith_inflate.dll", "libpith_inflate.so", "libpith_inflate.dylib"];

const PKG_ROOT = path.join(__dirname);
const REPO_ROOT = path.resolve(__dirname, "..", "..");

/** FfiError: a non-zero status code came back from the cdylib. */
class FfiError extends Error {
  /**
   * @param {string} op the FFI operation name
   * @param {number} status the raw status code
   */
  constructor(op, status) {
    const kind = { [STATUS_INVALID]: "invalid argument", [STATUS_REJECTED]: "input rejected" }[status] ?? "unknown failure";
    super(`${op} failed: ${kind} (status ${status})`);
    this.name = "FfiError";
    /** The raw status code the FFI returned. */
    this.status = status;
  }
}

/**
 * Locates the cdylib through the suite's discovery chain.
 * @returns {string} an absolute path to the cdylib file
 * @throws {Error} when nothing is found
 */
function findCdylib() {
  const explicit = process.env.PITH_CDYLIB;
  if (explicit && fs.statSync(explicit, { throwIfNoEntry: false })?.isFile()) {
    return path.resolve(explicit);
  }
  /** @type {string[]} */
  const dirs = [];
  const envDir = process.env.PITH_CDYLIB_DIR;
  if (envDir) {
    dirs.push(envDir);
    if (!path.isAbsolute(envDir)) {
      dirs.push(path.join(REPO_ROOT, envDir));
    }
  }
  const osArch = `${process.platform}-${process.arch}`;
  dirs.push(path.join(PKG_ROOT, "prebuilds", osArch));
  dirs.push(path.join(PKG_ROOT, "prebuilds"));
  dirs.push(path.join(REPO_ROOT, "target", "release"));
  for (const dir of dirs) {
    for (const name of CDYLIB_NAMES) {
      const p = path.join(dir, name);
      if (fs.statSync(p, { throwIfNoEntry: false })?.isFile()) return p;
    }
  }
  throw new Error(
    "no pith-inflate cdylib found (searched PITH_CDYLIB, PITH_CDYLIB_DIR, prebuilds/ and <repo>/target/release); " +
      "run `cargo build --release` first",
  );
}

let cached = undefined;

/**
 * Loads the cdylib and binds the exported symbols (lazily, once).
 * @returns {{inflateRaw: Function, inflateZlib: Function, inflateAuto: Function,
 *   adler32: Function, free: Function}}
 */
function loadLibrary() {
  if (cached) return cached;
  const lib = koffi.load(findCdylib());
  /** @type {[string, string]} */
  const inflateSignature = [
    "const uint8_t *",
    "size_t",
    koffi.out(koffi.pointer("void *")),
    koffi.out(koffi.pointer("size_t")),
  ];
  const inflateRaw = lib.func("pith_inflate_inflate_raw", "int32_t", inflateSignature);
  const inflateZlib = lib.func("pith_inflate_inflate_zlib", "int32_t", inflateSignature);
  const inflateAuto = lib.func("pith_inflate_inflate_auto", "int32_t", inflateSignature);
  const inflateGzip = lib.func("pith_inflate_inflate_gzip", "int32_t", inflateSignature);
  const adler32 = lib.func("pith_inflate_adler32", "int32_t", [
    "const uint8_t *",
    "size_t",
    koffi.out(koffi.pointer("uint32_t")),
  ]);
  const free = lib.func("void pith_inflate_free(void *ptr, size_t len)");
  cached = { inflateRaw, inflateZlib, inflateAuto, inflateGzip, adler32, free };
  return cached;
}

/**
 * Runs one decompress entry point and copies the handed-out cdylib
 * buffer into a JS Buffer before releasing it.
 *
 * @param {Function} op the bound decompress symbol
 * @param {string} opName the FFI operation name, for error messages
 * @param {Buffer} data the compressed stream
 * @returns {Buffer} the decompressed bytes (empty for an empty stream)
 * @throws {FfiError} with `status === -2` for any malformed input
 */
function decompress(op, opName, data) {
  if (!Buffer.isBuffer(data)) {
    throw new TypeError("data must be a Buffer");
  }
  const out = [null];
  const outLen = [0];
  const status = op(data, data.length, out, outLen);
  if (status !== STATUS_OK) {
    throw new FfiError(opName, status);
  }
  try {
    // koffi.decode hands back a Uint8Array view over the external
    // buffer; copy it into a Buffer before the cdylib buffer is freed.
    return Buffer.from(koffi.decode(out[0], "uint8_t", Number(outLen[0])));
  } finally {
    const { free } = loadLibrary();
    free(out[0], Number(outLen[0]));
  }
}

/**
 * Decompresses a raw RFC 1951 DEFLATE stream (trailing bytes after the
 * final block are ignored).
 *
 * @param {Buffer} data the compressed stream
 * @returns {Buffer} the decompressed bytes (empty for an empty stream)
 * @throws {FfiError} with `status === -2` for any malformed input
 */
function inflateRaw(data) {
  return decompress(loadLibrary().inflateRaw, "pith_inflate_inflate_raw", data);
}

/**
 * Decompresses a zlib (RFC 1950) stream: two-byte header, DEFLATE
 * payload, big-endian Adler-32 trailer.
 *
 * @param {Buffer} data the compressed stream
 * @returns {Buffer} the decompressed bytes (empty for an empty stream)
 * @throws {FfiError} with `status === -2` for any malformed input,
 *   including a trailer that does not match the Adler-32 of the output
 */
function inflateZlib(data) {
  return decompress(loadLibrary().inflateZlib, "pith_inflate_inflate_zlib", data);
}

/**
 * Decompresses either framing, sniffing the two-byte zlib header. gzip
 * is recognised and refused (status -2), never mis-decoded.
 *
 * @param {Buffer} data the compressed stream
 * @returns {Buffer} the decompressed bytes (empty for an empty stream)
 * @throws {FfiError} with `status === -2` for any malformed input
 */
function inflateAuto(data) {
  return decompress(loadLibrary().inflateAuto, "pith_inflate_inflate_auto", data);
}

/**
 * Decompresses a gzip (RFC 1952) container: the ten-byte header
 * (FEXTRA/FNAME/FCOMMENT/FHCRC optional fields included), the DEFLATE
 * payload, and the CRC-32 + ISIZE trailer. A multi-member stream
 * (concatenated .gz files) decodes to the concatenation of every
 * member's payload.
 *
 * @param {Buffer} data the compressed stream
 * @returns {Buffer} the decompressed bytes (empty for an empty stream)
 * @throws {FfiError} with `status === -2` for any malformed input,
 *   including a trailer whose CRC-32 or ISIZE does not match the
 *   produced output
 */
function inflateGzip(data) {
  return decompress(loadLibrary().inflateGzip, "pith_inflate_inflate_gzip", data);
}

/**
 * Incremental-feed decode canonicalized through the one-shot FFI.
 *
 * The binding holds no decoder state across the C ABI - the suite's
 * FFI is deliberately stateless - so this helper implements the
 * feed/finish shape by re-decoding the bytes buffered so far through
 * the framing's one-shot export on every call. That is O(n^2) across a
 * stream in the worst case, correct by construction, and
 * byte-identical to the Rust `StreamingDecoder` by the chunked-feed
 * equivalence property the Rust test suite pins. Throughput paths
 * should call the one-shot functions directly.
 *
 * A feed reports whether the bytes fed so far already decode
 * completely - for a multi-member stream that is true at every member
 * boundary, so keep feeding and let `finish()` give the final verdict.
 * `output()` tracks the longest decoded prefix. An incomplete - or so
 * far invalid - stream reports false: the stateless FFI cannot tell
 * "needs more input" from "broken input". `finish()` gives the
 * verdict, throwing `FfiError` for a stream that never decoded.
 */
class StreamingInflate {
  /** @type {Function} the one-shot decode this canonicalization runs */
  #op;
  /** @type {Buffer[]} every byte fed so far */
  #chunks;
  /** @type {Buffer | null} the longest decoded prefix */
  #output;

  /**
   * @param {"raw" | "zlib" | "gzip"} [framing] the container framing
   */
  constructor(framing = "gzip") {
    const ops = { raw: inflateRaw, zlib: inflateZlib, gzip: inflateGzip };
    if (!(framing in ops)) {
      throw new TypeError(`framing must be one of ${Object.keys(ops).join(", ")}, got ${framing}`);
    }
    this.#op = ops[framing];
    this.#chunks = [];
    this.#output = null;
  }

  /**
   * Feeds the next chunk.
   * @param {Buffer} data the next bytes of the stream
   * @returns {boolean} true when the bytes fed so far decode completely
   * @throws {TypeError} on a non-Buffer chunk
   */
  feed(data) {
    if (!Buffer.isBuffer(data)) {
      throw new TypeError("data must be a Buffer");
    }
    this.#chunks.push(data);
    try {
      this.#output = this.#op(Buffer.concat(this.#chunks));
    } catch (err) {
      if (!(err instanceof FfiError)) throw err;
      return false;
    }
    return true;
  }

  /**
   * The decoded bytes; empty until the stream decoded.
   * @returns {Buffer}
   */
  output() {
    return this.#output ?? Buffer.alloc(0);
  }

  /**
   * Ends the stream: the decoded bytes, or the decode's refusal as an
   * `FfiError` (malformed, truncated or over-limit).
   * @returns {Buffer}
   * @throws {FfiError} when the stream never decoded
   */
  finish() {
    if (this.#output === null) {
      this.#output = this.#op(Buffer.concat(this.#chunks));
    }
    return this.#output;
  }
}

/**
 * Computes the Adler-32 checksum (RFC 1950 section 9) of `data`. The
 * checksum of an empty input is 1, so `adler32(Buffer.alloc(0)) === 1`.
 *
 * @param {Buffer} data the bytes to checksum
 * @returns {number} the checksum as an unsigned 32-bit integer
 * @throws {FfiError} on a refused call
 */
function adler32(data) {
  if (!Buffer.isBuffer(data)) {
    throw new TypeError("data must be a Buffer");
  }
  const out = [0];
  const status = loadLibrary().adler32(data, data.length, out);
  if (status !== STATUS_OK) {
    throw new FfiError("pith_inflate_adler32", status);
  }
  return out[0] >>> 0;
}

module.exports = {
  STATUS_OK,
  STATUS_INVALID,
  STATUS_REJECTED,
  CDYLIB_NAMES,
  FfiError,
  findCdylib,
  inflateRaw,
  inflateZlib,
  inflateAuto,
  inflateGzip,
  StreamingInflate,
  adler32,
};
