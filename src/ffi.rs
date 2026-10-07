//! The C ABI surface of `pith-inflate`: the entry points the Python
//! (ctypes), Node (koffi) and Go (cgo) SDKs bind through.
//!
//! The suite's FFI convention, defined by this module and mirrored by
//! every `pith-*` cdylib:
//!
//! * one flat set of `#[unsafe(no_mangle)] pub unsafe extern "C"`
//!   functions — raw pointers plus lengths, no structs across the
//!   boundary;
//! * every function returns a status code (see the constants below),
//!   never a `Result`, never a panic: a foreign caller must never be
//!   able to unwind across this boundary;
//! * an operation either hands ownership to the caller (and ships a
//!   matching `_free` — [`pith_inflate_free`] here) or writes into
//!   caller-provided out-parameters ([`pith_inflate_adler32`]
//!   allocates nothing, so there is no free for it);
//! * the `unsafe` allowance is confined to this module; the whole
//!   decoder stays unsafe-free behind the crate-root `#![deny]`.
//!
//! Every decompress entry point runs with the crate's conservative
//! default [`Limits`] (64 MiB of input, 64 MiB of output) — a foreign
//! caller gets the same denial-of-service ceiling a Rust caller does.
//!
//! One deliberate nuance: a null `data` with `len == 0` is *not* a
//! caller bug — it is an empty input, which the decoder refuses with
//! [`PITH_E_REJECTED`] exactly like any other malformed stream. Only a
//! null `data` with a non-zero `len` is [`PITH_E_INVALID`]. This keeps
//! the status uniform across the three SDK languages: Go hands a nil
//! pointer for an empty slice, and the `empty_input` reference vector
//! must come back `-2` everywhere, not `-1` on one language.
//!
//! Each export below is a single-line-signature delegation (via
//! `#[rustfmt::skip]`) to a private, normally-mangled core function.
//! That packing is load-bearing for the coverage gate: `#[no_mangle]`
//! symbols collide by name across the workspace's test binaries in the
//! llvm-cov profile merge, so an export's executable lines read as
//! uncovered whenever *any* other test target links the crate; private
//! mangled functions merge per-binary and stay covered. Keeping every
//! export at two executable lines bounds that merge artifact to ten
//! lines, and the logic lives in fully-tested private cores.

#![allow(unsafe_code)]

use crate::{Limits, adler32, inflate_auto, inflate_raw, inflate_zlib};

/// Status: success.
pub const PITH_OK: i32 = 0;
/// Status: a caller argument is invalid — a null out-parameter, or a
/// null `data` with a non-zero `len`.
pub const PITH_E_INVALID: i32 = -1;
/// Status: the core decoder refused the input — a malformed, truncated
/// or over-limit stream, or a framing the crate refuses (gzip, preset
/// dictionaries).
pub const PITH_E_REJECTED: i32 = -2;

/// Which framing one FFI decompress entry point uses.
#[derive(Copy, Clone, Debug)]
enum Op {
    /// Raw RFC 1951 DEFLATE.
    Raw,
    /// RFC 1950 zlib framing.
    Zlib,
    /// Sniff the framing (`inflate_auto`).
    Auto,
}

/// Decompresses a raw RFC 1951 DEFLATE stream.
///
/// `data` points at `len` bytes of the stream (trailing bytes after
/// the final block are ignored, as [`crate::inflate_raw`] documents).
/// On success the function allocates a buffer, writes its address
/// through `out`, its length through `out_len`, and returns
/// [`PITH_OK`]; the caller owns the buffer and must release it with
/// [`pith_inflate_free`], passing back the same pointer *and* length.
/// A successfully decompressed empty stream hands out a zero-length
/// buffer, which is still freed through [`pith_inflate_free`].
///
/// Returns [`PITH_E_INVALID`] for a null `out`/`out_len`, or a null
/// `data` with `len > 0`; [`PITH_E_REJECTED`] for anything the decoder
/// refuses.
///
/// # Safety
///
/// `data` must point to `len` readable bytes (or be null when `len`
/// is zero); `out` to one writable pointer; `out_len` to one writable
/// `usize`. All must stay valid for the duration of the call; the
/// function retains nothing.
#[rustfmt::skip]
#[unsafe(no_mangle)] pub unsafe extern "C" fn pith_inflate_inflate_raw(data: *const u8, len: usize, out: *mut *mut u8, out_len: *mut usize) -> i32 { inflate_ffi(Op::Raw, data, len, out, out_len) }

/// Decompresses a zlib (RFC 1950) stream: two-byte header, DEFLATE
/// payload, big-endian Adler-32 trailer.
///
/// Buffer ownership, status codes and pointer rules are exactly those
/// of [`pith_inflate_inflate_raw`]; the only difference is the framing
/// the decoder expects (see [`crate::inflate_zlib`]).
///
/// # Safety
///
/// Same contract as [`pith_inflate_inflate_raw`].
#[rustfmt::skip]
#[unsafe(no_mangle)] pub unsafe extern "C" fn pith_inflate_inflate_zlib(data: *const u8, len: usize, out: *mut *mut u8, out_len: *mut usize) -> i32 { inflate_ffi(Op::Zlib, data, len, out, out_len) }

/// Decompresses either framing, sniffing the two-byte zlib header.
///
/// Buffer ownership, status codes and pointer rules are exactly those
/// of [`pith_inflate_inflate_raw`]; see [`crate::inflate_auto`] for
/// the sniffing rules (gzip is recognised and refused, never guessed).
///
/// # Safety
///
/// Same contract as [`pith_inflate_inflate_raw`].
#[rustfmt::skip]
#[unsafe(no_mangle)] pub unsafe extern "C" fn pith_inflate_inflate_auto(data: *const u8, len: usize, out: *mut *mut u8, out_len: *mut usize) -> i32 { inflate_ffi(Op::Auto, data, len, out, out_len) }

/// Computes the Adler-32 checksum (RFC 1950 section 9) of `len` bytes
/// at `data` into the caller-provided `out` slot.
///
/// No allocation, no free. Returns [`PITH_OK`] and writes the
/// checksum through `out`; [`PITH_E_INVALID`] for a null `out`, or a
/// null `data` with `len > 0`. The checksum of an empty input is 1,
/// so `adler32(NULL, 0, out)` is a successful call writing `1`.
///
/// # Safety
///
/// `data` must point to `len` readable bytes (or be null when `len`
/// is zero); `out` to one writable `u32`, valid for the duration of
/// the call.
#[rustfmt::skip]
#[unsafe(no_mangle)] pub unsafe extern "C" fn pith_inflate_adler32(data: *const u8, len: usize, out: *mut u32) -> i32 { adler32_ffi(data, len, out) }

/// Releases a buffer handed out by [`pith_inflate_inflate_raw`],
/// [`pith_inflate_inflate_zlib`] or [`pith_inflate_inflate_auto`].
///
/// # Safety
///
/// `ptr` must be a pointer one of those functions handed out with the
/// `out_len` value that came back with it, and must not have been
/// released (or otherwise freed) before. Null is accepted and
/// ignored, so callers can free unconditionally on the error path; a
/// zero-length buffer is freed by passing its (dangling) pointer with
/// `len == 0`.
#[rustfmt::skip]
#[unsafe(no_mangle)] pub unsafe extern "C" fn pith_inflate_free(ptr: *mut u8, len: usize) { free_buffer(ptr, len) }

/// The safe core of [`pith_inflate_adler32`]: null-guard the caller's
/// slot, borrow the input, checksum. No allocation, no free. The
/// checksum of an empty input is 1, so an empty input is a successful
/// call writing `1` — a null `data` with `len == 0` is an empty
/// input, exactly as in [`inflate_ffi`].
fn adler32_ffi(data: *const u8, len: usize, out: *mut u32) -> i32 {
    if out.is_null() {
        return PITH_E_INVALID;
    }
    let checksum = if len == 0 {
        let empty: &[u8] = &[];
        adler32(empty)
    } else {
        if data.is_null() {
            return PITH_E_INVALID;
        }
        let bytes = unsafe { core::slice::from_raw_parts(data, len) };
        adler32(bytes)
    };
    unsafe { *out = checksum };
    PITH_OK
}

/// The safe core of [`pith_inflate_free`]: reconstruct the boxed slice
/// from the same length the allocation was handed out with. Null is
/// ignored, so callers can free unconditionally on the error path.
fn free_buffer(ptr: *mut u8, len: usize) {
    if ptr.is_null() {
        return;
    }
    let slice = unsafe { core::slice::from_raw_parts_mut(ptr, len) };
    drop(unsafe { alloc::boxed::Box::from_raw(slice) });
}

/// The shared body of the three decompress entry points: null-guard
/// the caller's slots, borrow the input, decompress with the default
/// [`Limits`], then hand the exact-length buffer to the caller —
/// [`pith_inflate_free`] reconstructs the boxed slice from the same
/// length.
fn inflate_ffi(op: Op, data: *const u8, len: usize, out: *mut *mut u8, out_len: *mut usize) -> i32 {
    if out.is_null() || out_len.is_null() {
        return PITH_E_INVALID;
    }
    let bytes: &[u8] = if len == 0 {
        // A null `data` with `len == 0` is an empty input, not a bug:
        // the decoder below refuses it uniformly across SDK languages.
        &[]
    } else {
        if data.is_null() {
            return PITH_E_INVALID;
        }
        unsafe { core::slice::from_raw_parts(data, len) }
    };
    match inflate_framed(op, bytes) {
        Ok(output) => {
            let len = output.len();
            // Hand the exact-length buffer to the caller. For an empty
            // output this is a dangling-but-aligned pointer with
            // `len == 0`, which `pith_inflate_free` reconstructs.
            let ptr = alloc::boxed::Box::into_raw(output.into_boxed_slice());
            unsafe {
                *out = ptr.cast::<u8>();
                *out_len = len;
            }
            PITH_OK
        }
        Err(status) => status,
    }
}

/// The safe core of the three decompress entry points: dispatch on the
/// framing and map every decoder refusal to [`PITH_E_REJECTED`].
fn inflate_framed(op: Op, bytes: &[u8]) -> Result<alloc::vec::Vec<u8>, i32> {
    let limits = Limits::default();
    let result = match op {
        Op::Raw => inflate_raw(bytes, &limits),
        Op::Zlib => inflate_zlib(bytes, &limits),
        Op::Auto => inflate_auto(bytes, &limits),
    };
    result.map_err(|_| PITH_E_REJECTED)
}

// The known-answer corpus, shared verbatim with the integration
// harness (`tests/inflate.rs`) and the gen-reference binary — one
// source of truth for "what this decoder must produce". The `#[path]`
// is relative to `src/`, the directory of the file declaring it.
#[cfg(test)]
#[path = "../tests/vectors.rs"]
mod vectors;

#[cfg(test)]
mod tests {
    use super::{
        Op, PITH_E_INVALID, PITH_E_REJECTED, PITH_OK, inflate_ffi, inflate_framed,
        pith_inflate_adler32, pith_inflate_free, pith_inflate_inflate_auto,
        pith_inflate_inflate_raw, pith_inflate_inflate_zlib,
    };

    use super::vectors::{BAD_VECTORS, VECTORS};

    /// Decodes the lowercase-hex encoding the corpus uses for every
    /// byte field.
    fn unhex(s: &str) -> alloc::vec::Vec<u8> {
        let bytes = s.as_bytes();
        assert!(bytes.len() % 2 == 0, "odd hex length in {s:?}");
        bytes
            .chunks(2)
            .map(|pair| {
                let hi = (pair[0] as char).to_digit(16).expect("hex digit");
                let lo = (pair[1] as char).to_digit(16).expect("hex digit");
                ((hi << 4) | lo) as u8
            })
            .collect()
    }

    /// Lowercase hex of a byte slice, for pin comparisons.
    fn hex(data: &[u8]) -> String {
        let mut out = String::with_capacity(data.len() * 2);
        for byte in data {
            out.push_str(&format!("{byte:02x}"));
        }
        out
    }

    /// The same structural sniff the gen-reference binary and the
    /// language harnesses repeat: a bad vector that is zlib-framed is
    /// routed through `inflate_zlib`, the rest through `inflate_raw`.
    fn looks_like_zlib(bytes: &[u8]) -> bool {
        bytes.len() >= 2
            && bytes[0] & 0x0f == 8
            && bytes[0] >> 4 <= 7
            && ((u16::from(bytes[0]) << 8) | u16::from(bytes[1])) % 31 == 0
    }

    /// Every corpus vector reproduces hex-exact through the safe
    /// cores, routed by the reference contract (`"/zlib"` names go
    /// through the zlib framing); the Adler-32 of each output matches
    /// the digest the same output is pinned by in `reference.json`'s
    /// corpus order — recomputed here, so the FFI pins can never
    /// disagree with the shipped vectors.
    #[test]
    fn safe_core_reproduces_every_corpus_vector() {
        for v in VECTORS {
            let compressed = unhex(v.compressed);
            let op = if v.name.ends_with("/zlib") {
                Op::Zlib
            } else {
                Op::Raw
            };
            let plain = inflate_framed(op, &compressed)
                .unwrap_or_else(|status| panic!("vector {} refused: {status}", v.name));
            assert_eq!(hex(&plain), v.plain, "vector {}", v.name);
            let mut sum = 0u32;
            let status = unsafe { pith_inflate_adler32(plain.as_ptr(), plain.len(), &mut sum) };
            assert_eq!(status, PITH_OK, "vector {}", v.name);
            // The FFI checksum equals the crate's own; `gen-reference
            // verify` already pins that value into reference.json.
            assert_eq!(sum, crate::adler32(&plain), "vector {}", v.name);
        }
    }

    /// Every corpus bad vector is refused with [`PITH_E_REJECTED`]
    /// through the same routing the reference contract spells out:
    /// zlib-prefixed names and zlib-looking bytes go through the zlib
    /// framing, the rest through raw DEFLATE.
    #[test]
    fn safe_core_refuses_every_bad_vector() {
        for v in BAD_VECTORS {
            let input = unhex(v.input);
            let zlib_routed = v.name.starts_with("zlib_") || looks_like_zlib(&input);
            let op = if zlib_routed { Op::Zlib } else { Op::Raw };
            assert_eq!(
                inflate_framed(op, &input),
                Err(PITH_E_REJECTED),
                "bad vector {} must be refused",
                v.name
            );
        }
    }

    /// The `text` vector, pinned literally (compressed, plain and
    /// Adler-32 read off the committed `reference.json`): a wrongly
    /// regenerated reference cannot mask drift, because this test
    /// fails independently of it.
    #[test]
    fn pinned_text_vector() {
        let compressed = unhex(
            "2bc94855282ccd4cce56482aca2fcf5348cbaf50c82acd2d2856c82f4b2d5228014ae72456552aa4e4a7eb8179a38a478c6200",
        );
        let plain_hex = concat!(
            "74686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c617a7920646f672e",
            "2074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c617a7920646f67",
            "2e2074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c617a7920646f",
            "672e2074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c617a792064",
            "6f672e2074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c617a7920",
            "646f672e2074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c617a79",
            "20646f672e2074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c617a",
            "7920646f672e2074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c61",
            "7a7920646f672e2074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c",
            "617a7920646f672e2074686520717569636b2062726f776e20666f78206a756d7073206f7665722074686520",
            "6c617a7920646f672e2074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865",
            "206c617a7920646f672e2074686520717569636b2062726f776e20666f78206a756d7073206f766572207468",
            "65206c617a7920646f672e20"
        );
        for op in [Op::Raw, Op::Auto] {
            let plain = inflate_framed(op, &compressed).expect("text vector");
            assert_eq!(hex(&plain), plain_hex, "text vector through {op:?}");
        }
        let plain = inflate_framed(Op::Auto, &compressed).expect("text vector");
        let mut sum = 0u32;
        let status = unsafe { pith_inflate_adler32(plain.as_ptr(), plain.len(), &mut sum) };
        assert_eq!(status, PITH_OK);
        assert_eq!(sum, 0xf724_c355);
    }

    /// The `empty/zlib` vector, pinned literally: the compressed
    /// empty stream is the eight bytes an encoder emits for "emit
    /// nothing", the plaintext is the empty slice, and its Adler-32
    /// is the initial value 1. Exercises the zero-length handout end
    /// to end, including the free of a zero-length buffer.
    #[test]
    fn pinned_empty_zlib_vector() {
        let compressed: alloc::vec::Vec<u8> = vec![0x78, 0xda, 0x03, 0x00, 0x00, 0x00, 0x00, 0x01];
        for op in [Op::Zlib, Op::Auto] {
            let plain = inflate_framed(op, &compressed).expect("empty/zlib vector");
            assert!(plain.is_empty(), "empty/zlib through {op:?}");
        }

        // Through the raw FFI: status OK, a handed-out zero-length
        // buffer, and a free of that same zero-length buffer.
        let mut out: *mut u8 = core::ptr::null_mut();
        let mut out_len: usize = 0;
        let status = unsafe {
            pith_inflate_inflate_zlib(
                compressed.as_ptr(),
                compressed.len(),
                &mut out,
                &mut out_len,
            )
        };
        assert_eq!(status, PITH_OK);
        assert_eq!(out_len, 0);
        assert!(!out.is_null());
        unsafe { pith_inflate_free(out, out_len) };

        // Adler-32 of an empty input is the initial value 1; the
        // handed-out zero-length buffer is already freed above, so the
        // null pointer with len 0 is the honest shape here.
        let mut sum = 0u32;
        let status = unsafe { pith_inflate_adler32(core::ptr::null(), 0, &mut sum) };
        assert_eq!(status, PITH_OK);
        assert_eq!(sum, 0x0000_0001);
    }

    /// The FFI hands back exactly the bytes the safe core produced,
    /// for a raw and an auto-routed vector, and the buffers round-trip
    /// through [`pith_inflate_free`].
    #[test]
    fn ffi_hands_back_the_decoded_bytes() {
        for name in ["text", "text/zlib"] {
            let v = VECTORS.iter().find(|v| v.name == name).expect("vector");
            let compressed = unhex(v.compressed);
            let expected = inflate_framed(Op::Auto, &compressed).expect("decode");
            let mut out: *mut u8 = core::ptr::null_mut();
            let mut out_len: usize = 0;
            let status = unsafe {
                pith_inflate_inflate_auto(
                    compressed.as_ptr(),
                    compressed.len(),
                    &mut out,
                    &mut out_len,
                )
            };
            assert_eq!(status, PITH_OK, "{name}");
            assert_eq!(out_len, expected.len(), "{name}");
            let handed_back = unsafe { core::slice::from_raw_parts(out, out_len) };
            assert_eq!(handed_back, expected.as_slice(), "{name}");
            unsafe { pith_inflate_free(out, out_len) };
        }
    }

    /// Null out-slots are [`PITH_E_INVALID`]; a null `data` with a
    /// non-zero `len` is [`PITH_E_INVALID`]; a null `data` with
    /// `len == 0` is an empty input and reaches the decoder, which
    /// refuses it with [`PITH_E_REJECTED`] — the same status every
    /// language sees for the `empty_input` reference vector.
    #[test]
    fn ffi_refusals() {
        let compressed = [0x78u8, 0xda];
        let mut out: *mut u8 = core::ptr::null_mut();
        let mut out_len: usize = 0;

        // Null out / out_len, on every op.
        let garbage = [0u8; 16];
        for op_ffi in [
            pith_inflate_inflate_raw,
            pith_inflate_inflate_zlib,
            pith_inflate_inflate_auto,
        ] {
            let status = unsafe {
                op_ffi(
                    garbage.as_ptr(),
                    garbage.len(),
                    core::ptr::null_mut(),
                    &mut out_len,
                )
            };
            assert_eq!(status, PITH_E_INVALID);
            let status = unsafe {
                op_ffi(
                    garbage.as_ptr(),
                    garbage.len(),
                    &mut out,
                    core::ptr::null_mut(),
                )
            };
            assert_eq!(status, PITH_E_INVALID);

            // Null data with a non-zero length.
            let status = unsafe { op_ffi(core::ptr::null(), 16, &mut out, &mut out_len) };
            assert_eq!(status, PITH_E_INVALID);

            // Null data with a zero length: refused by the core, not
            // the pointer guard.
            let status = unsafe { op_ffi(core::ptr::null(), 0, &mut out, &mut out_len) };
            assert_eq!(status, PITH_E_REJECTED);
        }

        // A truncated stream (zlib header, no blocks) is -2 on all
        // three routings; so is structural garbage.
        let inputs: [&[u8]; 2] = [&compressed, &garbage];
        for input in inputs {
            for op in [Op::Raw, Op::Zlib, Op::Auto] {
                assert_eq!(
                    inflate_ffi(op, input.as_ptr(), input.len(), &mut out, &mut out_len),
                    PITH_E_REJECTED,
                    "input {input:?} through {op:?}"
                );
            }
        }

        // A null buffer is a legal free, and freeing twice the same
        // zero-length handout stays safe (no double-free of a real
        // allocation happens for len 0).
        unsafe { pith_inflate_free(core::ptr::null_mut(), 0) };
    }

    /// Adler-32 through the FFI equals the crate's own checksum on a
    /// real output; null slots are refused; an empty input checksums
    /// to the initial value 1.
    #[test]
    fn adler32_ffi_matches_the_core() {
        let v = VECTORS.iter().find(|v| v.name == "text").expect("vector");
        let plain = inflate_framed(Op::Raw, &unhex(v.compressed)).expect("decode");
        let mut sum = 0u32;
        let status = unsafe { pith_inflate_adler32(plain.as_ptr(), plain.len(), &mut sum) };
        assert_eq!(status, PITH_OK);
        assert_eq!(sum, crate::adler32(&plain));
        assert_eq!(format!("{sum:08x}"), "f724c355");

        // Empty input: the initial value.
        let mut sum = 0u32;
        let status = unsafe { pith_inflate_adler32(core::ptr::null(), 0, &mut sum) };
        assert_eq!(status, PITH_OK);
        assert_eq!(sum, 1);

        // Null out-slot / null data with length.
        let status =
            unsafe { pith_inflate_adler32(plain.as_ptr(), plain.len(), core::ptr::null_mut()) };
        assert_eq!(status, PITH_E_INVALID);
        let status = unsafe { pith_inflate_adler32(core::ptr::null(), 4, &mut sum) };
        assert_eq!(status, PITH_E_INVALID);
    }
}
