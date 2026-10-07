//! The C ABI surface exercised from OUTSIDE the crate, the way the
//! language SDKs bind it: every export called through the rlib's
//! public `ffi` module against the committed vectors, plus the refusal
//! paths. Complements the in-crate `src/ffi.rs` unit tests, which
//! reach the same code through the test build — having both means the
//! exports stay covered (and pinned) whichever binary the coverage
//! merge keys on.

use pith_inflate::ffi::{
    PITH_E_INVALID, PITH_E_REJECTED, PITH_OK, pith_inflate_adler32, pith_inflate_free,
    pith_inflate_inflate_auto, pith_inflate_inflate_raw, pith_inflate_inflate_zlib,
};

/// The `text` vector of `reference.json`, pinned literally: its
/// compressed stream, its plaintext and its Adler-32 — the same pin
/// the crate's unit tests and the three language SDKs re-derive.
const TEXT_COMPRESSED: &[u8] = &[
    0x2b, 0xc9, 0x48, 0x55, 0x28, 0x2c, 0xcd, 0x4c, 0xce, 0x56, 0x48, 0x2a, 0xca, 0x2f, 0xcf, 0x53,
    0x48, 0xcb, 0xaf, 0x50, 0xc8, 0x2a, 0xcd, 0x2d, 0x28, 0x56, 0xc8, 0x2f, 0x4b, 0x2d, 0x52, 0x28,
    0x01, 0x4a, 0xe7, 0x24, 0x56, 0x55, 0x2a, 0xa4, 0xe4, 0xa7, 0xeb, 0x81, 0x79, 0xa3, 0x8a, 0x47,
    0x8c, 0x62, 0x00,
];
/// The `text` vector's plaintext: the pangram, 12 times (540 bytes).
fn text_plain() -> Vec<u8> {
    b"the quick brown fox jumps over the lazy dog. ".repeat(12)
}

/// The `empty/zlib` vector, pinned literally: the empty stream, the
/// empty output and the initial Adler-32 value.
const EMPTY_ZLIB: &[u8] = &[0x78, 0xda, 0x03, 0x00, 0x00, 0x00, 0x00, 0x01];

/// Decompresses `input` through the raw FFI and hands the buffer back
/// to `pith_inflate_free`, returning a caller-owned copy.
fn decode(
    op: unsafe extern "C" fn(*const u8, usize, *mut *mut u8, *mut usize) -> i32,
    input: &[u8],
) -> Result<Vec<u8>, i32> {
    let mut out: *mut u8 = core::ptr::null_mut();
    let mut out_len: usize = 0;
    let status = unsafe { op(input.as_ptr(), input.len(), &mut out, &mut out_len) };
    if status != PITH_OK {
        return Err(status);
    }
    let handed = unsafe { core::slice::from_raw_parts(out, out_len) };
    let copy = handed.to_vec();
    unsafe { pith_inflate_free(out, out_len) };
    Ok(copy)
}

/// Every decompress export reproduces the pinned vectors hex-exact,
/// and the FFI's own Adler-32 matches the recorded digests.
#[test]
fn exports_reproduce_the_pinned_vectors() {
    let text = text_plain();
    type Op = unsafe extern "C" fn(*const u8, usize, *mut *mut u8, *mut usize) -> i32;
    for (name, op, input, plain) in [
        (
            "text/raw",
            pith_inflate_inflate_raw as Op,
            TEXT_COMPRESSED,
            text.as_slice(),
        ),
        (
            "text/auto",
            pith_inflate_inflate_auto as Op,
            TEXT_COMPRESSED,
            text.as_slice(),
        ),
        (
            "empty/zlib",
            pith_inflate_inflate_zlib as Op,
            EMPTY_ZLIB,
            b"".as_slice(),
        ),
    ] {
        let out = decode(op, input).unwrap_or_else(|s| panic!("{name}: status {s}"));
        assert_eq!(out, plain, "{name}");
        let mut sum: u32 = 0;
        let status = unsafe { pith_inflate_adler32(out.as_ptr(), out.len(), &mut sum) };
        assert_eq!(status, PITH_OK, "{name}");
        let expected = if plain.is_empty() { 1 } else { 0xf724_c355 };
        assert_eq!(sum, expected, "{name}");
    }
}

/// Null out-slots and a null `data` with a non-zero `len` are caller
/// bugs (`-1`); a null `data` with `len == 0` is an empty input the
/// core refuses (`-2`); truncated and garbage streams are `-2` —
/// never a crash, from any export.
#[test]
fn exports_refuse_without_crashing() {
    let mut out: *mut u8 = core::ptr::null_mut();
    let mut out_len: usize = 0;
    let garbage = [0u8; 16];

    for op in [
        pith_inflate_inflate_raw,
        pith_inflate_inflate_zlib,
        pith_inflate_inflate_auto,
    ] {
        assert_eq!(
            unsafe {
                op(
                    garbage.as_ptr(),
                    garbage.len(),
                    core::ptr::null_mut(),
                    &mut out_len,
                )
            },
            PITH_E_INVALID
        );
        assert_eq!(
            unsafe {
                op(
                    garbage.as_ptr(),
                    garbage.len(),
                    &mut out,
                    core::ptr::null_mut(),
                )
            },
            PITH_E_INVALID
        );
        assert_eq!(
            unsafe { op(core::ptr::null(), 16, &mut out, &mut out_len) },
            PITH_E_INVALID
        );
        assert_eq!(
            unsafe { op(core::ptr::null(), 0, &mut out, &mut out_len) },
            PITH_E_REJECTED
        );
        assert_eq!(decode(op, b"\x78\xda"), Err(PITH_E_REJECTED));
        assert_eq!(decode(op, &garbage), Err(PITH_E_REJECTED));
    }

    let mut sum: u32 = 0;
    assert_eq!(
        unsafe { pith_inflate_adler32(garbage.as_ptr(), garbage.len(), core::ptr::null_mut()) },
        PITH_E_INVALID
    );
    assert_eq!(
        unsafe { pith_inflate_adler32(core::ptr::null(), 4, &mut sum) },
        PITH_E_INVALID
    );
    assert_eq!(
        unsafe { pith_inflate_adler32(core::ptr::null(), 0, &mut sum) },
        PITH_OK
    );
    assert_eq!(sum, 1);

    // A null buffer is a legal free; the empty/zlib handout frees with
    // its zero length.
    unsafe { pith_inflate_free(core::ptr::null_mut(), 0) };
    let empty = decode(pith_inflate_inflate_zlib, EMPTY_ZLIB).expect("empty/zlib");
    assert!(empty.is_empty());
}
