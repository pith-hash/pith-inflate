//! Fake-JNI-environment coverage for the glue in `src/ffi_jni.rs`.
//!
//! `src/ffi_jni.rs` is compiled out of the unit-test build (the
//! `#[no_mangle]` exports would collide with the unit-test binary), so
//! this integration test drives every export through an `unsafe extern`
//! declaration against a synthetic environment: a zeroed function
//! table whose slots the glue calls carry test-local implementations
//! backed by in-test buffers. The real-JVM proof is the Java suite
//! (`sdk/java`, `mvn test` against the built cdylib); this file keeps
//! the glue executed and visible to the coverage gate with zero new
//! dependencies (the suite's `check-zero-deps.py` gate forbids
//! registry crates, so the plain `std` mutexes stay unwrapped here).

#![allow(unsafe_code)]
// The JNI typedefs keep the jni.h spelling.
#![allow(non_camel_case_types)]

use core::ffi::c_void;
use std::sync::{LazyLock, Mutex};

use pith_inflate::ffi::{PITH_E_INVALID, PITH_E_REJECTED, PITH_OK};
use pith_inflate::{adler32, inflate_gzip, inflate_raw};

type JNIEnv = *const FakeTable;
type JArray = *mut c_void;
type JIntArray = *mut c_void;
type JClass = *mut c_void;
type jbyte = i8;
type jint = i32;
type jlong = i64;

/// Mirror of `src/ffi_jni.rs`'s function table — the same slot
/// positions (`GetArrayLength` = 171, `NewByteArray` = 176,
/// `GetByteArrayRegion` = 200, `SetByteArrayRegion` = 208,
/// `SetIntArrayRegion` = 211, four reserved pointers in the prefix).
#[repr(C)]
struct FakeTable {
    /// Slots 0..=170.
    _prefix: [*mut c_void; 171],
    /// Slot 171.
    get_array_length: unsafe extern "system" fn(env: *mut JNIEnv, array: JArray) -> jint,
    /// Slots 172..=175.
    _gap_before_new_byte_array: [*mut c_void; 4],
    /// Slot 176.
    new_byte_array: unsafe extern "system" fn(env: *mut JNIEnv, len: jint) -> JArray,
    /// Slots 177..=199.
    _gap_before_byte_region: [*mut c_void; 23],
    /// Slot 200.
    get_byte_array_region: unsafe extern "system" fn(
        env: *mut JNIEnv,
        array: JArray,
        start: jint,
        len: jint,
        buf: *mut jbyte,
    ),
    /// Slots 201..=207.
    _gap_before_set_byte_region: [*mut c_void; 7],
    /// Slot 208.
    set_byte_array_region: unsafe extern "system" fn(
        env: *mut JNIEnv,
        array: JArray,
        start: jint,
        len: jint,
        buf: *const jbyte,
    ),
    /// Slots 209..=210.
    _gap_before_int_region: [*mut c_void; 2],
    /// Slot 211.
    set_int_array_region: unsafe extern "system" fn(
        env: *mut JNIEnv,
        array: JIntArray,
        start: jint,
        len: jint,
        buf: *const jint,
    ),
}

// The exports under test (linked from the crate's rlib).
unsafe extern "system" {
    fn Java_hash_pith_inflate_PithInflate_inflateRawNative(
        env: *mut JNIEnv,
        class: JClass,
        data: JArray,
        status: JIntArray,
    ) -> JArray;
    fn Java_hash_pith_inflate_PithInflate_inflateZlibNative(
        env: *mut JNIEnv,
        class: JClass,
        data: JArray,
        status: JIntArray,
    ) -> JArray;
    fn Java_hash_pith_inflate_PithInflate_inflateAutoNative(
        env: *mut JNIEnv,
        class: JClass,
        data: JArray,
        status: JIntArray,
    ) -> JArray;
    fn Java_hash_pith_inflate_PithInflate_inflateGzipNative(
        env: *mut JNIEnv,
        class: JClass,
        data: JArray,
        status: JIntArray,
    ) -> JArray;
    fn Java_hash_pith_inflate_PithInflate_adler32Native(
        env: *mut JNIEnv,
        class: JClass,
        data: JArray,
        status: JIntArray,
    ) -> jlong;
}

/// The state of one native call under test: the input bytes the fake
/// `GetByteArrayRegion` hands out, and the output `byte[]` the fake
/// `NewByteArray`/`SetByteArrayRegion` pair materializes.
struct FakeCall {
    input: Vec<u8>,
    negative_length: bool,
    out_bytes: Vec<u8>,
    out_len: usize,
    out_ints: Vec<i32>,
}

static CALL: LazyLock<Mutex<Option<FakeCall>>> = LazyLock::new(|| Mutex::new(None));
static SERIAL: Mutex<()> = Mutex::new(());

/// `GetArrayLength` (slot 171): the current input's length.
unsafe extern "system" fn fake_get_array_length(_env: *mut JNIEnv, _array: JArray) -> jint {
    let guard = CALL.lock().unwrap();
    let current = guard.as_ref().expect("no fake call state installed");
    if current.negative_length {
        -1
    } else {
        current.input.len() as jint
    }
}

/// `NewByteArray` (slot 176): allocates the fake array's backing store
/// and returns a non-null sentinel handle.
unsafe extern "system" fn fake_new_byte_array(_env: *mut JNIEnv, len: jint) -> JArray {
    let mut guard = CALL.lock().unwrap();
    let current = guard.as_mut().expect("no fake call state installed");
    current.out_len = len as usize;
    current.out_bytes = vec![0u8; len as usize];
    1usize as JArray
}

/// `GetByteArrayRegion` (slot 200): copies the current input bytes.
unsafe extern "system" fn fake_get_byte_array_region(
    _env: *mut JNIEnv,
    _array: JArray,
    start: jint,
    len: jint,
    buf: *mut jbyte,
) {
    let guard = CALL.lock().unwrap();
    let current = guard.as_ref().expect("no fake call state installed");
    let start = start as usize;
    let bytes = &current.input[start..start + len as usize];
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf.cast(), bytes.len()) };
}

/// `SetByteArrayRegion` (slot 208): copies into the fake array's
/// backing store.
unsafe extern "system" fn fake_set_byte_array_region(
    _env: *mut JNIEnv,
    _array: JArray,
    start: jint,
    len: jint,
    buf: *const jbyte,
) {
    let mut guard = CALL.lock().unwrap();
    let current = guard.as_mut().expect("no fake call state installed");
    let start = start as usize;
    let slice = unsafe { std::slice::from_raw_parts(buf.cast(), len as usize) };
    current.out_bytes[start..start + len as usize].copy_from_slice(slice);
}

/// `SetIntArrayRegion` (slot 211): copies into the target array —
/// faithful to the JVM — and records the written values.
unsafe extern "system" fn fake_set_int_array_region(
    _env: *mut JNIEnv,
    array: JIntArray,
    _start: jint,
    len: jint,
    buf: *const jint,
) {
    unsafe { std::ptr::copy_nonoverlapping(buf, array as *mut jint, len as usize) };
    let mut guard = CALL.lock().unwrap();
    let current = guard.as_mut().expect("no fake call state installed");
    for i in 0..len as usize {
        current.out_ints.push(unsafe { *buf.add(i) });
    }
}

/// A zero-initialized `FakeTable`, leaked.
///
/// Raw `alloc_zeroed` bytes rather than `mem::zeroed`: the latter
/// runtime-refuses zeroed fn-pointer fields, while the former is just
/// memory — every slot the glue calls is assigned below before use.
fn zeroed_table() -> *mut FakeTable {
    let raw = unsafe { std::alloc::alloc_zeroed(std::alloc::Layout::new::<FakeTable>()) };
    assert!(!raw.is_null(), "alloc_zeroed failed");
    raw.cast::<FakeTable>()
}

/// Runs `f` against a synthetic environment backed by `input`,
/// returning its result and the state the fake slots recorded.
///
/// The big lock serializes sections across test threads: the fake
/// table callbacks address this one global call state.
fn with_fake_env<T>(input: Vec<u8>, negative_length: bool, f: impl FnOnce(*mut JNIEnv) -> T) -> T {
    let _serial = SERIAL.lock().unwrap();
    *CALL.lock().unwrap() = Some(FakeCall {
        input,
        negative_length,
        out_bytes: Vec::new(),
        out_len: usize::MAX,
        out_ints: Vec::new(),
    });

    let table = zeroed_table();
    unsafe {
        (*table).get_array_length = fake_get_array_length;
        (*table).new_byte_array = fake_new_byte_array;
        (*table).get_byte_array_region = fake_get_byte_array_region;
        (*table).set_byte_array_region = fake_set_byte_array_region;
        (*table).set_int_array_region = fake_set_int_array_region;
    }
    let functions: *const FakeTable = table;
    let env: *mut JNIEnv = Box::into_raw(Box::new(functions));

    let result = f(env);
    CALL.lock().unwrap().take().expect("fake call state");
    result
}

/// A live one-element status array; read it back with [`read_status`].
fn status_slot() -> JIntArray {
    Box::into_raw(Box::new([i32::MIN; 1])) as JIntArray
}

/// Reads the slot's content and releases it.
fn read_status(slot: JIntArray) -> i32 {
    let value = unsafe { *(slot as *mut jint) };
    unsafe { drop(Box::from_raw(slot as *mut jint)) };
    value
}

/// The output `byte[]` the most recent fake call materialized.
fn take_out_bytes() -> (Vec<u8>, usize) {
    let mut guard = CALL.lock().unwrap();
    let current = guard.as_mut().expect("no fake call state installed");
    (current.out_bytes.clone(), current.out_len)
}

/// A non-null opaque array handle (the glue only checks nullness).
const SOME_ARRAY: JArray = 1usize as JArray;

/// The `one_byte` raw vector's compressed bytes and plaintext, from
/// the committed corpus.
fn one_byte_raw() -> (Vec<u8>, Vec<u8>) {
    let vector = vectors::VECTORS
        .iter()
        .find(|v| v.name == "one_byte")
        .expect("vector");
    (unhex(vector.compressed), unhex(vector.plain))
}

/// The `one_byte/zlib` vector, likewise.
fn one_byte_zlib() -> (Vec<u8>, Vec<u8>) {
    let vector = vectors::VECTORS
        .iter()
        .find(|v| v.name == "one_byte/zlib")
        .expect("vector");
    (unhex(vector.compressed), unhex(vector.plain))
}

/// The `empty/zlib` vector — a stream whose decode is a zero-length
/// output, exercising the empty `byte[]` construction.
fn empty_zlib() -> (Vec<u8>, Vec<u8>) {
    let vector = vectors::VECTORS
        .iter()
        .find(|v| v.name == "empty/zlib")
        .expect("vector");
    (unhex(vector.compressed), unhex(vector.plain))
}

/// The `gzip/hello` fixture — the committed `.gz` bytes the Java suite
/// replays through the same export.
fn gzip_hello() -> Vec<u8> {
    std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/gzip/hello.gz"
    ))
    .expect("fixture")
}

/// Decodes the lowercase-hex encoding the corpus module uses.
fn unhex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).expect("hex digit"))
        .collect()
}

#[path = "vectors.rs"]
mod vectors;
use vectors::VECTORS;

#[test]
fn jni_inflate_raw_matches_the_corpus_vector() {
    let (compressed, plain) = one_byte_raw();
    let (out, status) = with_fake_env(compressed, false, |env| unsafe {
        let slot = status_slot();
        let array = Java_hash_pith_inflate_PithInflate_inflateRawNative(
            env,
            std::ptr::null_mut(),
            SOME_ARRAY,
            slot,
        );
        assert!(!array.is_null());
        let (bytes, len) = take_out_bytes();
        assert_eq!(len, plain.len());
        (bytes, read_status(slot))
    });
    assert_eq!(status, PITH_OK);
    assert_eq!(out, plain);
    // The one-shot decode of the same bytes agrees.
    assert_eq!(
        inflate_raw(
            &unhex(
                VECTORS
                    .iter()
                    .find(|v| v.name == "one_byte")
                    .expect("v")
                    .compressed
            ),
            &pith_inflate::Limits::default()
        )
        .expect("one-shot"),
        plain
    );
}

#[test]
fn jni_inflate_zlib_and_auto_agree() {
    let (compressed, plain) = one_byte_zlib();
    for which in ["zlib", "auto"] {
        let (out, status) = with_fake_env(compressed.clone(), false, |env| unsafe {
            let slot = status_slot();
            let array = match which {
                "zlib" => Java_hash_pith_inflate_PithInflate_inflateZlibNative(
                    env,
                    std::ptr::null_mut(),
                    SOME_ARRAY,
                    slot,
                ),
                _ => Java_hash_pith_inflate_PithInflate_inflateAutoNative(
                    env,
                    std::ptr::null_mut(),
                    SOME_ARRAY,
                    slot,
                ),
            };
            assert!(!array.is_null(), "{which}");
            let (bytes, _) = take_out_bytes();
            (bytes, read_status(slot))
        });
        assert_eq!(status, PITH_OK, "{which}");
        assert_eq!(out, plain, "{which}");
    }
}

#[test]
fn jni_inflate_gzip_matches_the_fixture() {
    let container = gzip_hello();
    let expected = inflate_gzip(&container, &pith_inflate::Limits::default()).expect("one-shot");
    let (out, status) = with_fake_env(container, false, |env| unsafe {
        let slot = status_slot();
        let array = Java_hash_pith_inflate_PithInflate_inflateGzipNative(
            env,
            std::ptr::null_mut(),
            SOME_ARRAY,
            slot,
        );
        assert!(!array.is_null());
        let (bytes, len) = take_out_bytes();
        assert_eq!(len, expected.len());
        (bytes, read_status(slot))
    });
    assert_eq!(status, PITH_OK);
    assert_eq!(out, expected);
}

#[test]
fn jni_empty_output_builds_an_empty_byte_array() {
    let (compressed, plain) = empty_zlib();
    assert!(plain.is_empty());
    let (out, len, status) = with_fake_env(compressed, false, |env| unsafe {
        let slot = status_slot();
        let array = Java_hash_pith_inflate_PithInflate_inflateZlibNative(
            env,
            std::ptr::null_mut(),
            SOME_ARRAY,
            slot,
        );
        assert!(!array.is_null());
        let (bytes, len) = take_out_bytes();
        (bytes, len, read_status(slot))
    });
    assert_eq!(status, PITH_OK);
    assert_eq!(len, 0);
    assert!(out.is_empty());
}

#[test]
fn jni_adler32_matches_the_crate_checksum() {
    let (compressed, _plain) = one_byte_raw();
    // The export checksums the bytes handed to it — the compressed
    // stream here — exactly like the C ABI's `pith_inflate_adler32`.
    let expected = adler32(&compressed);
    let (sum, status) = with_fake_env(compressed, false, |env| unsafe {
        let slot = status_slot();
        let sum = Java_hash_pith_inflate_PithInflate_adler32Native(
            env,
            std::ptr::null_mut(),
            SOME_ARRAY,
            slot,
        );
        (sum, read_status(slot))
    });
    assert_eq!(status, PITH_OK);
    assert_eq!(sum, i64::from(expected));
}

#[test]
fn jni_null_data_is_invalid_on_every_inflate_export() {
    for which in ["raw", "zlib", "auto", "gzip", "adler"] {
        let (result, status) = with_fake_env(Vec::new(), false, |env| unsafe {
            let slot = status_slot();
            let value = match which {
                "raw" => Java_hash_pith_inflate_PithInflate_inflateRawNative(
                    env,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    slot,
                ),
                "zlib" => Java_hash_pith_inflate_PithInflate_inflateZlibNative(
                    env,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    slot,
                ),
                "auto" => Java_hash_pith_inflate_PithInflate_inflateAutoNative(
                    env,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    slot,
                ),
                "gzip" => Java_hash_pith_inflate_PithInflate_inflateGzipNative(
                    env,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    slot,
                ),
                _ => {
                    let sum = Java_hash_pith_inflate_PithInflate_adler32Native(
                        env,
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                        slot,
                    );
                    assert_eq!(sum, 0, "{which}");
                    std::ptr::null_mut()
                }
            };
            assert!(value.is_null(), "{which}");
            (value, read_status(slot))
        });
        assert!(result.is_null(), "{which}");
        assert_eq!(status, PITH_E_INVALID, "{which}");
    }
}

#[test]
fn jni_negative_array_length_is_invalid() {
    let (out, status) = with_fake_env(b"whatever".to_vec(), true, |env| unsafe {
        let slot = status_slot();
        let array = Java_hash_pith_inflate_PithInflate_inflateRawNative(
            env,
            std::ptr::null_mut(),
            SOME_ARRAY,
            slot,
        );
        (array, read_status(slot))
    });
    assert!(out.is_null());
    assert_eq!(status, PITH_E_INVALID);
}

#[test]
fn jni_rejected_stream_is_rejected_not_crashing() {
    let (out, status) = with_fake_env(b"\x00\x00".to_vec(), false, |env| unsafe {
        let slot = status_slot();
        let array = Java_hash_pith_inflate_PithInflate_inflateRawNative(
            env,
            std::ptr::null_mut(),
            SOME_ARRAY,
            slot,
        );
        (array, read_status(slot))
    });
    assert!(out.is_null());
    assert_eq!(status, PITH_E_REJECTED);
}
