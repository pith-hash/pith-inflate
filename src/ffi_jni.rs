//! The JNI surface of `pith-inflate`: the `Java_hash_pith_inflate_*`
//! exports the Java SDK's `PithInflate` class binds through.
//!
//! The C ABI of [`crate::ffi`] is untouched: JNI requires exports named
//! `Java_<package>_<Class>_<method>`, so the Java-facing shims live
//! here and forward to the flat C functions. The Java-side contract:
//!
//! * the export takes the JNI environment first and the receiving
//!   class second (static methods);
//! * input bytes arrive as a `byte[]`, copied through the function
//!   table before any decode runs;
//! * the status (`PITH_OK`, `PITH_E_INVALID`, `PITH_E_REJECTED`)
//!   crosses back through the first slot of an `int[]` the caller
//!   provides;
//! * the decompressed bytes cross back as a fresh `byte[]` (null when
//!   the status is not [`PITH_OK`]); the Adler-32 crosses back
//!   bit-cast to `jlong` (0 unless the status is [`PITH_OK`]);
//! * a null `data` array maps to [`PITH_E_INVALID`] exactly as the C
//!   ABI maps a null pointer; a null environment or status array
//!   short-circuits to a zero return without touching memory (both
//!   are unreachable through the Java wrapper, which always passes
//!   live arrays from a live JVM).
//!
//! The suite is zero-third-party (CI's `check-zero-deps.py` fails any
//! registry crate), so the JNI function table is hand-declared below:
//! every slot is pointer-sized and the positions are the fixed
//! `JNINativeInterface_` member order of `jni.h`. The slot indices
//! were parsed mechanically from the JDK 21 header and are validated
//! end-to-end against a live JVM every time the Java suite runs
//! (`sdk/java`, `mvn test`).

#![allow(unsafe_code)]
// The JNI typedefs keep the jni.h spelling (jint, jbyte, ...).
#![allow(non_camel_case_types)]

use core::ffi::c_void;

use pith_digest::Result as DecodeResult;

use crate::ffi::{PITH_E_INVALID, PITH_E_REJECTED, PITH_OK};
use crate::{Limits, adler32, inflate_auto, inflate_gzip, inflate_raw, inflate_zlib};

/// A JNI environment handle — C-mode `JNIEnv*`, a pointer to the
/// function table.
type JNIEnv = *const JniTable;

/// Any Java array reference; the glue only checks nullness before
/// handing arrays through the table.
type JArray = *mut c_void;

/// A Java `int[]` reference.
type JIntArray = *mut c_void;

/// A Java class object reference (static methods receive the class).
type JClass = *mut c_void;

/// `jbyte` per `jni.h`.
type jbyte = i8;
/// `jint`/`jsize` per `jni.h`.
type jint = i32;
/// `jlong` per `jni.h`.
type jlong = i64;

/// The JNI function-table slots this module calls.
///
/// Underscore-prefixed gap fields hold the slots between the used ones
/// (slot = field position; the four reserved pointers are part of the
/// prefix). Slot indices parsed from the JDK 21 `include/jni.h`:
/// `GetArrayLength` = 171, `NewByteArray` = 176,
/// `GetByteArrayRegion` = 200, `SetByteArrayRegion` = 208,
/// `SetIntArrayRegion` = 211.
#[repr(C)]
struct JniTable {
    /// Slots 0..=170: the four reserved pointers through
    /// `ReleaseStringUTFChars`.
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

/// The JNI 1.1-era function table behind an environment handle.
///
/// # Safety
///
/// `env` must be a live JNI environment pointer.
unsafe fn table<'a>(env: *mut JNIEnv) -> &'a JniTable {
    // `env` points at the function-table pointer (C-mode `JNIEnv*`):
    // deref twice to reach the table itself.
    unsafe { &**env }
}

/// Copies a Java `byte[]` through the environment into an owned
/// buffer.
///
/// # Safety
///
/// `env` must be a live JNI environment and `array` a live `byte[]`
/// reference for the duration of the call; a null array is
/// [`PITH_E_INVALID`], mirroring the C ABI's null-pointer rule.
unsafe fn java_bytes(env: *mut JNIEnv, array: JArray) -> Result<Vec<u8>, i32> {
    if array.is_null() {
        return Err(PITH_E_INVALID);
    }
    let functions = unsafe { table(env) };
    let len = unsafe { (functions.get_array_length)(env, array) };
    if len < 0 {
        return Err(PITH_E_INVALID);
    }
    let mut bytes = vec![0u8; len as usize];
    unsafe { (functions.get_byte_array_region)(env, array, 0, len, bytes.as_mut_ptr().cast()) };
    Ok(bytes)
}

/// Writes `value` into the one-element `int[]` status slot.
///
/// # Safety
///
/// `status` must be a live `int[]` of length ≥ 1 (checked by the
/// caller).
unsafe fn set_status(env: *mut JNIEnv, status: JIntArray, value: jint) {
    let functions = unsafe { table(env) };
    unsafe { (functions.set_int_array_region)(env, status, 0, 1, &value) };
}

/// Builds a fresh Java `byte[]` carrying `bytes`, or null when the
/// environment refuses the allocation (unreachable through a live
/// JVM's short arrays).
///
/// # Safety
///
/// `env` must be a live JNI environment.
unsafe fn jbyte_array(env: *mut JNIEnv, bytes: &[u8]) -> JArray {
    let functions = unsafe { table(env) };
    let array = unsafe { (functions.new_byte_array)(env, bytes.len() as jint) };
    if array.is_null() {
        return core::ptr::null_mut();
    }
    unsafe {
        (functions.set_byte_array_region)(
            env,
            array,
            0,
            bytes.len() as jint,
            bytes.as_ptr().cast(),
        );
    }
    array
}

/// The shared body of the four decompress exports: copy the `byte[]`
/// in, run the framing's one-shot decode, and hand the output back as
/// a fresh `byte[]` with the status in `status[0]`.
///
/// # Safety
///
/// `env` must be a live JNI environment and `data`/`status` live Java
/// array references for the duration of the call.
unsafe fn inflate_native(
    env: *mut JNIEnv,
    data: JArray,
    status: JIntArray,
    op: fn(&[u8], &Limits) -> DecodeResult<Vec<u8>>,
) -> JArray {
    if env.is_null() || status.is_null() {
        return core::ptr::null_mut();
    }
    let status_slot = status;
    let bytes = match unsafe { java_bytes(env, data) } {
        Ok(bytes) => bytes,
        Err(code) => {
            unsafe { set_status(env, status_slot, code) };
            return core::ptr::null_mut();
        }
    };
    match op(&bytes, &Limits::default()) {
        Ok(out) => {
            unsafe { set_status(env, status_slot, PITH_OK) };
            unsafe { jbyte_array(env, &out) }
        }
        Err(_) => {
            // Every decode refusal — malformed, truncated, over-limit,
            // a framing the crate refuses — collapses to PITH_E_REJECTED
            // at this boundary, exactly as the C ABI does.
            unsafe { set_status(env, status_slot, PITH_E_REJECTED) };
            core::ptr::null_mut()
        }
    }
}

/// The Java binding of [`inflate_raw`]: a raw RFC 1951 DEFLATE stream
/// in, the decompressed bytes out as a fresh `byte[]`, the status
/// (`PITH_OK`, `PITH_E_INVALID` for a null array, `PITH_E_REJECTED`
/// for a refused stream) through `status[0]`.
///
/// # Safety
///
/// `env` must be a live JNI environment and `data`/`status` live Java
/// array references for the duration of the call.
//
// Private: the JVM links the export by symbol name; a public Rust
// signature over the private table type would trip `private_interfaces`.
#[unsafe(no_mangle)]
unsafe extern "system" fn Java_hash_pith_inflate_PithInflate_inflateRawNative(
    env: *mut JNIEnv,
    _class: JClass,
    data: JArray,
    status: JIntArray,
) -> JArray {
    unsafe { inflate_native(env, data, status, inflate_raw) }
}

/// The Java binding of [`inflate_zlib`] — the wire contract of
/// [`Java_hash_pith_inflate_PithInflate_inflateRawNative`].
///
/// # Safety
///
/// `env` must be a live JNI environment and `data`/`status` live Java
/// array references for the duration of the call.
#[unsafe(no_mangle)]
unsafe extern "system" fn Java_hash_pith_inflate_PithInflate_inflateZlibNative(
    env: *mut JNIEnv,
    _class: JClass,
    data: JArray,
    status: JIntArray,
) -> JArray {
    unsafe { inflate_native(env, data, status, inflate_zlib) }
}

/// The Java binding of [`inflate_auto`] — the wire contract of
/// [`Java_hash_pith_inflate_PithInflate_inflateRawNative`].
///
/// # Safety
///
/// `env` must be a live JNI environment and `data`/`status` live Java
/// array references for the duration of the call.
#[unsafe(no_mangle)]
unsafe extern "system" fn Java_hash_pith_inflate_PithInflate_inflateAutoNative(
    env: *mut JNIEnv,
    _class: JClass,
    data: JArray,
    status: JIntArray,
) -> JArray {
    unsafe { inflate_native(env, data, status, inflate_auto) }
}

/// The Java binding of [`inflate_gzip`] — the wire contract of
/// [`Java_hash_pith_inflate_PithInflate_inflateRawNative`].
///
/// # Safety
///
/// `env` must be a live JNI environment and `data`/`status` live Java
/// array references for the duration of the call.
#[unsafe(no_mangle)]
unsafe extern "system" fn Java_hash_pith_inflate_PithInflate_inflateGzipNative(
    env: *mut JNIEnv,
    _class: JClass,
    data: JArray,
    status: JIntArray,
) -> JArray {
    unsafe { inflate_native(env, data, status, inflate_gzip) }
}

/// The Java binding of [`adler32`]: the checksum bit-cast to `jlong`
/// (0 unless the status is [`PITH_OK`]) and the status through
/// `status[0]`.
///
/// # Safety
///
/// `env` must be a live JNI environment and `data`/`status` live Java
/// array references for the duration of the call.
#[unsafe(no_mangle)]
unsafe extern "system" fn Java_hash_pith_inflate_PithInflate_adler32Native(
    env: *mut JNIEnv,
    _class: JClass,
    data: JArray,
    status: JIntArray,
) -> jlong {
    if env.is_null() || status.is_null() {
        return 0;
    }
    let status_slot = status;
    match unsafe { java_bytes(env, data) } {
        Err(code) => {
            unsafe { set_status(env, status_slot, code) };
            0
        }
        Ok(bytes) => {
            unsafe { set_status(env, status_slot, PITH_OK) };
            i64::from(adler32(&bytes))
        }
    }
}
