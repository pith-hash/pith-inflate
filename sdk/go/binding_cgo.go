// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pith-hash

//go:build !windows && cgo

package pithinflate

/*
#include <dlfcn.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

typedef int32_t (*pith_inflate_fn)(const uint8_t *, size_t, uint8_t **, size_t *);
typedef int32_t (*pith_adler_fn)(const uint8_t *, size_t, uint32_t *);
typedef void (*pith_free_fn)(uint8_t *, size_t);

static int32_t pith_call_inflate(void *fn, const uint8_t *data, size_t len,
                                 uint8_t **out, size_t *out_len) {
    return ((pith_inflate_fn)fn)(data, len, out, out_len);
}

static int32_t pith_call_adler(void *fn, const uint8_t *data, size_t len, uint32_t *out) {
    return ((pith_adler_fn)fn)(data, len, out);
}

static void pith_call_free(void *fn, uint8_t *ptr, size_t len) {
    ((pith_free_fn)fn)(ptr, len);
}
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// ffiSymbol dlsyms one exported symbol of an open cdylib handle.
func ffiSymbol(handle unsafe.Pointer, name, libPath string) (unsafe.Pointer, error) {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	sym := C.dlsym(handle, cName)
	if sym == nil {
		return nil, fmt.Errorf("pithinflate: symbol %s missing from %s", name, libPath)
	}
	return sym, nil
}

// openCdylib dlopens libPath with error text surfaced verbatim.
func openCdylib(libPath string) (unsafe.Pointer, error) {
	cPath := C.CString(libPath)
	defer C.free(unsafe.Pointer(cPath))
	handle := C.dlopen(cPath, C.RTLD_NOW|C.RTLD_LOCAL)
	if handle == nil {
		msg := "unknown dlopen failure"
		if e := C.dlerror(); e != nil {
			msg = C.GoString(e)
		}
		return nil, fmt.Errorf("pithinflate: dlopen(%s): %s", libPath, msg)
	}
	return handle, nil
}

// ffiInflate opens the cdylib, resolves symbol and calls it. The
// handle is released before returning; repeated calls reuse the
// loader's own refcount.
func ffiInflate(libPath, symbol string, data *byte, n int, out **byte, outLen *uintptr) (int32, error) {
	handle, err := openCdylib(libPath)
	if err != nil {
		return 0, err
	}
	defer C.dlclose(handle)

	sym, err := ffiSymbol(handle, symbol, libPath)
	if err != nil {
		return 0, err
	}
	var cOut *C.uint8_t
	var cLen C.size_t
	rc := C.pith_call_inflate(sym, (*C.uint8_t)(unsafe.Pointer(data)), C.size_t(n), &cOut, &cLen)
	*out = (*byte)(unsafe.Pointer(cOut))
	*outLen = uintptr(cLen)
	return int32(rc), nil
}

// ffiAdler32 opens the cdylib, resolves pith_inflate_adler32 and calls
// it with a typed out slot.
func ffiAdler32(libPath string, data *byte, n int, out *uint32) (int32, error) {
	handle, err := openCdylib(libPath)
	if err != nil {
		return 0, err
	}
	defer C.dlclose(handle)

	sym, err := ffiSymbol(handle, "pith_inflate_adler32", libPath)
	if err != nil {
		return 0, err
	}
	var sum C.uint32_t
	rc := C.pith_call_adler(sym, (*C.uint8_t)(unsafe.Pointer(data)), C.size_t(n), &sum)
	*out = uint32(sum)
	return int32(rc), nil
}

// ffiFree opens the cdylib, resolves pith_inflate_free and releases a
// buffer handed out by ffiInflate. Null is accepted (the cdylib
// ignores it), matching the C contract.
func ffiFree(libPath string, ptr *byte, n uintptr) {
	handle, err := openCdylib(libPath)
	if err != nil {
		return // the library vanished mid-flight; nothing to free
	}
	defer C.dlclose(handle)

	sym, err := ffiSymbol(handle, "pith_inflate_free", libPath)
	if err != nil {
		return
	}
	C.pith_call_free(sym, (*C.uint8_t)(unsafe.Pointer(ptr)), C.size_t(n))
}
