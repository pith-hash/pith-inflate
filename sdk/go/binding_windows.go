// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pith-hash

//go:build windows

package pithinflate

import (
	"fmt"
	"syscall"
	"unsafe"
)

// openProc loads libPath and resolves name. The library is released
// before returning: on Windows FreeLibrary unmaps the cdylib, so the
// proc must be used (and its buffer copied out) inside the caller.
func openProc(libPath, name string) (proc uintptr, release func(), err error) {
	lib, err := syscall.LoadLibrary(libPath)
	if err != nil {
		return 0, nil, fmt.Errorf("pithinflate: LoadLibrary(%s): %w", libPath, err)
	}
	release = func() { syscall.FreeLibrary(lib) }
	proc, err = syscall.GetProcAddress(lib, name)
	if err != nil {
		release()
		return 0, nil, fmt.Errorf("pithinflate: symbol %s missing from %s: %w", name, libPath, err)
	}
	return proc, release, nil
}

// ffiInflate loads the cdylib with LoadLibrary (absolute path, no PATH
// involvement), resolves symbol and calls it. The returned buffer stays
// alive in the cdylib until ffiFree.
func ffiInflate(libPath, symbol string, data *byte, n int, out **byte, outLen *uintptr) (int32, error) {
	proc, release, err := openProc(libPath, symbol)
	if err != nil {
		return 0, err
	}
	defer release()

	var cOut *byte
	var cLen uintptr
	rc, _, _ := syscall.SyscallN(proc,
		uintptr(unsafe.Pointer(data)),
		uintptr(n),
		uintptr(unsafe.Pointer(&cOut)),
		uintptr(unsafe.Pointer(&cLen)),
	)
	*out = cOut
	*outLen = cLen
	return int32(rc), nil
}

// ffiAdler32 resolves pith_inflate_adler32 and writes the checksum
// through the typed out slot.
func ffiAdler32(libPath string, data *byte, n int, out *uint32) (int32, error) {
	proc, release, err := openProc(libPath, "pith_inflate_adler32")
	if err != nil {
		return 0, err
	}
	defer release()

	var sum uint32
	rc, _, _ := syscall.SyscallN(proc,
		uintptr(unsafe.Pointer(data)),
		uintptr(n),
		uintptr(unsafe.Pointer(&sum)),
	)
	*out = sum
	return int32(rc), nil
}

// ffiFree resolves pith_inflate_free and releases a buffer handed out
// by ffiInflate. Null is accepted (the cdylib ignores it).
func ffiFree(libPath string, ptr *byte, n uintptr) {
	proc, release, err := openProc(libPath, "pith_inflate_free")
	if err != nil {
		return // the library vanished mid-flight; nothing to free
	}
	defer release()
	syscall.SyscallN(proc, uintptr(unsafe.Pointer(ptr)), n)
}
