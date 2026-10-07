// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pith-hash

//go:build !windows && !cgo

package pithinflate

import "fmt"

// ffiInflate is unavailable without cgo on unix: there is no
// pure-Go dlopen in the standard library. Build with CGO_ENABLED=1
// (the CD pipeline always does).
func ffiInflate(string, string, *byte, int, **byte, *uintptr) (int32, error) {
	return 0, fmt.Errorf("pithinflate: cgo is required to load the cdylib on this platform (build with CGO_ENABLED=1)")
}

// ffiAdler32 mirrors the unavailable inflate.
func ffiAdler32(string, *byte, int, *uint32) (int32, error) {
	return 0, fmt.Errorf("pithinflate: cgo is required to load the cdylib on this platform (build with CGO_ENABLED=1)")
}

// ffiFree mirrors the unavailable inflate.
func ffiFree(string, *byte, uintptr) {}
