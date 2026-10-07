// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pith-hash

// Package pithinflate provides Go bindings for the pith-inflate Rust
// cdylib: DEFLATE and zlib decompression under hard size limits.
//
// The single Rust core (built by `cargo build --release`) is loaded at
// runtime; the package carries zero module dependencies. On unix the
// cdylib is opened with dlopen through cgo, on Windows with
// LoadLibrary through the standard syscall package — both resolve the
// library through the same discovery chain, so `go build ./... &&
// go test ./...` works unchanged on every OS the CD matrix builds.
//
// Discovery order (the suite's cdylib convention):
//
//  1. PITH_CDYLIB — an explicit cdylib file path;
//  2. PITH_CDYLIB_DIR — a directory scanned for the cdylib names (the
//     CD pipeline points this at target/release);
//  3. <repo root>/target/release — the repository working-tree layout,
//     anchored at this package's source directory, so a source
//     checkout runs against a local cargo build unconfigured.
//
// The FFI surface is three decompress operations, one checksum and one
// free: InflateRaw, InflateZlib and InflateAuto run the crate's
// conservative default limits (64 MiB of input, 64 MiB of output) and
// return the decompressed bytes as a Go copy (the handed-out cdylib
// buffer is released before returning); Adler32 checksums any buffer
// without allocating. A stream the decoder refuses comes back as an
// *FfiError with Status StatusRejected — never a panic.
package pithinflate

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"unsafe"
)

// Status codes returned by the cdylib's C ABI.
const (
	// StatusOK: success.
	StatusOK int32 = 0
	// StatusInvalid: a caller argument is invalid (a null pointer).
	StatusInvalid int32 = -1
	// StatusRejected: the core decoder refused the input (a malformed,
	// truncated or over-limit stream, or a framing the crate refuses).
	StatusRejected int32 = -2
)

// cdylibNames are the file names cargo may drop into the build
// directory, per platform (windows / linux / macOS).
var cdylibNames = []string{"pith_inflate.dll", "libpith_inflate.so", "libpith_inflate.dylib"}

// FfiError reports a non-zero status code from the cdylib.
type FfiError struct {
	// Op is the FFI operation name.
	Op string
	// Status is the raw status code the FFI returned.
	Status int32
}

func (e *FfiError) Error() string {
	kind := "unknown failure"
	switch e.Status {
	case StatusInvalid:
		kind = "invalid argument"
	case StatusRejected:
		kind = "input rejected"
	}
	return fmt.Sprintf("%s failed: %s (status %d)", e.Op, kind, e.Status)
}

// FindCdylib locates the cdylib through the suite's discovery chain.
func FindCdylib() (string, error) {
	if p := os.Getenv("PITH_CDYLIB"); p != "" {
		if st, err := os.Stat(p); err == nil && st.Mode().IsRegular() {
			return filepath.Abs(p)
		}
	}
	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		return "", fmt.Errorf("pithinflate: cannot locate the package source directory")
	}
	pkgDir := filepath.Dir(thisFile)
	repoRoot := filepath.Dir(filepath.Dir(pkgDir)) // sdk/go -> sdk -> repo root

	var dirs []string
	if env := os.Getenv("PITH_CDYLIB_DIR"); env != "" {
		dirs = append(dirs, env)
		if !filepath.IsAbs(env) {
			dirs = append(dirs, filepath.Join(repoRoot, env))
		}
	}
	dirs = append(dirs, filepath.Join(repoRoot, "target", "release"))
	for _, dir := range dirs {
		for _, name := range cdylibNames {
			p := filepath.Join(dir, name)
			if st, err := os.Stat(p); err == nil && st.Mode().IsRegular() {
				return p, nil
			}
		}
	}
	return "", fmt.Errorf(
		"pithinflate: no cdylib found (searched PITH_CDYLIB, PITH_CDYLIB_DIR and <repo>/target/release); run `cargo build --release` first",
	)
}

// locate resolves the cdylib path once per process.
var locate = sync.OnceValues(FindCdylib)

// InflateRaw decompresses a raw RFC 1951 DEFLATE stream (trailing
// bytes after the final block are ignored). The returned slice is a Go
// copy; the handed-out cdylib buffer is released before returning. An
// error carries StatusRejected for any malformed, truncated or
// over-limit input.
func InflateRaw(data []byte) ([]byte, error) {
	return inflate("pith_inflate_inflate_raw", data)
}

// InflateZlib decompresses a zlib (RFC 1950) stream: two-byte header,
// DEFLATE payload, big-endian Adler-32 trailer. Buffer handling and
// error semantics are those of InflateRaw.
func InflateZlib(data []byte) ([]byte, error) {
	return inflate("pith_inflate_inflate_zlib", data)
}

// InflateAuto decompresses either framing, sniffing the two-byte zlib
// header; gzip is recognised and refused (StatusRejected), never
// mis-decoded.
func InflateAuto(data []byte) ([]byte, error) {
	return inflate("pith_inflate_inflate_auto", data)
}

// inflate resolves the cdylib, runs one decompress symbol and copies
// the handed-out buffer into a Go slice before releasing it.
func inflate(symbol string, data []byte) ([]byte, error) {
	libPath, err := locate()
	if err != nil {
		return nil, err
	}
	var out *byte
	var outLen uintptr
	var dataPtr *byte
	if len(data) > 0 {
		dataPtr = &data[0]
	}
	status, err := ffiInflate(libPath, symbol, dataPtr, len(data), &out, &outLen)
	if err != nil {
		return nil, err
	}
	if status != StatusOK {
		return nil, &FfiError{Op: symbol, Status: status}
	}
	buf := make([]byte, outLen)
	copy(buf, unsafe.Slice(out, outLen))
	ffiFree(libPath, out, outLen)
	return buf, nil
}

// Adler32 computes the Adler-32 checksum (RFC 1950 section 9) of data
// without allocating. The checksum of an empty input is 1, so
// Adler32(nil) == 1.
func Adler32(data []byte) (uint32, error) {
	libPath, err := locate()
	if err != nil {
		return 0, err
	}
	var dataPtr *byte
	if len(data) > 0 {
		dataPtr = &data[0]
	}
	var sum uint32
	status, err := ffiAdler32(libPath, dataPtr, len(data), &sum)
	if err != nil {
		return 0, err
	}
	if status != StatusOK {
		return 0, &FfiError{Op: "pith_inflate_adler32", Status: status}
	}
	return sum, nil
}
