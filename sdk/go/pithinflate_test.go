// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pith-hash

package pithinflate

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// repoRoot resolves the repository root relative to this package
// (sdk/go -> sdk -> repo root), the anchor for reference.json.
func repoRoot(t *testing.T) string {
	t.Helper()
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	if st, err := os.Stat(filepath.Join(root, "reference.json")); err != nil || st.IsDir() {
		t.Fatalf("reference.json not found at %s", root)
	}
	return root
}

// reference parses the committed reference.json: vectors is a list of
// good entries (compressed -> plain, pinned by Adler-32), badVectors a
// list of malformed streams.
func reference(t *testing.T) (vectors []struct {
	Name       string `json:"name"`
	Block      string `json:"block"`
	Compressed string `json:"compressed"`
	Plain      string `json:"plain"`
	Adler32    string `json:"adler32"`
}, bad []struct {
	Name  string `json:"name"`
	Input string `json:"input"`
	Kind  string `json:"kind"`
}) {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(repoRoot(t), "reference.json"))
	if err != nil {
		t.Fatal(err)
	}
	var parsed struct {
		Vectors []struct {
			Name       string `json:"name"`
			Block      string `json:"block"`
			Compressed string `json:"compressed"`
			Plain      string `json:"plain"`
			Adler32    string `json:"adler32"`
		} `json:"vectors"`
		BadVectors []struct {
			Name  string `json:"name"`
			Input string `json:"input"`
			Kind  string `json:"kind"`
		} `json:"bad_vectors"`
	}
	if err := json.Unmarshal(raw, &parsed); err != nil {
		t.Fatal(err)
	}
	return parsed.Vectors, parsed.BadVectors
}

// looksLikeZlib is the reference contract's structural sniff: a bad
// vector that is zlib-framed must be routed through InflateZlib, the
// rest through InflateRaw.
func looksLikeZlib(b []byte) bool {
	return len(b) >= 2 &&
		b[0]&0x0f == 8 &&
		b[0]>>4 <= 7 &&
		(uint16(b[0])<<8|uint16(b[1]))%31 == 0
}

// TestReferenceVectorsHexExact replays every committed reference.json
// vector through the cdylib and compares byte-exact: the decompressed
// output against the recorded plain hex, and the FFI's own Adler-32 of
// that output against the recorded digest — the same vectors the Rust
// gen-reference verify gate and the Python/Node SDKs check.
func TestReferenceVectorsHexExact(t *testing.T) {
	vectors, _ := reference(t)
	for _, want := range vectors {
		t.Run(want.Name, func(t *testing.T) {
			compressed, err := hex.DecodeString(want.Compressed)
			if err != nil {
				t.Fatal(err)
			}
			op := InflateRaw
			if strings.HasSuffix(want.Name, "/zlib") {
				op = InflateZlib
			}
			out, err := op(compressed)
			if err != nil {
				t.Fatalf("%s: %v", want.Name, err)
			}
			if plain, err := hex.DecodeString(want.Plain); err != nil {
				t.Fatal(err)
			} else if !bytes.Equal(out, plain) {
				t.Errorf("%s: output %d bytes, want %d (hex mismatch)", want.Name, len(out), len(plain))
			}
			sum, err := Adler32(out)
			if err != nil {
				t.Fatal(err)
			}
			if got := formatAdler(sum); got != want.Adler32 {
				t.Errorf("%s: adler32 %s, want %s", want.Name, got, want.Adler32)
			}
		})
	}
}

// TestBadVectorsRefused replays every malformed reference vector: the
// routed operation must return an *FfiError with StatusRejected — a
// status, never a crash.
func TestBadVectorsRefused(t *testing.T) {
	_, bad := reference(t)
	for _, want := range bad {
		t.Run(want.Name, func(t *testing.T) {
			input, err := hex.DecodeString(want.Input)
			if err != nil {
				t.Fatal(err)
			}
			op := InflateRaw
			if strings.HasPrefix(want.Name, "zlib_") || looksLikeZlib(input) {
				op = InflateZlib
			}
			if _, err := op(input); err == nil {
				t.Fatalf("%s: unexpectedly decoded", want.Name)
			} else {
				var ffi *FfiError
				if !asFfiError(err, &ffi) || ffi.Status != StatusRejected {
					t.Fatalf("%s: want status %d, got %v", want.Name, StatusRejected, err)
				}
			}
		})
	}
}

// asFfiError is errors.As without importing errors twice in one file's
// readability budget.
func asFfiError(err error, target **FfiError) bool {
	for err != nil {
		if e, ok := err.(*FfiError); ok {
			*target = e
			return true
		}
		u, ok := err.(interface{ Unwrap() error })
		if !ok {
			return false
		}
		err = u.Unwrap()
	}
	return false
}

// formatAdler is the eight-digit lowercase hex the reference pins.
func formatAdler(sum uint32) string {
	const digits = "0123456789abcdef"
	out := make([]byte, 8)
	for i := 7; i >= 0; i-- {
		out[i] = digits[sum&0xf]
		sum >>= 4
	}
	return string(out)
}

// TestPinnedTextVector pins the `text` vector literally — compressed
// stream, plaintext and Adler-32 as committed to reference.json — so
// the binding fails loudly even if reference.json were regenerated
// wrongly. The same pin the Rust unit tests re-derive.
func TestPinnedTextVector(t *testing.T) {
	const compressedHex = "2bc94855282ccd4cce56482aca2fcf5348cbaf50c82acd2d2856c82f4b2d5228014ae72456552aa4e4a7eb8179a38a478c6200"
	compressed, err := hex.DecodeString(compressedHex)
	if err != nil {
		t.Fatal(err)
	}
	wantPlain := bytes.Repeat([]byte("the quick brown fox jumps over the lazy dog. "), 12)
	for name, op := range map[string]func([]byte) ([]byte, error){
		"raw":  InflateRaw,
		"auto": InflateAuto,
	} {
		out, err := op(compressed)
		if err != nil {
			t.Fatalf("text/%s: %v", name, err)
		}
		if !bytes.Equal(out, wantPlain) {
			t.Errorf("text/%s: output %d bytes, want %d", name, len(out), len(wantPlain))
		}
		sum, err := Adler32(out)
		if err != nil {
			t.Fatal(err)
		}
		if sum != 0xf724c355 {
			t.Errorf("text/%s: adler32 %08x, want f724c355", name, sum)
		}
	}
}

// TestPinnedEmptyZlibVector pins the `empty/zlib` vector literally:
// the empty stream decodes to a zero-length buffer that round-trips
// through the free path, and its Adler-32 is the initial value 1.
func TestPinnedEmptyZlibVector(t *testing.T) {
	compressed, err := hex.DecodeString("78da030000000001")
	if err != nil {
		t.Fatal(err)
	}
	for name, op := range map[string]func([]byte) ([]byte, error){
		"zlib": InflateZlib,
		"auto": InflateAuto,
	} {
		out, err := op(compressed)
		if err != nil {
			t.Fatalf("empty/zlib/%s: %v", name, err)
		}
		if len(out) != 0 {
			t.Errorf("empty/zlib/%s: output %d bytes, want 0", name, len(out))
		}
		sum, err := Adler32(out)
		if err != nil {
			t.Fatal(err)
		}
		if sum != 1 {
			t.Errorf("empty/zlib/%s: adler32 %d, want 1", name, sum)
		}
	}
}

// TestRefusals checks the decoder's refusal paths through every
// routing: a status code, never a crash. A null data pointer with a
// zero length is an empty input (refused by the core), not a caller
// bug.
func TestRefusals(t *testing.T) {
	truncated := []byte{0x78, 0xda}
	garbage := make([]byte, 16)
	for name, op := range map[string]func([]byte) ([]byte, error){
		"raw":  InflateRaw,
		"zlib": InflateZlib,
		"auto": InflateAuto,
	} {
		for _, input := range [][]byte{truncated, garbage, {}} {
			if _, err := op(input); err == nil {
				t.Fatalf("%s: input %v unexpectedly decoded", name, input)
			} else {
				var ffi *FfiError
				if !asFfiError(err, &ffi) || ffi.Status != StatusRejected {
					t.Fatalf("%s: want status %d, got %v", name, StatusRejected, err)
				}
			}
		}
	}
	if _, err := Adler32(nil); err != nil {
		t.Fatalf("adler32(empty): %v", err)
	}
}
