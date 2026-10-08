//! Regenerates and verifies `reference.json`, the hex-exact vector
//! reference this suite ships beside every SDK artifact.
//!
//! The corpus itself lives in `tests/vectors.rs` (regenerate it with
//! `python gen_vectors.py`); this binary turns that corpus into a
//! language-neutral JSON document by *decoding every entry through
//! `pith-inflate`* and pinning each decoded output with its Adler-32
//! digest. The result is a fixed point: `reference.json` is exactly what
//! the current decoder produces for the current corpus, so any regression
//! in either shows up as a `verify` failure instead of a silently stale
//! file.
//!
//! - `gen-reference gen` writes `reference.json` at the repository root.
//! - `gen-reference verify` recomputes it and fails on any drift; this is
//!   the mode the CI gate runs.
//!
//! The binary uses `std` (it touches the filesystem) but adds no
//! dependencies: the JSON emission is hand-rolled, and the digests are
//! the suite's own checksums through `pith-digest` - Adler-32 pins the
//! raw/zlib corpus, CRC-32 pins the gzip corpus, the same checksums the
//! framings themselves validate with.

#[path = "../../tests/vectors.rs"]
mod vectors;

use std::fs;
use std::path::Path;

use pith_digest::{Error, crc32};
use pith_inflate::{Limits, inflate_gzip, inflate_raw, inflate_zlib};
use vectors::{BAD_VECTORS, GZIP_BAD_VECTORS, GZIP_VECTORS, VECTORS};

/// Where the reference file lives: the repository root, next to the crate
/// manifest, so the CD workflow can ship it with the SDK artifacts
/// regardless of the directory `cargo run` was invoked from.
fn reference_path() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("reference.json")
}

/// The error variant's name, matching the harness in `tests/inflate.rs`.
fn kind_of(err: &Error) -> &'static str {
    match err {
        Error::Truncated { .. } => "Truncated",
        Error::InvalidMagic { .. } => "InvalidMagic",
        Error::BadValue(_) => "BadValue",
        Error::Unsupported(_) => "Unsupported",
        Error::TooLarge { .. } => "TooLarge",
    }
}

/// The same structural sniff `inflate_auto` documents and the harness
/// repeats: a bad vector that is zlib-framed must be routed through
/// `inflate_zlib`, the rest through `inflate_raw`.
fn looks_like_zlib(bytes: &[u8]) -> bool {
    bytes.len() >= 2
        && bytes[0] & 0x0f == 8
        && bytes[0] >> 4 <= 7
        && ((u16::from(bytes[0]) << 8) | u16::from(bytes[1])) % 31 == 0
}

/// Decodes the lowercase-hex encoding the corpus uses for every byte field.
fn unhex(s: &str) -> Vec<u8> {
    assert!(s.len() % 2 == 0, "odd hex length in {s:?}");
    s.as_bytes()
        .chunks(2)
        .map(|pair| {
            let hi = (pair[0] as char).to_digit(16).expect("hex digit");
            let lo = (pair[1] as char).to_digit(16).expect("hex digit");
            ((hi << 4) | lo) as u8
        })
        .collect()
}

/// Lowercase hex of a byte slice.
fn hex(data: &[u8]) -> String {
    let mut out = String::with_capacity(data.len() * 2);
    for byte in data {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

/// Escapes one string for a JSON string literal. The corpus strings are
/// plain ASCII names and prose, but the escaper is complete so the output
/// can never be corrupted by a future edit to either.
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// The eight-digit lowercase hex of the Adler-32 digest of a decoded
/// output: the fixed point each corpus entry is pinned by.
fn digest_hex(data: &[u8]) -> String {
    format!("{:08x}", pith_inflate::adler32(data))
}

/// Decodes one valid corpus entry and returns its output bytes. A vector
/// the decoder refuses is a broken build, not a reference to write.
fn decode_valid(name: &str, compressed_hex: &str) -> Vec<u8> {
    let compressed = unhex(compressed_hex);
    let limits = Limits::default();
    let result = if name.ends_with("/zlib") {
        inflate_zlib(&compressed, &limits)
    } else {
        inflate_raw(&compressed, &limits)
    };
    result.unwrap_or_else(|e| panic!("vector {name} failed to decode: {e}"))
}

/// Decodes one malformed corpus entry and returns the observed error
/// variant name. A stream that decodes, or fails for the wrong reason, is
/// a broken build: the reference must never paper over a decoder change.
fn decode_bad(name: &str, input_hex: &str, zlib_routed: bool) -> &'static str {
    let input = unhex(input_hex);
    let result = if zlib_routed {
        inflate_zlib(&input, &Limits::default())
    } else {
        inflate_raw(&input, &Limits::default())
    };
    match result {
        Err(e) => kind_of(&e),
        Ok(_) => panic!("bad vector {name} unexpectedly decoded"),
    }
}

/// Decodes one gzip corpus entry and returns its output bytes. A member
/// the decoder refuses is a broken build, not a reference to write.
fn decode_gzip_valid(name: &str, input_hex: &str) -> Vec<u8> {
    let input = unhex(input_hex);
    inflate_gzip(&input, &Limits::default())
        .unwrap_or_else(|e| panic!("gzip vector {name} failed to decode: {e}"))
}

/// Decodes one malformed gzip corpus entry and returns the observed
/// error variant name, with the same wrong-reason panic contract as
/// [`decode_bad`].
fn decode_gzip_bad(name: &str, input_hex: &str) -> &'static str {
    let input = unhex(input_hex);
    match inflate_gzip(&input, &Limits::default()) {
        Err(e) => kind_of(&e),
        Ok(_) => panic!("gzip bad vector {name} unexpectedly decoded"),
    }
}

/// The exact bytes of `reference.json` for the current corpus and decoder:
/// two-space indent, corpus order, one trailing newline. Nothing here is
/// sorted or deduplicated on purpose - the file is a transcript of the
/// corpus, in corpus order, so a diff against a previous commit reads as a
/// changelog of the corpus.
fn reference_json() -> String {
    let mut out = String::with_capacity(96 * 1024);
    out.push_str("{\n");
    out.push_str("  \"schema\": 2,\n");
    out.push_str("  \"crate\": \"pith-inflate\",\n");
    out.push_str(
        "  \"description\": \"hex-exact DEFLATE/zlib/gzip reference vectors: every \
tests/vectors.rs corpus entry decoded by pith-inflate, raw/zlib outputs pinned by \
their Adler-32 digest and gzip outputs by their CRC-32\",\n",
    );
    out.push_str("  \"vectors\": [\n");
    for v in VECTORS {
        let plain = decode_valid(v.name, v.compressed);
        assert_eq!(
            hex(&plain),
            v.plain,
            "vector {} ({}) decoded to bytes that differ from its recorded plaintext",
            v.name,
            v.why
        );
        out.push_str("    {\n");
        out.push_str(&format!("      \"name\": \"{}\",\n", json_escape(v.name)));
        out.push_str(&format!("      \"block\": \"{}\",\n", json_escape(v.block)));
        out.push_str(&format!(
            "      \"compressed\": \"{}\",\n",
            json_escape(v.compressed)
        ));
        out.push_str(&format!("      \"plain\": \"{}\",\n", json_escape(v.plain)));
        out.push_str(&format!("      \"adler32\": \"{}\"\n", digest_hex(&plain)));
        out.push_str("    },\n");
    }
    // Drop the final comma of the last entry to close the array. The
    // corpus is a non-empty const, so the comma is always there.
    let comma = out.rfind("},\n").expect("at least one vector entry") + 1;
    out.replace_range(comma..comma + 1, "");
    out.push_str("  ],\n");
    out.push_str("  \"bad_vectors\": [\n");
    for v in BAD_VECTORS {
        let zlib_routed = v.name.starts_with("zlib_") || looks_like_zlib(&unhex(v.input));
        let observed = decode_bad(v.name, v.input, zlib_routed);
        assert_eq!(
            observed, v.kind,
            "bad vector {} ({}) failed for the wrong reason",
            v.name, v.why
        );
        out.push_str("    {\n");
        out.push_str(&format!("      \"name\": \"{}\",\n", json_escape(v.name)));
        out.push_str(&format!("      \"input\": \"{}\",\n", json_escape(v.input)));
        out.push_str(&format!("      \"kind\": \"{}\"\n", json_escape(observed)));
        out.push_str("    },\n");
    }
    let comma = out.rfind("},\n").expect("at least one bad vector entry") + 1;
    out.replace_range(comma..comma + 1, "");
    out.push_str("  ],\n");
    out.push_str("  \"gzip_vectors\": [\n");
    for v in GZIP_VECTORS {
        let plain = decode_gzip_valid(v.name, v.input);
        assert_eq!(
            hex(&plain),
            v.plain,
            "gzip vector {} ({}) decoded to bytes that differ from its recorded plaintext",
            v.name,
            v.why
        );
        out.push_str("    {\n");
        out.push_str(&format!("      \"name\": \"{}\",\n", json_escape(v.name)));
        out.push_str(&format!("      \"file\": \"{}\",\n", json_escape(v.file)));
        out.push_str(&format!("      \"input\": \"{}\",\n", json_escape(v.input)));
        out.push_str(&format!("      \"plain\": \"{}\",\n", json_escape(v.plain)));
        out.push_str(&format!("      \"crc32\": \"{:08x}\"\n", crc32(&plain)));
        out.push_str("    },\n");
    }
    let comma = out.rfind("},\n").expect("at least one gzip vector entry") + 1;
    out.replace_range(comma..comma + 1, "");
    out.push_str("  ],\n");
    out.push_str("  \"gzip_bad_vectors\": [\n");
    for v in GZIP_BAD_VECTORS {
        let observed = decode_gzip_bad(v.name, v.input);
        assert_eq!(
            observed, v.kind,
            "gzip bad vector {} ({}) failed for the wrong reason",
            v.name, v.why
        );
        out.push_str("    {\n");
        out.push_str(&format!("      \"name\": \"{}\",\n", json_escape(v.name)));
        out.push_str(&format!("      \"input\": \"{}\",\n", json_escape(v.input)));
        out.push_str(&format!("      \"kind\": \"{}\"\n", json_escape(observed)));
        out.push_str("    },\n");
    }
    let comma = out
        .rfind("},\n")
        .expect("at least one gzip bad vector entry")
        + 1;
    out.replace_range(comma..comma + 1, "");
    out.push_str("  ]\n");
    out.push_str("}\n");
    out
}

/// Writes `reference.json`. Returns the number of entries pinned.
fn generate_at(path: &Path) -> std::io::Result<usize> {
    let json = reference_json();
    fs::write(path, json)?;
    Ok(entry_count())
}

/// Recomputes the reference and compares it byte-for-byte with the
/// committed copy. `Ok(entries)` means the committed file is current.
fn verify_at(path: &Path) -> Result<usize, String> {
    let committed =
        fs::read_to_string(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let recomputed = reference_json();
    if committed != recomputed {
        let first = committed
            .bytes()
            .zip(recomputed.bytes())
            .position(|(a, b)| a != b)
            .unwrap_or(committed.len().min(recomputed.len()));
        return Err(format!(
            "recomputed reference differs from {} (first difference at byte {})",
            path.display(),
            first
        ));
    }
    Ok(entry_count())
}

/// Every corpus entry the reference pins: valid and malformed, raw/zlib
/// and gzip.
fn entry_count() -> usize {
    VECTORS.len() + BAD_VECTORS.len() + GZIP_VECTORS.len() + GZIP_BAD_VECTORS.len()
}

/// The CLI body: the mode argument against the reference path. Returns the
/// process exit code so tests can exercise every branch in-process; `main`
/// is the only thing that actually exits.
fn run(path: &Path, mode: Option<&str>) -> i32 {
    match mode {
        Some("gen") => match generate_at(path) {
            Ok(entries) => {
                println!(
                    "wrote {} ({} vectors, {} bad vectors, {} gzip vectors, {} gzip bad vectors)",
                    path.display(),
                    VECTORS.len(),
                    BAD_VECTORS.len(),
                    GZIP_VECTORS.len(),
                    GZIP_BAD_VECTORS.len()
                );
                debug_assert_eq!(entries, entry_count());
                0
            }
            Err(e) => {
                eprintln!("gen-reference: {e}");
                1
            }
        },
        Some("verify") => match verify_at(path) {
            Ok(n) => {
                println!("reference.json is current ({n} entries verified)");
                0
            }
            Err(e) => {
                eprintln!("reference.json is stale: {e}");
                1
            }
        },
        other => {
            eprintln!("usage: gen-reference <gen|verify> (got {other:?})");
            2
        }
    }
}

fn main() {
    std::process::exit(run(&reference_path(), std::env::args().nth(1).as_deref()));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A scratch path unique to this test process, cleaned up by the caller.
    fn scratch(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("gen-reference-{}-{name}", std::process::id()))
    }

    #[test]
    fn json_escape_escapes_specials() {
        assert_eq!(json_escape("plain"), "plain");
        assert_eq!(json_escape("a\"b\\c\nd\te\rf"), "a\\\"b\\\\c\\nd\\te\\rf");
        assert_eq!(json_escape("\u{1}"), "\\u0001");
    }

    #[test]
    fn reference_pins_every_corpus_entry() {
        let json = reference_json();
        assert_eq!(json.matches("\"name\":").count(), entry_count());
        assert_eq!(json.matches("\"crc32\":").count(), GZIP_VECTORS.len());
        // The empty vector decodes to nothing, whose Adler-32 is 1.
        assert!(json.contains("\"compressed\": \"0300\""));
        assert!(json.contains("\"adler32\": \"00000001\""));
        // Every declared error kind is carried through verbatim.
        for v in BAD_VECTORS {
            assert!(json.contains(&format!("\"kind\": \"{}\"", v.kind)));
        }
        for v in GZIP_BAD_VECTORS {
            assert!(json.contains(&format!("\"kind\": \"{}\"", v.kind)));
        }
    }

    #[test]
    fn generate_then_verify_round_trips() {
        let path = scratch("roundtrip.json");
        let entries = generate_at(&path).expect("write reference");
        assert_eq!(entries, entry_count());
        assert_eq!(verify_at(&path), Ok(entries));
        let written = fs::read_to_string(&path).expect("read back");
        assert_eq!(written, reference_json());
        assert!(written.ends_with("}\n"));
        fs::remove_file(&path).ok();
    }

    #[test]
    fn verify_rejects_a_tampered_reference() {
        let path = scratch("tamper.json");
        let mut json = reference_json();
        // Corrupt one JSON key: any byte drift must fail the verify.
        let pos = json.find("\"adler32\"").expect("digest field present");
        json.replace_range(pos..pos + 10, "\"adler33\"");
        fs::write(&path, json).expect("write tampered reference");
        assert!(verify_at(&path).is_err());
        fs::remove_file(&path).ok();
    }

    #[test]
    fn verify_reports_a_missing_file() {
        let path = scratch("missing-does-not-exist.json");
        fs::remove_file(&path).ok();
        let err = verify_at(&path).expect_err("missing file must fail");
        assert!(err.contains("cannot read"));
    }

    #[test]
    fn run_dispatches_gen_verify_and_usage() {
        let path = scratch("run.json");
        fs::remove_file(&path).ok();
        assert_eq!(run(&path, Some("gen")), 0);
        assert!(path.exists());
        assert_eq!(run(&path, Some("verify")), 0);
        assert_eq!(run(&path, Some("polish")), 2);
        assert_eq!(run(&path, None), 2);
        fs::remove_file(&path).ok();
    }

    #[test]
    fn run_verify_fails_on_a_stale_reference() {
        let path = scratch("run-stale.json");
        fs::write(&path, "{\n  \"schema\": 0\n}\n").expect("write stale reference");
        assert_eq!(run(&path, Some("verify")), 1);
        fs::remove_file(&path).ok();
    }

    #[test]
    fn every_error_variant_has_a_named_kind() {
        assert_eq!(kind_of(&Error::truncated("x", 2, 1)), "Truncated");
        assert_eq!(kind_of(&Error::InvalidMagic { what: "x" }), "InvalidMagic");
        assert_eq!(kind_of(&Error::BadValue("x")), "BadValue");
        assert_eq!(kind_of(&Error::Unsupported("x")), "Unsupported");
        assert_eq!(kind_of(&Error::too_large("x", 1)), "TooLarge");
    }

    #[test]
    #[should_panic(expected = "odd hex length")]
    fn unhex_rejects_an_odd_hex_length() {
        let _ = unhex("abc");
    }

    #[test]
    #[should_panic(expected = "unexpectedly decoded")]
    fn a_bad_vector_that_decodes_is_a_broken_build() {
        // The empty raw stream decodes cleanly, so it must never be listed
        // as a bad vector.
        let _ = decode_bad("empty_stream", "0300", false);
    }

    #[test]
    fn reference_path_lands_beside_the_manifest() {
        assert!(reference_path().is_absolute());
        assert_eq!(
            reference_path().file_name(),
            Some("reference.json".as_ref())
        );
    }
}
