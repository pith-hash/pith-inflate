//! The gzip (RFC 1952) reader: committed fixtures through
//! `inflate_gzip` and through the streaming decoder, the malformed
//! corpus, multi-member streams, and the limits.

use pith_digest::Error;
use pith_inflate::{Framing, Limits, StreamingDecoder, inflate_auto, inflate_gzip};
use vectors::{GZIP_BAD_VECTORS, GZIP_VECTORS};

#[path = "vectors.rs"]
mod vectors;

/// Decodes the lowercase-hex encoding `vectors.rs` uses for every byte
/// field.
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

/// The variant name of an error, for comparison against a bad vector's
/// `kind`.
fn kind_of(err: &Error) -> &'static str {
    match err {
        Error::Truncated { .. } => "Truncated",
        Error::InvalidMagic { .. } => "InvalidMagic",
        Error::BadValue(_) => "BadValue",
        Error::Unsupported(_) => "Unsupported",
        Error::TooLarge { .. } => "TooLarge",
    }
}

/// The bytes of a committed fixture, by the name its vector records.
fn fixture(file: &str) -> &'static [u8] {
    match file {
        "hello.gz" => include_bytes!("fixtures/gzip/hello.gz").as_slice(),
        "empty.gz" => include_bytes!("fixtures/gzip/empty.gz").as_slice(),
        "two-members.gz" => include_bytes!("fixtures/gzip/two-members.gz").as_slice(),
        "binary.gz" => include_bytes!("fixtures/gzip/binary.gz").as_slice(),
        "fextra.gz" => include_bytes!("fixtures/gzip/fextra.gz").as_slice(),
        "fname.gz" => include_bytes!("fixtures/gzip/fname.gz").as_slice(),
        "fcomment.gz" => include_bytes!("fixtures/gzip/fcomment.gz").as_slice(),
        "fhcrc.gz" => include_bytes!("fixtures/gzip/fhcrc.gz").as_slice(),
        "all-flags.gz" => include_bytes!("fixtures/gzip/all-flags.gz").as_slice(),
        other => panic!("unknown fixture {other}"),
    }
}

/// Every gzip vector decodes byte-identically through `inflate_gzip`,
/// one-shot.
#[test]
fn gzip_vectors_decode_byte_for_byte() {
    for v in GZIP_VECTORS {
        let out = inflate_gzip(&unhex(v.input), &Limits::default())
            .unwrap_or_else(|e| panic!("vector {} failed: {e}", v.name));
        assert_eq!(out.hex(), v.plain, "vector {}", v.name);
    }
}

/// The committed fixture bytes and the corpus hex are the same bytes:
/// the `.gz` files are canonical, the corpus is their inline copy.
#[test]
fn fixtures_and_corpus_are_the_same_bytes() {
    for v in GZIP_VECTORS {
        assert_eq!(fixture(v.file), unhex(v.input).as_slice(), "{}", v.name);
    }
}

/// Every fixture also decodes through the streaming decoder, fed in
/// three uneven chunks, byte-identically.
#[test]
fn fixtures_stream_in_three_chunks() {
    for v in GZIP_VECTORS {
        let input = unhex(v.input);
        let plain = unhex(v.plain);
        let (a, b) = (input.len() / 3, (input.len() * 2) / 3);
        let mut d = StreamingDecoder::new(Framing::Gzip, Limits::default());
        d.feed(&input[..a]).expect(v.name);
        assert!(
            d.output().len() <= plain.len(),
            "{} produced too much",
            v.name
        );
        d.feed(&input[a..b]).expect(v.name);
        assert!(
            d.output().len() <= plain.len(),
            "{} produced too much",
            v.name
        );
        d.feed(&input[b..]).expect(v.name);
        let out = d
            .finish()
            .unwrap_or_else(|e| panic!("{} failed: {e}", v.name));
        assert_eq!(out, plain, "vector {}", v.name);
    }
}

/// Every malformed gzip container fails with exactly the named error
/// variant - never a crash, never a wrong reason.
#[test]
fn gzip_bad_vectors_produce_the_named_variant() {
    for v in GZIP_BAD_VECTORS {
        let result = inflate_gzip(&unhex(v.input), &Limits::default());
        match result {
            Err(e) => assert_eq!(kind_of(&e), v.kind, "bad vector {}", v.name),
            Ok(out) => panic!("bad vector {} decoded to {:?}", v.name, out.hex()),
        }
    }
}

/// A two-member stream built from two different fixtures: the output is
/// the concatenation of the member outputs, each member validated
/// alone.
#[test]
fn concatenated_members_concatenate_output() {
    let hello = include_bytes!("fixtures/gzip/hello.gz").as_slice();
    let fname = include_bytes!("fixtures/gzip/fname.gz").as_slice();
    let both = [hello, fname].concat();
    let out = inflate_gzip(&both, &Limits::default()).expect("two members");
    let hello_out = inflate_gzip(hello, &Limits::default()).expect("hello");
    let fname_out = inflate_gzip(fname, &Limits::default()).expect("fname");
    assert_eq!(out, [hello_out, fname_out].concat());
}

/// The FHCRC of the `fhcrc` fixture really covers the header bytes:
/// flipping any header byte after the magic check - the FLG bits and
/// the MTIME/OS tail - must be refused as [`Error::BadValue`].
#[test]
fn fhcrc_covers_the_header_bytes() {
    let original = include_bytes!("fixtures/gzip/fhcrc.gz");
    for position in [3usize, 4, 9] {
        let mut flipped = original.to_vec();
        flipped[position] ^= 0x01;
        match inflate_gzip(&flipped, &Limits::default()) {
            Err(Error::BadValue(_)) => {}
            other => panic!("flipped header byte {position} gave {other:?}"),
        }
    }
}

/// An output ceiling below the member's content is [`Error::TooLarge`],
/// through the one-shot and the streaming decoder alike.
#[test]
fn output_ceiling_is_enforced() {
    let hello = include_bytes!("fixtures/gzip/hello.gz");
    let limits = Limits {
        max_output: 8,
        ..Limits::default()
    };
    match inflate_gzip(hello, &limits) {
        Err(Error::TooLarge { .. }) => {}
        other => panic!("expected TooLarge, got {other:?}"),
    }
    let mut d = StreamingDecoder::new(Framing::Gzip, limits);
    match d.feed(hello) {
        Err(Error::TooLarge { .. }) => {}
        other => panic!("expected TooLarge, got {other:?}"),
    }
}

/// Input above the ceiling is refused before a bit is read, and the
/// streaming decoder enforces the ceiling cumulatively across feeds.
#[test]
fn input_ceiling_is_enforced() {
    let hello = include_bytes!("fixtures/gzip/hello.gz");
    let limits = Limits {
        max_input: 16,
        ..Limits::default()
    };
    match inflate_gzip(hello, &limits) {
        Err(Error::TooLarge { .. }) => {}
        other => panic!("expected TooLarge, got {other:?}"),
    }
    let mut d = StreamingDecoder::new(Framing::Gzip, limits);
    d.feed(&hello[..8]).expect("first half fits");
    match d.feed(&hello[8..]) {
        Err(Error::TooLarge { .. }) => {}
        other => panic!("expected TooLarge, got {other:?}"),
    }
}

/// `inflate_auto` still refuses gzip rather than mis-decoding it: the
/// explicit `inflate_gzip` entry point is the gzip reader.
#[test]
fn auto_sniff_still_reports_gzip_unsupported() {
    for v in GZIP_VECTORS {
        match inflate_auto(&unhex(v.input), &Limits::default()) {
            Err(Error::Unsupported("gzip")) => {}
            other => panic!("{} through auto gave {other:?}", v.name),
        }
    }
}

/// Round-tripping the same member through two framing paths agrees:
/// streaming raw/zlib containers over the fixture payloads matches the
/// one-shot zlib path (the equivalence property the streaming module
/// pins exhaustively, spot-checked here at the container level).
#[test]
fn streaming_gzip_matches_one_shot_at_every_offset() {
    let hello = include_bytes!("fixtures/gzip/hello.gz");
    let expected = inflate_gzip(hello, &Limits::default()).expect("one-shot");
    for split in 0..=hello.len() {
        let mut d = StreamingDecoder::new(Framing::Gzip, Limits::default());
        d.feed(&hello[..split]).expect("first half");
        d.feed(&hello[split..]).expect("second half");
        assert_eq!(d.finish().expect("finish"), expected, "split at {split}");
    }
}

/// Helper so assertions can compare hex strings without importing the
/// corpus generator's helpers.
trait Hex {
    fn hex(&self) -> String;
}

impl Hex for [u8] {
    fn hex(&self) -> String {
        // `write!` into one string: the older clippy pin in CI rejects
        // the `map(format!).collect()` shape this replaces.
        let mut out = String::with_capacity(self.len() * 2);
        for byte in self {
            use core::fmt::Write as _;
            let _ = write!(out, "{byte:02x}");
        }
        out
    }
}
