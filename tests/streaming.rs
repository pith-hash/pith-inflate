//! The chunked-feed equivalence property: a stream decoded through
//! `StreamingDecoder` in any chunking is byte-identical to the one-shot
//! decode of the same bytes, for every framing the crate ships.
//!
//! The corpus in `vectors.rs` is the input: every valid vector through
//! its own framing at every chunk size up to its full length (for the
//! streams short enough to sweep), every malformed vector ending in the
//! same error variant the one-shot names, and the container wrappers'
//! edge rules (trailing bytes, empty feeds, finish-before-feed) pinned
//! explicitly.

use pith_digest::Error;
use pith_inflate::{Framing, Limits, StreamingDecoder, inflate_gzip, inflate_raw, inflate_zlib};
use vectors::{BAD_VECTORS, GZIP_BAD_VECTORS, GZIP_VECTORS, VECTORS};

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

/// The variant name of an error.
fn kind_of(err: &Error) -> &'static str {
    match err {
        Error::Truncated { .. } => "Truncated",
        Error::InvalidMagic { .. } => "InvalidMagic",
        Error::BadValue(_) => "BadValue",
        Error::Unsupported(_) => "Unsupported",
        Error::TooLarge { .. } => "TooLarge",
    }
}

/// The reference contract's structural sniff: a bad vector that is
/// zlib-framed must route through the zlib framing, the rest through
/// raw DEFLATE.
fn looks_like_zlib(bytes: &[u8]) -> bool {
    bytes.len() >= 2
        && bytes[0] & 0x0f == 8
        && bytes[0] >> 4 <= 7
        && ((u16::from(bytes[0]) << 8) | u16::from(bytes[1])) % 31 == 0
}

/// The one-shot decode a vector is pinned by, through the framing its
/// name encodes (`/zlib` suffix means the zlib framing, else raw).
fn one_shot(framing: Framing, input: &[u8]) -> Result<Vec<u8>, Error> {
    match framing {
        Framing::Raw => inflate_raw(input, &Limits::default()),
        Framing::Zlib => inflate_zlib(input, &Limits::default()),
        Framing::Gzip => inflate_gzip(input, &Limits::default()),
        Framing::Auto => unreachable!("the corpus is routed by explicit framing"),
    }
}

/// Feeds `input` through a fresh decoder in chunks of exactly `size`
/// bytes, asserting after every feed that the incremental output is a
/// prefix of the final output - the incremental decoder never guesses.
fn feed_in_chunks(
    framing: Framing,
    input: &[u8],
    size: usize,
    expected: &[u8],
) -> Result<Vec<u8>, Error> {
    let mut d = StreamingDecoder::new(framing, Limits::default());
    for chunk in input.chunks(size.max(1)) {
        d.feed(chunk)?;
        let produced = d.output();
        assert!(
            expected.starts_with(produced),
            "incremental output {:02x?} is not a prefix of the final output",
            produced
        );
    }
    assert_eq!(d.output(), expected, "post-feed output must be complete");
    assert!(d.is_done(), "decoder must be done after the full feed");
    d.finish()
}

/// Chunk plans: an exhaustive sweep for streams short enough that
/// sweeping is cheap, a spread that keeps the worst cases (1, 2, 3 and
/// one-byte-past-structural-boundaries) for the large ones.
fn chunk_plans(len: usize) -> Vec<usize> {
    if len <= 512 {
        (1..=len).collect()
    } else {
        let mut plan: Vec<usize> = vec![1, 2, 3, 5, 8, 13, 21, 64, 1000, 4096];
        plan.push(len / 2);
        plan.push(len - 1);
        plan.push(len);
        plan.sort_unstable();
        plan.dedup();
        plan
    }
}

/// Every valid corpus vector, through its own framing, at every chunk
/// size, decodes byte-identically to the one-shot decode.
#[test]
fn chunked_feed_equals_one_shot_for_every_vector() {
    for v in VECTORS {
        let input = unhex(v.compressed);
        let expected = unhex(v.plain);
        let framing = if v.name.ends_with("/zlib") {
            Framing::Zlib
        } else {
            Framing::Raw
        };
        for size in chunk_plans(input.len()) {
            let out = feed_in_chunks(framing, &input, size, &expected)
                .unwrap_or_else(|e| panic!("vector {} at chunk {size}: {e}", v.name));
            assert_eq!(out, expected, "vector {} at chunk {size}", v.name);
        }
    }
}

/// Every valid gzip container, through the gzip framing, at every chunk
/// size, decodes byte-identically to `inflate_gzip`.
#[test]
fn chunked_feed_equals_one_shot_for_every_gzip_vector() {
    for v in GZIP_VECTORS {
        let input = unhex(v.input);
        let expected = unhex(v.plain);
        for size in chunk_plans(input.len()) {
            let out = feed_in_chunks(Framing::Gzip, &input, size, &expected)
                .unwrap_or_else(|e| panic!("vector {} at chunk {size}: {e}", v.name));
            assert_eq!(out, expected, "vector {} at chunk {size}", v.name);
        }
    }
}

/// Every malformed stream, routed through the streaming decoder with
/// the framing its one-shot routing uses, fails with the same error
/// variant - at every chunk size of the sweep for short streams.
#[test]
fn malformed_streams_end_in_the_same_variant_when_streamed() {
    for v in BAD_VECTORS {
        let input = unhex(v.input);
        let framing = if v.name.starts_with("zlib_") || looks_like_zlib(&input) {
            Framing::Zlib
        } else {
            Framing::Raw
        };
        let one_shot_kind = kind_of(&one_shot(framing, &input).expect_err(v.name));
        assert_eq!(one_shot_kind, v.kind, "bad vector {}", v.name);
        for size in chunk_plans(input.len().max(1)) {
            let mut d = StreamingDecoder::new(framing, Limits::default());
            let mut failure = None;
            for chunk in input.chunks(size.max(1)) {
                if let Err(e) = d.feed(chunk) {
                    failure = Some(e);
                    break;
                }
            }
            let kind = match failure {
                Some(e) => kind_of(&e),
                None => kind_of(&d.finish().expect_err(v.name)),
            };
            assert_eq!(kind, v.kind, "bad vector {} at chunk {size}", v.name);
        }
    }
}

/// Every malformed gzip container ends in the named variant through the
/// streaming decoder too - fed all at once or byte by byte.
#[test]
fn malformed_gzip_ends_in_the_same_variant_when_streamed() {
    for v in GZIP_BAD_VECTORS {
        let input = unhex(v.input);
        for size in [input.len().max(1), 1] {
            let mut d = StreamingDecoder::new(Framing::Gzip, Limits::default());
            let mut failure = None;
            for chunk in input.chunks(size) {
                if let Err(e) = d.feed(chunk) {
                    failure = Some(e);
                    break;
                }
            }
            let kind = match failure {
                Some(e) => kind_of(&e),
                None => kind_of(&d.finish().expect_err(v.name)),
            };
            assert_eq!(kind, v.kind, "bad vector {} at chunk {size}", v.name);
        }
    }
}

/// Finishing before any input is a truncation, in every framing.
#[test]
fn finish_before_any_feed_is_truncated() {
    for framing in [Framing::Raw, Framing::Zlib, Framing::Gzip, Framing::Auto] {
        let d = StreamingDecoder::new(framing, Limits::default());
        match d.finish() {
            Err(Error::Truncated { .. }) => {}
            other => panic!("{framing:?}: expected Truncated, got {other:?}"),
        }
    }
}

/// An empty feed is a no-op, not a decode step.
#[test]
fn empty_feed_changes_nothing() {
    let mut d = StreamingDecoder::new(Framing::Zlib, Limits::default());
    assert_eq!(d.feed(b"").expect("empty feed"), 0);
    assert!(d.output().is_empty());
    match d.finish() {
        Err(Error::Truncated { .. }) => {}
        other => panic!("expected Truncated, got {other:?}"),
    }
}

/// Raw ignores trailing bytes after the final block, one-shot and
/// streaming alike; zlib refuses them, one-shot and streaming alike.
#[test]
fn trailing_bytes_follow_the_framings_contract() {
    // Raw: "A" plus junk decodes to "A".
    let mut raw = vec![0x73, 0x04, 0x00];
    raw.extend_from_slice(b"trailing junk");
    assert_eq!(inflate_raw(&raw, &Limits::default()).expect("raw"), b"A");
    let mut d = StreamingDecoder::new(Framing::Raw, Limits::default());
    d.feed(&[0x73, 0x04, 0x00]).expect("stream");
    d.feed(b"trailing junk")
        .expect("raw ignores trailing bytes");
    assert_eq!(d.finish().expect("raw finish"), b"A");

    // zlib: the same junk after a complete zlib stream is refused.
    let mut zlib = unhex(
        VECTORS
            .iter()
            .find(|v| v.name == "one_byte/zlib")
            .expect("v")
            .compressed,
    );
    zlib.extend_from_slice(b"x");
    match inflate_zlib(&zlib, &Limits::default()) {
        Err(Error::BadValue(_)) => {}
        other => panic!("expected BadValue, got {other:?}"),
    }
    let mut d = StreamingDecoder::new(Framing::Zlib, Limits::default());
    let exact = &unhex(
        VECTORS
            .iter()
            .find(|v| v.name == "one_byte/zlib")
            .expect("v")
            .compressed,
    );
    d.feed(exact).expect("clean stream");
    match d.feed(b"x") {
        Err(Error::BadValue(_)) => {}
        other => panic!("expected BadValue, got {other:?}"),
    }
}

/// Auto sniffs streamed input exactly like the one-shot sniff: the zlib
/// vector decodes, gzip magic is refused with the same Unsupported.
#[test]
fn auto_streaming_matches_the_one_shot_sniff() {
    let zlib = unhex(
        VECTORS
            .iter()
            .find(|v| v.name == "one_byte/zlib")
            .expect("v")
            .compressed,
    );
    let expected = inflate_zlib(&zlib, &Limits::default()).expect("one-shot");
    let mut d = StreamingDecoder::new(Framing::Auto, Limits::default());
    d.feed(&zlib).expect("auto feed");
    assert_eq!(d.finish().expect("auto finish"), expected);

    let gzip = unhex(
        GZIP_VECTORS
            .iter()
            .find(|v| v.name == "gzip/hello")
            .expect("v")
            .input,
    );
    let mut d = StreamingDecoder::new(Framing::Auto, Limits::default());
    match d.feed(&gzip) {
        Err(Error::Unsupported("gzip")) => {}
        other => panic!("auto streaming must refuse gzip, got {other:?}"),
    }
}

/// A one-byte-at-a-time gzip decode of a multi-member fixture matches
/// the one-shot decode - the extreme of the chunk sweep, spelled out so
/// the multi-member path is visibly covered at chunk size 1.
#[test]
fn byte_by_byte_multi_member_matches_one_shot() {
    let two = include_bytes!("fixtures/gzip/two-members.gz");
    let expected = inflate_gzip(two, &Limits::default()).expect("one-shot");
    let mut d = StreamingDecoder::new(Framing::Gzip, Limits::default());
    for byte in two.iter() {
        d.feed(core::slice::from_ref(byte)).expect("feed one byte");
    }
    assert_eq!(d.finish().expect("finish"), expected);
}

/// Output ceilings and input ceilings bind the streaming decoder just
/// as they bind the one-shot paths.
#[test]
fn limits_bind_the_streaming_decoder() {
    // Output: "the quick brown fox..." repeats 12 times in the text
    // vector; a 16-byte ceiling must die long before the end.
    let text = VECTORS.iter().find(|v| v.name == "text").expect("v");
    let input = unhex(text.compressed);
    let limits = Limits {
        max_output: 16,
        ..Limits::default()
    };
    let mut d = StreamingDecoder::new(Framing::Raw, limits);
    match d.feed(&input) {
        Err(Error::TooLarge { .. }) => {}
        other => panic!("expected TooLarge, got {other:?}"),
    }
    // A poisoned decoder stays poisoned.
    match d.feed(&input) {
        Err(Error::TooLarge { .. }) => {}
        other => panic!("poisoned decoder must stay TooLarge, got {other:?}"),
    }
    match d.finish() {
        Err(Error::TooLarge { .. }) => {}
        other => panic!("poisoned finish must stay TooLarge, got {other:?}"),
    }
}

/// Half a stream stalls without error and reports only the output
/// prefix; the other half completes it.
#[test]
fn partial_feeds_stall_without_error() {
    for v in GZIP_VECTORS {
        let input = unhex(v.input);
        let expected = unhex(v.plain);
        let mut d = StreamingDecoder::new(Framing::Gzip, Limits::default());
        d.feed(&input[..input.len() / 2]).expect(v.name);
        assert!(
            expected.starts_with(d.output()),
            "{}: partial output is not a prefix",
            v.name
        );
        assert!(!d.is_done(), "{}: half a stream is not done", v.name);
        d.feed(&input[input.len() / 2..]).expect(v.name);
        assert_eq!(d.finish().expect(v.name), expected, "{}", v.name);
    }
}

/// Auto that never sees two bytes stalls at `feed` without resolving;
/// once two bytes arrive it resolves raw and decodes, exactly like the
/// one-shot sniff's raw arm.
#[test]
fn auto_resolves_raw_after_a_one_byte_stall() {
    let raw = unhex(
        VECTORS
            .iter()
            .find(|v| v.name == "one_byte")
            .expect("v")
            .compressed,
    );
    let mut d = StreamingDecoder::new(Framing::Auto, Limits::default());
    d.feed(&raw[..1]).expect("one byte stalls the sniff");
    d.feed(&raw[1..]).expect("second byte resolves raw");
    let expected = inflate_raw(&raw, &Limits::default()).expect("one-shot");
    assert_eq!(d.finish().expect("finish"), expected);
}

/// Trailing junk that arrives in the same feed as the stream's final
/// bytes is refused for zlib, byte-for-byte like the one-shot decode of
/// the same input.
#[test]
fn zlib_refuses_junk_fed_with_the_stream() {
    let mut input = unhex(
        VECTORS
            .iter()
            .find(|v| v.name == "one_byte/zlib")
            .expect("v")
            .compressed,
    );
    input.push(b'x');
    match inflate_zlib(&input, &Limits::default()) {
        Err(Error::BadValue(_)) => {}
        other => panic!("one-shot expected BadValue, got {other:?}"),
    }
    let mut d = StreamingDecoder::new(Framing::Zlib, Limits::default());
    match d.feed(&input) {
        Err(Error::BadValue(_)) => {}
        other => panic!("streaming expected BadValue, got {other:?}"),
    }
}

/// A zlib stream whose CMF advertises a window larger than the decoder
/// supports is refused at feed time, naming the field exactly like the
/// one-shot.
#[test]
fn streaming_refuses_an_oversized_cinfo() {
    let input = [0x98u8, 0x00, 0x9d, 0x2d]; // CM=8, CINFO=9: invalid
    match inflate_zlib(&input, &Limits::default()) {
        Err(Error::InvalidMagic { .. }) => {}
        other => panic!("one-shot expected InvalidMagic, got {other:?}"),
    }
    let mut d = StreamingDecoder::new(Framing::Zlib, Limits::default());
    match d.feed(&input) {
        Err(Error::InvalidMagic { .. }) => {}
        other => panic!("streaming expected InvalidMagic, got {other:?}"),
    }
}

/// A stream of more than one block - a stored block that is not final,
/// then a final one - decodes across the block boundary at every chunk
/// size, matching the one-shot decode.
#[test]
fn non_final_blocks_stream_across_the_boundary() {
    let mut stream = vec![0x00]; // bfinal=0, btype=00 (stored)
    stream.extend_from_slice(&5u16.to_le_bytes());
    stream.extend_from_slice(&(!5u16).to_le_bytes());
    stream.extend_from_slice(b"12345");
    stream.push(0x01); // bfinal=1, btype=00 (stored)
    stream.extend_from_slice(&2u16.to_le_bytes());
    stream.extend_from_slice(&(!2u16).to_le_bytes());
    stream.extend_from_slice(b"AB");
    assert_eq!(
        inflate_raw(&stream, &Limits::default()).expect("one-shot"),
        b"12345AB"
    );
    for size in 1..=stream.len() {
        let mut d = StreamingDecoder::new(Framing::Raw, Limits::default());
        for chunk in stream.chunks(size) {
            d.feed(chunk).expect("feed");
        }
        assert_eq!(d.finish().expect("finish"), b"12345AB", "chunk size {size}");
    }
}

/// A finish that stops inside a structure names that structure and the
/// honest byte count it still needs: every byte phase the state machine
/// can wait in gets its message exercised here.
#[test]
fn finish_names_the_stalled_structure() {
    let zlib = unhex(
        VECTORS
            .iter()
            .find(|v| v.name == "one_byte/zlib")
            .expect("v")
            .compressed,
    );
    // zlib: the Adler-32 trailer, cut by two bytes.
    let mut d = StreamingDecoder::new(Framing::Zlib, Limits::default());
    d.feed(&zlib[..zlib.len() - 2]).expect("feed");
    match d.finish() {
        Err(Error::Truncated { what, needed, .. }) => {
            assert_eq!((what, needed), ("zlib Adler-32 trailer", 2))
        }
        other => panic!("expected zlib trailer Truncated, got {other:?}"),
    }

    let cut = |name: &str, keep: usize| -> Vec<u8> {
        let v = GZIP_VECTORS.iter().find(|v| v.name == name).expect("v");
        unhex(&v.input[..keep * 2])
    };
    // gzip: eleven bytes of the fextra fixture - the fixed header plus
    // one FEXTRA length byte - stops in the FEXTRA length read.
    let mut d = StreamingDecoder::new(Framing::Gzip, Limits::default());
    d.feed(&cut("gzip/fextra", 11)).expect("feed");
    match d.finish() {
        Err(Error::Truncated { what, needed, .. }) => {
            assert_eq!((what, needed), ("gzip FEXTRA length", 2))
        }
        other => panic!("expected FEXTRA Truncated, got {other:?}"),
    }

    // The fhcrc fixture ends its header at byte 10; without the two
    // CRC bytes that follow it, the decoder stops in the FHCRC read.
    let fhcrc = include_bytes!("fixtures/gzip/fhcrc.gz");
    let mut d = StreamingDecoder::new(Framing::Gzip, Limits::default());
    d.feed(&fhcrc[..10]).expect("feed");
    match d.finish() {
        Err(Error::Truncated { what, needed, .. }) => assert_eq!((what, needed), ("gzip FHCRC", 2)),
        other => panic!("expected FHCRC Truncated, got {other:?}"),
    }

    // hello.gz without its last six bytes: the CRC-32 is half read.
    let hello = include_bytes!("fixtures/gzip/hello.gz");
    let mut d = StreamingDecoder::new(Framing::Gzip, Limits::default());
    d.feed(&hello[..hello.len() - 6]).expect("feed");
    match d.finish() {
        Err(Error::Truncated { what, needed, .. }) => {
            assert_eq!((what, needed), ("gzip CRC-32 trailer", 2))
        }
        other => panic!("expected CRC Truncated, got {other:?}"),
    }

    // hello.gz without its last three bytes: the ISIZE is one byte in.
    let mut d = StreamingDecoder::new(Framing::Gzip, Limits::default());
    d.feed(&hello[..hello.len() - 3]).expect("feed");
    match d.finish() {
        Err(Error::Truncated { what, needed, .. }) => {
            assert_eq!((what, needed), ("gzip ISIZE trailer", 3))
        }
        other => panic!("expected ISIZE Truncated, got {other:?}"),
    }

    // The two-member fixture: member 2 starts at byte 63 (member 1 is
    // 63 bytes end to end); five bytes into its fixed header, a finish
    // names the member header, not the first.
    let two = include_bytes!("fixtures/gzip/two-members.gz");
    let mut d = StreamingDecoder::new(Framing::Gzip, Limits::default());
    d.feed(&two[..68]).expect("feed");
    match d.finish() {
        Err(Error::Truncated { what, needed, .. }) => {
            assert_eq!((what, needed), ("gzip member header", 10))
        }
        other => panic!("expected member header Truncated, got {other:?}"),
    }
}
