//! Hand-built streams for the decoder's refusal paths that the generated
//! corpus cannot reach: Python's `zlib` only ever emits well-formed
//! streams, so every error branch here is packed bit by bit with the same
//! `Bits` builder the main harness uses for its hand-built blocks.
//!
//! These tests exist to keep the suite's >= 95% line gate honest; the
//! behaviours they pin are the same checked errors the API documents.

use pith_digest::Error;
use pith_inflate::{Limits, inflate_auto, inflate_raw, inflate_zlib};

/// DEFLATE packs bits least-significant first; Huffman codes go in MSB
/// first. This builder is the mirror of the implementation's reader.
struct Bits {
    bytes: Vec<u8>,
    nbits: usize,
}

impl Bits {
    fn new() -> Self {
        Bits {
            bytes: Vec::new(),
            nbits: 0,
        }
    }

    /// Pushes one bit at the current position.
    fn push_bit(&mut self, bit: u32) {
        if self.nbits % 8 == 0 {
            self.bytes.push(0);
        }
        if bit != 0 {
            self.bytes[self.nbits / 8] |= 1 << (self.nbits % 8);
        }
        self.nbits += 1;
    }

    /// A value field, least-significant bit first.
    fn field(&mut self, value: u32, count: u32) {
        for i in 0..count {
            self.push_bit((value >> i) & 1);
        }
    }

    /// A Huffman code, most-significant bit of the code first.
    fn code(&mut self, value: u32, count: u32) {
        for i in (0..count).rev() {
            self.push_bit((value >> i) & 1);
        }
    }

    /// Pads to the next byte boundary, as the reader's `align` does.
    fn align(&mut self) {
        while self.nbits % 8 != 0 {
            self.push_bit(0);
        }
    }
}

/// Asserts the stream fails with the named variant and nothing else.
fn assert_kind(bits: &Bits, kind: &'static str) {
    match inflate_raw(&bits.bytes, &Limits::default()) {
        Err(e) => assert_eq!(variant(&e), kind, "stream failed for the wrong reason: {e}"),
        Ok(out) => panic!("stream unexpectedly decoded to {out:?}"),
    }
}

/// The variant name of an error, mirroring the harness's `kind_of`.
fn variant(err: &Error) -> &'static str {
    match err {
        Error::Truncated { .. } => "Truncated",
        Error::InvalidMagic { .. } => "InvalidMagic",
        Error::BadValue(_) => "BadValue",
        Error::Unsupported(_) => "Unsupported",
        Error::TooLarge { .. } => "TooLarge",
    }
}

#[test]
fn raw_input_ceiling_is_enforced() {
    let limits = Limits {
        max_input: 8,
        ..Limits::default()
    };
    match inflate_raw(&[0u8; 16], &limits) {
        Err(Error::TooLarge { .. }) => {}
        other => panic!("oversized raw input must be TooLarge, got {other:?}"),
    }
}

#[test]
fn auto_input_ceiling_is_enforced() {
    let limits = Limits {
        max_input: 8,
        ..Limits::default()
    };
    match inflate_auto(&[0u8; 16], &limits) {
        Err(Error::TooLarge { .. }) => {}
        other => panic!("oversized auto input must be TooLarge, got {other:?}"),
    }
}

#[test]
fn one_byte_zlib_header_is_truncated() {
    match inflate_zlib(&[0x78], &Limits::default()) {
        Err(Error::Truncated { .. }) => {}
        other => panic!("a lone CMF byte must be Truncated, got {other:?}"),
    }
}

#[test]
fn zlib_window_size_above_seven_is_invalid_magic() {
    // CMF 0x88: method 8, CINFO 8. FLG is irrelevant: the window test runs
    // before the FCHECK test.
    match inflate_zlib(&[0x88, 0x00], &Limits::default()) {
        Err(Error::InvalidMagic { .. }) => {}
        other => panic!("CINFO 8 must be InvalidMagic, got {other:?}"),
    }
}

#[test]
fn zlib_trailer_shorter_than_four_bytes_is_truncated() {
    // A complete stored-block stream ("abc") followed by only two of the
    // four Adler-32 trailer bytes.
    let input = [
        0x78, 0x01, 0x01, 0x03, 0x00, 0xfc, 0xff, 0x61, 0x62, 0x63, 0x00, 0x01,
    ];
    match inflate_zlib(&input, &Limits::default()) {
        Err(Error::Truncated { .. }) => {}
        other => panic!("a short trailer must be Truncated, got {other:?}"),
    }
}

#[test]
fn stored_block_data_cut_short_is_truncated() {
    // LEN claims 3 bytes, NLEN is correct, but only 1 data byte follows.
    let input = [0x01, 0x03, 0x00, 0xfc, 0xff, 0x61];
    match inflate_raw(&input, &Limits::default()) {
        Err(Error::Truncated { .. }) => {}
        other => panic!("cut stored data must be Truncated, got {other:?}"),
    }
}

#[test]
fn bits_ending_inside_a_huffman_code_is_truncated() {
    // BFINAL=1, BTYPE=fixed, then the first three bits of literal 'A'
    // (8-bit code 0b01110001): the decode walk runs off the end of input.
    let mut bits = Bits::new();
    bits.field(1, 1);
    bits.field(1, 2);
    bits.code(0b011, 3);
    match inflate_raw(&bits.bytes, &Limits::default()) {
        Err(Error::Truncated { .. }) => {}
        other => panic!("a cut code must be Truncated, got {other:?}"),
    }
}

/// A dynamic header whose code-length code is the complete four-symbol
/// tree the harness's one-symbol test uses: canonical codes 1=>00, 2=>01,
/// 17=>10, 18=>11.
fn cl_tree_1_2_17_18(bits: &mut Bits, hlit: u32, hdist: u32) {
    let mut cl = [0u8; 19];
    cl[1] = 2;
    cl[2] = 2;
    cl[17] = 2;
    cl[18] = 2;
    // dynamic_header's logic, inlined: HCLEN covers entries up to the last
    // nonzero, minimum 4.
    const ORDER: [usize; 19] = [
        16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
    ];
    bits.field(1, 1); // BFINAL
    bits.field(2, 2); // BTYPE = dynamic
    bits.field(hlit, 5);
    bits.field(hdist, 5);
    let entries = ORDER
        .iter()
        .rposition(|&sym| cl[sym] != 0)
        .map(|i| i + 1)
        .unwrap_or(4)
        .max(4);
    bits.field(entries as u32 - 4, 4);
    for &sym in &ORDER[..entries] {
        bits.field(u32::from(cl[sym]), 3);
    }
}

#[test]
fn incomplete_multi_symbol_distance_tree_is_bad_value() {
    // Two distance symbols at length 2 use half the code space; that is
    // under-subscribed and is not the RFC's one-symbol special case.
    let mut bits = Bits::new();
    cl_tree_1_2_17_18(&mut bits, 0, 1); // 257 literal/length, 2 distance codes
    bits.code(0b00, 2); // 1: literal length 1
    bits.code(0b11, 2); // 18
    bits.field(138 - 11, 7); // zero x138
    bits.code(0b11, 2); // 18
    bits.field(119 - 11, 7); // zero x119: 258 literal lengths total
    bits.code(0b01, 2); // 2: distance length 2
    bits.code(0b01, 2); // 2: distance length 2
    assert_kind(&bits, "BadValue");
}

#[test]
fn code_length_repeat_with_no_previous_is_bad_value() {
    // Symbol 16 (repeat the previous length) as the very first code-length
    // symbol: there is no previous length to repeat.
    let mut bits = Bits::new();
    let mut cl = [0u8; 19];
    cl[16] = 1;
    cl[18] = 1;
    // Header with the two-symbol one-bit code-length tree: 16=>0, 18=>1.
    const ORDER: [usize; 19] = [
        16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
    ];
    bits.field(1, 1);
    bits.field(2, 2);
    bits.field(0, 5);
    bits.field(0, 5);
    let entries = ORDER
        .iter()
        .rposition(|&sym| cl[sym] != 0)
        .map(|i| i + 1)
        .unwrap_or(4)
        .max(4);
    bits.field(entries as u32 - 4, 4);
    for &sym in &ORDER[..entries] {
        bits.field(u32::from(cl[sym]), 3);
    }
    bits.code(0, 1); // 16: repeat with nothing behind it
    assert_kind(&bits, "BadValue");
}

#[test]
fn code_length_run_over_the_table_is_bad_value() {
    // 319 filled lengths (257 literal + 32 distance + 30 extra) followed by
    // one 138-long run: the run crosses the 320-entry table and must be
    // refused instead of wrapping into it.
    let mut bits = Bits::new();
    cl_tree_1_2_17_18(&mut bits, 31, 31); // 288 literal/length + 32 distance
    bits.code(0b11, 2); // 18
    bits.field(138 - 11, 7); // zero x138
    bits.code(0b11, 2); // 18
    bits.field(138 - 11, 7); // zero x138: 276 so far
    for _ in 0..4 {
        bits.code(0b10, 2); // 17
        bits.field(10 - 3, 3); // zero x10: 316 so far
    }
    bits.code(0b10, 2); // 17
    bits.field(3, 3); // zero x3: 319 filled, 320 claimed
    bits.code(0b11, 2); // 18
    bits.field(138 - 11, 7); // a run that cannot fit: refused
    assert_kind(&bits, "BadValue");
}

#[test]
fn literal_at_the_output_ceiling_is_too_large() {
    // Block one is a stored block that fills the ceiling exactly; block
    // two is a fixed block whose first literal would exceed it.
    let mut bits = Bits::new();
    bits.field(0, 1); // BFINAL = 0
    bits.field(0, 2); // BTYPE = stored
    bits.align();
    bits.field(4, 16); // LEN
    bits.field(0xfffb, 16); // NLEN = !LEN
    for byte in b"abcd" {
        bits.field(u32::from(*byte), 8);
    }
    bits.field(1, 1); // BFINAL = 1
    bits.field(1, 2); // BTYPE = fixed
    bits.code(0b00110000 + 0x58, 8); // literal 'X'
    let limits = Limits {
        max_output: 4,
        ..Limits::default()
    };
    match inflate_raw(&bits.bytes, &limits) {
        Err(Error::TooLarge { .. }) => {}
        other => panic!("a literal past the ceiling must be TooLarge, got {other:?}"),
    }
}

#[test]
fn reserved_length_code_is_bad_value() {
    // Fixed-block symbols 286 and 287 are not length codes; symbol 286 must
    // be refused, not read as length 3.
    let mut bits = Bits::new();
    bits.field(1, 1); // BFINAL
    bits.field(1, 2); // BTYPE = fixed
    bits.code(0b01110001, 8); // literal 'A'
    bits.code(0b11000110, 8); // symbol 286
    assert_kind(&bits, "BadValue");
}

#[test]
fn reserved_distance_code_is_bad_value() {
    // The fixed distance table has 32 five-bit codes; codes 30 and 31 are
    // unassigned and must be refused.
    let mut bits = Bits::new();
    bits.field(1, 1); // BFINAL
    bits.field(1, 2); // BTYPE = fixed
    bits.code(0b01110001, 8); // literal 'A'
    bits.code(0b0000001, 7); // length code 257
    bits.code(0b11110, 5); // distance code 30
    assert_kind(&bits, "BadValue");
}
