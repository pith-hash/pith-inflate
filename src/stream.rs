//! Incremental decode: a `feed`/`finish` state machine over the same
//! DEFLATE core the one-shot entry points run.
//!
//! [`StreamingDecoder`] accepts input as it arrives (a socket, a pipe, a
//! chunked file read), decodes everything the buffered bytes allow after
//! every `feed`, and validates the container's trailer at `finish`. The
//! DEFLATE core underneath is the crate's only one: [`crate::inflate_raw`]
//! and [`crate::inflate_zlib`] drive the very same block-level functions
//! (`read_block_header`, `stored_header`, `inflate_one_symbol`), so
//! streamed and one-shot output of the same stream are byte-identical by
//! construction, not by test tolerance.
//!
//! Resume mechanics: every step of the state machine reads input before
//! it writes output, so a truncated read leaves the output untouched and
//! the bit position restorable. A step that runs out of input rolls back
//! and reports "stalled"; the bytes fed next are appended and the step
//! retried whole. Decoded output is retained (bounded by
//! [`Limits::max_output`], exactly like the one-shot API) and handed to
//! the caller at `finish`; the input buffer is compacted at `feed`
//! boundaries so retained input tracks the stream tail, not the whole
//! stream.
//!
//! gzip framing keeps the RFC 1952 container rules: optional FEXTRA /
//! FNAME / FCOMMENT / FHCRC header fields, the CRC-32 + ISIZE trailer
//! per member (checksums through `pith-digest`), and multi-member
//! streams, whose outputs concatenate.

use alloc::boxed::Box;
use alloc::vec::Vec;
use pith_digest::Error;

use crate::{
    BitReader, Huffman, Limits, dynamic_tables, fixed_tables, inflate_one_symbol, looks_like_zlib,
    read_block_header, stored_header,
};

/// Which container framing a [`StreamingDecoder`] accepts.
///
/// The variants mirror the one-shot entry points: [`Framing::Raw`] is
/// [`crate::inflate_raw`], [`Framing::Zlib`] is [`crate::inflate_zlib`],
/// [`Framing::Gzip`] is [`crate::inflate_gzip`], and [`Framing::Auto`]
/// sniffs the first two bytes exactly like [`crate::inflate_auto`] -
/// a structurally valid zlib header picks zlib, anything else is
/// decoded as raw DEFLATE, and gzip magic is refused with
/// [`Error::Unsupported`] so the caller learns what it actually has.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Framing {
    /// Raw DEFLATE (RFC 1951): the block sequence and nothing else.
    Raw,
    /// zlib (RFC 1950): two-byte header, DEFLATE, big-endian Adler-32.
    Zlib,
    /// gzip (RFC 1952): header, optional fields, DEFLATE, CRC-32 + ISIZE
    /// per member; multi-member streams concatenate.
    Gzip,
    /// Sniff the framing off the first two bytes, like
    /// [`crate::inflate_auto`].
    Auto,
}

/// One step's outcome: `true` means the state machine advanced and may
/// advance again, `false` means it stalled waiting for input, an error
/// fails the stream.
type Step = Result<bool, Error>;

/// The phases of the state machine. `byte_pos` (byte-aligned container
/// phases) and `bitpos` (bit-aligned DEFLATE phases) are absolute stream
/// positions: `base` counts input bytes already compacted out of `buf`,
/// so both survive buffer reallocation and compaction unchanged.
enum Phase {
    /// `Auto` only: waiting for the two sniff bytes.
    Sniff,
    /// zlib: the CMF/FLG header.
    ZlibHeader,
    /// gzip: the ten-byte fixed header.
    GzipFixedHeader,
    /// gzip FEXTRA: the little-endian XLEN field.
    GzipExtraLen,
    /// gzip FEXTRA: the extra-field bytes, ending at absolute offset
    /// `end`.
    GzipExtraBody { end: usize },
    /// gzip FNAME or FCOMMENT: a NUL-terminated run of bytes; `what`
    /// names which, for the truncation report.
    GzipZeroTerm { what: &'static str },
    /// gzip FHCRC: the little-endian header CRC16.
    GzipHeaderCrc,
    /// The DEFLATE payload proper.
    Deflate(DeflatePhase),
    /// zlib: the big-endian Adler-32 trailer, `got` of four bytes read.
    ZlibTrailer { got: u8, bytes: [u8; 4] },
    /// gzip: the little-endian CRC-32 trailer, `got` of four bytes read.
    GzipCrc { got: u8, bytes: [u8; 4] },
    /// gzip: the little-endian ISIZE trailer, `got` of four bytes read.
    GzipIsize { got: u8, bytes: [u8; 4] },
    /// The stream decoded completely. Raw stays here whatever is fed
    /// next (trailing bytes are ignored, as the one-shot documents);
    /// zlib only reaches this phase with the trailer at the exact end.
    Done,
}

/// The phases of the DEFLATE payload itself.
enum DeflatePhase {
    /// The two header bits of the next block.
    BlockHeader,
    /// A stored block's LEN/NLEN header.
    StoredLen { bfinal: u32 },
    /// A stored block's LEN payload bytes.
    StoredBody { bfinal: u32, len: usize },
    /// A Huffman block's body.
    Huffman {
        bfinal: u32,
        literal: Box<Huffman>,
        distance: Box<Huffman>,
    },
    /// The final block finished.
    Finished,
}

/// An incremental decoder for one stream of one [`Framing`].
///
/// Constructed with hard [`Limits`] (there is no unlimited mode,
/// matching the one-shot API); fed bytes as they arrive; finished
/// exactly once. After any hard error the decoder is poisoned: every
/// later call returns that same error, because the stream state that
/// would produce anything else is gone. A stall - the stream merely
/// needing more bytes - is never an error.
///
/// ```
/// use pith_inflate::{Framing, Limits, StreamingDecoder};
///
/// let mut d = StreamingDecoder::new(Framing::Raw, Limits::default());
/// d.feed(&[0x73, 0x04]).unwrap();
/// d.feed(&[0x00]).unwrap();
/// assert_eq!(d.finish().unwrap(), b"A");
/// ```
pub struct StreamingDecoder {
    limits: Limits,
    /// The framing in force: the requested one, or the sniff result for
    /// `Auto`.
    resolved: Option<Framing>,
    phase: Phase,
    /// Input bytes not yet fully consumed.
    buf: Vec<u8>,
    /// How many input bytes have been compacted out of `buf`'s front.
    base: usize,
    /// Absolute byte position for byte-aligned phases.
    byte_pos: usize,
    /// Absolute bit position for bit-aligned phases.
    bitpos: usize,
    out: Vec<u8>,
    /// gzip: output offset where the current member's output began.
    member_out_start: usize,
    /// gzip: how many members decoded completely so far.
    members: usize,
    /// gzip: absolute input offset where the current member's header
    /// starts - the byte range the FHCRC checksums. `usize::MAX` when
    /// the header is past its CRC or the framing is not gzip.
    header_crc_start: usize,
    /// gzip: FLG bits of the pending member's optional fields that are
    /// still to be consumed, in RFC order (FEXTRA, FNAME, FCOMMENT,
    /// FHCRC).
    flg_pending: u8,
    /// FNAME/FCOMMENT: how far the NUL scan has looked, absolute, so a
    /// long unterminated field is not rescanned from scratch per feed.
    scan_upto: usize,
    /// Total bytes ever fed, checked against `limits.max_input`.
    total_input: usize,
    failed: Option<Error>,
}

impl StreamingDecoder {
    /// A decoder for one stream of `framing`, under hard `limits`.
    pub fn new(framing: Framing, limits: Limits) -> Self {
        let phase = match framing {
            Framing::Raw => Phase::Deflate(DeflatePhase::BlockHeader),
            Framing::Zlib => Phase::ZlibHeader,
            Framing::Gzip => Phase::GzipFixedHeader,
            Framing::Auto => Phase::Sniff,
        };
        StreamingDecoder {
            limits,
            resolved: match framing {
                Framing::Auto => None,
                other => Some(other),
            },
            phase,
            buf: Vec::new(),
            base: 0,
            byte_pos: 0,
            bitpos: 0,
            out: Vec::new(),
            member_out_start: 0,
            members: 0,
            header_crc_start: usize::MAX,
            flg_pending: 0,
            scan_upto: 0,
            total_input: 0,
            failed: None,
        }
    }

    /// Feeds more input and decodes everything the buffered bytes
    /// allow. Returns the number of decoded output bytes so far - the
    /// [`output`](Self::output) prefix grows monotonically across calls.
    ///
    /// An empty `input` is legal and changes nothing. Hard errors (bad
    /// magic, corrupt checksums, ceilings, trailing bytes after a zlib
    /// stream) fail the call and poison the decoder; needing more input
    /// never does - that is what `feed` exists for.
    pub fn feed(&mut self, input: &[u8]) -> Result<usize, Error> {
        if let Some(e) = self.failed {
            return Err(e);
        }
        if input.is_empty() {
            return Ok(self.out.len());
        }
        self.total_input += input.len();
        if self.total_input > self.limits.max_input {
            let e = Error::too_large("streaming input", self.limits.max_input);
            self.failed = Some(e);
            return Err(e);
        }
        if matches!(self.phase, Phase::Done) {
            // Framing-specific trailing-byte rules, mirroring the
            // one-shot entry points: raw ignores what follows the final
            // block; zlib already proved nothing may follow it.
            if self.resolved == Some(Framing::Zlib) {
                let e = Error::BadValue("trailing bytes after the zlib stream");
                self.failed = Some(e);
                return Err(e);
            }
            return Ok(self.out.len());
        }
        self.buf.extend_from_slice(input);
        self.compact();
        if let Err(e) = self.run() {
            self.failed = Some(e);
            return Err(e);
        }
        self.compact();
        Ok(self.out.len())
    }

    /// Ends the stream: validates completeness and trailers, and returns
    /// the decoded output. Consumes the decoder.
    ///
    /// A stream that stops inside any structure - a half-read header, a
    /// block that never ends, a trailer missing bytes - is
    /// [`Error::Truncated`], never a partial success.
    pub fn finish(mut self) -> Result<Vec<u8>, Error> {
        if let Some(e) = self.failed {
            return Err(e);
        }
        // Auto that never saw two bytes resolves to raw, exactly like
        // the one-shot sniff, and the decoder then fails or succeeds on
        // the merits of those bytes.
        if matches!(self.phase, Phase::Sniff) {
            self.resolved = Some(Framing::Raw);
            self.phase = Phase::Deflate(DeflatePhase::BlockHeader);
            self.bitpos = 0;
        }
        self.run()?;
        match self.phase {
            // zlib at its exact Adler-32 end, or raw after its final
            // block (trailing bytes ignored).
            Phase::Done => Ok(self.out),
            // gzip at a member boundary with nothing fed past it: the
            // multi-member end state.
            Phase::GzipFixedHeader if self.members > 0 && self.available() == 0 => Ok(self.out),
            Phase::Deflate(_) => Err(Error::truncated(
                "DEFLATE stream",
                self.available() + 1,
                self.available(),
            )),
            _ => Err(self.truncation_error()),
        }
    }

    /// The decoded output produced so far. It is always a prefix of the
    /// stream's final output.
    pub fn output(&self) -> &[u8] {
        &self.out
    }

    /// Whether the stream decoded completely - framing checks included:
    /// a gzip stream is done only once a member's CRC-32 and ISIZE have
    /// both validated (and nothing has been fed past that member
    /// boundary), a zlib stream only at its exact Adler-32 end.
    pub fn is_done(&self) -> bool {
        match self.phase {
            Phase::Done => true,
            Phase::GzipFixedHeader => self.members > 0 && self.available() == 0,
            _ => false,
        }
    }

    /// Bytes available from the byte cursor to the end of the buffer.
    fn available(&self) -> usize {
        self.base + self.buf.len() - self.byte_pos
    }

    /// Drops consumed bytes from the buffer front, keeping every
    /// position the state machine still references: the byte containing
    /// the next unread bit during DEFLATE phases, the byte cursor
    /// otherwise, and a gzip header still pending its CRC16.
    fn compact(&mut self) {
        let mut keep_from = if matches!(self.phase, Phase::Deflate(_)) {
            self.bitpos / 8
        } else {
            self.byte_pos
        };
        if self.header_crc_start != usize::MAX {
            keep_from = keep_from.min(self.header_crc_start);
        }
        let drop = keep_from.saturating_sub(self.base);
        if drop > 0 {
            self.buf.drain(..drop);
            self.base += drop;
        }
    }

    /// Drives steps until one stalls or a hard error fires. A stalled
    /// step is not an error here: `feed` simply waits for more bytes.
    fn run(&mut self) -> Result<(), Error> {
        while self.step()? {}
        Ok(())
    }

    /// The truncation error for the phase the stream stopped in, with a
    /// `what` naming the structure and honest needed/found counts.
    fn truncation_error(&self) -> Error {
        let (what, needed) = match &self.phase {
            Phase::ZlibHeader => ("zlib header", 2),
            Phase::ZlibTrailer { got, .. } => ("zlib Adler-32 trailer", 4 - *got as usize),
            Phase::GzipFixedHeader => {
                if self.members > 0 {
                    ("gzip member header", 10)
                } else {
                    ("gzip header", 10)
                }
            }
            Phase::GzipExtraLen => ("gzip FEXTRA length", 2),
            Phase::GzipExtraBody { end } => ("gzip FEXTRA data", end - self.byte_pos),
            Phase::GzipZeroTerm { what } => (*what, 1),
            Phase::GzipHeaderCrc => ("gzip FHCRC", 2),
            Phase::GzipCrc { got, .. } => ("gzip CRC-32 trailer", 4 - *got as usize),
            Phase::GzipIsize { got, .. } => ("gzip ISIZE trailer", 4 - *got as usize),
            Phase::Sniff | Phase::Deflate(_) | Phase::Done => {
                unreachable!("handled by finish before this runs")
            }
        };
        Error::truncated(what, needed, self.available())
    }

    /// Runs one state-machine step. Every step owns its transition: a
    /// step that stalls leaves `self.phase` exactly where it was, a
    /// progress step writes the next phase before returning, and a hard
    /// error poisons the decoder. Truncation never escapes a step: each
    /// step checks its input up front (byte phases) or rolls its bit
    /// cursor back (DEFLATE) and reports a stall instead, so `run` can
    /// simply loop until a step stalls.
    fn step(&mut self) -> Step {
        self.take_step()
    }

    /// The step dispatcher: one phase, one transition.
    fn take_step(&mut self) -> Step {
        match self.phase {
            Phase::Sniff => self.sniff_step(),
            Phase::ZlibHeader => self.zlib_header_step(),
            Phase::GzipFixedHeader => self.gzip_fixed_header_step(),
            Phase::GzipExtraLen => self.gzip_extra_len_step(),
            Phase::GzipExtraBody { end } => self.gzip_extra_body_step(end),
            Phase::GzipZeroTerm { what } => self.gzip_zero_term_step(what),
            Phase::GzipHeaderCrc => self.gzip_header_crc_step(),
            Phase::ZlibTrailer { got, bytes } => self.zlib_trailer_step(got, bytes),
            Phase::GzipCrc { got, bytes } => self.gzip_crc_step(got, bytes),
            Phase::GzipIsize { got, bytes } => self.gzip_isize_step(got, bytes),
            Phase::Done => Ok(false),
            Phase::Deflate(_) => {
                let taken = core::mem::replace(&mut self.phase, Phase::Done);
                let Phase::Deflate(dp) = taken else {
                    unreachable!("the match arm guarantees the DEFLATE phase")
                };
                self.deflate_step(dp)
            }
        }
    }

    /// `Auto`: pick the framing off the first two bytes.
    fn sniff_step(&mut self) -> Step {
        if self.available() < 2 {
            return Ok(false);
        }
        let start = self.byte_pos - self.base;
        let two = [self.buf[start], self.buf[start + 1]];
        if two[0] == 0x1f && two[1] == 0x8b {
            // Exactly the one-shot sniff's verdict: auto frames raw
            // DEFLATE and zlib only, and names what it refused.
            return Err(Error::Unsupported("gzip"));
        }
        if looks_like_zlib(&two) {
            self.resolved = Some(Framing::Zlib);
            self.phase = Phase::ZlibHeader;
        } else {
            self.resolved = Some(Framing::Raw);
            self.phase = Phase::Deflate(DeflatePhase::BlockHeader);
            self.bitpos = self.byte_pos * 8;
        }
        Ok(true)
    }

    /// zlib: the CMF/FLG header - the same checks, in the same order,
    /// as the one-shot `inflate_zlib`: FDICT before FCHECK, method and
    /// window before both.
    fn zlib_header_step(&mut self) -> Step {
        if self.available() < 2 {
            return Ok(false);
        }
        let start = self.byte_pos - self.base;
        let (cmf, flg) = (self.buf[start], self.buf[start + 1]);
        if cmf & 0x0f != 8 {
            return Err(Error::InvalidMagic {
                what: "zlib CMF compression method",
            });
        }
        if cmf >> 4 > 7 {
            return Err(Error::InvalidMagic {
                what: "zlib CMF window size (CINFO)",
            });
        }
        if flg & 0x20 != 0 {
            return Err(Error::Unsupported("zlib preset dictionary (FDICT)"));
        }
        if ((u16::from(cmf) << 8) | u16::from(flg)) % 31 != 0 {
            return Err(Error::InvalidMagic {
                what: "zlib FCHECK header check bits",
            });
        }
        self.byte_pos += 2;
        self.bitpos = self.byte_pos * 8;
        self.phase = Phase::Deflate(DeflatePhase::BlockHeader);
        Ok(true)
    }

    /// gzip: the ten-byte fixed header - magic, method, reserved FLG
    /// bits - then chain into the optional fields.
    fn gzip_fixed_header_step(&mut self) -> Step {
        if self.available() < 10 {
            return Ok(false);
        }
        let start = self.byte_pos - self.base;
        let head = &self.buf[start..start + 10];
        if head[0] != 0x1f || head[1] != 0x8b {
            return Err(Error::InvalidMagic { what: "gzip magic" });
        }
        if head[2] != 8 {
            return Err(Error::InvalidMagic {
                what: "gzip CM compression method",
            });
        }
        if head[3] & 0xe0 != 0 {
            return Err(Error::InvalidMagic {
                what: "gzip FLG reserved bits",
            });
        }
        self.byte_pos += 10;
        self.flg_pending = head[3];
        self.header_crc_start = self.byte_pos - 10;
        self.scan_upto = self.byte_pos;
        self.advance_header_chain();
        Ok(true)
    }

    /// gzip FEXTRA: the little-endian XLEN field.
    fn gzip_extra_len_step(&mut self) -> Step {
        if self.available() < 2 {
            return Ok(false);
        }
        let start = self.byte_pos - self.base;
        let xlen = u16::from_le_bytes([self.buf[start], self.buf[start + 1]]) as usize;
        self.byte_pos += 2;
        self.scan_upto = self.byte_pos;
        self.phase = Phase::GzipExtraBody {
            end: self.byte_pos + xlen,
        };
        Ok(true)
    }

    /// gzip FEXTRA: the XLEN extra-field bytes, skipped byte-exactly.
    fn gzip_extra_body_step(&mut self, end: usize) -> Step {
        if self.available() < end - self.byte_pos {
            return Ok(false);
        }
        self.byte_pos = end;
        self.scan_upto = self.byte_pos;
        self.advance_header_chain();
        Ok(true)
    }

    /// gzip FNAME/FCOMMENT: skip to and past the NUL terminator. The
    /// scan resumes where the previous feed stopped, so a long
    /// unterminated field costs one pass per fed byte, not one pass per
    /// feed over the whole field.
    fn gzip_zero_term_step(&mut self, _what: &'static str) -> Step {
        let from = self.scan_upto.max(self.byte_pos) - self.base;
        if let Some(nul) = self.buf[from..].iter().position(|&b| b == 0) {
            self.byte_pos = self.base + from + nul + 1;
            self.scan_upto = self.byte_pos;
            self.advance_header_chain();
            Ok(true)
        } else {
            self.scan_upto = self.base + self.buf.len();
            Ok(false)
        }
    }

    /// gzip FHCRC: the little-endian CRC16 of every header byte before
    /// it - CRC-32's low half, through pith-digest like every other
    /// checksum in this crate.
    fn gzip_header_crc_step(&mut self) -> Step {
        if self.available() < 2 {
            return Ok(false);
        }
        let start = self.byte_pos - self.base;
        let stored = u16::from_le_bytes([self.buf[start], self.buf[start + 1]]);
        let header = &self.buf[self.header_crc_start - self.base..start];
        if (pith_digest::crc32(header) & 0xffff) as u16 != stored {
            return Err(Error::BadValue("gzip FHCRC header check mismatch"));
        }
        self.byte_pos += 2;
        self.header_crc_start = usize::MAX;
        self.bitpos = self.byte_pos * 8;
        self.phase = Phase::Deflate(DeflatePhase::BlockHeader);
        Ok(true)
    }

    /// Chains to the next pending optional header field in FLG bit
    /// order (FEXTRA, FNAME, FCOMMENT, FHCRC), or starts the DEFLATE
    /// payload at the current byte boundary.
    fn advance_header_chain(&mut self) {
        let flags = self.flg_pending;
        let (consumed, phase) = if flags & 0x04 != 0 {
            (0x04, Phase::GzipExtraLen)
        } else if flags & 0x08 != 0 {
            (0x08, Phase::GzipZeroTerm { what: "gzip FNAME" })
        } else if flags & 0x10 != 0 {
            (
                0x10,
                Phase::GzipZeroTerm {
                    what: "gzip FCOMMENT",
                },
            )
        } else if flags & 0x02 != 0 {
            (0x02, Phase::GzipHeaderCrc)
        } else {
            self.bitpos = self.byte_pos * 8;
            (0, Phase::Deflate(DeflatePhase::BlockHeader))
        };
        self.flg_pending = flags & !consumed;
        self.phase = phase;
    }

    /// zlib trailer: one byte per step, then the Adler-32 check and the
    /// exact-end rule the one-shot applies.
    fn zlib_trailer_step(&mut self, mut got: u8, mut bytes: [u8; 4]) -> Step {
        if self.available() < 1 {
            return Ok(false);
        }
        let start = self.byte_pos - self.base;
        bytes[got as usize] = self.buf[start];
        self.byte_pos += 1;
        got += 1;
        if got < 4 {
            self.phase = Phase::ZlibTrailer { got, bytes };
            return Ok(true);
        }
        let stored = u32::from_be_bytes(bytes);
        if crate::adler32(&self.out) != stored {
            return Err(Error::BadValue("zlib Adler-32 trailer mismatch"));
        }
        self.phase = Phase::Done;
        if self.byte_pos < self.base + self.buf.len() {
            return Err(Error::BadValue("trailing bytes after the zlib stream"));
        }
        Ok(true)
    }

    /// gzip CRC-32 trailer: one little-endian byte per step, then the
    /// check over the current member's output.
    fn gzip_crc_step(&mut self, mut got: u8, mut bytes: [u8; 4]) -> Step {
        if self.available() < 1 {
            return Ok(false);
        }
        let start = self.byte_pos - self.base;
        bytes[got as usize] = self.buf[start];
        self.byte_pos += 1;
        got += 1;
        if got < 4 {
            self.phase = Phase::GzipCrc { got, bytes };
            return Ok(true);
        }
        let stored = u32::from_le_bytes(bytes);
        if pith_digest::crc32(&self.out[self.member_out_start..]) != stored {
            return Err(Error::BadValue("gzip CRC-32 trailer mismatch"));
        }
        self.phase = Phase::GzipIsize {
            got: 0,
            bytes: [0; 4],
        };
        Ok(true)
    }

    /// gzip ISIZE trailer: one little-endian byte per step, then the
    /// check against this member's output length modulo 2^32, then the
    /// member boundary - the next member's header follows.
    fn gzip_isize_step(&mut self, mut got: u8, mut bytes: [u8; 4]) -> Step {
        if self.available() < 1 {
            return Ok(false);
        }
        let start = self.byte_pos - self.base;
        bytes[got as usize] = self.buf[start];
        self.byte_pos += 1;
        got += 1;
        if got < 4 {
            self.phase = Phase::GzipIsize { got, bytes };
            return Ok(true);
        }
        let stored = u32::from_le_bytes(bytes);
        if (self.out.len() - self.member_out_start) as u32 != stored {
            return Err(Error::BadValue("gzip ISIZE trailer mismatch"));
        }
        self.members += 1;
        self.member_out_start = self.out.len();
        self.header_crc_start = self.byte_pos;
        self.scan_upto = self.byte_pos;
        self.phase = Phase::GzipFixedHeader;
        Ok(true)
    }

    /// DEFLATE: one step through a fresh [`BitReader`] over the buffer.
    fn deflate_step(&mut self, dp: DeflatePhase) -> Step {
        let saved = self.bitpos;
        let mut reader = BitReader::new(&self.buf);
        reader.bitpos = self.bitpos - self.base * 8;
        let mut dp = dp;
        let outcome = deflate_step_inner(&mut dp, &mut reader, &mut self.out, &self.limits);
        match outcome {
            Ok(progress) => {
                self.bitpos = self.base * 8 + reader.bitpos;
                self.phase = match dp {
                    // The final block ended: the bit stream rounds up to
                    // the next byte boundary and hands over to the
                    // framing's trailer (or ends the raw stream, whose
                    // trailing bytes are ignored). The header-CRC span
                    // is over either way.
                    DeflatePhase::Finished => {
                        self.byte_pos = self.bitpos.div_ceil(8);
                        self.header_crc_start = usize::MAX;
                        match self.resolved {
                            Some(Framing::Zlib) => Phase::ZlibTrailer {
                                got: 0,
                                bytes: [0; 4],
                            },
                            Some(Framing::Gzip) => Phase::GzipCrc {
                                got: 0,
                                bytes: [0; 4],
                            },
                            _ => Phase::Done,
                        }
                    }
                    other => Phase::Deflate(other),
                };
                Ok(progress)
            }
            Err(Error::Truncated { .. }) => {
                // The whole step re-runs when more input arrives.
                self.bitpos = saved;
                self.phase = Phase::Deflate(dp);
                Ok(false)
            }
            Err(e) => Err(e),
        }
    }
}

/// One DEFLATE state-machine step: a block header, a stored header or
/// body, a table build, or one Huffman symbol. Input reads all precede
/// output effects, so a truncated step is cleanly retriable.
fn deflate_step_inner(
    dp: &mut DeflatePhase,
    reader: &mut BitReader<'_>,
    out: &mut Vec<u8>,
    limits: &Limits,
) -> Step {
    match dp {
        DeflatePhase::BlockHeader => {
            let (bfinal, btype) = read_block_header(reader)?;
            *dp = match btype {
                0 => DeflatePhase::StoredLen { bfinal },
                1 => {
                    let (literal, distance) = fixed_tables()?;
                    DeflatePhase::Huffman {
                        bfinal,
                        literal: Box::new(literal),
                        distance: Box::new(distance),
                    }
                }
                2 => {
                    let (literal, distance) = dynamic_tables(reader)?;
                    DeflatePhase::Huffman {
                        bfinal,
                        literal: Box::new(literal),
                        distance: Box::new(distance),
                    }
                }
                // RFC 1951 3.2.5: a compliant decoder must refuse type 3.
                _ => return Err(Error::BadValue("reserved block type 3")),
            };
            Ok(true)
        }
        DeflatePhase::StoredLen { bfinal } => {
            let len = stored_header(reader, out, limits)?;
            *dp = DeflatePhase::StoredBody {
                bfinal: *bfinal,
                len,
            };
            Ok(true)
        }
        DeflatePhase::StoredBody { bfinal, len } => {
            let bytes = reader.take(*len)?;
            out.extend_from_slice(bytes);
            *dp = next_block_or_finish(*bfinal);
            Ok(true)
        }
        DeflatePhase::Huffman {
            bfinal,
            literal,
            distance,
        } => {
            let end_of_block = inflate_one_symbol(reader, out, literal, distance, limits)?;
            if end_of_block {
                *dp = next_block_or_finish(*bfinal);
            }
            Ok(true)
        }
        DeflatePhase::Finished => Ok(false),
    }
}

/// Where the DEFLATE phase goes after a block ends: the next block's
/// header, or finished.
fn next_block_or_finish(bfinal: u32) -> DeflatePhase {
    if bfinal == 1 {
        DeflatePhase::Finished
    } else {
        DeflatePhase::BlockHeader
    }
}
