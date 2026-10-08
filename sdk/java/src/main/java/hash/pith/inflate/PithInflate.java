// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pith-hash
package hash.pith.inflate;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Java JNI bindings for the {@code pith-inflate} cdylib: DEFLATE,
 * zlib and gzip decompression under hard size limits — the same C
 * library the Python (ctypes), Node (koffi) and Go (cgo) SDKs bind
 * through.
 *
 * <p>The cdylib is resolved once at class-load time, mirroring the
 * discovery chain of the other SDKs: (1) the {@code PITH_CDYLIB}
 * environment variable — the explicit file; (2) {@code PITH_CDYLIB_DIR}
 * — a directory holding one of the platform library names; (3) a
 * {@code target/release} directory at the working directory or up to
 * six ancestors above it. {@link LinkageError} names every candidate
 * when nothing matches.</p>
 *
 * <p>Every decompress operation returns the decompressed bytes; a
 * stream the core refuses raises {@link FfiError} with the C ABI
 * status code. Refusals never crash the JVM.</p>
 */
public final class PithInflate {

    /** Status: success. */
    public static final int PITH_OK = 0;

    /** Status: invalid argument — a null array. */
    public static final int PITH_E_INVALID = -1;

    /** Status: the core decoder refused the input (malformed, truncated or over-limit stream, or a framing the crate refuses). */
    public static final int PITH_E_REJECTED = -2;

    /** Platform cdylib file names, in probe order. */
    private static final String[] CDYLIB_NAMES = {
        "pith_inflate.dll", "libpith_inflate.so", "libpith_inflate.dylib",
    };

    private static final String CDYLIB_PATH = findCdylib();

    static {
        System.load(CDYLIB_PATH);
    }

    private PithInflate() {
    }

    /** The absolute path of the loaded cdylib (tests and diagnostics). */
    public static String cdylibPath() {
        return CDYLIB_PATH;
    }

    private static native byte[] inflateRawNative(byte[] data, int[] status);

    private static native byte[] inflateZlibNative(byte[] data, int[] status);

    private static native byte[] inflateAutoNative(byte[] data, int[] status);

    private static native byte[] inflateGzipNative(byte[] data, int[] status);

    private static native long adler32Native(byte[] data, int[] status);

    /**
     * Decompresses a raw RFC 1951 DEFLATE stream (trailing bytes after
     * the final block are ignored).
     *
     * @throws FfiError with status {@code PITH_E_REJECTED} for any
     *     malformed, truncated or over-limit input
     */
    public static byte[] inflateRaw(byte[] data) {
        return decompress("pith_inflate_inflate_raw", data, "inflateRawNative");
    }

    /**
     * Decompresses a zlib (RFC 1950) stream: two-byte header, DEFLATE
     * payload, big-endian Adler-32 trailer.
     *
     * @throws FfiError with status {@code PITH_E_REJECTED} for any
     *     malformed input, including a trailer that does not match the
     *     Adler-32 of the produced output
     */
    public static byte[] inflateZlib(byte[] data) {
        return decompress("pith_inflate_inflate_zlib", data, "inflateZlibNative");
    }

    /**
     * Decompresses either framing, sniffing the two-byte zlib header.
     * gzip is recognised and refused ({@code PITH_E_REJECTED}), never
     * mis-decoded.
     *
     * @throws FfiError with status {@code PITH_E_REJECTED} for any
     *     malformed input
     */
    public static byte[] inflateAuto(byte[] data) {
        return decompress("pith_inflate_inflate_auto", data, "inflateAutoNative");
    }

    /**
     * Decompresses a gzip (RFC 1952) container: the ten-byte header
     * (FEXTRA/FNAME/FCOMMENT/FHCRC optional fields included), the
     * DEFLATE payload, and the CRC-32 + ISIZE trailer. A multi-member
     * stream (concatenated {@code .gz} files) decodes to the
     * concatenation of every member's payload.
     *
     * @throws FfiError with status {@code PITH_E_REJECTED} for any
     *     malformed input, including a trailer whose CRC-32 or ISIZE
     *     does not match the produced output
     */
    public static byte[] inflateGzip(byte[] data) {
        return decompress("pith_inflate_inflate_gzip", data, "inflateGzipNative");
    }

    /**
     * Computes the Adler-32 checksum (RFC 1950 section 9) of {@code
     * data}. The checksum of an empty input is 1, so {@code
     * adler32(new byte[0]) == 1}.
     */
    public static long adler32(byte[] data) {
        int[] status = new int[1];
        long sum = adler32Native(data, status);
        if (status[0] != PITH_OK) {
            throw new FfiError("pith_inflate_adler32", status[0]);
        }
        return sum;
    }

    /** Runs one decompress wrapper and maps the status slot to FfiError. */
    private static byte[] decompress(String op, byte[] data, String nativeName) {
        if (data == null) {
            throw new FfiError(op, PITH_E_INVALID);
        }
        int[] status = new int[1];
        byte[] out = invoke(nativeName, data, status);
        if (status[0] != PITH_OK) {
            throw new FfiError(op, status[0]);
        }
        return out;
    }

    /** Dispatches to the named native method (all four share a shape). */
    private static byte[] invoke(String nativeName, byte[] data, int[] status) {
        switch (nativeName) {
            case "inflateRawNative":
                return inflateRawNative(data, status);
            case "inflateZlibNative":
                return inflateZlibNative(data, status);
            case "inflateAutoNative":
                return inflateAutoNative(data, status);
            default:
                return inflateGzipNative(data, status);
        }
    }

    /**
     * Incremental-feed decode canonicalized through the stateless
     * one-shot JNI surface.
     *
     * <p>The C ABI carries no decoder state — every export is a
     * one-shot — so this helper implements the feed/finish shape by
     * re-decoding the bytes buffered so far through the framing's
     * one-shot export on every call. That is O(n&sup2;) across a
     * stream in the worst case, correct by construction, and
     * byte-identical to the Rust {@code StreamingDecoder} by the
     * chunked-feed equivalence property the Rust test suite pins.
     * Throughput paths should call the one-shot methods directly.</p>
     *
     * <p>{@link #feed(byte[])} reports whether the bytes fed so far
     * already decode completely — for a multi-member stream that is
     * true at every member boundary, so keep feeding and let
     * {@link #finish()} give the final verdict. {@link #output()}
     * tracks the longest decoded prefix. An incomplete — or so far
     * invalid — stream reports {@code false}: the stateless FFI cannot
     * tell "needs more input" from "broken input".
     * {@link #finish()} gives the verdict, raising {@link FfiError}
     * for a stream that never decoded.</p>
     */
    public static final class StreamingInflater {
        /** The one-shot decode this canonicalization runs. */
        private final java.util.function.Function<byte[], byte[]> op;

        /** Every byte fed so far. */
        private byte[] buffer = new byte[0];

        /** The longest decoded prefix, or {@code null}. */
        private byte[] output;

        /**
         * @param framing "raw", "zlib" or "gzip"
         * @throws IllegalArgumentException for an unknown framing
         */
        public StreamingInflater(String framing) {
            switch (framing) {
                case "raw":
                    this.op = PithInflate::inflateRaw;
                    break;
                case "zlib":
                    this.op = PithInflate::inflateZlib;
                    break;
                case "gzip":
                    this.op = PithInflate::inflateGzip;
                    break;
                default:
                    throw new IllegalArgumentException(
                            "framing must be one of gzip, raw, zlib, got " + framing);
            }
        }

        /** Convenience constructor: the gzip framing. */
        public StreamingInflater() {
            this("gzip");
        }

        /**
         * Feeds the next chunk.
         *
         * @return true when the bytes fed so far decode completely
         * @throws FfiError with status {@code PITH_E_INVALID} for a
         *     null chunk
         */
        public boolean feed(byte[] chunk) {
            if (chunk == null) {
                throw new FfiError("pith_inflate_streaming_feed", PITH_E_INVALID);
            }
            byte[] grown = new byte[buffer.length + chunk.length];
            System.arraycopy(buffer, 0, grown, 0, buffer.length);
            System.arraycopy(chunk, 0, grown, buffer.length, chunk.length);
            buffer = grown;
            try {
                output = op.apply(buffer);
            } catch (FfiError e) {
                return false;
            }
            return true;
        }

        /** The decoded bytes; empty until the stream decoded. */
        public byte[] output() {
            return output == null ? new byte[0] : output.clone();
        }

        /**
         * Ends the stream: the decoded bytes.
         *
         * @throws FfiError when the stream never decoded (malformed,
         *     truncated or over-limit input)
         */
        public byte[] finish() {
            if (output == null) {
                output = op.apply(buffer);
            }
            return output.clone();
        }
    }

    /**
     * A native call refused or failed: the C ABI status code plus the
     * operation that reported it — the Java face of the ctypes/koffi/
     * cgo {@code FfiError}.
     */
    public static final class FfiError extends RuntimeException {
        private static final long serialVersionUID = 1L;

        /** The refusing operation (its C ABI name). */
        public final String op;

        /** The C ABI status code ({@code -1} invalid, {@code -2} rejected). */
        public final int status;

        FfiError(String op, int status) {
            super(op + " failed with status " + status);
            this.op = op;
            this.status = status;
        }
    }

    /**
     * Resolves the cdylib path: {@code PITH_CDYLIB} (explicit file),
     * then {@code PITH_CDYLIB_DIR} + platform name, then a
     * {@code target/release} directory at the working directory or up
     * to six ancestors above it.
     */
    private static String findCdylib() {
        Path cwd = Paths.get("").toAbsolutePath();

        String explicitFile = System.getenv("PITH_CDYLIB");
        if (explicitFile != null && !explicitFile.isEmpty()) {
            Path file = Paths.get(explicitFile).toAbsolutePath();
            if (Files.isRegularFile(file)) {
                return file.toString();
            }
        }

        String dir = System.getenv("PITH_CDYLIB_DIR");
        if (dir != null && !dir.isEmpty()) {
            Path base = Paths.get(dir).toAbsolutePath();
            for (String name : CDYLIB_NAMES) {
                Path candidate = base.resolve(name);
                if (Files.isRegularFile(candidate)) {
                    return candidate.toString();
                }
            }
        }

        for (Path base = cwd; base != null; base = base.getParent()) {
            for (String name : CDYLIB_NAMES) {
                Path candidate = base.resolve("target").resolve("release").resolve(name);
                if (Files.isRegularFile(candidate)) {
                    return candidate.toString();
                }
            }
        }

        throw new LinkageError(
                "cannot locate the pith_inflate cdylib; set PITH_CDYLIB or PITH_CDYLIB_DIR"
                + " (probed PITH_CDYLIB, PITH_CDYLIB_DIR, and target/release at "
                + cwd + " and its ancestors)");
    }
}
