// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pith-hash
package hash.pith.inflate;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.zip.CRC32;
import org.junit.jupiter.api.Test;

/**
 * Hex-exact conformance: the committed {@code reference.json} vectors
 * through JNI — the same vectors the Rust {@code gen-reference verify}
 * gate and the Python/Node/Go SDK suites replay, plus the refusal
 * paths (never a crash) and the streaming canonicalization's
 * chunked-feed equivalence.
 */
class PithInflateTest {

    private static final Path REPO_ROOT =
            Paths.get(System.getProperty("user.dir")).toAbsolutePath()
                    .getParent().getParent();

    private static JsonNode reference() throws Exception {
        Path file = REPO_ROOT.resolve("reference.json");
        assertTrue(Files.isRegularFile(file), file.toString());
        return new ObjectMapper().readTree(Files.readAllBytes(file));
    }

    private static byte[] hex(String s) {
        byte[] out = new byte[s.length() / 2];
        for (int i = 0; i < out.length; i++) {
            out[i] = (byte) Integer.parseInt(s.substring(i * 2, i * 2 + 2), 16);
        }
        return out;
    }

    /** The reference contract's structural sniff for bad vectors. */
    private static boolean looksLikeZlib(byte[] b) {
        return b.length >= 2
                && (b[0] & 0x0f) == 8
                && (b[0] >> 4) <= 7
                && ((((b[0] & 0xff) << 8) | (b[1] & 0xff)) % 31) == 0;
    }

    @Test
    void cdylibIsDiscoverable() {
        assertTrue(Files.isRegularFile(Paths.get(PithInflate.cdylibPath())));
    }

    @Test
    void referenceVectorsAreReproducedHexExact() throws Exception {
        for (JsonNode vector : reference().path("vectors")) {
            byte[] compressed = hex(vector.path("compressed").asText());
            byte[] expected = hex(vector.path("plain").asText());
            boolean zlib = vector.path("name").asText().endsWith("/zlib");

            byte[] out = zlib ? PithInflate.inflateZlib(compressed) : PithInflate.inflateRaw(compressed);
            assertArrayEquals(expected, out, vector.path("name").asText());
            assertEquals(vector.path("adler32").asText(),
                    String.format("%08x", PithInflate.adler32(out)),
                    vector.path("name").asText());
        }
    }

    @Test
    void gzipVectorsAreReproducedHexExact() throws Exception {
        for (JsonNode vector : reference().path("gzip_vectors")) {
            byte[] data = hex(vector.path("input").asText());
            byte[] expected = hex(vector.path("plain").asText());

            byte[] out = PithInflate.inflateGzip(data);
            assertArrayEquals(expected, out, vector.path("name").asText());
            // The CRC-32 the Rust decoder validated the trailer with,
            // cross-checked against the JDK's own java.util.zip.CRC32
            // so a wrongly generated reference fails loudly here too.
            CRC32 crc = new CRC32();
            crc.update(out);
            assertEquals(vector.path("crc32").asText(),
                    String.format("%08x", crc.getValue()),
                    vector.path("name").asText());
        }
    }

    @Test
    void badVectorsAreRefusedNotCrashing() throws Exception {
        JsonNode reference = reference();
        for (JsonNode vector : reference.path("bad_vectors")) {
            byte[] data = hex(vector.path("input").asText());
            String name = vector.path("name").asText();
            PithInflate.FfiError e = assertThrows(PithInflate.FfiError.class, () -> {
                byte[] ignored = looksLikeZlib(data)
                        ? PithInflate.inflateZlib(data)
                        : PithInflate.inflateRaw(data);
            }, () -> name);
            assertEquals(PithInflate.PITH_E_REJECTED, e.status, name);
        }
        for (JsonNode vector : reference().path("gzip_bad_vectors")) {
            byte[] data = hex(vector.path("input").asText());
            PithInflate.FfiError e = assertThrows(PithInflate.FfiError.class,
                    () -> PithInflate.inflateGzip(data),
                    vector.path("name").asText());
            assertEquals(PithInflate.PITH_E_REJECTED, e.status, vector.path("name").asText());
        }
    }

    @Test
    void truncatedStreamIsRefusedOnEveryRouting() {
        byte[] headerOnly = hex("78da");
        for (String[] routing : new String[][] {{"raw", "r"}, {"zlib", "z"}, {"auto", "a"}}) {
            PithInflate.FfiError e = assertThrows(PithInflate.FfiError.class, () -> {
                switch (routing[0]) {
                    case "raw":
                        PithInflate.inflateRaw(headerOnly);
                        break;
                    case "zlib":
                        PithInflate.inflateZlib(headerOnly);
                        break;
                    default:
                        PithInflate.inflateAuto(headerOnly);
                }
            });
            assertEquals(PithInflate.PITH_E_REJECTED, e.status);
        }
        // The auto sniff refuses gzip magic instead of guessing.
        PithInflate.FfiError gzipAuto = assertThrows(PithInflate.FfiError.class,
                () -> PithInflate.inflateAuto(hex("1f8b0800000000000203")));
        assertEquals(PithInflate.PITH_E_REJECTED, gzipAuto.status);
    }

    @Test
    void emptyStreamRoundTrips() {
        byte[] out = PithInflate.inflateZlib(hex("78da030000000001"));
        assertEquals(0, out.length);
        assertEquals(1L, PithInflate.adler32(out));
    }

    @Test
    void gzipStreamingFeedMatchesOneShot() throws Exception {
        for (JsonNode vector : reference().path("gzip_vectors")) {
            byte[] data = hex(vector.path("input").asText());
            byte[] expected = PithInflate.inflateGzip(data);
            for (int size : new int[] {1, 7, 64}) {
                PithInflate.StreamingInflater streamer = new PithInflate.StreamingInflater("gzip");
                for (int start = 0; start < data.length; start += size) {
                    int end = Math.min(start + size, data.length);
                    streamer.feed(java.util.Arrays.copyOfRange(data, start, end));
                    byte[] prefix = streamer.output();
                    assertTrue(equalsPrefix(expected, prefix), vector.path("name").asText());
                }
                assertArrayEquals(expected, streamer.finish(), vector.path("name").asText());
            }
        }
    }

    @Test
    void streamingRawAndZlibMatchOneShot() throws Exception {
        Map<String, String> framings = new LinkedHashMap<>();
        framings.put("one_byte", "raw");
        framings.put("one_byte/zlib", "zlib");
        for (Map.Entry<String, String> entry : framings.entrySet()) {
            JsonNode vector = null;
            for (JsonNode candidate : reference().path("vectors")) {
                if (entry.getKey().equals(candidate.path("name").asText())) {
                    vector = candidate;
                    break;
                }
            }
            byte[] data = hex(vector.path("compressed").asText());
            byte[] expected = hex(vector.path("plain").asText());
            PithInflate.StreamingInflater streamer = new PithInflate.StreamingInflater(entry.getValue());
            for (int i = 0; i < data.length; i++) {
                streamer.feed(java.util.Arrays.copyOfRange(data, i, i + 1));
            }
            assertArrayEquals(expected, streamer.finish(), entry.getKey());
        }
    }

    @Test
    void streamingFinishSurfacesTheRejection() throws Exception {
        JsonNode vector = reference().path("gzip_vectors").get(0);
        byte[] data = hex(vector.path("input").asText());
        byte[] quarter = java.util.Arrays.copyOfRange(data, 0, data.length / 4);
        PithInflate.StreamingInflater streamer = new PithInflate.StreamingInflater("gzip");
        assertFalse(streamer.feed(quarter), "a quarter of a stream must not decode");
        assertEquals(0, streamer.output().length);
        PithInflate.FfiError e = assertThrows(PithInflate.FfiError.class, streamer::finish);
        assertEquals(PithInflate.PITH_E_REJECTED, e.status);
    }

    @Test
    void streamingRejectsUnknownFraming() {
        assertThrows(IllegalArgumentException.class,
                () -> new PithInflate.StreamingInflater("bzip2"));
    }

    /** {@code array} equals the first {@code array.length} bytes of {@code full}. */
    private static boolean equalsPrefix(byte[] full, byte[] prefix) {
        if (prefix.length > full.length) {
            return false;
        }
        for (int i = 0; i < prefix.length; i++) {
            if (full[i] != prefix[i]) {
                return false;
            }
        }
        return true;
    }
}
