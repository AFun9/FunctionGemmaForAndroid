package com.gemma.functiongemma;

import org.junit.Test;

import java.io.ByteArrayInputStream;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class TokenizerJsonParserTest {
    private final TokenizerJsonParser parser = new TokenizerJsonParser();

    @Test
    public void parsesVocabAndMergesFromInputStream() throws Exception {
        String tokenizerJson = "{"
                + "\"model\":{"
                + "\"vocab\":{\"<bos>\":2,\"foo\":7},"
                + "\"merges\":[[\"f\",\"oo\"],\"fo o\"]"
                + "}"
                + "}";

        Map<String, Integer> vocab = parser.parseVocab(new ByteArrayInputStream(tokenizerJson.getBytes(StandardCharsets.UTF_8)));
        List<String> merges = parser.parseMerges(new ByteArrayInputStream(tokenizerJson.getBytes(StandardCharsets.UTF_8)));

        assertEquals(Integer.valueOf(2), vocab.get("<bos>"));
        assertEquals(Integer.valueOf(7), vocab.get("foo"));
        assertEquals(2, merges.size());
        assertEquals("f oo", merges.get(0));
        assertEquals("fo o", merges.get(1));
    }

    @Test
    public void parsesConfigFromInputStream() throws Exception {
        String configJson = "{"
                + "\"eos_token_id\":[1,106],"
                + "\"bos_token\":{\"content\":\"<bos>\"}"
                + "}";

        Map<String, Object> config = parser.parseConfig(new ByteArrayInputStream(configJson.getBytes(StandardCharsets.UTF_8)));

        assertTrue(config.containsKey("eos_token_id"));
        assertTrue(config.get("eos_token_id") instanceof List);
        assertTrue(config.get("bos_token") instanceof Map);
    }
}
