package com.gemma.functiongemma;

import java.util.List;
import java.util.Map;

public interface Tokenizer {
    TokenizerResult applyChatTemplate(
            List<Map<String, Object>> messages,
            List<Map<String, Object>> tools,
            boolean addGenerationPrompt);
    String decode(long[] tokenIds, boolean skipSpecialTokens);
    int getEosTokenId();

    boolean hasToken(String token);

    int convertTokenToId(String token);

    List<Integer> getEosTokenIds();

    public static class TokenizerResult {
        public long[] inputIds;
        public long[] attentionMask;

        public TokenizerResult(long[] inputIds, long[] attentionMask) {
            this.inputIds = inputIds;
            this.attentionMask = attentionMask;
        }
    }

}
