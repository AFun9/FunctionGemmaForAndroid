package com.gemma.functiongemma.android;

import com.google.gson.Gson;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.LinkedHashMap;
import java.util.Map;

final class ToolsetFingerprint {
    private static final int CACHE_FORMAT_VERSION = 1;
    private static final Gson GSON = new Gson();

    private ToolsetFingerprint() {
    }

    static String create(ToolsetDefinition toolset, String modelSignature, String tokenizerSignature) {
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("cacheFormatVersion", CACHE_FORMAT_VERSION);
        payload.put("modelSignature", modelSignature);
        payload.put("tokenizerSignature", tokenizerSignature);
        payload.put("toolset", toolset.canonicalFingerprintPayload());
        return sha256(GSON.toJson(payload));
    }

    private static String sha256(String input) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] bytes = digest.digest(input.getBytes(StandardCharsets.UTF_8));
            StringBuilder builder = new StringBuilder(bytes.length * 2);
            for (byte value : bytes) {
                builder.append(Character.forDigit((value >>> 4) & 0xF, 16));
                builder.append(Character.forDigit(value & 0xF, 16));
            }
            return builder.toString();
        } catch (NoSuchAlgorithmException error) {
            throw new IllegalStateException("SHA-256 unavailable", error);
        }
    }
}
