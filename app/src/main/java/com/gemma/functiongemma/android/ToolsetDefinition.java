package com.gemma.functiongemma.android;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

public final class ToolsetDefinition {
    private final String id;
    private final String displayName;
    private final String systemPrompt;
    private final List<Map<String, Object>> tools;

    public ToolsetDefinition(String id, String displayName, String systemPrompt, List<Map<String, Object>> tools) {
        if (id == null || id.trim().isEmpty()) {
            throw new IllegalArgumentException("toolset id must not be empty");
        }
        this.id = id.trim();
        this.displayName = displayName == null || displayName.trim().isEmpty() ? this.id : displayName.trim();
        this.systemPrompt = systemPrompt == null ? "" : systemPrompt;
        this.tools = deepCopyToolList(tools);
    }

    public String id() {
        return id;
    }

    public String displayName() {
        return displayName;
    }

    public String systemPrompt() {
        return systemPrompt;
    }

    public List<Map<String, Object>> tools() {
        return deepCopyToolList(tools);
    }

    Object canonicalFingerprintPayload() {
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("systemPrompt", systemPrompt);
        payload.put("tools", canonicalizeValue(tools));
        return payload;
    }

    private static List<Map<String, Object>> deepCopyToolList(List<Map<String, Object>> tools) {
        List<Map<String, Object>> copy = new ArrayList<>();
        if (tools == null) {
            return copy;
        }
        for (Map<String, Object> tool : tools) {
            @SuppressWarnings("unchecked")
            Map<String, Object> canonicalMap = (Map<String, Object>) canonicalizeValue(tool);
            copy.add(canonicalMap);
        }
        return copy;
    }

    static Object canonicalizeValue(Object value) {
        if (value instanceof Map<?, ?> mapValue) {
            Map<String, Object> sorted = new TreeMap<>();
            for (Map.Entry<?, ?> entry : mapValue.entrySet()) {
                sorted.put(String.valueOf(entry.getKey()), canonicalizeValue(entry.getValue()));
            }
            Map<String, Object> linked = new LinkedHashMap<>();
            for (Map.Entry<String, Object> entry : sorted.entrySet()) {
                linked.put(entry.getKey(), entry.getValue());
            }
            return linked;
        }
        if (value instanceof List<?> listValue) {
            List<Object> copy = new ArrayList<>(listValue.size());
            for (Object item : listValue) {
                copy.add(canonicalizeValue(item));
            }
            return copy;
        }
        return value;
    }
}
