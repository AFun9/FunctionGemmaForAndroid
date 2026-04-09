package com.gemma.functiongemma.android;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

public final class ToolCall {
    private final String name;
    private final Map<String, Object> arguments;

    public ToolCall(String name, Map<String, Object> arguments) {
        if (name == null || name.trim().isEmpty()) {
            throw new IllegalArgumentException("Tool call name must not be empty");
        }
        this.name = name.trim();
        Map<String, Object> copy = new LinkedHashMap<>();
        if (arguments != null) {
            copy.putAll(arguments);
        }
        this.arguments = Collections.unmodifiableMap(copy);
    }

    public String name() {
        return name;
    }

    public Map<String, Object> arguments() {
        return arguments;
    }
}
