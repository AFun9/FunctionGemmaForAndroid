package com.gemma.functiongemma.android;

import androidx.annotation.Nullable;

final class ToolCallResolution {
    private final @Nullable ToolCall toolCall;
    private final String source;

    ToolCallResolution(@Nullable ToolCall toolCall, String source) {
        this.toolCall = toolCall;
        this.source = source;
    }

    @Nullable
    ToolCall toolCall() {
        return toolCall;
    }

    String source() {
        return source;
    }
}
