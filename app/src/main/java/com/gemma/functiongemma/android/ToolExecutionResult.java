package com.gemma.functiongemma.android;

public final class ToolExecutionResult {
    private final boolean executed;
    private final String summary;

    public ToolExecutionResult(boolean executed, String summary) {
        this.executed = executed;
        this.summary = summary;
    }

    public boolean executed() {
        return executed;
    }

    public String summary() {
        return summary;
    }
}
