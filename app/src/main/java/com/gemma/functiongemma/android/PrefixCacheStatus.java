package com.gemma.functiongemma.android;

public final class PrefixCacheStatus {
    private final String toolsetId;
    private final boolean persistedCacheAvailable;
    private final boolean lastActivationCacheHit;
    private final int systemSequenceLength;
    private final long kvFileSizeBytes;
    private final long builtAtEpochMs;

    PrefixCacheStatus(
            String toolsetId,
            boolean persistedCacheAvailable,
            boolean lastActivationCacheHit,
            int systemSequenceLength,
            long kvFileSizeBytes,
            long builtAtEpochMs) {
        this.toolsetId = toolsetId;
        this.persistedCacheAvailable = persistedCacheAvailable;
        this.lastActivationCacheHit = lastActivationCacheHit;
        this.systemSequenceLength = systemSequenceLength;
        this.kvFileSizeBytes = kvFileSizeBytes;
        this.builtAtEpochMs = builtAtEpochMs;
    }

    public String toolsetId() {
        return toolsetId;
    }

    public boolean persistedCacheAvailable() {
        return persistedCacheAvailable;
    }

    public boolean lastActivationCacheHit() {
        return lastActivationCacheHit;
    }

    public int systemSequenceLength() {
        return systemSequenceLength;
    }

    public long kvFileSizeBytes() {
        return kvFileSizeBytes;
    }

    public long builtAtEpochMs() {
        return builtAtEpochMs;
    }
}
