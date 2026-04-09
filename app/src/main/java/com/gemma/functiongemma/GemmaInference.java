package com.gemma.functiongemma;

public class GemmaInference {
    private long nativeHandle;

    static {
        try {
            System.loadLibrary("onnxruntime");
        } catch (UnsatisfiedLinkError ignored) {
            // Some packaging setups let the linker resolve libonnxruntime.so automatically.
        }
        try {
            System.loadLibrary("gemma_inference");
        } catch (UnsatisfiedLinkError error) {
            throw new UnsatisfiedLinkError("Failed to load native libraries for FunctionGemma: " + error.getMessage());
        }
    }

    // 本地方法声明
    private native long initNative(String modelPath, int numThreads);
    private native void releaseNative(long handle);
    private native int[] greedyGenerateNative(long handle,
                                              long[] inputIds, int[] stopIds, int maxNewTokens);
    private native boolean precomputeSystemKVCacheNative(long handle, long[] systemTokens);
    private native int[] generateWithSystemCacheNative(long handle,
                                                       long[] fullInputIds, int[] stopIds, int maxNewTokens);
    private native boolean exportSystemKVCacheNative(long handle, String outputPath);
    private native boolean importSystemKVCacheNative(long handle, String inputPath);
    private native int getSystemCacheSequenceLengthNative(long handle);

    public void init(String modelPath, int numThreads) {
        nativeHandle = initNative(modelPath, numThreads);
        if (nativeHandle == 0) {
            throw new RuntimeException("Failed to initialize native inference engine");
        }
    }

    public int[] greedyGenerate(long[] inputIds, int[] stopIds, int maxNewTokens) {
        if (nativeHandle == 0) {
            throw new IllegalStateException("Not initialized");
        }
        return greedyGenerateNative(nativeHandle, inputIds, stopIds, maxNewTokens);
    }

    public boolean precomputeSystemKVCache(long[] systemTokens) {
        if (nativeHandle == 0) {
            throw new IllegalStateException("Not initialized");
        }
        return precomputeSystemKVCacheNative(nativeHandle, systemTokens);
    }

    public int[] generateWithSystemCache(long[] fullInputIds, int[] stopIds, int maxNewTokens) {
        if (nativeHandle == 0) {
            throw new IllegalStateException("Not initialized");
        }
        return generateWithSystemCacheNative(nativeHandle, fullInputIds, stopIds, maxNewTokens);
    }

    public boolean exportSystemKVCache(String outputPath) {
        if (nativeHandle == 0) {
            throw new IllegalStateException("Not initialized");
        }
        return exportSystemKVCacheNative(nativeHandle, outputPath);
    }

    public boolean importSystemKVCache(String inputPath) {
        if (nativeHandle == 0) {
            throw new IllegalStateException("Not initialized");
        }
        return importSystemKVCacheNative(nativeHandle, inputPath);
    }

    public int getSystemCacheSequenceLength() {
        if (nativeHandle == 0) {
            throw new IllegalStateException("Not initialized");
        }
        return getSystemCacheSequenceLengthNative(nativeHandle);
    }

    public void release() {
        if (nativeHandle != 0) {
            releaseNative(nativeHandle);
            nativeHandle = 0;
        }
    }

    @Override
    protected void finalize() throws Throwable {
        try {
            release();
        } finally {
            super.finalize();
        }
    }
}
