package com.gemma.functiongemma;

public class CppInferenceEngine implements InferenceEngine {
    private final GemmaInference inference = new GemmaInference();

    @Override
    public void init(String modelPath, int numThreads) {
        inference.init(modelPath, numThreads);
    }

    @Override
    public long[] generate(long[] inputIds, long[] stopIds, int maxNewTokens) {
        return toLongArray(inference.greedyGenerate(inputIds, toIntArray(stopIds), maxNewTokens));
    }

    public boolean precomputeSystemPrefix(long[] systemTokens) {
        return inference.precomputeSystemKVCache(systemTokens);
    }

    public long[] generateWithPrefixCache(long[] fullInputIds, long[] stopIds, int maxNewTokens) {
        return toLongArray(inference.generateWithSystemCache(fullInputIds, toIntArray(stopIds), maxNewTokens));
    }

    public boolean exportSystemPrefixCache(String outputPath) {
        return inference.exportSystemKVCache(outputPath);
    }

    public boolean importSystemPrefixCache(String inputPath) {
        return inference.importSystemKVCache(inputPath);
    }

    public int getSystemPrefixSequenceLength() {
        return inference.getSystemCacheSequenceLength();
    }

    private static int[] toIntArray(long[] stopIds) {
        int[] intStopIds = new int[stopIds.length];
        for (int i = 0; i < stopIds.length; i++) {
            intStopIds[i] = (int) stopIds[i];
        }
        return intStopIds;
    }

    private static long[] toLongArray(int[] output) {
        long[] result = new long[output.length];
        for (int i = 0; i < output.length; i++) {
            result[i] = output[i];
        }
        return result;
    }

    @Override
    public void close() {
        inference.release();
    }
}
