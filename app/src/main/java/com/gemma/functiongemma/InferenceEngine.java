package com.gemma.functiongemma;

public interface InferenceEngine extends AutoCloseable {
    void init(String modelPath, int numThreads) throws Exception;

    long[] generate(long[] inputIds, long[] stopIds, int maxNewTokens) throws Exception;

    @Override
    void close();
}
