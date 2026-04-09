#ifndef GEMMA_INFERENCE_H
#define GEMMA_INFERENCE_H

#include <jni.h>
#include <string>
#include <vector>
#include <memory>
#include <chrono>

#ifdef __cplusplus
extern "C" {
#endif

class GemmaInferenceImpl;

// JNI 函数声明
JNIEXPORT jlong JNICALL Java_com_gemma_functiongemma_GemmaInference_initNative(
        JNIEnv* env, jobject obj, jstring modelPath, jint numThreads);

JNIEXPORT void JNICALL Java_com_gemma_functiongemma_GemmaInference_releaseNative(
        JNIEnv* env, jobject obj, jlong handle);

JNIEXPORT jintArray JNICALL Java_com_gemma_functiongemma_GemmaInference_greedyGenerateNative(
        JNIEnv* env, jobject obj, jlong handle,
        jlongArray inputIds, jintArray stopIds, jint maxNewTokens);

JNIEXPORT jboolean JNICALL Java_com_gemma_functiongemma_GemmaInference_precomputeSystemKVCacheNative(
        JNIEnv* env, jobject obj, jlong handle, jlongArray systemTokens);

JNIEXPORT jintArray JNICALL Java_com_gemma_functiongemma_GemmaInference_generateWithSystemCacheNative(
        JNIEnv* env, jobject obj, jlong handle,
        jlongArray fullInputIds, jintArray stopIds, jint maxNewTokens);

JNIEXPORT jboolean JNICALL Java_com_gemma_functiongemma_GemmaInference_exportSystemKVCacheNative(
        JNIEnv* env, jobject obj, jlong handle, jstring outputPath);

JNIEXPORT jboolean JNICALL Java_com_gemma_functiongemma_GemmaInference_importSystemKVCacheNative(
        JNIEnv* env, jobject obj, jlong handle, jstring inputPath);

JNIEXPORT jint JNICALL Java_com_gemma_functiongemma_GemmaInference_getSystemCacheSequenceLengthNative(
        JNIEnv* env, jobject obj, jlong handle);

#ifdef __cplusplus
}
#endif

#endif // GEMMA_INFERENCE_H
