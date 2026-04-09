#include "gemma_inference.h"
#include <array>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <fstream>
#ifdef __ANDROID__
#include <android/log.h>
#endif

// ONNX Runtime 头文件
#include "onnxruntime/onnxruntime_c_api.h"
#include "onnxruntime/onnxruntime_cxx_api.h"
#include "onnxruntime/onnxruntime_float16.h"

#define LOG_TAG "GemmaInference"
#ifdef __ANDROID__
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#else
#define LOGI(...) do { std::fprintf(stdout, "[%s][INFO] ", LOG_TAG); std::fprintf(stdout, __VA_ARGS__); std::fprintf(stdout, "\n"); } while (0)
#define LOGE(...) do { std::fprintf(stderr, "[%s][ERROR] ", LOG_TAG); std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); } while (0)
#endif

// 定义常量
static const int BATCH_SIZE = 1;
static const int NUM_LAYERS = 18;    // 模型层数
static const int HEAD_DIM = 256;     // 模型Head_dim维度
static const int VOCAB_SIZE = 262144; // 词汇表大小
static const uint32_t KV_CACHE_MAGIC = 0x314b5647; // GVK1
static const uint32_t KV_CACHE_VERSION = 1;


class GemmaInferenceImpl {
private:
    using PastType = float;  // FP32 - 移动端性能最优
    // 根据PastType自动确定ONNX张量元素类型
    static constexpr ONNXTensorElementDataType PAST_ELEMENT_TYPE =
            // 仅支持FP16和FP32两种类型
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;

    // ONNX Runtime 相关
    std::unique_ptr<Ort::Env> ortEnv;
    std::unique_ptr<Ort::Session> ortSession;
    Ort::SessionOptions sessionOptions;
    Ort::MemoryInfo memoryInfo;

    // 输入输出名称
    std::vector<const char*> inputNames;
    std::vector<const char*> outputNames;

    // 高性能KV cache 存储结构 - 双缓冲区优化 (三维数组设计)
    struct KVCache {
        // 性能优先策略：预分配合理大小，减少运行时开销
        static constexpr size_t INITIAL_SEQ_LEN = 512;  // 预分配足够大的序列长度
        static constexpr size_t MAX_SEQ_LEN = 2048;     // 支持更长的序列
        static constexpr size_t NUM_BUFFERS = 2;        // 双缓冲区

        // 三维数组设计：[buffer_index][seq_pos * HEAD_DIM]
        // past_key_values[0][...] 作为 past_1, past_key_values[1][...] 作为 past_2
        std::vector<PastType> key_buffers;   // 自动选择FP16或FP32
        std::vector<PastType> value_buffers; // 自动选择FP16或FP32
        size_t current_seq_len;  // 当前有效序列长度
        size_t current_buffer;   // 当前使用的缓冲区索引 (0 或 1)

        KVCache() : current_seq_len(0), current_buffer(0) {
            // 预分配最大缓冲区大小的双缓冲区，实现真正的零拷贝
            size_t single_buffer_size = MAX_SEQ_LEN * HEAD_DIM;
            size_t total_buffer_size = NUM_BUFFERS * single_buffer_size;

            // 自动初始化为正确的类型和值
            PastType zero_value = static_cast<PastType>(0.0f);
            key_buffers.resize(total_buffer_size, zero_value);
            value_buffers.resize(total_buffer_size, zero_value);
        }

        size_t seq_len() const {
            return current_seq_len;
        }

        void clear() {
            current_seq_len = 0;
            current_buffer = 0;  // 重置为buffer 0作为输入
            // 缓冲区保持不变，无需清空
        }


        // 完全释放内存（用于长时间不使用的情况）
        void release_memory() {
            key_buffers.clear();
            value_buffers.clear();
            key_buffers.shrink_to_fit();
            value_buffers.shrink_to_fit();
            current_seq_len = 0;
            current_buffer = 0;
        }

        // 零拷贝扩展 - 仅检查边界，不进行内存分配
        bool extend(size_t new_seq_len) {
            if (new_seq_len > MAX_SEQ_LEN) {
                LOGE("Sequence length %zu exceeds maximum %zu", new_seq_len, MAX_SEQ_LEN);
                return false;
            }

            // 由于预分配了最大缓冲区，这里不需要实际的内存扩展
            // 注意：current_seq_len的设置由调用者负责
            return true;
        }

        // 单独的序列长度设置方法
        void set_seq_len(size_t new_seq_len) {
            current_seq_len = new_seq_len;
        }

    public:
        // 获取指定缓冲区的数据指针（三维数组访问：past[buffer_idx][...]）
        PastType* key_buffer_data(size_t buffer_idx) {
            return key_buffers.data() + buffer_idx * (key_buffers.size() / NUM_BUFFERS);
        }
        PastType* value_buffer_data(size_t buffer_idx) {
            return value_buffers.data() + buffer_idx * (value_buffers.size() / NUM_BUFFERS);
        }

        size_t key_size() const { return current_seq_len * HEAD_DIM; }
        size_t value_size() const { return current_seq_len * HEAD_DIM; }
    };

    std::vector<KVCache> kvCache;

    // 系统KV cache存储（用于分阶段prefill优化）
    std::vector<std::vector<PastType>> systemKVCache;  // [layer][seq_len * head_dim]
    size_t systemSeqLength;  // 系统提示词的序列长度
    bool systemCacheValid;   // 系统缓存是否有效

    // 统一的logits缓冲区，避免每次重新分配（prefill和decode阶段复用）
    std::vector<float> logitsBuffer;  // 大小: VOCAB_SIZE，存储logits（prefill/decode阶段复用）

    // 优化的attention_mask缓存，避免频繁重新分配
    std::vector<int64_t> attentionMaskBuffer;
    size_t currentAttentionMaskSize;

    // 当前输入缓冲区索引 (0 或 1，用于双缓冲区管理)
    size_t currentInputBufferIndex;

public:
    GemmaInferenceImpl()
            : memoryInfo(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
              currentAttentionMaskSize(0),
              currentInputBufferIndex(0) {
        initNames();
        initKVCache();
        // 初始化系统KV cache状态
        systemSeqLength = 0;
        systemCacheValid = false;

        // logitsBuffer 预分配词汇表大小（prefill和decode阶段复用）
        logitsBuffer.resize(VOCAB_SIZE);
        // attentionMaskBuffer 预分配合理大小
        attentionMaskBuffer.reserve(KVCache::INITIAL_SEQ_LEN * 2);  // 预分配足够空间
    }

    ~GemmaInferenceImpl() {
        release();
    }

    bool init(const std::string& modelPath, int numThreads) {
        try {
            // 创建 ONNX Runtime 环境
            ortEnv = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "GemmaInference");

            // 配置会话选项
            sessionOptions.SetIntraOpNumThreads(numThreads);
            sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

            // 创建会话
            ortSession = std::make_unique<Ort::Session>(*ortEnv, modelPath.c_str(), sessionOptions);

            return true;
        } catch (const Ort::Exception& e) {
            LOGE("Failed to initialize ONNX Runtime: %s", e.what());
            return false;
        } catch (const std::exception& e) {
            LOGE("Failed to initialize: %s", e.what());
            return false;
        }
    }


    // 预计算系统提示词的KV cache
    bool precomputeSystemKVCache(const std::vector<int64_t>& systemTokens) {
        try {
            LOGI("开始预计算系统KV cache，序列长度: %zu", systemTokens.size());

            // 清空之前的系统缓存
            systemKVCache.clear();
            systemSeqLength = 0;
            systemCacheValid = false;

            if (systemTokens.empty()) {
                LOGE("系统tokens为空");
                return false;
            }

            // 记录系统序列长度
            systemSeqLength = systemTokens.size();

            // 准备prefill输入
            std::vector<int64_t> positionIds(systemTokens.size());
            std::iota(positionIds.begin(), positionIds.end(), 0);

            // attention mask：全1向量，长度等于序列长度
            attentionMaskBuffer.resize(systemTokens.size());
            std::fill(attentionMaskBuffer.begin(), attentionMaskBuffer.end(), 1);

            // 准备输入输出
            std::vector<Ort::Value> inputs, outputs;
            preparePrefillInputsOutputs(systemTokens, positionIds, attentionMaskBuffer, inputs, outputs);

            // 执行推理
            ortSession->Run(Ort::RunOptions{nullptr},
                            inputNames.data(), inputs.data(), inputs.size(),
                            outputNames.data(), outputs.data(), outputs.size());

            // 缓存系统KV states
            systemKVCache.resize(NUM_LAYERS);
            for (int layer = 0; layer < NUM_LAYERS; ++layer) {
                // 从输出中获取KV cache
                const auto& keyOutput = outputs[1 + layer * 2];  // logits之后是key outputs
                const auto& valueOutput = outputs[2 + layer * 2]; // 然后是value outputs

                // 获取tensor数据
                const PastType* keyData = keyOutput.GetTensorData<PastType>();
                const PastType* valueData = valueOutput.GetTensorData<PastType>();

                // 计算数据大小：seq_len * head_dim
                size_t dataSize = systemSeqLength * HEAD_DIM;

                // 存储到系统缓存中
                systemKVCache[layer].resize(dataSize * 2); // key + value
                std::copy(keyData, keyData + dataSize, systemKVCache[layer].begin());
                std::copy(valueData, valueData + dataSize, systemKVCache[layer].begin() + dataSize);
            }

            systemCacheValid = true;
            LOGI("系统KV cache预计算完成，缓存了 %d 层，每层 %zu 个元素", NUM_LAYERS, systemSeqLength * HEAD_DIM * 2);

            return true;

        } catch (const std::exception& e) {
            LOGE("预计算系统KV cache失败: %s", e.what());
            systemCacheValid = false;
            return false;
        }
    }

    bool exportSystemKVCache(const std::string& outputPath) {
        if (!systemCacheValid || systemKVCache.size() != NUM_LAYERS || systemSeqLength == 0) {
            LOGE("No valid system KV cache available for export");
            return false;
        }

        std::ofstream stream(outputPath, std::ios::binary | std::ios::trunc);
        if (!stream.is_open()) {
            LOGE("Failed to open KV cache output path: %s", outputPath.c_str());
            return false;
        }

        const uint32_t numLayers = NUM_LAYERS;
        const uint32_t headDim = HEAD_DIM;
        const uint64_t seqLen = systemSeqLength;
        const uint32_t bytesPerElement = sizeof(PastType);

        stream.write(reinterpret_cast<const char*>(&KV_CACHE_MAGIC), sizeof(KV_CACHE_MAGIC));
        stream.write(reinterpret_cast<const char*>(&KV_CACHE_VERSION), sizeof(KV_CACHE_VERSION));
        stream.write(reinterpret_cast<const char*>(&numLayers), sizeof(numLayers));
        stream.write(reinterpret_cast<const char*>(&headDim), sizeof(headDim));
        stream.write(reinterpret_cast<const char*>(&bytesPerElement), sizeof(bytesPerElement));
        stream.write(reinterpret_cast<const char*>(&seqLen), sizeof(seqLen));

        for (const auto& layerBuffer : systemKVCache) {
            const size_t byteCount = layerBuffer.size() * sizeof(PastType);
            stream.write(reinterpret_cast<const char*>(layerBuffer.data()), static_cast<std::streamsize>(byteCount));
        }

        const bool success = stream.good();
        if (!success) {
            LOGE("Failed while writing system KV cache to disk");
        }
        return success;
    }

    bool importSystemKVCache(const std::string& inputPath) {
        std::ifstream stream(inputPath, std::ios::binary);
        if (!stream.is_open()) {
            LOGE("Failed to open KV cache input path: %s", inputPath.c_str());
            return false;
        }

        uint32_t magic = 0;
        uint32_t version = 0;
        uint32_t numLayers = 0;
        uint32_t headDim = 0;
        uint32_t bytesPerElement = 0;
        uint64_t seqLen = 0;

        stream.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        stream.read(reinterpret_cast<char*>(&version), sizeof(version));
        stream.read(reinterpret_cast<char*>(&numLayers), sizeof(numLayers));
        stream.read(reinterpret_cast<char*>(&headDim), sizeof(headDim));
        stream.read(reinterpret_cast<char*>(&bytesPerElement), sizeof(bytesPerElement));
        stream.read(reinterpret_cast<char*>(&seqLen), sizeof(seqLen));

        if (!stream.good()) {
            LOGE("Failed to read KV cache header");
            return false;
        }
        if (magic != KV_CACHE_MAGIC || version != KV_CACHE_VERSION) {
            LOGE("KV cache header mismatch");
            return false;
        }
        if (numLayers != NUM_LAYERS || headDim != HEAD_DIM || bytesPerElement != sizeof(PastType) || seqLen == 0) {
            LOGE("KV cache shape mismatch");
            return false;
        }

        const size_t layerElementCount = static_cast<size_t>(seqLen) * HEAD_DIM * 2;
        std::vector<std::vector<PastType>> loadedCache(NUM_LAYERS);
        for (auto& layerBuffer : loadedCache) {
            layerBuffer.resize(layerElementCount);
            const size_t byteCount = layerBuffer.size() * sizeof(PastType);
            stream.read(reinterpret_cast<char*>(layerBuffer.data()), static_cast<std::streamsize>(byteCount));
            if (!stream.good()) {
                LOGE("Failed to read KV cache payload");
                return false;
            }
        }

        systemKVCache = std::move(loadedCache);
        systemSeqLength = static_cast<size_t>(seqLen);
        systemCacheValid = true;
        return true;
    }

    int getSystemCacheSequenceLength() const {
        if (!systemCacheValid) {
            return 0;
        }
        return static_cast<int>(systemSeqLength);
    }

    std::vector<int> greedyGenerate(const std::vector<int64_t>& inputIds,
                                    const std::vector<int>& stopIds,
                                    int maxNewTokens) {
        std::vector<int> generatedTokens;

        // 优化的attention_mask处理 - 预分配并初始化
        size_t initialSeqLen = inputIds.size();
        attentionMaskBuffer.resize(initialSeqLen);
        std::fill(attentionMaskBuffer.begin(), attentionMaskBuffer.end(), 1);
        currentAttentionMaskSize = initialSeqLen;

        try {
            // Phase 1: Prefill
            std::vector<int64_t> positionIds(inputIds.size());
            std::iota(positionIds.begin(), positionIds.end(), 0);

            std::vector<Ort::Value> inputs, preallocatedOutputs;
            preparePrefillInputsOutputs(inputIds, positionIds, attentionMaskBuffer, inputs, preallocatedOutputs);

            ortSession->Run(Ort::RunOptions{nullptr},
                            inputNames.data(),
                            inputs.data(),
                            inputs.size(),
                            outputNames.data(),
                            preallocatedOutputs.data(),
                            preallocatedOutputs.size());

            const float* lastLogits = getLastLogitsFromOutput(preallocatedOutputs[0]);
            int nextToken = argmax(lastLogits, VOCAB_SIZE);
            generatedTokens.push_back(nextToken);


            // 检查停止词
            if (contains(stopIds, nextToken)) {
                return generatedTokens;
            }

            // 保存 KV cache - prefill阶段写入输出缓冲区
            updateKVCache(preallocatedOutputs);

            // prefill后，设置下次输入缓冲区为输出缓冲区
            currentInputBufferIndex = (currentInputBufferIndex + 1) % 2;

            // ===== Phase 2: Decode =====
            for (int step = 1; step < maxNewTokens; ++step) {
                // currentSeqLen = prefill输入长度 + 已生成的token数量
                // step从1开始，所以已生成的token数量是step-1
                int currentSeqLen = static_cast<int>(initialSeqLen) + step - 1;
                std::vector<int64_t> nextInputIds = {static_cast<int64_t>(nextToken)};
                std::vector<int64_t> nextPositionIds = {static_cast<int64_t>(currentSeqLen)};

                attentionMaskBuffer.push_back(1);
                currentAttentionMaskSize++;

                std::vector<Ort::Value> inputs, preallocatedOutputs;
                prepareDecodeInputsOutputs(nextInputIds, nextPositionIds, attentionMaskBuffer, inputs, preallocatedOutputs);

                ortSession->Run(Ort::RunOptions{nullptr},
                                inputNames.data(),
                                inputs.data(),
                                inputs.size(),
                                outputNames.data(),
                                preallocatedOutputs.data(),
                                preallocatedOutputs.size());
                nextToken = argmax(logitsBuffer.data(), VOCAB_SIZE);
                generatedTokens.push_back(nextToken);


                // 检查停止词
                if (contains(stopIds, nextToken)) {
                    break;
                }

                // 更新 KV cache - 写入输出缓冲区
                updateKVCache(preallocatedOutputs);

                // 切换缓冲区：下次使用输出缓冲区作为输入
                currentInputBufferIndex = (currentInputBufferIndex + 1) % 2;
            }

        } catch (const Ort::Exception& e) {
            LOGE("ONNX Runtime error: %s", e.what());
        } catch (const std::exception& e) {
            LOGE("Error during generation: %s", e.what());
        }

        return generatedTokens;
    }

    // 使用缓存的系统KV进行prefill
    std::vector<int> generateWithSystemCache(const std::vector<int64_t>& inputTokens,
                                             const std::vector<int>& stopIds,
                                             int maxNewTokens) {
        std::vector<int> generatedTokens;

        try {
            if (!systemCacheValid || systemKVCache.empty()) {
                LOGE("系统KV cache无效，使用标准prefill");
                return greedyGenerate(inputTokens, stopIds, maxNewTokens);
            }
            if (inputTokens.size() < systemSeqLength) {
                LOGE("完整输入长度(%zu)小于系统前缀长度(%zu)", inputTokens.size(), systemSeqLength);
                return generatedTokens;
            }

            return generateWithSystemCacheOptimized(inputTokens, stopIds, maxNewTokens);

        } catch (const std::exception& e) {
            return generatedTokens;
        }
    }

    // 分阶段优化的prefill实现
    std::vector<int> generateWithSystemCacheOptimized(const std::vector<int64_t>& fullTokens,
                                                      const std::vector<int>& stopIds,
                                                      int maxNewTokens) {
        std::vector<int> generatedTokens;

        try {
            LOGI("开始分阶段优化prefill，完整tokens数量: %zu", fullTokens.size());

            // 提取用户输入部分（完整tokens减去系统部分）
            size_t userInputLen = fullTokens.size() - systemSeqLength;
            std::vector<int64_t> userTokens(fullTokens.begin() + systemSeqLength, fullTokens.end());

            LOGI("分阶段优化 - 总上下文长度: %zu (系统缓存: %zu + 用户输入: %zu)",
                 fullTokens.size(), systemSeqLength, userInputLen);

            // 准备用户输入的position ids（从系统长度开始）
            std::vector<int64_t> userPositionIds(userInputLen);
            std::iota(userPositionIds.begin(), userPositionIds.end(), systemSeqLength);

            // attention mask：覆盖完整上下文，全1向量
            attentionMaskBuffer.resize(fullTokens.size());
            std::fill(attentionMaskBuffer.begin(), attentionMaskBuffer.end(), 1);

            // Phase 1: 用户Prefill（使用系统缓存的KV）
            std::vector<Ort::Value> inputs, prefillOutputs;
            prepareUserPrefillWithCache(userTokens, userPositionIds, attentionMaskBuffer, inputs, prefillOutputs);

            ortSession->Run(Ort::RunOptions{nullptr},
                            inputNames.data(), inputs.data(), inputs.size(),
                            outputNames.data(), prefillOutputs.data(), prefillOutputs.size());

            const float* lastLogits = getLastLogitsFromOutput(prefillOutputs[0]);
            int nextToken = argmax(lastLogits, VOCAB_SIZE);
            LOGI("分阶段优化prefill首个生成token: %d", nextToken);

            generatedTokens.push_back(nextToken);

            // 检查停止词
            if (contains(stopIds, nextToken)) {
                return generatedTokens;
            }


            // 更新KV cache - prefill阶段写入输出缓冲区
            updateKVCache(prefillOutputs);

            // prefill后，设置下次输入缓冲区为输出缓冲区
            currentInputBufferIndex = (currentInputBufferIndex + 1) % 2;

            // Phase 2: Decode循环
            for (int step = 1; step < maxNewTokens; ++step) {
                // currentSeqLen = 完整prefill长度 + 已生成的token数量
                int currentSeqLen = static_cast<int>(fullTokens.size()) + step - 1;
                std::vector<int64_t> nextInputIds = {static_cast<int64_t>(nextToken)};
                std::vector<int64_t> nextPositionIds = {static_cast<int64_t>(currentSeqLen)};

                attentionMaskBuffer.push_back(1);
                currentAttentionMaskSize++;

                std::vector<Ort::Value> inputs, decodeOutputs;
                prepareDecodeInputsOutputs(nextInputIds, nextPositionIds, attentionMaskBuffer, inputs, decodeOutputs);

                ortSession->Run(Ort::RunOptions{nullptr},
                                inputNames.data(), inputs.data(), inputs.size(),
                                outputNames.data(), decodeOutputs.data(), decodeOutputs.size());

                nextToken = argmax(logitsBuffer.data(), VOCAB_SIZE);
                generatedTokens.push_back(nextToken);

                // 检查停止词
                if (contains(stopIds, nextToken)) {
                    break;
                }

                // 更新KV cache
                updateKVCache(decodeOutputs);

                // 切换缓冲区
                currentInputBufferIndex = (currentInputBufferIndex + 1) % 2;
            }

            LOGI("分阶段优化生成完成，共生成 %zu 个tokens", generatedTokens.size());
            return generatedTokens;

        } catch (const std::exception& e) {
            LOGE("分阶段优化prefill失败: %s", e.what());
            return generatedTokens;
        }
    }


    // 只释放KV cache内存（用于每次推理结束时）
    void releaseKVCache() {
        // 释放所有KV缓存内存
        for (auto& cache : kvCache) {
            cache.release_memory();
        }
        kvCache.clear();

        // 释放系统缓存
        systemKVCache.clear();
        systemSeqLength = 0;
        systemCacheValid = false;
    }

    void release() {
        ortSession.reset();
        ortEnv.reset();
        inputNames.clear();
        outputNames.clear();

        // 释放所有KV缓存内存
        releaseKVCache();

        attentionMaskBuffer.clear();
        currentAttentionMaskSize = 0;
    }

private:
    // 初始化输入输出名称
    void initNames() {
        // 输入名称
        inputNames.push_back("input_ids");
        inputNames.push_back("position_ids");

        // past_key_values
        for (int i = 0; i < NUM_LAYERS; ++i) {
            std::string keyName = "past_key_values." + std::to_string(i) + ".key";
            std::string valueName = "past_key_values." + std::to_string(i) + ".value";

            inputNames.push_back(strdup(keyName.c_str()));
            inputNames.push_back(strdup(valueName.c_str()));
        }

        inputNames.push_back("attention_mask");

        // 输出名称
        outputNames.push_back("logits");

        for (int i = 0; i < NUM_LAYERS; ++i) {
            std::string keyName = "present." + std::to_string(i) + ".key";
            std::string valueName = "present." + std::to_string(i) + ".value";

            outputNames.push_back(strdup(keyName.c_str()));
            outputNames.push_back(strdup(valueName.c_str()));
        }
    }

    // 初始化 KV cache
    void initKVCache() {
        kvCache.resize(NUM_LAYERS);  // 构造函数自动推断类型
    }



    // 准备 prefill 输入和预分配输出
    void preparePrefillInputsOutputs(const std::vector<int64_t>& inputIds,
                                     const std::vector<int64_t>& positionIds,
                                     const std::vector<int64_t>& attentionMask,
                                     std::vector<Ort::Value>& inputs,
                                     std::vector<Ort::Value>& outputs) {
        inputs.clear();
        outputs.clear();

        // 预分配容量: input_ids + position_ids + past_kv(36) + attention_mask = 39
        inputs.reserve(2 + 2 * NUM_LAYERS + 1);

        // 1. input_ids
        inputs.push_back(Ort::Value::CreateTensor<int64_t>(
                memoryInfo, const_cast<int64_t*>(inputIds.data()), inputIds.size(),
                std::vector<int64_t>{BATCH_SIZE, static_cast<int64_t>(inputIds.size())}.data(), 2));

        // 2. position_ids
        inputs.push_back(Ort::Value::CreateTensor<int64_t>(
                memoryInfo, const_cast<int64_t*>(positionIds.data()), positionIds.size(),
                std::vector<int64_t>{BATCH_SIZE, static_cast<int64_t>(positionIds.size())}.data(), 2));

        // 3. past_key_values (初始为空) - 所有layer使用相同的空shape
        auto emptyKvShape = std::vector<int64_t>{BATCH_SIZE, 1, 0, static_cast<int64_t>(HEAD_DIM)};
        for (int i = 0; i < NUM_LAYERS; ++i) {
            // Key (空tensor)
            inputs.push_back(Ort::Value::CreateTensor(memoryInfo, nullptr, 0, emptyKvShape.data(), emptyKvShape.size(),
                                                      PAST_ELEMENT_TYPE));

            // Value (空tensor)
            inputs.push_back(Ort::Value::CreateTensor(memoryInfo, nullptr, 0, emptyKvShape.data(), emptyKvShape.size(),
                                                      PAST_ELEMENT_TYPE));
        }

        // 4. attention_mask
        inputs.push_back(Ort::Value::CreateTensor<int64_t>(
                memoryInfo, const_cast<int64_t*>(attentionMask.data()), attentionMask.size(),
                std::vector<int64_t>{BATCH_SIZE, static_cast<int64_t>(attentionMask.size())}.data(), 2));

        // logits 输出由 ORT 自动分配，避免与不同导出模型形状冲突
        size_t seqLen = inputIds.size();  // 提前声明seqLen变量
        outputs.push_back(Ort::Value{nullptr});

        // present_key_values - 使用输出缓冲区
        size_t outputBufferIndex = (currentInputBufferIndex + 1) % KVCache::NUM_BUFFERS;

        // 在prefill阶段，所有layer的输出shape都相同，可以预计算
        size_t required_elements = seqLen * HEAD_DIM;
        auto kvShape = std::vector<int64_t>{BATCH_SIZE, 1, static_cast<int64_t>(seqLen), static_cast<int64_t>(HEAD_DIM)};

        for (int i = 0; i < NUM_LAYERS; ++i) {
            // 预绑定缓冲区：让ONNX直接写入我们的缓冲区（零拷贝）
            // 注意：虽然缓冲区很大，但我们只告诉ONNX实际需要的大小
            outputs.push_back(Ort::Value::CreateTensor(memoryInfo, kvCache[i].key_buffer_data(outputBufferIndex),
                                                       required_elements * sizeof(PastType),
                                                       kvShape.data(), kvShape.size(),
                                                       PAST_ELEMENT_TYPE));
            outputs.push_back(Ort::Value::CreateTensor(memoryInfo, kvCache[i].value_buffer_data(outputBufferIndex),
                                                       required_elements * sizeof(PastType),
                                                       kvShape.data(), kvShape.size(),
                                                       PAST_ELEMENT_TYPE));
        }
    }

    // 准备用户prefill输入（使用缓存的系统KV）
    void prepareUserPrefillWithCache(const std::vector<int64_t>& userTokens,
                                     const std::vector<int64_t>& userPositionIds,
                                     const std::vector<int64_t>& attentionMask,
                                     std::vector<Ort::Value>& inputs,
                                     std::vector<Ort::Value>& outputs) {
        inputs.clear();
        outputs.clear();

        // 预分配容量: input_ids + position_ids + past_kv(36) + attention_mask = 39
        inputs.reserve(2 + 2 * NUM_LAYERS + 1);

        // 1. input_ids - 只有用户tokens
        inputs.push_back(Ort::Value::CreateTensor<int64_t>(
                memoryInfo, const_cast<int64_t*>(userTokens.data()), userTokens.size(),
                std::vector<int64_t>{BATCH_SIZE, static_cast<int64_t>(userTokens.size())}.data(), 2));

        // 2. position_ids - 用户的position ids（从系统长度开始）
        inputs.push_back(Ort::Value::CreateTensor<int64_t>(
                memoryInfo, const_cast<int64_t*>(userPositionIds.data()), userPositionIds.size(),
                std::vector<int64_t>{BATCH_SIZE, static_cast<int64_t>(userPositionIds.size())}.data(), 2));

        // 3. past_key_values - 使用缓存的系统KV
        // 系统KV的形状: [batch_size, 1, system_seq_len, head_dim]
        auto systemKvShape = std::vector<int64_t>{BATCH_SIZE, 1, static_cast<int64_t>(systemSeqLength), static_cast<int64_t>(HEAD_DIM)};

        for (int i = 0; i < NUM_LAYERS; ++i) {
            size_t dataSize = systemSeqLength * HEAD_DIM;

            // Key - 使用系统缓存
            inputs.push_back(Ort::Value::CreateTensor<PastType>(
                    memoryInfo, systemKVCache[i].data(), dataSize,
                    systemKvShape.data(), systemKvShape.size()));

            // Value - 使用系统缓存（在key之后）
            inputs.push_back(Ort::Value::CreateTensor<PastType>(
                    memoryInfo, systemKVCache[i].data() + dataSize, dataSize,
                    systemKvShape.data(), systemKvShape.size()));
        }

        // 4. attention_mask - 覆盖完整上下文（系统 + 用户）
        inputs.push_back(Ort::Value::CreateTensor<int64_t>(
                memoryInfo, const_cast<int64_t*>(attentionMask.data()), attentionMask.size(),
                std::vector<int64_t>{BATCH_SIZE, static_cast<int64_t>(attentionMask.size())}.data(), 2));

        // logits 输出由 ORT 自动分配，避免与不同导出模型形状冲突
        outputs.push_back(Ort::Value{nullptr});

        // present_key_values - 完整的上下文KV (系统缓存 + 用户新计算)
        size_t outputBufferIndex = (currentInputBufferIndex + 1) % KVCache::NUM_BUFFERS;
        size_t userSeqLen = userTokens.size();
        size_t totalContextLen = systemSeqLength + userSeqLen;
        auto outputKvShape = std::vector<int64_t>{BATCH_SIZE, 1, static_cast<int64_t>(totalContextLen), static_cast<int64_t>(HEAD_DIM)};

        for (int i = 0; i < NUM_LAYERS; ++i) {
            size_t required_elements = totalContextLen * HEAD_DIM;

            outputs.push_back(Ort::Value::CreateTensor(memoryInfo, kvCache[i].key_buffer_data(outputBufferIndex),
                                                       required_elements * sizeof(PastType),
                                                       outputKvShape.data(), outputKvShape.size(), PAST_ELEMENT_TYPE));
            outputs.push_back(Ort::Value::CreateTensor(memoryInfo, kvCache[i].value_buffer_data(outputBufferIndex),
                                                       required_elements * sizeof(PastType),
                                                       outputKvShape.data(), outputKvShape.size(), PAST_ELEMENT_TYPE));
        }
    }

    // 准备 decode 输入和预分配输出 - 使用当前缓冲区状态
    void prepareDecodeInputsOutputs(const std::vector<int64_t>& inputIds,
                                    const std::vector<int64_t>& positionIds,
                                    const std::vector<int64_t>& attentionMask,
                                    std::vector<Ort::Value>& inputs,
                                    std::vector<Ort::Value>& outputs) {
        inputs.clear();
        outputs.clear();

        // 预分配容量: input_ids + position_ids + past_kv(36) + attention_mask = 39
        inputs.reserve(2 + 2 * NUM_LAYERS + 1);

        // 1. input_ids
        inputs.push_back(Ort::Value::CreateTensor<int64_t>(
                memoryInfo, const_cast<int64_t*>(inputIds.data()), inputIds.size(),
                std::vector<int64_t>{BATCH_SIZE, static_cast<int64_t>(inputIds.size())}.data(), 2));

        // 2. position_ids
        inputs.push_back(Ort::Value::CreateTensor<int64_t>(
                memoryInfo, const_cast<int64_t*>(positionIds.data()), positionIds.size(),
                std::vector<int64_t>{BATCH_SIZE, static_cast<int64_t>(positionIds.size())}.data(), 2));

        // 3. past_key_values (双缓冲区：使用当前输入缓冲区)
        size_t outputBufferIndex = (currentInputBufferIndex + 1) % KVCache::NUM_BUFFERS;
        // 在第一次decode时，所有layer的seqLen都相同，可以预计算
        size_t seqLen = kvCache[0].seq_len();  // 所有layer的seqLen都相同
        auto kvShape = std::vector<int64_t>{BATCH_SIZE, 1, static_cast<int64_t>(seqLen), static_cast<int64_t>(HEAD_DIM)};

        for (int i = 0; i < NUM_LAYERS; ++i) {
            // Key - 使用当前输入缓冲区
            inputs.push_back(Ort::Value::CreateTensor(memoryInfo, kvCache[i].key_buffer_data(currentInputBufferIndex),
                                                      kvCache[i].key_size() * sizeof(PastType),
                                                      kvShape.data(), kvShape.size(),
                                                      PAST_ELEMENT_TYPE));

            // Value - 使用当前输入缓冲区
            inputs.push_back(Ort::Value::CreateTensor(memoryInfo, kvCache[i].value_buffer_data(currentInputBufferIndex),
                                                      kvCache[i].value_size() * sizeof(PastType),
                                                      kvShape.data(), kvShape.size(),
                                                      PAST_ELEMENT_TYPE));
        }

        // 4. attention_mask
        inputs.push_back(Ort::Value::CreateTensor<int64_t>(
                memoryInfo, const_cast<int64_t*>(attentionMask.data()), attentionMask.size(),
                std::vector<int64_t>{BATCH_SIZE, static_cast<int64_t>(attentionMask.size())}.data(), 2));

        // 预分配输出缓冲区 - 使用预分配的decode logits缓冲区
        outputs.push_back(Ort::Value::CreateTensor<float>(
                memoryInfo, logitsBuffer.data(), logitsBuffer.size(),
                std::vector<int64_t>{BATCH_SIZE, 1, VOCAB_SIZE}.data(), 3));

        // present_key_values - 使用输出缓冲区
        // 在第一次decode时，所有layer的newSeqLen都相同，可以预计算
        size_t newSeqLen = kvCache[0].seq_len() + 1;  // 所有layer的seqLen都相同
        size_t required_elements = newSeqLen * HEAD_DIM;
        auto outputKvShape = std::vector<int64_t>{BATCH_SIZE, 1, static_cast<int64_t>(newSeqLen), static_cast<int64_t>(HEAD_DIM)};

        for (int i = 0; i < NUM_LAYERS; ++i) {
            // 预绑定缓冲区：让ONNX直接写入我们的缓冲区（零拷贝）
            // 注意：虽然缓冲区很大，但我们只告诉ONNX实际需要的大小
            outputs.push_back(Ort::Value::CreateTensor(memoryInfo, kvCache[i].key_buffer_data(outputBufferIndex),
                                                       required_elements * sizeof(PastType),
                                                       outputKvShape.data(), outputKvShape.size(),
                                                       PAST_ELEMENT_TYPE));
            outputs.push_back(Ort::Value::CreateTensor(memoryInfo, kvCache[i].value_buffer_data(outputBufferIndex),
                                                       required_elements * sizeof(PastType),
                                                       outputKvShape.data(), outputKvShape.size(),
                                                       PAST_ELEMENT_TYPE));
        }
    }


    // 更新 KV cache
    void updateKVCache(const std::vector<Ort::Value>& outputs) {
        // 跳过 logits (索引0)
        for (int i = 0; i < NUM_LAYERS; ++i) {

            int keyIdx = 1 + i * 2;

            try {
                // 从输出tensor获取新的序列长度
                const auto& keyTensor = outputs[keyIdx];
                auto keyShape = keyTensor.GetTensorTypeAndShapeInfo().GetShape();
                size_t new_seq_len = static_cast<size_t>(keyShape[2]);

                // 扩展缓存长度（仅检查边界，无内存分配）
                if (!kvCache[i].extend(new_seq_len)) {
                    LOGE("Failed to extend KV cache for layer %d, seq_len=%zu", i, new_seq_len);
                    continue;
                }

                // 真正的零拷贝：ONNX已经直接写入了我们的缓冲区
                // 只需更新序列长度状态
                kvCache[i].set_seq_len(new_seq_len);

            } catch (const std::exception& e) {
                LOGE("Error updating KV cache for layer %d: %s", i, e.what());
            }
        }

    }

    // 兼容不同导出模型logits形状，统一返回最后一个位置的logits指针
    const float* getLastLogitsFromOutput(const Ort::Value& logitsValue) {
        auto shape = logitsValue.GetTensorTypeAndShapeInfo().GetShape();
        const float* logitsData = logitsValue.GetTensorData<float>();
        if (shape.size() == 3) {
            int64_t seqLen = shape[1];
            if (seqLen <= 0) {
                throw std::runtime_error("Invalid logits seq_len");
            }
            return logitsData + (seqLen - 1) * VOCAB_SIZE;
        }
        if (shape.size() == 2) {
            return logitsData;
        }
        throw std::runtime_error("Unexpected logits rank");
    }

    // 辅助函数：从数组中找到最大值索引（需要检查NaN）
    int argmax(const float* data, int size) {
        int maxIdx = 0;
        float maxVal = data[0];
        if (std::isnan(maxVal)) {
            maxVal = -INFINITY;
        }

        for (int i = 1; i < size; ++i) {
            float currentVal = data[i];

            // 跳过NaN值
            if (std::isnan(currentVal)) {
                continue;
            }

            if (currentVal > maxVal) {
                maxVal = currentVal;
                maxIdx = i;
            }
        }

        return maxIdx;
    }


    // 辅助函数：检查是否包含停止 token
    bool contains(const std::vector<int>& vec, int value) {
        return std::find(vec.begin(), vec.end(), value) != vec.end();
    }
};

// JNI 实现
extern "C" {

JNIEXPORT jlong JNICALL Java_com_gemma_functiongemma_GemmaInference_initNative(
        JNIEnv* env, jobject obj, jstring modelPath, jint numThreads) {

    const char* modelPathStr = env->GetStringUTFChars(modelPath, nullptr);
    if (!modelPathStr) {
        LOGE("Failed to get model path string");
        return 0;
    }

    auto* impl = new GemmaInferenceImpl();
    bool success = impl->init(modelPathStr, numThreads);

    env->ReleaseStringUTFChars(modelPath, modelPathStr);

    if (!success) {
        delete impl;
        return 0;
    }

    return reinterpret_cast<jlong>(impl);
}

JNIEXPORT void JNICALL Java_com_gemma_functiongemma_GemmaInference_releaseNative(
        JNIEnv* env, jobject obj, jlong handle) {

    if (handle) {
        auto* impl = reinterpret_cast<GemmaInferenceImpl*>(handle);
        delete impl;
    }
}

JNIEXPORT jintArray JNICALL Java_com_gemma_functiongemma_GemmaInference_greedyGenerateNative(
        JNIEnv* env, jobject obj, jlong handle,
        jlongArray inputIds, jintArray stopIds, jint maxNewTokens) {

    if (!handle) {
        LOGE("Invalid handle");
        return nullptr;
    }

    auto* impl = reinterpret_cast<GemmaInferenceImpl*>(handle);

    // 获取输入数据
    jlong* inputIdsArr = env->GetLongArrayElements(inputIds, nullptr);
    jint* stopIdsArr = stopIds ? env->GetIntArrayElements(stopIds, nullptr) : nullptr;

    jsize inputIdsLen = env->GetArrayLength(inputIds);
    jsize stopIdsLen = stopIds ? env->GetArrayLength(stopIds) : 0;

    // 转换为 C++ vector
    std::vector<int64_t> inputIdsVec(inputIdsArr, inputIdsArr + inputIdsLen);

    std::vector<int> stopIdsVec;
    if (stopIdsArr) {
        stopIdsVec.assign(stopIdsArr, stopIdsArr + stopIdsLen);
    }


    // 运行生成
    std::vector<int> generatedTokens = impl->greedyGenerate(
            inputIdsVec, stopIdsVec, maxNewTokens);

    // 创建返回数组
    jintArray result = env->NewIntArray(static_cast<jsize>(generatedTokens.size()));
    if (result && !generatedTokens.empty()) {
        env->SetIntArrayRegion(result, 0, static_cast<jsize>(generatedTokens.size()),
                               reinterpret_cast<const jint*>(generatedTokens.data()));
    }

    // 释放资源
    env->ReleaseLongArrayElements(inputIds, inputIdsArr, 0);
    if (stopIdsArr) {
        env->ReleaseIntArrayElements(stopIds, stopIdsArr, 0);
    }

    return result;
}

JNIEXPORT jboolean JNICALL Java_com_gemma_functiongemma_GemmaInference_precomputeSystemKVCacheNative(
        JNIEnv* env, jobject obj, jlong handle, jlongArray systemTokens) {

    if (!handle) {
        LOGE("Invalid handle");
        return JNI_FALSE;
    }

    auto* impl = reinterpret_cast<GemmaInferenceImpl*>(handle);

    // 获取输入数据
    jlong* tokensArr = env->GetLongArrayElements(systemTokens, nullptr);
    jsize tokensLen = env->GetArrayLength(systemTokens);

    // 转换为 C++ vector
    std::vector<int64_t> systemTokensVec(tokensArr, tokensArr + tokensLen);

    // 执行预计算
    bool success = impl->precomputeSystemKVCache(systemTokensVec);

    // 释放资源
    env->ReleaseLongArrayElements(systemTokens, tokensArr, 0);

    return success ? JNI_TRUE : JNI_FALSE;
}


JNIEXPORT jintArray JNICALL Java_com_gemma_functiongemma_GemmaInference_generateWithSystemCacheNative(
        JNIEnv* env, jobject obj, jlong handle,
        jlongArray fullInputIds, jintArray stopIds, jint maxNewTokens) {

    if (!handle) {
        LOGE("Invalid handle");
        return nullptr;
    }

    auto* impl = reinterpret_cast<GemmaInferenceImpl*>(handle);

    // 获取输入数据
    jlong* userIdsArr = env->GetLongArrayElements(fullInputIds, nullptr);
    jint* stopIdsArr = stopIds ? env->GetIntArrayElements(stopIds, nullptr) : nullptr;

    jsize userIdsLen = env->GetArrayLength(fullInputIds);
    jsize stopIdsLen = stopIds ? env->GetArrayLength(stopIds) : 0;

    // 转换为 C++ vector
    std::vector<int64_t> userIdsVec(userIdsArr, userIdsArr + userIdsLen);
    std::vector<int> stopIdsVec;
    if (stopIdsArr) {
        stopIdsVec.assign(stopIdsArr, stopIdsArr + stopIdsLen);
    }

    // 执行生成
    std::vector<int> generatedTokens = impl->generateWithSystemCache(
            userIdsVec, stopIdsVec, maxNewTokens);

    // 创建返回数组
    jintArray result = env->NewIntArray(static_cast<jsize>(generatedTokens.size()));
    if (result && !generatedTokens.empty()) {
        env->SetIntArrayRegion(result, 0, static_cast<jsize>(generatedTokens.size()),
                               reinterpret_cast<const jint*>(generatedTokens.data()));
    }

    // 释放资源
    env->ReleaseLongArrayElements(fullInputIds, userIdsArr, 0);
    if (stopIdsArr) {
        env->ReleaseIntArrayElements(stopIds, stopIdsArr, 0);
    }

    return result;
}

JNIEXPORT jboolean JNICALL Java_com_gemma_functiongemma_GemmaInference_exportSystemKVCacheNative(
        JNIEnv* env, jobject obj, jlong handle, jstring outputPath) {
    if (!handle) {
        LOGE("Invalid handle");
        return JNI_FALSE;
    }

    const char* path = env->GetStringUTFChars(outputPath, nullptr);
    if (!path) {
        LOGE("Failed to read export path");
        return JNI_FALSE;
    }

    auto* impl = reinterpret_cast<GemmaInferenceImpl*>(handle);
    const bool success = impl->exportSystemKVCache(path);
    env->ReleaseStringUTFChars(outputPath, path);
    return success ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL Java_com_gemma_functiongemma_GemmaInference_importSystemKVCacheNative(
        JNIEnv* env, jobject obj, jlong handle, jstring inputPath) {
    if (!handle) {
        LOGE("Invalid handle");
        return JNI_FALSE;
    }

    const char* path = env->GetStringUTFChars(inputPath, nullptr);
    if (!path) {
        LOGE("Failed to read import path");
        return JNI_FALSE;
    }

    auto* impl = reinterpret_cast<GemmaInferenceImpl*>(handle);
    const bool success = impl->importSystemKVCache(path);
    env->ReleaseStringUTFChars(inputPath, path);
    return success ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jint JNICALL Java_com_gemma_functiongemma_GemmaInference_getSystemCacheSequenceLengthNative(
        JNIEnv* env, jobject obj, jlong handle) {
    if (!handle) {
        LOGE("Invalid handle");
        return 0;
    }

    auto* impl = reinterpret_cast<GemmaInferenceImpl*>(handle);
    return static_cast<jint>(impl->getSystemCacheSequenceLength());
}

} // extern "C"
