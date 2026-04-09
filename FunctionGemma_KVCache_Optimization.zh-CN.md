# FunctionGemma KV Cache 优化说明

English version: [FunctionGemma_KVCache_Optimization.md](FunctionGemma_KVCache_Optimization.md)

## 概述

本文档描述 FunctionGemmaForAndroid 所采用的 KV Cache 优化方案。其核心目标是预先计算对话中稳定前缀部分的 key-value cache，尤其是 developer prompt 与工具定义，从而减少重复计算。

## 背景

在传统对话推理流程中，每次生成都需要重新处理完整上下文，包括：

- developer 或 system prompt
- 工具定义
- 用户输入
- 历史生成内容

其中 developer prompt 和工具定义在多轮对话中通常长期稳定。如果每次请求都重复计算这些部分，在移动端上的代价会较高。

## 核心策略

该优化将推理拆分为两个阶段：

1. **前缀预计算阶段**  
   预先计算稳定前缀的 KV cache。
2. **用户输入推理阶段**  
   复用前缀缓存，仅处理与当前用户请求相关的后缀部分。

## 前缀组成

被缓存的前缀通常包括：

- developer prompt
- 工具声明
- 对话模板中所需的结构性 token

其中不会包含当前用户输入。

## 两阶段运行流程

### 阶段一：前缀预计算

运行时会先构造仅包含前缀内容的消息列表，应用 chat template，完成 tokenization，然后送入 native inference，为所有 transformer 层生成对应的 KV cache。

生成后的缓存会被持久化并建立索引，以便后续对同一 toolset 的再次激活时直接复用。

### 阶段二：用户输入推理

当用户提交请求后，运行时会构建完整 prompt，并识别出超出缓存前缀之外的用户输入后缀。推理会从已预计算的状态继续，而不是从空上下文重新开始。

这样可以显著减少 attention 计算量以及重复 prompt 处理开销。

## 优化收益

这一设计主要带来三方面收益：

- 降低稳定 prompt 内容的重复计算成本
- 减少 prefill 阶段不必要的内存压力
- 提升工具调用场景下的重复交互响应速度

## 被持久化的内容

运行时会持久化以下内容：

- 前缀 KV cache 文件
- 描述 toolset 指纹的元数据
- 序列长度与缓存可用性信息

这些信息用于判断现有缓存是否能够被安全复用。

## 缓存失效条件

只要有效前缀发生变化，prefix cache 就必须重建。通常包括：

- 工具定义发生变化
- developer prompt 发生变化
- tokenizer 发生变化
- 模型文件发生变化

## 实现说明

相关实现可参考：

- [`FunctionGemmaEngine.java`](app/src/main/java/com/gemma/functiongemma/android/FunctionGemmaEngine.java)
- [`PrefixCacheIndexStore.java`](app/src/main/java/com/gemma/functiongemma/android/PrefixCacheIndexStore.java)
- [`ToolsetFingerprint.java`](app/src/main/java/com/gemma/functiongemma/android/ToolsetFingerprint.java)
- [`main/cpp`](main/cpp) 下的 native inference 代码

## 适用范围

该优化在以下场景中特别有价值：

- 工具声明体积较大
- prompt 前缀在多次请求中保持稳定
- 设备对内存较为敏感
- 交互模式会重复使用同一套 toolset
