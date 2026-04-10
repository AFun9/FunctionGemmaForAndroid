# FunctionGemmaForAndroid

English version: [README.md](README.md)

## 项目概述

FunctionGemmaForAndroid 是一个以 Android 为核心的本地推理项目，基于 Gemma、ONNX Runtime 和原生 C++ 推理链路构建。该仓库聚焦于离线工具调用、轻量级移动端交互，以及适合端侧助手场景的运行时架构。

## 仓库结构

- [`app`](app)：Android 应用模块，包含界面、工具调用解析、校验与执行逻辑
- [`main/cpp`](main/cpp)：原生推理实现以及 CMake 源码
- [`model`](model)：本地模型资源目录，构建时会作为 Android assets 打包
- [`python`](python)：将官方 Gemma 模型转换为本项目所需 ONNX 结构的工具脚本
- [`FunctionGemma_KVCache_Optimization.md`](FunctionGemma_KVCache_Optimization.md)：描述运行时 KV Cache 优化方案的技术说明文档

## 本地开发环境

### 必需工具链

- Android Studio Ladybug 或更高版本
- JDK 17
- Android SDK 34
- Android NDK `27.0.12077973`
- CMake `3.22.1`

### 关键运行时依赖

- Android Gradle Plugin `8.5.2`
- ONNX Runtime Android `1.23.2`
- `minSdk 26`
- `targetSdk 34`

## SDK 配置

仓库中不包含本机专用的 Android SDK 路径配置。首次本地构建前，需要在项目根目录创建 `local.properties`，并写入 SDK 路径：

```properties
sdk.dir=/home/yourname/Android/Sdk
```

也可以通过 `ANDROID_HOME` 或 `ANDROID_SDK_ROOT` 提供路径，但对于 Android Studio 工程来说，`local.properties` 通常是最直接的方式。

## 模型资源

Android 应用运行时要求 [`model`](model) 目录中已经放好转换后的 ONNX 模型资源。通常至少需要：

- `tokenizer.json`
- `tokenizer_config.json`
- `onnx/model.onnx`
- `onnx/model.onnx_data`（如果导出时采用外部张量数据）

如果你当前只有官方原始模型，而不是 ONNX 结构，需要先使用 [`python`](python) 目录中的脚本完成转换。

## 构建与运行

在项目根目录执行：

```bash
./gradlew :app:assembleDebug
./gradlew :app:installDebug
```

也可以直接在 Android Studio 中打开工程并运行 `app` 模块。

## 运行时说明

- 首次启动时会先从 APK 中提取模型资源
- 初始化过程中会加载 tokenizer
- 首次创建 native model session 时可能会有明显耗时
- prefix KV cache 的首次构建会增加启动时间和内存占用

## 精度支持说明

本项目需要区分 **导出精度支持** 和 **运行时精度支持** 两个概念。

- 当前仓库中的模型转换链路只保留两种导出精度：`fp32` 和 `fp16`
- 但这并不意味着 Android 运行时已经对这些精度全部实现了端到端支持

在当前仓库中，Android 应用采用的是“Java 控制层 + JNI/C++ 数据面”的结构。这个边界对低精度模型尤其重要：

- Java 负责界面、配置和调度
- Native 层负责 ONNX Runtime session、张量创建和 KV cache 管理

### 为什么低精度推理不适合走 Java 主路径

Java 对现代 ONNX 低精度张量类型并没有自然、直接的数组表示：

- `fp16` 往往需要用 `short[]` 这类“位模式容器”来存储，而不是直接的半精度浮点数组
- `int4` 需要手动打包到 `byte[]`
- KV cache 的输入输出张量还必须严格匹配导出模型所要求的 I/O dtype

因此，本项目并不将 Java-only 推理路径作为低精度运行时方案。更合适的架构是：

- Java 负责应用逻辑
- JNI/C++ 负责低精度张量和运行时缓存

### 为什么 FP16 仍然可能占用较高内存

低精度权重并不等于整个推理链路都会自动变成低精度。实际运行时中，内存占用仍可能较高，甚至在某些情况下高于预期，常见原因包括：

- KV cache 仍使用比权重更宽的 dtype
- 导出图中的部分算子仍然在 `float32` 下运行
- ONNX Runtime 在执行过程中会分配额外的临时 buffer 与 cast workspace

因此，即使模型权重是 `fp16`，如果整个运行时链路没有做到精度一致，进程 RSS 仍然可能偏高。

## 主要代码入口

- [`MainActivity.java`](app/src/main/java/com/gemma/functiongemma/android/MainActivity.java)：Android 界面与交互入口
- [`FunctionGemmaEngine.java`](app/src/main/java/com/gemma/functiongemma/android/FunctionGemmaEngine.java)：推理协调与 toolset 激活
- [`ToolCallParser.java`](app/src/main/java/com/gemma/functiongemma/android/ToolCallParser.java)：模型输出解析
- [`ToolCallFallbackResolver.java`](app/src/main/java/com/gemma/functiongemma/android/ToolCallFallbackResolver.java)：对不完整或错误工具调用的解析后纠偏
- [`ToolExecutor.java`](app/src/main/java/com/gemma/functiongemma/android/ToolExecutor.java)：工具执行层
- [`BuiltInToolsets.java`](app/src/main/java/com/gemma/functiongemma/android/BuiltInToolsets.java)：内置工具定义

## 工具调用输出格式

当前运行时适配的是 FunctionGemma 风格的函数调用模板，而不是普通 JSON。一个典型的工具调用输出格式如下：

```text
<start_function_call>call:launch_app{app_name:<escape>微信<escape>}<end_function_call>
```

## 应用映射机制

当前项目支持两层应用匹配机制：

1. 面向常见应用的内置别名映射
2. 基于 launcher 可见应用名的已安装软件兜底匹配

用户还可以在 `Tools` 页面中添加自定义应用别名，将自己习惯说的名字长期绑定到某个已安装应用。

## 常见问题

### SDK location not found

该报错通常表示 `local.properties`、`ANDROID_HOME` 或 `ANDROID_SDK_ROOT` 缺失，或者配置路径不正确。

### 不应提交大型模型文件

模型权重和生成后的 ONNX 文件通常会超过普通 Git 托管平台的限制。除非仓库已经明确配置大文件存储方案，否则应将这些文件保留在本地 [`model`](model) 目录，而不是直接提交到版本库。
