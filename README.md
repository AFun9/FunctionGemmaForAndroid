# FunctionGemmaForAndroid

本项目是一个基于 Gemma / ONNX Runtime 的 Android 本地函数调用实验项目，当前目标是 Android 端离线推理。

## 项目结构

- [`app`](app): Android 应用入口、界面、工具执行逻辑。
- [`main/cpp`](main/cpp): 原生 C++ 推理实现。
- [`model`](model): 本地模型资产目录，Android 会把它当作 assets 使用。
- [`python`](python): 把官方模型转换成 ONNX 结构的脚本，需保留。

## 本地开发环境

### Android 开发必需

- Android Studio Ladybug 或更高版本
- JDK 17
- Android SDK 34
- Android NDK `27.0.12077973`
- CMake `3.22.1`

### 关键依赖

- Android Gradle Plugin `8.5.2`
- ONNX Runtime Android `1.23.2`
- `minSdk 26`
- `targetSdk 34`

### 需要你本机自己提供的内容

这个仓库当前**不包含**你的本机 SDK 路径配置，所以首次拉起前需要补一项：

1. 在项目根目录创建 `local.properties`
2. 写入你的 Android SDK 路径，例如：

```properties
sdk.dir=/home/yourname/Android/Sdk
```

或者你也可以通过环境变量提供 SDK 路径，但对 Android Studio 来说，`local.properties` 通常更直接。

## 如何运行 Android 版本

### 1. 准备模型目录

Android 工程会把 [`model`](model) 当作 assets 打包。目录下至少需要有这些内容：

- `tokenizer.json`
- `tokenizer_config.json`
- `onnx/model.onnx`
- 可能还包括 `onnx/model.onnx_data`

如果你手里是官方模型而不是 ONNX 目录，先用 [`python`](python) 里的转换流程生成 ONNX 结构，再放入 [`model`](model)。

### 2. 构建并安装

在项目根目录执行：

```bash
./gradlew :app:assembleDebug
./gradlew :app:installDebug
```

或者直接在 Android Studio 里打开项目后运行 `app`。

### 3. 首次启动说明

- 首次启动会从 assets 中准备模型文件
- 会加载 tokenizer
- 会初始化 native model session
- 首次加载耗时和内存占用都会明显高于后续运行

## Android 开发说明

### 主要入口

- 主界面：[`MainActivity.java`](app/src/main/java/com/gemma/functiongemma/android/MainActivity.java)
- 推理协调：[`FunctionGemmaEngine.java`](app/src/main/java/com/gemma/functiongemma/android/FunctionGemmaEngine.java)
- 工具调用解析：[`ToolCallParser.java`](app/src/main/java/com/gemma/functiongemma/android/ToolCallParser.java)
- 工具执行：[`ToolExecutor.java`](app/src/main/java/com/gemma/functiongemma/android/ToolExecutor.java)
- 内置工具集：[`BuiltInToolsets.java`](app/src/main/java/com/gemma/functiongemma/android/BuiltInToolsets.java)

### 当前函数调用输出格式

当前模型侧使用的是 FunctionGemma 风格模板，不是 JSON。典型输出类似：

```text
<start_function_call>call:launch_app{app_name:<escape>微信<escape>}<end_function_call>
```

解析逻辑已经按这个格式适配。

## 常见问题

### 1. `SDK location not found`

说明缺少 `local.properties` 或 `ANDROID_HOME` 配置。

### 2. GitHub 推送失败，提示模型文件过大

[`model`](model) 里的 ONNX 权重通常不适合直接提交到 GitHub。建议：

- 本地保留模型目录
- 在 git 中忽略 `model/`
- 仓库只保留代码和转换脚本

### 3. 首次运行内存占用很高

这是模型加载和 ONNX Runtime 初始化阶段的正常现象，尤其是首启和 prefix cache 初始化时更明显。
