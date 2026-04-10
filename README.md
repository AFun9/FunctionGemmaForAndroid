# FunctionGemmaForAndroid

Chinese version: [README.zh-CN.md](README.zh-CN.md)

## Overview

FunctionGemmaForAndroid is an Android-first local inference project built around Gemma, ONNX Runtime, and a native C++ execution path. The repository focuses on offline tool calling, lightweight mobile interaction, and a runtime architecture designed for on-device assistant workflows.

## Repository Structure

- [`app`](app): Android application module, UI, tool parsing, validation, and execution flow
- [`main/cpp`](main/cpp): native inference implementation and CMake sources
- [`model`](model): local model assets packaged as Android application assets
- [`python`](python): conversion utilities used to transform official Gemma checkpoints into the ONNX layout expected by this project
- [`FunctionGemma_KVCache_Optimization.md`](FunctionGemma_KVCache_Optimization.md): technical note describing the KV cache optimization strategy used by the runtime

## Development Environment

### Required Toolchain

- Android Studio Ladybug or newer
- JDK 17
- Android SDK 34
- Android NDK `27.0.12077973`
- CMake `3.22.1`

### Project Runtime Dependencies

- Android Gradle Plugin `8.5.2`
- ONNX Runtime Android `1.23.2`
- `minSdk 26`
- `targetSdk 34`

## SDK Configuration

This repository does not include machine-specific Android SDK paths. Before building locally, create `local.properties` in the project root and define your SDK directory:

```properties
sdk.dir=/home/yourname/Android/Sdk
```

You may also use `ANDROID_HOME` or `ANDROID_SDK_ROOT`, but `local.properties` is the most direct option for Android Studio projects.

## Model Assets

The Android app expects the [`model`](model) directory to contain the converted ONNX runtime assets. At minimum, the project usually requires:

- `tokenizer.json`
- `tokenizer_config.json`
- `onnx/model.onnx`
- `onnx/model.onnx_data` when external tensor data is produced

If you only have the original official model checkpoint, convert it first using the scripts under [`python`](python).

## Build and Run

From the project root:

```bash
./gradlew :app:assembleDebug
./gradlew :app:installDebug
```

You may also open the project in Android Studio and run the `app` module directly.

## Runtime Notes

- The first launch extracts packaged model assets from the APK.
- The tokenizer is loaded during initialization.
- Native model session creation may take noticeable time on first load.
- Prefix KV cache preparation can increase startup time and memory usage during initial activation.

## Main Entry Points

- [`MainActivity.java`](app/src/main/java/com/gemma/functiongemma/android/MainActivity.java): Android UI and interaction entry point
- [`FunctionGemmaEngine.java`](app/src/main/java/com/gemma/functiongemma/android/FunctionGemmaEngine.java): inference orchestration and toolset activation
- [`ToolCallParser.java`](app/src/main/java/com/gemma/functiongemma/android/ToolCallParser.java): model output parsing
- [`ToolExecutor.java`](app/src/main/java/com/gemma/functiongemma/android/ToolExecutor.java): tool execution layer
- [`BuiltInToolsets.java`](app/src/main/java/com/gemma/functiongemma/android/BuiltInToolsets.java): built-in tool definitions
- [`UserToolsetTemplate.java`](app/src/main/java/com/gemma/functiongemma/android/UserToolsetTemplate.java): default user-editable tool template

## Tool Call Output Format

The runtime is adapted to the FunctionGemma-style function call template rather than plain JSON. A typical tool call looks like:

```text
<start_function_call>call:open_target{target:<escape>微信<escape>}<end_function_call>
```

## App Mapping Behavior

The default built-in toolset uses `open_target(target)` for "open X" requests.

- For `open_target`, the runtime uses the original text after `打开` from the user message as `target`.
- If `target` is a settings target such as `settings`, `wifi`, `bluetooth`, or `internet`, the app opens the corresponding settings page.
- Otherwise the app tries to launch an installed app by alias or launcher-visible application label.

Users may also create custom app aliases from the `Tools` page. This allows a user-defined spoken name to be permanently mapped to a selected installed application.

## Common Issues

### SDK location not found

This usually means `local.properties`, `ANDROID_HOME`, or `ANDROID_SDK_ROOT` is missing or incorrect.

### Large model files should not be committed

Model weights and generated ONNX artifacts can exceed normal Git hosting limits. Keep them in the local [`model`](model) directory and out of version control unless your repository is explicitly configured for large file storage.


### Large model files cannot be pushed to GitHub

**English**

The ONNX model files under [`model`](model) are often too large for normal GitHub repository history. In most cases:

- keep model artifacts locally
- ignore `model/` in Git
- commit only source code, build logic, and conversion scripts


### High memory usage on first load

Elevated memory usage during the first model load is expected, especially during ONNX Runtime initialization and prefix cache preparation.
