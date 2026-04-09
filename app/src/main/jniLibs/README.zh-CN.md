# JNI 库目录说明

English version: [README.md](README.md)

## 目录用途

该目录作为手动提供 JNI 动态库的兜底位置保留。

当前标准构建流程已经会从本地 Gradle 缓存中自动解包 `com.microsoft.onnxruntime:onnxruntime-android:1.23.2`，并将其头文件与 `libonnxruntime.so` 自动提供给 native 构建过程。

## 何时需要使用

只有在你明确希望覆盖默认依赖解包流程，并手动提供特定 ABI 的共享库时，才需要使用这个目录。

## 预期内容

如果确实需要手动覆盖，请将对应 ABI 的 `.so` 文件放入该目录中。
