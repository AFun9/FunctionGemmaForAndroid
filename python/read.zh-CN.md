# Python 转换说明

English version: [read.md](read.md)

## 目录用途

[`python`](.) 目录包含将官方 Gemma 模型转换为本项目所需 ONNX 资源结构的脚本工具。

这些脚本属于模型准备流程的一部分，并不参与 Android 应用运行时逻辑，但对于生成兼容的模型资源仍然是必要文件。

## 为什么仓库中需要保留这部分脚本

本仓库中的 Android 运行时对模型导出结果有明确要求，包括：

- 特定的 ONNX 目录结构
- 与当前 tokenizer 路径一致的 tokenizer 资源
- 与本项目 prefill 优化流程匹配的导出行为

虽然转换逻辑基于官方 Gemma 导出思路，但本项目保留了一个针对运行时的 prefill 优化。

## 导出精度与运行时支持的区别

当前转换脚本支持导出多种精度，包括：

- `fp32`
- `fp16`
- `q4`
- `q4f16`

但“能够导出”并不等于“运行时已经完整支持”。

一个低精度模型即使已经成功导出，如果运行时没有同时对齐以下部分，实际仍然很难正确运行：

- 模型 I/O dtype
- KV cache dtype
- 临时 buffer dtype
- 宿主侧张量创建方式

### 为什么这对 Java 特别重要

本项目并不把 Java 当作低精度张量执行的主要承载层。

实际原因包括：

- Java 没有自然的 `float16[]` 数组类型
- `fp16` 通常只能借助 `short[]` 这类位模式容器表示
- `int4` 张量需要手动打包到 `byte[]`
- 运行时 cache 张量必须严格匹配 ONNX 模型声明的 dtype

因此，低精度推理更适合放在 JNI/C++ 层完成，而不是依赖 Java-only 的 ONNX Runtime 执行链路。

### 为什么 FP16 仍可能出现较高内存占用

即使权重已经降低到更低精度，运行时内存占用仍可能偏高，常见原因包括：

- KV cache 仍然以 `float32` 形式存储
- 部分导出图会在执行时提升到 `float32`
- 运行时还会额外申请 cast buffer 或 workspace

所以在评估精度收益时，不能只看导出后的模型体积，还要结合真实运行时实现一起分析。

## 面向运行时的导出调整

默认导出逻辑通常会保留完整序列的 logits：

```python
logits_shape = ["batch_size", "sequence_length", "vocab_size"]
```

而本项目在 prefill 阶段只需要最后一个 token 的 logits，因此导出会调整为：

```python
last_logits_shape = ["batch_size", 1, "vocab_size"]
```

这一调整可以减少不必要的内存占用，并更贴合当前移动端运行时的自回归生成路径。

## 前置条件

执行转换前，请先准备：

- 可用的 Python 环境
- 你要转换的官方 Gemma 模型
- 转换脚本 [`build_gemma.py`](build_gemma.py)

## 示例命令

一个典型的转换命令如下：

```bash
python build_gemma.py \
    --model_name "your-org/your-gemma-model" \
    --output "/path/to/output/model-onnx" \
    -p fp32 fp16 q4 q4f16
```

其中模型标识与输出目录需要根据你的本地环境和模型来源自行调整。

## 预期产物

转换完成后，输出目录中应包含 Android 应用运行所需的模型和 tokenizer 资源，通常包括：

- `tokenizer.json`
- `tokenizer_config.json`
- `onnx/model.onnx`
- `onnx/model.onnx_data`（当采用外部张量数据时）

如果还生成了额外的配置文件，也建议一并保留。

## 接入本项目

请将转换后的文件放入项目根目录的 [`model`](../model) 目录中。

Android 构建时会将该目录作为应用资源打包；运行时应用会从 assets 中提取并加载 ONNX 模型和 tokenizer 文件。

## 备注

- 如果采用与本项目不同的导出结构，生成的 ONNX 结果可能无法直接兼容当前推理运行时。
- 该目录应视为可复用的模型准备工具链，而不是临时辅助文件。
