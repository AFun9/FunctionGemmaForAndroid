# Python 转换说明

[`python`](.) 目录用于把官方 Gemma 模型转换成当前项目可直接使用的 ONNX 目录结构。

## 来源

转换脚本基于官方提供的 `build_gemma.py` 思路整理而来。本项目保留这部分脚本，是因为仓库内的 Android 推理链路依赖转换后的 ONNX 结构，不能直接拿官方原始模型即用。

## 为什么需要项目内这份转换脚本

官方转换逻辑在 prefill 阶段会输出完整序列长度的 logits：

```python
logits_shape = ["batch_size", "sequence_length", "vocab_size"]
```

但本项目的实际推理场景里，prefill 阶段只需要最后一个 token 的 logits，因此这里做了针对性优化，输出形状改为：

```python
last_logits_shape = ["batch_size", 1, "vocab_size"]
```

这样做的好处是：

- 减少 prefill 阶段不必要的 logits 内存占用
- 更适合本项目这种自回归生成场景
- 更容易在移动端内存预算下运行

## 转换前准备

你需要先准备：

- Python 运行环境
- 官方 Gemma 模型
- 本项目中的 [`build_gemma.py`](build_gemma.py)

## 基本转换方式

下面是一种典型的转换方式示例：

```python
model_author = ""
gemma_model = "myemoji-gemma-3-270m-it"

repo_id = f"{model_author}/{gemma_model}"
save_path = f"/content/{gemma_model}-onnx"
```

执行转换：

```bash
python build_gemma.py \
    --model_name "${repo_id}" \
    --output "${save_path}" \
    -p fp32 fp16 q4 q4f16
```

转换完成后，会得到一个 ONNX 模型目录。

## 如何接入本项目

转换后的目录至少应包含：

- `tokenizer.json`
- `tokenizer_config.json`
- `onnx/model.onnx`
- 可能还包括 `onnx/model.onnx_data`

之后把这些文件放入项目根目录的 [`model`](../model) 中，Android 工程会把它们作为 assets 打包。

## 备注

- 如果不使用本项目这套转换脚本，生成出来的 ONNX 结构可能无法直接匹配当前推理逻辑
- 这部分脚本属于模型准备流程，不是运行期代码，但对本项目仍然是必要文件
