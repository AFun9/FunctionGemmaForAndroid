# Python Conversion Guide

Chinese version: [read.zh-CN.md](read.zh-CN.md)

## Purpose

The [`python`](.) directory contains the conversion utilities used to transform official Gemma checkpoints into the ONNX asset layout expected by this project.

These scripts are part of the model preparation workflow. They are not runtime dependencies of the Android application, but they remain necessary for generating compatible model assets.

## Why This Repository Keeps These Scripts

The Android runtime in this repository expects:

- a specific ONNX model layout
- a tokenizer bundle matching the runtime tokenizer path
- export behavior aligned with the project's optimized prefill workflow

The conversion logic is based on the official Gemma export approach, but this project retains a runtime-specific optimization for prefill logits.

## Precision Export vs Runtime Support

The conversion script supports multiple export precisions, including:

- `fp32`
- `fp16`
- `q4`
- `q4f16`

However, export support should not be confused with complete runtime support.

An exported low-precision model can still be difficult to run correctly if the inference runtime does not align:

- model I/O dtype
- KV cache dtype
- temporary buffer dtype
- tensor creation APIs on the host side

### Why This Matters for Java

This project does not treat Java as the primary execution layer for low-precision tensors.

In practice:

- Java has no natural `float16[]` array type
- `fp16` usually has to be represented through bit-packed `short[]`
- `int4` tensors require manual packing into `byte[]`
- runtime cache tensors must match the exact ONNX model contract

Because of these constraints, low-precision execution is better handled in JNI/C++ rather than in a Java-only ONNX Runtime path.

### Why FP16 Can Still Show High Memory Usage

Even when the exported weight precision is reduced, total runtime memory can remain high if:

- KV cache is still stored in `float32`
- some graph segments are promoted to `float32`
- the runtime allocates additional cast or workspace buffers

For that reason, precision conversion should always be evaluated together with the actual runtime implementation, not only with the exported model size.

## Runtime-Oriented Export Adjustment

The default export pattern may keep logits for the entire sequence:

```python
logits_shape = ["batch_size", "sequence_length", "vocab_size"]
```

In this project, prefill only needs the logits of the final token, so the export is adjusted to:

```python
last_logits_shape = ["batch_size", 1, "vocab_size"]
```

This reduces unnecessary memory usage and better matches the autoregressive generation path used by the mobile runtime.

## Prerequisites

Before running the conversion workflow, prepare:

- a working Python environment
- access to the official Gemma model checkpoint you intend to convert
- the conversion script [`build_gemma.py`](build_gemma.py)

## Example Command

A typical conversion command may look like this:

```bash
python build_gemma.py \
    --model_name "your-org/your-gemma-model" \
    --output "/path/to/output/model-onnx" \
    -p fp32 fp16 q4 q4f16
```

Adjust the model identifier and output path according to your own environment and model source.

## Expected Output

After conversion, the generated output should contain the model and tokenizer assets required by the Android application, usually including:

- `tokenizer.json`
- `tokenizer_config.json`
- `onnx/model.onnx`
- `onnx/model.onnx_data` when external tensor data is emitted

Additional configuration files may also be produced and should be preserved.

## Integration into This Project

Move the converted files into the project-level [`model`](../model) directory.

During the Android build, this directory is packaged as application assets. At runtime, the app extracts and loads the ONNX model and tokenizer files from that asset bundle.

## Notes

- A different export layout may not match the assumptions of this repository's inference runtime.
- This directory should be treated as part of the reproducible model preparation toolchain, not as temporary helper files.
