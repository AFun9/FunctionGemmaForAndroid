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

The conversion logic is based on the official Gemma export approach, but the local script is intentionally narrower than the original experimental exporter. It only keeps the export paths that are still used by this Android runtime.

## What Was Simplified

The exporter in this repository was reduced to the minimum feature set that still matches the Android app:

- only `fp32` and `fp16` export are kept
- old experimental branches that were not part of the current mobile runtime path were removed
- the CLI now focuses on a small number of runtime-relevant export controls instead of a wide set of one-off switches

This makes the script easier to reason about and reduces the chance of exporting a model shape or dtype combination that the Android runtime does not actually use.

## Precision Export vs Runtime Support

The conversion script in this repository currently supports two export precisions:

- `fp32`
- `fp16`

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

## Runtime-Oriented Export Adjustments

### 1. Last-token LM head by default

The main optimization kept in this repository is that the default LM head export path is `last_token`.

The older full-sequence export pattern keeps logits for the entire sequence:

```python
logits_shape = ["batch_size", "sequence_length", "vocab_size"]
```

In this project, prefill only needs the logits of the final token, so the default export is adjusted to:

```python
last_logits_shape = ["batch_size", 1, "vocab_size"]
```

Why this was done:

- the Android runtime only needs the next-token decision after prefill
- computing full-sequence logits creates unnecessary temporary memory pressure on mobile

Result:

- lower temporary logits memory during generation
- exported graph behavior is closer to the actual autoregressive runtime path

### 2. Optional mobile-oriented export controls

The script still keeps a few focused controls because they directly affect runtime cost:

- `--layernorm-policy`: keeps LayerNorm in `fp32` by default, with an `io` option to reduce casts
- `--rope-cache-length`: allows exporting a shorter precomputed RoPE cache to reduce fixed initializer size
- `--lm-head-dtype`: controls whether the LM head computes in model I/O dtype or `fp32`
- `--pretranspose-lm-head`: optionally stores a transposed LM head weight to reduce runtime transpose work

These options remain because they are tied to concrete runtime tradeoffs, not because the exporter is intended to be a general-purpose research tool.

## Prerequisites

Before running the conversion workflow, prepare:

- a working Python environment
- access to the official Gemma model checkpoint you intend to convert
- the conversion script [`build_gemma.py`](build_gemma.py)

## Example Commands

Typical export:

```bash
python build_gemma.py \
    --model_name "your-org/your-gemma-model" \
    --output "/path/to/output/model-onnx" \
    -p fp32 fp16
```

Mobile-oriented export with explicit runtime-related knobs:

```bash
python build_gemma.py \
    --model_name "your-org/your-gemma-model" \
    --output "/path/to/output/model-onnx" \
    -p fp16 \
    --lm-head-policy last_token \
    --layernorm-policy io \
    --rope-cache-length 4096
```

Adjust the model identifier, output path, and export options according to your own environment and runtime target.

## Expected Output

After conversion, the generated output should contain the model and tokenizer assets required by the Android application, usually including:

- `tokenizer.json`
- `tokenizer_config.json`
- `onnx/model.onnx`
- `onnx/model.onnx_data` when external tensor data is emitted

Additional configuration files may also be produced and should be preserved.

## Why These Optimizations Matter

For this project, the goal of export is not just "produce a valid ONNX file". The exported graph should also better match how the Android runtime actually runs:

- smaller and simpler export surface
- fewer unused experimental branches
- lower temporary memory pressure during prefill
- better alignment between exported tensor behavior and the JNI/C++ execution path

## Integration into This Project

Move the converted files into the project-level [`model`](../model) directory.

During the Android build, this directory is packaged as application assets. At runtime, the app extracts and loads the ONNX model and tokenizer files from that asset bundle.

## Notes

- A different export layout may not match the assumptions of this repository's inference runtime.
- This directory should be treated as part of the reproducible model preparation toolchain, not as temporary helper files.
