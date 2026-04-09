# FunctionGemma KV Cache Optimization

Chinese version: [FunctionGemma_KVCache_Optimization.zh-CN.md](FunctionGemma_KVCache_Optimization.zh-CN.md)

## Overview

This document describes the KV cache optimization strategy used by FunctionGemmaForAndroid. The primary goal is to reduce repeated computation by precomputing the key-value cache for the stable prefix of a conversation, especially the developer prompt and tool declarations.

## Motivation

In a conventional chat inference flow, every generation step reprocesses the entire prompt context:

- developer or system prompt
- tool definitions
- user input
- previously generated output

The developer prompt and tool declarations are typically stable across many turns. Recomputing them for every request is unnecessarily expensive on mobile devices.

## Core Strategy

The optimization splits inference into two phases:

1. **Prefix precomputation**  
   Precompute the KV cache for the stable prefix.
2. **User-stage inference**  
   Reuse the cached prefix state and process only the user-specific suffix.

## Prefix Composition

The cached prefix generally includes:

- the developer prompt
- tool declarations
- template-controlled structural tokens required by the chat format

It intentionally excludes the current user request.

## Two-Phase Runtime Flow

### Phase 1: Prefix Precomputation

The runtime constructs a prefix-only message list, applies the chat template, tokenizes the result, and feeds it into native inference to generate the KV cache for all transformer layers.

The exported cache is then persisted and indexed so it can be reused across future activations of the same toolset.

### Phase 2: User-Stage Inference

When the user submits a message, the runtime rebuilds the full prompt, determines the user-specific suffix beyond the cached prefix, and executes inference starting from the precomputed state instead of from an empty context.

This reduces both attention computation and repeated prompt handling.

## Why It Helps

This design improves efficiency in three ways:

- lowers repeated compute cost for stable prompt content
- reduces unnecessary memory pressure during prefill
- improves responsiveness for repeated tool-based interactions

## What Is Persisted

The runtime stores:

- the prefix KV cache artifact
- metadata describing the toolset fingerprint
- sequence length and cache availability information

This allows the runtime to determine whether an existing cache can be reused safely.

## Cache Invalidation

The prefix cache must be rebuilt whenever the effective prefix changes. In practice, that usually includes:

- tool definition changes
- developer prompt changes
- tokenizer changes
- model file changes

## Implementation Notes

Relevant implementation can be found in:

- [`FunctionGemmaEngine.java`](app/src/main/java/com/gemma/functiongemma/android/FunctionGemmaEngine.java)
- [`PrefixCacheIndexStore.java`](app/src/main/java/com/gemma/functiongemma/android/PrefixCacheIndexStore.java)
- [`ToolsetFingerprint.java`](app/src/main/java/com/gemma/functiongemma/android/ToolsetFingerprint.java)
- native inference code under [`main/cpp`](main/cpp)

## Scope

This optimization is most valuable when:

- tool declarations are large
- the prompt prefix remains stable for many requests
- the device is memory-sensitive
- the interaction style repeatedly reuses the same toolset
