# JNI Library Directory

Chinese version: [README.zh-CN.md](README.zh-CN.md)

## Purpose

This directory is kept as an escape hatch for manually supplied JNI libraries.

The standard build path already unpacks `com.microsoft.onnxruntime:onnxruntime-android:1.23.2` from the local Gradle cache and provides its headers and `libonnxruntime.so` to the native build automatically.

## When to Use It

You only need this directory when you intentionally want to override the default dependency-unpack flow and provide ABI-specific shared libraries manually.

## Expected Contents

If manual override is required, place ABI-specific `.so` files here.
