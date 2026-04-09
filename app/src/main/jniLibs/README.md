This directory is kept as an escape hatch for manually supplied JNI libraries.

The app build now unpacks `com.microsoft.onnxruntime:onnxruntime-android:1.23.2`
from the local Gradle cache and feeds its headers and `libonnxruntime.so`
to CMake automatically.

If you ever need to override that flow, place ABI-specific `.so` files here.
