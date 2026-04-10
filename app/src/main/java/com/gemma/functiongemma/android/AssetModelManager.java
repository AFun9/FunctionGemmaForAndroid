package com.gemma.functiongemma.android;

import android.content.Context;
import android.content.res.AssetManager;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

final class AssetModelManager {
    interface ProgressListener {
        void onProgress(String status);
    }

    private static final String EXTRACTION_DIR_NAME = "functiongemma_model_v4";
    private static final String EXTRACTION_MARKER_NAME = ".extract_complete";
    private static final int COPY_BUFFER_SIZE = 1024 * 1024;
    private static final String[] REQUIRED_FILES = {
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
            "chat_template.jinja",
            "tokenizer.model",
            "onnx/model.onnx",
            "onnx/model.onnx_data"
    };

    private AssetModelManager() {
    }

    static File ensureExtracted(Context context, ProgressListener progressListener) throws IOException {
        File outputDir = new File(context.getNoBackupFilesDir(), EXTRACTION_DIR_NAME);
        if (!outputDir.exists() && !outputDir.mkdirs()) {
            throw new IOException("Failed to create model directory: " + outputDir);
        }
        if (isExtractionComplete(outputDir)) {
            return outputDir;
        }

        for (String requiredFile : REQUIRED_FILES) {
            copyAssetFile(context.getAssets(), requiredFile, outputDir, progressListener);
        }
        writeExtractionMarker(outputDir);
        return outputDir;
    }

    private static void copyAssetFile(
            AssetManager assetManager,
            String assetPath,
            File outputRoot,
            ProgressListener progressListener) throws IOException {
        File outputFile = new File(outputRoot, assetPath);
        long expectedLength = getAssetLength(assetManager, assetPath);
        if (outputFile.exists() && outputFile.length() > 0
                && (expectedLength < 0 || outputFile.length() == expectedLength)) {
            return;
        }
        File parent = outputFile.getParentFile();
        if (parent != null && !parent.exists() && !parent.mkdirs()) {
            throw new IOException("Failed to create parent directory: " + parent);
        }

        if (progressListener != null) {
            progressListener.onProgress("Status: extracting asset " + assetPath + "...");
        }

        File tempFile = new File(outputFile.getAbsolutePath() + ".tmp");
        if (tempFile.exists() && !tempFile.delete()) {
            throw new IOException("Failed to clear temp file: " + tempFile);
        }

        try (InputStream inputStream = assetManager.open(assetPath, AssetManager.ACCESS_STREAMING);
             FileOutputStream outputStream = new FileOutputStream(tempFile)) {
            byte[] buffer = new byte[COPY_BUFFER_SIZE];
            int read;
            while ((read = inputStream.read(buffer)) != -1) {
                outputStream.write(buffer, 0, read);
            }
            outputStream.flush();
        }

        if (expectedLength >= 0 && tempFile.length() != expectedLength) {
            throw new IOException(
                    "Extracted asset length mismatch for " + assetPath
                            + ": expected " + expectedLength
                            + " bytes, got " + tempFile.length() + " bytes"
            );
        }

        if (outputFile.exists() && !outputFile.delete()) {
            throw new IOException("Failed to replace existing file: " + outputFile);
        }
        if (!tempFile.renameTo(outputFile)) {
            throw new IOException("Failed to finalize extracted file: " + outputFile);
        }
    }

    private static boolean isExtractionComplete(File outputDir) {
        File markerFile = new File(outputDir, EXTRACTION_MARKER_NAME);
        if (!markerFile.exists()) {
            return false;
        }
        for (String requiredFile : REQUIRED_FILES) {
            File file = new File(outputDir, requiredFile);
            if (!file.exists() || file.length() == 0) {
                return false;
            }
        }
        return true;
    }

    private static void writeExtractionMarker(File outputDir) throws IOException {
        File markerFile = new File(outputDir, EXTRACTION_MARKER_NAME);
        try (OutputStream outputStream = new FileOutputStream(markerFile, false)) {
            outputStream.flush();
        }
    }

    private static long getAssetLength(AssetManager assetManager, String assetPath) {
        try (var assetFileDescriptor = assetManager.openFd(assetPath)) {
            return assetFileDescriptor.getLength();
        } catch (IOException ignored) {
            return -1L;
        }
    }
}
