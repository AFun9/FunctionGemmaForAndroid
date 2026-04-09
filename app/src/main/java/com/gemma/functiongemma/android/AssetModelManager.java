package com.gemma.functiongemma.android;

import android.content.Context;
import android.content.res.AssetManager;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

final class AssetModelManager {
    private static final String EXTRACTION_DIR_NAME = "functiongemma_model";
    private static final int COPY_BUFFER_SIZE = 16 * 1024;

    private AssetModelManager() {
    }

    static File ensureExtracted(Context context) throws IOException {
        File outputDir = new File(context.getNoBackupFilesDir(), EXTRACTION_DIR_NAME);
        if (!outputDir.exists() && !outputDir.mkdirs()) {
            throw new IOException("Failed to create model directory: " + outputDir);
        }
        copyAssetTree(context.getAssets(), "", outputDir);
        return outputDir;
    }

    private static void copyAssetTree(AssetManager assetManager, String assetPath, File outputRoot) throws IOException {
        String[] children = assetManager.list(assetPath);
        if (children != null && children.length > 0) {
            File targetDir = assetPath.isEmpty() ? outputRoot : new File(outputRoot, assetPath);
            if (!targetDir.exists() && !targetDir.mkdirs()) {
                throw new IOException("Failed to create directory: " + targetDir);
            }
            for (String child : children) {
                String childAssetPath = assetPath.isEmpty() ? child : assetPath + "/" + child;
                copyAssetTree(assetManager, childAssetPath, outputRoot);
            }
            return;
        }

        File outputFile = new File(outputRoot, assetPath);
        if (outputFile.exists() && outputFile.length() > 0) {
            return;
        }
        File parent = outputFile.getParentFile();
        if (parent != null && !parent.exists() && !parent.mkdirs()) {
            throw new IOException("Failed to create parent directory: " + parent);
        }

        try (InputStream inputStream = assetManager.open(assetPath);
             FileOutputStream outputStream = new FileOutputStream(outputFile)) {
            byte[] buffer = new byte[COPY_BUFFER_SIZE];
            int read;
            while ((read = inputStream.read(buffer)) != -1) {
                outputStream.write(buffer, 0, read);
            }
            outputStream.flush();
        }
    }
}
