package com.gemma.functiongemma.android;

import android.content.Context;
import android.util.Log;

import com.gemma.functiongemma.CppInferenceEngine;
import com.gemma.functiongemma.ManualBPETokenizer;
import com.gemma.functiongemma.Tokenizer;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class FunctionGemmaEngine implements AutoCloseable {
    private static final String TAG = "FunctionGemmaEngine";
    private static final String DEFAULT_TOOLSET_ID = BuiltInToolsets.MOBILE_ASSISTANT_ID;
    private static final String MODEL_PATH = "onnx/model.onnx";
    private static final String CACHE_DIR_NAME = "prefix_kv_cache";
    private static final String[] MODEL_SIGNATURE_FILES = {
            "onnx/model.onnx",
            "onnx/model.onnx_data",
            "config.json",
            "generation_config.json"
    };
    private static final String[] TOKENIZER_SIGNATURE_FILES = {
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
            "chat_template.jinja",
            "tokenizer.model"
    };

    private final Context appContext;
    private final CppInferenceEngine inferenceEngine;
    private final Map<String, ToolsetDefinition> toolsets = new LinkedHashMap<>();
    private final ToolsetStore toolsetStore;

    private Tokenizer tokenizer;
    private File extractedDir;
    private File prefixCacheDir;
    private PrefixCacheIndexStore prefixCacheIndexStore;
    private String modelSignature;
    private String tokenizerSignature;
    private String activeToolsetId;
    private String activeFingerprint;
    private boolean lastActivationUsedPersistedCache;
    private PrefixCacheStatus activeCacheStatus;
    private boolean loaded;

    public interface StatusListener {
        void onStatus(String status);
    }

    public FunctionGemmaEngine(Context context) {
        this.appContext = context.getApplicationContext();
        this.inferenceEngine = new CppInferenceEngine();
        this.toolsetStore = new ToolsetStore(appContext.getNoBackupFilesDir());
        registerOrReplaceToolsetInternal(createDefaultToolset(), false);
        loadPersistedToolsets();
        this.activeToolsetId = DEFAULT_TOOLSET_ID;
    }

    public synchronized void load() throws Exception {
        load(null);
    }

    public synchronized void load(StatusListener statusListener) throws Exception {
        if (loaded) {
            notifyStatus(statusListener, "Status: model already loaded");
            return;
        }
        long startTime = System.currentTimeMillis();
        notifyStatus(statusListener, "Status: extracting model assets...");
        extractedDir = AssetModelManager.ensureExtracted(appContext);
        notifyStatus(statusListener, "Status: preparing prefix cache directory...");
        prefixCacheDir = new File(appContext.getNoBackupFilesDir(), CACHE_DIR_NAME);
        if (!prefixCacheDir.exists() && !prefixCacheDir.mkdirs()) {
            throw new IOException("Failed to create prefix cache directory: " + prefixCacheDir);
        }
        prefixCacheIndexStore = new PrefixCacheIndexStore(prefixCacheDir);
        notifyStatus(statusListener, "Status: loading tokenizer...");
        tokenizer = ManualBPETokenizer.fromAssets(appContext, "");
        notifyStatus(statusListener, "Status: initializing native model session...");
        inferenceEngine.init(new File(extractedDir, MODEL_PATH).getAbsolutePath(), Runtime.getRuntime().availableProcessors());
        notifyStatus(statusListener, "Status: building model fingerprints...");
        modelSignature = buildDirectorySignature(extractedDir, MODEL_SIGNATURE_FILES);
        tokenizerSignature = buildDirectorySignature(extractedDir, TOKENIZER_SIGNATURE_FILES);
        loaded = true;
        notifyStatus(statusListener, "Status: preparing default prefix KV cache...");
        activateToolset(activeToolsetId);
        notifyStatus(
                statusListener,
                "Status: model loaded in " + (System.currentTimeMillis() - startTime) + " ms"
        );
    }

    public synchronized boolean isLoaded() {
        return loaded;
    }

    public synchronized void registerOrReplaceToolset(ToolsetDefinition toolset) {
        registerOrReplaceToolsetInternal(toolset, true);
    }

    public synchronized ToolsetDefinition registerOrReplaceToolset(
            String toolsetId,
            String displayName,
            String systemPrompt,
            List<Map<String, Object>> tools) {
        ToolsetDefinition toolset = new ToolsetDefinition(toolsetId, displayName, systemPrompt, tools);
        registerOrReplaceToolset(toolset);
        return toolset;
    }

    public synchronized List<String> getRegisteredToolsetIds() {
        return new ArrayList<>(toolsets.keySet());
    }

    public synchronized ToolsetDefinition getToolset(String toolsetId) {
        ToolsetDefinition toolset = toolsets.get(toolsetId);
        if (toolset == null) {
            return null;
        }
        return new ToolsetDefinition(toolset.id(), toolset.displayName(), toolset.systemPrompt(), toolset.tools());
    }

    public synchronized String getActiveToolsetId() {
        return activeToolsetId;
    }

    public synchronized boolean wasLastActivationCacheHit() {
        return lastActivationUsedPersistedCache;
    }

    public synchronized PrefixCacheStatus getActiveToolsetCacheStatus() {
        return activeCacheStatus;
    }

    public synchronized void activateToolset(String toolsetId) throws Exception {
        ToolsetDefinition toolset = toolsets.get(toolsetId);
        if (toolset == null) {
            throw new IllegalArgumentException("Unknown toolset: " + toolsetId);
        }
        if (!loaded) {
            activeToolsetId = toolsetId;
            load();
            return;
        }
        ensurePrefixCacheReady(toolset, false);
        activeToolsetId = toolsetId;
    }

    public synchronized boolean deleteToolset(String toolsetId) throws Exception {
        if (toolsetId == null || toolsetId.trim().isEmpty()) {
            throw new IllegalArgumentException("Toolset id must not be empty");
        }
        if (DEFAULT_TOOLSET_ID.equals(toolsetId)) {
            return false;
        }
        ToolsetDefinition removed = toolsets.remove(toolsetId);
        if (removed == null) {
            return false;
        }

        persistCustomToolsets();
        removePrefixCache(toolsetId);

        if (toolsetId.equals(activeToolsetId)) {
            activeToolsetId = DEFAULT_TOOLSET_ID;
            activeFingerprint = null;
            activeCacheStatus = null;
            if (loaded) {
                activateToolset(DEFAULT_TOOLSET_ID);
            }
        }
        return true;
    }

    public synchronized void rebuildToolsetCache(String toolsetId) throws Exception {
        ToolsetDefinition toolset = toolsets.get(toolsetId);
        if (toolset == null) {
            throw new IllegalArgumentException("Unknown toolset: " + toolsetId);
        }
        if (!loaded) {
            load();
        }
        ensurePrefixCacheReady(toolset, true);
        activeToolsetId = toolsetId;
    }

    public synchronized String generate(String userMessage, int maxNewTokens) throws Exception {
        return generate(activeToolsetId, userMessage, maxNewTokens);
    }

    public synchronized String generate(String toolsetId, String userMessage, int maxNewTokens) throws Exception {
        if (!loaded) {
            load();
        }
        ToolsetDefinition toolset = toolsets.get(toolsetId);
        if (toolset == null) {
            throw new IllegalArgumentException("Unknown toolset: " + toolsetId);
        }
        ensurePrefixCacheReady(toolset, false);

        List<Map<String, Object>> messages = new ArrayList<>();
        messages.add(createMessage("developer", toolset.systemPrompt()));
        messages.add(createMessage("user", userMessage));

        Tokenizer.TokenizerResult tokenizerResult = tokenizer.applyChatTemplate(messages, toolset.tools(), true);
        long[] stopIds = buildStopIds(tokenizer);
        long[] result = inferenceEngine.generateWithPrefixCache(tokenizerResult.inputIds, stopIds, maxNewTokens);
        return tokenizer.decode(result, false);
    }

    @Override
    public synchronized void close() {
        inferenceEngine.close();
        activeFingerprint = null;
        activeCacheStatus = null;
        loaded = false;
    }

    private void ensurePrefixCacheReady(ToolsetDefinition toolset, boolean forceRebuild) throws Exception {
        long[] prefixTokens = buildPrefixTokens(toolset);
        String fingerprint = ToolsetFingerprint.create(toolset, modelSignature, tokenizerSignature);
        if (!forceRebuild
                && fingerprint.equals(activeFingerprint)
                && toolset.id().equals(activeToolsetId)
                && inferenceEngine.getSystemPrefixSequenceLength() == prefixTokens.length) {
            lastActivationUsedPersistedCache = true;
            return;
        }

        PrefixCacheEntry entry = forceRebuild ? null : prefixCacheIndexStore.find(toolset.id(), fingerprint);
        if (entry != null) {
            File cacheFile = new File(prefixCacheDir, entry.kvFileName);
            if (cacheFile.exists()
                    && inferenceEngine.importSystemPrefixCache(cacheFile.getAbsolutePath())
                    && inferenceEngine.getSystemPrefixSequenceLength() == prefixTokens.length) {
                activeToolsetId = toolset.id();
                activeFingerprint = fingerprint;
                lastActivationUsedPersistedCache = true;
                activeCacheStatus = buildCacheStatus(toolset.id(), entry, true);
                return;
            }
        }

        if (!inferenceEngine.precomputeSystemPrefix(prefixTokens)) {
            throw new IllegalStateException("Failed to precompute prefix KV cache for toolset: " + toolset.id());
        }
        if (inferenceEngine.getSystemPrefixSequenceLength() != prefixTokens.length) {
            throw new IllegalStateException("Prefix KV cache length mismatch for toolset: " + toolset.id());
        }

        String kvFileName = fingerprint + ".kvbin";
        File cacheFile = new File(prefixCacheDir, kvFileName);
        if (!inferenceEngine.exportSystemPrefixCache(cacheFile.getAbsolutePath())) {
            throw new IOException("Failed to export prefix KV cache: " + cacheFile);
        }

        PrefixCacheEntry newEntry = new PrefixCacheEntry();
        newEntry.toolsetId = toolset.id();
        newEntry.displayName = toolset.displayName();
        newEntry.fingerprint = fingerprint;
        newEntry.kvFileName = kvFileName;
        newEntry.systemSequenceLength = prefixTokens.length;
        newEntry.createdAtEpochMs = System.currentTimeMillis();
        prefixCacheIndexStore.upsert(newEntry);

        activeToolsetId = toolset.id();
        activeFingerprint = fingerprint;
        lastActivationUsedPersistedCache = false;
        activeCacheStatus = buildCacheStatus(toolset.id(), newEntry, false);
    }

    private static void notifyStatus(StatusListener listener, String status) {
        Log.d(TAG, status);
        if (listener != null) {
            listener.onStatus(status);
        }
    }

    private long[] buildPrefixTokens(ToolsetDefinition toolset) {
        List<Map<String, Object>> messages = new ArrayList<>();
        messages.add(createMessage("developer", toolset.systemPrompt()));
        return tokenizer.applyChatTemplate(messages, toolset.tools(), false).inputIds;
    }

    private static String buildDirectorySignature(File baseDir, String[] relativePaths) {
        StringBuilder builder = new StringBuilder();
        for (String relativePath : relativePaths) {
            File file = new File(baseDir, relativePath);
            builder.append(relativePath)
                    .append(':')
                    .append(file.exists() ? file.length() : -1L)
                    .append(':')
                    .append(file.exists() ? file.lastModified() : -1L)
                    .append(';');
        }
        return builder.toString();
    }

    private static long[] buildStopIds(Tokenizer tokenizer) {
        List<Long> stopIds = new ArrayList<>();
        stopIds.add((long) tokenizer.getEosTokenId());
        stopIds.add((long) tokenizer.convertTokenToId("<start_function_response>"));
        stopIds.add((long) tokenizer.convertTokenToId("<end_of_turn>"));
        if (tokenizer.hasToken("<end_function_call>")) {
            stopIds.add((long) tokenizer.convertTokenToId("<end_function_call>"));
        }

        long[] result = new long[stopIds.size()];
        for (int i = 0; i < stopIds.size(); i++) {
            result[i] = stopIds.get(i);
        }
        return result;
    }

    private static Map<String, Object> createMessage(String role, String content) {
        Map<String, Object> message = new LinkedHashMap<>();
        message.put("role", role);
        message.put("content", content);
        return message;
    }

    private static ToolsetDefinition createDefaultToolset() {
        return BuiltInToolsets.createMobileAssistantToolset();
    }

    private void registerOrReplaceToolsetInternal(ToolsetDefinition toolset, boolean persist) {
        toolsets.put(toolset.id(), toolset);
        if (activeToolsetId == null) {
            activeToolsetId = toolset.id();
        }
        if (toolset.id().equals(activeToolsetId)) {
            activeFingerprint = null;
        }
        if (persist) {
            persistCustomToolsets();
        }
    }

    private void loadPersistedToolsets() {
        try {
            for (ToolsetDefinition toolset : toolsetStore.readAll()) {
                registerOrReplaceToolsetInternal(toolset, false);
            }
        } catch (IOException ignored) {
            // Keep the built-in default toolset even if persistence is unavailable.
        }
    }

    private void persistCustomToolsets() {
        try {
            List<ToolsetDefinition> customToolsets = new ArrayList<>();
            for (ToolsetDefinition toolset : toolsets.values()) {
                if (!DEFAULT_TOOLSET_ID.equals(toolset.id())) {
                    customToolsets.add(toolset);
                }
            }
            toolsetStore.writeAll(customToolsets);
        } catch (IOException ignored) {
            // Persistence failure should not block registering a toolset in memory.
        }
    }

    private void removePrefixCache(String toolsetId) {
        if (prefixCacheIndexStore == null || prefixCacheDir == null) {
            return;
        }
        try {
            PrefixCacheEntry removed = prefixCacheIndexStore.remove(toolsetId);
            if (removed != null && removed.kvFileName != null) {
                File cacheFile = new File(prefixCacheDir, removed.kvFileName);
                if (cacheFile.exists()) {
                    //noinspection ResultOfMethodCallIgnored
                    cacheFile.delete();
                }
            }
        } catch (IOException ignored) {
            // Cache cleanup failure should not block toolset deletion.
        }
    }

    private PrefixCacheStatus buildCacheStatus(String toolsetId, PrefixCacheEntry entry, boolean cacheHit) {
        File cacheFile = new File(prefixCacheDir, entry.kvFileName);
        return new PrefixCacheStatus(
                toolsetId,
                cacheFile.exists(),
                cacheHit,
                entry.systemSequenceLength,
                cacheFile.exists() ? cacheFile.length() : 0L,
                entry.createdAtEpochMs
        );
    }
}
