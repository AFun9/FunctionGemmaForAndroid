package com.gemma.functiongemma.android;

import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ProgressBar;
import android.widget.Spinner;
import android.widget.TextView;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import com.gemma.functiongemma.R;

import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public final class MainActivity extends AppCompatActivity {
    private static final String TAG = "FunctionGemmaUI";
    private static final int DEFAULT_MAX_NEW_TOKENS = 64;
    private static final int LOG_CHUNK_SIZE = 3000;
    private static final int PAGE_ASSISTANT = 0;
    private static final int PAGE_TOOLS = 1;
    private static final int OUTPUT_TAB_MODEL = 0;
    private static final int OUTPUT_TAB_PARSED = 1;
    private static final int OUTPUT_TAB_EXECUTION = 2;

    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private final Handler mainHandler = new Handler(Looper.getMainLooper());

    private FunctionGemmaEngine engine;
    private AppAliasStore appAliasStore;
    private ToolExecutor toolExecutor;
    private TextView statusText;
    private TextView cacheStatusText;
    private ProgressBar inferenceProgress;
    private TextView modelOutputText;
    private TextView parsedToolText;
    private TextView executionOutputText;
    private TextView pageAssistantTab;
    private TextView pageToolsTab;
    private TextView tabModelOutput;
    private TextView tabParsedTool;
    private TextView tabExecutionResult;
    private TextView studioToggleText;
    private TextView studioSummaryTitleText;
    private TextView studioSummaryMetaText;
    private View assistantPage;
    private View toolsPage;
    private View studioContent;
    private Spinner toolsetSpinner;
    private EditText toolsetIdEditText;
    private EditText toolsetNameEditText;
    private EditText systemPromptEditText;
    private EditText toolsJsonEditText;
    private EditText inputEditText;
    private Button loadButton;
    private Button saveToolsetButton;
    private Button useSelectedToolsetButton;
    private Button deleteToolsetButton;
    private Button rebuildPrefixCacheButton;
    private Button resetTemplateButton;
    private Button saveAppAliasButton;
    private Button runButton;
    private ArrayAdapter<String> toolsetAdapter;
    private ArrayAdapter<InstalledAppOption> installedAppAdapter;
    private Spinner installedAppSpinner;
    private EditText appAliasEditText;
    private TextView appAliasSummaryText;
    private final List<AppAliasEntry> userAppAliases = new ArrayList<>();
    private final SimpleDateFormat timeFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault());
    private PrefixCacheStatus lastCacheStatus;

    private interface BackgroundTask {
        void run() throws Exception;
    }

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        engine = new FunctionGemmaEngine(getApplicationContext());
        appAliasStore = new AppAliasStore(getApplicationContext().getNoBackupFilesDir());
        loadPersistedAppAliases();
        rebuildAliasDependencies();
        bindViews();
        setupAdapters();
        bindActions();
        initializeUiState();
    }

    private void bindViews() {
        statusText = findViewById(R.id.statusText);
        cacheStatusText = findViewById(R.id.cacheStatusText);
        inferenceProgress = findViewById(R.id.inferenceProgress);
        modelOutputText = findViewById(R.id.modelOutputText);
        parsedToolText = findViewById(R.id.parsedToolText);
        executionOutputText = findViewById(R.id.executionOutputText);
        pageAssistantTab = findViewById(R.id.pageAssistantTab);
        pageToolsTab = findViewById(R.id.pageToolsTab);
        tabModelOutput = findViewById(R.id.tabModelOutput);
        tabParsedTool = findViewById(R.id.tabParsedTool);
        tabExecutionResult = findViewById(R.id.tabExecutionResult);
        studioToggleText = findViewById(R.id.studioToggleText);
        studioSummaryTitleText = findViewById(R.id.studioSummaryTitleText);
        studioSummaryMetaText = findViewById(R.id.studioSummaryMetaText);
        assistantPage = findViewById(R.id.assistantPage);
        toolsPage = findViewById(R.id.toolsPage);
        studioContent = findViewById(R.id.studioContent);
        toolsetSpinner = findViewById(R.id.toolsetSpinner);
        toolsetIdEditText = findViewById(R.id.toolsetIdEditText);
        toolsetNameEditText = findViewById(R.id.toolsetNameEditText);
        systemPromptEditText = findViewById(R.id.systemPromptEditText);
        toolsJsonEditText = findViewById(R.id.toolsJsonEditText);
        inputEditText = findViewById(R.id.inputEditText);
        loadButton = findViewById(R.id.loadButton);
        saveToolsetButton = findViewById(R.id.saveToolsetButton);
        useSelectedToolsetButton = findViewById(R.id.useSelectedToolsetButton);
        deleteToolsetButton = findViewById(R.id.deleteToolsetButton);
        rebuildPrefixCacheButton = findViewById(R.id.rebuildPrefixCacheButton);
        resetTemplateButton = findViewById(R.id.resetTemplateButton);
        saveAppAliasButton = findViewById(R.id.saveAppAliasButton);
        runButton = findViewById(R.id.runButton);
        installedAppSpinner = findViewById(R.id.installedAppSpinner);
        appAliasEditText = findViewById(R.id.appAliasEditText);
        appAliasSummaryText = findViewById(R.id.appAliasSummaryText);
    }

    private void setupAdapters() {
        toolsetAdapter = new ArrayAdapter<>(this, android.R.layout.simple_spinner_item, new ArrayList<>());
        toolsetAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        toolsetSpinner.setAdapter(toolsetAdapter);
        installedAppAdapter = new ArrayAdapter<>(this, android.R.layout.simple_spinner_item, new ArrayList<>());
        installedAppAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        installedAppSpinner.setAdapter(installedAppAdapter);
    }

    private void bindActions() {
        loadButton.setOnClickListener(view -> loadModel());
        saveToolsetButton.setOnClickListener(view -> saveToolset());
        useSelectedToolsetButton.setOnClickListener(view -> useSelectedToolset());
        deleteToolsetButton.setOnClickListener(view -> deleteSelectedToolset());
        rebuildPrefixCacheButton.setOnClickListener(view -> rebuildSelectedToolsetCache());
        resetTemplateButton.setOnClickListener(view -> resetTemplate());
        saveAppAliasButton.setOnClickListener(view -> saveAppAlias());
        runButton.setOnClickListener(view -> runInference());
        pageAssistantTab.setOnClickListener(view -> selectMainPage(PAGE_ASSISTANT));
        pageToolsTab.setOnClickListener(view -> selectMainPage(PAGE_TOOLS));
        tabModelOutput.setOnClickListener(view -> selectOutputTab(OUTPUT_TAB_MODEL));
        tabParsedTool.setOnClickListener(view -> selectOutputTab(OUTPUT_TAB_PARSED));
        tabExecutionResult.setOnClickListener(view -> selectOutputTab(OUTPUT_TAB_EXECUTION));
        studioToggleText.setOnClickListener(view -> setStudioExpanded(studioContent == null || studioContent.getVisibility() != View.VISIBLE));
    }

    private void initializeUiState() {
        syncToolsetUi(engine.getActiveToolsetId());
        refreshInstalledApps();
        refreshAppAliasSummary();
        selectMainPage(PAGE_ASSISTANT);
        setStudioExpanded(false);
        selectOutputTab(OUTPUT_TAB_MODEL);
        clearOutputPanels();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        executor.shutdownNow();
        engine.close();
    }

    private void loadModel() {
        executeBusyTask("Status: extracting assets and loading model...", () -> {
            engine.load(status -> mainHandler.post(() -> updateStatus(status)));
            finishBusyTask(() -> {
                updateStatus("Status: model loaded and ready");
                syncToolsetUi(engine.getActiveToolsetId());
            });
        });
    }

    private void runInference() {
        String toolsetId = toolsetIdEditText.getText().toString().trim();
        String userMessage = inputEditText.getText().toString().trim();
        if (toolsetId.isEmpty()) {
            updateStatus("Status: please enter or save a toolset first");
            return;
        }
        if (userMessage.isEmpty()) {
            updateStatus("Status: please enter a message first");
            return;
        }

        setBusy(true);
        updateStatus("Status: running inference...");
        clearOutputPanels();
        executor.execute(() -> {
            try {
                String output = engine.generate(toolsetId, userMessage, DEFAULT_MAX_NEW_TOKENS);
                final ToolCall parsedToolCall = ToolCallParser.parse(output);
                final ToolCall effectiveToolCall = normalizeToolCall(userMessage, parsedToolCall);
                final String toolSource = effectiveToolCall == null ? "none" : "model";
                logInferenceDebug(userMessage, output, effectiveToolCall, toolSource);
                mainHandler.post(() -> {
                    try {
                        ToolExecutionResult executionResult = null;
                        if (effectiveToolCall != null) {
                            executionResult = toolExecutor.execute(effectiveToolCall);
                        }
                        renderOutputSummary(output, effectiveToolCall, toolSource, executionResult);
                        if (effectiveToolCall != null && executionResult != null && executionResult.executed()) {
                            updateStatus("Status: inference finished and tool executed");
                        } else if (effectiveToolCall != null) {
                            updateStatus("Status: inference finished but tool was not executed");
                        } else {
                            updateStatus("Status: inference finished with no tool call");
                        }
                    } catch (Exception executionError) {
                        renderOutputSummary(
                                output,
                                effectiveToolCall,
                                toolSource,
                                new ToolExecutionResult(false, "Post-inference validation failed: " + executionError.getMessage())
                        );
                        updateStatus("Status: inference finished but validation failed");
                    } finally {
                        setBusy(false);
                    }
                });
            } catch (Exception error) {
                showError(error);
            }
        });
    }

    private ToolCall normalizeToolCall(String userMessage, @Nullable ToolCall toolCall) {
        if (toolCall == null || !"open_target".equals(toolCall.name())) {
            return toolCall;
        }
        String extractedTarget = extractOpenTarget(userMessage);
        if (extractedTarget == null) {
            return toolCall;
        }
        java.util.Map<String, Object> arguments = new java.util.LinkedHashMap<>(toolCall.arguments());
        arguments.put("target", extractedTarget);
        return new ToolCall(toolCall.name(), arguments);
    }

    @Nullable
    private String extractOpenTarget(String userMessage) {
        if (userMessage == null) {
            return null;
        }
        int marker = userMessage.lastIndexOf("打开");
        if (marker < 0) {
            return null;
        }
        String target = userMessage.substring(marker + 2).trim();
        return target.isEmpty() ? null : target;
    }

    private void saveToolset() {
        String toolsetId = toolsetIdEditText.getText().toString().trim();
        String displayName = toolsetNameEditText.getText().toString().trim();
        String systemPrompt = systemPromptEditText.getText().toString().trim();
        String toolsJson = toolsJsonEditText.getText().toString().trim();

        if (toolsetId.isEmpty()) {
            updateStatus("Status: please enter a toolset id");
            return;
        }

        executeBusyTask("Status: validating and caching toolset...", () -> {
            engine.registerOrReplaceToolset(new ToolsetDefinition(
                    toolsetId,
                    displayName,
                    systemPrompt,
                    UserToolsetTemplate.parseToolsJson(toolsJson)
            ));
            engine.activateToolset(toolsetId);
            finishBusyTask(() -> {
                syncToolsetUi(toolsetId);
                updateStatus("Status: toolset " + toolsetId + " saved and activated");
            });
        });
    }

    private void useSelectedToolset() {
        String toolsetId = requireSelectedToolsetId("Status: no saved toolset selected");
        if (toolsetId == null) {
            return;
        }
        executeBusyTask("Status: activating toolset " + toolsetId + "...", () -> {
            engine.activateToolset(toolsetId);
            finishBusyTask(() -> {
                syncToolsetUi(toolsetId);
                updateStatus("Status: activated toolset " + toolsetId);
            });
        });
    }

    private void resetTemplate() {
        toolsetNameEditText.setText("");
        systemPromptEditText.setText(UserToolsetTemplate.defaultSystemPrompt());
        toolsJsonEditText.setText(UserToolsetTemplate.defaultToolsJsonTemplate());
        updateStatus("Status: template reset");
    }

    private void saveAppAlias() {
        String alias = appAliasEditText.getText().toString().trim();
        Object selected = installedAppSpinner.getSelectedItem();
        if (alias.isEmpty()) {
            updateStatus("Status: please enter an app alias");
            return;
        }
        if (!(selected instanceof InstalledAppOption option)) {
            updateStatus("Status: please select an installed app");
            return;
        }

        AppAliasEntry entry = new AppAliasEntry(alias, option.packageName(), option.label());
        upsertUserAlias(entry);
        try {
            appAliasStore.writeAll(userAppAliases);
            rebuildAliasDependencies();
            refreshAppAliasSummary();
            appAliasEditText.setText("");
            updateStatus("Status: saved app alias \"" + alias + "\" -> " + option.label());
        } catch (Exception error) {
            updateStatus("Status: failed to save app alias - " + error.getMessage());
        }
    }

    private void deleteSelectedToolset() {
        String toolsetId = requireSelectedToolsetId("Status: no toolset selected to delete");
        if (toolsetId == null) {
            return;
        }
        executeBusyTask("Status: deleting toolset " + toolsetId + "...", () -> {
            boolean deleted = engine.deleteToolset(toolsetId);
            finishBusyTask(() -> {
                syncToolsetUi(engine.getActiveToolsetId());
                if (deleted) {
                    updateStatus("Status: deleted toolset " + toolsetId);
                } else {
                    updateStatus("Status: built-in toolset cannot be deleted");
                }
            });
        });
    }

    private void rebuildSelectedToolsetCache() {
        String toolsetId = requireSelectedToolsetId("Status: no toolset selected to rebuild");
        if (toolsetId == null) {
            return;
        }
        executeBusyTask("Status: rebuilding prefix KV for " + toolsetId + "...", () -> {
            engine.rebuildToolsetCache(toolsetId);
            finishBusyTask(() -> {
                syncToolsetUi(toolsetId);
                updateStatus("Status: rebuilt prefix KV for " + toolsetId);
            });
        });
    }

    private void showError(Exception error) {
        mainHandler.post(() -> {
            updateStatus("Status: error - " + error.getMessage());
            selectMainPage(PAGE_ASSISTANT);
            renderOutputPanels("", "", error.toString());
            selectOutputTab(OUTPUT_TAB_EXECUTION);
            setBusy(false);
        });
    }

    private void executeBusyTask(String status, BackgroundTask task) {
        setBusy(true);
        updateStatus(status);
        executor.execute(() -> {
            try {
                task.run();
            } catch (Exception error) {
                showError(error);
            }
        });
    }

    private void finishBusyTask(Runnable task) {
        mainHandler.post(() -> {
            task.run();
            setBusy(false);
        });
    }

    private void updateStatus(String status) {
        statusText.setText(status);
    }

    private void setBusy(boolean busy) {
        loadButton.setEnabled(!busy);
        saveToolsetButton.setEnabled(!busy);
        useSelectedToolsetButton.setEnabled(!busy);
        deleteToolsetButton.setEnabled(!busy);
        rebuildPrefixCacheButton.setEnabled(!busy);
        resetTemplateButton.setEnabled(!busy);
        saveAppAliasButton.setEnabled(!busy);
        runButton.setEnabled(!busy);
        inferenceProgress.setVisibility(busy ? View.VISIBLE : View.GONE);
        runButton.setText(busy ? R.string.button_running : R.string.button_run);
    }

    private void setStudioExpanded(boolean expanded) {
        studioContent.setVisibility(expanded ? View.VISIBLE : View.GONE);
        studioToggleText.setText(expanded ? R.string.studio_toggle_expanded : R.string.studio_toggle_collapsed);
    }

    private void selectMainPage(int page) {
        boolean showAssistant = page == PAGE_ASSISTANT;
        assistantPage.setVisibility(showAssistant ? View.VISIBLE : View.GONE);
        toolsPage.setVisibility(showAssistant ? View.GONE : View.VISIBLE);
        applyTabVisual(pageAssistantTab, showAssistant);
        applyTabVisual(pageToolsTab, !showAssistant);
    }

    private void refreshToolsetList() {
        List<String> toolsetIds = engine.getRegisteredToolsetIds();
        toolsetAdapter.clear();
        toolsetAdapter.addAll(toolsetIds);
        toolsetAdapter.notifyDataSetChanged();
        String activeToolsetId = engine.getActiveToolsetId();
        if (activeToolsetId != null) {
            for (int i = 0; i < toolsetAdapter.getCount(); i++) {
                if (activeToolsetId.equals(toolsetAdapter.getItem(i))) {
                    toolsetSpinner.setSelection(i);
                    break;
                }
            }
        }
        updateStudioSummary();
    }

    private void syncToolsetUi(String toolsetId) {
        refreshToolsetList();
        populateFieldsFromToolset(toolsetId);
        updateCacheStatus(engine.getActiveToolsetCacheStatus());
    }

    private String requireSelectedToolsetId(String emptyStatus) {
        Object selected = toolsetSpinner.getSelectedItem();
        if (selected == null) {
            updateStatus(emptyStatus);
            return null;
        }
        return selected.toString();
    }

    private void populateFieldsFromToolset(String toolsetId) {
        ToolsetDefinition toolset = engine.getToolset(toolsetId);
        if (toolset == null) {
            updateStudioSummary();
            return;
        }
        toolsetIdEditText.setText(toolset.id());
        toolsetNameEditText.setText(toolset.displayName());
        systemPromptEditText.setText(toolset.systemPrompt());
        toolsJsonEditText.setText(UserToolsetTemplate.toPrettyJson(toolset.tools()));
        updateStudioSummary();
    }

    private void updateCacheStatus(PrefixCacheStatus cacheStatus) {
        lastCacheStatus = cacheStatus;
        if (cacheStatus == null || !cacheStatus.persistedCacheAvailable()) {
            cacheStatusText.setText(getString(R.string.cache_status_idle));
            updateStudioSummary();
            return;
        }
        StringBuilder builder = new StringBuilder();
        builder.append("Toolset: ").append(cacheStatus.toolsetId()).append('\n');
        builder.append("Last activation: ")
                .append(cacheStatus.lastActivationCacheHit() ? "cache hit" : "rebuilt and persisted")
                .append('\n');
        builder.append("System seq length: ").append(cacheStatus.systemSequenceLength()).append('\n');
        builder.append("KV size: ").append(formatBytes(cacheStatus.kvFileSizeBytes())).append('\n');
        builder.append("Built at: ").append(timeFormat.format(new Date(cacheStatus.builtAtEpochMs())));
        cacheStatusText.setText(builder.toString());
        updateStudioSummary();
    }

    private static String formatBytes(long bytes) {
        if (bytes < 1024) {
            return bytes + " B";
        }
        if (bytes < 1024 * 1024) {
            return String.format(Locale.US, "%.1f KB", bytes / 1024.0);
        }
        return String.format(Locale.US, "%.2f MB", bytes / (1024.0 * 1024.0));
    }

    private void renderOutputSummary(
            String rawOutput,
            ToolCall toolCall,
            String toolSource,
            ToolExecutionResult executionResult) {
        selectMainPage(PAGE_ASSISTANT);
        String safeOutput = rawOutput == null || rawOutput.isBlank() ? "(empty)" : rawOutput.trim();
        StringBuilder parsedBuilder = new StringBuilder();
        if (toolCall == null) {
            parsedBuilder.append("None").append('\n');
        } else {
            parsedBuilder.append("name=").append(toolCall.name()).append('\n');
            parsedBuilder.append("arguments=").append(toolCall.arguments()).append('\n');
        }
        parsedBuilder.append("source=").append(toolSource == null ? "none" : toolSource);

        StringBuilder executionBuilder = new StringBuilder();
        if (executionResult == null) {
            executionBuilder.append("No execution result");
        } else {
            executionBuilder.append(executionResult.executed() ? "Executed" : "Not executed")
                    .append(": ")
                    .append(executionResult.summary());
        }
        renderOutputPanels(safeOutput, parsedBuilder.toString(), executionBuilder.toString());
        selectBestOutputTab(toolCall, executionResult);
    }

    private void renderOutputPanels(String modelOutput, String parsedToolOutput, String executionOutput) {
        setPanelText(modelOutputText, modelOutput);
        setPanelText(parsedToolText, parsedToolOutput);
        setPanelText(executionOutputText, executionOutput);
    }

    private void clearOutputPanels() {
        renderOutputPanels("", "", "");
    }

    private void selectOutputTab(int tab) {
        boolean showModel = tab == OUTPUT_TAB_MODEL;
        boolean showParsed = tab == OUTPUT_TAB_PARSED;
        boolean showExecution = tab == OUTPUT_TAB_EXECUTION;

        modelOutputText.setVisibility(showModel ? View.VISIBLE : View.GONE);
        parsedToolText.setVisibility(showParsed ? View.VISIBLE : View.GONE);
        executionOutputText.setVisibility(showExecution ? View.VISIBLE : View.GONE);

        applyTabVisual(tabModelOutput, showModel);
        applyTabVisual(tabParsedTool, showParsed);
        applyTabVisual(tabExecutionResult, showExecution);
    }

    private void selectBestOutputTab(@Nullable ToolCall toolCall, @Nullable ToolExecutionResult executionResult) {
        selectOutputTab(executionResult != null
                ? OUTPUT_TAB_EXECUTION
                : toolCall != null ? OUTPUT_TAB_PARSED : OUTPUT_TAB_MODEL);
    }

    private void applyTabVisual(TextView tabView, boolean selected) {
        tabView.setBackgroundResource(selected ? R.drawable.bg_tab_active : R.drawable.bg_tab_inactive);
        tabView.setTextColor(getColor(selected ? R.color.fg_primary : R.color.fg_secondary));
    }

    private static void setPanelText(TextView textView, String value) {
        textView.setText(value == null || value.isBlank() ? "(empty)" : value);
    }

    private void updateStudioSummary() {
        String activeToolsetId = engine.getActiveToolsetId();
        ToolsetDefinition activeToolset = activeToolsetId == null ? null : engine.getToolset(activeToolsetId);
        if (activeToolset == null) {
            studioSummaryTitleText.setText(R.string.studio_summary_empty_title);
            studioSummaryMetaText.setText(R.string.studio_summary_empty_meta);
            return;
        }

        studioSummaryTitleText.setText(activeToolset.displayName() + "  ·  " + activeToolset.id());

        String cacheLine;
        if (lastCacheStatus == null || !lastCacheStatus.persistedCacheAvailable()) {
            cacheLine = "Prefix KV not prepared";
        } else if (activeToolset.id().equals(lastCacheStatus.toolsetId())) {
            cacheLine = lastCacheStatus.lastActivationCacheHit()
                    ? "Prefix KV ready · cache hit on last activation"
                    : "Prefix KV ready · rebuilt on last activation";
        } else {
            cacheLine = "Prefix KV available for " + lastCacheStatus.toolsetId();
        }

        String summary = "Tools: " + activeToolset.tools().size()
                + " · Prompt chars: " + activeToolset.systemPrompt().length()
                + "\n" + cacheLine;
        studioSummaryMetaText.setText(summary);
    }

    private void loadPersistedAppAliases() {
        userAppAliases.clear();
        try {
            userAppAliases.addAll(appAliasStore.readAll());
        } catch (Exception error) {
            Log.w(TAG, "Failed to load app aliases", error);
        }
    }

    private void rebuildAliasDependencies() {
        toolExecutor = new ToolExecutor(this, new AppAliasResolver(userAppAliases));
    }

    private void refreshInstalledApps() {
        installedAppAdapter.clear();
        installedAppAdapter.addAll(queryInstalledApps());
        installedAppAdapter.notifyDataSetChanged();
    }

    private List<InstalledAppOption> queryInstalledApps() {
        List<InstalledAppOption> result = new ArrayList<>();
        var launcherIntent = new android.content.Intent(android.content.Intent.ACTION_MAIN);
        launcherIntent.addCategory(android.content.Intent.CATEGORY_LAUNCHER);
        var packageManager = getPackageManager();
        var resolveInfos = packageManager.queryIntentActivities(launcherIntent, 0);
        for (var resolveInfo : resolveInfos) {
            if (resolveInfo.activityInfo == null || resolveInfo.activityInfo.packageName == null) {
                continue;
            }
            CharSequence label = resolveInfo.loadLabel(packageManager);
            String safeLabel = label == null ? resolveInfo.activityInfo.packageName : label.toString();
            result.add(new InstalledAppOption(safeLabel, resolveInfo.activityInfo.packageName));
        }
        result.sort(Comparator.comparing(InstalledAppOption::label, String.CASE_INSENSITIVE_ORDER));
        return result;
    }

    private void refreshAppAliasSummary() {
        if (userAppAliases.isEmpty()) {
            appAliasSummaryText.setText(R.string.app_alias_summary_empty);
            return;
        }
        StringBuilder builder = new StringBuilder();
        int count = Math.min(userAppAliases.size(), 8);
        for (int i = 0; i < count; i++) {
            AppAliasEntry entry = userAppAliases.get(i);
            builder.append(entry.alias())
                    .append(" -> ")
                    .append(entry.appLabel() == null || entry.appLabel().isBlank() ? entry.packageName() : entry.appLabel())
                    .append('\n');
        }
        if (userAppAliases.size() > count) {
            builder.append("... ").append(userAppAliases.size() - count).append(" more");
        } else if (builder.length() > 0) {
            builder.setLength(builder.length() - 1);
        }
        appAliasSummaryText.setText(builder.toString());
    }

    private void upsertUserAlias(AppAliasEntry newEntry) {
        String normalizedAlias = normalizeAlias(newEntry.alias());
        for (int i = 0; i < userAppAliases.size(); i++) {
            AppAliasEntry existing = userAppAliases.get(i);
            if (normalizeAlias(existing.alias()).equals(normalizedAlias)) {
                userAppAliases.set(i, newEntry);
                return;
            }
        }
        userAppAliases.add(0, newEntry);
    }

    private static String normalizeAlias(String value) {
        return value == null ? "" : value.trim().replace(" ", "").toLowerCase(Locale.ROOT);
    }

    private static void logInferenceDebug(
            String userMessage,
            String rawOutput,
            ToolCall toolCall,
            String toolSource) {
        Log.d(TAG, "User message: " + (userMessage == null ? "(null)" : userMessage));
        Log.d(TAG, "Parsed tool source: " + (toolSource == null ? "none" : toolSource));
        if (toolCall == null) {
            Log.d(TAG, "Parsed tool call: none");
        } else {
            Log.d(TAG, "Parsed tool call name: " + toolCall.name());
            Log.d(TAG, "Parsed tool call arguments: " + toolCall.arguments());
        }

        String safeOutput = rawOutput == null ? "(null)" : rawOutput;
        if (safeOutput.isEmpty()) {
            Log.d(TAG, "Model output: (empty)");
            return;
        }
        for (int start = 0; start < safeOutput.length(); start += LOG_CHUNK_SIZE) {
            int end = Math.min(start + LOG_CHUNK_SIZE, safeOutput.length());
            Log.d(TAG, "Model output [" + start + "," + end + "]: " + safeOutput.substring(start, end));
        }
    }
}
