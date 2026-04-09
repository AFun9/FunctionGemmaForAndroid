package com.gemma.functiongemma.android;

import androidx.annotation.Nullable;

import java.util.Locale;
import java.util.Map;

final class ToolCallFallbackResolver {
    private final AppAliasResolver appAliasResolver;

    ToolCallFallbackResolver(AppAliasResolver appAliasResolver) {
        this.appAliasResolver = appAliasResolver;
    }

    ToolCallResolution resolve(String userMessage, @Nullable ToolCall parsedToolCall) {
        ToolCall corrected = correctFromUserMessage(userMessage, parsedToolCall);
        if (corrected != null) {
            return new ToolCallResolution(corrected, "fallback");
        }
        if (parsedToolCall != null) {
            return new ToolCallResolution(parsedToolCall, "model");
        }
        return new ToolCallResolution(null, "none");
    }

    @Nullable
    private ToolCall correctFromUserMessage(String userMessage, @Nullable ToolCall parsedToolCall) {
        String normalizedMessage = normalize(userMessage);
        boolean appLaunchIntent = isExplicitLaunchIntent(normalizedMessage);
        boolean systemPanelIntent = isSystemPanelIntent(normalizedMessage);
        boolean navigationIntent = isNavigationIntent(normalizedMessage);

        if (navigationIntent) {
            String destination = extractDestination(userMessage);
            if (destination != null && (parsedToolCall == null || missingArgument(parsedToolCall, "destination"))) {
                return new ToolCall("navigate", Map.of("destination", destination));
            }
        }

        if (appLaunchIntent && !systemPanelIntent) {
            String appName = inferAppName(userMessage);
            if (appName != null && shouldRewriteAsLaunchApp(parsedToolCall)) {
                return new ToolCall("launch_app", Map.of("app_name", appName));
            }
        }

        if (systemPanelIntent && (parsedToolCall == null || missingArgument(parsedToolCall, "panel"))) {
            String panel = inferSystemPanel(normalizedMessage);
            if (panel != null) {
                return new ToolCall("open_system_panel", Map.of("panel", panel));
            }
        }

        return null;
    }

    private boolean shouldRewriteAsLaunchApp(@Nullable ToolCall parsedToolCall) {
        if (parsedToolCall == null) {
            return true;
        }
        if ("open_system_panel".equals(parsedToolCall.name())) {
            return true;
        }
        if (!"launch_app".equals(parsedToolCall.name())) {
            return false;
        }
        String appName = stringArgument(parsedToolCall, "app_name");
        return appName == null || isSuspiciousAppName(appName);
    }

    @Nullable
    private String inferAppName(String userMessage) {
        String alias = appAliasResolver.findKnownAliasInText(userMessage);
        if (alias != null) {
            return alias;
        }

        String candidate = normalizeWhitespace(userMessage)
                .replaceFirst("^(帮我|请|麻烦你)?\\s*(打开|启动|进入|点开|运行)\\s*", "")
                .replaceFirst("^(open|launch|start)\\s+", "")
                .replaceFirst("\\s*(应用|软件|app)\\s*$", "")
                .replaceFirst("\\s*(一下|下)\\s*$", "")
                .trim();
        if (candidate.isEmpty()) {
            return null;
        }
        String normalizedCandidate = normalize(candidate);
        if (normalizedCandidate.isEmpty()) {
            return null;
        }
        if (isGenericAppReference(normalizedCandidate) || isSystemPanelIntent(normalizedCandidate)) {
            return null;
        }
        return candidate;
    }

    @Nullable
    private static String extractDestination(String userMessage) {
        String candidate = normalizeWhitespace(userMessage)
                .replaceFirst("^(帮我|请|麻烦你)?\\s*(导航到|导航去|带我去|前往|去|到)\\s*", "")
                .replaceFirst("^(navigate to|take me to|go to)\\s+", "")
                .trim();
        return candidate.isEmpty() ? null : candidate;
    }

    @Nullable
    private static String inferSystemPanel(String normalizedMessage) {
        if (containsAny(normalizedMessage, "wifi", "wi-fi", "wlan")) {
            return "wifi";
        }
        if (containsAny(normalizedMessage, "bluetooth", "蓝牙")) {
            return "bluetooth";
        }
        if (containsAny(normalizedMessage, "internet", "互联网", "网络")) {
            return "internet";
        }
        if (containsAny(normalizedMessage, "settings", "设置", "系统设置")) {
            return "settings";
        }
        return null;
    }

    private static boolean missingArgument(ToolCall toolCall, String key) {
        String value = stringArgument(toolCall, key);
        return value == null || value.isBlank();
    }

    @Nullable
    private static String stringArgument(ToolCall toolCall, String key) {
        Object value = toolCall.arguments().get(key);
        if (value == null) {
            return null;
        }
        return String.valueOf(value);
    }

    private boolean isSuspiciousAppName(String appName) {
        String normalized = normalize(appName);
        if (normalized.isEmpty()) {
            return true;
        }
        if (appAliasResolver.resolvePackage(appName) != null) {
            return false;
        }
        return normalized.equals("app")
                || normalized.equals("应用")
                || normalized.equals("软件")
                || normalized.endsWith("todoapp")
                || normalized.contains("mytodoapp")
                || normalized.contains("example")
                || normalized.contains("sample")
                || normalized.contains("demo")
                || normalized.contains("testapp");
    }

    private static boolean isGenericAppReference(String normalizedCandidate) {
        return normalizedCandidate.equals("app")
                || normalizedCandidate.equals("应用")
                || normalizedCandidate.equals("软件")
                || normalizedCandidate.equals("某个软件");
    }

    private static boolean isExplicitLaunchIntent(String normalizedMessage) {
        return containsAny(normalizedMessage, "打开", "启动", "进入", "点开", "open", "launch", "start");
    }

    private static boolean isNavigationIntent(String normalizedMessage) {
        return containsAny(normalizedMessage, "导航", "前往", "带我去", "navigate", "route", "directions");
    }

    private static boolean isSystemPanelIntent(String normalizedMessage) {
        return containsAny(
                normalizedMessage,
                "wifi", "wi-fi", "wlan", "蓝牙", "网络", "互联网", "设置", "系统设置", "控制中心",
                "bluetooth", "internet", "settings"
        );
    }

    private static boolean containsAny(String input, String... keywords) {
        for (String keyword : keywords) {
            if (input.contains(keyword)) {
                return true;
            }
        }
        return false;
    }

    private static String normalizeWhitespace(String value) {
        return value == null ? "" : value.trim().replaceAll("\\s+", " ");
    }

    private static String normalize(String value) {
        return value == null
                ? ""
                : value.trim()
                        .replace(" ", "")
                        .replace("_", "")
                        .replace("-", "")
                        .toLowerCase(Locale.ROOT);
    }
}
