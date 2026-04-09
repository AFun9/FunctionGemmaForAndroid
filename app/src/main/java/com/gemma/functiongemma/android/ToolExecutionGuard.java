package com.gemma.functiongemma.android;

import java.util.Locale;

final class ToolExecutionGuard {
    private final AppAliasResolver appAliasResolver;

    ToolExecutionGuard(AppAliasResolver appAliasResolver) {
        this.appAliasResolver = appAliasResolver;
    }

    ToolExecutionResult validate(String userMessage, ToolCall toolCall) {
        if (toolCall == null) {
            return new ToolExecutionResult(false, "No executable tool call parsed");
        }

        String normalizedMessage = normalize(userMessage);
        boolean musicIntent = isMusicIntent(normalizedMessage);
        boolean appLaunchIntent = isExplicitLaunchIntent(normalizedMessage);
        boolean navigationIntent = isNavigationIntent(normalizedMessage);
        boolean systemPanelIntent = isSystemPanelIntent(normalizedMessage);

        if (musicIntent && "launch_app".equals(toolCall.name())) {
            String appName = stringArgument(toolCall, "app_name");
            if (appName == null) {
                return new ToolExecutionResult(false, "Blocked launch_app because app_name is missing");
            }
            if (!appAliasResolver.isMusicAppName(appName)) {
                return new ToolExecutionResult(
                        false,
                        "Blocked launch_app for non-music app \"" + appName + "\" because the request looks like music playback"
                );
            }
        }

        if (appLaunchIntent && "play_music".equals(toolCall.name()) && !musicIntent) {
            return new ToolExecutionResult(
                    false,
                    "Blocked play_music because the request looks like opening an app, not playing music"
            );
        }

        if (navigationIntent && "launch_app".equals(toolCall.name())) {
            return new ToolExecutionResult(
                    false,
                    "Blocked launch_app because the request looks like navigation"
            );
        }

        if ("open_system_panel".equals(toolCall.name()) && appLaunchIntent && !systemPanelIntent) {
            return new ToolExecutionResult(
                    false,
                    "Blocked open_system_panel because the request looks like opening an app, not a system panel"
            );
        }

        return new ToolExecutionResult(true, "Tool call passed validation");
    }

    private static boolean isMusicIntent(String normalizedMessage) {
        return containsAny(
                normalizedMessage,
                "听", "音乐", "歌曲", "歌", "播放", "来一首", "唱", "专辑",
                "listen", "music", "song", "play"
        );
    }

    private static boolean isExplicitLaunchIntent(String normalizedMessage) {
        return containsAny(
                normalizedMessage,
                "打开", "启动", "进入", "点开",
                "open", "launch", "start"
        );
    }

    private static boolean isNavigationIntent(String normalizedMessage) {
        return containsAny(
                normalizedMessage,
                "导航", "去", "到", "前往", "路线", "怎么走", "带我去",
                "navigate", "navigation", "route", "directions", "take me to"
        );
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

    private static String stringArgument(ToolCall toolCall, String key) {
        Object value = toolCall.arguments().get(key);
        if (value == null) {
            return null;
        }
        return String.valueOf(value);
    }

    private static String normalize(String value) {
        if (value == null) {
            return "";
        }
        return value.trim().toLowerCase(Locale.ROOT);
    }
}
