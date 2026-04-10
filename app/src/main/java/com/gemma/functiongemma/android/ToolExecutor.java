package com.gemma.functiongemma.android;

import android.app.Activity;
import android.content.ActivityNotFoundException;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.pm.ResolveInfo;
import android.net.Uri;
import android.provider.Settings;

import java.util.List;
import java.util.Locale;
import java.util.Map;

final class ToolExecutor {
    private final Context appContext;
    private final Activity activity;
    private final PackageManager packageManager;
    private final AppAliasResolver appAliasResolver;

    ToolExecutor(Activity activity, AppAliasResolver appAliasResolver) {
        this.activity = activity;
        this.appContext = activity.getApplicationContext();
        this.packageManager = appContext.getPackageManager();
        this.appAliasResolver = appAliasResolver;
    }

    ToolExecutionResult execute(ToolCall toolCall) {
        if (toolCall == null) {
            return new ToolExecutionResult(false, "No tool call to execute");
        }
        return switch (toolCall.name()) {
            case "open_target" -> openTarget(toolCall.arguments());
            case "launch_app" -> launchApp(toolCall.arguments());
            case "navigate" -> navigate(toolCall.arguments());
            case "play_music" -> playMusic(toolCall.arguments());
            case "web_search" -> webSearch(toolCall.arguments());
            case "open_system_panel", "open_settings" -> openSystemPanel(toolCall.arguments());
            default -> new ToolExecutionResult(false, "Unsupported tool: " + toolCall.name());
        };
    }

    private ToolExecutionResult launchApp(Map<String, Object> arguments) {
        String appName = readString(arguments, "app_name");
        if (appName == null) {
            return new ToolExecutionResult(false, "launch_app requires app_name");
        }
        return launchAppByName(appName, "launch_app requires app_name");
    }

    private ToolExecutionResult openTarget(Map<String, Object> arguments) {
        String target = readString(arguments, "target");
        if (target == null || target.isBlank()) {
            return new ToolExecutionResult(false, "open_target requires target");
        }
        String settingsTarget = normalizeSettingsTarget(target);
        if (settingsTarget != null) {
            return openSystemPanelByName(settingsTarget, "open_target requires target");
        }
        return launchAppByName(target, "open_target requires target");
    }

    private ToolExecutionResult launchAppByName(String appName, String missingMessage) {
        if (appName == null || appName.isBlank()) {
            return new ToolExecutionResult(false, missingMessage);
        }

        String packageName = appAliasResolver.resolvePackage(appName);
        if (packageName == null) {
            packageName = findInstalledLaunchablePackage(appName);
        }
        if (packageName == null) {
            return new ToolExecutionResult(false, "No package mapping found for " + appName);
        }

        Intent launchIntent = createLaunchIntent(packageName);
        if (launchIntent == null) {
            return new ToolExecutionResult(false, "App is not installed: " + appName + " (" + packageName + ")");
        }
        return startActivitySafely(launchIntent, "Opened " + appName, "Unable to open " + appName);
    }

    private String findInstalledLaunchablePackage(String appName) {
        String normalizedTarget = normalizeAppName(appName);
        if (normalizedTarget.isEmpty()) {
            return null;
        }

        Intent launcherIntent = new Intent(Intent.ACTION_MAIN);
        launcherIntent.addCategory(Intent.CATEGORY_LAUNCHER);
        List<ResolveInfo> candidates = packageManager.queryIntentActivities(launcherIntent, 0);

        String containsMatch = null;
        for (ResolveInfo candidate : candidates) {
            CharSequence label = candidate.loadLabel(packageManager);
            String normalizedLabel = normalizeAppName(label == null ? "" : label.toString());
            String packageName = candidate.activityInfo == null ? null : candidate.activityInfo.packageName;
            if (packageName == null || packageName.isBlank()) {
                continue;
            }
            if (normalizedTarget.equals(normalizedLabel)) {
                return packageName;
            }
            if (containsMatch == null
                    && (normalizedLabel.contains(normalizedTarget) || normalizedTarget.contains(normalizedLabel))) {
                containsMatch = packageName;
            }
        }
        return containsMatch;
    }

    private ToolExecutionResult playMusic(Map<String, Object> arguments) {
        String song = readString(arguments, "song");
        String artist = readString(arguments, "artist");
        String appName = readString(arguments, "app_name");

        StringBuilder queryBuilder = new StringBuilder();
        if (artist != null && !artist.isEmpty()) {
            queryBuilder.append(artist).append(' ');
        }
        if (song != null && !song.isEmpty()) {
            queryBuilder.append(song);
        }
        String query = queryBuilder.toString().trim();
        if (query.isEmpty()) {
            return new ToolExecutionResult(false, "play_music requires song or artist");
        }

        String packageName = appAliasResolver.resolvePackage(appName != null ? appName : "音乐");
        if (packageName != null) {
            Intent appIntent = createLaunchIntent(packageName);
            if (appIntent != null) {
                return startActivitySafely(
                        appIntent,
                        "Opened music app for " + query,
                        "Unable to open music app for " + query
                );
            }
        }

        return openWebSearch(
                query,
                "Searched music for " + query,
                "No browser available to search music for " + query
        );
    }

    private ToolExecutionResult navigate(Map<String, Object> arguments) {
        String destination = readString(arguments, "destination");
        if (destination == null || destination.isBlank()) {
            return new ToolExecutionResult(false, "navigate requires destination");
        }

        String mode = readString(arguments, "mode");
        String routeType = switch (mode == null ? "" : mode.trim().toLowerCase(Locale.ROOT)) {
            case "walking", "walk" -> "2";
            case "transit", "bus", "public_transit" -> "1";
            default -> "0";
        };

        Uri uri = Uri.parse(
                "androidamap://route?sourceApplication=FunctionGemma"
                        + "&dname=" + Uri.encode(destination)
                        + "&dev=0"
                        + "&t=" + routeType
        );
        Intent intent = new Intent(Intent.ACTION_VIEW, uri);
        intent.setPackage("com.autonavi.minimap");
        intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
        return startActivitySafely(intent, "Started navigation to " + destination, "Unable to navigate to " + destination);
    }

    private ToolExecutionResult webSearch(Map<String, Object> arguments) {
        String query = readString(arguments, "query");
        if (query == null || query.isBlank()) {
            return new ToolExecutionResult(false, "web_search requires query");
        }
        return openWebSearch(
                query,
                "Opened web search for " + query,
                "No browser available to search for " + query
        );
    }

    private ToolExecutionResult openSystemPanel(Map<String, Object> arguments) {
        String panel = readString(arguments, "panel");
        return openSystemPanelByName(panel, "open_system_panel requires panel");
    }

    private ToolExecutionResult openSystemPanelByName(String panel, String missingMessage) {
        if (panel == null || panel.isBlank()) {
            return new ToolExecutionResult(false, missingMessage);
        }
        String normalized = normalizeSettingsTarget(panel);
        if (normalized == null) {
            return new ToolExecutionResult(false, "Unsupported system panel: " + panel);
        }
        Intent intent;
        switch (normalized) {
            case "bluetooth":
                intent = new Intent(Settings.ACTION_BLUETOOTH_SETTINGS);
                break;
            case "wifi":
                intent = new Intent(Settings.Panel.ACTION_WIFI);
                break;
            case "internet":
                intent = new Intent(Settings.Panel.ACTION_INTERNET_CONNECTIVITY);
                break;
            case "settings":
                intent = new Intent(Settings.ACTION_SETTINGS);
                break;
            default:
                return new ToolExecutionResult(false, "Unsupported system panel: " + panel);
        }
        intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
        return startActivitySafely(intent, "Opened panel: " + panel, "Unable to open panel: " + panel);
    }

    private static String readString(Map<String, Object> arguments, String key) {
        Object value = arguments.get(key);
        if (value == null) {
            return null;
        }
        return String.valueOf(value);
    }

    private static String normalizeAppName(String value) {
        if (value == null) {
            return "";
        }
        return value.trim().replace(" ", "").toLowerCase(Locale.ROOT);
    }

    private static String normalizeSettingsTarget(String value) {
        String normalized = normalizeAppName(value);
        return switch (normalized) {
            case "settings", "设置", "系统设置" -> "settings";
            case "wifi", "wi-fi", "wlan", "无线网络" -> "wifi";
            case "bluetooth", "蓝牙" -> "bluetooth";
            case "internet", "网络", "互联网" -> "internet";
            default -> null;
        };
    }

    private ToolExecutionResult openWebSearch(String query, String successMessage, String failureMessage) {
        Intent intent = new Intent(Intent.ACTION_VIEW, Uri.parse("https://www.google.com/search?q=" + Uri.encode(query)));
        intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
        return startActivitySafely(intent, successMessage, failureMessage);
    }

    private Intent createLaunchIntent(String packageName) {
        Intent launchIntent = packageManager.getLaunchIntentForPackage(packageName);
        if (launchIntent == null) {
            return null;
        }
        launchIntent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
        return launchIntent;
    }

    private ToolExecutionResult startActivitySafely(Intent intent, String successMessage, String failureMessage) {
        if (intent.resolveActivity(packageManager) == null) {
            return new ToolExecutionResult(false, failureMessage);
        }
        try {
            activity.startActivity(intent);
            return new ToolExecutionResult(true, successMessage);
        } catch (ActivityNotFoundException | SecurityException error) {
            return new ToolExecutionResult(false, failureMessage + " (" + error.getClass().getSimpleName() + ")");
        }
    }
}
