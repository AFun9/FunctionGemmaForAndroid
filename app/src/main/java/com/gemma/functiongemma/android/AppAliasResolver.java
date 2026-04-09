package com.gemma.functiongemma.android;

import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

final class AppAliasResolver {
    private final Map<String, String> packageByAlias = new LinkedHashMap<>();
    private final Set<String> musicPackages = new HashSet<>();

    AppAliasResolver() {
        register("微信", "com.tencent.mm");
        register("wechat", "com.tencent.mm");
        register("高德", "com.autonavi.minimap");
        register("高德地图", "com.autonavi.minimap");
        register("amap", "com.autonavi.minimap");
        registerMusicApp("qq音乐", "com.tencent.qqmusic");
        registerMusicApp("qq music", "com.tencent.qqmusic");
        registerMusicApp("网易云", "com.netease.cloudmusic");
        registerMusicApp("网易云音乐", "com.netease.cloudmusic");
        registerMusicApp("netease music", "com.netease.cloudmusic");
        registerMusicApp("spotify", "com.spotify.music");
        registerMusicApp("youtube music", "com.google.android.apps.youtube.music");
        register("chrome", "com.android.chrome");
        register("浏览器", "com.android.chrome");
        registerMusicApp("music", "com.tencent.qqmusic");
        registerMusicApp("音乐", "com.tencent.qqmusic");
    }

    void register(String alias, String packageName) {
        packageByAlias.put(normalize(alias), packageName);
    }

    void registerMusicApp(String alias, String packageName) {
        register(alias, packageName);
        musicPackages.add(packageName);
    }

    String resolvePackage(String appName) {
        if (appName == null || appName.trim().isEmpty()) {
            return null;
        }
        return packageByAlias.get(normalize(appName));
    }

    boolean isMusicAppName(String appName) {
        String packageName = resolvePackage(appName);
        return packageName != null && musicPackages.contains(packageName);
    }

    String findKnownAliasInText(String text) {
        if (text == null || text.trim().isEmpty()) {
            return null;
        }
        String normalizedText = normalize(text);
        String bestAlias = null;
        for (String alias : packageByAlias.keySet()) {
            if (!normalizedText.contains(alias)) {
                continue;
            }
            if (bestAlias == null || alias.length() > bestAlias.length()) {
                bestAlias = alias;
            }
        }
        return bestAlias;
    }

    private static String normalize(String value) {
        return value.trim().toLowerCase(Locale.ROOT);
    }
}
