package com.gemma.functiongemma.android;

public final class AppAliasEntry {
    private final String alias;
    private final String packageName;
    private final String appLabel;

    public AppAliasEntry(String alias, String packageName, String appLabel) {
        this.alias = alias;
        this.packageName = packageName;
        this.appLabel = appLabel;
    }

    public String alias() {
        return alias;
    }

    public String packageName() {
        return packageName;
    }

    public String appLabel() {
        return appLabel;
    }
}
