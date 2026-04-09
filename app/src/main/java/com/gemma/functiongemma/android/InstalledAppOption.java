package com.gemma.functiongemma.android;

final class InstalledAppOption {
    private final String label;
    private final String packageName;

    InstalledAppOption(String label, String packageName) {
        this.label = label;
        this.packageName = packageName;
    }

    String label() {
        return label;
    }

    String packageName() {
        return packageName;
    }

    @Override
    public String toString() {
        return label + " · " + packageName;
    }
}
