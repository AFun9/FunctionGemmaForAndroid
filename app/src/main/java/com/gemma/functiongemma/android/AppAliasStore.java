package com.gemma.functiongemma.android;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

final class AppAliasStore {
    private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();
    private final File file;

    AppAliasStore(File directory) {
        this.file = new File(directory, "app_aliases.json");
    }

    synchronized List<AppAliasEntry> readAll() throws IOException {
        if (!file.exists()) {
            return new ArrayList<>();
        }
        try (FileReader reader = new FileReader(file)) {
            AppAliasEntry[] entries = GSON.fromJson(reader, AppAliasEntry[].class);
            List<AppAliasEntry> result = new ArrayList<>();
            if (entries != null) {
                for (AppAliasEntry entry : entries) {
                    if (entry != null
                            && entry.alias() != null
                            && !entry.alias().isBlank()
                            && entry.packageName() != null
                            && !entry.packageName().isBlank()) {
                        result.add(entry);
                    }
                }
            }
            return result;
        }
    }

    synchronized void writeAll(List<AppAliasEntry> entries) throws IOException {
        File parent = file.getParentFile();
        if (parent != null && !parent.exists() && !parent.mkdirs()) {
            throw new IOException("Failed to create app alias directory: " + parent);
        }
        try (FileWriter writer = new FileWriter(file, false)) {
            GSON.toJson(entries, writer);
        }
    }
}
