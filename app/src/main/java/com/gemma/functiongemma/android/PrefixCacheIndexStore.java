package com.gemma.functiongemma.android;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

final class PrefixCacheIndexStore {
    private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();
    private final File indexFile;

    PrefixCacheIndexStore(File cacheDir) {
        this.indexFile = new File(cacheDir, "index.json");
    }

    synchronized PrefixCacheEntry find(String toolsetId, String fingerprint) throws IOException {
        for (PrefixCacheEntry entry : readAll()) {
            if (fingerprint.equals(entry.fingerprint) && toolsetId.equals(entry.toolsetId)) {
                return entry;
            }
        }
        return null;
    }

    synchronized void upsert(PrefixCacheEntry entry) throws IOException {
        List<PrefixCacheEntry> entries = readAll();
        for (Iterator<PrefixCacheEntry> iterator = entries.iterator(); iterator.hasNext(); ) {
            PrefixCacheEntry existing = iterator.next();
            if (entry.toolsetId.equals(existing.toolsetId)) {
                iterator.remove();
            }
        }
        entries.add(entry);
        writeAll(entries);
    }

    synchronized PrefixCacheEntry remove(String toolsetId) throws IOException {
        List<PrefixCacheEntry> entries = readAll();
        PrefixCacheEntry removed = null;
        for (Iterator<PrefixCacheEntry> iterator = entries.iterator(); iterator.hasNext(); ) {
            PrefixCacheEntry existing = iterator.next();
            if (toolsetId.equals(existing.toolsetId)) {
                removed = existing;
                iterator.remove();
            }
        }
        writeAll(entries);
        return removed;
    }

    private List<PrefixCacheEntry> readAll() throws IOException {
        if (!indexFile.exists()) {
            return new ArrayList<>();
        }
        try (FileReader reader = new FileReader(indexFile)) {
            PrefixCacheEntry[] entries = GSON.fromJson(reader, PrefixCacheEntry[].class);
            List<PrefixCacheEntry> result = new ArrayList<>();
            if (entries != null) {
                for (PrefixCacheEntry entry : entries) {
                    if (entry != null) {
                        result.add(entry);
                    }
                }
            }
            return result;
        }
    }

    private void writeAll(List<PrefixCacheEntry> entries) throws IOException {
        File parent = indexFile.getParentFile();
        if (parent != null && !parent.exists() && !parent.mkdirs()) {
            throw new IOException("Failed to create cache index directory: " + parent);
        }
        try (FileWriter writer = new FileWriter(indexFile, false)) {
            GSON.toJson(entries, writer);
        }
    }
}
