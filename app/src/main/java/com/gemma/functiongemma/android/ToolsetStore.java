package com.gemma.functiongemma.android;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

final class ToolsetStore {
    private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();
    private final File file;

    ToolsetStore(File directory) {
        this.file = new File(directory, "toolsets.json");
    }

    synchronized List<ToolsetDefinition> readAll() throws IOException {
        if (!file.exists()) {
            return new ArrayList<>();
        }
        try (FileReader reader = new FileReader(file)) {
            ToolsetRecord[] records = GSON.fromJson(reader, ToolsetRecord[].class);
            List<ToolsetDefinition> result = new ArrayList<>();
            if (records != null) {
                for (ToolsetRecord record : records) {
                    if (record != null && record.id != null) {
                        result.add(new ToolsetDefinition(
                                record.id,
                                record.displayName,
                                record.systemPrompt,
                                record.tools
                        ));
                    }
                }
            }
            return result;
        }
    }

    synchronized void writeAll(List<ToolsetDefinition> toolsets) throws IOException {
        File parent = file.getParentFile();
        if (parent != null && !parent.exists() && !parent.mkdirs()) {
            throw new IOException("Failed to create toolset directory: " + parent);
        }

        List<ToolsetRecord> records = new ArrayList<>(toolsets.size());
        for (ToolsetDefinition toolset : toolsets) {
            ToolsetRecord record = new ToolsetRecord();
            record.id = toolset.id();
            record.displayName = toolset.displayName();
            record.systemPrompt = toolset.systemPrompt();
            record.tools = toolset.tools();
            records.add(record);
        }

        try (FileWriter writer = new FileWriter(file, false)) {
            GSON.toJson(records, writer);
        }
    }

    private static final class ToolsetRecord {
        String id;
        String displayName;
        String systemPrompt;
        List<Map<String, Object>> tools;
    }
}
