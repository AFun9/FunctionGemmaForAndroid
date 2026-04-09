package com.gemma.functiongemma.android;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.util.List;

import static org.junit.Assert.assertEquals;

public class ToolsetStoreTest {
    @Rule
    public TemporaryFolder temporaryFolder = new TemporaryFolder();

    @Test
    public void writesAndReadsToolsets() throws Exception {
        ToolsetStore store = new ToolsetStore(temporaryFolder.newFolder("toolsets"));
        ToolsetDefinition toolset = new ToolsetDefinition(
                "demo",
                "Demo",
                "system prompt",
                UserToolsetTemplate.parseToolsJson(UserToolsetTemplate.defaultToolsJsonTemplate())
        );

        store.writeAll(List.of(toolset));
        List<ToolsetDefinition> loaded = store.readAll();

        assertEquals(1, loaded.size());
        assertEquals("demo", loaded.get(0).id());
        assertEquals("Demo", loaded.get(0).displayName());
        assertEquals("system prompt", loaded.get(0).systemPrompt());
    }
}
