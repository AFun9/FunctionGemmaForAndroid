package com.gemma.functiongemma.android;

import org.junit.Test;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;

public class ToolsetFingerprintTest {
    @Test
    public void staysStableAcrossEquivalentMapOrdering() {
        ToolsetDefinition first = new ToolsetDefinition("demo", "Demo", "system", List.of(tool("query", "action")));
        ToolsetDefinition second = new ToolsetDefinition("demo", "Demo", "system", List.of(tool("action", "query")));

        String firstFingerprint = ToolsetFingerprint.create(first, "modelSig", "tokenizerSig");
        String secondFingerprint = ToolsetFingerprint.create(second, "modelSig", "tokenizerSig");

        assertEquals(firstFingerprint, secondFingerprint);
    }

    @Test
    public void changesWhenSystemPromptChanges() {
        ToolsetDefinition first = new ToolsetDefinition("demo", "Demo", "system-a", List.of(tool("action", "query")));
        ToolsetDefinition second = new ToolsetDefinition("demo", "Demo", "system-b", List.of(tool("action", "query")));

        assertNotEquals(
                ToolsetFingerprint.create(first, "modelSig", "tokenizerSig"),
                ToolsetFingerprint.create(second, "modelSig", "tokenizerSig")
        );
    }

    private static Map<String, Object> tool(String firstProperty, String secondProperty) {
        Map<String, Object> properties = new LinkedHashMap<>();
        properties.put(firstProperty, property("string"));
        properties.put(secondProperty, property("string"));

        Map<String, Object> parameters = new LinkedHashMap<>();
        parameters.put("type", "object");
        parameters.put("properties", properties);

        Map<String, Object> function = new LinkedHashMap<>();
        function.put("name", "tool");
        function.put("description", "desc");
        function.put("parameters", parameters);

        Map<String, Object> tool = new LinkedHashMap<>();
        tool.put("type", "function");
        tool.put("function", function);
        return tool;
    }

    private static Map<String, Object> property(String type) {
        Map<String, Object> property = new LinkedHashMap<>();
        property.put("type", type);
        property.put("description", type + "-desc");
        return property;
    }
}
