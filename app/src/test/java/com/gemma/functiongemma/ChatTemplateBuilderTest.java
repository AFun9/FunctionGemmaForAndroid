package com.gemma.functiongemma;

import org.junit.Test;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertTrue;

public class ChatTemplateBuilderTest {
    @Test
    public void buildsDeveloperToolAndUserTurns() {
        List<Map<String, Object>> messages = new ArrayList<>();
        messages.add(message("developer", "System prompt"));
        messages.add(message("user", "Play music"));

        List<Map<String, Object>> tools = new ArrayList<>();
        tools.add(tool());

        String prompt = ChatTemplateBuilder.build("<bos>", messages, tools, true);

        assertTrue(prompt.startsWith("<bos><start_of_turn>developer"));
        assertTrue(prompt.contains("<start_function_declaration>"));
        assertTrue(prompt.contains("<start_of_turn>user\nPlay music<end_of_turn>\n"));
        assertTrue(prompt.endsWith("<start_of_turn>model\n"));
    }

    @Test
    public void sortsToolPropertiesDeterministically() {
        List<Map<String, Object>> messages = new ArrayList<>();
        messages.add(message("developer", "System prompt"));

        String prompt = ChatTemplateBuilder.build("<bos>", messages, List.of(toolWithPropertiesOutOfOrder()), false);

        assertTrue(prompt.contains("properties:{action:"));
        assertTrue(prompt.indexOf("action:{") < prompt.indexOf("query:{"));
        assertTrue(prompt.indexOf("query:{") < prompt.indexOf("volume:{"));
    }

    private static Map<String, Object> message(String role, String content) {
        Map<String, Object> message = new HashMap<>();
        message.put("role", role);
        message.put("content", content);
        return message;
    }

    private static Map<String, Object> tool() {
        Map<String, Object> parameters = new HashMap<>();
        parameters.put("type", "object");
        parameters.put("properties", new HashMap<String, Object>());

        Map<String, Object> function = new HashMap<>();
        function.put("name", "control_music");
        function.put("description", "desc");
        function.put("parameters", parameters);

        Map<String, Object> tool = new HashMap<>();
        tool.put("type", "function");
        tool.put("function", function);
        return tool;
    }

    private static Map<String, Object> toolWithPropertiesOutOfOrder() {
        Map<String, Object> properties = new LinkedHashMap<>();
        properties.put("volume", property("integer"));
        properties.put("query", property("string"));
        properties.put("action", property("string"));

        Map<String, Object> parameters = new LinkedHashMap<>();
        parameters.put("type", "object");
        parameters.put("properties", properties);

        Map<String, Object> function = new LinkedHashMap<>();
        function.put("name", "control_music");
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
