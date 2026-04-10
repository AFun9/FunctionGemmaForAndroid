package com.gemma.functiongemma.android;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

final class BuiltInToolsets {
        static final String MOBILE_ASSISTANT_ID = "mobile-assistant";

        private BuiltInToolsets() {
        }

        static ToolsetDefinition createMobileAssistantToolset() {
                return new ToolsetDefinition(
                                MOBILE_ASSISTANT_ID,
                                "Mobile Assistant",
                                "You are a mobile assistant with tool calling. ",
                                List.of(
                                                openTargetTool(),
                                                navigateTool(),
                                                playMusicTool(),
                                                webSearchTool()));
        }

        private static Map<String, Object> openTargetTool() {
                Map<String, Object> parameters = properties(
                                property("target", "string",
                                                "Copy the exact target text from the user's request after 打开. Keep the original text as-is. Do not translate, paraphrase, romanize, or normalize it."));
                parameters.put("required", List.of("target"));
                return functionTool(
                                "open_target",
                                "Open an app or a system settings page. Copy the exact target text from the user's request into target.",
                                parameters);
        }

        private static Map<String, Object> playMusicTool() {
                return functionTool(
                                "play_music",
                                "Play or search music. Do not use this just to open a music app.",
                                properties(
                                                property("song", "string", "Song title"),
                                                property("artist", "string", "Artist name"),
                                                property("app_name", "string", "Optional music app name")));
        }

        private static Map<String, Object> navigateTool() {
                Map<String, Object> parameters = properties(
                                property("destination", "string", "Destination name or address"),
                                property("mode", "string",
                                                "Optional mode such as driving, walking, transit"));
                parameters.put("required", List.of("destination"));
                return functionTool(
                                "navigate",
                                "Navigate to a destination in the map app.",
                                parameters);
        }

        private static Map<String, Object> webSearchTool() {
                Map<String, Object> parameters = properties(
                                property("query", "string", "Search query"));
                parameters.put("required", List.of("query"));
                return functionTool(
                                "web_search",
                                "Search the web in a browser.",
                                parameters);
        }

        private static Map<String, Object> functionTool(String name, String description,
                        Map<String, Object> parameters) {
                Map<String, Object> function = new LinkedHashMap<>();
                function.put("name", name);
                function.put("description", description);
                function.put("parameters", parameters);

                Map<String, Object> tool = new LinkedHashMap<>();
                tool.put("type", "function");
                tool.put("function", function);
                return tool;
        }

        private static Map<String, Object> properties(Map<String, Object>... propertyEntries) {
                Map<String, Object> properties = new LinkedHashMap<>();
                for (Map<String, Object> property : propertyEntries) {
                        properties.put(String.valueOf(property.remove("_name")), property);
                }
                Map<String, Object> parameters = new LinkedHashMap<>();
                parameters.put("type", "object");
                parameters.put("properties", properties);
                return parameters;
        }

        private static Map<String, Object> property(String name, String type, String description) {
                Map<String, Object> property = new LinkedHashMap<>();
                property.put("_name", name);
                property.put("type", type);
                property.put("description", description);
                return property;
        }
}
