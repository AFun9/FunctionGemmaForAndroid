package com.gemma.functiongemma.android;

import java.util.Arrays;
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
                "You are a mobile assistant with tool calling. "
                        + "If one tool matches the user's main intent, call exactly one tool. "
                        + "Otherwise reply normally. "
                        + "When calling a tool, output only this format: "
                        + "<start_function_call>call:tool_name{arg_name:<escape>value<escape>}<end_function_call>. "
                        + "Do not output JSON. Do not add explanation before or after the tool call.",
                List.of(
                        launchAppTool(),
                        navigateTool(),
                        playMusicTool(),
                        webSearchTool(),
                        openSystemPanelTool()
                )
        );
    }

    private static Map<String, Object> launchAppTool() {
        return functionTool(
                "launch_app",
                "Open an app itself. Do not use this for navigation, music playback, web search, or system settings.",
                properties(
                        property("app_name", "string", "App name")
                )
        );
    }

    private static Map<String, Object> playMusicTool() {
        return functionTool(
                "play_music",
                "Play or search music. If both artist and song appear, extract them separately. Prefer this over launch_app for music requests.",
                properties(
                        property("song", "string", "Song title"),
                        property("artist", "string", "Artist name"),
                        property("app_name", "string", "Optional music app name")
                )
        );
    }

    private static Map<String, Object> navigateTool() {
        return functionTool(
                "navigate",
                "Navigate to a destination. The default map app is 高德地图.",
                properties(
                        property("destination", "string", "Destination name or address"),
                        property("mode", "string", "Optional mode such as driving, walking, transit")
                )
        );
    }

    private static Map<String, Object> webSearchTool() {
        return functionTool(
                "web_search",
                "Search the web.",
                properties(
                        property("query", "string", "Search query")
                )
        );
    }

    private static Map<String, Object> openSystemPanelTool() {
        Map<String, Object> panel = property("panel", "string", "Panel name");
        panel.put("enum", Arrays.asList("bluetooth", "wifi", "internet", "settings"));
        return functionTool(
                "open_system_panel",
                "Open a system panel such as bluetooth, wifi, internet, or settings.",
                properties(panel)
        );
    }

    private static Map<String, Object> functionTool(String name, String description, Map<String, Object> parameters) {
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
