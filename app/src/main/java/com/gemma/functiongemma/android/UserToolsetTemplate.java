package com.gemma.functiongemma.android;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;

import java.lang.reflect.Type;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

final class UserToolsetTemplate {
    private static final Gson GSON = new Gson();
    private static final Gson PRETTY_GSON = new GsonBuilder().setPrettyPrinting().create();
    private static final Type TOOL_LIST_TYPE = new TypeToken<List<Map<String, Object>>>() { }.getType();
    private static final String DEFAULT_SYSTEM_PROMPT =
            "You are a mobile assistant with tool calling. "
                    + "If one tool matches the user's main intent, call exactly one tool. "
                    + "Otherwise reply normally. "
                    + "When calling a tool, output only this format: "
                    + "<start_function_call>call:tool_name{arg_name:<escape>value<escape>}<end_function_call>. "
                    + "Do not output JSON. Do not add explanation before or after the tool call.";
    private static final String DEFAULT_TOOLS_TEMPLATE = """
            [
              {
                "type": "function",
                "function": {
                  "name": "open_target",
                  "description": "Open an app or a system settings page. Copy the exact target text from the user's request into target.",
                  "parameters": {
                    "type": "object",
                    "properties": {
                      "target": {
                        "type": "string",
                        "description": "Copy the exact target text from the user's request after 打开. Keep the original text as-is. Do not translate, paraphrase, romanize, or normalize it."
                      }
                    },
                    "required": ["target"]
                  }
                }
              },
              {
                "type": "function",
                "function": {
                  "name": "navigate",
                  "description": "Navigate to a destination in the map app.",
                  "parameters": {
                    "type": "object",
                    "properties": {
                      "destination": {
                        "type": "string",
                        "description": "Destination name or address"
                      },
                      "mode": {
                        "type": "string",
                        "description": "Optional mode such as driving, walking, transit"
                      }
                    },
                    "required": ["destination"]
                  }
                }
              },
              {
                "type": "function",
                "function": {
                  "name": "play_music",
                  "description": "Play or search music. Do not use this just to open a music app.",
                  "parameters": {
                    "type": "object",
                    "properties": {
                      "song": {
                        "type": "string",
                        "description": "Song title"
                      },
                      "artist": {
                        "type": "string",
                        "description": "Artist name"
                      },
                      "app_name": {
                        "type": "string",
                        "description": "Optional music app name"
                      }
                    }
                  }
                }
              },
              {
                "type": "function",
                "function": {
                  "name": "web_search",
                  "description": "Search the web in a browser.",
                  "parameters": {
                    "type": "object",
                    "properties": {
                      "query": {
                        "type": "string",
                        "description": "Search query"
                      }
                    },
                    "required": ["query"]
                  }
                }
              },
            ]
            """;

    private UserToolsetTemplate() {
    }

    static String defaultSystemPrompt() {
        return DEFAULT_SYSTEM_PROMPT;
    }

    static String defaultToolsJsonTemplate() {
        return DEFAULT_TOOLS_TEMPLATE;
    }

    static List<Map<String, Object>> parseToolsJson(String rawJson) {
        if (rawJson == null || rawJson.trim().isEmpty()) {
            throw new IllegalArgumentException("Tools JSON must not be empty");
        }

        List<Map<String, Object>> tools = GSON.fromJson(rawJson, TOOL_LIST_TYPE);
        if (tools == null || tools.isEmpty()) {
            throw new IllegalArgumentException("Tools JSON must contain at least one function");
        }

        java.util.Set<String> functionNames = new java.util.HashSet<>();
        for (int i = 0; i < tools.size(); i++) {
            Map<String, Object> tool = tools.get(i);
            validateTool(tool, i, functionNames);
        }
        return tools;
    }

    static String toPrettyJson(List<Map<String, Object>> tools) {
        return PRETTY_GSON.toJson(tools);
    }

    private static void validateTool(Map<String, Object> tool, int index, java.util.Set<String> functionNames) {
        if (tool == null) {
            throw new IllegalArgumentException("Tool at index " + index + " is null");
        }

        Object type = tool.get("type");
        if (!"function".equals(type)) {
            throw new IllegalArgumentException("Tool at index " + index + " must have type=function");
        }

        Object functionObject = tool.get("function");
        if (!(functionObject instanceof Map<?, ?> functionMap)) {
            throw new IllegalArgumentException("Tool at index " + index + " must contain a function object");
        }

        Object name = functionMap.get("name");
        if (!(name instanceof String) || ((String) name).trim().isEmpty()) {
            throw new IllegalArgumentException("Tool at index " + index + " must define function.name");
        }
        if (!functionNames.add(((String) name).trim())) {
            throw new IllegalArgumentException("Duplicate function.name detected: " + name);
        }

        Object description = functionMap.get("description");
        if (!(description instanceof String) || ((String) description).trim().isEmpty()) {
            throw new IllegalArgumentException("Tool at index " + index + " must define function.description");
        }

        Object parametersObject = functionMap.get("parameters");
        if (!(parametersObject instanceof Map<?, ?> parametersMap)) {
            throw new IllegalArgumentException("Tool at index " + index + " must define function.parameters");
        }

        Object parametersType = parametersMap.get("type");
        if (!(parametersType instanceof String) || ((String) parametersType).trim().isEmpty()) {
            throw new IllegalArgumentException("Tool at index " + index + " must define parameters.type");
        }

        Object propertiesObject = parametersMap.get("properties");
        if (!(propertiesObject instanceof Map<?, ?> propertiesMap) || propertiesMap.isEmpty()) {
            throw new IllegalArgumentException("Tool at index " + index + " must define at least one parameter property");
        }

        for (Map.Entry<?, ?> entry : propertiesMap.entrySet()) {
            String propertyName = String.valueOf(entry.getKey()).trim();
            if (propertyName.isEmpty()) {
                throw new IllegalArgumentException("Tool at index " + index + " contains an empty property name");
            }
            if (!(entry.getValue() instanceof Map<?, ?> propertyMap)) {
                throw new IllegalArgumentException("Property " + propertyName + " must be an object");
            }
            Object propertyType = propertyMap.get("type");
            if (!(propertyType instanceof String) || ((String) propertyType).trim().isEmpty()) {
                throw new IllegalArgumentException("Property " + propertyName + " must define type");
            }
            Object enumValue = propertyMap.get("enum");
            if (enumValue != null && (!(enumValue instanceof List<?>) || ((List<?>) enumValue).isEmpty())) {
                throw new IllegalArgumentException("Property " + propertyName + " has an invalid enum definition");
            }
        }
    }
}
