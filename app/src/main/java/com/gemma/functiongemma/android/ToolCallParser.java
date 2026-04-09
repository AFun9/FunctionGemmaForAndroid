package com.gemma.functiongemma.android;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

final class ToolCallParser {
    private static final Gson GSON = new Gson();
    private static final Pattern TAGGED_JSON_PATTERN = Pattern.compile(
            "<start_function_call>\\s*(\\{.*?\\})\\s*<end_function_call>",
            Pattern.DOTALL
    );
    private static final Pattern OPEN_TAGGED_JSON_PATTERN = Pattern.compile(
            "<start_function_call>\\s*(\\{.*\\})",
            Pattern.DOTALL
    );
    private static final Pattern FENCED_JSON_PATTERN = Pattern.compile(
            "```(?:json)?\\s*(\\{.*?\\})\\s*```",
            Pattern.DOTALL | Pattern.CASE_INSENSITIVE
    );
    private static final Pattern TEMPLATE_CALL_PATTERN = Pattern.compile(
            "<start_function_call>\\s*call:([a-zA-Z_][a-zA-Z0-9_]*)\\s*\\{(.*?)\\}\\s*<end_function_call>",
            Pattern.DOTALL
    );
    private static final Pattern OPEN_TEMPLATE_CALL_PATTERN = Pattern.compile(
            "<start_function_call>\\s*call:([a-zA-Z_][a-zA-Z0-9_]*)\\s*\\{(.*)\\}",
            Pattern.DOTALL
    );
    private static final Pattern FUNCTION_LIKE_PATTERN = Pattern.compile(
            "([a-zA-Z_][a-zA-Z0-9_]*)\\s*\\((\\{.*\\})\\)",
            Pattern.DOTALL
    );

    private ToolCallParser() {
    }

    static ToolCall parse(String rawText) {
        if (rawText == null || rawText.trim().isEmpty()) {
            return null;
        }

        try {
            String trimmed = rawText.trim();
            Matcher taggedMatcher = TAGGED_JSON_PATTERN.matcher(trimmed);
            if (taggedMatcher.find()) {
                return parseJsonCall(taggedMatcher.group(1));
            }

            Matcher openTaggedMatcher = OPEN_TAGGED_JSON_PATTERN.matcher(trimmed);
            if (openTaggedMatcher.find()) {
                ToolCall openTaggedCall = parseJsonCall(openTaggedMatcher.group(1));
                if (openTaggedCall != null) {
                    return openTaggedCall;
                }
            }

            Matcher fencedMatcher = FENCED_JSON_PATTERN.matcher(trimmed);
            if (fencedMatcher.find()) {
                ToolCall fencedCall = parseJsonCall(fencedMatcher.group(1));
                if (fencedCall != null) {
                    return fencedCall;
                }
            }

            Matcher templateMatcher = TEMPLATE_CALL_PATTERN.matcher(trimmed);
            if (templateMatcher.find()) {
                ToolCall templateCall = parseTemplateCall(
                        templateMatcher.group(1),
                        templateMatcher.group(2)
                );
                if (templateCall != null) {
                    return templateCall;
                }
            }

            Matcher openTemplateMatcher = OPEN_TEMPLATE_CALL_PATTERN.matcher(trimmed);
            if (openTemplateMatcher.find()) {
                ToolCall templateCall = parseTemplateCall(
                        openTemplateMatcher.group(1),
                        openTemplateMatcher.group(2)
                );
                if (templateCall != null) {
                    return templateCall;
                }
            }

            if (trimmed.startsWith("{") && trimmed.endsWith("}")) {
                ToolCall directJson = parseJsonCall(trimmed);
                if (directJson != null) {
                    return directJson;
                }
            }

            ToolCall embeddedJson = parseEmbeddedJsonCall(trimmed);
            if (embeddedJson != null) {
                return embeddedJson;
            }

            Matcher functionMatcher = FUNCTION_LIKE_PATTERN.matcher(trimmed);
            if (functionMatcher.find()) {
                String name = functionMatcher.group(1);
                String argsJson = functionMatcher.group(2);
                @SuppressWarnings("unchecked")
                Map<String, Object> arguments = GSON.fromJson(argsJson, Map.class);
                return new ToolCall(name, arguments);
            }

            return null;
        } catch (RuntimeException error) {
            return null;
        }
    }

    private static ToolCall parseEmbeddedJsonCall(String text) {
        int start = text.indexOf('{');
        while (start >= 0 && start < text.length()) {
            String candidate = extractBalancedJsonObject(text, start);
            if (candidate != null) {
                ToolCall parsed = parseJsonCall(candidate);
                if (parsed != null) {
                    return parsed;
                }
            }
            start = text.indexOf('{', start + 1);
        }
        return null;
    }

    private static ToolCall parseTemplateCall(String name, String argumentsBody) {
        if (name == null || name.trim().isEmpty()) {
            return null;
        }
        Map<String, Object> arguments = new LinkedHashMap<>();
        if (argumentsBody != null && !argumentsBody.trim().isEmpty()) {
            for (String part : splitTemplateArguments(argumentsBody)) {
                int separator = part.indexOf(':');
                if (separator <= 0) {
                    continue;
                }
                String key = part.substring(0, separator).trim();
                String rawValue = part.substring(separator + 1).trim();
                if (key.isEmpty()) {
                    continue;
                }
                arguments.put(key, unescapeTemplateValue(rawValue));
            }
        }
        return new ToolCall(name, arguments);
    }

    private static java.util.List<String> splitTemplateArguments(String text) {
        java.util.List<String> parts = new java.util.ArrayList<>();
        StringBuilder current = new StringBuilder();
        boolean inEscape = false;
        for (int index = 0; index < text.length(); ) {
            if (text.startsWith("<escape>", index)) {
                inEscape = !inEscape;
                current.append("<escape>");
                index += "<escape>".length();
                continue;
            }
            char currentChar = text.charAt(index);
            if (currentChar == ',' && !inEscape) {
                String item = current.toString().trim();
                if (!item.isEmpty()) {
                    parts.add(item);
                }
                current.setLength(0);
            } else {
                current.append(currentChar);
            }
            index++;
        }
        String tail = current.toString().trim();
        if (!tail.isEmpty()) {
            parts.add(tail);
        }
        return parts;
    }

    private static String unescapeTemplateValue(String rawValue) {
        String value = rawValue.trim();
        if (value.startsWith("<escape>") && value.endsWith("<escape>") && value.length() >= 16) {
            value = value.substring("<escape>".length(), value.length() - "<escape>".length());
        }
        return value;
    }

    private static String extractBalancedJsonObject(String text, int startIndex) {
        boolean inString = false;
        boolean escaping = false;
        int depth = 0;
        for (int i = startIndex; i < text.length(); i++) {
            char current = text.charAt(i);
            if (escaping) {
                escaping = false;
                continue;
            }
            if (current == '\\') {
                escaping = true;
                continue;
            }
            if (current == '"') {
                inString = !inString;
                continue;
            }
            if (inString) {
                continue;
            }
            if (current == '{') {
                depth++;
            } else if (current == '}') {
                depth--;
                if (depth == 0) {
                    return text.substring(startIndex, i + 1);
                }
            }
        }
        return null;
    }

    private static ToolCall parseJsonCall(String json) {
        JsonObject object = JsonParser.parseString(json).getAsJsonObject();
        String name = readString(object, "name");
        if (name == null) {
            name = readString(object, "function");
        }
        if (name == null) {
            name = readString(object, "tool_name");
        }
        if (name == null && object.has("function") && object.get("function").isJsonObject()) {
            JsonObject functionObject = object.getAsJsonObject("function");
            name = readString(functionObject, "name");
            if (name != null) {
                object = functionObject;
            }
        }
        if (name == null) {
            return null;
        }

        Map<String, Object> arguments = new LinkedHashMap<>();
        if (object.has("arguments") && object.get("arguments").isJsonObject()) {
            @SuppressWarnings("unchecked")
            Map<String, Object> parsedArgs = GSON.fromJson(object.get("arguments"), Map.class);
            arguments.putAll(parsedArgs);
        } else if (object.has("args") && object.get("args").isJsonObject()) {
            @SuppressWarnings("unchecked")
            Map<String, Object> parsedArgs = GSON.fromJson(object.get("args"), Map.class);
            arguments.putAll(parsedArgs);
        } else if (object.has("parameters") && object.get("parameters").isJsonObject()) {
            @SuppressWarnings("unchecked")
            Map<String, Object> parsedArgs = GSON.fromJson(object.get("parameters"), Map.class);
            arguments.putAll(parsedArgs);
        }
        return new ToolCall(name, arguments);
    }

    private static String readString(JsonObject object, String key) {
        if (!object.has(key) || !object.get(key).isJsonPrimitive()) {
            return null;
        }
        return object.get(key).getAsString();
    }
}
