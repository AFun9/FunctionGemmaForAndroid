package com.gemma.functiongemma;

import java.util.List;
import java.util.Map;
import java.util.TreeMap;

final class ChatTemplateBuilder {
    private ChatTemplateBuilder() {
    }

    static String build(String bosToken,
                        List<Map<String, Object>> messages,
                        List<Map<String, Object>> tools,
                        boolean addGenerationPrompt) {
        StringBuilder sb = new StringBuilder();
        sb.append(bosToken);

        List<Map<String, Object>> messagesToProcess = messages;
        boolean hasSystemBlock = (tools != null && !tools.isEmpty())
                || (!messages.isEmpty() && isSystemRole((String) messages.get(0).get("role")));

        if (hasSystemBlock) {
            sb.append("<start_of_turn>developer\n");
            if (!messages.isEmpty() && isSystemRole((String) messages.get(0).get("role"))) {
                appendContent(sb, messages.get(0).get("content"));
                messagesToProcess = messages.subList(1, messages.size());
            }
            appendToolDeclarations(sb, tools);
            sb.append("<end_of_turn>\n");
        }

        for (Map<String, Object> message : messagesToProcess) {
            String role = (String) message.get("role");
            if ("assistant".equals(role)) {
                role = "model";
            }
            sb.append("<start_of_turn>").append(role).append("\n");
            appendContent(sb, message.get("content"));
            sb.append("<end_of_turn>\n");
        }

        if (addGenerationPrompt) {
            sb.append("<start_of_turn>model\n");
        }
        return sb.toString();
    }

    private static boolean isSystemRole(String role) {
        return "developer".equals(role) || "system".equals(role);
    }

    private static void appendToolDeclarations(StringBuilder sb, List<Map<String, Object>> tools) {
        if (tools == null || tools.isEmpty()) {
            return;
        }
        for (Map<String, Object> tool : tools) {
            Object functionObj = tool.get("function");
            if (!(functionObj instanceof Map)) {
                continue;
            }
            @SuppressWarnings("unchecked")
            Map<String, Object> function = (Map<String, Object>) functionObj;
            sb.append("<start_function_declaration>");
            sb.append("declaration:").append(function.get("name"));
            sb.append("{description:<escape>").append(function.get("description")).append("<escape>");
            appendParameters(sb, function);
            sb.append("}");
            sb.append("<end_function_declaration>");
        }
    }

    private static void appendParameters(StringBuilder sb, Map<String, Object> function) {
        @SuppressWarnings("unchecked")
        Map<String, Object> parameters = (Map<String, Object>) function.get("parameters");
        if (parameters == null) {
            return;
        }
        sb.append(",parameters:{");

        @SuppressWarnings("unchecked")
        Map<String, Object> properties = (Map<String, Object>) parameters.get("properties");
        if (properties != null) {
            sb.append("properties:{");
            boolean first = true;
            for (Map.Entry<String, Object> entry : new TreeMap<>(properties).entrySet()) {
                if (!first) {
                    sb.append(",");
                }
                appendProperty(sb, entry.getKey(), entry.getValue());
                first = false;
            }
            sb.append("}");
        }

        @SuppressWarnings("unchecked")
        List<String> required = (List<String>) parameters.get("required");
        if (required != null && !required.isEmpty()) {
            sb.append(",required:[");
            for (int i = 0; i < required.size(); i++) {
                if (i > 0) {
                    sb.append(",");
                }
                sb.append("<escape>").append(required.get(i)).append("<escape>");
            }
            sb.append("]");
        }

        String type = (String) parameters.get("type");
        if (type != null) {
            sb.append(",type:<escape>").append(type.toUpperCase()).append("<escape>");
        }
        sb.append("}");
    }

    private static void appendProperty(StringBuilder sb, String propName, Object propValue) {
        if (!(propValue instanceof Map)) {
            return;
        }
        @SuppressWarnings("unchecked")
        Map<String, Object> propMap = (Map<String, Object>) propValue;
        sb.append(propName).append(":{description:<escape>")
                .append(propMap.get("description")).append("<escape>")
                .append(",type:<escape>").append(propMap.get("type")).append("<escape>");
        if ("action".equals(propName)) {
            @SuppressWarnings("unchecked")
            List<String> enumValues = (List<String>) propMap.get("enum");
            if (enumValues != null && !enumValues.isEmpty()) {
                sb.append(",enum:[");
                for (int i = 0; i < enumValues.size(); i++) {
                    if (i > 0) {
                        sb.append(",");
                    }
                    sb.append("<escape>").append(enumValues.get(i)).append("<escape>");
                }
                sb.append("]");
            }
        }
        sb.append("}");
    }

    private static void appendContent(StringBuilder sb, Object content) {
        if (content == null) {
            return;
        }
        if (content instanceof String) {
            sb.append(((String) content).trim());
        } else {
            sb.append(content.toString());
        }
    }
}
