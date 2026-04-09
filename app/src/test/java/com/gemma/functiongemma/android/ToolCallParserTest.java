package com.gemma.functiongemma.android;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;

public class ToolCallParserTest {
    @Test
    public void parsesTaggedJsonFunctionCall() {
        ToolCall toolCall = ToolCallParser.parse("""
                <start_function_call>
                {"name":"launch_app","arguments":{"app_name":"微信"}}
                <end_function_call>
                """);

        assertNotNull(toolCall);
        assertEquals("launch_app", toolCall.name());
        assertEquals("微信", toolCall.arguments().get("app_name"));
    }

    @Test
    public void parsesTaggedJsonFunctionCallWithoutClosingTag() {
        ToolCall toolCall = ToolCallParser.parse("""
                <start_function_call>
                {"name":"launch_app","arguments":{"app_name":"微信"}}
                """);

        assertNotNull(toolCall);
        assertEquals("launch_app", toolCall.name());
        assertEquals("微信", toolCall.arguments().get("app_name"));
    }

    @Test
    public void parsesJsonInCodeFence() {
        ToolCall toolCall = ToolCallParser.parse("""
                ```json
                {"name":"launch_app","arguments":{"app_name":"微信"}}
                ```
                """);

        assertNotNull(toolCall);
        assertEquals("launch_app", toolCall.name());
        assertEquals("微信", toolCall.arguments().get("app_name"));
    }

    @Test
    public void parsesEmbeddedJsonInsideText() {
        ToolCall toolCall = ToolCallParser.parse(
                "I will use a tool now: {\"name\":\"launch_app\",\"arguments\":{\"app_name\":\"微信\"}}"
        );

        assertNotNull(toolCall);
        assertEquals("launch_app", toolCall.name());
        assertEquals("微信", toolCall.arguments().get("app_name"));
    }

    @Test
    public void parsesNestedFunctionObject() {
        ToolCall toolCall = ToolCallParser.parse(
                "{\"function\":{\"name\":\"launch_app\",\"arguments\":{\"app_name\":\"微信\"}}}"
        );

        assertNotNull(toolCall);
        assertEquals("launch_app", toolCall.name());
        assertEquals("微信", toolCall.arguments().get("app_name"));
    }

    @Test
    public void parsesToolNameAndParametersShape() {
        ToolCall toolCall = ToolCallParser.parse(
                "{\"tool_name\":\"launch_app\",\"parameters\":{\"app_name\":\"微信\"}}"
        );

        assertNotNull(toolCall);
        assertEquals("launch_app", toolCall.name());
        assertEquals("微信", toolCall.arguments().get("app_name"));
    }

    @Test
    public void parsesFunctionGemmaTemplateCall() {
        ToolCall toolCall = ToolCallParser.parse(
                "<start_function_call>call:launch_app{app_name:<escape>微信<escape>}<end_function_call>"
        );

        assertNotNull(toolCall);
        assertEquals("launch_app", toolCall.name());
        assertEquals("微信", toolCall.arguments().get("app_name"));
    }

    @Test
    public void parsesOpenFunctionGemmaTemplateCall() {
        ToolCall toolCall = ToolCallParser.parse(
                "<start_function_call>call:launch_app{app_name:<escape>微信<escape>}"
        );

        assertNotNull(toolCall);
        assertEquals("launch_app", toolCall.name());
        assertEquals("微信", toolCall.arguments().get("app_name"));
    }

    @Test
    public void parsesFunctionStyleCall() {
        ToolCall toolCall = ToolCallParser.parse("play_music({\"song\":\"稻香\",\"artist\":\"周杰伦\"})");

        assertNotNull(toolCall);
        assertEquals("play_music", toolCall.name());
        assertEquals("稻香", toolCall.arguments().get("song"));
        assertEquals("周杰伦", toolCall.arguments().get("artist"));
    }

    @Test
    public void returnsNullForPlainText() {
        assertNull(ToolCallParser.parse("I think the user wants to hear a song."));
    }

    @Test
    public void returnsNullForMalformedFunctionJson() {
        assertNull(ToolCallParser.parse("play_music({\"song\":\"逆战\",})"));
        assertNull(ToolCallParser.parse("{\"name\":\"play_music\",\"arguments\":"));
    }
}
