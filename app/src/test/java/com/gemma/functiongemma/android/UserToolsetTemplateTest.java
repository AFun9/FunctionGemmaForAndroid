package com.gemma.functiongemma.android;

import org.junit.Test;

import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class UserToolsetTemplateTest {
    @Test
    public void parsesValidToolTemplate() {
        List<Map<String, Object>> tools = UserToolsetTemplate.parseToolsJson(UserToolsetTemplate.defaultToolsJsonTemplate());

        assertEquals(5, tools.size());
        assertEquals("function", tools.get(0).get("type"));
        assertTrue(tools.get(0).containsKey("function"));
        assertEquals(
                "navigate",
                ((Map<?, ?>) tools.get(1).get("function")).get("name")
        );
    }

    @Test(expected = IllegalArgumentException.class)
    public void rejectsMissingFunctionName() {
        UserToolsetTemplate.parseToolsJson("""
                [
                  {
                    "type": "function",
                    "function": {
                      "description": "desc",
                      "parameters": {
                        "type": "object",
                        "properties": {
                          "query": {
                            "type": "string",
                            "description": "desc"
                          }
                        }
                      }
                    }
                  }
                ]
                """);
    }
}
