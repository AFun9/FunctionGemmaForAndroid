package com.gemma.functiongemma.android;

import org.junit.Test;

import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

public class ToolCallFallbackResolverTest {
    @Test
    public void rewritesSystemPanelMistakeToLaunchApp() {
        ToolCallFallbackResolver resolver = new ToolCallFallbackResolver(new AppAliasResolver());

        ToolCallResolution resolution = resolver.resolve(
                "打开微信",
                new ToolCall("open_system_panel", Map.of())
        );

        assertEquals("fallback", resolution.source());
        assertNotNull(resolution.toolCall());
        assertEquals("launch_app", resolution.toolCall().name());
        assertEquals("微信", resolution.toolCall().arguments().get("app_name"));
    }

    @Test
    public void fillsMissingLaunchAppNameFromUserMessage() {
        ToolCallFallbackResolver resolver = new ToolCallFallbackResolver(new AppAliasResolver());

        ToolCallResolution resolution = resolver.resolve(
                "打开wechat",
                new ToolCall("launch_app", Map.of())
        );

        assertEquals("fallback", resolution.source());
        assertNotNull(resolution.toolCall());
        assertEquals("launch_app", resolution.toolCall().name());
        assertEquals("wechat", resolution.toolCall().arguments().get("app_name"));
    }

    @Test
    public void replacesSuspiciousPlaceholderAppName() {
        ToolCallFallbackResolver resolver = new ToolCallFallbackResolver(new AppAliasResolver());

        ToolCallResolution resolution = resolver.resolve(
                "打开支付宝",
                new ToolCall("launch_app", Map.of("app_name", "my_todo_app"))
        );

        assertEquals("fallback", resolution.source());
        assertNotNull(resolution.toolCall());
        assertEquals("支付宝", resolution.toolCall().arguments().get("app_name"));
    }

    @Test
    public void keepsValidSystemPanelCall() {
        ToolCallFallbackResolver resolver = new ToolCallFallbackResolver(new AppAliasResolver());

        ToolCallResolution resolution = resolver.resolve(
                "打开wifi",
                new ToolCall("open_system_panel", Map.of("panel", "wifi"))
        );

        assertEquals("model", resolution.source());
        assertNotNull(resolution.toolCall());
        assertEquals("open_system_panel", resolution.toolCall().name());
        assertEquals("wifi", resolution.toolCall().arguments().get("panel"));
    }
}
