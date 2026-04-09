package com.gemma.functiongemma.android;

import org.junit.Test;

import java.util.Map;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public class ToolExecutionGuardTest {
    @Test
    public void blocksNonMusicAppForMusicRequest() {
        ToolExecutionGuard guard = new ToolExecutionGuard(new AppAliasResolver());
        ToolCall toolCall = new ToolCall("launch_app", Map.of("app_name", "微信"));

        ToolExecutionResult result = guard.validate("我想听张杰的逆战", toolCall);

        assertFalse(result.executed());
    }

    @Test
    public void allowsPlayMusicForMusicRequest() {
        ToolExecutionGuard guard = new ToolExecutionGuard(new AppAliasResolver());
        ToolCall toolCall = new ToolCall("play_music", Map.of("song", "逆战", "artist", "张杰"));

        ToolExecutionResult result = guard.validate("我想听张杰的逆战", toolCall);

        assertTrue(result.executed());
    }
}
