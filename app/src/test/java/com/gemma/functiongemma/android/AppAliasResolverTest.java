package com.gemma.functiongemma.android;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import java.util.List;

public class AppAliasResolverTest {
    @Test
    public void resolvesKnownAliases() {
        AppAliasResolver resolver = new AppAliasResolver();

        assertEquals("com.tencent.mm", resolver.resolvePackage("微信"));
        assertEquals("com.tencent.mobileqq", resolver.resolvePackage("QQ"));
        assertEquals("com.eg.android.AlipayGphone", resolver.resolvePackage("支付宝"));
        assertEquals("com.tencent.qqmusic", resolver.resolvePackage("QQ音乐"));
        assertEquals("com.android.chrome", resolver.resolvePackage("浏览器"));
    }

    @Test
    public void normalizesWhitespaceInAlias() {
        AppAliasResolver resolver = new AppAliasResolver();

        assertEquals("com.tencent.mobileqq", resolver.resolvePackage(" q q "));
        assertEquals("com.tencent.mobileqq", resolver.resolvePackage("QQ"));
    }

    @Test
    public void returnsNullForUnknownAlias() {
        AppAliasResolver resolver = new AppAliasResolver();

        assertNull(resolver.resolvePackage("不存在的应用"));
    }

    @Test
    public void recognizesMusicApps() {
        AppAliasResolver resolver = new AppAliasResolver();

        assertTrue(resolver.isMusicAppName("QQ音乐"));
        assertTrue(resolver.isMusicAppName("网易云音乐"));
        assertFalse(resolver.isMusicAppName("微信"));
    }

    @Test
    public void findsKnownAliasInText() {
        AppAliasResolver resolver = new AppAliasResolver();

        assertEquals("qq音乐", resolver.findKnownAliasInText("请打开QQ音乐播放逆战"));
        assertEquals("微信", resolver.findKnownAliasInText("帮我打开微信"));
        assertEquals("支付宝", resolver.findKnownAliasInText("请直接打开支付宝"));
        assertNull(resolver.findKnownAliasInText("帮我打开一个不存在的应用"));
    }

    @Test
    public void resolvesUserDefinedAlias() {
        AppAliasResolver resolver = new AppAliasResolver(List.of(
                new AppAliasEntry("工作聊天", "com.tencent.wework", "企业微信")
        ));

        assertEquals("com.tencent.wework", resolver.resolvePackage("工作聊天"));
        assertEquals("com.tencent.wework", resolver.resolvePackage("企业微信"));
    }
}
