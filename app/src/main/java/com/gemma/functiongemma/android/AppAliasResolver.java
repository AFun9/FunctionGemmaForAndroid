package com.gemma.functiongemma.android;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

final class AppAliasResolver {
    private final Map<String, String> packageByAlias = new LinkedHashMap<>();

    AppAliasResolver(List<AppAliasEntry> userAliases) {
        for (String[] entry : new String[][] {
                {"天气", "com.coloros.weather2"},
                {"家人守护", "com.coloros.familyguard"},
                {"美柚", "com.lingan.seeyou"},
                {"百度极速版", "com.baidu.searchbox.lite"},
                {"58同城", "com.wuba"},
                {"知乎", "com.zhihu.android"},
                {"滴滴出行", "com.sdu.didi.psnger"},
                {"计算器", "com.coloros.calculator"},
                {"飞猪旅行", "com.taobao.trip"},
                {"网易有道词典", "com.youdao.dict"},
                {"百度贴吧", "com.baidu.tieba"},
                {"腾讯新闻", "com.tencent.news"},
                {"饿了么", "me.ele"},
                {"优酷视频", "com.youku.phone"},
                {"抖音", "com.ss.android.ugc.aweme"},
                {"今日头条", "com.ss.android.article.news"},
                {"夸克", "com.quark.browser"},
                {"邮件", "com.android.email"},
                {"美团", "com.sankuai.meituan"},
                {"剪映", "com.lemon.lv"},
                {"酷狗音乐", "com.kugou.android"},
                {"网易邮箱大师", "com.netease.mail"},
                {"番茄免费小说", "com.dragon.read"},
                {"yy", "com.duowan.mobile"},
                {"qq", "com.tencent.mobileqq"},
                {"QQ", "com.tencent.mobileqq"},
                {"qq浏览器", "com.tencent.mtt"},
                {"QQ浏览器", "com.tencent.mtt"},
                {"文件管理", "com.coloros.filemanager"},
                {"豆瓣", "com.douban.frodo"},
                {"网易云音乐", "com.netease.cloudmusic"},
                {"喜马拉雅", "com.ximalaya.ting.android"},
                {"美团外卖", "com.sankuai.meituan.takeoutnew"},
                {"飞书", "com.ss.android.lark"},
                {"全民K歌", "com.tencent.karaoke"},
                {"微博", "com.sina.weibo"},
                {"墨迹天气", "com.moji.mjweather"},
                {"起点读书", "com.qidian.QDReader"},
                {"豆包", "com.larus.nova"},
                {"腾讯微视", "com.tencent.weishi"},
                {"keep", "com.gotokeep.keep"},
                {"腾讯地图", "com.tencent.map"},
                {"虎牙直播", "com.duowan.kiwi"},
                {"芒果TV", "com.hunantv.imgo.activity"},
                {"UC浏览器", "com.UCMobile"},
                {"腾讯文档", "com.tencent.docs"},
                {"携程旅行", "ctrip.android.view"},
                {"哈啰", "com.jingyao.easybike"},
                {"支付宝", "com.eg.android.AlipayGphone"},
                {"爱奇艺", "com.qiyi.video"},
                {"番茄畅听", "com.xs.fm"},
                {"得物", "com.shizhuang.duapp"},
                {"西瓜视频", "com.ss.android.article.video"},
                {"网易新闻", "com.netease.newsreader.activity"},
                {"腾讯视频", "com.tencent.qqlive"},
                {"淘宝", "com.taobao.taobao"},
                {"快手", "com.smile.gifmaker"},
                {"扫描全能王", "com.intsig.camscanner"},
                {"菜鸟", "com.cainiao.wireless"},
                {"盒马", "com.wudaokou.hippo"},
                {"阿里巴巴", "com.alibaba.wireless"},
                {"闲鱼", "com.taobao.idlefish"},
                {"QQ邮箱", "com.tencent.androidqqmail"},
                {"百度网盘", "com.baidu.netdisk"},
                {"酷安", "com.coolapk.market"},
                {"QQ音乐", "com.tencent.qqmusic"},
                {"百度", "com.baidu.searchbox"},
                {"铁路12306", "com.MobileTicket"},
                {"腾讯会议", "com.tencent.wemeet.app"},
                {"企业微信", "com.tencent.wework"},
                {"微信", "com.tencent.mm"},
                {"wechat", "com.tencent.mm"},
                {"京东", "com.jingdong.app.mall"},
                {"搜狐视频", "com.sohu.sohuvideo"},
                {"百度地图", "com.baidu.BaiduMap"},
                {"设置", "com.android.settings"},
                {"浏览器", "com.android.chrome"},
                {"chrome", "com.android.chrome"},
                {"高德", "com.autonavi.minimap"},
                {"高德地图", "com.autonavi.minimap"},
                {"amap", "com.autonavi.minimap"},
                {"qq music", "com.tencent.qqmusic"},
                {"网易云", "com.netease.cloudmusic"},
                {"netease music", "com.netease.cloudmusic"},
                {"spotify", "com.spotify.music"},
                {"youtube music", "com.google.android.apps.youtube.music"},
                {"music", "com.tencent.qqmusic"},
                {"音乐", "com.tencent.qqmusic"}
        }) {
            register(entry[0], entry[1]);
        }
        registerUserAliases(userAliases);
    }

    private void registerUserAliases(List<AppAliasEntry> aliases) {
        if (aliases == null) {
            return;
        }
        for (AppAliasEntry entry : aliases) {
            if (entry == null) {
                continue;
            }
            register(entry.alias(), entry.packageName());
            if (entry.appLabel() != null && !entry.appLabel().isBlank()) {
                register(entry.appLabel(), entry.packageName());
            }
        }
    }

    private void register(String alias, String packageName) {
        packageByAlias.put(normalize(alias), packageName);
    }

    String resolvePackage(String appName) {
        if (appName == null || appName.trim().isEmpty()) {
            return null;
        }
        return packageByAlias.get(normalize(appName));
    }

    private static String normalize(String value) {
        return value.trim()
                .replace(" ", "")
                .toLowerCase(Locale.ROOT);
    }
}
