package com.gemma.functiongemma;

/**
 * 手动实现的 BPE Tokenizer（基于 tokenizer.json）
 *
 * 这个类实现了完整的Byte Pair Encoding (BPE)分词算法，包括：
 * - 从tokenizer.json文件加载词汇表和BPE合并规则
 * - 应用预分词器（按空格分割文本）
 * - 执行BPE编码和解码
 * - 处理特殊token（BOS、EOS、PAD、UNK等）
 * - 应用聊天模板格式化消息
 */

import android.content.Context;
import android.content.res.AssetManager;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.*;


public class ManualBPETokenizer implements Tokenizer {
    private static final int MAX_BPE_MERGE_ITERATIONS = 50;
    private static final int MAX_BPE_CACHE_SIZE = 4096;
    private static final List<String> ALL_SPECIAL_TOKENS_SORTED = createSortedSpecialTokens();
    private static final boolean DEBUG_TEMPLATE = Boolean.getBoolean("functiongemma.debug.template");
    private static final Map<String, Integer> TOKEN_DEFAULTS = Map.of(
            "bos_token", 2,
            "pad_token", 0,
            "unk_token", 3
    );

    // ========== FunctionGemma 数据结构 ==========

    /** 词汇表：token字符串 -> token ID 的映射 */
    private Map<String, Integer> vocab;

    /** 反向词汇表：token ID -> token字符串 的映射 */
    private List<String> idToToken;

    /** BPE合并规则列表，每个规则是"token1 token2"格式 */
    private List<String> merges;

    /** EOS token ID列表，支持多个结束标记 */
    private List<Integer> eosTokenIds;
    /** BOS (Beginning of Sequence) token ID */
    private int bosTokenId;
    /** PAD (Padding) token ID */
    private int padTokenId;
    /** UNK (Unknown) token ID */
    private int unkTokenId;

    /** BPE编码结果缓存，避免重复计算相同的文本 */
    private Map<String, List<String>> bpeCache;
    private Map<String, MergeRule> mergeRulesMap;
    private List<String> activeSpecialTokens;
    private Set<Integer> eosTokenIdSet;
    private String unkTokenText;
    private final TokenizerJsonParser parser;

    /**
     * 从 tokenizer 目录创建 tokenizer 实例
     *
     * 这个方法是创建ManualBPETokenizer的主要入口点
     * 1. 创建ManualBPETokenizer实例
     * 2. 自动加载对应目录下的tokenizer.json和相关配置文件
     * 3. 初始化所有数据结构
     *
     * tokenizerDir 包含 tokenizer.json 等文件的目录路径
     * 返回配置好的 ManualBPETokenizer 实例，包含词汇表、BPE规则和特殊token配置，如果文件读取失败则抛出 IOException
     */
    public static ManualBPETokenizer fromDirectory(String tokenizerDir) throws IOException {
        ManualBPETokenizer tokenizer = new ManualBPETokenizer(); // 创建新实例
        tokenizer.loadTokenizer(new DirectoryResourceLoader(tokenizerDir)); // 加载配置
        return tokenizer; // 返回配置完成的ManualBPETokenizer
    }

    public static ManualBPETokenizer fromAssets(Context context, String assetBasePath) throws IOException {
        ManualBPETokenizer tokenizer = new ManualBPETokenizer();
        tokenizer.loadTokenizer(new AssetResourceLoader(context.getAssets(), assetBasePath));
        return tokenizer;
    }

    /**
     * 初始化所有核心数据结构
     *
     * 只能通过fromDirectory方法创建实例，确保ManualBPETokenizer被正确初始化
     */
    private ManualBPETokenizer() {
        this.vocab = new HashMap<>();
        this.idToToken = new ArrayList<>();
        this.mergeRulesMap = new HashMap<>();
        this.merges = new ArrayList<>();  // 初始化BPE合并规则列表
        this.bpeCache = new LinkedHashMap<>(1024, 0.75f, true) {
            @Override
            protected boolean removeEldestEntry(Map.Entry<String, List<String>> eldest) {
                return size() > MAX_BPE_CACHE_SIZE;
            }
        };
        this.activeSpecialTokens = new ArrayList<>();
        this.eosTokenIdSet = new HashSet<>();
        this.parser = new TokenizerJsonParser();
    }

    /**
     * 加载 tokenizer 配置
     *
     * 从 tokenizer.json 文件中读取词汇表、BPE 合并规则和特殊 token 配置。
     * 这个方法是ManualBPETokenizer初始化的核心，会依次加载：
     * 1. tokenizer.json - 主要的词汇表和BPE规则
     * 2. 特殊token配置 - 从各种配置文件中读取
     * 3. 额外词汇表 - added_tokens.json
     *
     * tokenizerDir tokenizer 目录路径，包含所有配置文件
     * 抛出 IOException 如果文件读取或解析失败
     */
    private void loadTokenizer(ResourceLoader resourceLoader) throws IOException {
        // 检查tokenizer.json文件是否存在
        if (!resourceLoader.exists("tokenizer.json")) {
            throw new IOException("tokenizer.json 不存在");
        }

        // ===== 加载词汇表 =====
        Map<String, Integer> vocabMap;
        try (InputStream inputStream = resourceLoader.open("tokenizer.json")) {
            vocabMap = parser.parseVocab(inputStream);
        }
        for (Map.Entry<String, Integer> entry : vocabMap.entrySet()) {
            String token = entry.getKey(); // token字符串
            int id = entry.getValue(); // token ID
            putToken(token, id);
        }

        // ===== 加载 BPE 合并规则 =====
        List<String> mergesList;
        try (InputStream inputStream = resourceLoader.open("tokenizer.json")) {
            mergesList = parser.parseMerges(inputStream);
        }
        merges.addAll(mergesList);

        // 处理合并规则映射
        for (int i = 0; i < merges.size(); i++) {
            String merge = merges.get(i);
            String[] parts = merge.split(" ");
            if (parts.length != 2) continue;
            String first = parts[0];
            String second = parts[1];
            String mergedToken = first + second;
            // 检查合并后的token是否在词汇表中，如果不在，则跳过这个规则
            if (vocab.containsKey(mergedToken)) {
                mergeRulesMap.put(first + " " + second, new MergeRule(i, mergedToken));
            }
        }

        // ===== 加载特殊token配置 =====
        loadSpecialTokens(resourceLoader); // 从配置文件加载BOS、EOS、PAD等特殊token

        // ===== 加载额外词汇表 =====
        loadAddedTokens(resourceLoader); // 加载added_tokens.json中的额外token

        // 仅保留词表中存在的特殊token，后续编码直接复用，避免每次动态排序/过滤
        activeSpecialTokens.clear();
        for (String token : ALL_SPECIAL_TOKENS_SORTED) {
            if (vocab.containsKey(token)) {
                activeSpecialTokens.add(token);
            }
        }
        eosTokenIdSet.clear();
        eosTokenIdSet.addAll(eosTokenIds);
        unkTokenText = getTokenById(unkTokenId);

        // ===== 加载完成，输出统计信息 =====
        System.out.println("Tokenizer 加载完成:");
        System.out.println("  - 词汇表大小: " + vocab.size()); // 显示词汇表大小
        System.out.println("  - BPE merges 数量: " + merges.size()); // 显示合并规则数量
        System.out.println("  - EOS token IDs: " + eosTokenIds); // 显示EOS token列表
//        System.out.println("vocab冲突： " + countSameHashCodeGroups(vocab));
//        System.out.println("mergeRulesMap冲突： " + countSameHashCodeGroups(mergeRulesMap));
    }

    /**
     * 加载特殊 token 配置
     *
     * 从多个配置文件中读取特殊token的设置，包括：
     * - config.json: 主要的模型配置
     * - special_tokens_map.json: 特殊token映射
     * - tokenizer_config.json: tokenizer配置
     *
     * 支持的特殊token：
     * - BOS (Beginning of Sequence)
     * - EOS (End of Sequence) - 支持多个
     * - PAD (Padding)
     * - UNK (Unknown)
     */
    private void loadSpecialTokens(ResourceLoader resourceLoader) throws IOException {
        // 初始化 eosTokenIds 列表，支持多个EOS token
        eosTokenIds = new ArrayList<>();

        // 从配置文件加载特殊token
        String[] configFiles = {"config.json", "special_tokens_map.json", "tokenizer_config.json"};
        for (String configFile : configFiles) {
            if (!resourceLoader.exists(configFile)) continue;

            Map<String, Object> config = loadJsonConfig(resourceLoader, configFile);

            // 处理 EOS token（可能是一个或多个）
            if (eosTokenIds.isEmpty()) {
                Object eosTokenObj = config.get("eos_token_id");
                if (eosTokenObj != null) {
                    if (eosTokenObj instanceof List) {
                        @SuppressWarnings("unchecked")
                        List<Object> eosList = (List<Object>) eosTokenObj;
                        for (Object eos : eosList) {
                            eosTokenIds.add(((Number) eos).intValue());
                        }
                    } else if (eosTokenObj instanceof Number) {
                        eosTokenIds.add(((Number) eosTokenObj).intValue());
                    }
                }
            }

            // 处理其他特殊token
            for (Map.Entry<String, Integer> entry : TOKEN_DEFAULTS.entrySet()) {
                String tokenName = entry.getKey();
                int defaultId = entry.getValue();

                if (getTokenId(tokenName) == 0) { // 只有在未设置时才设置
                    String tokenStr = getSpecialTokenContent(config, tokenName);
                    if (tokenStr != null) {
                        setTokenId(tokenName, vocab.getOrDefault(tokenStr,defaultId));
                    }
                }
            }
        }

        // 如果没有从配置中获取到EOS token，使用默认值
        if (eosTokenIds.isEmpty()) {
            eosTokenIds.add(1); // 默认EOS token
        }

        // 设置其他token的默认值
        setDefaultTokensIfUnset(TOKEN_DEFAULTS);
    }

    private Map<String, Object> loadJsonConfig(ResourceLoader resourceLoader, String configFile) throws IOException {
        try (InputStream inputStream = resourceLoader.open(configFile)) {
            return parser.parseConfig(inputStream);
        }
    }

    private int getTokenId(String tokenName) {
        switch (tokenName) {
            case "bos_token": return bosTokenId;
            case "pad_token": return padTokenId;
            case "unk_token": return unkTokenId;
            default: return 0; // eos_token 现在使用列表，不再是单个int
        }
    }

    private void setTokenId(String tokenName, int id) {
        switch (tokenName) {
            case "bos_token": bosTokenId = id; break;
            case "pad_token": padTokenId = id; break;
            case "unk_token": unkTokenId = id; break;
            // eos_token 不再使用单个int
        }
    }

    private void setDefaultTokensIfUnset(Map<String, Integer> defaults) {
        // EOS token 已经在loadSpecialTokens中处理了
        if (bosTokenId == 0) bosTokenId = vocab.getOrDefault("<bos>", defaults.get("bos_token"));
        if (padTokenId == 0) padTokenId = vocab.getOrDefault("<pad>", defaults.get("pad_token"));
        if (unkTokenId == 0) unkTokenId = vocab.getOrDefault("<unk>", defaults.get("unk_token"));
    }

    private String getSpecialTokenContent(Map<String, Object> node, String key) {
        if (!node.containsKey(key))
            return null;
        Object tokenObj = node.get(key);
        if (tokenObj instanceof String) {
            return (String) tokenObj;
        } else if (tokenObj instanceof Map) {
            @SuppressWarnings("unchecked")
            Map<String, Object> tokenMap = (Map<String, Object>) tokenObj;
            if (tokenMap.containsKey("content")) {
                return (String) tokenMap.get("content");
            }
        }
        return null;
    }

    /**
     * 加载额外的 token
     * 从 added_tokens.json 文件中读取额外的词汇表条目
     *  tokenizerDir tokenizer 目录路径
     *  抛出 IOException 如果文件读取失败
     */
    private void loadAddedTokens(ResourceLoader resourceLoader) throws IOException {
        if (!resourceLoader.exists("added_tokens.json")) {
            return;
        }

        Map<String, Integer> addedTokens;
        try (InputStream inputStream = resourceLoader.open("added_tokens.json")) {
            addedTokens = parser.parseAddedTokens(inputStream);
        }
        for (Map.Entry<String, Integer> entry : addedTokens.entrySet()) {
            String token = entry.getKey();
            int id = entry.getValue();
            if (!vocab.containsKey(token)) {
                putToken(token, id);
            }
        }
    }


    /**
     * 应用预分词器
     *
     * BPE算法的第一步：将输入文本分割成基本的词单元。
     * 这个方法按空格和换行符分割文本，为BPE编码做准备。
     *
     * 处理逻辑：
     * - 按空格分割单词
     * - 保留换行符作为单独的token（如果在词汇表中）
     * - 处理连续空格和换行符
     *
     *  text 输入文本
     *  返回分词后的字符串列表，每个元素是一个词或换行符
     */
    private List<String> applyPreTokenizer(String text) {
        List<String> words = new ArrayList<>();
        boolean hasNewlineToken = vocab.containsKey("\n");

        if (text.isEmpty()) {
            return words;
        }

        // 按空格和换行符分割
        int i = 0;
        while (i < text.length()) {
            // 查找下一个空格或换行符
            int nextSpace = text.indexOf(' ', i);
            int nextNewline = text.indexOf('\n', i);

            int nextSplit = text.length();
            boolean isNewline = false;

            if (nextNewline != -1 && (nextSpace == -1 || nextNewline < nextSpace)) {
                nextSplit = nextNewline;
                isNewline = true;
            } else if (nextSpace != -1) {
                nextSplit = nextSpace;
                isNewline = false;
            }

            // 提取当前部分
            if (nextSplit > i) {
                String segment = text.substring(i, nextSplit);
                if (!segment.isEmpty()) {
                    words.add(segment);
                }
            }

            if (isNewline) {
                // 添加换行符 token
                if (hasNewlineToken) {
                    words.add("\n");
                }
                i = nextSplit + 1;
            } else if (nextSpace != -1) {
                i = nextSplit + 1;
            } else {
                break;
            }
        }

        return words;
    }

    /**
     * 对单个词应用 BPE 编码
     *
     * 这是BPE算法的核心实现：
     * 1. 如果整个词已经在词汇表中，直接返回
     * 2. 将词分解为字符序列
     * 3. 按照合并规则，迭代地合并最频繁的字符对
     * 4. 确保所有子词都在词汇表中
     *
     * 算法特点：
     * - 贪婪策略：每次选择第一个可合并的规则
     * - 迭代限制：最多50次迭代避免无限循环
     * - 回退策略：如果找不到匹配，使用UNK token
     *
     *  word 输入的单词字符串
     *  返回 BPE编码后的子词列表
     */
    private List<String> applyBPEToWord(String word) {
        // 检查整个词是否已经在vocab中
        if (vocab.containsKey(word)) {
            return Arrays.asList(word);
        }

        // 将词分解为字符序列
        List<String> tokens = new ArrayList<>();
        for (char c : word.toCharArray()) {
            tokens.add(String.valueOf(c));
        }

        // 限制迭代次数
        for (int iter = 0; iter < Math.min(MAX_BPE_MERGE_ITERATIONS, merges.size()); iter++) {
            int minRuleIndex = Integer.MAX_VALUE;
            int minIndex = -1;
            String mergedToken = null;

            // 扫描当前tokens序列，找到可以合并的相邻对，并选择规则索引最小的那个
            for (int i = 0; i < tokens.size() - 1; i++) {
                String key = tokens.get(i) + " " + tokens.get(i + 1);
                MergeRule rule = mergeRulesMap.get(key);
                if (rule != null && rule.index < minRuleIndex) {
                    minRuleIndex = rule.index;
                    minIndex = i;
                    mergedToken = rule.mergedToken;
                }
            }

            // 如果没有找到可以合并的相邻对，退出循环
            if (minIndex == -1) {
                break;
            }

            // 合并
            tokens.set(minIndex, mergedToken);
            tokens.remove(minIndex + 1);
        }

        // 确保所有token都在vocab中，否则尝试最长匹配或使用UNK
        List<String> result = new ArrayList<>();
        for (String token : tokens) {
            if (vocab.containsKey(token)) {
                result.add(token);
            } else {
                result.add(unkTokenText);
            }
        }

        return result;
    }

    /**
     * 提取特殊 token
     *
     * 在BPE编码前，优先识别和提取特殊标记。
     * 这些特殊token具有特定的语义含义，不能被BPE分割。
     *
     * 处理的特殊token包括：
     * - 函数调用标记：<start_function_call>、<end_function_call>
     * - 对话标记：<start_of_turn>、<end_of_turn>
     * - 转义标记：<escape>
     * - 其他控制标记
     *
     *
     *  text 输入文本
     *  返回处理后的字符串列表，特殊token和普通文本片段交替出现
     */
    private List<String> extractSpecialTokens(String text) {
        List<String> parts = new ArrayList<>();
        int i = 0;
        while (i < text.length()) {
            boolean found = false;
            for (String specialToken : activeSpecialTokens) {
                if (text.startsWith(specialToken, i)) {
                    parts.add(specialToken);
                    i += specialToken.length();
                    found = true;
                    break;
                }
            }
            if (!found) {
                // 找到下一个特殊 token 的位置，或者到文本末尾
                int nextSpecial = text.length();
                for (String specialToken : activeSpecialTokens) {
                    int pos = text.indexOf(specialToken, i);
                    if (pos != -1 && pos < nextSpecial) {
                        nextSpecial = pos;
                    }
                }
                if (nextSpecial > i) {
                    parts.add(text.substring(i, nextSpecial));
                    i = nextSpecial;
                } else {
                    parts.add(text.substring(i));
                    break;
                }
            }
        }
        return parts;
    }

    /**
     * 执行完整的 BPE 编码流程
     *  text 输入文本
     *  返回编码后的token列表
     */
    private List<String> bpeEncode(String text) {
        List<String> cached = bpeCache.get(text);
        if (cached != null) {
            return cached;
        }

        // 步骤 0: 提取特殊 token
        List<String> parts = extractSpecialTokens(text);

        List<String> allTokens = new ArrayList<>(Math.max(16, text.length() / 2));
        for (String part : parts) {
            if (part.isEmpty())
                continue;

            // 检查是否是特殊 token（已经在 vocab 中）
            if (vocab.containsKey(part)) {
                allTokens.add(part);
                continue;
            }

            // 应用 pre-tokenizer（按空格分割）
            List<String> words = applyPreTokenizer(part);

            // 步骤 3: 对每个词应用 BPE
            for (String word : words) {
                if (word.isEmpty())
                    continue;

                // 检查词是否已经在 vocab 中（完整词）
                if (vocab.containsKey(word)) {
                    allTokens.add(word);
                } else {
                    // 应用 BPE 编码
                    List<String> wordTokens = applyBPEToWord(word);
                    // 将 BPE tokens 转换为 vocab 中的 tokens
                    for (String token : wordTokens) {
                        if (vocab.containsKey(token)) {
                            allTokens.add(token);
                        } else {
                            // 如果 token 不在 vocab 中，直接使用 UNK
                            allTokens.add(unkTokenText);
                        }
                    }
                }
            }
        }

        // 缓存结果
        bpeCache.put(text, allTokens);
        return allTokens;
    }

    /**
     * 应用聊天模板并进行BPE编码
     *
     * 这是ManualBPETokenizer的核心接口方法，将结构化的对话数据转换为模型可处理的token序列。
     *
     * 处理流程：
     * 1. 应用FunctionGemma聊天模板格式化消息和工具定义
     * 2. 生成符合FunctionGemma模型要求的对话字符串
     * 3. 对格式化文本进行BPE编码
     * 4. 生成attention_mask和其他必要输入
     *
     * 模板格式：
     * - BOS token开始
     * - 工具声明
     * - developer/system消息
     * - user/model轮替的对话
     * - 特殊的控制token标记对话结构
     *
     *  messages 消息列表，每个消息包含role和content
     *  tools 工具定义列表，每个工具包含name、description、parameters
     *  addGenerationPrompt 是否添加生成提示（用于对话开始）
     *  返回 TokenizerResult 包含inputIds、attentionMask等编码结果
     */
    @Override
    public TokenizerResult applyChatTemplate(
            List<Map<String, Object>> messages,
            List<Map<String, Object>> tools,
            boolean addGenerationPrompt) {

        // 使用简化的聊天模板实现
        String text = buildSimpleChatTemplate(messages, tools, addGenerationPrompt);
        if (DEBUG_TEMPLATE) {
            System.out.println("调试: 输入文本 = " + text);
        }

        // 使用 BPE 编码
        List<String> tokens = bpeEncode(text);

        // 转换为 token IDs
        long[] inputIds = new long[tokens.size()];
        long[] attentionMask = new long[tokens.size()];

        for (int i = 0; i < tokens.size(); i++) {
            inputIds[i] = vocab.getOrDefault(tokens.get(i),unkTokenId);
            attentionMask[i] = 1;
        }

        return new Tokenizer.TokenizerResult(inputIds, attentionMask);
    }

    private String buildSimpleChatTemplate(
            List<Map<String, Object>> messages,
            List<Map<String, Object>> tools,
            boolean addGenerationPrompt) {
        return ChatTemplateBuilder.build(
                getTokenById(bosTokenId),
                messages,
                tools,
                addGenerationPrompt
        );
    }



    /**
     * 解码 token IDs 为文本
     */
    /**
     * 解码 token IDs 为文本
     *  tokenIds token ID 数组
     *  skipSpecialTokens 是否跳过特殊token
     *  返回解码后的文本
     */
    @Override
    public String decode(long[] tokenIds, boolean skipSpecialTokens) {
        StringBuilder sb = new StringBuilder();
        for (long id : tokenIds) {
            int tokenId = (int) id;
            if (skipSpecialTokens) {
                if (eosTokenIdSet.contains(tokenId) || tokenId == bosTokenId || tokenId == padTokenId) {
                    continue;
                }
            }

            String token = getTokenById((int) id);
            if (token == null)
                continue;

            // 移除 BPE 标记（简化处理）
            token = token.replace("Ġ", " ");
            sb.append(token);
        }
        return sb.toString();
    }

    /**
     * 获取结束token的ID
     *  返回主要的EOS token ID
     */
    @Override
    public int getEosTokenId() {
        return eosTokenIds.isEmpty() ? 1 : eosTokenIds.get(0);
    }

    /**
     * 获取所有EOS token IDs
     *  返回 EOS token ID列表
     */
    @Override
    public List<Integer> getEosTokenIds() {
        return new ArrayList<>(eosTokenIds);
    }

    /**
     * 检查token是否存在于词汇表中
     *  token 要检查的token
     *  返回是否存在
     */
    @Override
    public boolean hasToken(String token) {
        return vocab.containsKey(token);
    }

    /**
     * 将token转换为ID
     *  token 输入token
     *  返回 token ID，不存在则返回UNK token ID
     */
    @Override
    public int convertTokenToId(String token) {
        return vocab.getOrDefault(token, unkTokenId);
    }

    /**
     * 获取词汇表大小
     *  返回词汇表中的token数量
     */
    public int getVocabSize() {
        return vocab.size();
    }

    private void putToken(String token, int id) {
        vocab.put(token, id);
        ensureTokenCapacity(id);
        idToToken.set(id, token);
    }

    private void ensureTokenCapacity(int id) {
        while (idToToken.size() <= id) {
            idToToken.add(null);
        }
    }

    private String getTokenById(int id) {
        if (id < 0 || id >= idToToken.size()) {
            return null;
        }
        return idToToken.get(id);
    }

    private interface ResourceLoader {
        boolean exists(String relativePath) throws IOException;

        InputStream open(String relativePath) throws IOException;
    }

    private static final class DirectoryResourceLoader implements ResourceLoader {
        private final File baseDir;

        private DirectoryResourceLoader(String tokenizerDir) {
            this.baseDir = new File(tokenizerDir);
        }

        @Override
        public boolean exists(String relativePath) {
            return new File(baseDir, relativePath).exists();
        }

        @Override
        public InputStream open(String relativePath) throws IOException {
            return new FileInputStream(new File(baseDir, relativePath));
        }
    }

    private static final class AssetResourceLoader implements ResourceLoader {
        private final AssetManager assetManager;
        private final String assetBasePath;

        private AssetResourceLoader(AssetManager assetManager, String assetBasePath) {
            this.assetManager = assetManager;
            this.assetBasePath = normalizeAssetBasePath(assetBasePath);
        }

        @Override
        public boolean exists(String relativePath) {
            try (InputStream ignored = open(relativePath)) {
                return true;
            } catch (IOException ignored) {
                return false;
            }
        }

        @Override
        public InputStream open(String relativePath) throws IOException {
            String path = assetBasePath.isEmpty() ? relativePath : assetBasePath + "/" + relativePath;
            return assetManager.open(path);
        }

        private static String normalizeAssetBasePath(String raw) {
            if (raw == null) {
                return "";
            }
            String normalized = raw.trim();
            while (normalized.startsWith("/")) {
                normalized = normalized.substring(1);
            }
            while (normalized.endsWith("/")) {
                normalized = normalized.substring(0, normalized.length() - 1);
            }
            return normalized;
        }
    }

    private static List<String> createSortedSpecialTokens() {
        List<String> tokens = new ArrayList<>(Arrays.asList(
                "<start_function_response>", "<end_function_response>",
                "<start_function_call>", "<end_function_call>",
                "<start_function_declaration>", "<end_function_declaration>",
                "<start_of_turn>", "<end_of_turn>",
                "<start_of_image>", "<end_of_image>",
                "<escape>",
                "<bos>", "<eos>", "<pad>", "<unk>"
        ));
        tokens.sort((a, b) -> Integer.compare(b.length(), a.length()));
        return Collections.unmodifiableList(tokens);
    }

}
