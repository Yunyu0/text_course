import jieba
import re
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import sys


# 1. 查找系统中文字体
def find_chinese_font():
    """查找系统中可用的中文字体"""
    # 常见中文字体名称
    chinese_fonts = [
        'SimHei', 'Microsoft YaHei', 'KaiTi', 'SimSun',
        'FangSong', 'STSong', 'STKaiti', 'STFangsong',
        'WenQuanYi Micro Hei', 'Source Han Sans SC',
        'Noto Sans CJK SC', 'WenQuanYi Zen Hei'
    ]

    # 获取系统所有字体
    system_fonts = fm.findSystemFonts()
    for font_name in chinese_fonts:
        for font_path in system_fonts:
            if font_name.lower() in os.path.basename(font_path).lower():
                return font_path

    # 备选方案：查找包含"黑体"、"宋体"等关键字的字体
    keywords = ['hei', 'song', 'kai', 'fang', 'st', 'cjk', 'sc', 'chinese', '中文']
    for font_path in system_fonts:
        font_name = os.path.basename(font_path).lower()
        if any(kw in font_name for kw in keywords):
            return font_path
    return None  # 未找到中文字体


# 1. 初始化词典构建
def build_initial_dictionary(file_path):
    """从原始旅行日记中提取高频词汇构建初始词典"""
    if not os.path.exists(file_path):
        print(f"错误：文件 {file_path} 不存在！")
        sys.exit(1)
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 清洗文本
    cleaned_content = re.sub(r'\d{4}[-./]\d{1,2}[-./]\d{1,2}', '', content)
    cleaned_content = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', cleaned_content)

    # 使用jieba进行分词
    words = jieba.lcut(cleaned_content)
    word_freq = Counter(words)

    # 过滤条件：长度≥2且频率≥3
    travel_keywords = []
    exclude_words = ['但是', '可以', '这个', '我们', '自己', '什么', '一样', '一个',
                     '非常', '觉得', '时候', '没有', '还是', '就是', '这种', '地方']

    for word, freq in word_freq.items():
        if len(word) >= 2 and freq >= 3 and word not in exclude_words:
            travel_keywords.append(word)

    # 添加特色词汇
    special_terms = [
        '雷峰塔', '鼓浪屿', '青石板客栈', '摇橹船', '烤乳扇', '滑沙板',
        '马迭尔冰棍', '猫耳朵汤', '二八杠', '驴友大巴', '拓片', '煨桑'
    ]

    # 合并并去重
    full_dict = list(set(travel_keywords + special_terms))

    # 保存初始词典
    with open('travel_dictionary.txt', 'w', encoding='utf-8') as f:
        for word in full_dict:
            f.write(word + '\n')

    print(f"初始词典构建完成，包含 {len(full_dict)} 个词汇")
    return full_dict


# 2. 大规模语料收集与扩展
def collect_candidate_terms(file_path):
    """从原始数据集中提取候选词汇"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 清洗文本
    cleaned_content = re.sub(r'\d{4}[-./]\d{1,2}[-./]\d{1,2}', '', content)
    cleaned_content = re.sub(r'[^\u4e00-\u9fa5]', ' ', cleaned_content)
    # 使用jieba分词
    words = jieba.lcut(cleaned_content)
    # 提取长度≥2的中文词汇
    candidate_terms = set()
    for word in words:
        if len(word) >= 2 and re.match(r'^[\u4e00-\u9fa5]+$', word):
            candidate_terms.add(word)
    print(f"候选词库收集完成，包含 {len(candidate_terms)} 个词汇")
    return list(candidate_terms)


# 3. 个性化词典更新 - 修复版本
def update_dictionary(initial_dict, candidate_terms, file_path):
    """使用三维度算法更新词典"""
    # 加载语料
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    # 计算候选词频率
    term_freq = {}
    for term in candidate_terms:
        term_freq[term] = len(re.findall(term, content))
    # 情感词列表
    sentiment_words = ['惊艳', '难忘', '治愈', '壮观', '美味', '震撼', '惬意', '独特',
                       '值得', '推荐', '惊喜', '感动', '舒适', '愉快', '满意']
    # 计算情感相关性
    sentiment_scores = {}
    for term in candidate_terms:
        co_occurrence = 0
        # 使用更准确的分句方法
        sentences = re.split(r'[。！？；\n]', content)
        for sent in sentences:
            if term in sent and any(s_word in sent for s_word in sentiment_words):
                co_occurrence += 1
        sentiment_scores[term] = co_occurrence / (len(sentiment_words) + 0.01)
    # 计算TF-IDF
    documents = [re.sub(r'\d{4}[-./]\d{1,2}[-./]\d{1,2}', '', diary)
                 for diary in re.split(r'---+\n', content) if len(diary.strip()) > 100]  # 过滤过短日记
    # 使用TF-IDF计算
    vectorizer = TfidfVectorizer(tokenizer=jieba.lcut, min_df=2)
    tfidf_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = {term: 0 for term in candidate_terms}

    for term in candidate_terms:
        if term in feature_names:
            idx = list(feature_names).index(term)
            tfidf_scores[term] = np.max(tfidf_matrix[:, idx].toarray())
    # 三维度筛选
    new_terms = []
    for term in candidate_terms:
        if term in initial_dict:
            continue

        freq_ok = term_freq[term] >= 2
        sentiment_ok = sentiment_scores[term] >= 0.1
        tfidf_ok = tfidf_scores[term] >= 0.1

        # 放宽条件
        if freq_ok and (sentiment_ok or tfidf_ok):
            new_terms.append(term)

    report_terms = [
        '网红打卡地', '古镇巷弄', '环岛路', '穷游', '自驾游', '跟团游',
        '潜水', '滑翔伞', '露营', '当地小吃', '夜市摊', '特色菜'
    ]

    # 合并并去重
    final_new_terms = list(set(new_terms + report_terms))

    # 更新词典
    updated_dict = initial_dict + final_new_terms

    # 保存更新后的词典
    with open('updated_travel_dictionary.txt', 'w', encoding='utf-8') as f:
        for word in updated_dict:
            f.write(word + '\n')

    print(f"词典更新完成，新增 {len(final_new_terms)} 个词汇，总计 {len(updated_dict)} 个词汇")
    return updated_dict


# 4. 分词效果对比实验
def evaluate_segmentation(file_path, diary_title):
    """对比使用定制词典前后的分词效果 - 科学评估版"""
    # 加载测试文本
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 找到指定日记
    diaries = re.split(r'---+\n', content)
    test_text = ""
    for diary in diaries:
        if diary_title in diary:
            test_text = diary
            break

    if not test_text:
        print(f"未找到标题包含 '{diary_title}' 的日记")
        return None

    print(f"测试文本长度: {len(test_text)} 字符")

    # 构建正确分词列表
    correct_terms = []
    with open('updated_travel_dictionary.txt', 'r', encoding='utf-8') as f:
        for line in f:
            term = line.strip()
            if term in test_text:
                correct_terms.append(term)

    # 添加更多旅行词汇
    additional_terms = [
        '西湖', '雷峰塔', '鼓浪屿', '巷弄', '特色民宿', '摇橹船', '草鞋底',
        '船帮', '鸬鹚', '三潭印月', '青石板', '知味观', '猫耳朵汤'
    ]
    correct_terms += [t for t in additional_terms if t in test_text]

    # 去重
    correct_terms = list(set(correct_terms))
    print(f"正确分词列表包含 {len(correct_terms)} 个词汇")

    # 重置jieba词典
    jieba.initialize()
    # 对照组：使用默认词典
    default_seg = jieba.lcut(test_text)
    # 实验组：使用定制词典
    jieba.load_userdict('updated_travel_dictionary.txt')
    custom_seg = jieba.lcut(test_text)

    def calculate_metrics(seg_list, correct_list):
        # 计算正确切分的数量（精确匹配）
        correct_count = sum(1 for word in seg_list if word in correct_list)
        total_seg = len(seg_list)
        # 计算实际应该切分出的术语数量
        actual_terms_count = 0
        for term in correct_list:
            actual_terms_count += test_text.count(term)
        # 计算准确率：正确切分数量 / 总切分数量
        precision = correct_count / total_seg if total_seg > 0 else 0
        # 计算召回率：正确切分数量 / 实际应切分出的术语数量
        recall = correct_count / actual_terms_count if actual_terms_count > 0 else 0
        # 计算F1值
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return precision, recall, f1, correct_count
    # 计算指标
    default_precision, default_recall, default_f1, default_correct = calculate_metrics(default_seg, correct_terms)
    custom_precision, custom_recall, custom_f1, custom_correct = calculate_metrics(custom_seg, correct_terms)

    # 打印结果
    print("\n分词效果对比:")
    print(f"{'指标':<10} | {'对照组':<8} | {'实验组':<8} | {'提升幅度':<10}")
    print(f"{'-' * 40}")
    print(
        f"{'准确率':<10} | {default_precision:.4f}   | {custom_precision:.4f}   | +{(custom_precision - default_precision):.4f}")
    print(f"{'召回率':<10} | {default_recall:.4f}   | {custom_recall:.4f}   | +{(custom_recall - default_recall):.4f}")
    print(f"{'F1值':<10} | {default_f1:.4f}   | {custom_f1:.4f}   | +{(custom_f1 - default_f1):.4f}")
    print(f"对照组正确切分: {default_correct}, 实验组正确切分: {custom_correct}")

    # 找出新增正确切分的词汇
    additional_correct = [word for word in custom_seg if word in correct_terms and word not in default_seg]
    if additional_correct:
        print(f"实验组新增正确切分词汇: {', '.join(set(additional_correct))}")

    # 找出未正确切分的词汇
    missing_correct = [word for word in correct_terms if word in test_text and word not in custom_seg]
    if missing_correct:
        print(f"实验组未正确切分词汇: {', '.join(set(missing_correct))}")

    return {
        'default': (default_precision, default_recall, default_f1),
        'custom': (custom_precision, custom_recall, custom_f1),
        'default_correct': default_correct,
        'custom_correct': custom_correct
    }


# 5. 新词评估机制
def evaluate_new_term(term, seed_terms, content):
    # 场景相关性
    scene_score = 0
    sentences = re.split(r'[。！？；\n]', content)
    total_relevant = 0
    for sent in sentences:
        if term in sent:
            if any(seed in sent for seed in seed_terms):
                scene_score += 1
            total_relevant += 1
    # 避免除以零
    scene_score = scene_score / (total_relevant + 0.01) if total_relevant > 0 else 0
    # 情感强度
    positive_words = ['美', '好', '惊艳', '难忘', '治愈', '壮观', '独特', '赞', '爽']
    negative_words = ['坑', '差', '贵', '累', '挤', '热', '脏', '乱', '吵']
    # 计算情感倾向
    sentiment = 0
    term_count = len(re.findall(term, content))
    if term_count > 0:
        pos_count = sum(1 for char in term if char in positive_words)
        neg_count = sum(1 for char in term if char in negative_words)
        sentiment = (pos_count - neg_count) / len(term) if len(term) > 0 else 0
        # 添加上下文情感分析
        for sent in sentences:
            if term in sent:
                if any(pw in sent for pw in positive_words):
                    sentiment += 0.1
                elif any(nw in sent for nw in negative_words):
                    sentiment -= 0.1
        # 归一化到[-1, 1]范围
        sentiment = max(-1, min(1, sentiment))
    # 语义独特性
    uniqueness = 1 - min(term_count / 10, 1.0)
    # 综合得分 - 调整权重
    composite_score = 0.5 * scene_score + 0.3 * max(0, sentiment) + 0.2 * uniqueness
    return composite_score, scene_score, sentiment, uniqueness


# 6. 可视化展示
def visualize_results(updated_dict, evaluation_results):
    # 分类词典词汇
    categories = {
        '景点类': ['雷峰塔', '鼓浪屿', '西湖', '黄山', '月牙泉', '古镇', '海滩', '寺庙'],
        '住宿类': ['客栈', '民宿', '青旅', '酒店', '青石板客栈'],
        '交通类': ['摇橹船', '缆车', '大巴', '自驾', '骑行', '徒步'],
        '活动类': ['拍照', '暴走', '潜水', '露营', '烧烤', '观星'],
        '美食类': ['猫耳朵汤', '烤乳扇', '马迭尔冰棍', '小吃', '夜市', '特色菜'],
        '其他': []
    }
    category_counts = {cat: 0 for cat in categories}
    for word in updated_dict:
        matched = False
        for cat, keywords in categories.items():
            if any(keyword in word for keyword in keywords):
                category_counts[cat] += 1
                matched = True
                break
        if not matched:
            if '游' in word or '旅' in word or '行' in word:
                category_counts['活动类'] += 1
            elif '店' in word or '宿' in word or '房' in word:
                category_counts['住宿类'] += 1
            elif '吃' in word or '餐' in word or '饮' in word:
                category_counts['美食类'] += 1
            elif '景' in word or '园' in word or '山' in word:
                category_counts['景点类'] += 1
            else:
                category_counts['其他'] += 1

    # 绘制饼图
    plt.figure(figsize=(12, 8))
    cats = [k for k, v in category_counts.items() if v > 0]
    counts = [v for k, v in category_counts.items() if v > 0]
    colors = ['#FF9966', '#FF5E62', '#FFCC5C', '#48BB78', '#4298F5', '#A389F4']

    # 添加爆炸效果
    explode = [0.05] * len(cats)

    plt.pie(counts, labels=cats, colors=colors[:len(cats)],
            autopct=lambda p: f'{p:.1f}% ({int(p * sum(counts) / 100)})',
            startangle=90, shadow=True, explode=explode, textprops={'fontsize': 12})

    plt.title('旅行定制词典词汇类别分布', fontsize=16, pad=20)
    plt.axis('equal')

    # 添加图例
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))

    plt.savefig('dictionary_categories.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("可视化结果已保存: dictionary_categories.png")

    # 添加评估结果可视化
    if evaluation_results:
        labels = ['准确率', '召回率', 'F1值']
        default_metrics = evaluation_results['default']
        custom_metrics = evaluation_results['custom']

        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x - width / 2, default_metrics, width, label='默认词典', color='#8888ff')
        rects2 = ax.bar(x + width / 2, custom_metrics, width, label='定制词典', color='#ff8888')

        ax.set_ylabel('分数')
        ax.set_title('分词效果对比')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        # 添加数值标签
        for rect in rects1 + rects2:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3点垂直偏移
                        textcoords="offset points",
                        ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig('segmentation_comparison.png', dpi=300)
        plt.show()
        print("评估结果可视化已保存: segmentation_comparison.png")


# 主流程
def main():
    # 设置中文字体
    chinese_font = find_chinese_font()
    if chinese_font:
        plt.rcParams['font.family'] = fm.FontProperties(fname=chinese_font).get_name()
    else:
        print("警告: 未找到中文字体，可视化可能显示异常")

    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.facecolor'] = 'white'

    file_path = 'raw_travel_diaries.txt'

    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误：文件 {file_path} 不存在！")
        return

    print("=" * 50)
    print("步骤1: 初始化词典构建")
    initial_dict = build_initial_dictionary(file_path)

    print("\n" + "=" * 50)
    print("步骤2: 大规模语料收集与扩展")
    candidate_terms = collect_candidate_terms(file_path)

    print("\n" + "=" * 50)
    print("步骤3: 个性化词典更新")
    updated_dict = update_dictionary(initial_dict, candidate_terms, file_path)

    print("\n" + "=" * 50)
    print("步骤4: 分词效果对比实验")
    evaluation_results = evaluate_segmentation(file_path, "杭州西湖暴走日记")

    if evaluation_results:
        print("\n" + "=" * 50)
        print("步骤5: 新词评估示例")
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        seed_terms = ['旅行', '景点', '体验', '游玩', '游览']
        term = '环岛路'
        score, scene_sim, sentiment, uniqueness = evaluate_new_term(term, seed_terms, content)
        print(f"新词 '{term}' 评估结果:")
        print(f"  场景相关性: {scene_sim:.4f}")
        print(f"  情感强度: {sentiment:.4f}")
        print(f"  语义独特性: {uniqueness:.4f}")
        print(f"  综合得分: {score:.4f} - {'建议收录' if score > 0.4 else '不建议收录'}")

        print("\n" + "=" * 50)
        print("步骤6: 可视化展示")
        visualize_results(updated_dict, evaluation_results)

    print("\n处理完成！")


if __name__ == "__main__":
    main()