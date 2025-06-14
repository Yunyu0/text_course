import re
import jieba
import numpy as np
import matplotlib.pyplot as plt
from gensim import corpora, models
from sklearn.metrics.pairwise import cosine_similarity
import umap.umap_ as umap
from collections import defaultdict
import os
import warnings
from wordcloud import WordCloud
from matplotlib.font_manager import FontProperties
import logging

# 配置日志
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger()

# 忽略警告
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# 设置中文字体
def set_chinese_font():
    """设置中文字体支持"""
    try:
        # 尝试查找系统中文字体
        from matplotlib.font_manager import findfont, FontProperties
        font = findfont(FontProperties(family=['SimHei', 'Microsoft YaHei', 'KaiTi', 'SimSun']))
        font_path = font
        if os.path.exists(font_path):
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = [os.path.basename(font_path).split('.')[0]]
            return font_path
    except Exception as e:
        logger.warning(f"设置中文字体失败: {e}")
    return None


# 数据预处理
def preprocess_data(file_path, dict_path, stopwords_path):
    # 加载词典
    if os.path.exists(dict_path):
        jieba.load_userdict(dict_path)
        logger.info(f"已加载自定义词典: {dict_path}")

    # 加载停用词
    stopwords = set()
    if os.path.exists(stopwords_path):
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            stopwords = set([line.strip() for line in f if line.strip()])

    # 添加旅行专用停用词
    travel_stopwords = {'日记', '实录', '血泪史', '作死', '小姑奶奶', '女娃子', '拍照', '看到', '时候',
                        '什么', '一个', '觉得', '时候', '什么', '一个', '突然', '结果', '活像', '仿佛'}
    all_stopwords = stopwords.union(travel_stopwords)

    # 读取数据
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件不存在: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 更健壮的分割方式 - 匹配多种分隔符格式
    diaries = re.split(r'(?:\n-{3,}\s*\n)|(?:\n-{3,}\n)', content)
    diaries = [d.strip() for d in diaries if d.strip()]

    logger.info(f"找到 {len(diaries)} 个日记片段")

    diary_titles = []
    diary_texts = []

    # 提取标题和内容
    for diary in diaries:
        if len(diary) < 100:
            continue  # 跳过过短的片段

        # 尝试提取标题 - 多种格式
        title = ""
        title_match = re.search(r'^(\d{4}[-./]\d{1,2}[-./]\d{1,2}\s+([^\n]+))', diary)
        if not title_match:
            title_match = re.search(r'^([^\n]+日记)\s*\n', diary)
        if not title_match:
            title_match = re.search(r'^([^\n]+实录)\s*\n', diary)
        if not title_match:
            title_match = re.search(r'^([^\n]+日志)\s*\n', diary)

        if title_match:
            title = title_match.group(1).strip()
            # 提取日记内容（去掉标题部分）
            diary_content = diary.replace(title_match.group(0), '').strip()
        else:
            # 使用第一行作为标题
            first_line = diary.split('\n', 1)[0].strip()
            title = first_line
            diary_content = diary[len(first_line):].strip()

        # 简化标题 - 保留中文部分
        simple_title = re.sub(r'[^\u4e00-\u9fa5]', '', title)
        if not simple_title:
            simple_title = title[:15]  # 如果无中文，截取前15个字符

        diary_titles.append(simple_title)
        diary_texts.append(diary_content)

    logger.info(f"预处理完成，共处理 {len(diary_titles)} 篇旅行日记")
    logger.info(f"前5个标题示例: {diary_titles[:5]}")
    return diary_titles, diary_texts


# 构建词典和语料库
def build_corpus(diary_texts, stopwords):
    # 对每篇日记进行分词
    tokenized_diaries = []
    for text in diary_texts:
        # 清洗文本
        cleaned = re.sub(r'\d{4}[-./]\d{1,2}[-./]\d{1,2}', '', text)
        cleaned = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s,。!?:;、]', ' ', cleaned)
        # 分词
        words = jieba.lcut(cleaned)
        # 过滤停用词和短词
        filtered = [word for word in words if word not in stopwords and len(word) > 1]
        tokenized_diaries.append(filtered)

    # 创建词典，调整过滤参数
    dictionary = corpora.Dictionary(tokenized_diaries)

    # 更严格的过滤策略
    dictionary.filter_extremes(
        no_below=2,  # 至少出现在2篇文档中
        no_above=0.6,  # 最多出现在60%文档中
        keep_n=1000  # 保留最多1000个词
    )

    # 如果词典为空，使用未过滤的词典
    if len(dictionary) == 0:
        logger.warning("过滤后词典为空，使用未过滤词典")
        dictionary = corpora.Dictionary(tokenized_diaries)

    corpus = [dictionary.doc2bow(text) for text in tokenized_diaries]
    logger.info(f"词典构建完成，包含 {len(dictionary)} 个词")
    return dictionary, corpus, tokenized_diaries


# LSA主题建模
def train_lsa_model(corpus, dictionary, num_topics=4):
    # 构建TF-IDF模型
    tfidf = models.TfidfModel(corpus)
    tfidf_corpus = tfidf[corpus]

    # 训练LSA模型
    lsa = models.LsiModel(tfidf_corpus, id2word=dictionary, num_topics=num_topics)

    # 打印主题
    logger.info("LSA主题分析结果:")
    topics = []
    for i, topic in lsa.print_topics(num_topics=num_topics, num_words=10):
        topic_str = f"主题 {i + 1}: {topic}"
        logger.info(topic_str)
        topics.append(topic_str)

    return lsa, tfidf_corpus, topics


# LDA主题建模
def train_lda_model(corpus, dictionary, num_topics=4, passes=20):
    # 训练LDA模型，增加迭代次数
    lda = models.LdaModel(
        corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=passes,
        alpha='auto',
        eta='auto',
        random_state=42,
        iterations=400,  # 增加迭代次数
        minimum_probability=0.01  # 设置最小概率阈值
    )

    # 打印主题
    logger.info("LDA主题分析结果:")
    topics = []
    for i, topic in lda.print_topics(num_topics=num_topics, num_words=10):
        topic_str = f"主题 {i + 1}: {topic}"
        logger.info(topic_str)
        topics.append(topic_str)

    return lda, topics


# 获取文档主题向量
def get_topic_vectors(model, corpus, model_type='lda'):
    topic_vectors = []
    for doc in corpus:
        if model_type == 'lda':
            topic_dist = model[doc]
            vec = [0] * model.num_topics
            for topic_id, prob in topic_dist:
                vec[topic_id] = prob
            topic_vectors.append(vec)
        else:  # lsa
            vec = model[doc]
            vec = [val for _, val in vec]
            # 确保向量长度一致
            if len(vec) < model.num_topics:
                vec += [0] * (model.num_topics - len(vec))
            topic_vectors.append(vec[:model.num_topics])
    return np.array(topic_vectors)


# 主题可视化
def visualize_topics(topic_vectors, titles, method='UMAP', filename='topic_umap.png'):
    if len(topic_vectors) == 0:
        logger.warning("无法可视化，主题向量为空")
        return None

    # 确保有足够的文档进行可视化
    n_neighbors = min(5, len(topic_vectors) - 1) if len(topic_vectors) > 1 else 1

    if method == 'UMAP':
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors, min_dist=0.3)
        embedding = reducer.fit_transform(topic_vectors)
    else:  # PCA
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
        embedding = reducer.fit_transform(topic_vectors)

    plt.figure(figsize=(12, 10))
    plt.scatter(embedding[:, 0], embedding[:, 1], s=150, alpha=0.8, c='skyblue', edgecolor='navy')

    # 添加标签
    for i, title in enumerate(titles):
        short_title = title[:4] if len(title) > 4 else title
        plt.annotate(short_title,
                     (embedding[i, 0], embedding[i, 1]),
                     fontsize=10, ha='center', va='bottom')

    plt.title('旅行日记主题分布可视化', fontsize=16)
    plt.xlabel(f'{method}维度1', fontsize=12)
    plt.ylabel(f'{method}维度2', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"已保存主题分布图: {filename}")
    return embedding


# 生成词云
def generate_topic_wordcloud(lda_model, topic_id, font_path=None, filename='topic_wordcloud.png'):
    try:
        # 获取主题词分布
        topic_words = lda_model.show_topic(topic_id, topn=20)
        word_freq = {word: weight for word, weight in topic_words}

        # 生成词云
        wordcloud = WordCloud(
            font_path=font_path,
            width=800,
            height=600,
            background_color='white',
            max_words=50,
            colormap='viridis',
            prefer_horizontal=0.8
        ).generate_from_frequencies(word_freq)

        plt.figure(figsize=(10, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'主题 {topic_id + 1} 关键词云', fontsize=16)
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"已保存词云图: {filename}")
    except Exception as e:
        logger.error(f"生成词云时出错: {e}")


# 构建推荐系统
def build_recommendation_system(topic_vectors, diary_titles):
    if len(topic_vectors) < 2:
        logger.warning("文档数量不足，无法构建推荐系统")
        return None

    # 计算相似度矩阵
    similarity_matrix = cosine_similarity(topic_vectors)

    # 推荐函数
    def recommend(diary_idx, k=3):
        # 获取当前日记的相似度排序
        sim_scores = list(enumerate(similarity_matrix[diary_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        # 排除自身
        sim_scores = sim_scores[1:k + 1]
        # 返回推荐结果
        recommendations = [(diary_titles[idx], score) for idx, score in sim_scores]
        return recommendations

    return recommend


# 主函数
def main():
    # 文件路径
    data_file = 'raw_travel_diaries.txt'
    dict_file = 'updated_travel_dictionary.txt'
    stopwords_file = 'chinese_stopwords.txt'

    # 设置中文字体
    font_path = set_chinese_font()

    try:
        # 数据预处理 - 返回原始文本
        diary_titles, diary_texts = preprocess_data(data_file, dict_file, stopwords_file)

        # 加载停用词
        stopwords = set()
        if os.path.exists(stopwords_file):
            with open(stopwords_file, 'r', encoding='utf-8') as f:
                stopwords = set([line.strip() for line in f if line.strip()])

        # 添加旅行专用停用词
        travel_stopwords = {'日记', '实录', '血泪史', '作死', '小姑奶奶', '女娃子', '拍照',  '看到', '时候',
                            '什么', '一个', '觉得', '时候', '什么', '一个', '突然', '结果', '活像','仿佛'}
        all_stopwords = stopwords.union(travel_stopwords)

        # 构建词典和语料库
        dictionary, corpus, tokenized_diaries = build_corpus(diary_texts, all_stopwords)

        # 检查语料库是否有效
        if len(corpus) == 0 or len(dictionary) == 0:
            logger.error("语料库或词典为空，无法进行主题建模")
            return

        # 确定主题数量
        num_topics = min(4, len(diary_titles), len(dictionary))
        if num_topics < 2:
            logger.warning(f"主题数量太少 ({num_topics})，无法进行主题建模")
            return

        # 训练LSA模型
        lsa_model, lsa_corpus, lsa_topics = train_lsa_model(corpus, dictionary, num_topics=num_topics)

        # 训练LDA模型
        lda_model, lda_topics = train_lda_model(corpus, dictionary, num_topics=num_topics, passes=50)

        # 获取LDA主题向量
        lda_topic_vectors = get_topic_vectors(lda_model, corpus, 'lda')

        # 可视化LDA主题分布
        if lda_topic_vectors is not None and len(lda_topic_vectors) > 0:
            visualize_topics(lda_topic_vectors, diary_titles, 'UMAP', 'lda_topic_umap.png')

        # 生成主题词云
        if lda_model and lda_model.num_topics > 0:
            for i in range(lda_model.num_topics):
                generate_topic_wordcloud(lda_model, i, font_path, f'topic_{i + 1}_wordcloud.png')

        # 构建推荐系统
        if lda_topic_vectors is not None and len(lda_topic_vectors) > 1:
            recommend = build_recommendation_system(lda_topic_vectors, diary_titles)

            if recommend:
                logger.info("推荐示例:")
                # 创建标题映射
                example_titles = [
                    "杭州西湖", "西安城墙", "敦煌沙漠", "大理洱海", "黄山作死",
                    "哈尔滨", "鼓浪屿", "拉萨", "婺源", "草原蚊子"
                ]

                # 只使用实际存在的标题
                valid_titles = []
                for title in example_titles:
                    if any(title in t for t in diary_titles):
                        valid_titles.append(title)
                    else:
                        logger.warning(f"未找到包含 '{title}' 的日记")

                # 对存在的标题进行推荐
                for title in valid_titles:
                    # 查找包含关键词的标题索引
                    indices = [i for i, t in enumerate(diary_titles) if title in t]
                    if indices:
                        idx = indices[0]
                        logger.info(f"\n与《{diary_titles[idx]}》相似的旅行日记：")
                        k = min(3, len(diary_titles) - 1)
                        recommendations = recommend(idx, k=k)
                        for rec_title, score in recommendations:
                            logger.info(f"- 《{rec_title}》, 相似度: {score:.4f}")
                    else:
                        logger.warning(f"未找到日记: {title}")

        # 保存主题模型
        if lsa_model:
            lsa_model.save('lsa_model.model')
        if lda_model:
            lda_model.save('lda_model.model')
        logger.info("主题模型已保存")

        # 保存主题分析结果
        with open('topic_analysis.txt', 'w', encoding='utf-8') as f:
            if lsa_topics:
                f.write("LSA主题分析结果:\n")
                f.write("\n".join(lsa_topics))
            if lda_topics:
                f.write("\n\nLDA主题分析结果:\n")
                f.write("\n".join(lda_topics))

        logger.info("主题建模与推荐系统构建完成！")

    except Exception as e:
        logger.error(f"程序运行出错: {e}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()