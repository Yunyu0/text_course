import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import re
import os
import sys
from collections import defaultdict, Counter
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import umap.umap_ as umap
import seaborn as sns
import pandas as pd
import nltk
from nltk.corpus import stopwords
import jieba

# 设置中文字体
zh_fonts = [
    'SimHei', 'Microsoft YaHei', 'KaiTi', 'STKaiti',
    'STXihei', 'STHeiti', 'STSong', 'STFangsong'
]
for font in zh_fonts:
    if font in fm.get_font_names():
        plt.rcParams['font.sans-serif'] = [font]
        plt.rcParams['axes.unicode_minus'] = False
        break
else:
    try:
        font_path = "SimHei.ttf"
        font_prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = font_prop.get_name()
        plt.rcParams['axes.unicode_minus'] = False
    except:
        print("警告: 无法加载中文字体，图表可能无法正确显示中文")

# 下载中文停用词
nltk.download('stopwords')
cn_stopwords = set(stopwords.words('chinese'))


# 1. 数据加载与预处理
def load_data():
    # 加载旅行日记
    with open('raw_travel_diaries.txt', 'r', encoding='utf-8') as f:
        diaries = f.read().split('---\n')


    diary_titles = []
    for diary in diaries:
        if diary.strip():
            # 获取第一行作为标题行
            title_line = diary.strip().split('\n')[0]
            date_patterns = [
                r'^\d{4}[-./]\d{1,2}[-./]\d{1,2}\s*',  # 2023.3.15
                r'^\d{4}[-./]\d{1,2}\s*',  # 2023.3
                r'^\d{4}\s*'  # 2023
            ]
            for pattern in date_patterns:
                title_line = re.sub(pattern, '', title_line)
            title = title_line.strip() if title_line.strip() else diary.strip().split('\n')[0]
            diary_titles.append(title)

    # 加载定制词典
    with open('updated_travel_dictionary.txt', 'r', encoding='utf-8') as f:
        custom_dict = [line.strip() for line in f.readlines()]

    # 添加到jieba词典
    for word in custom_dict:
        jieba.add_word(word)

    # 文本预处理函数
    def preprocess_text(text):
        # 分词
        words = jieba.cut(text)
        # 过滤停用词和非中文
        words = [word for word in words if word not in cn_stopwords and re.match(r'^[\u4e00-\u9fa5]+$', word)]
        return words

    # 预处理所有日记
    processed_diaries = []
    for diary in diaries:
        if diary.strip():
            processed_diaries.append(preprocess_text(diary))

    return processed_diaries, diary_titles, custom_dict

# 2. 构建词语共现图
def build_word_graph(words_list, window_size=5, min_cooccur=1):  # 降低阈值到1
    graph = defaultdict(dict)
    word_freq = Counter()
    # 统计词频
    for words in words_list:
        for word in words:
            word_freq[word] += 1
    # 构建共现图
    for words in words_list:
        for i, word in enumerate(words):
            if word not in custom_dict:
                continue
            # 滑动窗口内词语共现
            start = max(0, i - window_size)
            end = min(len(words), i + window_size + 1)
            for j in range(start, end):
                if i != j and j < len(words):
                    neighbor = words[j]
                    if neighbor in custom_dict and neighbor != word:
                        # 更新共现计数
                        graph[word][neighbor] = graph[word].get(neighbor, 0) + 1
    # 转换为NetworkX图
    G = nx.Graph()
    for word, neighbors in graph.items():
        for neighbor, weight in neighbors.items():
            if weight >= min_cooccur:
                G.add_edge(word, neighbor, weight=weight)
    # 添加孤立节点
    for word in custom_dict:
        if word not in G:
            G.add_node(word)
    return G, word_freq


# 3. 随机游走生成序列
def random_walk(start_node, graph, walk_length):
    if start_node not in graph:
        return [start_node]
    walk = [start_node]
    current = start_node
    for _ in range(walk_length - 1):
        neighbors = list(graph.neighbors(current))
        if not neighbors:
            break
        # 获取边权重
        weights = [graph.get_edge_data(current, n)['weight'] for n in neighbors]
        total_weight = sum(weights)
        # 计算选择概率
        probs = [w / total_weight for w in weights]
        next_node = np.random.choice(neighbors, p=probs)
        walk.append(next_node)
        current = next_node
    return walk


# 4. DeepWalk算法实现
def deepwalk(graph, num_walks=15, walk_length=20, vector_dim=100):
    all_walks = []
    nodes = list(graph.nodes())

    if len(nodes) < 20:
        num_walks = max(num_walks, 50)
    for _ in range(num_walks):
        np.random.shuffle(nodes)
        for node in nodes:
            walk = random_walk(node, graph, walk_length)
            if len(walk) > 1:
                all_walks.append(walk)
    # 训练Word2Vec模型
    model = Word2Vec(
        sentences=all_walks,
        vector_size=vector_dim,
        window=3,
        min_count=0,
        sg=1,
        workers=4,
        epochs=20
    )
    return model


# 5. 词语相似度分析
def word_similarity_analysis(model, target_word):
    if target_word in model.wv:
        try:
            similar_words = model.wv.most_similar(target_word, topn=5)
            print(f"与'{target_word}'最相似的词语：")
            for word, similarity in similar_words:
                print(f"  {word}: {similarity:.4f}")
            return similar_words
        except KeyError:
            print(f"词语'{target_word}'不在词汇表中")
    else:
        print(f"词语'{target_word}'不在词向量模型中")
    return []


# 6. 可视化词向量
def visualize_word_vectors(model, custom_dict):
    words = [word for word in custom_dict if word in model.wv]
    if not words:
        print("没有可用的词向量进行可视化")
        return None, None
    vectors = np.array([model.wv[word] for word in words])
    # UMAP降维
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(5, len(words) - 1))
    embeddings = reducer.fit_transform(vectors)
    # 创建聚类标签
    partitions = {}
    category_keywords = {
        "景点": ["西湖", "鼓浪屿", "黄山", "雷峰塔", "长城", "故宫", "草原", "大学城"],
        "活动": ["潜水", "露营", "骑行", "暴走", "滑雪", "徒步"],
        "美食": ["冰棍", "烤乳扇", "泡馍", "酥油", "火锅", "小吃", "臭豆腐", "幽兰拿铁"],
        "情感": ["难忘", "惊艳", "值得", "突然", "惊喜", "感动"],
        "交通": ["缆车", "大巴", "自驾", "骑行", "徒步", "飞机"]
    }
    for word in words:
        for category, keywords in category_keywords.items():
            if any(kw in word for kw in keywords):
                partitions[word] = category
                break
        else:
            partitions[word] = "其他"
    # 可视化
    plt.figure(figsize=(12, 10))
    categories = set(partitions.values())
    colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
    for i, category in enumerate(categories):
        idx = [j for j, word in enumerate(words) if partitions[word] == category]
        if not idx:
            continue
        plt.scatter(
            embeddings[idx, 0],
            embeddings[idx, 1],
            color=colors[i],
            label=category,
            s=100,
            alpha=0.7
        )

        # 添加词语标签 - 只添加部分重要词
        for j in idx:
            if word_freq.get(words[j], 0) > 3:
                plt.annotate(
                    words[j],
                    (embeddings[j, 0], embeddings[j, 1]),
                    fontsize=9,
                    alpha=0.8
                )
    plt.title("旅行词汇语义空间可视化 (UMAP降维)", fontsize=16)
    plt.xlabel("UMAP维度1", fontsize=14)
    plt.ylabel("UMAP维度2", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.2)
    plt.savefig('travel_words_umap.png', dpi=300, bbox_inches='tight')
    plt.show()

    return embeddings, words


# 7. 旅行日记推荐系统
def diary_recommendation(model, processed_diaries, diary_titles, custom_dict):
    diary_vectors = []
    valid_indices = []
    for idx, diary in enumerate(processed_diaries):
        # 获取日记中在词汇表中的词向量
        vectors = [model.wv[word] for word in diary if word in model.wv and word in custom_dict]
        if vectors:
            # 计算平均向量
            diary_vector = np.mean(vectors, axis=0)
            diary_vectors.append(diary_vector)
            valid_indices.append(idx)
    if not diary_vectors:
        print("没有足够的词向量进行日记推荐")
        return None
    diary_vectors = np.array(diary_vectors)
    # 计算相似度矩阵
    similarity_matrix = cosine_similarity(diary_vectors)
    # 如果日记数量少，调整热力图大小
    n_diaries = len(valid_indices)
    fig_size = (max(8, n_diaries), max(6, n_diaries * 0.8))
    # 可视化相似度矩阵
    plt.figure(figsize=fig_size)
    ax = sns.heatmap(
        similarity_matrix,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        xticklabels=[diary_titles[i] for i in valid_indices],
        yticklabels=[diary_titles[i] for i in valid_indices]
    )
    plt.title("旅行日记相似度矩阵", fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    # 保存图片
    plt.savefig('diary_recommendation.png', dpi=300, bbox_inches='tight')
    plt.show()
    # 使用KNN进行推荐
    knn = NearestNeighbors(n_neighbors=min(4, n_diaries), metric='cosine')
    knn.fit(diary_vectors)
    recommendations = {}
    for i, vec in enumerate(diary_vectors):
        distances, indices = knn.kneighbors([vec])
        # 排除自身
        rec_indices = []
        for idx in indices[0]:
            if valid_indices[idx] != valid_indices[i]:
                rec_indices.append(valid_indices[idx])
        rec_indices = rec_indices[:3]  # 最多推荐3篇
        recommendations[valid_indices[i]] = rec_indices
    # 打印推荐结果
    print("\n旅行日记推荐结果:")
    for idx, recs in recommendations.items():
        print(f"\n日记: {diary_titles[idx]}")
        if recs:
            print("推荐日记:")
            for rec_idx in recs:
                # 获取相似度
                i_idx = valid_indices.index(idx)
                r_idx = valid_indices.index(rec_idx)
                sim = similarity_matrix[i_idx, r_idx]
                print(f"  - {diary_titles[rec_idx]} (相似度: {sim:.4f})")
        else:
            print("没有找到相关推荐")
    return recommendations


# 8. 三维度新词评估机制
def new_word_evaluation(new_word, model, graph, seed_words=["景点", "体验", "推荐", "活动"]):
    result = {
        "新词": new_word,
        "在词典中": new_word in custom_dict,
        "在图中": new_word in graph,
        "在词向量中": new_word in model.wv,
        "图中心性": 0,
        "语义距离": 0,
        "共现频率": 0,
        "综合评分": 0
    }
    # 1. 图中心性 (PageRank)
    if result["在图中"]:
        try:
            pagerank_scores = nx.pagerank(graph, weight='weight')
            result["图中心性"] = pagerank_scores.get(new_word, 0)
        except Exception as e:
            print(f"PageRank计算错误: {e}")
    # 2. 语义距离
    if result["在词向量中"]:
        new_word_vector = model.wv[new_word]
        seed_vectors = [model.wv[word] for word in seed_words if word in model.wv]
        if seed_vectors:
            similarities = [cosine_similarity([new_word_vector], [vec])[0][0] for vec in seed_vectors]
            result["语义距离"] = np.mean(similarities)
    # 3. 共现频率
    if result["在图中"]:
        try:
            result["共现频率"] = sum([graph[new_word][n] for n in graph[new_word]])
        except:
            pass
    # 加权综合评分
    weights = [0.4, 0.3, 0.3]
    normalized_freq = min(result["共现频率"] / 50, 1.0) if result["共现频率"] > 0 else 0
    result["综合评分"] = (
            weights[0] * result["图中心性"] +
            weights[1] * result["语义距离"] +
            weights[2] * normalized_freq
    )

    return result


# 主函数 - 增强版本
def main():
    global custom_dict, word_freq

    print("=" * 50)
    print("基于随机游走的旅行文本语义关联分析与推荐系统")
    print("=" * 50)

    # 1. 加载数据
    print("\n[步骤1] 加载数据...")
    processed_diaries, diary_titles, custom_dict = load_data()
    print(f"  加载了 {len(processed_diaries)} 篇旅行日记")
    print(f"  定制词典包含 {len(custom_dict)} 个词汇")

    # 2. 构建词语共现图
    print("\n[步骤2] 构建词语共现图...")
    graph, word_freq = build_word_graph(processed_diaries)
    print(f"  构建了包含 {graph.number_of_nodes()} 个节点和 {graph.number_of_edges()} 条边的共现图")

    # 检查关键词是否在图中
    keywords = ["环岛路", "雷峰塔", "潜水", "西湖", "露营"]
    print("\n关键词检查:")
    for word in keywords:
        if word in graph:
            print(f"  '{word}' 在图中 (度: {graph.degree(word)})")
        else:
            print(f"  '{word}' 不在图中")

    # 3. 实现DeepWalk算法
    print("\n[步骤3] 训练DeepWalk模型...")
    model = deepwalk(graph)
    print(f"  DeepWalk模型训练完成 (词汇表大小: {len(model.wv)})")

    # 4. 词语相似度分析
    print("\n[步骤4] 词语相似度分析:")
    word_similarity_analysis(model, "环岛路")
    word_similarity_analysis(model, "雷峰塔")
    word_similarity_analysis(model, "潜水")

    # 5. 可视化词向量
    print("\n[步骤5] 生成词向量可视化...")
    embeddings, words = visualize_word_vectors(model, custom_dict)

    # 6. 旅行日记推荐系统
    print("\n[步骤6] 生成旅行日记推荐...")
    recommendations = diary_recommendation(model, processed_diaries, diary_titles, custom_dict)

    # 7. 新词评估示例
    print("\n[步骤7] 新词评估示例:")
    new_words = ["网红打卡", "亲子游", "背包客", "自由行", "美食探店"]
    evaluation_results = []

    for word in new_words:
        eval_result = new_word_evaluation(word, model, graph)
        evaluation_results.append(eval_result)
        print(f"\n评估结果: {word}")
        print(f"  在词典中: {eval_result['在词典中']}")
        print(f"  在图中: {eval_result['在图中']}")
        print(f"  在词向量中: {eval_result['在词向量中']}")
        print(f"  图中心性: {eval_result['图中心性']:.4f}")
        print(f"  语义距离: {eval_result['语义距离']:.4f}")
        print(f"  共现频率: {eval_result['共现频率']}")
        print(f"  综合评分: {eval_result['综合评分']:.4f}")

    # 保存评估结果
    if evaluation_results:
        eval_df = pd.DataFrame(evaluation_results)
        eval_df.to_csv('new_word_evaluation.csv', index=False, encoding='utf-8-sig')
        print("\n新词评估结果已保存到 new_word_evaluation.csv")

    print("\n" + "=" * 50)
    print("处理完成!")
    print("=" * 50)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()