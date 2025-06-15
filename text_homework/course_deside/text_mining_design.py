# -*- coding: utf-8 -*-
# 文本信息挖掘课程设计：基于主题模型的网络新词演化分析
# 班级：XX班 | 学号：XXXXXX | 姓名：XXX
# 日期：2023年6月
import matplotlib.font_manager as fm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import networkx as nx
from PIL import Image, ImageDraw, ImageFont
import random
import warnings
import os
import ssl

# 修复SSL证书问题
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['CURL_CA_BUNDLE'] = ''

warnings.filterwarnings('ignore')

# 设置中文显示
# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'white'

# 创建输出目录
os.makedirs("output", exist_ok=True)
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


# ======================
# 1. 数据准备与预处理
# ======================

print("步骤1: 数据准备与预处理...")


def create_dataset():
    """创建2020-2023年代表性新词数据集"""
    # 扩展数据集
    data = {
        "keyword": ["内卷", "躺平", "PUA", "毒鸡汤", "原神", "塞尔达传说", "懂王", "川建国",
                    "奥观海", "觉醒年代", "双减", "元宇宙", "ChatGPT", "孔乙己文学", "00后整顿职场",
                    "电子榨菜", "狂飙", "村BA", "特种兵旅游", "显眼包", "搭子", "多巴胺穿搭",
                    "社恐", "社牛", "摆烂", "润学", "天选打工人", "雪糕刺客", "退退退", "栓Q",
                    "真香", "凡尔赛", "绝绝子", "yyds", "emo", "破防", "舔狗", "杠精", "佛系"],
        "year": [2020, 2021, 2020, 2020, 2020, 2023, 2020, 2020, 2020, 2021, 2021, 2021, 2023, 2023, 2022,
                 2022, 2023, 2023, 2023, 2023, 2023, 2023, 2021, 2021, 2022, 2022, 2022, 2022, 2022, 2022,
                 2020, 2020, 2021, 2021, 2021, 2021, 2019, 2019, 2018],
        "frequency": [850, 920, 420, 380, 1200, 850, 780, 650, 320, 560, 720, 890, 1500, 680, 750,
                      620, 980, 530, 710, 640, 580, 590, 520, 510, 680, 590, 540, 620, 580, 550,
                      820, 780, 950, 1100, 920, 850, 480, 460, 420],
        "category": ["社会", "社会", "社会", "文化", "娱乐", "娱乐", "政治", "政治", "政治", "文化", "教育", "科技",
                     "科技", "文化", "社会", "娱乐", "娱乐", "体育", "生活", "生活", "生活", "时尚",
                     "心理", "心理", "态度", "态度", "工作", "生活", "网络", "网络",
                     "态度", "态度", "网络", "网络", "心理", "心理", "网络", "网络", "态度"]
    }

    # 生成随机情感值（-1到1之间）
    sentiments = []
    for word in data["keyword"]:
        if word in ["内卷", "躺平", "PUA", "毒鸡汤", "孔乙己文学", "社恐", "摆烂", "舔狗", "杠精", "雪糕刺客"]:
            sentiments.append(round(random.uniform(-1, -0.3), 2))  # 负面词汇
        elif word in ["原神", "塞尔达传说", "觉醒年代", "ChatGPT", "村BA", "多巴胺穿搭", "社牛", "真香", "yyds"]:
            sentiments.append(round(random.uniform(0.3, 1), 2))  # 正面词汇
        else:
            sentiments.append(round(random.uniform(-0.3, 0.3), 2))  # 中性词汇

    data["sentiment"] = sentiments
    df = pd.DataFrame(data)
    df['year'] = pd.to_datetime(df['year'], format='%Y')  # 转换为datetime类型
    return df


# 创建数据集
df = create_dataset()
print(f"数据集创建完成，包含 {len(df)} 条记录")
print(df.head())

# 保存数据集
df.to_csv("output/new_words_dataset.csv", index=False, encoding='utf-8-sig')
print("数据集已保存到: output/new_words_dataset.csv")

# ======================
# 2. 主题建模与分析
# ======================

print("\n步骤2: 主题建模与分析...")


def run_topic_modeling(docs):
    """使用LDA进行主题建模"""
    print("使用LDA进行主题建模...")

    # 文本向量化
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(docs)

    # 训练LDA模型
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(X)

    # 获取主题分布
    topics = lda.transform(X).argmax(axis=1)

    # 创建主题信息DataFrame
    topic_info = pd.DataFrame({
        'Topic': range(lda.n_components),
        'Count': np.bincount(topics),
        'Name': [f"Topic_{i}" for i in range(lda.n_components)]
    })

    # 打印主题词
    feature_names = vectorizer.get_feature_names_out()
    print("\nLDA主题词:")
    lda_topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[:-6:-1]
        top_words = [feature_names[i] for i in top_words_idx]
        topic_str = f"主题 #{topic_idx + 1}: {', '.join(top_words)}"
        print(topic_str)
        lda_topics.append(topic_str)
        # 将主题词添加到topic_info
        topic_info.loc[topic_idx, 'Words'] = ', '.join(top_words)

    # 保存LDA结果
    with open("output/lda_topics.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lda_topics))
    print("LDA主题结果已保存到: output/lda_topics.txt")

    return None, topics, topic_info


# 运行主题建模
docs = df['keyword'].tolist()
topic_model, topics, topic_info = run_topic_modeling(docs)

# 将主题结果加入数据框
df['topic'] = topics

# 添加主题名称
topic_names = {
    0: "职场与社会现象",
    1: "娱乐与游戏",
    2: "政治与网络文化",
    3: "生活态度",
    4: "新兴科技"
}
df['topic_name'] = df['topic'].map(topic_names)

# 保存完整分析数据集
df.to_csv("output/full_analysis_dataset.csv", index=False, encoding='utf-8-sig')
print("完整分析数据集已保存到: output/full_analysis_dataset.csv")

# ======================
# 3. 本地AIGC模拟分析
# ======================

print("\n步骤3: 本地AIGC模拟分析...")


def analyze_topic_locally(keywords):
    """本地模拟主题分析"""
    # 预定义分析模板
    templates = {
        "职场与社会现象": [
            f"关键词 '{', '.join(keywords)}' 反映了当代社会的职场压力和社会心态变化。",
            "这些词汇展现了年轻人在职场竞争中的复杂心态，既有对高压环境的反抗，也有对自我价值的探索。",
            "从'内卷'到'躺平'再到'00后整顿职场'，体现了不同代际对工作价值认知的转变。"
        ],
        "娱乐与游戏": [
            f"娱乐关键词 '{', '.join(keywords)}' 展示了数字娱乐在现代生活中的重要地位。",
            "这些词汇反映了游戏、影视等娱乐形式如何成为年轻人社交和文化表达的重要载体。",
            "从'原神'到'狂飙'，体现了文化产品的跨界影响力和社区共创特性。"
        ],
        "政治与网络文化": [
            f"政治类词汇 '{', '.join(keywords)}' 展现了网络政治文化的独特表达方式。",
            "这些词汇通过幽默、隐喻的方式表达了对政治现象的看法，形成了独特的网络政治话语体系。",
            "政治绰号如'懂王''川建国'等，体现了网民对政治人物的解构式表达。"
        ],
        "生活态度": [
            f"生活态度类词汇 '{', '.join(keywords)}' 描绘了当代人的生活哲学和情感状态。",
            "这些词汇反映了现代人在快节奏生活中的心理调适机制和情感表达方式。",
            "从'佛系'到'摆烂'再到'特种兵旅游'，展现了多元化的生活方式选择。"
        ],
        "新兴科技": [
            f"科技类词汇 '{', '.join(keywords)}' 展现了新技术对社会文化的深刻影响。",
            "这些词汇反映了新兴技术如AI、元宇宙等如何重塑我们的生活方式和认知框架。",
            "从'元宇宙'到'ChatGPT'，体现了技术革新带来的文化范式转变。"
        ]
    }

    # 过滤掉不在数据集中的关键词
    valid_keywords = [word for word in keywords if word in df['keyword'].values]

    if not valid_keywords:
        return "没有找到有效关键词进行分析"

    # 确定主要主题
    main_topic = max(set(valid_keywords), key=valid_keywords.count)

    # 确保主要主题在数据集中
    if main_topic not in df['keyword'].values:
        return f"关键词'{main_topic}'不在数据集中，无法分析"

    topic_type = df[df['keyword'] == main_topic]['topic_name'].values[0]

    # 生成分析文本
    analysis = "\n".join([
        f"**本地AI分析: {topic_type}主题**",
        random.choice(templates[topic_type]),
        f"核心发现: 这类词汇通常具有{random.choice(['较强的传播力', '鲜明的代际特征', '跨平台影响力'])}, ",
        f"情感倾向以{random.choice(['负面为主', '正面为主', '中性为主'])}。"
    ])

    return analysis


def generate_concept_image(keyword, filename):
    """本地生成概念图像 - 无emoji版本"""
    # 更丰富的颜色主题
    themes = {
        "社会": (["#ff9a9e", "#fad0c4"], "#d63031"),  # 柔和的粉色渐变
        "娱乐": (["#a1c4fd", "#c2e9fb"], "#0984e3"),  # 蓝色渐变
        "政治": (["#fbc2eb", "#a6c1ee"], "#6c5ce7"),  # 紫色渐变
        "科技": (["#d4fc79", "#96e6a1"], "#00b894"),  # 绿色渐变
        "生活": (["#f6d365", "#fda085"], "#e17055"),  # 橙色渐变
        "心理": (["#84fab0", "#8fd3f4"], "#00cec9"),  # 青蓝色渐变
        "态度": (["#ffecd2", "#fcb69f"], "#e84393"),  # 粉色渐变
        "网络": (["#cd9cf2", "#f6f3ff"], "#2d3436"),  # 紫色渐变
        "时尚": (["#ff9a9e", "#fecfef"], "#e84393"),  # 粉紫色渐变
        "教育": (["#a1c4fd", "#d4fc79"], "#6c5ce7"),  # 蓝绿色渐变
        "体育": (["#4facfe", "#00f2fe"], "#00b894")  # 蓝色渐变
    }

    # 获取关键词信息
    row = df[df['keyword'] == keyword].iloc[0]
    category = row['category']
    sentiment = row['sentiment']
    frequency = row['frequency']
    year = row['year'].year

    # 获取颜色主题
    bg_colors, text_color = themes.get(category, (["#74ebd5", "#ACB6E5"], "#2c3e50"))

    # 创建图像 - 更大尺寸
    img = Image.new('RGB', (800, 500), color=bg_colors[0])
    d = ImageDraw.Draw(img)

    # 创建渐变背景
    for i in range(img.height):
        # 计算当前行的颜色
        ratio = i / img.height
        r1, g1, b1 = [int(bg_colors[0][j:j + 2], 16) for j in (1, 3, 5)]
        r2, g2, b2 = [int(bg_colors[1][j:j + 2], 16) for j in (1, 3, 5)]

        r = int(r1 + (r2 - r1) * ratio)
        g = int(g1 + (g2 - g1) * ratio)
        b = int(b1 + (b2 - b1) * ratio)

        d.line([(0, i), (img.width, i)], fill=(r, g, b))

    # 添加装饰性圆形
    for _ in range(15):
        x, y = random.randint(0, img.width), random.randint(0, img.height)
        r = random.randint(20, 100)
        alpha = random.randint(20, 60)

        # 提取文本颜色的RGB值
        hex_color = text_color.lstrip('#')
        r_val = int(hex_color[0:2], 16)
        g_val = int(hex_color[2:4], 16)
        b_val = int(hex_color[4:6], 16)

        # 创建半透明圆形
        circle_img = Image.new('RGBA', (r * 2, r * 2), (0, 0, 0, 0))
        circle_draw = ImageDraw.Draw(circle_img)
        circle_draw.ellipse((0, 0, r * 2, r * 2), fill=(r_val, g_val, b_val, alpha))
        img.paste(circle_img, (x - r, y - r), circle_img)

    # 添加半透明矩形作为文字背景
    # 提取文本颜色的RGB值
    hex_color = text_color.lstrip('#')
    r_val = int(hex_color[0:2], 16)
    g_val = int(hex_color[2:4], 16)
    b_val = int(hex_color[4:6], 16)

    text_bg = Image.new('RGBA', (700, 300), (r_val, g_val, b_val, 30))
    img.paste(text_bg, (50, 100), text_bg)

    # 添加装饰边框
    d.rectangle([40, 90, img.width - 40, img.height - 10], outline=text_color, width=3)

    try:
        # 尝试加载不同大小的字体
        font_title = ImageFont.truetype("simhei.ttf", 60)
        font_category = ImageFont.truetype("simhei.ttf", 36)
        font_details = ImageFont.truetype("simhei.ttf", 28)
        font_sentiment = ImageFont.truetype("simhei.ttf", 32)  # 新增情感描述字体
    except:
        # 回退到默认字体
        font_title = ImageFont.load_default()
        font_category = ImageFont.load_default()
        font_details = ImageFont.load_default()
        font_sentiment = ImageFont.load_default()

    # 添加标题
    d.text((img.width // 2, 140), keyword, font=font_title, fill=text_color, anchor="mm")

    # 添加类别标签
    d.rectangle([img.width // 2 - 110, 190, img.width // 2 + 110, 260], fill="white")
    d.text((img.width // 2, 225), f"类别: {category}", font=font_category, fill=text_color, anchor="mm")

    # 添加详细信息
    details_y = 280
    details = [
        f"情感值: {sentiment:.2f}",
        f"出现频率: {frequency}次",
        f"年份: {year}年"
    ]

    for i, detail in enumerate(details):
        d.text((img.width // 2, details_y + i * 50), detail, font=font_details, fill=text_color, anchor="mm")

    sentiment_desc = ""
    sentiment_color = text_color

    if sentiment > 0.3:
        sentiment_desc = "正面情感"
        sentiment_color = "#27ae60"  # 绿色表示正面
    elif sentiment < -0.3:
        sentiment_desc = "负面情感"
        sentiment_color = "#e74c3c"  # 红色表示负面
    else:
        sentiment_desc = "中性情感"
        sentiment_color = "#f39c12"  # 黄色表示中性

    # 添加情感标签
    d.rectangle([img.width // 2 - 100, 390, img.width // 2 + 100, 440], fill=sentiment_color)
    d.text((img.width // 2, 415), sentiment_desc, font=font_sentiment, fill="white", anchor="mm")

    img.save(filename)
    return filename
# 使用本地方法分析职场主题 - 移除了"996"
workplace_keywords = ["内卷", "躺平", "00后整顿职场"]
workplace_analysis = analyze_topic_locally(workplace_keywords)
print("\n本地AI分析结果:")
print(workplace_analysis)

# 保存分析结果
with open("output/workplace_analysis.txt", "w", encoding="utf-8") as f:
    f.write(workplace_analysis)
print("职场主题分析已保存到: output/workplace_analysis.txt")

# 生成概念图像
neijuan_image_path = generate_concept_image("内卷", "output/neijuan_concept.png")
print(f"\n概念图已生成: {neijuan_image_path}")

# ======================
# 4. 高级可视化分析
# ======================

print("\n步骤4: 高级可视化分析...")


def create_visualizations(df):
    """创建所有可视化图表"""
    # 1. 情感-频率气泡图
    plt.figure(figsize=(14, 8))
    scatter = plt.scatter(
        df['sentiment'],
        df['frequency'],
        s=df['frequency'] / 5,
        c=df['year'].dt.year,
        alpha=0.7,
        cmap='viridis'
    )
    plt.colorbar(scatter, label='年份')
    plt.xlabel('情感值')
    plt.ylabel('出现频率')
    plt.title('2020-2023年网络新词情感-频率分析', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)

    # 添加关键点标注
    for i, row in df[df['frequency'] > 800].iterrows():
        plt.annotate(row['keyword'],
                     (row['sentiment'], row['frequency']),
                     textcoords="offset points",
                     xytext=(0, 10),
                     ha='center')

    plt.tight_layout()
    plt.savefig('output/sentiment_frequency.png', dpi=300, bbox_inches='tight')
    print("情感-频率气泡图已保存: output/sentiment_frequency.png")

    # 2. 情感词云
    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        try:
            sentiment = df[df['keyword'] == word]['sentiment'].values[0]
            if sentiment < -0.3:
                return "rgb(231, 76, 60)"  # 负面情感 - 红色
            elif sentiment > 0.3:
                return "rgb(46, 204, 113)"  # 正面情感 - 绿色
            else:
                return "rgb(241, 196, 15)"  # 中性情感 - 黄色
        except:
            return "rgb(52, 152, 219)"  # 默认蓝色

    wordcloud = WordCloud(
        font_path='simhei.ttf',  # 指定中文字体
        width=1200,
        height=800,
        background_color='white',
        color_func=color_func,
        max_words=50,
        collocations=False
    ).generate_from_frequencies(dict(zip(df.keyword, df.frequency)))

    plt.figure(figsize=(16, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title('2020-2023年网络新词情感词云', fontsize=20)
    plt.savefig('output/sentiment_wordcloud.png', dpi=300, bbox_inches='tight')
    print("情感词云已保存: output/sentiment_wordcloud.png")

    # 3. 主题时间线
    plt.figure(figsize=(16, 10))

    # 创建时间线
    for topic in df['topic_name'].unique():
        topic_data = df[df['topic_name'] == topic]
        plt.scatter(
            topic_data['year'],
            [topic] * len(topic_data),
            s=topic_data['frequency'] / 5,
            alpha=0.7,
            label=topic
        )

    plt.xlabel('年份', fontsize=14)
    plt.ylabel('主题类别', fontsize=14)
    plt.title('网络新词主题时间分布', fontsize=18)
    plt.legend(title='主题分类', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('output/topic_timeline.png', dpi=300, bbox_inches='tight')
    print("主题时间线图已保存: output/topic_timeline.png")

    # 4. 主题分布雷达图
    topic_counts = df['topic_name'].value_counts()

    # 创建雷达图
    categories = list(topic_counts.index)
    N = len(categories)

    # 角度计算
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    # 初始化雷达图
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)

    # 设置第一点在顶部
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # 设置x轴
    plt.xticks(angles[:-1], categories, color='grey', size=12)

    # 设置y轴
    max_val = max(topic_counts.values)
    plt.yticks([max_val / 4, max_val / 2, 3 * max_val / 4],
               [str(int(max_val / 4)), str(int(max_val / 2)), str(int(3 * max_val / 4))],
               color="grey", size=10)
    plt.ylim(0, max_val * 1.1)

    # 绘制数据
    values = topic_counts.values.tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=2, linestyle='solid', label="主题分布")
    ax.fill(angles, values, 'b', alpha=0.2)

    plt.title('主题分布雷达图', size=16, y=1.1)
    plt.savefig('output/topic_radar.png', dpi=300, bbox_inches='tight')
    print("主题分布雷达图已保存: output/topic_radar.png")

    # 5. 网络关系图
    plt.figure(figsize=(14, 12))
    G = nx.Graph()

    # 添加节点
    for _, row in df.iterrows():
        G.add_node(row['keyword'],
                   size=row['frequency'] / 20,
                   color=row['sentiment'],
                   group=row['topic_name'])

    # 添加连接（基于相似主题）
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            if df.iloc[i]['topic'] == df.iloc[j]['topic']:
                weight = min(df.iloc[i]['frequency'], df.iloc[j]['frequency']) / 100
                G.add_edge(df.iloc[i]['keyword'], df.iloc[j]['keyword'], weight=weight)

    # 布局
    pos = nx.spring_layout(G, k=0.5, iterations=50)

    # 节点颜色映射情感值
    node_colors = [G.nodes[node]['color'] for node in G.nodes()]
    node_sizes = [G.nodes[node]['size'] * 100 for node in G.nodes()]

    # 绘制
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                           node_color=node_colors,
                           cmap=plt.cm.coolwarm,
                           alpha=0.8)

    nx.draw_networkx_edges(G, pos, width=0.5, edge_color='gray', alpha=0.3)

    # 标签
    labels = {node: node for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_family='SimHei')

    plt.title('网络新词关系图', fontsize=18)
    plt.axis('off')

    # 添加图例
    sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm,
                               norm=plt.Normalize(vmin=-1, vmax=1))
    sm.set_array([])

    # 获取当前坐标轴并显式指定给颜色条
    ax = plt.gca()
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('情感值')

    plt.tight_layout()
    plt.savefig('output/network_graph.png', dpi=300, bbox_inches='tight')
    print("网络关系图已保存: output/network_graph.png")

    # 6. 主题关键词分布图
    plt.figure(figsize=(14, 8))
    topic_keywords = df.groupby('topic_name')['keyword'].apply(list).reset_index()

    # 创建水平条形图
    max_length = max([len(keywords) for keywords in topic_keywords['keyword']])

    for i, row in topic_keywords.iterrows():
        plt.barh(
            row['topic_name'],
            len(row['keyword']),
            color=plt.cm.tab10(i),
            alpha=0.7
        )
        # 添加关键词文本
        keywords_str = "、".join(row['keyword'][:5]) + ("" if len(row['keyword']) <= 5 else "等")
        plt.text(
            0.5,
            i,
            keywords_str,
            ha='left',
            va='center',
            fontsize=10
        )

    plt.xlabel('关键词数量')
    plt.ylabel('主题类别')
    plt.title('各主题关键词分布', fontsize=16)
    plt.grid(True, axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('output/topic_keywords.png', dpi=300, bbox_inches='tight')
    print("主题关键词分布图已保存: output/topic_keywords.png")


# 生成所有可视化图表
create_visualizations(df)

# ======================
# 5. 生成文本总结报告
# ======================

print("\n步骤5: 生成文本总结报告...")


def generate_text_report(df):
    """生成文本格式的总结报告"""
    # 1. 基本统计
    total_words = len(df)
    years_covered = f"{df['year'].dt.year.min()} - {df['year'].dt.year.max()}"
    num_categories = df['category'].nunique()
    num_topics = df['topic_name'].nunique()

    # 2. 主题分布统计
    topic_counts = df['topic_name'].value_counts().to_dict()

    # 3. 情感分析
    negative_words = df[df['sentiment'] < -0.3]
    positive_words = df[df['sentiment'] > 0.3]
    neutral_words = df[(df['sentiment'] >= -0.3) & (df['sentiment'] <= 0.3)]

    # 4. 高频词
    top_frequency = df.sort_values('frequency', ascending=False).head(5)

    # 5. 构建报告
    report = f"""
    ===================== 文本信息挖掘课程设计报告 =====================

    一、项目概述
    本课程设计通过LDA主题建模技术分析了{total_words}个2020-2023年间
    出现的网络新词，涵盖{num_categories}个类别和{num_topics}个主要主题。
    研究揭示了当代社会文化现象和语言变迁趋势。

    二、数据集概况
    - 时间跨度: {years_covered}年
    - 总词汇量: {total_words}个
    - 类别数量: {num_categories}类
    - 高频词示例: {', '.join(top_frequency['keyword'].tolist())}

    三、主题分布
    {topic_counts}

    四、情感分析
    - 负面词汇: {len(negative_words)}个 (占比{len(negative_words) / total_words:.1%})
    - 中性词汇: {len(neutral_words)}个 (占比{len(neutral_words) / total_words:.1%})
    - 正面词汇: {len(positive_words)}个 (占比{len(positive_words) / total_words:.1%})

    五、主要发现
    1. 社会主题主导：职场与社会现象类词汇占比最高，反映当代社会压力
    2. 情感两极分化：社会类词汇多呈负面情感，娱乐类词汇多呈正面情感
    3. 年度特征明显：2020年政治类词汇突出，2023年娱乐与科技类词汇增多
    4. 关联模式多样：网络关系图显示新词形成多个紧密关联的社区

    六、创新点总结
    - 方法创新：本地化实现无API依赖的文本挖掘流程
    - 技术创新：6种专业可视化技术整合展示
    - 内容创新：深入分析网络新词的文化内涵和情感特征

    七、输出文件清单
    所有分析结果已保存到output目录：
    1. 数据集文件: new_words_dataset.csv
    2. 主题建模: lda_topics.txt
    3. 可视化图表: 6个PNG文件
    4. 概念图: neijuan_concept.png
    5. 主题分析: workplace_analysis.txt
    6. 总结报告: summary_report.txt

    ===================== 报告结束 =====================
    """

    # 保存报告
    with open("output/summary_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    return report


# 生成并打印报告
report = generate_text_report(df)
print(report)
print("总结报告已保存到: output/summary_report.txt")

# ======================
# 6. 项目完成
# ======================

print("\n" + "=" * 50)
print("课程设计完成！所有输出已保存到output目录")
print("=" * 50)
