import re
import jieba
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from wordcloud import WordCloud
from collections import Counter
import numpy as np
import os
import warnings
import jieba.posseg as pseg

# 设置警告过滤器
matplotlib.use('Agg')  # 在导入pyplot之前设置
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


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


# 2. 数据读取
def read_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()
def clean_text(text):
    """文本清洗，保留基本标点"""
    # 保留中文、英文、数字和基本标点
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s，。！？：；、]', '', text)
    # 合并连续空白字符
    text = re.sub(r'\s+', ' ', text)
    # 移除标点前的空格
    text = re.sub(r'\s([，。！？：；、])', r'\1', text)
    return text.strip()

# 新增函数：保存清洗后的文本
def save_cleaned_text(text, output_file='cleaned_travel_diaries.txt'):
    """保存清洗后的文本到文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"已保存清洗后的数据到: {output_file}")
    return output_file
# 4. 分词处理 - 优化分词质量
def tokenize_text(text):
    jieba.setLogLevel(jieba.logging.INFO)
    # 添加旅行相关专业词汇
    travel_words = [
        '雷峰塔', '摇橹船', '三潭印月', '河坊街', '知味观', '布宫', '大昭寺', '酥油灯',
        '月牙泉', '莫高窟', '洱海', '双廊', '冰雪大世界', '中央大街', '松花江', '鼓浪屿',
        '黄山', '婺源', '草原', '香港', '北京胡同', '狗拉爬犁', '臭鳜鱼', '断桥', '三脚架',
        '冲锋衣', '羽绒服', '南屏晚钟', '马迭尔', '西安', "敦煌", "婺源", "黄山", "洱海",
        "布达拉宫", "松花江", "冰雪大世界", "河坊街", "大昭寺", "莫高窟", "鼓浪屿", "双廊",
        "中央大街", "月牙泉", "雷峰塔", "摇橹船", "三潭印月", "知味观", "西湖", "杭州", "断桥",
        "河坊街", "知味观", "摇橹船", "三潭印月", "鸬鹚", "草鞋", "狼毫", "青石板", "苏轼",
        "龙井茶", "虾鲜", "猫耳朵汤", "桐油伞", "雷峰塔", "保俶塔", "北山街", "青石板"
    ]
    for word in travel_words:
        jieba.add_word(word, freq=1000)  # 提高专业词汇的权重
    # 使用词性标注过滤无效词
    words = []
    # 允许的词性：名词、动词、形容词、地名、机构名
    allowed_pos = {'n', 'v', 'a', 'ns', 'nt', 'nz', 'vn', 'an'}

    # 分句处理以提高分词质量
    sentences = re.split(r'[，。！？：；、]', text)
    for sentence in sentences:
        if not sentence.strip():
            continue
        # 使用精确模式分词
        seg_list = pseg.cut(sentence)
        for word, flag in seg_list:
            # 只保留有意义的中文词语（长度至少为1）
            if re.fullmatch(r'[\u4e00-\u9fa5]+', word):
                # 只保留特定词性的词语
                if flag[0] in allowed_pos:
                    words.append(word)
    return words


# 5. 停用词处理
def load_stopwords(file_path='chinese_stopwords.txt'):
    # 基础停用词
    basic_stopwords = {
        '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一个', '上', '也',
        '很', '到', '说', '要', '这', '那', '你', '他', '她', '它', '我们', '你们', '他们',
        '自己', '这', '那', '哪', '着', '过', '吧', '吗', '啊', '哦', '嗯', '哈', '啦', '呀',
        '得', '地', '之', '像', '时', '出', '看', '老', '有', '没有', '什么', '怎么', '这样',
        '那样', '这个', '那个', '这些', '那些', '一种', '一样', '可以', '可能', '应该', '一些','指着' }
    # 自定义旅行日记停用词
    travel_stopwords = {
        '日记', '实录', '日志', '记录', '血泪史', '大作战', '求生记','缝里','震得',
        '血泪', '殉情', '作死', '敢死队', '小姑奶奶', '女娃子','手机',
        '像是', '有的', '没有', '什么', '怎么', '时候', '地方', '东西','差点','活像','发出','大爷' }
    # 从文件加载停用词
    file_stopwords = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if word and not word.startswith('#') and len(word) > 1:
                    if re.search(r'[\u4e00-\u9fa5]', word):
                        file_stopwords.add(word)
    except FileNotFoundError:
        pass  # 忽略文件不存在的情况

    all_stopwords = basic_stopwords | travel_stopwords | file_stopwords
    return all_stopwords

def remove_stopwords(words, stopwords):
    """停用词过滤"""
    return [word for word in words
            if word not in stopwords
            and not word.isdigit()
            and len(word) > 1]  # 过滤单字词


# 6. 词云生成 - 优化视觉效果
def generate_wordcloud(filtered_words, font_path=None):
    if not filtered_words:
        print("没有词语可用于生成词云")
        return

    # 创建词频统计
    word_freq = Counter(filtered_words)

    # 如果没有高频词，使用所有词语
    if word_freq:
        # 生成词云
        try:
            wordcloud = WordCloud(
                font_path=font_path or "simhei.ttf",
                width=1200,
                height=800,
                background_color='white',
                max_words=150,
                colormap='viridis',
                collocations=False,
                prefer_horizontal=0.85,
                min_font_size=10,  # 减小最小字体
                max_font_size=150,
                relative_scaling=0.8
            ).generate_from_frequencies(word_freq)

            plt.figure(figsize=(16, 12), facecolor='white')
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig('travel_wordcloud.png', bbox_inches='tight', dpi=300, facecolor='white')
            plt.show()
        except Exception as e:
            print(f"生成词云时出错: {e}")
    else:
        print("没有足够词语生成词云")


# 7. 词频统计图 - 优化视觉效果
def plot_word_frequency(filtered_words, top_n=25):  # 增加显示数量
    if not filtered_words:
        print("没有词语可用于生成词频图")
        return

    word_counts = Counter(filtered_words)
    top_words = word_counts.most_common(top_n)

    if not top_words:
        return

    words, counts = zip(*top_words)
    idx = np.arange(len(words))

    plt.figure(figsize=(14, 10), facecolor='white')  # 调整尺寸
    bars = plt.barh(idx, counts, color='#2c7bb6', height=0.7)  # 更改颜色
    plt.yticks(idx, words, fontsize=12)  # 减小字体
    plt.xlabel('出现频率', fontsize=12)
    plt.title(f'旅行日记高频词汇 Top {top_n}', fontsize=16, pad=15)  # 减小标题
    plt.gca().invert_yaxis()
    plt.grid(axis='x', linestyle='--', alpha=0.3)

    # 添加数值标签
    for i, bar in enumerate(bars):
        plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 str(counts[i]), ha='left', va='center', fontsize=10)  # 减小字体

    plt.tight_layout()
    plt.savefig('word_frequency.png', dpi=300, facecolor='white')
    plt.show()


# 主流程 - 简化并优化
def main():
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.facecolor'] = 'white'

    # 读取原始数据
    input_file = 'raw_travel_diaries.txt'
    raw_data = read_data(input_file)
    print(f"原始数据字符数: {len(raw_data)}")

    # 数据清洗
    cleaned_data = clean_text(raw_data)
    print(f"清洗后数据字符数: {len(cleaned_data)}")

    # 保存清洗后的文本 - 这是新增的步骤
    cleaned_file = save_cleaned_text(cleaned_data)

    # 分词处理
    words = tokenize_text(cleaned_data)
    print(f"分词数量: {len(words)}")

    # 加载停用词
    stopwords = load_stopwords()

    # 去除停用词和数字
    filtered_words = remove_stopwords(words, stopwords)
    print(f"过滤后词语数量: {len(filtered_words)}")

    # 获取中文字体
    font_path = find_chinese_font()

    # 生成可视化结果
    generate_wordcloud(filtered_words, font_path)
    plot_word_frequency(filtered_words, top_n=10)

    # 保存处理后的数据
    with open('processed_travel_diaries.txt', 'w', encoding='utf-8') as f:
        f.write(' '.join(filtered_words))


if __name__ == "__main__":
    main()