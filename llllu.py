import os
import pandas as pd
import re
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic.vectorizers import ClassTfidfTransformer
import nltk
from nltk import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from textblob import Word


def english_word_cut(mytext):
    stop_list = []
    try:
        with open(stop_file, encoding='utf-8') as stopword_list:
            stop_list = [line.strip() for line in stopword_list]
    except FileNotFoundError:
        print(f"Error: Stop file '{stop_file}' not found.")
    word_list = []

    words = word_tokenize(mytext)

    for word in words:

        word = word.lower()

        exceptions = ['data']
        if word in exceptions:
            word_list.append(word)
            continue

        word = Word(word).singularize()
        if word in stop_list or len(word) < 2:
            continue
        word_list.append(word)

    return " ".join(word_list)





if __name__ == '__main__':
    # 读取文件
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 100)
    pd.set_option('display.width', 1000)

    output_path = 'C:/Users/lujiawen/PycharmProjects/BERTopic/result'
    file_path = 'C:/Users/lujiawen/PycharmProjects/BERTopic/dataset'
    os.chdir(file_path)
    data = pd.read_excel("kfkx_3800.xlsx").astype(str)
    data = data.dropna()
    text = data["Abstract"]
    # 加载csv文件.values.tolist()
    # data=pd.read_csv("dataset.csv").astype(str)#content type
    os.chdir(output_path)
    dic_file = "D:/notebook/LDAsklearn_origin/stop_dic/dict.txt"  # ？
    stop_file = "C:/Users/lujiawen/PycharmProjects/BERTopic/stop_words.txt"
    # print(data[['Abstract', 'tokenized_text']])
    # 预处理
    nltk.download('punkt')
    print("预处理:")
    data['tokenized_text'] = data['Abstract'].apply(english_word_cut)
    print(data[['Abstract', 'tokenized_text']])

    # data["content_cutted"] = english_word_cut(text)
    # data["content_cutted"] = data.Abstract.apply(english_word_cut)
    # print(data["content_cutted"])

    # Step 1 - Embed documents 嵌入文档
    embedding_model = SentenceTransformer('C:/Users/lujiawen/PycharmProjects/BERTopic/models/all-MiniLM-L6-v2')

    # Step 2 - Reduce dimensionality 降维
    # umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine')
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=40)


    # Step 3 - Cluster reduced embeddings 聚类
    hdbscan_model = HDBSCAN(min_cluster_size=28, metric='euclidean', prediction_data=True) # 参数min_samples


    # Step 4 Create topic representation 构建表征主题 C-TF-IDF
    vectorizer_model = CountVectorizer(stop_words="english")
    ctfidf_model = ClassTfidfTransformer()

    # Step 5 - BERTOPIC
    topic_model = BERTopic(
        embedding_model=embedding_model,  # Step 1 - Extract embeddings
        umap_model=umap_model,  # Step 2 - Reduce dimensionality
        hdbscan_model=hdbscan_model,  # Step 3 - Cluster reduced embeddings
        vectorizer_model=vectorizer_model,  # Step 4 - Tokenize topics
        ctfidf_model=ctfidf_model,  # Step 5 - Extract topic words
        # diversity=0.5,  # Step 6 - Diversify topic words
        nr_topics='none',
        # nr_topics=8,
        top_n_words=10
        # min_topic_size
    )

    # Step 6 - 主题概率
    filtered_text = data["tokenized_text"].tolist()
    print("start bertopic")
    # filtered_text = filtered_text.astype(str)
    # filtered_text = data["tokenized_text"].astype(str)
    topics, probabilities = topic_model.fit_transform(filtered_text)


    # 结果
    print("主题文档概率:")
    print(topic_model.get_document_info(filtered_text))
    print("每个主题论文数量:")
    print(topic_model.get_topic_freq())
    print("主题-词概率：")
    print(topic_model.get_topic_info())
    print("第0个主题-词的概率分布：")
    print(topic_model.get_topic(0))

    # 可视化
    topic_model.visualize_barchart(top_n_topics=8).write_html('主题-词概率分布.html')
    topic_model.visualize_hierarchy(color_threshold=2).write_html('聚类分层.html')
    topic_model.visualize_heatmap().write_html('热力图.html')
    topic_model.visualize_topics().write_html('主题分布图.html')
    # topic_model.visualize_documents(titles, reduced_embeddings=reduced_embeddings, custom_1abe1s=True).write_html('主题分布图.html')

    embeddings = embedding_model.encode(filtered_text, show_progress_bar=False)
    # Run the visualization with the original embeddings
    topic_model.visualize_documents(filtered_text, embeddings=embeddings, hide_annotations=True).write_html('文档主题聚类.html')
    #