调参：
umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine')


hdbscan_model = HDBSCAN(min_cluster_size=30, metric='euclidean', prediction_data=True,min_samples=50)
①min_samples指定了在DBSCAN中被视为核心点的最小样本数。具体来说，一个点只有在以min_samples为半径的范围内至少包含min_samples个点时，才被认为是核心点。这个参数影响了DBSCAN算法对于密度高低的定义，从而影响了聚类的结果。
②min_cluster_size参数是HDBSCAN算法中的一个参数，它指定了一个聚类簇的最小样本数。min_cluster_size参数用于确定形成的簇至少包含多少个样本，小于这个数量的簇将被视为噪声。

①②③④⑤
主题数较少：
①参数设置，n_neighbors
②语料库
③文本预处理、停用词

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        nr_topics='none',
        top_n_words=10
    )