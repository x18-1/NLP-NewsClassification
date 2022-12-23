# 新闻分类

使用selenium进行数据爬取，使用字典树，jieba，hanlp，PCA进行数据处理，使用朴素贝叶斯，SVM，LightGBM，BiLSTM，TextCNN，BERT等模型进行建模，使用折交叉验证，warm-up 等训练技巧，使用word2vec，Tfidf	，词袋，Bert来表示词向量。


模型效果：

|               模型               | 测试集最好分数 |
| :------------------------------: | :------------: |
| 朴素贝叶斯（交叉验证的词袋表示） |     0.861      |
|     支持向量机（折交叉验证）     |     0.875      |
|   BiLSTM（预训练模型+warm-up）   |     0.885      |
|      TextCNN（预训练模型）       |     0.903      |
|         BERT（warm-up）          |      0.88      |




## DeepLearning

深度学习模型来进行新闻分类，使用的模型有Bert，TextCNN，BiLSTM。词向量的表示使用预训练模型Word2vec

## MachineLearning

使用机器学习的方法进行新闻分类，使用的模型有朴素贝叶斯、SVM、LightGBM。词向量的表示词袋、Tfidf、Word2vec（utils中的Text2vec方法）

## Get_data

使用selenium的爬虫，爬取的澎湃新新闻的数据

## Utils

一些封装的函数。vocab是词典。Trie是字典树。utils里面是函数的封装：分词函数、基于字典树的停用词过滤、Tetx2vec、TokenEmbedding、网络训练、数据加载函数、数据处理函数



