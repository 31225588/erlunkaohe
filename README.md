# inference文件说明

## 核心推理文件说明

- demo.py   demo展示

## 其他文件说明

- stopwords.txt    停用词表

- textcnn.h5   textcnn网络导入的模型
- word2vec.model   词嵌入导入的模型
- word2vec.model.syn1neg.npy   词嵌入训练中保存的参数
- word2vec.wec.model.wv.vectors.npy   词嵌入输出的词向量





# train文件说明

## 训练与预测文件说明

- lstm.py   lstm网络（包括模型训练和结果预测）
- textcnn.py   textcnn网络（包括模型训练和结果预测）

##其他文件说明

- baidu_stopwords.txt    停用词表1
- cn_stopwords.txt   停用词表2
- hit_stopwords.txt   停用词表3
- scu_stopwords.txt   停用词表4

- news_train.csv   训练集
- news_test_no_answer.csv   测试集
- textcnn_result.csv   textcnn网络的预测结果
- lstm_result.csv   lstm网络的预测结果
- lstm.h5   lstm网络导入的模型以及参数
- textcnn.h5   textcnn网络导入的模型以及参数





# model文件说明

## 模型文件说明

- lstm.h5   lstm网络导入的模型以及参数
- textcnn.h5   textcnn网络导入的模型以及参数

## 自建类文件说明

- my_layer.py   （包括自建全连接层和自建textcnn层）