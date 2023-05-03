# TEXTCNN新闻文本分类

## 导入所需的库

```python
#导入必要的库
import re
import jieba
import pickle
import random
import tensorflow as tf
import gensim
from gensim.models import  Word2Vec
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import  Conv1D, MaxPooling1D, Flatten, Dense
from keras.models import Sequential
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")
```

## 数据预处理

### **读取训练集和测试集**

```python
#读取训练集
train_ = pd.read_csv(r"news_train.csv")
train_x0 = pd.DataFrame(train_["新闻"])
train_x0 = train_x0.values.tolist()
```

```python
#读取测试集
test_ = pd.read_csv(r"news_test_no_answer.csv")
test_x0= pd.DataFrame(test_["新闻"])
test_x0 = test_x0.values.tolist()
```

### **构建停用词表**

  停用词表是baidu_stopwords.txt，cn_stopwords.txt，hit_stopwords.txt，scu_stopwords.txt这四个停用词表的集合。

```python
#构建停词表
stopwords = set()
with open('baidu_stopwords.txt', 'r', encoding='utf-8') as f:
    for line in f:
        stopwords.add(line.strip())
with open('cn_stopwords.txt', 'r', encoding='utf-8') as f:
    for line in f    :
        stopwords.add(line.strip())
with open('hit_stopwords.txt', 'r', encoding='utf-8') as f:
    for line in f:
        stopwords.add(line.strip())  
with open('scu_stopwords.txt', 'r', encoding='utf-8') as f:
    for line in f:
        stopwords.add(line.strip()) 
```

### **文本清洗**

  清洗文本中的超链接，英文字符，数字，空格以及特殊符号和标点符号。

  我认为将这些这些非中文字符删除可以提高预测准确率，因为训练集是中文文本且样本较少，所以对应的英文字符和数字较少，训练出来的词向量不够准确，所以要剔除。

```python
cleaned_train_x_list1 = []
for i in range(len(train_x0)):
    #文本清洗，清除文本中的超链接，英文字符，数字，空格，以及特殊符号和标点符号。
    train_x1 = str((train_x0[i])[0])
    cleaned_train_x = re.sub(r'<.*?>', '', train_x1)
    pattern = r'[^\u4e00-\u9fa5\s]+'
    cleaned_train_x = re.sub(pattern, '', cleaned_train_x)
    cleaned_train_x = re.sub(r"\s+", "", cleaned_train_x)
    #使用jieba分词，对句子进行分词。
    cleaned_train_x_list = list(jieba.cut(cleaned_train_x))
    #提取所有不在停用词表中的词
    cleaned_train_x_list = [word for word in cleaned_train_x_list if word not in stopwords]
    cleaned_train_x_list1.append(cleaned_train_x_list)
```

```python
#对测试集做同样的数据预处理
cleaned_test_x_list1 = []
for i in range(len(test_x0)):
    test_x1 = str((test_x0[i])[0])
    cleaned_test_x = re.sub(r'<.*?>', '', test_x1)
    pattern = r'[^\u4e00-\u9fa5\s]+'
    cleaned_test_x = re.sub(pattern, '', cleaned_test_x)
    cleaned_test_x = re.sub(r"\s+", "", cleaned_test_x)
    cleaned_test_x_list = list(jieba.cut(cleaned_test_x))
    cleaned_test_x_list = [word for word in cleaned_test_x_list if word not in stopwords]
    cleaned_test_x_list1.append(cleaned_test_x_list)
```

```python
#合并测试集与训练集中的词
clean_x_list = cleaned_train_x_list1 + cleaned_test_x_list1
```

### **jieba分词**

  jieba是一款基于Python的中文分词工具，它可以将中文文本快速、准确地切分成一系列单独的词语.

### 词嵌入

  词嵌入（Word Embedding）是NLP领域中的一种技术，用于将文本数据中的单词映射到低维的实数空间中。通过对文本中每个单词进行嵌入，可以将其转换为实数向量，从而更好地挖掘和利用文本中的语义信息。

  基于词嵌入的方法有很多，比如word2vec、GloVe等。

  本文选取的词嵌入方法是word2vec。

#### **word2vec构建词向量**

  CBOW和Skip-gram是Word2Vec模型中的两种经典算法。

  CBOW（Continuous Bag-of-Words）算法是通过给定上下文词来预测目标词汇的概率。它将文本序列中的每个词的上下文作为输入，并尝试预测当前词汇。CBOW在训练时，通过将各个上下文词汇的词向量进行加和，得到了一个平均向量，再将该平均向量通过线性变换映射成目标词汇的概率输出。CBOW相对于Skip-gram模型，更加适用于训练数据集较大、高频词多的情况下。

  Skip-gram（跳字模型）算法则是基于目标词汇预测其上下文单词的概率。即从中央单词中预测窗口内的周围单词。Skip-gram模型训练时，通过将中央词汇的词向量输入神经网络，得到与之关联的上下文词汇的概率分布。Skip-gram相对于CBOW模型，更加适用于训练数据集规模较小、低频词多的情况下。

  Negative Sampling和Hierarchical Softmax是Word2Vec模型中的两种优化方法。

  Negative Sampling算法通过一个二分类模型学习到每个单词的嵌入向量。在训练时，它只针对当前上下文中出现的负样本进行更新，以此优化效率。该优化方法一般适用于大规模数据集并且高频词汇较多的情况下。

  Hierarchical Softmax是一种哈夫曼树结构，被应用到Word2Vec中以实现更加高效的计算。在训练时，通过对目标单词在哈夫曼树中进行分布式表示，并采用分层的方式计算其概率。和Negative Sampling相比，Hierarchical Softmax适用于低频词汇较多、数据集较小的情况下。

  本文使用的模型是Skip-gram模型和Negative Sampling优化方法。

```python
#词表参数以及word2vec参数
embedding_dim = 200
window = 5
min_count = 5
negative = 10
epochs = 30
max_len = 100
len_train_x = 10000
len_train_y = 10000
len_test_x = 1999
num_classes = 10

#使用word2vec来构建词向量
model = Word2Vec(clean_x_list, 
                 vector_size = embedding_dim, 
                 window = window, 
                 min_count = min_count,
                 epochs = epochs , 
                 negative = negative)

#保存word2vec词向量模型
model.save("word2vec.model")
```

#### 转换张量

##### **训练集的填充和截断**

```python
#对分词后的句子做填充和截断，使句子大小都为100个词。
#句子长度不足100的，用unknown填充。
unknown = ['unknown']
cleaned_train_x_list2 = []
for sentence in cleaned_train_x_list1:
    if len(sentence) > max_len:
        sentence = sentence[:max_len]
        cleaned_train_x_list2.append(sentence)
    else:
        for i in range((max_len-len(sentence))):
            sentence.append(unknown)
        cleaned_train_x_list2.append(sentence)   
```

##### **训练集转化词向量**

```python
#将训练集文本中的词转化为对应词向量
train_x = []
all_vectors = []
default_vector = np.zeros(embedding_dim)
for sentence in cleaned_train_x_list2:
    for word in sentence:
        try:
            vector = model.wv[word]
            all_vectors.append(vector)
        #遇到不在词表中的词，或者unknown的时候，用全0数组填充。    
        except KeyError:
            all_vectors.append(default_vector)
            continue
train_x.append(all_vectors) 
```

##### **测试集转换(10000,100,200)张量**

对应后文cnn.fit的形状（num_sample,max_len,embedding_dim）

```python
#将训练集转换成(10000,100,200)的张量
train_x = [lst for lst in train_x]
train_x = train_x = np.array(train_x)
train_x = train_x.reshape((len_train_x,max_len,embedding_dim))
```

##### **测试集的填充和截断**

```python
#同样步骤处理测试集文本。
cleaned_test_x_list2 = []
for sentence in cleaned_test_x_list1:
    if len(sentence) > max_len:
        sentence = sentence[:max_len]
        cleaned_test_x_list2.append(sentence)
    else:
        for i in range((max_len-len(sentence))):
            sentence.append(unknown)
        cleaned_test_x_list2.append(sentence) 
```

##### **训练集转化词向量**

```python
test_x = []
all_vectors = []
default_vector = np.zeros(embedding_dim)
for sentence in cleaned_test_x_list2:
    for word in sentence:
        try:
            vector = model.wv[word]
            all_vectors.append(vector)
        except KeyError:
            all_vectors.append(default_vector)
            continue
test_x.append(all_vectors) 
```

##### **训练集转换(1999,100,200)张量**

```python
#将测试集转换成(1999,100,200)的张量
test_x = [elem for lst in test_x for elem in lst]
test_x = np.array(test_x)
test_x = test_x.reshape((len_test_x,max_len,embedding_dim))
```

##### **训练集label独热编码**

对应后文的cnn.fit形状(num_sample,num_classes)

```python
#训练集label的独热编码
num_classes = 10
train_y = pd.DataFrame(train_["标签"])
trian_y = np.array(train_y)
train_y = train_y.T - 1
y = np.zeros((len_train_x,num_classes))
m = np.arange(0,len_train_x).reshape(1,len_train_x)
y[m,train_y] += 1
train_y = y
```

#### 词嵌入遇到的问题

  问题：最终结果打榜分数一直在零点一几徘徊。

  问题分析：开始以为是predict()函数打乱了测试集的顺序，导致预测出来的类别对不上真实类别。但是经过打印确认发现没有问题。

  然后认为是过拟合导致训练集测试结果和预测结果相差过大，在添加验证集过后，观察验证集的val_accuracy，发现虽然图线拟合效果不是非常好，但是val_accuracy也超过0.9，那就证明不是过拟合导致验证集预测错误。

  最后发现是在构建词表的时候，我将训练集和测试集的词表分别构建，导致测试集词向量嵌入错误，在合并训练集和测试集词表后，问题解决。

### 验证集的分割

```python
#分割训练集和验证集
seed = 123
test_size = 0.2
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size = test_size, shuffle=False, random_state=seed)
```

### 构建train_generator

```python
#一批一批生成数据
def data_generator(train_x, train_y, batch_size):
    while True:
        # 随机抽样 batch_size 个数据
        indices = random.sample(range(len(train_x)), batch_size)
        batch_x = train_x[indices]
        batch_y = train_y[indices]

        # 将当前批次的数据打包成 numpy 数组，并返回
        yield (np.array(batch_x), np.array(batch_y))
```

```python
batch_size = 256
train_generator = data_generator(train_x, train_y, batch_size)
```

## TEXTCNN网络搭建

### TEXTCNN网络

  TextCNN是一种用于文本分类的卷积神经网络模型。它采用卷积神经网络的架构，在文本领域中通过卷积操作实现特征提取，并使用池化和全连接层完成分类任务。与其它传统的文本分类模型相比，TextCNN可以快速有效地学习到文本中的关键信息，同时也有很好的泛化能力。

  TextCNN模型的输入是一个序列，每个元素代表一个词，也就是一条文本记录。在模型中，首先通过将每个单词转为词向量进行表示。然后，将整个文本数据看作一个二维矩阵，并用一个卷积操作对其进行特征提取。接下来使用池化操作对提取的特征信息进行降维处理，再通过全连接层进行分类。

  TextCNN的结构优化主要有两个方向：一是词向量的构造，二是网络参数和超参数调优。其中，词向量的构造可以采用word2vec、GloVe等技术生成。而在网络参数方面，主要是对卷积核大小和数量、池化方式、全连接层和Dropout等进行调优。

  TextCNN通过卷积和池化操作对文本数据进行特征提取，使用全连接层进行分类。

  总的来说，TextCNN是一种简单而有效的文本分类模型，它可以通过卷积操作和池化操作较好地捕捉文本中的关键特征。它在许多文本分类任务上都取得了很好的效果，比如情感分析、垃圾邮件识别等。

#### 输入层

  输入层接受一段文本数据，每个单词被转换成了一个固定长度的向量表示，可采用预训练的词向量表示方法，如word2vec、GloVe等。

#### 卷积层

  TextCNN使用一维卷积层对文本数据进行特征提取，卷积核大小可根据具体的任务进行设置。卷积操作可以捕捉文本中的局部特征，类似于n-gram模型。在卷积操作中，卷积核对输入数据进行滑动窗口扫描，提取出每个窗口的特征并加以处理。

#### 池化层

  池化层即对特征图进行降维处理，将每个特征图中的最大值提取出来组成新的向量。这种方式可以有效地减少参数量和计算量，同时提高模型的泛化能力。

#### 全连接层

  经过卷积和池化操作后，文本数据被转换为一个固定长度的向量形式。该向量被送入全连接层进行分类，最后输出预测结果。

### sequential的构建

```python
#textcnn的sequential的参数

#卷积层参数
input_shape = (max_len,embedding_dim)
kernel_regularizer = regularizers.l2(l2 = 1e-3)
filters_1 = 32 
filters_2 = 64
filters_3 = 128
kernel_size1 = 5
kernel_size2 = 4
kernel_size3 = 3
strides_1 = 1

#池化层参数
strides_2 = 2
pool_size = 5

#全连接层参数
dense_size_1 = 128
dense_size_2 = 64
dense_size_3 = 32
num_classes = 10

#dropout层参数
dropout_1 =  0.5
dropout_2 =  0.3
dropout_3 = 0.2


#sequential的构建
cnn = tf.keras.Sequential(
    #卷积层1+池化层1的构建
    [tf.keras.layers.Conv1D(input_shape = input_shape,
        filters = filters_1,
        kernel_size = kernel_size1,
        strides = strides_1,
        padding='same',
        activation='relu',
        name='conv1',
        kernel_regularizer = kernel_regularizer),
     
    tf.keras.layers.MaxPool1D(pool_size = pool_size,
        strides = strides_2,
        name = 'pool1'),
     
     #卷积层2+池化层2的构建
    tf.keras.layers.Conv1D(filters = filters_2,
        kernel_size = kernel_size2,
        strides = strides_1,
        padding='same',
        activation='relu',
        name='conv2',
        kernel_regularizer = kernel_regularizer,),
     
    tf.keras.layers.MaxPool1D(pool_size = pool_size,
        strides = strides_2,
        name='pool2'),
     
     #卷积层3+池化层3的构建
    tf.keras.layers.Conv1D(filters = filters_3,
        kernel_size = kernel_size3 ,
        strides = strides_1,
        padding='same',
        activation='relu',
        name='conv3',
        kernel_regularizer = kernel_regularizer),
    tf.keras.layers.MaxPool1D(pool_size = pool_size,
        strides = strides_2,
        name = 'pool3'),
     
     #将三维输出拉平
    tf.keras.layers.Flatten(),
     
     #全连接层1+dropout层1的构建
    tf.keras.layers.Dense(dense_size_1,
        activation='relu',
        name='fc1'),
    tf.keras.layers.Dropout(dropout_1),
     
     #全连接层2+dropout层2的构建
    tf.keras.layers.Dense(dense_size_2,
        activation='relu',
        name='fc2'),
    tf.keras.layers.Dropout(dropout_2),
     
     #全连接层3+dropout层3的构建
    tf.keras.layers.Dense(dense_size_3,
        activation='relu',
        name='fc3'),
    tf.keras.layers.Dropout(dropout_3),
     
     #输出层的构建
    tf.keras.layers.Dense(num_classes,
        activation='softmax',
        name='output')]
                       )    
```

### compile的构建

```python
#配置优化器compile
nadam = tf.keras.optimizers.Nadam(lr=1e-4)
cnn.compile(optimizer=nadam, 
            loss='categorical_crossentropy',
            metrics=[tf.keras.metrics.CategoricalAccuracy(),tf.keras.metrics.Recall()]
            )
```

### 网络训练

```python
#放入训练测试数据，开始迭代

#动态学习率参数
factor = 0.7
patience = 5
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor = factor,                                                     patience = patience)

#训练参数
epochs = 100
batch_size = 256
history=cnn.fit(train_generator,
                steps_per_epoch=len(train_x) // batch_size,
                epochs = epochs ,
                validation_data=(val_x, val_y), 
                batch_size = batch_size,
                verbose = 1,
                callbacks=[reduce_lr],
                shuffle=False
                )
```

## 模型评估以及可视化

### loss/val_loss

```python
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['loss', 'val loss'], loc='upper left')
plt.show()
```

### categorical_accuracy/val_categorical_accuracy

```python
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.legend(['categorical accuracy','val categorical accuracy'], loc='upper left')
plt.show()
```

### recall/val_recall

```python
plt.plot(history.history['recall'])
plt.plot(history.history['val_recall'])
plt.legend(['recall','val recall'], loc='upper left')
plt.show()
```

### test_accuracy/test_f1-score

```python
answer = pd.read_csv(r"sample_submission.csv")
answer = pd.DataFrame(answer["label"])
answer = np.array(answer)
answer = answer.reshape((len_test_x,1))

#计算测试集accuracy
a = result - answer
accuracy = np.sum(a == np.zeros((len_test_x,1))) / len(result)
print(accuracy)

#计算打榜分数
from sklearn.metrics import f1_score
f1_score(answer,result,average='macro')
```

### 模型评估遇到的问题

问题：验证集的收敛较慢，而且震荡幅度较大。

问题解决：收敛较慢，更换优化器，放弃SGD，改用Nadam，收敛速度明显加快。

震荡幅度较大：更换优化器，增加dropout层，改用动态学习率，加入正则化，震荡幅度减小，loss值减小。

## 结果的预测以及保存

```python
#测试集的预测和结果的保存
result = cnn.predict(test_x)
result = result.argmax(axis = 1) + 1
result = result.reshape((len_test_x,1))
result_1 = result.reshape(len_test_x,1)
result_1 = np.hstack([np.arange(0,len_test_x).reshape(len_test_x,1),result_1])
result_1 = pd.DataFrame(result_1, columns=['id', 'label'])
result_1.to_csv('result.csv')
```

## 网络模型以及可视化参数的保存

```python
#保存网络模型
tf.keras.models.save_model(cnn, 'textcnn.h5')
```

```python
#保存可视化参数
with open('cnn.history.pkl', 'wb') as f:
    pickle.dump(history.history, f) 
```

