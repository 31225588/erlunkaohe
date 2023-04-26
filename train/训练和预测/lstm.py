#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")


# In[2]:


#读取训练集
train_ = pd.read_csv(r"news_train.csv")
train_x0 = pd.DataFrame(train_["新闻"])
train_x0 = train_x0.values.tolist()


# In[3]:


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


# In[4]:


#保存停用词表
with open('stopwords.txt', 'w', encoding='utf-8') as f:
    for word in stopwords:
        f.write(word + '\n')


# In[5]:


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


# In[6]:


#读取测试集
test_ = pd.read_csv(r"news_test_no_answer.csv")
test_x0= pd.DataFrame(test_["新闻"])
test_x0 = test_x0.values.tolist()


# In[7]:


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


# In[8]:


#合并测试集与训练集中的词
clean_x_list = cleaned_train_x_list1 + cleaned_test_x_list1


# In[9]:


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


# In[10]:


#对分词后的句子做填充和截断，使句子大小都为350个词。
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


# In[11]:


#将测试集文本中的词转化为对应词向量
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


# In[1]:


#将训练集转换成(10000,100,200)的张量
train_x = [lst for lst in train_x]
train_x = train_x = np.array(train_x)
train_x = train_x.reshape((len_train_x,max_len,embedding_dim))


# In[13]:


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


# In[14]:


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


# In[15]:


#将训练集转换成(1999,100,200)的张量
test_x = [elem for lst in test_x for elem in lst]
test_x = np.array(test_x)
test_x = test_x.reshape((len_test_x,max_len,embedding_dim))


# In[16]:


#训练集label的独热编码
num_classes = 10
train_y = pd.DataFrame(train_["标签"])
trian_y = np.array(train_y)
train_y = train_y.T - 1
y = np.zeros((len_train_x,num_classes))
m = np.arange(0,len_train_x).reshape(1,len_train_x)
y[m,train_y] += 1
train_y = y


# In[17]:


#分割训练集和验证集
seed = 123
test_size = 0.2
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size = test_size, shuffle=False, random_state=seed)


# In[18]:


def data_generator(train_x, train_y, batch_size):
    while True:
        # 随机抽样 batch_size 个数据
        indices = random.sample(range(len(train_x)), batch_size)
        batch_x = train_x[indices]
        batch_y = train_y[indices]

        # 将当前批次的数据打包成 numpy 数组，并返回
        yield (np.array(batch_x), np.array(batch_y))


# In[19]:


#一批一批生成数据
batch_size = 256
train_generator = data_generator(train_x, train_y, batch_size)


# In[21]:


#构建textcnn网络

#lstm的sequential的参数
num_classes = 10
dropout_1 =  0.5
dropout_2 =  0.3
recurrent_dropout_1 = 0.2
input_shape = (max_len,embedding_dim)
kernel_regularizer = regularizers.l2(l2 = 1e-3)

lstm = tf.keras.Sequential(
    #lstm层1的构建
    [tf.keras.layers.LSTM(64, input_shape = input_shape, return_sequences = True,kernel_regularizer = kernel_regularizer,
                          recurrent_dropout = recurrent_dropout_1),
     tf.keras.layers.Dropout(dropout_1),
     
     #lstm层2的构建
     tf.keras.layers.LSTM(128,kernel_regularizer = kernel_regularizer,recurrent_dropout = recurrent_dropout_1),
     tf.keras.layers.Dropout(dropout_2),
     
     #输出层的构建
     tf.keras.layers.Dense(num_classes, activation='softmax', name='output')])

#配置优化器compile
nadam = tf.keras.optimizers.Nadam(lr=1e-3)
lstm.compile(optimizer=nadam, loss='categorical_crossentropy',
             metrics=[tf.keras.metrics.CategoricalAccuracy(),
                      tf.keras.metrics.Recall()])

#放入训练测试数据，开始迭代

#动态学习率参数
factor = 0.7
patience = 5
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor = factor, patience = patience)

#训练参数
epochs = 100
batch_size = 256
history=lstm.fit(train_generator,
                steps_per_epoch=len(train_x) // batch_size,
                epochs = epochs ,
                validation_data=(val_x, val_y), 
                batch_size = batch_size,
                verbose = 1,
                callbacks=[reduce_lr],
                shuffle=False
                )


# In[29]:


#loss，categorical_accuracy，recall在训练集和验证集上结果可视化。
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['loss', 'val loss'], loc='upper left')
plt.show()

plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.legend(['categorical accuracy','val categorical accuracy'], loc='upper left')
plt.show()

plt.plot(history.history['recall_1'])
plt.plot(history.history['val_recall_1'])
plt.legend(['recall','val recall'], loc='upper left')
plt.show()


# In[30]:


#测试集的预测和结果的保存
result = lstm.predict(test_x)
result = result.argmax(axis = 1)
result = result + 1
result = result.reshape((len_test_x,1))
result_1 = result.reshape(len_test_x,1)
result_1 = np.hstack([np.arange(0,len_test_x).reshape(len_test_x,1),result_1])
result_1 = pd.DataFrame(result_1, columns=['id', 'label'])
result_1.to_csv('result.csv')


# In[31]:


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


# In[35]:


#保存网络模型
tf.keras.models.save_model(lstm, 'lstm.h5')


# In[36]:


#保存可视化参数
with open('lstm.history.pkl', 'wb') as f:
    pickle.dump(history.history, f) 

