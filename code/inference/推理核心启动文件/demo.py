#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import jieba
import pickle
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


import gradio as gr

def textcnn(input_x):
    #与textcnn文本预处理和网络构建方法相同
    max_len = 100
    embedding_dim = 200

    model_Word2Vec = Word2Vec.load("word2vec.model")
    loaded_cnn = tf.keras.models.load_model('textcnn.h5')
    
    stopwords = set()
    with open('stopwords.txt', 'r', encoding='utf-8') as f:
        for line in f:
            stopwords.add(line.strip())
            
    cleaned_input_x = re.sub(r'<.*?>', '', input_x)
    pattern = r'[^\u4e00-\u9fa5\s]+'
    cleaned_input_x = re.sub(pattern, '', cleaned_input_x)
    cleaned_input_x = re.sub(r"\s+", "", cleaned_input_x)
    cleaned_input_x_list = list(jieba.cut(cleaned_input_x))
    cleaned_input_x_list = [word for word in cleaned_input_x_list if word not in stopwords]
    
    cleaned_input_x_list1 = []

    if len(cleaned_input_x_list) > max_len:
        cleaned_input_x_list1 = cleaned_input_x_list[:max_len]
    else:
        cleaned_input_x_list1 = cleaned_input_x_list + ["unknown"] * (max_len - len(cleaned_input_x_list))
        
    all_vectors = []
    default_vector = np.zeros(embedding_dim)
    for word in cleaned_input_x_list1:
        try:
            vector = model_Word2Vec.wv[word]
            all_vectors.append(vector)
        except KeyError:
            all_vectors.append(default_vector)
            continue
    input_x = all_vectors
    input_x = np.array(input_x)
    input_x = input_x.reshape(1,max_len,embedding_dim)   
    
    result = loaded_cnn.predict(input_x)
    result = result.argmax(axis = 1)
    result = result + 1
    
    #得出结果后，将label转换为对应的类别
    news_types = {'娱乐':1, '财经':2, '时尚':3, '房产':4,
              '游戏':5, '科技':6, '家居':7, '时政':8, '教育':9, '体育':10}
    
    result = list(news_types.keys())[list(news_types.values()).index(result)]
    
    return result


# In[3]:


#给出两个示例
examples = [
    ["三大因素促房地产复苏 机构认为仍可增持目前主流地产股的动态市盈率已接近或超过了30倍，相当于2007年上半年的水平。机构预计，在供需紧张的情况下，下半年房价仍将报复性上涨。上周，两市地产板块表现异常强势，周涨幅达14%，累计资金净流入约135亿元。尤其是周五，地产板块涨势疯狂，达5%以上。在通胀预期、流动性和经济复苏预期的推动下，房地产板块半年以来累计增长113.27%，超过上证综指涨幅55个点。目前，各家机构对房地产板块一致看好。地产率先复苏引领经济回暖在经济衰退期，为挽救经济，一国政府通过实施宽松货币政策带来市场流动性充裕之后，该国房地产业极有可能率先复苏，并形成对其他产业的带动进而引领经济走出低谷。第一创业举例分析，在1987年签订广场协议之后的日本和2001年新经济泡沫破灭之后的美国。我国从2008年11月份以来，正面临着类似的政策与经济环境，因此他们认为2009年中国的房地产市场步入复苏是大概率事件，上半年的市场表现已经完全印证了他们的看法。安信证券的高善文认为，中国房地产市场可能正在进入泡沫化过程，这有利于短期内的经济恢复，但在中期内对宏观经济管理可能形成巨大挑战。","房产"],
    ["印度批准以23.5亿美元购买俄罗斯航母(图)新华网孟买3月10日电 (记者聂云) 印度国防部10日说，印度政府当天批准以23.5亿美元购买俄罗斯“戈尔什科夫海军上将”号航空母舰。印度海军于2004年向俄罗斯购买“戈尔什科夫海军上将”号航母并由俄进行改装。然而，2007年以来，由于物价上涨等原因，俄方一直要求将航母的购买和改装费用由原来所定的9亿多美元提高到29亿美元。印度有关部门曾多次对俄方报价提出疑问。“戈尔什科夫海军上将”号航母满载排水量约4万吨，目前正在俄罗斯进行现代化改装，预计2012年前后交付印度海军使用。","时政"]
]

#搭建demo
demo = gr.Interface(
    fn = textcnn,
    inputs = [gr.inputs.Textbox(label="新闻文本输入处")],
    outputs = [gr.outputs.Textbox(label="预测结果")],
    title = "基于 TextCNN 的新闻分类器",
    description = "请在下方输入想要进行分类的新闻文本",
    examples=examples
)

demo.launch()

