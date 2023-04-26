#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf

class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self, units=32, activation=tf.nn.relu):
        super(MyDenseLayer, self).__init__()
        self.units = units
        self.activation = activation

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='random_normal',
                                 trainable=True)

    def call(self, inputs):
        linear_output = tf.matmul(inputs, self.w) + self.b
        if self.activation is not None:
            return self.activation(linear_output)
        return linear_output


# In[2]:


class TextCNN(tf.keras.layers.Layer):
    
    def __init__(self, input_shape,
                 filters_1, filters_2, filters_3, 
                 kernel_size1, kernel_size2, kernel_size3, 
                 strides_1, strides_2, pool_size, 
                 dense_size_1, dense_size_2, dense_size_3, num_classes, 
                 dropout_1, dropout_2, dropout_3, 
                 kernel_regularizer, **kwargs):
        
        super(TextCNN, self).__init__(**kwargs)
        
        self.conv1 = tf.keras.layers.Conv1D(input_shape=input_shape, 
                                            filters=filters_1, 
                                            kernel_size=kernel_size1, 
                                            strides=strides_1, 
                                            padding='same', 
                                            activation='relu', 
                                            name='conv1', 
                                            kernel_regularizer=kernel_regularizer)
        
        self.pool1 = tf.keras.layers.MaxPool1D(pool_size=pool_size, 
                                                strides=strides_2, 
                                                name='pool1')
        
        self.conv2 = tf.keras.layers.Conv1D(filters=filters_2, 
                                            kernel_size=kernel_size2, 
                                            strides=strides_1, 
                                            padding='same', 
                                            activation='relu', 
                                            name='conv2', 
                                            kernel_regularizer=kernel_regularizer)
        
        self.pool2 = tf.keras.layers.MaxPool1D(pool_size=pool_size, 
                                                strides=strides_2, 
                                                name='pool2')
        
        self.conv3 = tf.keras.layers.Conv1D(filters=filters_3, 
                                            kernel_size=kernel_size3, 
                                            strides=strides_1, 
                                            padding='same', 
                                            activation='relu', 
                                            name='conv3', 
                                            kernel_regularizer=kernel_regularizer)
        
        self.pool3 = tf.keras.layers.MaxPool1D(pool_size=pool_size, 
                                                strides=strides_2, 
                                                name='pool3')
        
        self.flatten = tf.keras.layers.Flatten()
        
        self.fc1 = tf.keras.layers.Dense(dense_size_1, 
                                          activation='relu', 
                                          name='fc1')
        self.dropout1 = tf.keras.layers.Dropout(dropout_1)
        
        self.fc2 = tf.keras.layers.Dense(dense_size_2, 
                                          activation='relu', 
                                          name='fc2')
        self.dropout2 = tf.keras.layers.Dropout(dropout_2)
        
        self.fc3 = tf.keras.layers.Dense(dense_size_3, 
                                          activation='relu', 
                                          name='fc3')
        self.dropout3 = tf.keras.layers.Dropout(dropout_3)
        
        self.output_layer = tf.keras.layers.Dense(num_classes, 
                                                  activation='softmax', 
                                                  name='output')
    
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.dropout3(x)
        output = self.output_layer(x)
        return output


# In[ ]:




