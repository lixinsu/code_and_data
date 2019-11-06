#!/usr/bin/env python
# coding: utf-8

# ## 图像识别任务 -- MNIST手写数字识别
# ---
# 
# ### 第0步：导入运行所需要的Python包

# In[1]:


get_ipython().system('which python')


# In[2]:


import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tqdm
get_ipython().run_line_magic('matplotlib', 'inline')


# ### 第1步：加载数据 可视化数据

# In[3]:


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./data/', one_hot=True)


# In[4]:


plt.figure()
for i in range(18):
    plt.subplot(3,6,i+1)
    plt.imshow(np.reshape(mnist.train.images[i], [28, 28]), cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('%s' % np.argwhere(mnist.train.labels[i]==1)[0][0], fontsize=15)
plt.show()


# ### 第2步：构建模型（Softmax回归模型，卷积神经网络模型）

# In[5]:


class SoftmaxModel:
    def __init__(self):
        # Create the softmax regression model
        self.x = tf.placeholder(tf.float32, [None, 784])
        self.is_train = tf.placeholder(tf.bool, [])
        
        W = self.weight_variable([784, 10])
        b = self.bias_variable([10])
        self.y = tf.matmul(self.x, W) + b
        
        # Define loss 
        self.y_ = tf.placeholder(tf.float32, [None, 10])
        self.cross_entropy = tf.reduce_mean(
          tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y))
        
        # Define optimizer
        self.train_step = tf.train.GradientDescentOptimizer(0.5).minimize(self.cross_entropy)
        
        correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        tf.summary.scalar('Loss', self.cross_entropy)
        tf.summary.scalar('Accuracy', self.accuracy)
        
        self.merged = tf.summary.merge_all()
        
    def weight_variable(self, shape):
        """Create a weight variable with appropriate initialization."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        """Create a bias variable with appropriate initialization."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)


# In[6]:


class CNNModel:
    def __init__(self):
        # Create the softmax regression model
        self.x = tf.placeholder(tf.float32, [None, 784])
        self.is_train = tf.placeholder(tf.bool, [])
        
        input_layer = tf.reshape(self.x, [-1, 28, 28, 1])
        
        # Convolutional Layer #1
        conv1 = tf.layers.conv2d(
          inputs=input_layer,
          filters=16,
          kernel_size=[3, 3],
          padding="same",
          activation=tf.nn.relu)

        # Pooling Layer #1
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

        # Convolutional Layer #2 and Pooling Layer #2
        conv2 = tf.layers.conv2d(
          inputs=pool1,
          filters=16,
          kernel_size=[3, 3],
          padding="same",
          activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

        # Dense Layer
        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 16])
        dense = tf.layers.dense(inputs=pool2_flat, units=128, activation=tf.nn.relu)
        dropout = tf.layers.dropout(
          inputs=dense, rate=0.2, training=self.is_train)

        # Logits Layer
        self.y = tf.layers.dense(inputs=dropout, units=10)
        
        # Define loss 
        self.y_ = tf.placeholder(tf.float32, [None, 10])
        self.cross_entropy = tf.reduce_mean(
          tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y))
        
        # Define optimizer
        self.train_step = tf.train.AdamOptimizer(0.001).minimize(self.cross_entropy)
        
        correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        tf.summary.scalar('Loss', self.cross_entropy)
        tf.summary.scalar('Accuracy', self.accuracy)
        
        self.merged = tf.summary.merge_all() 


# ### 第3步：训练与测试模型

# In[7]:


get_ipython().system(' rm -r ./result/softmax/*')


# In[8]:


tf.reset_default_graph()

print('开始训练Softmax回归模型...')

sess = tf.InteractiveSession()

train_writer = tf.summary.FileWriter('./result/softmax/train', flush_secs=1)
test_writer = tf.summary.FileWriter('./result/softmax/test', flush_secs=1)

model = SoftmaxModel()

graph_writer = tf.summary.FileWriter('./result/softmax/graph', sess.graph)
graph_writer.close()

tf.global_variables_initializer().run()

# Train
for i in tqdm.tqdm(range(200)):
    batch_xs, batch_ys = mnist.train.next_batch(32)

    summary_train, _ = sess.run([model.merged, model.train_step], 
                                feed_dict={model.x: batch_xs, model.y_: batch_ys, 
                                           model.is_train:True})
    
    # Test trained model
    summary_test = sess.run(model.merged, 
                            feed_dict={model.x: mnist.test.images, model.y_: mnist.test.labels,
                                      model.is_train:False})
    
    train_writer.add_summary(summary_train, i)
    test_writer.add_summary(summary_test, i)

train_writer.close()
test_writer.close()

print('训练完毕.')


# In[9]:


get_ipython().system(' rm -r ./result/cnn/*')


# In[ ]:


# CNN very slow, try it later
tf.reset_default_graph()

print('开始训练卷积神经网络模型...')

sess = tf.InteractiveSession()

train_writer = tf.summary.FileWriter('./result/cnn/train', flush_secs=1)
test_writer = tf.summary.FileWriter('./result/cnn/test', flush_secs=1)

model = CNNModel()

graph_writer = tf.summary.FileWriter('./result/cnn/graph', sess.graph)
graph_writer.close()

tf.global_variables_initializer().run()

# Train
for i in tqdm.tqdm(range(200)):
    batch_xs, batch_ys = mnist.train.next_batch(32)

    summary_train, _ = sess.run([model.merged, model.train_step], 
                                feed_dict={model.x: batch_xs, model.y_: batch_ys, 
                                           model.is_train: True})
    
    # Test trained model
    summary_test = sess.run(model.merged, 
                            feed_dict={model.x: mnist.test.images, model.y_: mnist.test.labels, 
                                       model.is_train: False})
    
    train_writer.add_summary(summary_train, i)
    test_writer.add_summary(summary_test, i)

train_writer.close()
test_writer.close()

print('训练完毕.')


# ### 最后一步：TensorBoard可视化
# 
# 模型图：
#     ![b](./data/model.png)
# 
# 模型训练过程：
#     ![a](./data/example.png)
# 
# 

# ### 大功告成！最后看看我们的模型预测对了多少？

# In[11]:


pred_labels = sess.run(model.y, feed_dict={model.x: mnist.test.images[:18], model.is_train: False})

plt.figure()
for i in range(18):
    plt.subplot(3,6,i+1)
    plt.imshow(np.reshape(mnist.test.images[i], [28, 28]), cmap='gray')
    plt.xticks([])
    plt.yticks([])
    pred_num = np.argmax(pred_labels[i])
    real_num = np.argwhere(mnist.test.labels[i]==1)[0][0]
    if pred_num == real_num:
        plt.xlabel('%s %s' % (pred_num, real_num), fontsize=15, color='green')
    else:
        plt.xlabel('%s %s' % (pred_num, real_num), fontsize=15, color='red')
plt.show()


# In[ ]:




