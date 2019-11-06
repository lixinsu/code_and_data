#!/usr/bin/env python
# coding: utf-8

# # Policy Gradient 解决 Cart Pole V0

# <h3>Car pole游戏目标</h3>
# 在car（小车）上立pole（柱子）
# 

# <h3> Cart Pole V0 </h3>
# 
# <img src="https://cdn-images-1.medium.com/max/1200/1*G_whtIrY9fGlw3It6HFfhA.gif" alt="Cart Pole game" />
# 
# 4 种状态信息:
# <ul>
#     <li> 车的位置</li>
#     <li> 车的速度 </li>
#     <li> 柱子的位置 </li>
#     <li> 柱子的速度 </li>
# </ul>
# <br>
# agent的动作:
# <ul>
#     <li> 0: 向左 </li>
#     <li> 1: 向右 </ul>
# 
# 

# 参考资源
# <ul>
#     <li> <a href="https://medium.com/@awjuliani/super-simple-reinforcement-learning-tutorial-part-2-ded33892c724">Simple Reinforcement Learning with Tensorflow: Part 2 - Policy-based Agents </a> </li>
#     
#    
#   <li> <a href="https://gist.github.com/shanest/535acf4c62ee2a71da498281c2dfc4f4" >Policy gradients for reinforcement learning in TensorFlow</a></li>
#   </ul>

# ### 导入依赖的包

# In[1]:


import gym
import numpy as np
import tensorflow as tf


# ### Our game environment

# In[2]:


env = gym.make("CartPole-v0")

# 先看下agent随机策略的效果
env.reset()
rewards = []

for _ in range(100):
    env.render()
    
    # 采用随机动作，左右采样一个动作
    state, reward, done, info = env.step(env.action_space.sample())
env.close()


# ### 指定超参数

# In[3]:


input_size = 4 # state的种类
action_size = 2 # 2 actions 种类
hidden_size = 64 # 隐层单元数

learning_rate = 0.001 
gamma = 0.99 # 打折比例

train_episodes = 5000 # 游戏轮数
max_steps = 900 # 最大步数
batch_size = 5


# ### 构建网络

# In[4]:


class PGAgent():
    def __init__(self, input_size, action_size, hidden_size, learning_rate, gamma):
        
        self.input_size = input_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        
        # 构建网络
        self.inputs = tf.placeholder(tf.float32, 
                      shape = [None, input_size])
                              
        self.hidden_layer_1 = tf.contrib.layers.fully_connected(inputs = self.inputs,
                                                  num_outputs = hidden_size,
                                                  activation_fn = tf.nn.elu,
                                                  weights_initializer = tf.random_normal_initializer())

        self.output_layer = tf.contrib.layers.fully_connected(inputs = self.hidden_layer_1,
                                                         num_outputs = action_size,
                                                 activation_fn = tf.nn.softmax)
        
        # Log prob output
        self.output_log_prob = tf.log(self.output_layer)
        
        
        ### 损失函数 : 把reward 和 chosen action 输入 DNN
        # 参考实现 https://gist.github.com/shanest/535acf4c62ee2a71da498281c2dfc4f4
        
        self.actions = tf.placeholder(tf.int32, shape = [None])
        self.rewards = tf.placeholder(tf.float32, shape = [None])
        
        # 获得 log probability of actions from episode : 
        self.indices = tf.range(0, tf.shape(self.output_log_prob)[0]) * tf.shape(self.output_log_prob)[1] + self.actions
        
        self.actions_probability = tf.gather(tf.reshape(self.output_layer, [-1]), self.indices)
        
        self.loss = -tf.reduce_mean(tf.log(self.actions_probability) * self.rewards)
        
  

        # 收集梯度 after some training episodes outside the graph and then apply them.
        # 参考 https://medium.com/@awjuliani/super-simple-reinforcement-learning-tutorial-part-2-ded33892c724#.mtwpvfi8b
        tvars = tf.trainable_variables()
        self.gradient_holders = []
        for idx,var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32, name=str(idx)+ '_holder')
            self.gradient_holders.append(placeholder)
        
        self.gradients = tf.gradients(self.loss,tvars)
        
        
        ### 优化器
        
        #  RMSProp效果较好
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders,tvars))
        


# ### 定义优势函数   
# 通过该优势函数告诉代理怎么做正确的

# <p>早期的reward优于长期reward，对长期reward进行打折.</b>
# </p>
# <img src="assets/discountreward.png" alt="Discount reward"/>

# 延迟的reward影响更小：举个例子5步之后柱子已经非常倾斜，那之后的reward无意义，因为柱子已经无法修正

# <img src="assets/d1.png"/>

# <img src="assets/d2.png"/>

# In[5]:


# 对即时reward的和延迟reward应用不同的权重

def discount_rewards(r):
    # 初始化reward打折矩阵
    discounted_reward= np.zeros_like(r) 
    
    # 存储reward 的和
    running_add = 0
    
    # 遍历rewards
    for t in reversed(range(0, r.size)):
        
        running_add = running_add * gamma + r[t] # sum * y (gamma) + reward
        discounted_reward[t] = running_add
    return discounted_reward


# ### 训练agent

# In[6]:


# 清除图

tf.reset_default_graph()

agent = PGAgent(input_size, action_size, hidden_size, learning_rate, gamma)

# 定义tf图
with tf.Session() as sess:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    
    nb_episodes = 0
    
    # 定义 total_rewards 和 total_length
    total_reward = []
    total_length = []
    
 
    gradBuffer = sess.run(tf.trainable_variables())
    for ix,grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0
        
    
    # While we have episodes to train
    while nb_episodes < train_episodes:
        state = env.reset()
        running_reward = 0
        episode_history = [] # Init the array that keep track the history in an episode
        
        for step in range(max_steps):
            #Probabilistically pick an action given our network outputs.
            # Not my implementation: taken from Udacity Q-learning quart https://github.com/udacity/deep-learning/blob/master/reinforcement/Q-learning-cart.ipynb 
            action_distribution = sess.run(agent.output_layer ,feed_dict={agent.inputs:[state]})
            action = np.random.choice(action_distribution[0],p=action_distribution[0])
            action = np.argmax(action_distribution == action)
            
            state_1, reward, done, info = env.step(action)
            
            # 把当前步加入到 episode历史
            episode_history.append([state, action, reward, state_1])
            
            #  state 现在为 state 1
            state = state_1
            
            running_reward += reward
            
            if done == True:
                # 更新网络参数
                episode_history = np.array(episode_history)
                episode_history[:,2] = discount_rewards(episode_history[:,2])
                feed_dict={agent.rewards:episode_history[:,2],
                        agent.actions:episode_history[:,1],agent.inputs:np.vstack(episode_history[:,0])}
                grads = sess.run(agent.gradients, feed_dict=feed_dict)
                
                
                for idx,grad in enumerate(grads):
                    gradBuffer[idx] += grad

                if nb_episodes % batch_size == 0 and nb_episodes != 0:
                    feed_dict= dictionary = dict(zip(agent.gradient_holders, gradBuffer))
                    _ = sess.run(agent.update_batch, feed_dict=feed_dict)
                    for ix,grad in enumerate(gradBuffer):
                        gradBuffer[ix] = grad * 0
                
                #(running_reward))
                total_reward.append(running_reward)
                total_length.append(step)
                break
                
        # 每 100 episodes 打印
        if nb_episodes % 100 == 0:
            print("Episode: {}".format(nb_episodes),
                    "Total reward: {}".format(np.mean(total_reward[-100:])))
        nb_episodes += 1
    
    saver.save(sess, "checkpoints/cartPoleGame.ckpt")
        
        
  


# ### 测试agent

# In[7]:


test_episodes = 10
test_max_steps = 400
env.reset()
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
    
    for episode in range(1, test_episodes):
        t = 0
        while t < test_max_steps:
            env.render() 
            
        
            
            #从输出中概率性选择action.
            # 参考 https://github.com/udacity/deep-learning/blob/master/reinforcement/Q-learning-cart.ipynb 
            action_distribution = sess.run(agent.output_layer ,feed_dict={agent.inputs:[state]})
            action = np.random.choice(action_distribution[0],p=action_distribution[0])
            action = np.argmax(action_distribution == action)
            
            state_1, reward, done, info = env.step(action)
           
            
            if done:
                t = test_max_steps
                env.reset()
                # 采样动作
                state, reward, done, info = env.step(env.action_space.sample())

            else:
                state = state_1 # Next state
                t += 1
                
env.close()


# In[ ]:




