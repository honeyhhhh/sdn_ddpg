# 导入相关模块
import numpy as np
import math
import tensorflow as tf
import keras.backend as K
from keras.layers import Input, Dense
from keras.models import Model
from keras.utils import plot_model

# Actor神经网络
class ActorNetwork(object):
    def __init__(self, sess, state_size, action_size, batch_size, tau, learning_rate):
        self.sess = sess
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.tau = tau  # soft update discount factor

        self.HIDDEN1_UNITS = 50
        self.HIDDEN2_UNITS = 50

        K.set_session(sess)

        # 创建神经网络
        # eval网络
        self.model, self.weights, self.state =self.creat_actor_network(state_size, action_size)
        # target网络
        self.target_model, self.target_weights, self.target_state = self.creat_actor_network(state_size, action_size)
        # 从Critic传过来的梯度dQ/da
        # the gradients of the policy to get more Q
        self.action_gradient = tf.placeholder(tf.float32, [None, action_size])
        # 计算Actor的梯度
        # tf.gradients will calculate dys/dxs with a initial gradients for ys, so this is dq/da * da/dparams
        self.policy_grad =tf.gradients(ys=self.model.output, xs=self.weights, grad_ys=-self.action_gradient)
        grads = zip(self.policy_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)
        self.sess.run(tf.global_variables_initializer())

    # 训练eval网络
    def train(self, states, action_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.action_gradient: action_gradient
        })

    # 训练target网络，即soft更新网络
    def target_train(self):
        eval_weights = self.model.get_weights()
        target_weight = self.target_model.get_weights()
        # soft update target network weights
        for i in range(len(eval_weights)):
            target_weight[i] = self.tau * eval_weights[i] + (1 - self.tau) * target_weight[i]
        self.target_model.set_weights(target_weight)

    # 创建Actor网络模型
    def creat_actor_network(self, state_size, action_size):
        # debug
        print("Now we will creat Actor network")
        # 初始化
        # 初始化卷积核权重和偏置参数
        init_w = tf.random_normal_initializer(0., 0.1)
        init_b = tf.constant_initializer(0.01)

        S = Input(shape=[state_size])   # 创建一个输入Tensor
        h0 = Dense(self.HIDDEN1_UNITS, activation='relu', kernel_initializer=init_w, bias_initializer=init_b)(S)
        h1 = Dense(self.HIDDEN2_UNITS, activation='relu', kernel_initializer=init_w, bias_initializer=init_b)(h0)
        # 这里需要实现输出SDN节点的个数和SDN的部署位置之间的数量是同步的

        Link_weights = Dense(action_size, activation='tanh', kernel_initializer=init_w, bias_initializer=init_b)(h1)

        model = Model(input=S, output=Link_weights)

        plot_model(model, to_file="model.png", show_shapes=True)

        return model, model.trainable_weights, S


if __name__ == '__main__':
    # 神经网络相关操作定义
    sess = tf.Session()
    from keras import backend as K

    K.set_session(sess)
    # 初始化四个个网络
    actor = ActorNetwork(sess, 56, 10, 16, 0.01, 0.01)
