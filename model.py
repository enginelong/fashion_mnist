import os

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets


class FashionMnist:
    out_features1 = 12  # 第一个卷积池化层输出特征数量
    out_features2 = 24  # 第二个卷积池化层输出特征数量
    con_neurons = 512  # 全连接层神经元数量

    def __init__(self, path):
        """
        类初始化
        :param path: 数据集路径
        """
        self.sess = tf.Session()  # 定义会话
        self.data = read_data_sets(path, one_hot=True)

    def init_weight_variable(self, shape):
        """
        权重初始化
        :param shape: 待初始化张量的形状
        :return: 初始化之后张量的形状
        """
        inital = tf.truncated_normal(shape, stddev=0.1)  # 截尾正态分布

        return tf.Variable(inital)

    def init_bias_variable(self, shape):
        """
        偏置初始化
        :param shape: 待初始化张量的形状
        :return: 初始化之后张量的形状
        """
        inital = tf.constant(1.0, shape=shape)

        return tf.Variable(inital)

    def conv2d(self, x, w):
        """
        卷积操作
        :param x: 输入张量形状
        :param w: 卷积核 [filter_height,filter_width,in_channels,out_channels]
        :return: 卷积后结果
        """
        return tf.nn.conv2d(
            x, w,
            strides=[1, 1, 1, 1],  # 卷积时各个维度的步长 [b,w,h,c]
            padding='SAME'  # 填充使得输入和输出矩阵维度相同
        )

    def max_pool_2x2(self, x):
        """
        池化操作
        :param x: 输入张量形状
        :return: 池化后结果
        """
        return tf.nn.max_pool(
            x,
            ksize=[1, 2, 2, 1],  # 池化区域大小
            strides=[1, 2, 2, 1],  # 池化时各个维度的步长
            padding='SAME'  # 填充使得输入和输出矩阵维度相同
        )

    def create_conv_pool_layer(self, input, input_features, out_features):
        """
        创建卷积池化层
        :param input: 原始数据
        :param input_features: 输入特征维度
        :param out_features: 输出特征维度
        :return:
        """
        filter = self.init_weight_variable([5, 5, input_features, out_features])
        b = self.init_bias_variable([out_features])
        relu = tf.nn.relu(self.conv2d(input, filter) + b)
        pool = self.max_pool_2x2(relu)

        return pool

    def create_fc_layer(self, conv_flat, input_features, output_features):
        """
        创建全连接层
        :param conv_flat: 来自卷积层后拉伸的数据
        :param input_features: 输入特征的维度
        :param output_features: 输出特征的维度
        :return: 全连接层处理后的结果
        """
        w = self.init_weight_variable([input_features, output_features])
        b = self.init_bias_variable([output_features])
        relu = tf.nn.relu(tf.matmul(conv_flat, w) + b)

        return relu

    def build(self):
        """组网"""
        self.x = tf.placeholder(tf.float32, shape=[None, 784])
        input = tf.reshape(self.x, [-1, 28, 28, 1])
        self.label = tf.placeholder(tf.float32, [None, 10])  # 标签使用独热编码

        # conv-1
        conv1 = self.create_conv_pool_layer(input, 1, self.out_features1)

        # conv-2
        conv2 = self.create_conv_pool_layer(conv1, self.out_features1, self.out_features2)

        # fc-1
        conv_flat_shape = 7 * 7 * self.out_features2
        conv_flat = tf.reshape(conv2, [-1, conv_flat_shape])
        fc1 = self.create_fc_layer(conv_flat, conv_flat_shape, self.con_neurons)

        ## dropout(随机丢弃一些参数防止过拟合)
        self.keep_prob = tf.placeholder('float')
        dropout = tf.nn.dropout(fc1, self.keep_prob)

        # out
        out_w = self.init_weight_variable([self.con_neurons, 10])  # 产生10个输出
        out_b = self.init_bias_variable([10])
        out = tf.matmul(dropout, out_w) + out_b

        # 评价模型
        correct_pred = tf.equal(tf.argmax(out, 1),
                                tf.argmax(self.label, 1))
        self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # 定义损失函数
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.label,
                                                       logits=out)
        cross_entropy = tf.reduce_mean(loss)

        # 优化器
        optimizer = tf.train.AdamOptimizer(0.001)  # 动量优化器
        self.train_process = optimizer.minimize(cross_entropy)

    def train(self):
        """模型训练"""
        self.sess.run(tf.global_variables_initializer())  # 变量初始化
        batch_size = 100
        print('start training')

        saver = tf.train.Saver()  # 实例化模型加载/保存对象
        # 如果存在已经训练的模型则加载继续训练(增量训练)
        if os.path.exists('./model_save/checkpoint'):
            saver.restore(self.sess, './model_save/')

        for i in range(20):
            total_batchs = int(self.data.train.num_examples / batch_size)
            for j in range(total_batchs):
                batch = self.data.train.next_batch(batch_size)  # 获取一个批次样本
                params = {self.x: batch[0], self.label: batch[1], self.keep_prob: 0.5}
                _, acc = self.sess.run([self.train_process, self.acc], params)

                if j % 100 == 0:
                    print(f'epoch:{i}, pass:{j}, acc:{acc}')

        saver.save(self.sess, './model_save/')

    def eval(self, x, label):
        """模型评价"""
        params = {self.x: x, self.label: label, self.keep_prob: 1.0}
        test_acc = self.sess.run(self.acc, params)
        print(f'Test accuracy is %f'%test_acc)

        return test_acc

    def close_sess(self):
        """关闭会话"""
        self.sess.close()
