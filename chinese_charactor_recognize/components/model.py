# coding=utf-8
# date: 2018-10-13,10:35:01
# name: smz

import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import LSTMCell
from chinese_charactor_recognize.configure.configuration import *


class ZHModel(object):
    def __init__(self, isTrain=True):
        self.isTrain = isTrain
                                                         # (batchSize, 192, 32)
        self.inputs = tf.placeholder(dtype=tf.float32, shape=[None, None, IMAGE_SHAPE[0]],name='Inputs')
        self.labels = tf.sparse_placeholder(dtype=tf.int32, name='SparseLabels')  # shape默认为None
        self.seqLens = tf.placeholder(dtype=tf.int32, name='SeqLens')

    def build(self):

        # lstm
        cell = LSTMCell(NUM_HIDDEN)
        outputs, _ = tf.nn.dynamic_rnn(cell, self.inputs, self.seqLens, dtype=tf.float32)
        # (N, maxTimeSteps, numHidden)

        inputShape = tf.shape(self.inputs)  # (batchSize, 192, 32)
        # 接一层全连接求出每个序列的时间步的对应结果
        self.outputs = tf.reshape(outputs, [-1, NUM_HIDDEN])  # (batchSize*maxTimeSteps, numHidden)
        self.W = tf.Variable(tf.truncated_normal(shape=[NUM_HIDDEN, NUM_CLASSES], stddev=0.1), name='W')
        self.b = tf.Variable(tf.constant(0., shape=[NUM_CLASSES]), name='b')
        self.logits = tf.matmul(self.outputs, self.W) + self.b

        self.logits = tf.reshape(self.logits, [inputShape[0], -1, NUM_CLASSES])
        # 要使用ctc计算loss,换轴
        self.logits = tf.transpose(self.logits, (1, 0, 2))   # (maxTimeSteps, batchSize, numHidden)

        if self.isTrain:
            self.globalStep = tf.Variable(0, trainable=False)
            self.learningRate = tf.train.exponential_decay(INITIAL_LEARNING_RATE, self.globalStep,
                                                           DECAY_STEPS, LERNING_RATE_DECAY_FACTOR,
                                                           staircase=True)
            self.loss = tf.nn.ctc_loss(labels=self.labels, inputs=self.logits, sequence_length=self.seqLens)
            self.cost = tf.reduce_mean(self.loss, name='cost')
            tf.summary.scalar('cost', self.cost)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learningRate).minimize(self.cost, global_step=self.globalStep)
            self.mergeSummary = tf.summary.merge_all()

        self.decoded, self.logProb = tf.nn.ctc_beam_search_decoder(self.logits, self.seqLens, merge_repeated=False)
        self.saver = tf.train.Saver()

        # FileWriter和Saver都必须要在构图完成之后在建立,否则收集不到想要收集的数据
    def restoreModel(self, sess, modelPath):
        print("restoring model from %s"%(modelPath))
        self.saver.restore(sess, modelPath)

    def train(self, sess, inputs, labels, seqLens):

        feedDict = {self.inputs:inputs,
                    self.labels:labels,
                    self.seqLens:seqLens}
        _cost, _, mergeSummary, globalStep = sess.run(fetches=[self.cost, self.optimizer,self.mergeSummary, self.globalStep],
                                     feed_dict=feedDict)
        print("globalStep:%d, cost:%.6f"%(globalStep, _cost))
        # self.writer.add_summary(mergeSummary, globalStep)
        return mergeSummary, globalStep

    def test(self, sess, inputs, labels, seqLens):

        feedDict = {self.inputs:inputs,
                    self.labels:labels,
                    self.seqLens:seqLens}

        decoded = sess.run(fetches=[self.decoded], feed_dict=feedDict)

        return decoded






