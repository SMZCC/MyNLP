# coding=utf-8
# date: 2018-10-14,15:34:32
# name: smz

import numpy as np
import tensorflow as tf
from chinese_charactor_recognize.configure.configuration import *
from chinese_charactor_recognize.components.model import ZHModel
from chinese_charactor_recognize.components.datasets import TrainDataSet


def main():
    dictPath = "/media/smz/SMZ_WORKING_SPACE/MyNLP/chinese_charactor_recognize/data/zhDict.pkl"
    dataPath = "/media/smz/SMZ_WORKING_SPACE/MyNLP/chinese_charactor_recognize/data/char_data_train_8057.pkl"
    trainDataSet = TrainDataSet(dictPath, dataPath, BATCH_SIZE)

    zhModel = ZHModel()
    zhModel.build()

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        summaryWriter = tf.summary.FileWriter('/media/smz/SMZ_WORKING_SPACE/MyNLP/chinese_charactor_recognize/logs', graph=sess.graph)
        for epoch in range(NUM_EPOCHS):
            print("Epoch:%d"%(epoch+1))
            for imgs, sparseTensorArgs, seqLens in trainDataSet:
                imgs = np.asarray(imgs, dtype=np.float32)
                imgs = imgs.transpose((0, 2, 1))
                mergeSummary, globStep = zhModel.train(sess, imgs, sparseTensorArgs, seqLens)
                summaryWriter.add_summary(mergeSummary, global_step=globStep)

            if (epoch+1) % 5 == 0:
                zhModel.saver.save(sess, '/media/smz/SMZ_WORKING_SPACE/MyNLP/chinese_charactor_recognize/trained_models/ZHModel.ckpt', global_step=globStep)

        summaryWriter.close()

if __name__ == "__main__":
    main()