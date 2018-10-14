# coding=utf-8
# date: 2018-10-14,16:37:17
# name: smz

import numpy as np
import pickle as pkl
import tensorflow as tf
import matplotlib.pyplot as plt
from chinese_charactor_recognize.components.model import ZHModel
from chinese_charactor_recognize.components.misc import decodeToCodes
from chinese_charactor_recognize.components.datasets import TestDataSet


def main():
    dictPath = "/media/smz/SMZ_WORKING_SPACE/MyNLP/chinese_charactor_recognize/data/zhDict.pkl"
    dataPath = "/media/smz/SMZ_WORKING_SPACE/MyNLP/chinese_charactor_recognize/data/char_data_test_2015.pkl"
    zhDictPath = "/media/smz/SMZ_WORKING_SPACE/MyNLP/chinese_charactor_recognize/data/zhDict.pkl"
    modelPath = "/media/smz/SMZ_WORKING_SPACE/MyNLP/chinese_charactor_recognize/trained_models/ZHModel.ckpt"

    testDataSet = TestDataSet(dictPath, dataPath)
    with open(zhDictPath, 'rb') as f:
        zhDict = pkl.load(f)

    zhModel = ZHModel(isTrain=False)
    zhModel.build()

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        zhModel.restoreModel(sess, modelPath=modelPath)
        imgs, sparseTensorArgs, seqLens = testDataSet.next()
        imgs_ = np.asarray(imgs.copy(), dtype=np.float32)
        imgs_ = imgs_.transpose((0, 2, 1))
        decodes = zhModel.test(sess, imgs_, sparseTensorArgs, seqLens)
        codes = decodeToCodes(decodes[0][0])
        charIdxs = codes[0]
        chars = ''   # 识别的结果
        for charIdx in charIdxs:
            char = zhDict[charIdx]
            chars += char

        fig = plt.figure()
        ax_im = fig.add_subplot(1, 1, 1)
        ax_im.axis("off")
        ax_im.imshow(imgs[0])
        ax_im.set_title("prediction result:%s"%(chars), fontdict={"family":"SimHei"})

        plt.show()


if __name__ == "__main__":
    main()