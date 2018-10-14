# coding=utf-8
# date: 2018-10-13,21:05:24
# name: smz

import pickle as pkl
from chinese_charactor_recognize.components.misc import imgToGray
from chinese_charactor_recognize.components.misc import encodeChars
from chinese_charactor_recognize.components.misc import codesToSparseTensorArgs


class TrainDataSet(object):
    def __init__(self, dictPath=None, dataPath=None, batchSize=None):
        self.batchSize = batchSize
        self.pointer = 0  # 指向当前batch头元素的地址

        with open(dictPath, 'rb') as f:
            self.ZHDict = pkl.load(f)

        with open(dataPath, 'rb') as f:
            self.data = pkl.load(f)

    def __iter__(self):
        return self

    def __next__(self):
        nextPointer = self.pointer + self.batchSize
        if nextPointer > 8057:   # 我自己的数据集,我随意
            self.pointer = 0
            raise StopIteration
        images = []
        codes = []    # [[18, 27], [13], [56, 78, 45]]
        seqLens = []
        for sample in self.data[self.pointer:nextPointer]:
            img = sample['image']
            img = imgToGray(img)
            label = sample['label']
            code = encodeChars(self.ZHDict, label)
            images.append(img)    # (n, h, w)
            codes.append(code)
            seqLens.append(192)
        self.pointer = nextPointer
        return images, codesToSparseTensorArgs(codes), seqLens
    next = __next__  # 将next函数重新指向我实现的函数,该next需要与def齐平


class TestDataSet(object):
    def __init__(self, dictPath=None, dataPath=None):
        self.pointer = 0  # 指向当前batch头元素的地址

        with open(dictPath, 'rb') as f:
            self.ZHDict = pkl.load(f)

        with open(dataPath, 'rb') as f:
            self.data = pkl.load(f)

    def __iter__(self):
        return self

    def __next__(self):
        if self.pointer == 2015:
            self.pointer = 0
            raise StopIteration
        images = []     # (n, h, w)
        codes = []
        seqLens = []
        sample = self.data[self.pointer]
        img = imgToGray(sample['image'])
        code = encodeChars(self.ZHDict, sample['label'])
        images.append(img)
        codes.append(code)
        seqLens.append(len(code))
        self.pointer += 1
        return images, codesToSparseTensorArgs(codes, 1), seqLens
    next = __next__


if __name__ == "__main__":
    dictPath = "/media/smz/SMZ_WORKING_SPACE/MyNLP/chinese_charactor_recognize/data/zhDict.pkl"
    dataPath = '/media/smz/SMZ_WORKING_SPACE/MyNLP/chinese_charactor_recognize/data/char_data_train_8057.pkl'
    testDataPath = "/media/smz/SMZ_WORKING_SPACE/MyNLP/chinese_charactor_recognize/data/char_data_test_2015.pkl"
    trainDataSet = TrainDataSet(dictPath, dataPath, 16)
    testDataSet = TestDataSet(dictPath, testDataPath)
    counter = 0
    for imgs, codes in testDataSet:
        counter += 1
        print("pass")
    print(counter)