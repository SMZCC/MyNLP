# coding=utf-8
# date: 2018-10-13,15:51:26
# name: smz


import numpy as np
import tensorflow as tf


def splitZH(char):
    """将多个连在一起的中文字符分割开来
    如： “你好呀”,分割成 '你‘, '好', '呀'
    """
    charLength = len(char)
    for i in range(charLength):
        yield char[i]


def encodeChars(ZHDict, chars):
    """使用字典ZHDict来编码chars
    args:
        ZHDict: <ZHDict>
        chars: for example: 'hello world'
    """
    encodes = []
    for char in splitZH(chars):
        if char in ZHDict:
            code = ZHDict.getCharIdx(char)
            encodes.append(code)

    return encodes


def imgToGray(img):
    """img: ndarray, bgr"""
    img_shape = img.shape
    if len(img_shape) == 2:
        return img
    else:
        return img[:, :, 0] * 0.114 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.299


def codesToSparseTensor(codes, batchSize=3):
    """序列标签 --> 稀疏张量
    args:
        codes: 2-dim,指的是一个batch的样本中每个样本与汉字的映射编码
    """
    idxs = []
    values = []
    codes = np.asarray(codes)
    for i in range(batchSize):
        for j in range(len(codes[i])):
            idxs.append((i, j))
            values.append(codes[i][j])
    print("idxs:", idxs)
    print("values:", values)
    sparseTensorShape = (batchSize, codes.max(0)[1]+1)
    return tf.SparseTensor(idxs, values, sparseTensorShape)


def decodeToCodes(sparseTensor):
    """py3,sparseTensor ---> indices, values"""
    idxs = sparseTensor[0]
    codes = []    # 每个样本的编码
    values = sparseTensorValue[1]   # [1, 2, 4, 5, 6, 7]
    numH = idxs.max(0)[0] + 1  # sparseTensor的height
    for i in range(numH):
        sample = []
        for valueIdx, idx in enumerate(idxs):
            if idx[0] == i:
                sample.append(values[valueIdx])
        codes.append(sample)

    return codes


if __name__ == "__main__":
    import pickle as pkl
    zhDictPath = "/media/smz/SMZ_WORKING_SPACE/MyNLP/chinese_charactor_recognize/data/zhDict.pkl"
    with open(zhDictPath, 'rb') as f:
        zhDict = pkl.load(f)
        print(zhDict.getAllChars())

    chars = '张莫'
    encodes = encodeChars(zhDict, chars)
    print("张莫:{}".format(encodes))

    codes = [[1, 2], [4], [5, 6, 7]]
    sparseTensor = codesToSparseTensor(codes)
    print("sparseTensor:\n", sparseTensor)
    with tf.Session() as sess:
        sparseTensorValue = sess.run(sparseTensor)
        decodes = decodeToCodes(sparseTensorValue)
        print("decodeToCodes:\n", decodes)

