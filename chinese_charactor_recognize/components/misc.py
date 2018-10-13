# coding=utf-8
# date: 2018-10-13,15:51:26
# name: smz


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


def codesToSparseTensor(batchSize, codes):
    """序列标签 --> 稀疏张量"""




if __name__ == "__main__":
    import pickle as pkl
    zhDictPath = "/media/smz/SMZ_WORKING_SPACE/MyNLP/chinese_charactor_recognize/data/zhDict.pkl"
    with open(zhDictPath, 'rb') as f:
        zhDict = pkl.load(f)
        print(zhDict.getAllChars())

    chars = '张莫'
    encodes = encodeChars(zhDict, chars)
    print("张莫:{}".format(encodes))


