# coding=utf-8
# date: 2018-10-12,18:40:54
# name: smz

"""
1.从训练数据中抽取出所有汉字,构成字典
"""
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from chinese_charactor_recognize.components.misc import splitZH
from chinese_charactor_recognize.components.zhdict import ZHDict


def constrackZHDict(dataPath, ZHDictSavePath):
    with open(dataPath, 'rb') as f:
        data = pkl.load(f)
    zhDict = ZHDict()
    for sample in data:
        label = sample['label']
        for char in label:
            if char not in zhDict:
                zhDict.addChar(char)

    with open(ZHDictSavePath, 'wb') as f:
        pkl.dump(zhDict, f)


def splitData(dataPath, trainRatio=0.8):
    """将10072个样本拆分成训练集和测试集"""
    with open(dataPath, 'rb') as f:
        data = pkl.load(f)

    num_data = len(data)
    num_train = int(num_data * trainRatio)
    idxs = np.random.permutation(range(num_data))
    train_idxs =  idxs[:num_train]
    test_idxs = idxs[num_train:]

    train_data = []
    test_data = []
    for idx in train_idxs:
        train_data.append(data[idx])
    for idx in test_idxs:
        test_data.append(data[idx])

    with open('/media/smz/SMZ_WORKING_SPACE/MyNLP/chinese_charactor_recognize/data/char_data_train_%d.pkl'%(num_train), 'wb') as f:
        pkl.dump(train_data, f)

    with open("/media/smz/SMZ_WORKING_SPACE/MyNLP/chinese_charactor_recognize/data/char_data_test_%d.pkl"%(num_data-num_train), 'wb') as f:
        pkl.dump(test_data, f)


def findNone(dataPath):
    """查找数据集中图片读取失败的样本"""
    counter = 0
    idxs = []
    with open(dataPath, 'rb') as f:
        data = pkl.load(f)

    for idx, sample in enumerate(data):
        if sample['image'] is None:
            print("idx:", idx)
            idxs.append(idx)
            counter += 1
    print("counter:", counter)
    return idxs


def clearCharData(dataPath):
    with open(dataPath, 'rb') as f:
        data = pkl.load(f)
    total = len(data)
    idxs = findNone(dataPath)
    clearData = []
    for sample in data:
        if not sample['image'] is None:
            clearData.append(sample)
    print("clear %d samples, remain %d samples" % (len(idxs), total - len(idxs)))
    print("clearCharDate:%d" % (len(clearData)))

    with open("/media/smz/SMZ_WORKING_SPACE/MyNLP/chinese_charactor_recognize/data/char_data_cleared.pkl", 'wb') as f:
        pkl.dump(clearData, f)


############################以下检查测试用 ############################
def checkTrainData(dataPath):
    with open(dataPath, 'rb') as f:
        data = pkl.load(f)

    img = data[0]['image']
    label = data[0]['label']
    print("label:", label)
    print("img.shape:", img.shape)

    for idx, ch in enumerate(splitZH(label)):
        print("ch_%d:%s"%(idx, ch))

    # gray
    imgGray = img[:, :, 0] * 0.114 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.299

    imgRed = img[:, :, 2]
    imgGreen = img[:, :, 1]
    imgBlue = img[:, :, 0]

    fig = plt.figure()
    ax = fig.add_subplot(231)
    ax.imshow(img)
    ax.axis('off')
    ax.set_title("%s"%(label), fontdict={'family':'SimHei'})

    ax_gray = fig.add_subplot(2, 3, 2)
    ax_gray.imshow(imgGray)
    ax_gray.axis("off")
    ax_gray.set_title("Gray mode", fontdict={'fontsize':10})

    ax_red = fig.add_subplot(2, 3, 4)
    ax_red.imshow(imgRed, cmap='gray')
    ax_red.axis("off")
    ax_red.set_title("Red channel")

    ax_green = fig.add_subplot(2, 3, 5)
    ax_green.imshow(imgGreen, cmap='gray')
    ax_green.axis("off")
    ax_green.set_title("Green channel")

    ax_blue = fig.add_subplot(2, 3, 6)
    ax_blue.imshow(imgBlue, cmap='gray')
    ax_blue.axis("off")
    ax_blue.set_title("Blue channel")

    plt.show()


def checkZHDict(ZHDictPath):
    with open(ZHDictPath, 'rb') as f:
        zhDict = pkl.load(f)
    print(zhDict.getAllChars())
    print(type(zhDict))
    print("len(zhDict):", len(zhDict))


if __name__ == "__main__":
    dataPath = "/media/smz/SMZ_WORKING_SPACE/MyNLP/chinese_charactor_recognize/data/char_data.pkl"
    zhDictSavePath = '/media/smz/SMZ_WORKING_SPACE/MyNLP/chinese_charactor_recognize/data/zhDict.pkl'
    clearedDataPath = "/media/smz/SMZ_WORKING_SPACE/MyNLP/chinese_charactor_recognize/data/char_data_cleared.pkl"
    # checkTrainData(dataPath)
    # clearCharData(dataPath)
    # constrackZHDict(dataPath, zhDictSavePath)
    # checkZHDict(zhDictSavePath)
    # splitData(clearedDataPath)
    # findNone(dataPath)


