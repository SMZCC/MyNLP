# coding=utf-8
# date: 2018年10月12日,22点08分
# name: smz

from collections import OrderedDict
import glob
import cv2
import os
import pickle as pkl

"""
抽取出训练文件的data, pkl的序列化格式为：[[<Dict:'image'-ndarray, 'label'-str>], [], []]
注： 当文件因为一些奇怪的编码而打不开的时候,试试使用二进制的形式打开文件
"""


def extractData(dataPath):
    img_paths = sorted(glob.glob(os.path.join(dataPath, '*.jpg')))
    txt_paths = sorted(glob.glob(os.path.join(dataPath, '*.txt')))

    train_data = []
    for idx, img_path in enumerate(img_paths):
        sample = OrderedDict()
        if os.path.splitext(img_path)[0] == os.path.splitext(txt_paths[idx])[0]:
            image = cv2.imread(img_path)
            with open(txt_paths[idx], 'r') as f:
                labelLine = f.readline()
        else:
            print("s"%(img_path))
            continue
        sample['image'] = image
        sample['label'] = labelLine

        train_data.append(sample)

    with open("./char_train_data.pkl", 'wb') as f:
        pkl.dump(train_data, f, protocol=2)


if __name__ == "__main__":
    dataPath = "E:\\test\\part2"
    extractData(dataPath)