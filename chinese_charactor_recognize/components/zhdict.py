# coding=utf-8
# date: 2018-10-12,18:32:39
# name: smz

"""
中文字典类,用于保存已有的汉字
支持方法：
    [idx]         获取索引为idx的
    in            判断某个字符是否在字典中
    addChar(char) 将char添加到索引中
    getIdx(char)  返回Char的索引
    for xx in <ZHDict>
    len
"""


class ZHDict(object):
    def __init__(self):
        self.__charactors__ = [' ']

    def addChar(self, char):
        """将字符char添加到索引中"""
        if char not in self.__charactors__:
            self.__charactors__.append(char)

    def getCharIdx(self, char):
        """获取字符char对应的索引"""
        for idx, _char in enumerate(self.__charactors__):
            if _char == char: return idx

        assert False, "%s is not in ZHDict"%(char)

    def getAllIdx(self):
        """获取所有字符的索引"""
        return range(len(self.__charactors__))

    def getAllChars(self):
        return self.__charactors__

    def __contains__(self, item):
        return item in self.__charactors__

    def __getitem__(self, idx):
        return self.__charactors__[idx]

    def __iter__(self):
        return iter(self.__charactors__)

    def __len__(self):
        return len(self.__charactors__)


if __name__ == '__main__':
    zhDict = ZHDict()

    print("zhDict[0]:", zhDict[0])
    zhDict.addChar('你好')
    print("zhDict[1]:", zhDict[1])
    print("getIdx(' ')", zhDict.getIdx(' '))
    i = 0
    for char in zhDict:
        i += 1
        print("第%d个：%s"%(i, char))

    print("'你好' in zhDict:", '你好' in zhDict)



