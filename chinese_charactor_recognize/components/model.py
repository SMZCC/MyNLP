# coding=utf-8
# date: 2018-10-13,10:35:01
# name: smz

import tensorflow as tf


class model(object):
    def __init__(self):
        pass

    @property
    def isTrain(self):
        return self.__isTrain__

    @isTrain.setter
    def isTrain(self, value):
        self.__isTrain__ = value

    def build(self):
        pass

