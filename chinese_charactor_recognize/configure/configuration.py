# coding=utf-8
# date: 2018-10-13,14:30:01
# name: smz

# img
OUTPUT_SHAPE = (32, 192)  # a gray img of shape (32, 192)

# train loop
NUM_EPOCHS = 10000
BATCH_SIZE = 16   # 修改这里的值的时候,注意修改datasets文件中对应的值
BATCHES = 503     # 训练样本总数是8057, 使用503*16=8048,舍弃部分数据
TRAIN_SIZE = 8057

# lstm
NUM_HIDDEN = 64
NUM_CLASSES = 1002  # 一个空格+1001个字,没算ctc的空字符‘-’,我觉得那个东西,与我自己的标签表示根本没有关系

# initialize learning rate
INITIAL_LEARNING_RATE = 1e-3
DECAY_STEPS = 5000
REPORT_STEPS = 100
LERNING_RATE_DECAY_FACTOR = 0.9
MOMENTUM = 0.9




