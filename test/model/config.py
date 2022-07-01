# -*- coding:utf-8 -*-
# @Time    : 2022/7/1 8:56
# @Author  : Yinkai Yang
# @FileName: config.py
# @Software: PyCharm
# @Description: this is a program related to
import torch

# 初始化参数设置
PAD = 0                             # padding占位符的索引
UNK = 1                             # 未登录词标识符的索引
BATCH_SIZE = 128                    # 批次大小
EPOCHS = 20                         # 训练轮数
LAYERS = 6                          # transformer中encoder、decoder层数
H_NUM = 8                           # 多头注意力个数
D_MODEL = 256                       # 输入、输出词向量维数(其实transformer中这个是512，需要根据实际问题进行一些简化)
D_FF = 1024                         # feed forward全连接层维数
DROPOUT = 0.1                       # dropout比例
MAX_LENGTH = 60                     # 语句最大长度

TRAIN_FILE = 'data/en-cn/train.txt'  # 训练集
DEV_FILE = "data/en-cn/dev.txt"      # 验证集
SAVE_FILE = 'save/model.pt'         # 模型保存路径
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")