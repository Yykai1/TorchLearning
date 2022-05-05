# -*- coding:utf-8 -*-
# @Time    : 2022/5/3 14:36
# @Author  : Yinkai Yang
# @FileName: 11-RNN_GPU.py
# @Software: PyCharm
# @Description: this is a program related to
# 注意：1.构建网络时，把网络，与损失函数转换到GPU上（第一肯定是判断GPU是否可用）
#      2.训练网络时，把数据转换到GPU上
#      3.只有tensor类型才能上传到GPU上，故需要对numpy数据进行转换成rensor类型
#      4.对训练的输出结果有些需要使用np的函数进行操作，需先将输出结果转到CPU上，并转成numpy类型，再使用np的函数

import os
import string
import unicodedata
import glob
import torch
import torch.nn as nn
from torch import Tensor
import random
import time
import math

all_letters = string.ascii_letters + ",.;'-"
n_letters = len(all_letters) + 1  # plus EOS maker
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

category_lines = {}  # 字典
all_categories = []  # 每一个种类的名称


def find_files(path):
    return glob.glob(path)


def unicode2ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


def read_lines(filename):
    txt_lines = open(filename, 'r', encoding='utf-8').read().strip().split('\n')
    return [unicode2ascii(txt_line) for txt_line in txt_lines]


def construct_dictionary() -> int:
    for filename in find_files('data/names/*.txt'):
        type_name = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(type_name)
        lines = read_lines(filename)
        category_lines[type_name] = lines
    number = len(all_categories)
    if number == 0:
        raise RuntimeError('Data not found. Make sure that you downloaded data '
                           'from https://download.pytorch.org/tutorial/data.zip and extract it to '
                           'the current directory.')
    return number


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_hize = hidden_size

        self.i2h = nn.Linear(n_categories + input_size + hidden_size, hidden_size).cuda()
        self.i2o = nn.Linear(n_categories + input_size + hidden_size, output_size).cuda()
        self.o2o = nn.Linear(hidden_size + output_size, output_size).cuda()
        self.dropout = nn.Dropout(0.1).cuda()
        self.softmax = nn.LogSoftmax(dim=1).cuda()

    def forward(self, category, input, hidden):
        input_combined = torch.cat((category, input, hidden), 1).cuda()
        hidden = self.i2h(input_combined).cuda()
        output = self.i2o(input_combined).cuda()
        output_combined = torch.cat((hidden, output), 1).cuda()
        output = self.o2o(output_combined).cuda()
        output = self.dropout(output).cuda()
        output = self.softmax(output).cuda()
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_hize).cuda()


def random_choice(thing):
    return thing[random.randint(0, len(thing) - 1)]


def random_train_pair() -> [str, str]:
    """

    :return: name类别, line具体的名字
    """
    name = random_choice(all_categories)
    line = random_choice(category_lines[name])
    return name, line


# One-hot vector for category
def category2tensor(category) -> Tensor:
    li = all_categories.index(category)
    tensor = torch.zeros(1, n_categories).cuda()
    tensor[0][li] = 1
    return tensor  # 二维[1 * 18]


# One-hot matrix of first to last letters (not including EOS) for input
def input2tensor(line):
    tensor = torch.zeros(len(line), 1, n_letters).cuda()
    for i in range(len(line)):
        c = line[i]
        tensor[i][0][all_letters.find(c)] = 1
    return tensor  # 三维[len * 1 * 18]


# LongTensor of second letter to end (EOS) for target
def target2tensor(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]  # 从第二个字母开始读取，最后补充一个EOS，仍然等长
    letter_indexes.append(n_letters - 1)  # 加入EOS的下标位置
    torch.LongTensor(letter_indexes).cuda()
    return torch.LongTensor(letter_indexes)  # 一维[len]


# Make category, input, and target tensors
def random_train_example():
    category, line = random_train_pair()
    category_tensor = category2tensor(category).cuda()
    input_line_tensor = input2tensor(line).cuda()
    target_line_tensor = target2tensor(line).cuda()
    return category_tensor, input_line_tensor, target_line_tensor


def train(category_tensor, input_line_tensor, target_line_tensor):
    target_line_tensor.unsqueeze_(-1)  # 增加一个维度
    hidden = rnn.init_hidden()
    target_line_tensor.cuda()
    hidden.cuda()

    rnn.zero_grad()

    loss = 0

    for i in range(input_line_tensor.size(0)):
        output, hidden = rnn(category_tensor, input_line_tensor[i], hidden)
        output.cuda(), hidden.cuda()
        loss_temp = criterion(output, target_line_tensor[i]).cuda()
        loss += loss_temp
        loss.cuda()

    loss.backward()

    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item() / input_line_tensor.size(0)


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


n_categories = construct_dictionary()
# print('# categories:', n_categories, all_categories), print(unicode_to_ascii("O'Néàl"))
criterion = nn.NLLLoss().cuda()
learning_rate = 0.0005

rnn = RNN(n_letters, 128, n_letters).cuda()

n_iters = 100000
print_every = 5000
plot_every = 500
all_losses = []
total_loss = 0  # Reset every plot_every iters

start = time.time()

for iter in range(1, n_iters + 1):
    output, loss = train(*random_train_example())
    total_loss += loss

    if iter % print_every == 0:
        print('%s (%d %d%%) %.4f' % (time_since(start), iter, iter / n_iters * 100, loss))

    if iter % plot_every == 0:
        all_losses.append(total_loss / plot_every)
        total_loss = 0
