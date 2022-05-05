# -*- coding:utf-8 -*-
# @Time    : 2022/5/2 14:10
# @Author  : Yinkai Yang
# @FileName: 09-character_level_RNN.py
# @Software: PyCharm
# @Description: this is a program related to
# unicode_literals: 2.x版本str类型的数据在调用len方法时是按照字节进行读取的，unicode类型的才是按照字符进行读取的
# print_function: 从2.x版本超前使用3.x版本的print函数
# division: 从未来的版本中导入精确除法 7 / 5 = 1.4
# 说实话我感觉这个没有必要
import random
from io import open
import glob
import os

import unicodedata
import string

import torch
import torch.nn as nn

import time
import math

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

all_letters = string.ascii_letters + " .,;'"  # a~z A~Z . , ; '
n_letters = len(all_letters)

# Build the category_lines dictionary, a list of names per language
category_lines = {}  # # 字典,一个语言对应一个字符串list
all_categories = []  # 每一种语言，一共是18


def find_files(path):
    """

    :param path: 路径
    :return: 路径下面的所有符合的文件构成的列表
    """
    return glob.glob(path)


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)  # 将Unicode文本标准化
        # 条件，c在字典Mn中以及c在all_letters中
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )
# print(unicode_to_ascii('Ślusàrski'))  # 测试结果：Slusarski


def read_lines(filename):
    lines = open(filename, 'r', encoding='utf-8').read().strip().split('\n')
    return [unicode_to_ascii(line) for line in lines]


for filename in find_files('data/names/*.txt'):
    name = os.path.splitext(os.path.basename(filename))[0]
    # os.path.basename(filename)：xxx.txt
    # os.path.splitext(os.path.basename(filename)) :('xxx', '.txt')
    all_categories.append(name)
    lines = read_lines(filename)
    category_lines[name] = lines
# print(category_lines['Italian'][:5])  # 测试: ['Abandonato', 'Abatangelo', 'Abatantuono', 'Abate', 'Abategiovanni']


def letter_to_index(letter):
    return all_letters.find(letter)  # 找到字母的下标


def letter_to_tensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letter_to_index(letter)] = 1  # 下标位置置为1
    return tensor


# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letter_to_index(letter)] = 1
    return tensor
# print(letter_to_tensor('j')), print(line_to_tensor('Yyk').size())  # 测试


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)  # [1 * 128 + 57] * [128 + 57, 128] = [1, 128]
        self.i2o = nn.Linear(input_size + hidden_size, output_size)  # [1 * 128 + 57] * [128 + 57, 18] = [1, 18]
        self.softmax = nn.LogSoftmax(dim=1)  # 第二个维度进行归一化 [1 * 18]

    def forward(self, input, hidden):
        # 在给定维度上对输入的张量序列进行连接操作
        combined = torch.cat((input, hidden), 1)  # 在第二个维度上进行连接
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)  # print(f'output {output.size()}')
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)


# 单词的测试
# input = letter_to_tensor('A')  # [1 * 57]
# hidden = torch.zeros(1, n_hidden)  # [1 * 128]
#
# output, next_hidden = rnn(input, hidden)
# print(output)

# 句子的测试
# input = line_to_tensor('Abate')  # Italian中的第4个，下标为3
# hidden = torch.zeros(1, n_hidden)
#
# output, next_hidden = rnn(input[0], hidden)
# print(output)


# 输出结果处理，找到最大的哪一个值确定分类结果
def category_from_output(output):
    top_n, top_i = output.topk(1)  # top_i是index,top_n是数据
    category_i = top_i[0].item()
    return all_categories[category_i], category_i
# print(category_from_output(output))


def random_choice(l):
    return l[random.randint(0, len(l) - 1)]


def random_training_example():
    """

    :return: name_type->int, r_line->int, r_category_tensor->tensor, r_line_tensor->tensor
    """
    name_type = random_choice(all_categories)  # 1~18个类别中的一个
    r_line = random_choice(category_lines[name_type])  # name_type类别中随机选一个
    r_category_tensor = torch.tensor([all_categories.index(name_type)], dtype=torch.long)  # 将类别转换成tensor
    r_line_tensor = line_to_tensor(r_line)
    return name_type, r_line, r_category_tensor, r_line_tensor


def train(category_tensor, line_tensor):
    hidden = rnn.init_hidden()
    rnn.zero_grad()   # 把模型中参数的梯度设为0

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


n_categories = len(all_categories)
n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)  # (57,128,18)

# for i in range(10):
#     category, line, category_tensor, line_tensor = random_training_example()
#     print('type =', category, '/ name =', line)

# 训练网络
criterion = nn.NLLLoss()
learning_rate = 0.01

n_iters = 100000
print_every = 5000
plot_every = 1000

# Keep track of losses for plotting
current_loss = 0
all_losses = []

start = time.time()

for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = random_training_example()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    # Print iter number, loss, name and guess
    if iter % print_every == 0:
        guess, guess_i = category_from_output(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, time_since(start), loss, line, guess, correct))

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

plt.figure()
plt.plot(all_losses)

# Keep track of correct guesses in a confusion matrix
confusion = torch.zeros(n_categories, n_categories)
n_confusion = 10000

# Just return an output given a line
def evaluate(line_tensor):
    hidden = rnn.init_hidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output

# Go through a bunch of examples and record which are correctly guessed
for i in range(n_confusion):
    category, line, category_tensor, line_tensor = random_training_example()
    output = evaluate(line_tensor)
    guess, guess_i = category_from_output(output)
    category_i = all_categories.index(category)
    confusion[category_i][guess_i] += 1

# Normalize by dividing every row by its sum
for i in range(n_categories):
    confusion[i] = confusion[i] / confusion[i].sum()

# Set up plot
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

# Set up axes
ax.set_xticklabels([''] + all_categories, rotation=90)
ax.set_yticklabels([''] + all_categories)

# Force label at every tick
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

# sphinx_gallery_thumbnail_number = 2
plt.show()