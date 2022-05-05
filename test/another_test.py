# -*- coding:utf-8 -*-
# @Time    : 2022/5/4 15:35
# @Author  : Yinkai Yang
# @FileName: another_test.py
# @Software: PyCharm
# @Description: this is a program related to
import unicodedata
import re

# index2word = {0: 'SOS', 1: 'EOS'}
# word2index = {'SOS': 0, 'EOS': 1}
# word = 'SOS'
# print(word2index[word])
# print(index2word[0])


def unicode2ascii(s):
    # Turn a Unicode string to plain ASCII
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalize_string(s):
    # 将所有内容变为小写，并修剪大多数标点符号
    s = unicode2ascii(s.lower().strip())
    s = re.sub(r'([.!?])', r' \1', s)
    s = re.sub(r'[^a-zA-Z.!?]+', r' ', s)
    return s


lines = open('../learn/data/eng-fra.txt', 'r', encoding='utf-8').read().strip().split('\n')

# 划分每一行并且normalize
pairs = [[normalize_string(s) for s in line.split('\t')] for line in lines]  # 二位列表
# print(pairs)
MAX_LENGTH = 10
eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

def filter_pair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH


def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]


pairs_change = filter_pairs(pairs)
print(pairs_change[9990:10000])
