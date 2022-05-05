# -*- coding:utf-8 -*-
# @Time    : 2022/5/5 13:59
# @Author  : Yinkai Yang
# @FileName: 12-translation_Seq2Seq.py
# @Software: PyCharm
# @Description: this is a program related to
import unicodedata
import re
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

# 数据集在data/eng-fra.txt
# 类似于字符编码的方式，将单词进行独热向量的编码，但因为单词的数量比较多，one-hot vector比较大，所以每一种语言仅用一部分
# 为每一单词的创建唯一索引，建立了一个辅助类Lang
# 功能:1.word->index(word2index)字典
#     2.index->word(index2word)字典
#     3.每一个单词的计数
#     4.word2count用于替换稀有的单词

MAX_LENGTH = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SOS_token = 0
EOS_token = 1


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: 'SOS', 1: 'EOS'}
        self.n_words = 2  # count SOS and EOS

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


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


def read_langs(lang1, lang2, reverse=False):
    print('Reading Lines...')

    # 读取文件，按行划分 lang1=eng,lang2=fra
    lines = open('data/%s-%s.txt' % (lang1, lang2), 'r', encoding='utf-8').read().strip().split('\n')

    # 划分每一行并且normalize，生成的是二位list
    pairs = [[normalize_string(s) for s in line.split('\t')] for line in lines]  # 二位列表

    if reverse:
        # 调换顺序
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)  # fra
        output_lang = Lang(lang1)  # eng
    else:
        input_lang = Lang(lang1)  # fra
        output_lang = Lang(lang2)  # eng
    return input_lang, output_lang, pairs


eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filter_pair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH and p[1].startswith(eng_prefixes)


def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]


def prepare_data(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = read_langs(lang1, lang2, reverse)
    print('Read %s sentence pairs' % len(pairs))
    pairs = filter_pairs(pairs)
    print(pairs[:4])
    print('Trimmed to %s sentence pairs' % len(pairs))
    print('Counting words...')
    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])
    print('Counted words:')
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


input_lang_dictionary, output_lang_dictionary, sentence_pairs = prepare_data('eng', 'fra', True)
print(random.choice(sentence_pairs))


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        # output = self.embedding(input).view(1,1,-1)
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class decoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(decoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class attn_decoder_RNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(attn_decoder_RNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
