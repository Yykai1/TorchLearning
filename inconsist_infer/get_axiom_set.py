# -*- coding:utf-8 -*-
# @Time    : 2022/5/12 16:39
# @Author  : Yinkai Yang
# @FileName: get_axiom_set.py
# @Software: PyCharm
# @Description: this is a program related to get 公理集合的字典（下标到句子，句子到下标），公理集合K，各个最大一致子集的总集合mc_K
# 只需要调用给定文件路径就可以获得对应的字典、公理集合、一致子集的集合
import glob
import os


def get_files(path):
    """获得路径下各个文件的名字

    :param path: 文件路径
    :return: 路径下的文件名字
    """
    tmp = []
    for file in glob.glob(path):
        file_name = os.path.basename(file)
        tmp.append(file_name)
    return tmp


def get_axiom_set(basic, filename):
    """获得公理集合K和一致子集的总集合mc_K

    :param basic: 文件路径
    :param filename: 文件名字
    :return: 公理集合K，各个一致子集的集合
    """
    tmp = set()  # 保证公理集合内容不能重复
    mc_K = []  # 二维list，每一个子list中包含了一个文件夹的所有公理
    for i in filename:
        path = basic + i
        with open(path, 'r+', encoding='utf-8') as f:
            k = f.read().split("\n\n")
            for axiom in k:
                tmp.add(axiom)
            f.close()
        mc_K.append(k)
    return list(tmp), mc_K


def get_axioms_dictionary(basic):
    """

    :param basic: 文件路径
    :return: 公理集合的字典（下标到句子，句子到下标），公理集合K，各个最大一致子集的总集合mc_K
    """
    filename = get_files(basic + '*.txt')
    global K
    global mc_K
    K, mc_K = get_axiom_set(basic, filename)  # 获得公理集合K，一致子集的总集合mc_K
    index = 0
    sent_index = {}
    index_sent = {}
    for sentence in K:  # 创建字典
        sent_index[sentence] = index  # 每一个sentence的index
        index_sent[index] = sentence  # 每一个index的sentence
        index += 1  # 下标+1
    return sent_index, index_sent, K, mc_K

# if __name__ == '__main__':
#     main('AUTOMCS/')
