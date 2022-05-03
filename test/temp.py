# -*- coding:utf-8 -*-
# @Time    : 2022/5/1 10:54
# @Author  : Yinkai Yang
# @FileName: temp.py
# @Software: PyCharm
# @Description: this is a program related to
# data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
# print(data)
# result = tuple(filter(lambda t: t > 4, data))
# print(result)
import string

# all_letters = string.ascii_letters + ",.;'-"
# line = 'bcdefg,.;-'
# letter_indexes = [all_letters.find(line[li]) for li in range(1,len(line))]
# print(letter_indexes)

import torch

print(torch.__version__)
print(torch.cuda.is_available())