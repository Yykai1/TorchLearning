# -*- coding:utf-8 -*-
# @Time    : 2022/7/1 11:15
# @Author  : Yinkai Yang
# @FileName: tx.py
# @Software: PyCharm
# @Description: this is a program related to
import numpy as np
import torch

a = np.array([[[0,1,2,3,],[4,5,6,7]],[[0,1,2,3,],[4,5,6,7]],[[0,1,2,3,],[4,5,6,7]]])
b = torch.tensor(a)
print(b.size(-1))
print(b.size(0))
print(b.size(1))
print(b.size(2))