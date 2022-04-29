# -*- coding:utf-8 -*-
# @Time    : 2022/4/25 22:03
# @Author  : Yinkai Yang
# @FileName: test-02.py
# @Software: PyCharm
# @Description: this is a program related to
# 所有神经网络的核心是autograd包

import torch

# 用户手动创建,没有grad_fn属性
# x = torch.ones(2, 2, requires_grad=True)
x = torch.tensor([[1, 2.], [3, 4]], requires_grad=True)
print(x)
y = x + 2
print(y)  # 有grad_fn属性
# print(y.grad_fn)

z = y * y * 3
print(z)
# out = z.mean()  # 求平均了
# print(out)

# 第一种求导方式
# out.backward()  # 意思是只含有一个元素的张量,即1*1
# print(x.grad)  # 对x里面的元素分别求偏导

# 第二种求导方式
# a = torch.sum(z)
# grads = torch.autograd.grad(outputs=a, inputs=x)
# print(grads)

# 第三种求导方式
b = torch.tensor([[1., 1.], [1., 1.]], dtype=torch.float)
z.backward(b)
print(x.grad)

# out = z.mean()
# print(z, out)  # 所有的均值
# out = z.mean(0)  # 第一个维度的均值
# print(z, out)
# out = z.mean(1)  # 第二个维度的均值
# print(z, out)

# a = torch.randn(2, 2)
# a = ((a * 3) / (a - 1))
# print(a.requires_grad)
# a.requires_grad_(True)
# print(a.requires_grad)
# b = (a * a).sum()
# print(a.grad_fn)
# print(b.grad_fn)
