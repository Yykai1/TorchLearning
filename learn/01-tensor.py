# -*- coding:utf-8 -*-
# @Time    : 2022/4/25 18:24
# @Author  : Yinkai Yang
# @FileName: test-01.py
# @Software: PyCharm
# @Description: this is a program related to
import torch
import numpy as np

# 测试pytorch是否安装成功
# print(torch.__version__)  # 1.11.0
# print(torch.cuda.is_available())  # True


# x = torch.empty(5, 3)  # 不一定都是0.
# print(x)
#
# y = torch.rand(5, 3)
# print(y)
#
# a = torch.zeros(5, 3, dtype=torch.long)
# print(a)
#
# # 直接从数据构造张量
# b = torch.tensor([1, 2, 3, 4, 5])
# c = torch.tensor([5.0, 1])
# d = torch.tensor(1.0)
# e = torch.tensor([1.0])
# print(b)
# print(c)
# print(d)
# print(e)
# print(d == e)  # True
# print(x.size())

# 运算
# x = torch.rand(4, 4)
# y = torch.rand(4, 4)
# print(x)
# print(y)
# 加法的两种形式
# print(x + y)
# print(torch.add(x, y))
# 给定一个输出张量
# result = torch.empty(4, 4)
# torch.add(x, y, out=result)
# print(result)
# 原位加
# print(x.add_(y))

# 注意：任何一个in-place改变张量的操作后面都固定一个_。例如x.copy_(y)、x.t_()将更改x

# 索引操作
# print(x[:, 0])  # 第一列
# print(x[:, 1])  # 第二列
# print(x[0, :])  # 第一行

# 改变形状
# y = x.view(16)
# # print(y)
# z = x.view(-1, 8)
# # print(z)
# print(x.size(),y.size(),z.size())
# a=torch.tensor(1)
# b=torch.tensor([2.0])
# # 注意:item只能得到仅包含一个元素的tensor
# print(a.item())
# print(b.item())

# torch将tensor转换成numpy
# x = torch.tensor([1, 2, 3, 4, 5, 6.])
# y = x.numpy()
# print(x)
# print(y)
# x.add_(1)
# print(x)
# print(y)

# numpy数组转torch张量
# a = np.ones(5)
# b = torch.from_numpy(a)
# print(a)
# print(b)
# np.add(a, 1, out=a)
# #上面我只改变了a,但是怎么说呢,a的变化对b也造成影响了
# print(a)
# print(b)

# 张量可以使用.to方法移动到任何设备(device）上：
# 当GPU可用时,我们可以运行以下代码
# 我们将使用`torch.device`来将tensor移入和移出GPU
# x=torch.tensor(np.pi)
# if torch.cuda.is_available():
#     device = torch.device("cuda")          # a CUDA device object
#     y = torch.ones_like(x, device=device)  # 直接在GPU上创建tensor
#     x = x.to(device)                       # 或者使用`.to("cuda")`方法
#     z = x + y
#     print(z)
#     print(z.to("cpu", torch.double))       # `.to`也能在移动时改变dtype

# 矩阵的乘法
x = torch.tensor([[1., 2], [3, 4]])
y = torch.tensor([[1., 2], [3, 4]])
# 同一位置相乘
a = torch.mul(x, y)
print(a)
# 矩阵乘法
b = torch.matmul(x, y)
print(b)
