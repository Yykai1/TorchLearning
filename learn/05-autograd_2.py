# -*- coding:utf-8 -*-
# @Time    : 2022/4/27 18:16
# @Author  : Yinkai Yang
# @FileName: test-05.py
# @Software: PyCharm
# @Description: this is a program related to
# 自动微分Torch.atuograd
# We can only obtain the grad properties for the leaf nodes of the computational graph,
# which have requires_grad property set to True. For all other nodes in our graph, gradients will not be available.
# We can only perform gradient calculations using backward once on a given graph, for performance reasons.
# If we need to do several backward calls on the same graph, we need to pass retain_graph=True to the backward call.
import torch

# x = torch.ones(5)  # input tensor
# y = torch.zeros(3)  # expected output
# w = torch.randn(5, 3, requires_grad=True)
# b = torch.randn(3, requires_grad=True)

# 可以求导追踪
# z = torch.matmul(x, w) + b
# loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
# print(f"Gradient function for z = {z.grad_fn}")
# print(f"Gradient function for loss = {loss.grad_fn}")
# print(loss)
# loss.backward()
# print(w.grad)
# print(b.grad)

# 阻止求导追踪
# There are reasons you might want to disable gradient tracking:
# To mark some parameters in your neural network as frozen parameters. This is a very common scenario for finetuning a pretrained network
# To speed up computations when you are only doing forward pass, because computations on tensors that do not track gradients would be more efficient.
# z = torch.matmul(x, w) + b
# print(z.requires_grad)  # True
#
# z_det = z.detach()
# print(z_det.requires_grad)  # False
#
# with torch.no_grad():
#     z = torch.matmul(x, w) + b
# print(z.requires_grad)  # False


# 张量求导和雅各比矩阵
inp = torch.eye(5, requires_grad=True)
out = (inp+1).pow(2)
out.backward(torch.ones_like(inp), retain_graph=True)
print(f"First call\n{inp.grad}")
# inp.grad累计了
out.backward(torch.ones_like(inp), retain_graph=True)
print(f"\nSecond call\n{inp.grad}")
out.backward(torch.ones_like(inp), retain_graph=True)
print(f"\nThird call\n{inp.grad}")

# 正确操作，先归零
inp.grad.zero_()
# 以前我们是在backward()没有参数的情况下调用函数。这本质上等同于调用 backward(torch.tensor(1.0))，这是在标量值函数的情况下计算梯度的有用方法，例如神经网络训练期间的损失。
out.backward(torch.ones_like(inp), retain_graph=True)
print(f"\nCall after zeroing gradients\n{inp.grad}")
