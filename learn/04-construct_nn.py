# -*- coding:utf-8 -*-
# @Time    : 2022/4/26 12:20
# @Author  : Yinkai Yang
# @FileName: test-04.py
# @Software: PyCharm
# @Description: this is a program related to
# construct nn
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")  # Using cuda device,显然使用GPU


class NeutralNetwork(nn.Module):
    def __init__(self):
        super(NeutralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeutralNetwork().to(device)
# print(model)

X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")

"""
对Class中的函数细节进行一个详细的解释理解
"""
input_image = torch.rand(3, 28, 28)
# print(input_image.size())
#
flatten = nn.Flatten()
flat_image = flatten(input_image)
# print(flat_image.size())
#
layer1 = nn.Linear(in_features=28 * 28, out_features=20)
# # 使用自己内部的权重和偏差对输入进行线性变换，改变特征值（增加或者是减少）
# hidden1 = layer1(flat_image)
# print(hidden1.size())
#
# # ReLU函数<0取为0，>0求一个logistic函数
# print(f"Before ReLU: {hidden1}\n\n")
# hidden1 = nn.ReLU()(hidden1)
# print(f"After ReLU: {hidden1}")

# nn.Sequential()是一个有序的模块容器
seq_modules = nn.Sequential(
    flatten,
    layer1,
    # nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3, 28, 28)
logits = seq_modules(input_image)
print(logits)

# nn.Softmax()  # 将logits的值缩放为[0,1]之间
# dim=1表明是第二个维度的sun为1
softmax2=nn.Softmax(dim=1)
pred_probab = softmax2(logits)
print(pred_probab)

# 模型的参数
print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} -----> Size: {param.size()} -----> Values : {param[:2]} \n")
