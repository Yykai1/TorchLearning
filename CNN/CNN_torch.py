# -*- coding:utf-8 -*-
# @Time    : 2022/4/28 17:28
# @Author  : Yinkai Yang
# @FileName: CNN_torch.py
# @Software: PyCharm
# @Description: this is a program related to
# 利用CNN实现MNIST手写数字识别的分类问题

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision import datasets
import torch.nn.functional as F
import numpy as np

# define some parameters
learning_rate = 1e-3
batch_size = 50
epochs = 10
keep_prob_rate = 0.7

# obtain data
train_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

# load data
train_dataloader = DataLoader(train_data, batch_size=batch_size)
test_dataloader = DataLoader(train_data, batch_size=batch_size)


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Sequential(
            # 定义一个2维卷积层
            nn.Conv2d(
                in_channels=1,  # 输入通道数：RGB通道数
                out_channels=32,  # 输出通道数
                kernel_size=3,  # 卷积核
                stride=1,  # 步长
                padding=1  # 填充
            ),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 池化层，对邻域内特征点取最大
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.out1 = nn.Linear(64 * 7 * 7, 1024, bias=True)  # 第一个全连接层
        self.dropout = nn.Dropout(keep_prob_rate)
        self.out2 = nn.Linear(1024, 10, bias=True)  # 第二个全连接层

    def forward(self, x):
        # x:[50, 1, 28, 28]
        x = self.conv1(x)  # [50, 32, 14, 14]
        x = self.conv2(x)  # [50, 64, 7, 7]
        x = x.view(batch_size, 64 * 7 * 7)
        out1 = self.out1(x)  # [50, 1024]
        out1 = F.relu(out1)
        out1 = self.dropout(out1)
        out2 = self.out2(out1)
        logits = F.softmax(out2, dim=0)
        return logits


def train(dataloader, cnn, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # enumerate()用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。
    for batch, (X, y) in enumerate(dataloader):
        pred = cnn(X)
        loss = loss_fn(pred, y)

        # backpropagetion
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            # print("=" * 10, batch, "=" * 5, "=" * 5, "test accuracy is ", test(cnn), "=" * 10)


def test(dataloader, cnn, loss_fn):
    size = len(dataloader.dataset)

    ave_test_loss, correct = 0., 0.
    # batch = len(dataloader)
    # for(X,y) in enumerate(dataloader):
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            pred = cnn(X)
            # loss_fn的返回值是一个tensor，仅数据的话需要获取item属性
            ave_test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # for X,y in dataloader:
        #     pred = cnn(X)
        #         # loss_fn的返回值是一个tensor，仅数据的话需要获取item属性
        #     ave_test_loss += loss_fn(pred, y).item()
        #     correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        #
    ave_test_loss /= batch
    ave_correct = correct / size
    print(f"Test Error: \n Accuracy: {(100 * ave_correct):>0.1f}%, Avg loss: {ave_test_loss:>8f} \n")


def main():
    cnn = CNNModel()
    print(nn)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(cnn.parameters(), lr=learning_rate)
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, cnn, loss_fn, optimizer)
        test(train_dataloader, cnn, loss_fn)
    print("Done!")


if __name__ == '__main__':
    main()
