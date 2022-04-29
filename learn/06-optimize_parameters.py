# -*- coding:utf-8 -*-
# @Time    : 2022/4/27 20:21
# @Author  : Yinkai Yang
# @FileName: test-06.py
# @Software: PyCharm
# @Description: this is a program related to
# 优化模型参数

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

# obtain data
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

# load data
train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)


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


# 在训练循环中，优化分三个步骤进行：
# 调用optimizer.zero_grad()以重置模型参数的梯度。默认情况下渐变加起来；为了防止重复计算，我们在每次迭代时明确地将它们归零。
# 通过调用来反向传播预测损失loss.backward()。PyTorch 存储每个参数的损失梯度。
# 一旦我们有了我们的梯度，我们调用optimizer.step()通过在反向传递中收集的梯度来调整参数。

# full implementation
# define train_loop and evalute the performance of the model
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        # backpropagetion
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# define some hyperparameters (iteration==迭代,一个iteration指的是用batch_size个样本训练一次)
learning_rate = 1e-3  # 学习率
batch_size = 64  # 参数更新前通过网络传播的数据样本数,每一批数据量的大小
epochs = 10  # 优化循环过程中迭代数据集的次数,1个epoch指的是训练集的全部样本训练一次

# create a model
model = NeutralNetwork()
print(model)

# initialize the loss function
loss_fn = nn.CrossEntropyLoss()

# initialize the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")
