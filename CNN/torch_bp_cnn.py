# -*- coding:utf-8 -*-
# @Time    : 2022/5/16 19:48
# @Author  : 杨印凯
# @FileName: torch_bp_cnn.py
import torch  # 导入pytorch环境
from torch import nn  # 导入神经网络
from torch.utils.data import DataLoader  # 导入数据加载器
from torchvision import datasets  # 导入数据集
from torchvision.transforms import ToTensor  # Tensor格式转换
import torch.nn.functional as F  # 导入pytorch函数库
import matplotlib.pyplot as plt  # 导入作图环境
import matplotlib.ticker as ticker  # 用于坐标轴的设置
import numpy as np  # 导入数据格式numpy
import random  # 导入随机函数
from sklearn import metrics  # 导入sklearn工具的metrics进行准确率、召回率、F1值分析

learning_rate = 0.01  # 设置学习率
epochs = 10  # 设置训练次数
batch_size = 50  # 设置每一批次的大小
keep_prob_rate = 0.7  # 控制/调整训练神经网络时使用的丢失率，层之间的每个连接在训练时仅以0.7的概率使用

device = "cuda" if torch.cuda.is_available() else "cpu"  # 设置运行环境cuda或者是cpu
print(f"Using {device} device")  # 打印环境

# 下载训练集集
training_data = datasets.MNIST(
    root="data",  # 训练集数据存放路径
    train=True,  # 数据集名称：训练集
    download=True,  # 数据是否下载：True表明下载
    transform=ToTensor()  # 转换成Tensor格式
)

# 下载测试集
test_data = datasets.MNIST(
    root="data",  # 测试集数据存放路径
    train=False,  # 数据集名称：测试集
    download=True,  # 数据是否下载：True表明下载
    transform=ToTensor()  # 转换成Tensor格式
)

train_dataloader = DataLoader(training_data, batch_size=batch_size)  # 创建训练集数据加载器
test_dataloader = DataLoader(test_data, batch_size=batch_size)  # 创建测试集数据加载器


# 建立BP神经网络
class BPNNModel(nn.Module):
    def __init__(self):
        super(BPNNModel, self).__init__()  # 继承
        self.flatten = nn.Flatten()  # 数据展平
        self.linear_relu_stack = nn.Sequential(
            # 定义一个深度为3的神经网络
            nn.Linear(28 * 28, 512),  # 第一层
            nn.ReLU(),  # 激活函数
            nn.Linear(512, 512),  # 第二层隐藏层
            nn.ReLU(),  # 激活函数
            nn.Linear(512, 10)  # 输三层隐藏层
        )

    def forward(self, x):
        x = self.flatten(x)  # 数据展平
        logits = self.linear_relu_stack(x)  # 通过神经网络
        return logits  # 返回输出


# 创建卷积神经网络（CNN）
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()  # 继承
        self.flatten = nn.Flatten()  # 数据展平
        self.conv1 = nn.Sequential(
            # 定义一个2维卷积层
            nn.Conv2d(
                in_channels=1,  # 输入通道数：RGB通道数
                out_channels=8,  # 输出通道数
                kernel_size=3,  # 卷积核
                stride=1,  # 步长
                padding=1  # 填充为1，保证数据形状
            ),
            nn.ReLU(),  # 激活函数
            nn.MaxPool2d(2)  # 池化层，对邻域内特征点取最大
        )
        self.conv2 = nn.Sequential(
            # 再定义一个2维卷积层
            nn.Conv2d(
                in_channels=8,  # 输入通道数
                out_channels=16,  # 输出通道数
                kernel_size=3,  # 卷积核
                stride=1,  # 步长
                padding=1  # 填充为1，保证数据形状
            ),
            nn.ReLU(),  # 激活函数
            nn.MaxPool2d(2)  # 池化层，对邻域内特征点取最大
        )
        self.out1 = nn.Linear(16 * 7 * 7, 1024, bias=True)  # 第一个全连接层
        self.dropout = nn.Dropout(keep_prob_rate)  # 防止过拟合
        self.out2 = nn.Linear(1024, 10, bias=True)  # 第二个全连接层

    def forward(self, x):
        """前向通过整个卷积神经网络

        :param x: 输入
        :return: 输出结果
        """
        # 输入x数据格式[50, 1, 28, 28]
        x = self.conv1(x)  # 通过第一个卷积层，x格式[50, 8, 14, 14]
        x = self.conv2(x)  # 通过第二个卷积层，x格式[50, 16, 7, 7]
        x = self.flatten(x)  # 数据展平，x数据格式[50， 784]
        out1 = self.out1(x)  # 通过第一个全连接层，out1数据格式[50, 1024]
        out1 = F.relu(out1)  # out1通过激活函数
        out1 = self.dropout(out1)  # 防止过拟合
        out2 = self.out2(out1)  # 通过第二个全连接层
        logits = F.softmax(out2, dim=1)  # 对第二个维度数据进行softmax激活
        return logits  # 返回输出


def train(dataloader, model, loss_fn, optimizer, los):  # 网络训练
    """训练

    :param dataloader: 数据集
    :param model: 模型
    :param loss_fn: 损失函数
    :param optimizer: 优化器
    :param los: 记录损失
    :return: 无
    """
    size = len(dataloader.dataset)  # 获得数据集大小
    model.train()  # 设置模型为训练模式
    for batch, (X, y) in enumerate(dataloader):  # 批次训练
        X, y = X.to(device), y.to(device)  # 将数据放进device（cuda或者史cpu）

        pred = model(X)  # 训练
        loss = loss_fn(pred, y)  # 训练损失

        # 后向步进
        optimizer.zero_grad()  # 优化器梯度初始化为0
        loss.backward()  # 损失反向传播，根据loss来计算网络参数的梯度
        optimizer.step()  # 模型参数更新

        if batch % 10 == 0:  # loss, current打印控制
            loss, current = loss.item(), batch * len(X)  # 获得当前损失和批次
            los.append(loss)  # 记录损失，放进los中
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")  # 打印


def test(dataloader, model, loss_fn):  # 网络测试
    """测试

    :param dataloader: 数据集
    :param model: 模型
    :param loss_fn: 损失函数
    :return: 无
    """
    size = len(dataloader.dataset)  # 获得数据集大小
    num_batches = len(dataloader)  # 训练批次
    model.eval()  # 设置模型为测试模式
    test_loss, correct = 0, 0  # 记录测试损失和正确个数
    pred_cnn = []  # 记录预测结果，列表形式存储
    y_cnn = []  # 记录测试数据，列表形式存储
    with torch.no_grad():  # 不跟踪梯度
        for X, y in dataloader:  # 批次训练
            X, y = X.to(device), y.to(device)  # 将数据放进device（cuda或者史cpu）
            pred = model(X)  # 进行预测
            test_loss += loss_fn(pred, y).item()  # 测试损失
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()  # 记录测试正确个数
            y_cnn.append(y.cpu().numpy().tolist())  # 测试数据格式转换成二维列表
            pred_cnn.append(pred.cpu().argmax(1).numpy().tolist())  # 预测数据格式转换成二维列表

    test_loss /= num_batches  # 计算平均损失
    correct /= size  # 计算平均正确个数
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")  # 打印数据
    return y_cnn, pred_cnn  # 返回训练集数据和预测结果


def display_data(y_test, pred, s):
    """展示Confusion矩阵

    :param clf: 训练后的模型
    :param y_test: 测试集
    :param pred: 预测结果
    :param s: 模型的类型
    :return: 无
    """
    plt.figure(2)  # 画第二张图，避免所有图画在一个窗口（figure）中
    disp1 = metrics.ConfusionMatrixDisplay.from_predictions(y_test, pred)  # 展示混淆矩阵
    disp1.figure_.suptitle(f"{s} Confusion Matrix")  # 设置矩阵名称
    print(
        f"{metrics.classification_report(y_test, pred)}\n"  # 控制台展示
    )


def evaluate_pred(model, s):
    """对模型进行简单的测试，控制台打印预测结果和实际结果，进行一个简单的比较

    :param model: 模型
    :param s: 模型名称
    :return: 无
    """
    print('----------------------------------------------')
    print(f"{s}Prediction Start!")  # 完成
    for i in range(10):  # 10个简单的测试
        tmp = random.randint(0, 50)  # 生成随机数
        x, y = test_data[tmp][0], test_data[tmp][1]  # 获得数据以及类别
        x = x.unsqueeze(0)  # 增加维度[1 * 1 * 28 * 28]
        with torch.no_grad():  # 不对梯度进行跟踪
            pred = model(x)  # 预测
            predicted, actual = pred[0].argmax(0), y  # 获得预测结果和实际结果
            correct = '✓' if predicted == actual else '✗ (%s)' % actual  # 进行判断，正确✓，错误✗
            print(f'Predicted: "{predicted}", Actual: "{actual}", Judge:"{correct}"')  # 控制台打印
    print(f"{s}Prediction Done!")  # 完成
    print('----------------------------------------------')


def line_compare(a, b):
    """做折线图进行比较

    :param a: cnn各类别分类准确性数据
    :param b: bpnn各类别分类准确性数据
    :return: 无
    """
    plt.figure(1)  # 画第一张图，避免所有图画在一个窗口（figure）中
    x = np.array(range(0, 10, 1))  # 自变量数据范围
    plt.xticks(x, range(0, 10))  # 设置x轴显示数据
    plt.xlim((-0.5, 9.5))  # 规定x轴范围
    plt.ylim((0, 1.25))  # 规定y轴范围
    plt.xlabel('number class')  # 设置x轴名称
    plt.ylabel('accuracy')  # 设置y轴名称
    plt.plot(x, a, label='cnn', color='r')  # 做cnn准确性折线图，颜色为红
    plt.plot(x, b, label='bpnn', color='b')  # 做bpnn准确性折线图，颜色为蓝
    plt.title('Line Chart')  # 设置图片名称
    plt.legend()  # 为图像加上图例


def histogram_compare(a, b):
    """做柱状图（条形图）进行比较

    :param a: cnn各类别分类准确性数据
    :param b: bpnn各类别分类准确性数据
    :return: 无
    """
    plt.figure(2)  # 画第二张图，避免所有图画在一个窗口（figure）中
    x = np.array(range(0, 10, 1))  # 自变量数据范围
    total_width, n = 0.4, 2  # 柱的宽度，每一个柱的块（bar）数
    width = total_width / n  # 块（bar）的宽度
    x = x - width  # 确定自变量起始位置
    plt.xticks(x, range(0, 10))  # 设置x轴显示数据
    plt.ylim((0.6, 1.1))  # 规定y轴范围
    plt.bar(x, a, width=width, label='cnn', color='r')  # 做cnn准确性条形图，颜色为红
    plt.bar(x + width, b, width=width, label='bpnn', color='b')  # 做bpnn准确性条形图，颜色为蓝
    plt.xlabel('number class')  # 设置x轴名称
    plt.ylabel('accuracy')  # 设置y轴名称
    plt.title('Histogram Chart')  # 设置图片名称
    plt.legend()  # 为图像加上图例


def show_plot(los1, los2):
    """做损失去曲线

    :param los1: cnn的损失
    :param los2: bpnn的损失
    :return: 无
    """
    fig, ax = plt.subplots()  # 作图
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.xlabel('iteration number')  # 设置x轴名称
    plt.ylabel('loss')  # 设置y轴名称
    plt.title('Loss Chart')  # 设置图片名称
    plt.plot(los1, label='cnn_loss', color='r')  # 做cnn损失曲线，颜色为红
    plt.plot(los2, label='bpnn_loss', color='b')  # 做bpnn损失曲线，颜色为蓝
    plt.legend()  # 为图像加上图例


def main():
    cnn = CNNModel().to(device)  # 创建cnn，放进device中
    bpnn = BPNNModel().to(device)  # 创建bpnn，放进device中
    loss_fn1 = nn.CrossEntropyLoss()  # 创建交叉熵损失函数
    loss_fn2 = nn.CrossEntropyLoss()  # 创建交叉熵损失函数
    optimizer1 = torch.optim.SGD(cnn.parameters(), lr=learning_rate)  # 创建优化器
    # optimizer1 = torch.optim.SGD(cnn.parameters(), lr=learning_rate)
    optimizer2 = torch.optim.SGD(bpnn.parameters(), lr=learning_rate)  # 创建优化器
    # optimizer2 = torch.optim.SGD(bpnn.parameters(), lr=learning_rate)

    loss1 = []  # 记录损失
    for t in range(epochs):  # 循环epochs次进行训练
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, cnn, loss_fn1, optimizer1, loss1)  # 训练
        y_a, pred_a = test(test_dataloader, cnn, loss_fn1)  # 测试，接受返回值，此时返回数据二维的，不能直接使用
    torch.save(cnn.state_dict(), "cnn_model.pth")  # 保存模型
    print("Saved PyTorch Model State to cnn_model.pth")  # 模型保存成功

    loss2 = []  # 记录损失
    for t in range(epochs):  # 循环epochs次进行训练
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, bpnn, loss_fn2, optimizer2, loss2)  # 训练
        y_b, pred_b = test(test_dataloader, bpnn, loss_fn2)  # 测试，接受返回值，此时返回数据二维的，不能直接使用
    torch.save(bpnn.state_dict(), "bpnn_model.pth")  # 保存模型
    print("Saved PyTorch Model State to bpnn_model.pth")  # 模型保存成功

    y_a, y_b = np.array(sum(y_a, [])), np.array(sum(y_b, []))  # 数据处理，先将二维转成一维，让转成numpy格式
    pred_a, pred_b = np.array(sum(pred_a, [])), np.array(sum(pred_b, []))  # 数据处理，先将二维转成一维，让转成numpy格式
    a = metrics.precision_recall_fscore_support(y_a, pred_a)[0]  # 获得cnn的准确率
    b = metrics.precision_recall_fscore_support(y_b, pred_b)[0]  # 获得bpnn的准确率
    display_data(y_a, pred_a, 'cnn')  # 展示cnn的混淆矩阵
    display_data(y_b, pred_b, 'bpnn')  # 展示bpnn的混淆矩阵
    line_compare(a, b)  # 折线图性能比较
    histogram_compare(a, b)  # 柱状图性能比较
    show_plot(loss1, loss2)  # 损失曲线
    plt.show()  # 做出图片


if __name__ == '__main__':
    main()  # 启动
