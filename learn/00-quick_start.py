# -*- coding:utf-8 -*-
# @Time    : 2022/4/26 9:19
# @Author  : Yinkai Yang
# @FileName: test-00.py
# @Software: PyCharm
# @Description: this is a program related to
import torch

print(torch.ones(10))
# 流程
# 1.导入数据集，划分成训练集和测试集
#     环境导入 各种import
#     # Download training data from open datasets.
#     training_data = datasets.FashionMNIST(
#         root="data",
#         train=True,
#         download=True,
#         transform=ToTensor(),
#     )
#
#     # Download test data from open datasets.
#     test_data = datasets.FashionMNIST(
#         root="data",
#         train=False,
#         download=True,
#         transform=ToTensor(),
#     )
#     设置训练批次
# 2.创建模型
#     # Get cpu or gpu device for training.
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"Using {device} device")
#
#     # Define model
#     class NeuralNetwork(nn.Module):
#         def __init__(self):
#             super(NeuralNetwork, self).__init__()
#             self.flatten = nn.Flatten()
#             self.linear_relu_stack = nn.Sequential(
#                 nn.Linear(28*28, 512),
#                 nn.ReLU(),
#                 nn.Linear(512, 512),
#                 nn.ReLU(),
#                 nn.Linear(512, 10)
#             )
#
#         def forward(self, x):
#             x = self.flatten(x)
#             logits = self.linear_relu_stack(x)
#             return logits
#
#     model = NeuralNetwork().to(device)
#     print(model)
# 3.优化模型参数
#     定义损失函数和优化器
#         loss_fn = nn.CrossEntropyLoss()
#         optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
#     设置训练过程
#         def train(dataloader, model, loss_fn, optimizer):
#             size = len(dataloader.dataset)
#             model.train()
#             for batch, (X, y) in enumerate(dataloader):
#                 X, y = X.to(device), y.to(device)
#
#                 # Compute prediction error
#                 pred = model(X)
#                 loss = loss_fn(pred, y)
#
#                 # Backpropagation
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#
#                 if batch % 100 == 0:
#                     loss, current = loss.item(), batch * len(X)
#                     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
#     设置测试流程
#         def test(dataloader, model, loss_fn):
#             size = len(dataloader.dataset)
#             num_batches = len(dataloader)
#             model.eval()
#             test_loss, correct = 0, 0
#             with torch.no_grad():
#                 for X, y in dataloader:
#                     X, y = X.to(device), y.to(device)
#                     pred = model(X)
#                     test_loss += loss_fn(pred, y).item()
#                     correct += (pred.argmax(1) == y).type(torch.float).sum().item()
#             test_loss /= num_batches
#             correct /= size
#             print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
#     结束
#         epochs = 5
#         for t in range(epochs):
#             print(f"Epoch {t+1}\n-------------------------------")
#             train(train_dataloader, model, loss_fn, optimizer)
#             test(test_dataloader, model, loss_fn)
#         print("Done!")
# 4.保存模型
#     torch.save(model.state_dict(), "model.pth")
#     print("Saved PyTorch Model State to model.pth")
# 5.加载模型
#     model = NeuralNetwork()
#     model.load_state_dict(torch.load("model.pth"))
#     测试
#         classes = [
#             "T-shirt/top",
#             "Trouser",
#             "Pullover",
#             "Dress",
#             "Coat",
#             "Sandal",
#             "Shirt",
#             "Sneaker",
#             "Bag",
#             "Ankle boot",
#         ]
#
#         model.eval()
#         x, y = test_data[0][0], test_data[0][1]
#         with torch.no_grad():
#             pred = model(x)
#             predicted, actual = classes[pred[0].argmax(0)], classes[y]
#             print(f'Predicted: "{predicted}", Actual: "{actual}"')
