# -*- coding:utf-8 -*-
# @Time    : 2022/5/18 11:24
# @Author  : 杨印凯
# @FileName: evaluate_result.py
import matplotlib.pyplot as plt
import torch  # 导入pytorch
from torch_bp_cnn import CNNModel, BPNNModel, evaluate_pred  # 从torch_bp_cnn中导入模型和测试函数


def main():
    model1 = CNNModel()  # 创建cnn
    model2 = BPNNModel()  # 创建bpnn
    model1.load_state_dict(torch.load("cnn_model.pth"))  # 加载cnn
    model2.load_state_dict(torch.load("bpnn_model.pth"))  # 加载bpnn
    evaluate_pred(model1, 'cnn')  # 评估cnn
    evaluate_pred(model2, 'bpnn')  # 评估bpnn


if __name__ == '__main__':
    main()  # 启动函数
