# -*- coding:utf-8 -*-
# @Time    : 2022/4/28 8:45
# @Author  : Yinkai Yang
# @FileName: 07-save_load_model.py
# @Software: PyCharm
# @Description: this is a program related to
import torch
import torchvision.models as models

# save model weights
model = models.vgg16(pretrained=True)
torch.save(model.state_dict(), 'model_weights.pth')

# load model weights
model1 = models.vgg16()  # we do not specify pretrained=True, i.e. do not load default weights
model1.load_state_dict(torch.load('model_weights.pth'))
model1.eval()

# Saving and Loading Models with Shapes
torch.save(model, 'model.pth')
model2 = torch.load('model.pth')
model2.eval()
