import torch
import torch.nn as nn
from torchvision import models

# # 创建一个示例模型并保存
# def create_dummy_model():
#     model = models.resnet18(pretrained=False)
#     model.fc = nn.Linear(model.fc.in_features, 7)  # 7分类
#
#     # 随机初始化权重
#     for param in model.parameters():
#         if param.requires_grad:
#             torch.nn.init.xavier_uniform_(param.data)
#
#     # 保存模型
#     torch.save(model.state_dict(), 'best_model.pth')
#     print("已创建示例模型文件: best_model.pth")
#
# if __name__ == "__main__":
#     create_dummy_model()
