import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import resnet18
from PIL import Image
import matplotlib.pyplot as plt
import os
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sns  # 新增：用于混淆矩阵可视化

# 1. 优化数据增强（扩展对称处理类型）
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # 增强对称变换：增加垂直翻转和180度旋转（覆盖更多对称场景）
    transforms.RandomHorizontalFlip(p=0.5),  # 水平对称
    transforms.RandomVerticalFlip(p=0.3),    # 新增：垂直对称（概率30%）
    transforms.RandomRotation(180),          # 新增：0-180度旋转（覆盖旋转对称）
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 2. 加载数据集（保持不变）
dataset = datasets.ImageFolder('data', transform=train_transform)
classes = dataset.classes  # ['0', '1', ..., '6']

# 拆分：80% train, 10% val, 10% test
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

# 为val/test应用val_transform
val_ds.dataset.transform = val_transform
test_ds.dataset.transform = val_transform

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

print(f"数据集大小: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")
print(f"类别: {classes}")

# 3. 模型：优化微调层数（增强对对称特征的学习能力）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(classes))  # 动态适配类别数
model = model.to(device)

# 优化：解冻更多层（让模型学习对对称更鲁棒的特征）
for param in model.parameters():
    param.requires_grad = False  # 先冻结所有层
# 解冻layer2、layer3、layer4和全连接层（比原来多解冻2层）
for param in model.layer2.parameters():
    param.requires_grad = True
for param in model.layer3.parameters():
    param.requires_grad = True
for param in model.layer4.parameters():
    param.requires_grad = True
for param in model.fc.parameters():
    param.requires_grad = True

# 4. 损失/优化器/调度器（保持不变）
criterion = nn.CrossEntropyLoss()
# 只优化解冻的参数
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)  # 每10轮学习率衰减10倍


# 5. 训练函数（保持不变）
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return running_loss / len(loader), correct / total


# 验证函数（扩展：返回预测结果和真实标签，用于后续评估）
def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            # 收集预测结果和标签
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return running_loss / len(loader), correct / total, all_preds, all_labels


# 6. 训练循环（保持早停机制）
best_acc = 0.0
patience, counter = 10, 0
epochs = 50

for epoch in range(epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
    scheduler.step()

    print(
        f'Epoch {epoch + 1}/{epochs}: '
        f'Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | '
        f'Val Loss={val_loss:.4f}, Acc={val_acc:.4f}'
    )

    # 保存最佳模型
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')
        counter = 0
    else:
        counter += 1

    if counter >= patience:
        print(f"早停！最佳Val Acc: {best_acc:.4f}")
        break

# 7. 测试最佳模型（增强评估：添加分类报告和混淆矩阵）
model.load_state_dict(torch.load('best_model.pth'))
test_loss, test_acc, test_preds, test_labels = validate(model, test_loader, criterion, device)
print(f"\n=== 最终测试准确率: {test_acc:.4f} ({test_acc * 100:.2f}%) ===")

# 输出分类报告（详细展示每个类别的表现）
print("\n=== 分类报告 ===")
print(classification_report(
    test_labels, test_preds,
    target_names=classes,
    digits=4
))

# 绘制混淆矩阵（直观查看错误分类，尤其是对称样本）
cm = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=classes,
    yticklabels=classes
)
plt.xlabel('预测类别')
plt.ylabel('真实类别')
plt.title('测试集混淆矩阵')
plt.savefig('confusion_matrix.png')  # 保存混淆矩阵图片
plt.close()

print("混淆矩阵已保存为 'confusion_matrix.png'")