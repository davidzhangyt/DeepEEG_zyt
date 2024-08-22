'''
这个部分是EEGNet的模型，同conformer类似，整合了模型定义与模型训练的全部流程。
'''

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

class Conv2dWithNorm(nn.Conv2d):
    def __init__(self, *args, do_weight_norm=True, max_norm=1., p=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p
        self.max_norm = max_norm
        self.do_weight_norm = do_weight_norm

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.do_weight_norm:
            self.weight.data = torch.renorm(
                self.weight.data, self.p, 0, self.max_norm
            )
        return super().forward(input)

    def __repr__(self):
        repr = super().__repr__()
        if self.do_weight_norm:
            last_bracket_index = repr.rfind(')')
            self_repr = f', max_norm={self.max_norm}, p={self.p}'
            repr = repr[:last_bracket_index] + self_repr + ')'
        return repr

class LinearWithNorm(nn.Linear):
    def __init__(self, *args, do_weight_norm=True, max_norm=1., p=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p
        self.max_norm = max_norm
        self.do_weight_norm = do_weight_norm

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.do_weight_norm:
            self.weight.data = torch.renorm(
                self.weight.data, self.p, 0, self.max_norm
            )
        return super().forward(input)

    def __repr__(self):
        repr = super().__repr__()
        if self.do_weight_norm:
            last_bracket_index = repr.rfind(')')
            self_repr = f', max_norm={self.max_norm}, p={self.p}'
            repr = repr[:last_bracket_index] + self_repr + ')'
        return repr

# 定义数据集加载类
class EEGDataset(torch.utils.data.Dataset):
    def __init__(self, data_files, label_files):
        self.data_files = data_files
        self.label_files = label_files
        self.current_file_index = 0
        self.data = None
        self.labels = None
        self.load_next_file()

    def load_next_file(self):
        if self.current_file_index < len(self.data_files):
            self.data = np.load(self.data_files[self.current_file_index])
            self.labels = np.load(self.label_files[self.current_file_index])
            self.current_file_index += 1
        else:
            self.current_file_index = 0
            self.load_next_file()
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.labels = torch.tensor(np.argmax(self.labels, axis=1), dtype=torch.long)  # 将 one-hot 编码的标签转换为类索引

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        data = data.unsqueeze(0)  # 在数据的第一个维度增加一个通道维度，使其形状变为 (1, nCh, nTime)
        return data, self.labels[idx]

class EEGNet(nn.Module):
    def __init__(self, nCh=17, nTime=10000, C1=63, F1=8, D=2, F2=16, C2=15, 
                 P1=8, P2=16, p=0.5, cls=3) -> None:
        super().__init__()

        self.filter = nn.Sequential(
            nn.Conv2d(1, F1, (1, C1), padding=(0, C1//2), bias=False),
            nn.BatchNorm2d(F1)
        )

        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(F1, D*F1, (nCh, 1), groups=F1, bias=False),
            nn.BatchNorm2d(D*F1),
            nn.ELU(),
            nn.AvgPool2d((1, P1)),
            nn.Dropout(p)
        )

        self.separable_conv = nn.Sequential(
            nn.Conv2d(D*F1, D*F1, (1, C2), padding=(0, C2//2), groups=D*F1, bias=False),
            nn.Conv2d(D*F1, F2, 1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, P2)),
            nn.Dropout(p)
        )

        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(self.get_size(nCh, nTime), cls, bias=True),
            nn.LogSoftmax(dim=1)
        )

    def get_size(self, nCh, nTime):
        x = torch.randn(1, 1, nCh, nTime)
        out = self.filter(x)
        # print(f'After filter: {out.shape}')
        out = self.depthwise_conv(out)
        # print(f'After depthwise_conv: {out.shape}')
        out = self.separable_conv(out)
        # print(f'After separable_conv: {out.shape}')
        out = self.flatten(out)
        # print(f'After flatten: {out.shape}')
        return out.size(1)

    def forward(self, x):
        out = self.filter(x)
        # print(f'After filter: {out.shape}')  # 打印形状
        out = self.depthwise_conv(out)
        # print(f'After depthwise_conv: {out.shape}')  # 打印形状
        out = self.separable_conv(out)
        # print(f'After separable_conv: {out.shape}')  # 打印形状
        out = self.flatten(out)
        # print(f'After flatten: {out.shape}')  # 打印形状
        return self.fc(out)

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型并移动到GPU
model = EEGNet(nCh=17, nTime=5000, cls=3).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 设置保存模型路径
model_save_dir = "D:\\EEG-DL-master\\dpeeg-main\\dpeeg-main\\model_save_10000"
os.makedirs(model_save_dir, exist_ok=True)

# 定义数据文件路径
train_data_files = [
    "D:\\EEG-DL-master\\dpeeg-main\\dpeeg-main\\final_datasets_5000\\train_data_1.npy",
    "D:\\EEG-DL-master\\dpeeg-main\\dpeeg-main\\final_datasets_5000\\train_data_2.npy",
    "D:\\EEG-DL-master\\dpeeg-main\\dpeeg-main\\final_datasets_5000\\train_data_3.npy"
]
train_label_files = [
    "D:\\EEG-DL-master\\dpeeg-main\\dpeeg-main\\final_datasets_5000\\train_labels_1.npy",
    "D:\\EEG-DL-master\\dpeeg-main\\dpeeg-main\\final_datasets_5000\\train_labels_2.npy",
    "D:\\EEG-DL-master\\dpeeg-main\\dpeeg-main\\final_datasets_5000\\train_labels_3.npy"
]

test_data_file = "D:\\EEG-DL-master\\dpeeg-main\\dpeeg-main\\final_datasets_5000\\test_data.npy"
test_label_file = "D:\\EEG-DL-master\\dpeeg-main\\dpeeg-main\\final_datasets_5000\\test_labels.npy"

# 加载数据
train_dataset = EEGDataset(train_data_files, train_label_files)
test_data = np.load(test_data_file)
test_labels = np.load(test_label_file)

# 将测试数据转换为TensorDataset并确保标签是类索引
test_data = torch.tensor(test_data, dtype=torch.float32)
test_labels = torch.tensor(np.argmax(test_labels, axis=1), dtype=torch.long)  # 将 one-hot 编码的标签转换为类索引
test_dataset = TensorDataset(test_data, test_labels)

# 划分训练集和验证集
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# 创建DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 训练模型
num_epochs = 10
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        _, predicted = torch.max(outputs, 1)
        # print('predicted_train:',predicted)
        total += y_batch.size(0)
        # print('y_batch_train:',y_batch)
        correct += (predicted == y_batch).sum().item()

    train_losses.append(running_loss / len(train_loader))
    train_accuracy = 100 * correct / total
    train_accuracies.append(train_accuracy)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {train_accuracy:.2f}%')

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)

            # print('predicted_val:',predicted)
            total += y_batch.size(0)

            # print('y_batch_val:',y_batch)
            correct += (predicted == y_batch).sum().item()

    val_losses.append(val_loss / len(val_loader))
    val_accuracy = 100 * correct / total
    val_accuracies.append(val_accuracy)
    print(f'Validation Loss: {val_loss/len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.2f}%')

    # 保存模型
    torch.save(model.state_dict(), os.path.join(model_save_dir, f"model_epoch_{epoch+1}.pth"))

# 测试模型
model.eval()
test_loss = 0.0
correct = 0
total = 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        if X_batch.dim() == 3:
            X_batch = X_batch.unsqueeze(1)  # 确保数据形状正确
        
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        
        
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        test_loss += loss.item()
        
        _, predicted = torch.max(outputs, 1)
        
        # print('predicted_test:',predicted)
        total += y_batch.size(0)
        
        # print('y_batch_test:',y_batch)
        correct += (predicted == y_batch).sum().item()

test_accuracy = 100 * correct / total
print(f'Test Loss: {test_loss/len(test_loader):.4f}, Test Accuracy: {test_accuracy:.2f}%')

# 保存训练结果
np.save(os.path.join(model_save_dir, "train_losses.npy"), train_losses)
np.save(os.path.join(model_save_dir, "val_losses.npy"), val_losses)
np.save(os.path.join(model_save_dir, "test_accuracy.npy"), test_accuracy)
np.save(os.path.join(model_save_dir, "test_loss.npy"), test_loss)
