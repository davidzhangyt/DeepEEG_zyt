'''
这个部分是把PSD和原始信号整合进行训练的代码，使用的是EEGNet，然后参考的论文是A two-stage transformer based network for motor imagery classification
这篇论文没有源代码，我参考着自己写的，能训练能跑，但是准确率只有50%左右，所以这个代码仅供参考。
'''

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 设置数据路径
data_path = "D:/EEG-DL-master/dpeeg-main/dpeeg-main/data17"
folders = {
    "Cr37": "Cr-zzh-026",
    "SSD63": "SSD-zyx-037",
    "HL96": "HLygh-021"
}
model_save_path = "D:/EEG-DL-master/dpeeg-main/dpeeg-main/models"
os.makedirs(model_save_path, exist_ok=True)  # 确保模型保存路径存在

# 加载数据函数
def load_data(file_indices, folder, file_prefix):
    data_list, psd_list, label_list = [], [], []
    
    for idx in file_indices:
        data_file = os.path.join(data_path, folder, 'noise', f'{file_prefix}.cdt8C2_ICA_part{idx}_data.npy')
        psd_file = os.path.join(data_path, folder, 'noise', f'{file_prefix}.cdt8C2_ICA_part{idx}_psd.npy')
        label_file = os.path.join(data_path, folder, 'noise', f'{file_prefix}.cdt8C2_ICA_part{idx}_labels.npy')
        
        # 检查文件是否存在，如果存在则加载
        if os.path.exists(data_file) and os.path.exists(psd_file) and os.path.exists(label_file):
            data_list.append(np.load(data_file))
            psd_list.append(np.load(psd_file))
            label_list.append(np.load(label_file))
        else:
            print(f"File not found: {data_file}, {psd_file}, or {label_file}")
    
    if not data_list or not psd_list or not label_list:
        raise FileNotFoundError(f"Files for {file_prefix} in {folder} not found or incomplete.")
    
    data = np.concatenate(data_list, axis=0)
    psd = np.concatenate(psd_list, axis=0)
    labels = np.concatenate(label_list, axis=0)
    
    return data, psd, labels

# 选择数据集文件索引
train_indices = range(1, 9)  # 前8个文件
test_indices = range(9, 11)  # 后2个文件

# 加载所有数据
all_data_train, all_psd_train, all_labels_train = [], [], []
all_data_test, all_psd_test, all_labels_test = [], [], []

for folder, file_prefix in folders.items():
    try:
        data_train, psd_train, labels_train = load_data(train_indices, folder, file_prefix)
        data_test, psd_test, labels_test = load_data(test_indices, folder, file_prefix)
        
        all_data_train.append(data_train)
        all_psd_train.append(psd_train)
        all_labels_train.append(labels_train)
        
        all_data_test.append(data_test)
        all_psd_test.append(psd_test)
        all_labels_test.append(labels_test)
    except FileNotFoundError as e:
        print(e)
        continue

# 合并训练和测试集
X_train_data = np.concatenate(all_data_train, axis=0)
X_train_psd = np.concatenate(all_psd_train, axis=0)
y_train = np.concatenate(all_labels_train, axis=0)

X_test_data = np.concatenate(all_data_test, axis=0)
X_test_psd = np.concatenate(all_psd_test, axis=0)
y_test = np.concatenate(all_labels_test, axis=0)

# 转换为Tensor
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_train_data = torch.tensor(X_train_data, dtype=torch.float32).unsqueeze(1).to(device)
X_test_data = torch.tensor(X_test_data, dtype=torch.float32).unsqueeze(1).to(device)
X_train_psd = torch.tensor(X_train_psd, dtype=torch.float32).unsqueeze(1).to(device)
X_test_psd = torch.tensor(X_test_psd, dtype=torch.float32).unsqueeze(1).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)
y_test = torch.tensor(y_test, dtype=torch.long).to(device)

# 创建DataLoader
train_dataset = TensorDataset(X_train_data, X_train_psd, y_train)
test_dataset = TensorDataset(X_test_data, X_test_psd, y_test)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# 构建深度学习模型
class EEGNet(nn.Module):
    def __init__(self, input_shape_data, input_shape_psd):
        super(EEGNet, self).__init__()
        
        # 原始信号通道
        self.conv1_data = nn.Conv2d(1, 32, kernel_size=(3, 3))
        self.pool1_data = nn.MaxPool2d(kernel_size=(2, 2))
        self.flatten_data = nn.Flatten()

        # PSD特征通道
        self.conv1_psd = nn.Conv2d(1, 32, kernel_size=(3, 3))
        self.pool1_psd = nn.MaxPool2d(kernel_size=(2, 2))
        self.flatten_psd = nn.Flatten()

        # 融合层
        data_flatten_size = self._get_flatten_size(input_shape_data, 'data')
        psd_flatten_size = self._get_flatten_size(input_shape_psd, 'psd')
        self.fc1 = nn.Linear(data_flatten_size + psd_flatten_size, 64)
        self.fc2 = nn.Linear(64, 3)

    def _get_flatten_size(self, input_shape, layer_type):
        dummy_input = torch.zeros(1, 1, *input_shape)
        if layer_type == 'data':
            x = self.pool1_data(self.conv1_data(dummy_input))
        elif layer_type == 'psd':
            x = self.pool1_psd(self.conv1_psd(dummy_input))
        return int(np.prod(x.size()))

    def forward(self, x_data, x_psd):
        x_data = self.pool1_data(torch.relu(self.conv1_data(x_data)))
        x_data = self.flatten_data(x_data)

        x_psd = self.pool1_psd(torch.relu(self.conv1_psd(x_psd)))
        x_psd = self.flatten_psd(x_psd)

        x = torch.cat((x_data, x_psd), dim=1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 训练模型
model = EEGNet(input_shape_data=X_train_data.shape[2:], input_shape_psd=X_train_psd.shape[2:]).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

best_accuracy = 0.0

for epoch in range(50):
    model.train()
    running_loss = 0.0
    for X_batch_data, X_batch_psd, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch_data, X_batch_psd)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X_batch_data.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    
    # 评估模型在测试集上的性能
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch_data, X_batch_psd, y_batch in test_loader:
            outputs = model(X_batch_data, X_batch_psd)
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    
    accuracy = 100 * correct / total
    
    # 打印每个epoch的损失和准确率
    print(f"Epoch {epoch+1}/{50}, Loss: {epoch_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
    
    # 如果当前模型表现最好，则保存模型
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), os.path.join(model_save_path, 'best_model.pth'))
        print(f"Model saved with accuracy: {best_accuracy:.2f}%")

# 保存最终模型
torch.save(model.state_dict(), os.path.join(model_save_path, 'final_model.pth'))
print("Final model saved.")
