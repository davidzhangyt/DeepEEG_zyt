'''
这个部分是对论文conformer进行的复现，实测能到99%的训练准确率，但是测试集结果很差，只有40%。
论文原文：Conformer: Local Features Coupling Global Representations for Visual Recognition
github链接：https://github.com/pengzhiliang/Conformer
我的处理方法：把原始数据切分成了17个电极，500HZ下的2500个采样点，对应数据采用三分类，独热编码，训练集和测试集比例为8:2，点GPU即可运行，
如果想调整频域或者其他特征输入的话，可以直接把我这些代码复制给chatgpt，他会帮你修改。
'''
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from einops.layers.torch import Rearrange
from einops import rearrange
import time

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        self.data = torch.tensor(self.data, dtype=torch.float32).unsqueeze(1)  # 添加通道维度
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# 定义神经网络模型
class PatchEmbedding(nn.Module):
    def __init__(self, nCh, emb_size=40):
        super().__init__()

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (nCh, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.shallownet(x)
        x = self.projection(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=10,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size, num_heads, drop_p, forward_expansion, forward_drop_p):
        super().__init__(*[
            TransformerEncoderBlock(emb_size, num_heads, drop_p, forward_expansion, forward_drop_p)
            for _ in range(depth)
        ])

class ClassificationHead(nn.Sequential):
    def __init__(self, in_features, cls):
        super().__init__(
            nn.Flatten(),
            nn.Linear(in_features, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, cls),
            nn.LogSoftmax(dim=1)
        )

class EEGConformer(nn.Module):
    def __init__(
        self, 
        nCh, 
        nTime, 
        cls, 
        emb_size=40, 
        depth=6, 
        num_heads=10, 
        drop_p=0.5, 
        forward_expansion=4, 
        forward_drop_p=0.5
    ) -> None:
        super().__init__()
        self.nCh = nCh
        self.nTime = nTime
        self.patch_embedding = PatchEmbedding(nCh, emb_size)
        self.transformer_encoder = TransformerEncoder(
            depth, emb_size, num_heads, drop_p, forward_expansion, forward_drop_p
        )
        in_features = self._forward_transformer(torch.randn(1, 1, nCh, nTime)).size(1)
        self.classification_head = ClassificationHead(in_features, cls)

    def _forward_transformer(self, x) -> torch.Tensor:
        x = self.patch_embedding(x)
        x = self.transformer_encoder(x)
        return x.flatten(start_dim=1, end_dim=-1)

    def forward(self, x):
        out = self.patch_embedding(x)
        out = self.transformer_encoder(out)
        return self.classification_head(out)

# 初始化模型并移动到GPU
model = EEGConformer(17, 5000, 3).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 设置保存模型路径
model_save_dir = "D:\\EEG-DL-master\\dpeeg-main\\dpeeg-main\\model_save_5000"
os.makedirs(model_save_dir, exist_ok=True)

# 定义数据文件路径
train_data_files = [
    "D:\\EEG-DL-master\\dpeeg-main\\dpeeg-main\\final_datasets_5000\\train_data_1.npy",
    "D:\\EEG-DL-master\\dpeeg-main\\dpeeg-main\\final_datasets_5000\\train_data_2.npy",
    "D\\EEG-DL-master\\dpeeg-main\\dpeeg-main\\final_datasets_5000\\train_data_3.npy"
]
train_label_files = [
    "D:\\EEG-DL-master\\dpeeg-main\\dpeeg-main\\final_datasets_5000\\train_labels_1.npy",
    "D:\\EEG-DL-master\\dpeeg-main\\dpeeg-main\\final_datasets_5000\\train_labels_2.npy",
    "D\\EEG-DL-master\\dpeeg-main\\dpeeg-main\\final_datasets_5000\\train_labels_3.npy"
]

test_data_file = "D:\\EEG-DL-master\\dpeeg-main\\dpeeg-main\\final_datasets_5000\\test_data.npy"
test_label_file = "D:\\EEG-DL-master\\dpeeg-main\\dpeeg-main\\final_datasets_5000\\test_labels.npy"

# 加载数据
train_dataset = EEGDataset(train_data_files, train_label_files)
test_data = np.load(test_data_file)
test_labels = np.load(test_label_file)

# 将测试数据转换为TensorDataset
test_dataset = TensorDataset(torch.tensor(test_data, dtype=torch.float32).unsqueeze(1), torch.tensor(test_labels, dtype=torch.long))

# 划分训练集和验证集
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# 创建DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 训练模型
num_epochs = 30
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

start_time = time.time()

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, torch.max(y_batch, 1)[1])
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        _, predicted = torch.max(outputs, 1)
        _, labels = torch.max(y_batch, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

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
            loss = criterion(outputs, torch.max(y_batch, 1)[1])
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            _, labels = torch.max(y_batch, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_losses.append(val_loss / len(val_loader))
    val_accuracy = 100 * correct / total
    val_accuracies.append(val_accuracy)
    print(f'Validation Loss: {val_loss/len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.2f}%')

    # 保存模型
    torch.save(model.state_dict(), os.path.join(model_save_dir, f"model_epoch_{epoch+1}.pth"))

end_time = time.time()
print(f'Training complete in {end_time - start_time:.2f} seconds')

# 测试模型
model.eval()
test_loss = 0.0
correct = 0
total = 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        loss = criterion(outputs, torch.max(y_batch, 1)[1])
        test_loss += loss.item()
        
        _, predicted = torch.max(outputs, 1)
        _, labels = torch.max(y_batch, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total
print(f'Test Loss: {test_loss/len(test_loader):.4f}, Test Accuracy: {test_accuracy:.2f}%')

# 保存训练结果
np.save(os.path.join(model_save_dir, "train_losses.npy"), train_losses)
np.save(os.path.join(model_save_dir, "val_losses.npy"), val_losses)
np.save(os.path.join(model_save_dir, "train_accuracies.npy"), train_accuracies)
np.save(os.path.join(model_save_dir, "val_accuracies.npy"), val_accuracies)
