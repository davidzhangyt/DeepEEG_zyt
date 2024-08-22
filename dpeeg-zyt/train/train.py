import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
import time
from sklearn.preprocessing import LabelBinarizer

# 数据加载和预处理函数
def load_and_preprocess(file_path, label_binarizer):
    data = np.load(file_path)
    X = data['X']
    y = data['y']
    y = label_binarizer.transform(y)  # 将标签转换为独热编码
    return X, y

# 定义标签编码器
label_binarizer = LabelBinarizer()
label_binarizer.fit(["normal", "single_deaf", "double_deaf"])

# 自定义数据集类
class EEGDataset(Dataset):
    def __init__(self, data_files, label_binarizer):
        self.data_files = data_files
        self.label_binarizer = label_binarizer
        self.current_file_index = -1
        self.X = None
        self.y = None
        self.load_next_file()

    def load_next_file(self):
        self.current_file_index = (self.current_file_index + 1) % len(self.data_files)
        file = self.data_files[self.current_file_index]
        self.X, self.y = load_and_preprocess(file, self.label_binarizer)
        

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        x = x[np.newaxis, :]  # 添加一个新维度使其成为单通道
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# 定义训练和测试文件路径
train_files = ["D:\\EEG-DL-master\\dpeeg-main\\dpeeg-main\\temp_data\\train_0.npz",
               "D:\\EEG-DL-master\\dpeeg-main\\dpeeg-main\\temp_data\\train_1.npz",
               "D:\\EEG-DL-master\\dpeeg-main\\dpeeg-main\\temp_data\\train_2.npz",
               "D:\\EEG-DL-master\\dpeeg-main\\dpeeg-main\\temp_data\\train_3.npz",
               "D:\\EEG-DL-master\\dpeeg-main\\dpeeg-main\\temp_data\\train_4.npz"]

test_files = ["D:\\EEG-DL-master\\dpeeg-main\\dpeeg-main\\temp_data\\test_0.npz",
              "D:\\EEG-DL-master\\dpeeg-main\\dpeeg-main\\temp_data\\test_1.npz",
              "D:\\EEG-DL-master\\dpeeg-main\\dpeeg-main\\temp_data\\test_2.npz",
              "D:\\EEG-DL-master\\dpeeg-main\\dpeeg-main\\temp_data\\test_3.npz",
              "D:\\EEG-DL-master\\dpeeg-main\\dpeeg-main\\temp_data\\test_4.npz"]

# 创建数据集和数据加载器
train_dataset = EEGDataset(train_files, label_binarizer)
test_dataset = EEGDataset(test_files, label_binarizer)

batch_size = 64

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
model = EEGConformer(17, 1250, 3).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 设置保存模型路径
model_save_dir = "D:\\EEG-DL-master\\dpeeg-main\\dpeeg-main\\model_save"
os.makedirs(model_save_dir, exist_ok=True)

# 训练模型
num_epochs = 30
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

start_time = time.time()

for epoch in range(num_epochs):
    # 在每个 epoch 开始时重新加载一个新文件
    train_dataset.load_next_file()
    # val_dataset.load_next_file()

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

    # # 验证模型
    # model.eval()
    # val_running_loss = 0.0
    # val_correct = 0
    # val_total = 0
    # with torch.no_grad():
    #     for X_batch, y_batch in val_loader:
    #         X_batch, y_batch = X_batch.to(device), y_batch.to(device)
    #         outputs = model(X_batch)
    #         loss = criterion(outputs, torch.max(y_batch, 1)[1])
    #         val_running_loss += loss.item()
            
    #         _, predicted = torch.max(outputs, 1)
    #         _, labels = torch.max(y_batch, 1)
    #         val_total += labels.size(0)
    #         val_correct += (predicted == labels).sum().item()

    # val_losses.append(val_running_loss / len(val_loader))
    # val_accuracy = 100 * val_correct / val_total
    # val_accuracies.append(val_accuracy)
    # print(f'Validation Loss: {val_running_loss/len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.2f}%')

    # 保存中间模型
    model_save_path = os.path.join(model_save_dir, f"eeg_conformer_epoch_{epoch+1}.pth")
    if (epoch+1) % 5 == 0:
        torch.save(model.state_dict(), model_save_path)
        print(f'Model saved for epoch {epoch+1}.')

end_time = time.time()
print('Training finished.')
print(f'Total training time: {end_time - start_time:.2f} seconds')

# 绘制训练和验证损失
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_accuracies, label='Training Accuracy')
plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# 测试模型
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        _, labels = torch.max(y_batch, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the model on the test data: {100 * correct / total:.2f}%')

# 加载模型（如果需要）
# model.load_state_dict(torch.load(model_save_path))
# model.to(device)
# print('Model loaded.')


# 加载模型（如果需要）
# model.load_state_dict(torch.load(model_save_path))
# model.to(device)
# print('Model loaded.')



# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import torch
# from torch.utils.data import Dataset, DataLoader, random_split
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from einops import rearrange
# from einops.layers.torch import Rearrange
# import time
# from sklearn.preprocessing import LabelBinarizer

# # 数据加载和预处理函数
# def load_and_preprocess(file_path, label_binarizer):
#     data = np.load(file_path)
#     X = data['X']
#     y = data['y']
#     y = label_binarizer.transform(y)  # 将标签转换为独热编码
#     return X, y

# # 定义标签编码器
# label_binarizer = LabelBinarizer()
# label_binarizer.fit(["normal", "single_deaf", "double_deaf"])

# # 自定义数据集类
# class EEGDataset(Dataset):
#     def __init__(self, data_files, label_binarizer):
#         self.data_files = data_files
#         self.label_binarizer = label_binarizer
#         self.file_index = 0
#         self.X, self.y = self.load_data()

#     def load_data(self):
#         if self.file_index >= len(self.data_files):
#             self.file_index = 0
#         file = self.data_files[self.file_index]
#         self.file_index += 1
#         return load_and_preprocess(file, self.label_binarizer)
    
#     def __len__(self):
#         return len(self.X)
    
#     def __getitem__(self, idx):
#         x = self.X[idx]
#         y = self.y[idx]
#         x = x[np.newaxis, :]  # 添加一个新维度使其成为单通道
#         return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# # 定义训练和测试文件路径
# train_files = ["D:\\EEG-DL-master\\dpeeg-main\\dpeeg-main\\temp_data\\train_0.npz",
#                "D:\\EEG-DL-master\\dpeeg-main\\dpeeg-main\\temp_data\\train_1.npz",
#                "D\\EEG-DL-master\\dpeeg-main\\dpeeg-main\\temp_data\\train_2.npz",
#                "D\\EEG-DL-master\\dpeeg-main\\dpeeg-main\\temp_data\\train_3.npz",
#                "D\\EEG-DL-master\\dpeeg-main\\dpeeg-main\\temp_data\\train_4.npz"]

# test_files = ["D:\\EEG-DL-master\\dpeeg-main\\dpeeg-main\\temp_data\\test_0.npz",
#               "D\\EEG-DL-master\\dpeeg-main\\dpeeg-main\\temp_data\\test_1.npz",
#               "D\\EEG-DL-master\\dpeeg-main\\dpeeg-main\\temp_data\\test_2.npz",
#               "D\\EEG-DL-master\\dpeeg-main\\dpeeg-main\\temp_data\\test_3.npz",
#               "D\\EEG-DL-master\\dpeeg-main\\dpeeg-main\\temp_data\\test_4.npz"]

# # 创建数据集和数据加载器
# train_dataset = EEGDataset(train_files, label_binarizer)
# test_dataset = EEGDataset(test_files, label_binarizer)

# # 将训练数据集分割为训练集和验证集
# train_size = int(0.8 * len(train_dataset))
# val_size = len(train_dataset) - train_size
# train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# batch_size = 64

# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# # 检查是否有可用的GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # 定义神经网络模型
# class PatchEmbedding(nn.Module):
#     def __init__(self, nCh, emb_size=40):
#         super().__init__()

#         self.shallownet = nn.Sequential(
#             nn.Conv2d(1, 40, (1, 25), (1, 1)),
#             nn.Conv2d(40, 40, (nCh, 1), (1, 1)),
#             nn.BatchNorm2d(40),
#             nn.ELU(),
#             nn.AvgPool2d((1, 75), (1, 15)),
#             nn.Dropout(0.5),
#         )

#         self.projection = nn.Sequential(
#             nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),
#             Rearrange('b e (h) (w) -> b (h w) e'),
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.shallownet(x)
#         x = self.projection(x)
#         return x

# class MultiHeadAttention(nn.Module):
#     def __init__(self, emb_size, num_heads, dropout):
#         super().__init__()
#         self.emb_size = emb_size
#         self.num_heads = num_heads
#         self.keys = nn.Linear(emb_size, emb_size)
#         self.queries = nn.Linear(emb_size, emb_size)
#         self.values = nn.Linear(emb_size, emb_size)
#         self.att_drop = nn.Dropout(dropout)
#         self.projection = nn.Linear(emb_size, emb_size)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
#         keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
#         values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
#         energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)

#         scaling = self.emb_size ** (1 / 2)
#         att = F.softmax(energy / scaling, dim=-1)
#         att = self.att_drop(att)
#         out = torch.einsum('bhal, bhlv -> bhav ', att, values)
#         out = rearrange(out, "b h n d -> b n (h d)")
#         out = self.projection(out)
#         return out

# class ResidualAdd(nn.Module):
#     def __init__(self, fn):
#         super().__init__()
#         self.fn = fn

#     def forward(self, x, **kwargs):
#         res = x
#         x = self.fn(x, **kwargs)
#         x += res
#         return x

# class FeedForwardBlock(nn.Sequential):
#     def __init__(self, emb_size, expansion, drop_p):
#         super().__init__(
#             nn.Linear(emb_size, expansion * emb_size),
#             nn.GELU(),
#             nn.Dropout(drop_p),
#             nn.Linear(expansion * emb_size, emb_size),
#         )

# class TransformerEncoderBlock(nn.Sequential):
#     def __init__(self,
#                  emb_size,
#                  num_heads=10,
#                  drop_p=0.5,
#                  forward_expansion=4,
#                  forward_drop_p=0.5):
#         super().__init__(
#             ResidualAdd(nn.Sequential(
#                 nn.LayerNorm(emb_size),
#                 MultiHeadAttention(emb_size, num_heads, drop_p),
#                 nn.Dropout(drop_p)
#             )),
#             ResidualAdd(nn.Sequential(
#                 nn.LayerNorm(emb_size),
#                 FeedForwardBlock(
#                     emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
#                 nn.Dropout(drop_p)
#             )
#             ))

# class TransformerEncoder(nn.Sequential):
#     def __init__(self, depth, emb_size, num_heads, drop_p, forward_expansion, forward_drop_p):
#         super().__init__(*[
#             TransformerEncoderBlock(emb_size, num_heads, drop_p, forward_expansion, forward_drop_p)
#             for _ in range(depth)
#         ])

# class ClassificationHead(nn.Sequential):
#     def __init__(self, in_features, cls):
#         super().__init__(
#             nn.Flatten(),
#             nn.Linear(in_features, 256),
#             nn.ELU(),
#             nn.Dropout(0.5),
#             nn.Linear(256, 32),
#             nn.ELU(),
#             nn.Dropout(0.3),
#             nn.Linear(32, cls),
#             nn.LogSoftmax(dim=1)
#         )

# class EEGConformer(nn.Module):
#     def __init__(
#         self, 
#         nCh, 
#         nTime, 
#         cls, 
#         emb_size=40, 
#         depth=6, 
#         num_heads=10, 
#         drop_p=0.5, 
#         forward_expansion=4, 
#         forward_drop_p=0.5
#     ) -> None:
#         super().__init__()
#         self.nCh = nCh
#         self.nTime = nTime
#         self.patch_embedding = PatchEmbedding(nCh, emb_size)
#         self.transformer_encoder = TransformerEncoder(
#             depth, emb_size, num_heads, drop_p, forward_expansion, forward_drop_p
#         )
#         in_features = self._forward_transformer(torch.randn(1, 1, nCh, nTime)).size(1)
#         self.classification_head = ClassificationHead(in_features, cls)

#     def _forward_transformer(self, x) -> torch.Tensor:
#         x = self.patch_embedding(x)
#         x = self.transformer_encoder(x)
#         return x.flatten(start_dim=1, end_dim=-1)

#     def forward(self, x):
#         out = self.patch_embedding(x)
#         out = self.transformer_encoder(out)
#         return self.classification_head(out)

# # 初始化模型并移动到GPU
# model = EEGConformer(116, 2500, 3).to(device)

# # 定义损失函数和优化器
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # 设置保存模型路径
# model_save_dir = "D:\\EEG-DL-master\\dpeeg-main\\dpeeg-main\\model_save"
# os.makedirs(model_save_dir, exist_ok=True)

# # 训练模型
# num_epochs = 15
# train_losses = []
# val_losses = []
# train_accuracies = []
# val_accuracies = []

# start_time = time.time()

# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     correct = 0
#     total = 0
#     for X_batch, y_batch in train_loader:
#         X_batch, y_batch = X_batch.to(device), y_batch.to(device)
#         optimizer.zero_grad()
#         outputs = model(X_batch)
#         loss = criterion(outputs, torch.max(y_batch, 1)[1])
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
        
#         _, predicted = torch.max(outputs, 1)
#         _, labels = torch.max(y_batch, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#     print(f'train:correct: {correct}, total: {total}')

#     train_losses.append(running_loss / len(train_loader))
#     train_accuracy = 100 * correct / total
#     train_accuracies.append(train_accuracy)
#     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {train_accuracy:.2f}%')

#     # 验证模型
#     model.eval()
#     val_running_loss = 0.0
#     val_correct = 0
#     val_total = 0
#     with torch.no_grad():
#         for X_batch, y_batch in val_loader:
#             X_batch, y_batch = X_batch.to(device), y_batch.to(device)
#             outputs = model(X_batch)
#             loss = criterion(outputs, torch.max(y_batch, 1)[1])
#             val_running_loss += loss.item()
            
#             _, predicted = torch.max(outputs, 1)
#             _, labels = torch.max(y_batch, 1)
#             val_total += labels.size(0)
#             val_correct += (predicted == labels).sum().item()
#         print(f'correct: {val_correct}, total: {val_total}')

#     val_losses.append(val_running_loss / len(val_loader))
#     val_accuracy = 100 * val_correct / val_total
#     val_accuracies.append(val_accuracy)
#     print(f'Validation Loss: {val_running_loss/len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.2f}%')

#     # 保存中间模型
#     model_save_path = os.path.join(model_save_dir, f"eeg_conformer_epoch_{epoch+1}.pth")
#     if (epoch+1) % 5 == 0:
#         torch.save(model.state_dict(), model_save_path)
#         print(f'Model saved for epoch {epoch+1}.')

# end_time = time.time()
# print('Training finished.')
# print(f'Total training time: {end_time - start_time:.2f} seconds')

# # 绘制训练和验证损失
# plt.figure(figsize=(12, 4))
# plt.subplot(1, 2, 1)
# plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
# plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.subplot(1, 2, 2)
# plt.plot(range(1, num_epochs + 1), train_accuracies, label='Training Accuracy')
# plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()

# # 测试模型
# model.eval()
# correct = 0
# total = 0
# with torch.no_grad():
#     for X_batch, y_batch in test_loader:
#         X_batch, y_batch = X_batch.to(device), y_batch.to(device)
#         outputs = model(X_batch)
#         _, predicted = torch.max(outputs, 1)
#         _, labels = torch.max(y_batch, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
        
#     print(f'Test:correct: {correct}, total: {total}')

# print(f'Accuracy of the model on the test data: {100 * correct / total:.2f}%')

# # 加载模型（如果需要）
# # model.load_state_dict(torch.load(model_save_path))
# # model.to(device)
# # print('Model loaded.')
