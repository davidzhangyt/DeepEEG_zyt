import os
import numpy as np
import mne
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

# 定义目录和标签
data_dirs = [
    "D:\\EEG-DL-master\\dpeeg-main\\dpeeg-main\\C2_set_files\\CR",
    "D:\\EEG-DL-master\\dpeeg-main\\dpeeg-main\\C2_set_files\\SSD",
    # "D:\\EEG-DL-master\\dpeeg-main\\dpeeg-main\\C2_set_files\\HL"
]
labels = ["normal", "single_deaf", "double_deaf"]
save_dir = "D:\\EEG-DL-master\\dpeeg-main\\dpeeg-main\\C2_data"
os.makedirs(save_dir, exist_ok=True)

# 初始化数据和标签列表
all_data = []
all_labels = []

# 读取数据并标准化
for data_dir, label in zip(data_dirs, labels):
    for filename in os.listdir(data_dir):
        if filename.endswith(".set"):
            filepath = os.path.join(data_dir, filename)
            epochs = mne.io.read_epochs_eeglab(filepath)
            data = epochs.get_data().astype(np.float32)  # 获取数据
            
            # 标准化
            scaler = StandardScaler()
            num_samples, num_channels, num_times = data.shape
            data = data.reshape(num_samples, num_channels * num_times)  # Flatten for scaling
            data = scaler.fit_transform(data)
            data = data.reshape(num_samples, num_channels, num_times)  # Reshape back

            all_data.append(data)
            all_labels.extend([label] * len(data))  # Extend labels for each trial

# 转换为numpy数组并打乱顺序
all_data = np.concatenate(all_data, axis=0)
all_labels = np.array(all_labels)

# 独热编码标签
encoder = OneHotEncoder(sparse=False)
all_labels_encoded = encoder.fit_transform(all_labels.reshape(-1, 1))

# 打乱顺序
indices = np.random.permutation(len(all_data))
all_data = all_data[indices]
all_labels_encoded = all_labels_encoded[indices]

# 保存数据和标签
np.save(os.path.join(save_dir, "data.npy"), all_data)
np.save(os.path.join(save_dir, "labels.npy"), all_labels_encoded)

# 读取数据并切分为训练集、验证集和测试集
data = np.load(os.path.join(save_dir, "data.npy"), allow_pickle=True)
labels_encoded = np.load(os.path.join(save_dir, "labels.npy"), allow_pickle=True)

X_train, X_temp, y_train, y_temp = train_test_split(data, labels_encoded, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 保存分割后的数据
np.save(os.path.join(save_dir, "X_train.npy"), X_train)
np.save(os.path.join(save_dir, "X_val.npy"), X_val)
np.save(os.path.join(save_dir, "X_test.npy"), X_test)
np.save(os.path.join(save_dir, "y_train.npy"), y_train)
np.save(os.path.join(save_dir, "y_val.npy"), y_val)
np.save(os.path.join(save_dir, "y_test.npy"), y_test)

print("数据处理、独热编码和保存完成！")
