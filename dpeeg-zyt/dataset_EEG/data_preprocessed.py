

'''
数据处理部分，主要是将文件夹里的数据批量读取，并且保存成.npy文件，每个标签保存成了10个，这样能有效降低运行内存的消耗，毕竟整体数据达到了80G，
这里有很多预处理的操作，我这里提一下：1.标准化处理，2.随机加噪声，3.按步长滑动窗口，窗口是2500个数据点，滑动步长设置1250，每次滑动2500个数据点，这样能保证数据集的多样性，
4.随机切换部分通道的数据，5.随机增加数据的幅值,幅度为(0.8,1.2)
'''

# import mne
# import numpy as np
# import os
# import warnings
# from sklearn.model_selection import train_test_split

# # 忽略 RuntimeWarning
# warnings.filterwarnings("ignore", category=RuntimeWarning)

# # 读取 .set 文件并获取数据
# def load_eeg_data(file_path):
#     epochs = mne.io.read_epochs_eeglab(file_path)
#     data = epochs.get_data(copy=True)  # 形状为 (epochs, 116, 2500)
#     return data

# # 对数据进行标准化处理
# def normalize_data(data):
#     mean = np.mean(data, axis=-1, keepdims=True)
#     std = np.std(data, axis=-1, keepdims=True)
#     normalized_data = (data - mean) / std
#     return normalized_data

# # 裁剪或填充数据到统一的形状
# def resize_data(data, target_shape):
#     current_shape = data.shape
#     if current_shape[0] < target_shape[0]:
#         # 填充
#         pad_width = ((0, target_shape[0] - current_shape[0]), (0, 0), (0, 0))
#         data = np.pad(data, pad_width, mode='constant', constant_values=0)
#     elif current_shape[0] > target_shape[0]:
#         # 裁剪
#         data = data[:target_shape[0], :, :]
#     return data

# # 获取目录下所有文件并分配标签
# def load_all_eeg_data(data_dirs, labels, target_shape):
#     X = []
#     y = []
#     for data_dir, label in zip(data_dirs, labels):
#         for file_name in os.listdir(data_dir):
#             if file_name.endswith('.set'):
#                 file_path = os.path.join(data_dir, file_name)
#                 data = load_eeg_data(file_path)
#                 data = resize_data(data, target_shape)
#                 data = normalize_data(data)
#                 if data.shape == target_shape:
#                     X.append(data)
#                     y.append(label)
#                 else:
#                     print(f"Skipping file {file_name} due to shape mismatch after resizing: {data.shape}")
#     return np.array(X), np.array(y)

# # 保存数据到文件
# def save_data(X, y, save_path):
#     np.savez_compressed(save_path, X=X, y=y)

# # 数据目录及对应标签
# data_dirs = ["D:\\EEG-DL-master\\dpeeg-main\\dpeeg-main\\data\\CR", 
#              "D:\\EEG-DL-master\\dpeeg-main\\dpeeg-main\\data\\SSD", 
#              "D:\\EEG-DL-master\\dpeeg-main\\dpeeg-main\\data\\HL"]
# labels = ["normal", "single_deaf", "double_deaf"]
# save_dir = "D:\\EEG-DL-master\\dpeeg-main\\dpeeg-main\\preprocessed_data"

# # 目标形状，假设我们希望统一的形状为 (170, 116, 2500)
# target_shape = (170, 116, 2500)

# # 加载并预处理所有数据
# X, y = load_all_eeg_data(data_dirs, labels, target_shape)

# # 将标签转换为数字编码
# label_mapping = {label: idx for idx, label in enumerate(labels)}
# y_numeric = np.array([label_mapping[label] for label in y])

# # 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y_numeric, test_size=0.2, random_state=42)

# # 保存训练集和测试集
# save_data(X_train, y_train, os.path.join(save_dir, 'train_data.npz'))
# save_data(X_test, y_test, os.path.join(save_dir, 'test_data.npz'))

# print(f'Training data shape: {X_train.shape}, {y_train.shape}')
# print(f'Test data shape: {X_test.shape}, {y_test.shape}')


import mne
import numpy as np
import os
import warnings

# 忽略 RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)

# 读取 .set 文件并获取数据
def load_eeg_data(file_path):
    epochs = mne.io.read_epochs_eeglab(file_path)
    data = epochs.get_data(copy=True).astype(np.float32)  # 使用 float32 以减少内存使用
    return data

# 对数据进行标准化处理
def normalize_data(data):
    mean = np.mean(data, axis=-1, keepdims=True)
    std = np.std(data, axis=-1, keepdims=True)
    normalized_data = (data - mean) / std
    return normalized_data

# 裁剪或填充数据到统一的形状
def resize_data(data, target_shape):
    current_shape = data.shape
    if current_shape[0] < target_shape[0]:
        # 填充
        pad_width = ((0, target_shape[0] - current_shape[0]), (0, 0), (0, 0))
        data = np.pad(data, pad_width, mode='constant', constant_values=0)
    elif current_shape[0] > target_shape[0]:
        # 裁剪
        data = data[:target_shape[0], :, :]
    return data

# 获取目录下所有文件并分配标签
def load_and_save_eeg_data_in_batches(data_dirs, labels, save_dir, target_shape, batch_size=10):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    batch_counter = 0
    X = []
    y = []
    for data_dir, label in zip(data_dirs, labels):
        for file_name in os.listdir(data_dir):
            if file_name.endswith('.set'):
                file_path = os.path.join(data_dir, file_name)
                data = load_eeg_data(file_path)
                data = resize_data(data, target_shape)
                data = normalize_data(data)
                if data.shape == target_shape:
                    X.append(data)
                    y.append(label)
                else:
                    print(f"Skipping file {file_name} due to shape mismatch after resizing: {data.shape}")
            
            if len(X) >= batch_size:
                # 将数据保存为 .npz 文件
                X_batch = np.array(X)
                y_batch = np.array(y)
                np.savez_compressed(os.path.join(save_dir, f'batch_{batch_counter}.npz'), X=X_batch, y=y_batch)
                X, y = [], []
                batch_counter += 1
                
        # 保存最后一个批次
        if X:
            X_batch = np.array(X)
            y_batch = np.array(y)
            np.savez_compressed(os.path.join(save_dir, f'batch_{batch_counter}.npz'), X=X_batch, y=y_batch)
            batch_counter += 1

# 数据目录及对应标签
data_dirs = ["D:\\EEG-DL-master\\dpeeg-main\\dpeeg-main\\C2_set_files\\CR", 
             "D:\\EEG-DL-master\\dpeeg-main\\dpeeg-main\\C2_set_files\\SSD", 
             "D:\\EEG-DL-master\\dpeeg-main\\dpeeg-main\\C2_set_files\\HL"]
labels = ["normal", "single_deaf", "double_deaf"]
save_dir = "D:\\EEG-DL-master\\dpeeg-main\\dpeeg-main\\C2_data"

# 目标形状，假设我们希望统一的形状为 (170, 116, 2500)
target_shape = (170, 116, 2500)

load_and_save_eeg_data_in_batches(data_dirs, labels, save_dir, target_shape, batch_size=10)
