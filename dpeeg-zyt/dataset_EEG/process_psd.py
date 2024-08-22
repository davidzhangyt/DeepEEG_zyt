# import os
# import mne
# import numpy as np
# from sklearn.preprocessing import StandardScaler

# # 指定17个标准通道
# final_channels = ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'C3', 'CZ', 'C4', 'TP7', 'CP3', 'CP4', 'TP8', 'P3', 'P4', 'OZ']

# # 设置数据文件路径
# data_path = "D:/EEG-DL-master/dpeeg-main/dpeeg-main/data1-30"
# folders = ["Cr37", "SSD63", "HL96"]
# output_path = "D:/EEG-DL-master/dpeeg-main/dpeeg-main/data17"
# window_size = 2500  # 5秒，500Hz采样率
# step_size = 1250  # 窗口偏移为2.5秒
# n_splits = 10  # 每个类型的数据分成10个文件保存

# # 创建输出目录（如果不存在）
# os.makedirs(output_path, exist_ok=True)

# # 读取EEG数据并提取指定通道、标准化、窗口偏移
# def load_and_process_data(folder, label):
#     noise_path = os.path.join(data_path, folder, 'noise')
#     filepaths = [os.path.join(noise_path, f) for f in os.listdir(noise_path) if f.endswith('.set')]
#     all_data = []
    
#     for filepath in filepaths:
#         try:
#             # 使用 read_epochs_eeglab 读取含有多个 epoch 的 .set 文件
#             epochs = mne.io.read_epochs_eeglab(filepath)
#             all_channel_names = epochs.ch_names
            
#             # 获取选择的通道的索引
#             selected_indices = [all_channel_names.index(channel) for channel in final_channels if channel in all_channel_names]
            
#             # 提取选择的通道的数据
#             selected_eeg_data = epochs.get_data(picks=selected_indices)
            
#             # 检查数据的形状
#             n_epochs, n_channels, n_samples = selected_eeg_data.shape
#             if n_samples < window_size:
#                 print(f"Skipping {filepath} as it doesn't have enough data points")
#                 continue
            
#             # 标准化处理
#             scaler = StandardScaler()
#             reshaped_data = selected_eeg_data.reshape(-1, n_samples)
#             scaled_data = scaler.fit_transform(reshaped_data).reshape(n_epochs, n_channels, n_samples)
            
#             # 窗口偏移
#             for start in range(0, n_samples - window_size + 1, step_size):
#                 windowed_data = scaled_data[:, :, start:start + window_size]
#                 all_data.append(windowed_data)
        
#         except ValueError as e:
#             print(f"Error processing {filepath}: {e}")
#             continue
    
#     if all_data:
#         all_data = np.concatenate(all_data, axis=0)
        
#         # 为数据打上标签
#         labels = np.full((all_data.shape[0],), label)
        
#         # 将数据分成10个部分并保存
#         data_split = np.array_split(all_data, n_splits)
#         label_split = np.array_split(labels, n_splits)
        
#         for i, (data_part, label_part) in enumerate(zip(data_split, label_split)):
#             output_filepath = os.path.join(output_path, folder, 'noise')
#             os.makedirs(output_filepath, exist_ok=True)
#             data_filename = os.path.join(output_filepath, f'{os.path.basename(filepath).replace(".set", "")}_part{i+1}_data.npy')
#             label_filename = os.path.join(output_filepath, f'{os.path.basename(filepath).replace(".set", "")}_part{i+1}_labels.npy')
            
#             np.save(data_filename, data_part)
#             np.save(label_filename, label_part)
#             print(f"Saved {data_filename} and {label_filename}, shape: {data_part.shape}")

# # 处理所有文件夹的数据
# for folder, label in zip(folders, [0, 1, 2]):  # 0, 1, 2 分别对应 Cr37, SSD63, HL96
#     load_and_process_data(folder, label)


import os
import mne
import numpy as np
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler

# 指定17个标准通道
final_channels = ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'C3', 'CZ', 'C4', 'TP7', 'CP3', 'CP4', 'TP8', 'P3', 'P4', 'OZ']

# 设置数据文件路径
data_path = "D:/EEG-DL-master/dpeeg-main/dpeeg-main/data1-30"
folders = ["Cr37", "SSD63", "HL96"]
output_path = "D:/EEG-DL-master/dpeeg-main/dpeeg-main/data17"
window_size = 2500  # 5秒，500Hz采样率
step_size = 1250  # 窗口偏移为2.5秒
n_splits = 10  # 每个类型的数据分成10个文件保存

# 创建输出目录（如果不存在）
os.makedirs(output_path, exist_ok=True)

# 计算PSD特征
def compute_psd(data, sf=500, nperseg=256, noverlap=128):
    psd_list = []
    for epoch in data:
        freqs, psd = welch(epoch, sf, nperseg=nperseg, noverlap=noverlap, axis=-1)
        psd_list.append(psd)
    return np.array(psd_list)

# 读取EEG数据并提取指定通道、标准化、计算PSD、窗口偏移
def load_and_process_data(folder, label):
    noise_path = os.path.join(data_path, folder, 'noise')
    filepaths = [os.path.join(noise_path, f) for f in os.listdir(noise_path) if f.endswith('.set')]
    all_data = []
    all_psd = []
    
    for filepath in filepaths:
        try:
            # 使用 read_epochs_eeglab 读取含有多个 epoch 的 .set 文件
            epochs = mne.io.read_epochs_eeglab(filepath)
            all_channel_names = epochs.ch_names
            
            # 获取选择的通道的索引
            selected_indices = [all_channel_names.index(channel) for channel in final_channels if channel in all_channel_names]
            
            # 提取选择的通道的数据
            selected_eeg_data = epochs.get_data(picks=selected_indices)
            
            # 检查数据的形状
            n_epochs, n_channels, n_samples = selected_eeg_data.shape
            if n_samples < window_size:
                print(f"Skipping {filepath} as it doesn't have enough data points")
                continue
            
            # 标准化处理原始数据
            scaler = StandardScaler()
            reshaped_data = selected_eeg_data.reshape(-1, n_samples)
            scaled_data = scaler.fit_transform(reshaped_data).reshape(n_epochs, n_channels, n_samples)
            
            # 计算PSD特征
            psd_data = compute_psd(scaled_data)
            
            # 标准化处理PSD特征
            psd_scaler = StandardScaler()
            psd_reshaped = psd_data.reshape(-1, psd_data.shape[-1])
            scaled_psd = psd_scaler.fit_transform(psd_reshaped).reshape(psd_data.shape)
            
            # 窗口偏移
            for start in range(0, n_samples - window_size + 1, step_size):
                windowed_data = scaled_data[:, :, start:start + window_size]
                windowed_psd = scaled_psd[:, :, start:start + window_size]
                all_data.append(windowed_data)
                all_psd.append(windowed_psd)
        
        except ValueError as e:
            print(f"Error processing {filepath}: {e}")
            continue
    
    if all_data and all_psd:
        all_data = np.concatenate(all_data, axis=0)
        all_psd = np.concatenate(all_psd, axis=0)
        
        # 为数据打上标签
        labels = np.full((all_data.shape[0],), label)
        
        # 将数据分成10个部分并保存
        data_split = np.array_split(all_data, n_splits)
        psd_split = np.array_split(all_psd, n_splits)
        label_split = np.array_split(labels, n_splits)
        
        for i, (data_part, psd_part, label_part) in enumerate(zip(data_split, psd_split, label_split)):
            output_filepath = os.path.join(output_path, folder, 'noise')
            os.makedirs(output_filepath, exist_ok=True)
            data_filename = os.path.join(output_filepath, f'{os.path.basename(filepath).replace(".set", "")}_part{i+1}_data.npy')
            psd_filename = os.path.join(output_filepath, f'{os.path.basename(filepath).replace(".set", "")}_part{i+1}_psd.npy')
            label_filename = os.path.join(output_filepath, f'{os.path.basename(filepath).replace(".set", "")}_part{i+1}_labels.npy')
            
            np.save(data_filename, data_part)
            np.save(psd_filename, psd_part)
            np.save(label_filename, label_part)
            print(f"Saved {data_filename}, {psd_filename}, and {label_filename}, shape: {data_part.shape}")

# 处理所有文件夹的数据
for folder, label in zip(folders, [0, 1, 2]):  # 0, 1, 2 分别对应 Cr37, SSD63, HL96
    load_and_process_data(folder, label)
