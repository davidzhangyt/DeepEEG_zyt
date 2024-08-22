### CR 与 SSD
# import os
# import numpy as np
# import mne
# from sklearn.preprocessing import LabelBinarizer, StandardScaler

# # 指定包含 .set 文件的目录和对应的标签
# data_dirs = [
#     "D:\\EEG-DL-master\\dpeeg-main\\dpeeg-main\\data\\CR", 
#     "D:\\EEG-DL-master\\dpeeg-main\\dpeeg-main\\data\\SSD", 
#     "D:\\EEG-DL-master\\dpeeg-main\\dpeeg-main\\data\\HL"
# ]
# labels_list = ["normal", "single_deaf", "double_deaf"]
# save_dir = "D:\\EEG-DL-master\\dpeeg-main\\dpeeg-main\\preprocessed"

# # 指定17个标准通道
# final_channels = ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'C3', 'CZ', 'C4', 'TP7', 'CP3', 'CP4', 'TP8', 'P3', 'P4', 'OZ']

# # 数据增强函数
# def augment_data(selected_eeg_data, noise_std=0.1):
#     n_epochs, n_channels, n_samples = selected_eeg_data.shape
#     augmented_data = []
#     for i in range(n_epochs):
#         epoch_data = selected_eeg_data[i].copy()  # 使用copy以避免修改原始数据
#         if i % 2 == 0:
#             # 偶数epoch加高斯噪声，每个通道不同
#             for ch in range(n_channels):
#                 noise = np.random.normal(0, noise_std, epoch_data[ch].shape)
#                 epoch_data[ch] += noise
#         else:
#             # 奇数epoch随机变化幅值，每个通道不同
#             for ch in range(n_channels):
#                 scaling_factor = np.random.uniform(0.8, 1.2)
#                 epoch_data[ch] *= scaling_factor
#         # 下采样
#         downsampled_epoch1 = epoch_data[:, ::2]
#         downsampled_epoch2 = epoch_data[:, 1::2]
#         augmented_data.append(downsampled_epoch1)
#         augmented_data.append(downsampled_epoch2)
#     return np.array(augmented_data)

# # 独热编码器
# label_binarizer = LabelBinarizer()
# label_binarizer.fit(labels_list)

# # 保存数据和标签函数
# def save_data_and_labels(data, labels, label, part_start_index):
#     total_data = len(data)
#     batch_size = total_data // 10
#     for i in range(10):
#         start_index = i * batch_size
#         end_index = (i + 1) * batch_size if i < 9 else total_data
#         np.save(os.path.join(save_dir, f"{label}_data_part{part_start_index + i}.npy"), data[start_index:end_index])
#         np.save(os.path.join(save_dir, f"{label}_labels_part{part_start_index + i}.npy"), labels[start_index:end_index])

# # 处理HL数据
# def process_hl_data(data_dir, label, part_start_index, file_range):
#     all_data = []
#     all_labels = []
    
#     for file_name in sorted(os.listdir(data_dir))[file_range[0]:file_range[1]]:
#         if file_name.endswith(".set"):
#             set_file_path = os.path.join(data_dir, file_name)
            
#             # 读取 .set 文件
#             raw = mne.io.read_epochs_eeglab(set_file_path)
            
#             # 获取所有通道名称
#             all_channel_names = raw.ch_names
            
#             # 获取选择的通道的索引
#             selected_indices = [all_channel_names.index(channel) for channel in final_channels if channel in all_channel_names]
            
#             # 提取选择的通道的数据
#             selected_eeg_data = raw.get_data(picks=selected_indices)
            
#             # 对数据进行标准化
#             n_epochs, n_channels, n_samples = selected_eeg_data.shape
#             scaler = StandardScaler()
#             selected_eeg_data = selected_eeg_data.reshape(n_epochs * n_channels, n_samples)
#             selected_eeg_data = scaler.fit_transform(selected_eeg_data)
#             selected_eeg_data = selected_eeg_data.reshape(n_epochs, n_channels, n_samples)
            
#             # 对原始数据进行下采样
#             downsampled_original_data1 = selected_eeg_data[:, :, ::2]
#             downsampled_original_data2 = selected_eeg_data[:, :, 1::2]
            
#             # 增强数据
#             augmented_data = augment_data(selected_eeg_data)
            
#             # 合并原始数据和增强数据
#             combined_data = np.concatenate((downsampled_original_data1, downsampled_original_data2, augmented_data), axis=0)
            
#             # 相邻epoch拼接
#             new_epochs = []
#             for i in range(combined_data.shape[0] - 1):
#                 first_half = combined_data[i, :, -625:]  # 取前一个epoch的后半段
#                 second_half = combined_data[i + 1, :, :625]  # 取后一个epoch的前半段
#                 new_epoch = np.concatenate((first_half, second_half), axis=1)  # 拼接成新的epoch
#                 new_epochs.append(new_epoch)
            
#             new_epochs = np.array(new_epochs)
            
#             # 将新生成的epochs添加到combined_data的后面
#             final_combined_data = np.concatenate((combined_data, new_epochs), axis=0)
            
#             # 添加数据和标签
#             all_data.append(final_combined_data)
#             all_labels.extend([label] * final_combined_data.shape[0])

#     # 将所有数据合并到一个数组中
#     all_data = np.concatenate(all_data, axis=0)
#     all_labels = np.array(all_labels)
    
#     # 独热编码标签
#     all_labels_encoded = label_binarizer.transform(all_labels)
    
#     # 保存数据和标签
#     save_data_and_labels(all_data, all_labels_encoded, label, part_start_index)

# # 读取并处理CR和SSD数据
# for data_dir, label in zip(data_dirs[:2], labels_list[:2]):
#     all_data = []
#     all_labels = []

#     for file_name in os.listdir(data_dir):
#         if file_name.endswith(".set"):
#             set_file_path = os.path.join(data_dir, file_name)
            
#             # 读取 .set 文件
#             raw = mne.io.read_epochs_eeglab(set_file_path)
            
#             # 获取所有通道名称
#             all_channel_names = raw.ch_names
            
#             # 获取选择的通道的索引
#             selected_indices = [all_channel_names.index(channel) for channel in final_channels if channel in all_channel_names]
            
#             # 提取选择的通道的数据
#             selected_eeg_data = raw.get_data(picks=selected_indices)
            
#             # 对数据进行标准化
#             n_epochs, n_channels, n_samples = selected_eeg_data.shape
#             scaler = StandardScaler()
#             selected_eeg_data = selected_eeg_data.reshape(n_epochs * n_channels, n_samples)
#             selected_eeg_data = scaler.fit_transform(selected_eeg_data)
#             selected_eeg_data = selected_eeg_data.reshape(n_epochs, n_channels, n_samples)
            
#             # 对原始数据进行下采样
#             downsampled_original_data1 = selected_eeg_data[:, :, ::2]
#             downsampled_original_data2 = selected_eeg_data[:, :, 1::2]
            
#             # 增强数据
#             augmented_data = augment_data(selected_eeg_data)
            
#             # 合并原始数据和增强数据
#             combined_data = np.concatenate((downsampled_original_data1, downsampled_original_data2, augmented_data), axis=0)
            
#             # 相邻epoch拼接
#             new_epochs = []
#             for i in range(combined_data.shape[0] - 1):
#                 first_half = combined_data[i, :, -625:]  # 取前一个epoch的后半段
#                 second_half = combined_data[i + 1, :, :625]  # 取后一个epoch的前半段
#                 new_epoch = np.concatenate((first_half, second_half), axis=1)  # 拼接成新的epoch
#                 new_epochs.append(new_epoch)
            
#             new_epochs = np.array(new_epochs)
            
#             # 将新生成的epochs添加到combined_data的后面
#             final_combined_data = np.concatenate((combined_data, new_epochs), axis=0)
            
#             # 添加数据和标签
#             all_data.append(final_combined_data)
#             all_labels.extend([label] * final_combined_data.shape[0])

#     # 将所有数据合并到一个数组中
#     all_data = np.concatenate(all_data, axis=0)
#     all_labels = np.array(all_labels)
    
#     # 独热编码标签
#     all_labels_encoded = label_binarizer.transform(all_labels)

#     print(f"Final Data Shape for {label}:", all_data.shape)
#     print(f"Labels Shape for {label}:", all_labels_encoded.shape)

#     # 确保保存目录存在
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)

#     # 将数据分成10个文件保存
#     save_data_and_labels(all_data, all_labels_encoded, label, 1)

# # 分别处理HL数据的前后两部分
# hl_files = sorted(os.listdir(data_dirs[2]))

# # 处理前半段HL数据
# process_hl_data(data_dirs[2], labels_list[2], 1, (0, len(hl_files) // 2))

# # 处理后半段HL数据
# process_hl_data(data_dirs[2], labels_list[2], 11, (len(hl_files) // 2, len(hl_files)))


### HL
import os
import numpy as np
import mne
from sklearn.preprocessing import LabelBinarizer, StandardScaler

# 指定包含 .set 文件的目录和对应的标签
data_dirs = [
    "D:\\EEG-DL-master\\dpeeg-main\\dpeeg-main\\data\\HL"
]
labels_list = ["normal", "single_deaf", "double_deaf"]  # 更新为三个类别的标签列表
save_dir = "D:\\EEG-DL-master\\dpeeg-main\\dpeeg-main\\preprocessed"

# 指定17个标准通道
final_channels = ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'C3', 'CZ', 'C4', 'TP7', 'CP3', 'CP4', 'TP8', 'P3', 'P4', 'OZ']

# 数据增强函数
def augment_data(selected_eeg_data, noise_std=0.1):
    n_epochs, n_channels, n_samples = selected_eeg_data.shape
    augmented_data = []
    for i in range(n_epochs):
        epoch_data = selected_eeg_data[i].copy()  # 使用copy以避免修改原始数据
        if i % 2 == 0:
            # 偶数epoch加高斯噪声，每个通道不同
            for ch in range(n_channels):
                noise = np.random.normal(0, noise_std, epoch_data[ch].shape)
                epoch_data[ch] += noise
        else:
            # 奇数epoch随机变化幅值，每个通道不同
            for ch in range(n_channels):
                scaling_factor = np.random.uniform(0.8, 1.2)
                epoch_data[ch] *= scaling_factor
        # 下采样
        downsampled_epoch1 = epoch_data[:, ::2]
        downsampled_epoch2 = epoch_data[:, 1::2]
        augmented_data.append(downsampled_epoch1)
        augmented_data.append(downsampled_epoch2)
    return np.array(augmented_data)

# 独热编码器
label_binarizer = LabelBinarizer()
label_binarizer.fit(labels_list)

# 保存数据和标签函数
def save_data_and_labels(data, labels, part_start_index):
    total_data = len(data)
    batch_size = total_data // 10
    for i in range(10):
        start_index = i * batch_size
        end_index = (i + 1) * batch_size if i < 9 else total_data
        np.save(os.path.join(save_dir, f"double_deaf_data_part{part_start_index + i}.npy"), data[start_index:end_index])
        np.save(os.path.join(save_dir, f"double_deaf_labels_part{part_start_index + i}.npy"), labels[start_index:end_index])

# 处理HL数据
def process_hl_data(data_dir, part_start_index, file_range):
    all_data = []
    all_labels = []
    
    for file_name in sorted(os.listdir(data_dir))[file_range[0]:file_range[1]]:
        if file_name.endswith(".set"):
            set_file_path = os.path.join(data_dir, file_name)
            
            # 读取 .set 文件
            raw = mne.io.read_epochs_eeglab(set_file_path)
            
            # 获取所有通道名称
            all_channel_names = raw.ch_names
            
            # 获取选择的通道的索引
            selected_indices = [all_channel_names.index(channel) for channel in final_channels if channel in all_channel_names]
            
            # 提取选择的通道的数据
            selected_eeg_data = raw.get_data(picks=selected_indices)
            
            # 对数据进行标准化
            n_epochs, n_channels, n_samples = selected_eeg_data.shape
            scaler = StandardScaler()
            selected_eeg_data = selected_eeg_data.reshape(n_epochs * n_channels, n_samples)
            selected_eeg_data = scaler.fit_transform(selected_eeg_data)
            selected_eeg_data = selected_eeg_data.reshape(n_epochs, n_channels, n_samples)
            
            # 对原始数据进行下采样
            downsampled_original_data1 = selected_eeg_data[:, :, ::2]
            downsampled_original_data2 = selected_eeg_data[:, :, 1::2]
            
            # 增强数据
            augmented_data = augment_data(selected_eeg_data)
            
            # 合并原始数据和增强数据
            combined_data = np.concatenate((downsampled_original_data1, downsampled_original_data2, augmented_data), axis=0)
            
            # 相邻epoch拼接
            new_epochs = []
            for i in range(combined_data.shape[0] - 1):
                first_half = combined_data[i, :, -625:]  # 取前一个epoch的后半段
                second_half = combined_data[i + 1, :, :625]  # 取后一个epoch的前半段
                new_epoch = np.concatenate((first_half, second_half), axis=1)  # 拼接成新的epoch
                new_epochs.append(new_epoch)
            
            new_epochs = np.array(new_epochs)
            
            # 将新生成的epochs添加到combined_data的后面
            final_combined_data = np.concatenate((combined_data, new_epochs), axis=0)
            
            # 添加数据和标签
            all_data.append(final_combined_data)
            all_labels.extend([labels_list[2]] * final_combined_data.shape[0])

    # 将所有数据合并到一个数组中
    all_data = np.concatenate(all_data, axis=0)
    all_labels = np.array(all_labels)
    
    # 独热编码标签
    all_labels_encoded = label_binarizer.transform(all_labels)
    
    # 保存数据和标签
    save_data_and_labels(all_data, all_labels_encoded, part_start_index)

# 分别处理HL数据的前后两部分
hl_files = sorted(os.listdir(data_dirs[0]))

# 处理前半段HL数据
process_hl_data(data_dirs[0], 1, (0, len(hl_files) // 2))

# 处理后半段HL数据
process_hl_data(data_dirs[0], 11, (len(hl_files) // 2, len(hl_files)))
