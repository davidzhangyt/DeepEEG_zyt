import os
import numpy as np
import mne
from sklearn.preprocessing import LabelBinarizer, StandardScaler

# 指定包含 .set 文件的目录和对应的标签
data_dirs = ["D:\\EEG-DL-master\\dpeeg-main\\dpeeg-main\\C2_set_files\\CR", 
             "D:\\EEG-DL-master\\dpeeg-main\\dpeeg-main\\C2_set_files\\SSD", 
             "D:\\EEG-DL-master\\dpeeg-main\\dpeeg-main\\C2_set_files\\HL"]
labels_list = ["normal", "single_deaf", "double_deaf"]
save_dir = "D:\\EEG-DL-master\\dpeeg-main\\dpeeg-main\\preprocessed_5000"

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

# 读取并处理所有 .set 文件
for data_dir, label in zip(data_dirs, labels_list):
    all_data = []
    all_labels = []

    for file_name in os.listdir(data_dir):
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
            
            # 合并四个连续的 epoch
            num_full_epochs = combined_data.shape[0] // 4 * 4
            combined_data = combined_data[:num_full_epochs]  # 丢弃非整数的数据段
            
            # 将每四个连续的 epoch 合并成一个新的 epoch
            reshaped_data = combined_data.reshape(-1, 17, 5000)
            
            # 添加数据和标签
            all_data.append(reshaped_data)
            all_labels.extend([label] * reshaped_data.shape[0])

    # 将所有数据合并到一个数组中
    all_data = np.concatenate(all_data, axis=0)
    all_labels = np.array(all_labels)
    
    # 独热编码标签
    all_labels_encoded = label_binarizer.transform(all_labels)

    print(f"Final Data Shape for {label}:", all_data.shape)
    print(f"Labels Shape for {label}:", all_labels_encoded.shape)

    # 确保保存目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 将数据分成10个文件保存
    total_data = len(all_data)
    batch_size = total_data // 10
    for i in range(10):
        start_index = i * batch_size
        end_index = (i + 1) * batch_size if i < 9 else total_data
        np.save(os.path.join(save_dir, f"{label}_data_part{i+1}.npy"), all_data[start_index:end_index])
        np.save(os.path.join(save_dir, f"{label}_labels_part{i+1}.npy"), all_labels_encoded[start_index:end_index])



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

# # 读取并处理所有 .set 文件
# for data_dir, label in zip(data_dirs, labels_list):
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
#     total_data = len(all_data)
#     batch_size = total_data // 10
#     for i in range(10):
#         start_index = i * batch_size
#         end_index = (i + 1) * batch_size if i < 9 else total_data
#         np.save(os.path.join(save_dir, f"{label}_data_part{i+1}.npy"), all_data[start_index:end_index])
#         np.save(os.path.join(save_dir, f"{label}_labels_part{i+1}.npy"), all_labels_encoded[start_index:end_index])



# # 适用于HL，拆分成10个文件
# import os
# import numpy as np

# # 文件路径
# hl_data_files = [
#     "D:\\EEG-DL-master\\dpeeg-main\\dpeeg-main\\preprocessed_data\\HL\\double_deaf_data_part1.npy",
#     "D:\\EEG-DL-master\\dpeeg-main\\dpeeg-main\\preprocessed_data\\HL\\double_deaf_data_part2.npy"
# ]
# hl_label_files = [
#     "D:\\EEG-DL-master\\dpeeg-main\\dpeeg-main\\preprocessed_data\\HL\\double_deaf_labels_part1.npy",
#     "D:\\EEG-DL-master\\dpeeg-main\\dpeeg-main\\preprocessed_data\\HL\\double_deaf_labels_part2.npy"
# ]

# # 确保保存目录存在
# save_dir = "D:\\EEG-DL-master\\dpeeg-main\\dpeeg-main\\preprocessed_data\\HL"
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)

# # 读取文件
# def load_data(files):
#     data, labels = [], []
#     for file in files:
#         data.append(np.load(file))
#     return np.concatenate(data, axis=0)

# # 分割并保存数据和标签
# def split_and_save_data(data, labels, label_name, part_name):
#     total_data = len(data)
#     batch_size = total_data // 10
#     for i in range(10):
#         start_index = i * batch_size
#         end_index = (i + 1) * batch_size if i < 9 else total_data
#         np.save(os.path.join(save_dir, f"{label_name}_data_part{part_name}_{i+1}.npy"), data[start_index:end_index])
#         np.save(os.path.join(save_dir, f"{label_name}_labels_part{part_name}_{i+1}.npy"), labels[start_index:end_index])

# # 加载数据和标签
# for idx, (data_file, label_file) in enumerate(zip(hl_data_files, hl_label_files)):
#     data = np.load(data_file)
#     labels = np.load(label_file)
#     split_and_save_data(data, labels, "double_deaf", idx + 1)


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

# # 独热编码器
# label_binarizer = LabelBinarizer()
# label_binarizer.fit(labels_list)

# # 读取并处理所有 .set 文件
# for data_dir, label in zip(data_dirs, labels_list):
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
            
#             # 标准化数据
#             scaler = StandardScaler()
#             n_epochs, n_channels, n_samples = selected_eeg_data.shape
#             selected_eeg_data = selected_eeg_data.reshape(n_epochs * n_channels, n_samples)
#             selected_eeg_data = scaler.fit_transform(selected_eeg_data)
#             selected_eeg_data = selected_eeg_data.reshape(n_epochs, n_channels, n_samples)
            
#             # 相邻epoch拼接
#             new_epochs = []
#             for i in range(selected_eeg_data.shape[0] - 1):
#                 first_half = selected_eeg_data[i, :, -1250:]  # 取前一个epoch的后半段
#                 second_half = selected_eeg_data[i + 1, :, :1250]  # 取后一个epoch的前半段
#                 new_epoch = np.concatenate((first_half, second_half), axis=1)  # 拼接成新的epoch
#                 new_epochs.append(new_epoch)
            
#             new_epochs = np.array(new_epochs)
            
#             # 将新生成的epochs添加到原始数据的后面
#             final_combined_data = np.concatenate((selected_eeg_data, new_epochs), axis=0)
            
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
#     total_data = len(all_data)
#     batch_size = total_data // 10
#     for i in range(10):
#         start_index = i * batch_size
#         end_index = (i + 1) * batch_size if i < 9 else total_data
#         np.save(os.path.join(save_dir, f"{label}_data_part{i+1}.npy"), all_data[start_index:end_index])
#         np.save(os.path.join(save_dir, f"{label}_labels_part{i+1}.npy"), all_labels_encoded[start_index:end_index])

