'''
用来划分训练集和测试集的，节省训练时的压力，不过我比较推荐前面data_preprocessed.py既然已经每个标签都划分了10个npy文件，干脆前8个当训练集，
后两个当测试集就行，当然电脑有条件也可以用这个方法划分训练集和测试集，不过很吃性能，我32G运行内存都顶不住。
'''
import os
import numpy as np
from sklearn.model_selection import train_test_split

class EEGDataHandler:
    def __init__(self, data_dir, temp_dir, batch_size=32, n_splits=5,n_files=10, skip_step=2):
        self.data_dir = data_dir
        self.temp_dir = temp_dir
        self.batch_size = batch_size
        self.n_splits = n_splits
        self.n_files = n_files  # 每次处理的文件数
        self.skip_step = skip_step  # 跳跃选择的步长

    def load_and_reshape_batch(self, file_path):
        data = np.load(file_path)
        X = data['X'].reshape(-1, 116, 2500)  # 将 (10, 170, 116, 2500) 变为 (1700, 116, 2500)
        y = np.repeat(data['y'], 170)  # 对应的标签重复 170 次
        return X, y
    
    def save_batches(self):
        temp_index = 3
        file_names = [f for f in os.listdir(self.data_dir) if f.startswith('batch') and f.endswith('.npz')]
        
        for i in range(3, len(file_names), self.n_files * self.skip_step):
            X_list, y_list = [], []
            batch_files = file_names[i:i + self.n_files * self.skip_step:self.skip_step]  # 跳跃选择文件
            
            for file_name in batch_files:
                file_path = os.path.join(self.data_dir, file_name)
                X, y = self.load_and_reshape_batch(file_path)
                print(f'Loaded {file_name}: X shape {X.shape}, y shape {y.shape}')
                
                X_list.append(X)
                y_list.append(y)
            
            # 合并当前批次的数据
            X = np.concatenate(X_list, axis=0)
            y = np.concatenate(y_list, axis=0)
            
            # 划分训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # 保存当前批次的训练集和测试集
            np.savez_compressed(os.path.join(self.temp_dir, f'train_{temp_index}.npz'), X=X_train, y=y_train)
            np.savez_compressed(os.path.join(self.temp_dir, f'test_{temp_index}.npz'), X=X_test, y=y_test)
            
            temp_index += 1
            print(f'Saved batch {temp_index} data: X_train shape {X_train.shape}, y_train shape {y_train.shape}, X_test shape {X_test.shape}, y_test shape {y_test.shape}')

# # 处理最后3个文件
#         X_list, y_list = [], []
#         batch_files = file_names[-3:]
        
#         for file_name in batch_files:
#             file_path = os.path.join(self.data_dir, file_name)
#             X, y = self.load_and_reshape_batch(file_path)
#             print(f'Loaded {file_name}: X shape {X.shape}, y shape {y.shape}')
            
#             X_list.append(X)
#             y_list.append(y)
        
#         # 合并当前批次的数据
#         X = np.concatenate(X_list, axis=0)
#         y = np.concatenate(y_list, axis=0)
        
#         # 划分训练集和测试集
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
#         # 保存当前批次的训练集和测试集
#         np.savez_compressed(os.path.join(self.temp_dir, f'train_{temp_index}.npz'), X=X_train, y=y_train)
#         np.savez_compressed(os.path.join(self.temp_dir, f'test_{temp_index}.npz'), X=X_test, y=y_test)
        
#         print(f'Saved batch {temp_index} data: X_train shape {X_train.shape}, y_train shape {y_train.shape}, X_test shape {X_test.shape}, y_test shape {y_test.shape}')



# 使用示例
data_dir = "D:\\EEG-DL-master\\dpeeg-main\\dpeeg-main\\preprocessed_data"
temp_dir = "D:\\EEG-DL-master\\dpeeg-main\\dpeeg-main\\temp_data"
batch_size = 32
n_splits = 5
n_files = 10  # 每次处理10个文件
skip_step = 4  # 跳跃选择文件
# 创建临时文件目录
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

# 初始化数据处理器
data_handler = EEGDataHandler(data_dir, temp_dir, batch_size, n_splits, n_files, skip_step)

# 准备数据并保存中间文件
data_handler.save_batches()
