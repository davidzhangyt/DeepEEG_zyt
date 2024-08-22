## 随机森林

# import os
# import numpy as np
# import pywt
# from sklearn.model_selection import cross_val_score, KFold
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import make_pipeline
# import joblib

# # 设置数据路径
# data_path = "D:/EEG-DL-master/dpeeg-main/dpeeg-main/data17"
# folders = {
#     "Cr37": "Cr-zzh-026",
#     "SSD63": "SSD-zyx-037",
#     "HL96": "HLygh-021"
# }

# # 设置模型保存路径
# model_save_path = "D:/EEG-DL-master/dpeeg-main/dpeeg-main/models"
# os.makedirs(model_save_path, exist_ok=True)

# # 加载数据函数
# def load_data(file_indices, folder, file_prefix):
#     data_list, label_list = [], []
    
#     for idx in file_indices:
#         data_file = os.path.join(data_path, folder, 'noise', f'{file_prefix}.cdt8C2_ICA_part{idx}_data.npy')
#         label_file = os.path.join(data_path, folder, 'noise', f'{file_prefix}.cdt8C2_ICA_part{idx}_labels.npy')
        
#         if os.path.exists(data_file) and os.path.exists(label_file):
#             print(f"Loading data from: {data_file}")
#             data_list.append(np.load(data_file))
#             label_list.append(np.load(label_file))
#         else:
#             print(f"File not found: {data_file} or {label_file}")
    
#     if not data_list or not label_list:
#         raise FileNotFoundError(f"Files for {file_prefix} in {folder} not found or incomplete.")
    
#     data = np.concatenate(data_list, axis=0)
#     labels = np.concatenate(label_list, axis=0)
    
#     return data, labels

# # 小波变换提取特征
# def extract_dwt_features(data, wavelet='db4', level=5):
#     features = []
#     for i, epoch in enumerate(data):
#         coeffs = pywt.wavedec(epoch, wavelet, level=level, axis=-1)
#         coeffs_flat = np.concatenate(coeffs, axis=-1)
#         features.append(coeffs_flat.flatten())
#         if i % 100 == 0:
#             print(f"Processed {i+1}/{len(data)} epochs for current file.")
#     return np.array(features)

# # 选择数据集文件索引
# file_indices = range(1, 11)  # 使用所有10个文件

# # 加载并处理数据
# all_data, all_labels = [], []

# for folder, file_prefix in folders.items():
#     print(f"Processing folder: {folder}")
#     data, labels = load_data(file_indices, folder, file_prefix)
#     features = extract_dwt_features(data)
#     all_data.append(features)
#     all_labels.append(labels)
#     print(f"Completed feature extraction for folder: {folder}")

# X = np.concatenate(all_data, axis=0)
# y = np.concatenate(all_labels, axis=0)
# print(f"Completed loading and processing all data. Total samples: {X.shape[0]}")

# # 构建随机森林模型并使用标准化
# model = make_pipeline(StandardScaler(), RandomForestClassifier())

# # 10折交叉验证
# kf = KFold(n_splits=5, shuffle=True, random_state=42)

# # 训练和评估模型
# print("Starting cross-validation for RandomForest model")
# for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
#     X_train, X_test = X[train_idx], X[test_idx]
#     y_train, y_test = y[train_idx], y[test_idx]
#     model.fit(X_train, y_train)
#     score = model.score(X_test, y_test)
#     print(f"Fold {fold + 1}: Accuracy = {score:.4f}")
    
# scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
# print(f"RandomForest: Mean Accuracy = {scores.mean():.4f}, Std = {scores.std():.4f}")

# # 保存模型
# model_path = os.path.join(model_save_path, "RandomForest_model.pkl")
# joblib.dump(model, model_path)
# print(f"Model saved to {model_path}")

'''
这个部分是使用单个机器学习算法进行训练的，方法与machine_learning.py一致，只是这个是单一的而已，我们上面注释掉的代码使用随机森林，
这里使用的是逻辑回归。把上面的部分取消只需要全选上面的代码，然后'ctrl+/'即可
'''
##逻辑回归
import os
import numpy as np
import pywt
from sklearn.model_selection import cross_val_score, KFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib
from sklearn.linear_model import LogisticRegression
# 设置数据路径
data_path = "D:/EEG-DL-master/dpeeg-main/dpeeg-main/data17"
folders = {
    "Cr37": "Cr",
    "SSD63": "SSD",
    "HL96": "HL"
}

# 设置模型保存路径
model_save_path = "D:/EEG-DL-master/dpeeg-main/dpeeg-main/models"
os.makedirs(model_save_path, exist_ok=True)

# 加载数据函数
def load_data(file_indices, folder, file_prefix):
    data_list, label_list = [], []
    
    for idx in file_indices:
        data_file = os.path.join(data_path, folder, 'noise', f'{file_prefix}.cdt8C2_ICA_part{idx}_data.npy')
        label_file = os.path.join(data_path, folder, 'noise', f'{file_prefix}.cdt8C2_ICA_part{idx}_labels.npy')
        
        if os.path.exists(data_file) and os.path.exists(label_file):
            print(f"Loading data from: {data_file}")
            data_list.append(np.load(data_file))
            label_list.append(np.load(label_file))
        else:
            print(f"File not found: {data_file} or {label_file}")
    
    if not data_list or not label_list:
        raise FileNotFoundError(f"Files for {file_prefix} in {folder} not found or incomplete.")
    
    data = np.concatenate(data_list, axis=0)
    labels = np.concatenate(label_list, axis=0)
    
    return data, labels

# 小波变换提取特征
def extract_dwt_features(data, wavelet='db4', level=5):
    features = []
    for i, epoch in enumerate(data):
        coeffs = pywt.wavedec(epoch, wavelet, level=level, axis=-1)
        coeffs_flat = np.concatenate(coeffs, axis=-1)
        features.append(coeffs_flat.flatten())
        if i % 100 == 0:
            print(f"Processed {i+1}/{len(data)} epochs for current file.")
    return np.array(features)

# 选择数据集文件索引
file_indices = range(1, 11)  # 使用所有10个文件

# 加载并处理数据
all_data, all_labels = [], []

for folder, file_prefix in folders.items():
    print(f"Processing folder: {folder}")
    data, labels = load_data(file_indices, folder, file_prefix)
    features = extract_dwt_features(data)
    all_data.append(features)
    all_labels.append(labels)
    print(f"Completed feature extraction for folder: {folder}")

X = np.concatenate(all_data, axis=0)
y = np.concatenate(all_labels, axis=0)
print(f"Completed loading and processing all data. Total samples: {X.shape[0]}")

# 构建逻辑回归模型并使用标准化
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
# 5折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 训练和评估模型
print("Starting cross-validation for tree model")
for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"Fold {fold + 1}: Accuracy = {score:.4f}")
    
scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
print(f"逻辑回归: Mean Accuracy = {scores.mean():.4f}, Std = {scores.std():.4f}")

# 保存模型
model_path = os.path.join(model_save_path, "tree_model.pkl")
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")
