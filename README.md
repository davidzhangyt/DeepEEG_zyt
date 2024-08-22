# dpeeg

Dpeeg provides a complete workflow for deep learning decoding EEG tasks, including basic datasets (datasets
can be easily customized), basic network models, model training, rich experiments, and detailed experimental
result storage.

Each module in dpeeg is decoupled as much as possible to facilitate the use of separate modules.

# Usage

Installation dependencies are not written yet, please install as follows:

1. Create a new virtual environment named "dpeeg" with python3.10 using Anaconda3 and activate it：

```Shell
conda create --name dpeeg python=3.10
conda activate dpeeg
```

2. Install environment dependencies

```Shell
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c anaconda ipython ipykernel pandas scikit-learn
conda install -c conda-forge torchinfo mne-base tensorboard torchmetrics seaborn
pip install moabb==0.5.0
pip install dpeeg
```

# 使用方法

上面的基本环境可以直接配，pytorch=11.8那部分可以去pytorch的官网去下载对应的版本，直接复制就能用。https://pytorch.org/get-started/previous-versions/

dpeeg-main是一个github的开源项目，和那个metaBCI差不多，里面有更丰富的模型

dataset_EEG和train是我写的一些代码，可以直接拿来跑，在每个文件的上方均有注释这个代码是做什么的，并且代码内部也都写好了注释。数据处理很吃运行内存，这里我把我处理的一些文件夹删掉了，就只抽出来了这两个文件夹，结合着dpeeg-main，修改一下数据的一些路径。

我大概里面包含两种内容，一个是原始的eeg信号作为输入的训练，一个是psd结合原始数据作为输入进行的训练，我在train里的开头注释中都给出了对应复现的论文，后续遇到问题可以直接把报错问GPT，也可以顺便把代码一并给它，大概率它会给出解决方案。
