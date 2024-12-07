# CNN-BiLSTM-Attention-K-Line-Prediction

## 环境

经测试python 3.7没问题

```shell
pipreqs . --encoding=utf-8
```

```
pip install -r requirements.txt  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### TF1环境

requirements.txt如下，基于python3.7

```
Keras==2.3.1
numpy==1.16.5
pandas==0.25.1
scikit_learn==1.0.2
tensorflow==1.15.0
tensorflow_intel==2.11.0
matplotlib~=3.0.3
```

### TF2环境

基于python3.12

```
keras==3.5.0
numpy==2.1.3
pandas==2.2.3
scikit_learn==1.5.2
tensorflow==2.18.0
tensorflow_intel==2.18.0
tensorflow_intel==2.17.0
```

## 项目文件介绍

- `note.ipynb`：TF1下的所有功能实现
- `predict_T+1_tf1.py`：TF1下通过前t天数据预测T+1天开盘、收盘、高点、低点
- `predict_T+1_tf2.py`：TF2下通过前t天数据预测T+1天开盘、收盘、高点、低点

## 数据集要求

至少包含`['open', 'close', 'high', 'low']`这四列

## 功能-TF1

环境安装完成后打开`note.ipynb`即可进行代码运行

文件中对不同的代码模块进行了划分，方便二次开发时进行修改（如切换模型，切换数据集，切换数据集构造算法等

### 内置模型

内置如下五个模型，具体实现参考`note.ipynb`文件

```python
def attention_model():
# CNN-BiLSTM-Attention实现

def attention_model_with_norm():
# 在上一个模型基础上添加归一化层

def lstm(model_type):
    inputs = Input(shape=(TIME_STEPS, INPUT_DIMS))
    
    if model_type == 1:
        # single-layer LSTM
    
    if model_type == 2:
        # multi-layer LSTM
    
    if model_type == 3:
        # BiLSTM
    
    return model
```

### 功能实现

`note.ipynb`中实现了如下功能：

#### 通过第t+1天开盘、最高、最低预测收盘

在一段连续的时间内，某只股票会产生一个收盘价格的序列。我们记这段连续的价格序列为：

$$
C_T=c_1,c_2,...,c_T
$$
其中 $c_t$ 代表该股票在第 $t$ 个交易日的收盘价格。

此外我们定义开盘价序列 $O_T=o_1,o_2,...,o_T$ ，最高价序列 $H_T=h_1,h_2,...,h_T$ ，最低价序列 $L_T=l_1,l_2,...,l_T$

基于这四个序列数据，训练一个模型 $f_M$，以第 $T+1$ 天的开盘价 $o_{T+1}$、最高点 $h_{T+1}$、最低点 $l_{T+1}$ 作为输入，预测该天的收盘价格 $c_{T+1}$

即：
$$
\{O_T,H_T,L_T,C_T\}\to f_M
$$

$$
c_{T+1}=f_M(o_{T+1},h_{T+1},l_{T+1})
$$

#### 通过前t天数据预测T+1天开盘、收盘、高点、低点

考虑更现实一点的，训练一个模型 $f_N$，以前 $t$ 天数据为输入，预测第二天的开盘价 $o_{T+1}$、最高点 $h_{T+1}$、最低点 $l_{T+1}$ 、收盘价 $c_{T+1}$
$$
\{(O_{i,i+t},H_{i,i+t},,L_{i,i+t},C_{i,i+t}),(O_{i+t+1},H_{i+t+1},L_{i+t+1},C_{i+t+1})\}\to f_N,i\in (0,T-t-1)
$$

$$
(c_{T+1},o_{T+1},h_{T+1},l_{T+1})=f_N(O_{T-t,T},H_{T-t,T},,L_{T-t,T},C_{T-t,T})
$$

#### 通过前t天数据预测T+7天开盘、收盘、高点、低点

同理，同时也可以二次开发为预测T+n天的价格数据



### 不同功能实现的关键点

#### 数据集构造

针对不同场景构造不同数据集：

```python
def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, :])
    TrainX = np.array(dataX)
    Train_Y = np.array(dataY)

    return TrainX, Train_Y
train_X, _ = create_dataset(data, TIME_STEPS)
_, train_Y = create_dataset(close_column, TIME_STEPS)


def create_dataset_tomorrow(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, :])
    TrainX = np.array(dataX)
    Train_Y = np.array(dataY)
    return TrainX, Train_Y
train_X, train_Y = create_dataset_tomorrow(data, TIME_STEPS)

def create_dataset_7days(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 7):  # 预测7天后的数据
        a = dataset[i:(i + look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back + 6, :])  # 目标是7天后的数据
    TrainX = np.array(dataX)
    Train_Y = np.array(dataY)
    return TrainX, Train_Y
train_X, train_Y = create_dataset_7days(data, TIME_STEPS)
```

代码实现思路上基本类似，不再赘述

#### 神经网络实现

这个也没什么好说的其实，代码写的很明白。





## 特别致谢

本项目基于该项目进行二次开发：https://github.com/PatientEz/CNN-BiLSTM-Attention-Time-Series-Prediction_Keras
