from keras.layers import Input, Dense, LSTM, Conv1D, Dropout, Bidirectional, Multiply, BatchNormalization
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import *

import pandas as pd
import numpy as np

SINGLE_ATTENTION_VECTOR = False
INPUT_DIMS = 4
TIME_STEPS = 20
lstm_units = 64
epoch = 30
dropout = 0.3


def attention_3d_block(inputs):
    input_dim = int(inputs.shape[2])
    a = inputs
    a = Dense(input_dim, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((1, 2), name='attention_vec')(a)

    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul


def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, :])
    TrainX = np.array(dataX)
    Train_Y = np.array(dataY)

    return TrainX, Train_Y


# 多维归一化  返回数据和最大最小值
def NormalizeMult(data):
    # normalize 用于反归一化
    normalize = np.arange(2 * data.shape[1], dtype='float64')

    normalize = normalize.reshape(data.shape[1], 2)
    for i in range(0, data.shape[1]):
        # 第i列
        list = data[:, i]
        list_low, list_high = np.percentile(list, [0, 100])
        normalize[i, 0] = list_low
        normalize[i, 1] = list_high
        delta = list_high - list_low
        if delta != 0:
            # 第j行
            for j in range(0, data.shape[0]):
                data[j, i] = (data[j, i] - list_low) / delta
    return data, normalize


# 多维反归一化
def FNormalizeMult(data, normalize):
    data = np.array(data)
    for i in range(0, data.shape[1]):
        list_low = normalize[i, 0]
        list_high = normalize[i, 1]
        delta = list_high - list_low
        if delta != 0:
            # 第j行
            for j in range(0, data.shape[0]):
                data[j, i] = data[j, i] * delta + list_low

    return data


def mean_squared_error(test_Y, pred_Y):
    # 将输入转换为 NumPy 数组以支持数组运算
    test_Y = np.array(test_Y)
    pred_Y = np.array(pred_Y)

    # 计算均方误差
    mse = np.mean((test_Y - pred_Y) ** 2)
    return mse


def attention_model_with_norm():
    inputs = Input(shape=(TIME_STEPS, INPUT_DIMS))
    x = Conv1D(filters=64, kernel_size=1, activation='relu')(inputs)
    x = BatchNormalization()(x)  # Added Batch Normalization
    x = Dropout(dropout)(x)
    lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
    lstm_out = BatchNormalization()(lstm_out)  # Added Batch Normalization
    lstm_out = Dropout(dropout)(lstm_out)
    attention_mul = attention_3d_block(lstm_out)
    attention_mul = Flatten()(attention_mul)
    output = Dense(1, activation='sigmoid')(attention_mul)
    model = Model(inputs=[inputs], outputs=output)
    return model


def create_dataset_tomorrow(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, :])
    TrainX = np.array(dataX)
    Train_Y = np.array(dataY)
    return TrainX, Train_Y


def attention_model():
    inputs = Input(shape=(TIME_STEPS, INPUT_DIMS))
    x = Conv1D(filters=64, kernel_size=1, activation='relu')(inputs)
    x = Dropout(dropout)(x)
    lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
    lstm_out = Dropout(dropout)(lstm_out)
    attention_mul = attention_3d_block(lstm_out)
    attention_mul = Flatten()(attention_mul)
    output = Dense(4, activation='linear')(attention_mul)  # 修改了输出
    model = Model(inputs=[inputs], outputs=output)
    return model


def calculate_metrics(test_Y, pred_Y):
    # MAE
    mae = mean_absolute_error(test_Y, pred_Y)

    # MSE
    mse = mean_squared_error(test_Y, pred_Y)

    # 涨跌准确率
    test_diff = np.diff(test_Y[:, 1])  # 计算真实值的涨跌
    pred_diff = np.diff(pred_Y[:, 1])  # 计算预测值的涨跌
    test_sign = np.sign(test_diff)  # 获取真实值涨跌的符号
    pred_sign = np.sign(pred_diff)  # 获取预测值涨跌的符号
    accuracy = np.mean(test_sign == pred_sign) * 100  # 计算涨跌准确率

    return mae, mse, accuracy


# 加载数据
data = pd.read_csv("./data.csv")
data = data[['open', 'close', 'high', 'low']]

# 归一化
data = np.array(data)
data, normalize = NormalizeMult(data)
close_column = data[:, 1].reshape(len(data), 1)

train_X, train_Y = create_dataset_tomorrow(data, TIME_STEPS)
train_X, test_X, train_Y, test_Y = train_test_split(train_X, train_Y, test_size=0.2, random_state=42)

m = attention_model()

m.summary()
m.compile(optimizer='adam', loss='mse')
# m.save("./model.h5")
# np.save("normalize.npy", normalize)
m.fit([train_X], train_Y, epochs=epoch, batch_size=64, validation_split=0.1)
# 使用测试集进行预测
pred_Y = m.predict(test_X)

# 使用示例
mae, mse, accuracy = calculate_metrics(test_Y, pred_Y)
print(f"MAE: {mae}")
print(f"MSE: {mse}")
# print(f"MAPE: {mape}%")
print(f"涨跌准确率: {accuracy}%")
