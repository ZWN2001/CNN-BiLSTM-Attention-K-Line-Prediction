import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, Conv1D, Dropout, Bidirectional, Multiply, BatchNormalization, \
    Flatten, Lambda, Permute, RepeatVector
from tensorflow.keras.models import Model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import matplotlib.pyplot as plt

SINGLE_ATTENTION_VECTOR = False
INPUT_DIMS = 4
TIME_STEPS = 20
global lstm_units
global conv_filters
global epoch
global dropout
global temperature
global alpha


def attention_3d_block(inputs):
    input_dim = int(inputs.shape[2])
    a = inputs
    a = Dense(input_dim, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: tf.reduce_mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((1, 2), name='attention_vec')(a)

    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul


def NormalizeMult(data):
    normalize = np.arange(2 * data.shape[1], dtype='float64')

    normalize = normalize.reshape(data.shape[1], 2)
    for i in range(0, data.shape[1]):
        list = data[:, i]
        list_low, list_high = np.percentile(list, [0, 100])
        normalize[i, 0] = list_low
        normalize[i, 1] = list_high
        delta = list_high - list_low
        if delta != 0:
            for j in range(0, data.shape[0]):
                data[j, i] = (data[j, i] - list_low) / delta
    return data, normalize


def FNormalizeMult(data, normalize):
    data = np.array(data)
    for i in range(0, data.shape[1]):
        listlow = normalize[i, 0]
        listhigh = normalize[i, 1]
        delta = listhigh - listlow
        if delta != 0:
            #第j行
            for j in range(0, data.shape[0]):
                data[j, i] = data[j, i] * delta + listlow

    return data


def calculate_metrics(test_Y, pred_Y):
    mae = mean_absolute_error(test_Y, pred_Y)
    mse = mean_squared_error(test_Y, pred_Y)

    test_diff = np.diff(test_Y[:, 1])
    pred_diff = np.diff(pred_Y[:, 1])
    test_sign = np.sign(test_diff)
    pred_sign = np.sign(pred_diff)
    accuracy = np.mean(test_sign == pred_sign) * 100

    return mae, mse, accuracy


def create_dataset_tomorrow(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, :])
    TrainX = np.array(dataX)
    Train_Y = np.array(dataY)
    return TrainX, Train_Y


class Distiller(tf.keras.Model):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.teacher = teacher
        self.student = student

    def compile(self, optimizer, metrics, student_loss_fn, distillation_loss_fn, alpha=0.1, temperature=3):
        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):
        x, y = data

        # 教师模型输出
        teacher_predictions = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            # 学生模型输出
            student_predictions = self.student(x, training=True)

            # 计算学生的损失
            student_loss = self.student_loss_fn(y, student_predictions)

            # 计算蒸馏损失（使用温度处理的softmax）
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                tf.nn.softmax(student_predictions / self.temperature, axis=1),
            )

            # 总损失
            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        # 计算梯度并更新学生模型的权重
        gradients = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.student.trainable_variables))

        # 更新metrics
        self.compiled_metrics.update_state(y, student_predictions)

        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss, "distillation_loss": distillation_loss})
        return results

    def call(self, inputs):
        # 在这里，我们定义了Distiller的call方法，即通过学生模型进行前向传播
        student_predictions = self.student(inputs, training=False)
        return student_predictions

    def test_step(self, data):
        x, y = data
        y_prediction = self.student(x, training=False)
        student_loss = self.student_loss_fn(y, y_prediction)
        self.compiled_metrics.update_state(y, y_prediction)

        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        return results


def teacher_model():
    inputs = Input(shape=(TIME_STEPS, INPUT_DIMS))
    x = Conv1D(filters=conv_filters, kernel_size=1, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)
    lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
    lstm_out = BatchNormalization()(lstm_out)
    lstm_out = Dropout(dropout)(lstm_out)
    attention_mul = attention_3d_block(lstm_out)
    attention_mul = Flatten()(attention_mul)
    output = Dense(4, activation='linear')(attention_mul)  # 教师模型输出4个特征
    model = Model(inputs=[inputs], outputs=output)
    return model


# 定义学生模型
def student_model():
    inputs = Input(shape=(TIME_STEPS, INPUT_DIMS))
    x = Conv1D(filters=conv_filters // 2, kernel_size=1, activation='relu')(inputs)  # 较少的过滤器
    x = Dropout(dropout)(x)
    lstm_out = Bidirectional(LSTM(lstm_units // 3, return_sequences=True))(x)  # 更少的LSTM单元
    lstm_out = Dropout(dropout)(lstm_out)
    # attention_mul = attention_3d_block(lstm_out)
    attention_mul = Flatten()(lstm_out)
    output = Dense(4, activation='linear')(attention_mul)  # 学生模型输出4个特征
    model = Model(inputs=[inputs], outputs=output)
    return model


def do_test(test_name, test_Y, folder):
    # 创建教师模型和学生模型
    teacher = teacher_model()
    student = student_model()

    # 训练教师模型
    teacher.compile(optimizer='adam', loss='mse')
    teacher.fit([train_X], train_Y, epochs=epoch, batch_size=64, validation_split=0.1)

    # 创建蒸馏模型并训练学生模型
    distiller = Distiller(student=student, teacher=teacher)
    distiller.compile(
        optimizer='adam',
        metrics=['mae'],
        student_loss_fn=tf.keras.losses.MeanSquaredError(),
        distillation_loss_fn=tf.keras.losses.KLDivergence(),
        alpha=alpha,
        temperature=temperature
    )

    # 训练蒸馏模型
    distiller.fit(train_X, train_Y, epochs=epoch, batch_size=64, validation_split=0.1)

    # 预测
    pred_Y = distiller.predict(test_X)

    # 反归一化
    test_Y = FNormalizeMult(test_Y, normalize)
    pred_Y = FNormalizeMult(pred_Y, normalize)

    # 评估
    mae, mse, accuracy = calculate_metrics(test_Y, pred_Y)
    print('MAE:', mae)
    print('MSE:', mse)
    print('Accuracy:', accuracy)

    # 画图
    plt.figure(figsize=(10, 6))
    plt.plot(test_Y[:100, 1], label='True Values')
    plt.plot(pred_Y[:100, 1], label='Predicted Values')
    plt.title('Denormalized True vs Predicted Values')
    plt.xlabel('Date')
    plt.ylabel('Close')
    plt.legend()
    plt.savefig(f'{folder}/{test_name}-{mse}-{accuracy}.png')

    return mae, mse, accuracy


def draw_result(targets, result, name):
    # 根据result画图
    plt.figure(figsize=(10, 6))
    plt.plot(targets, [r[0] for r in result], label='MAE')
    plt.title(f'MAE vs {name}')
    plt.xlabel(name)
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(f'{name}/{name}-MAE.png')

    plt.figure(figsize=(10, 6))
    plt.plot(targets, [r[1] for r in result], label='MSE')
    plt.title(f'MSE vs {name}')
    plt.xlabel(name)
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(f'{name}/{name}-MSE.png')

    plt.figure(figsize=(10, 6))
    plt.plot(targets, [r[2] for r in result], label='Accuracy')
    plt.title(f'Accuracy vs {name}')
    plt.xlabel(name)
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(f'{name}/{name}-Accuracy.png')


def test_LSTM_units(test_Y):
    global lstm_units
    targets = [32, 48, 64, 80, 96, 112, 128]
    result = []
    if not os.path.exists('lstm_units'):
        os.mkdir('lstm_units')
    for target in targets:
        lstm_units = target
        mae, mse, accuracy = do_test(f'lstm_units-{target}', test_Y, 'lstm_units')
        result.append([mae, mse, accuracy])

    draw_result(targets, result, 'lstm_units')


def test_conv_filters(test_Y):
    global conv_filters
    targets = [16, 32, 48, 64, 80, 96, 112, 128]
    result = []
    if not os.path.exists('conv_filters'):
        os.mkdir('conv_filters')
    for target in targets:
        conv_filters = target
        mae, mse, accuracy = do_test(f'conv_filters-{target}', test_Y, 'conv_filters')
        result.append([mae, mse, accuracy])

    draw_result(targets, result, 'conv_filters')


def test_epoch(test_Y):
    global epoch
    targets = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    result = []
    if not os.path.exists('epoch'):
        os.mkdir('epoch')
    for target in targets:
        epoch = target
        mae, mse, accuracy = do_test(f'epoch-{target}', test_Y, 'epoch')
        result.append([mae, mse, accuracy])

    draw_result(targets, result, 'epoch')


def test_dropout(test_Y):
    global dropout
    targets = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    result = []
    if not os.path.exists('dropout'):
        os.mkdir('dropout')
    for target in targets:
        dropout = target
        mae, mse, accuracy = do_test(f'dropout-{target}', test_Y, 'dropout')
        result.append([mae, mse, accuracy])

    draw_result(targets, result, 'dropout')


def test_temperature(test_Y):
    global temperature
    targets = [5, 10, 15, 20, 25, 30]
    result = []
    if not os.path.exists('temperature'):
        os.mkdir('temperature')
    for target in targets:
        temperature = target
        mae, mse, accuracy = do_test(f'temperature-{target}', test_Y, 'temperature')
        result.append([mae, mse, accuracy])

    draw_result(targets, result, 'temperature')


def test_alpha(test_Y):
    global alpha
    targets = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    result = []
    if not os.path.exists('alpha'):
        os.mkdir('alpha')
    for target in targets:
        alpha = target
        mae, mse, accuracy = do_test(f'alpha-{target}', test_Y, 'alpha')
        result.append([mae, mse, accuracy])

    draw_result(targets, result, 'alpha')


def set_default():
    global lstm_units
    global conv_filters
    global epoch
    global dropout
    global temperature
    global alpha
    lstm_units = 64
    conv_filters = 64
    epoch = 30
    dropout = 0.4
    temperature = 10
    alpha = 0.1


if __name__ == '__main__':
    # 读取数据
    data = pd.read_csv('data.csv')
    data = data[['open', 'high', 'low', 'close']]
    data = np.array(data)

    # 归一化
    data, normalize = NormalizeMult(data)

    # 创建数据集
    look_back = TIME_STEPS
    trainX, trainY = create_dataset_tomorrow(data, look_back)

    # 划分训练集和测试集
    train_X, test_X, train_Y, test_Y = train_test_split(trainX, trainY, test_size=0.2, random_state=42)

    # set_default()
    # test_LSTM_units(test_Y)
    #
    # set_default()
    # test_conv_filters(test_Y)
    #
    # set_default()
    # test_epoch(test_Y)
    #
    # set_default()
    # test_dropout(test_Y)

    set_default()
    test_temperature(test_Y)

    # set_default()
    # test_alpha(test_Y)
