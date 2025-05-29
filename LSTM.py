import argparse
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.nn.utils import weight_norm
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
import seaborn as sns
from selectPolicy import modelPolicy
from plot_utils import plot_single_model_violin,plot_results
'''绘图基本设置'''
plt.rcParams['font.sans-serif'] = ['SimHei']  # 'Times New Roman' 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
font1 = {'weight': 'normal', 'size': 14}
# 随机数种子
np.random.seed(0)

class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.

    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)
        self.std[self.std == 0] = 1  # 防止标准差为零

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        if data.shape[-1] != mean.shape[-1]:
            mean = mean[-1:]
            std = std[-1:]
        return (data * std) + mean


def plot_loss_data(data):
    # 使用Matplotlib绘制线图
    plt.figure()
    plt.figure(figsize=(10, 5))
    plt.plot(data, marker='o')

    # 添加标题
    plt.title("loss results Plot")

    # 显示图例
    plt.legend(["Loss"])
    plt.show()


class TimeSeriesDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence, label = self.sequences[index]
        return torch.Tensor(sequence), torch.Tensor(label)


def create_inout_sequences(input_data, tw, pre_len, config):
    # 创建时间序列数据专用的数据分割器
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i:i + tw]
        if (i + tw + pre_len) > len(input_data):
            break
        if config.feature == 'MS':
            train_label = input_data[:, -1:][i + tw:i + tw + pre_len]
        else:
            train_label = input_data[i + tw:i + tw + pre_len]
        inout_seq.append((train_seq, train_label))
    return inout_seq


def calculate_mae(y_true, y_pred):
    # 平均绝对误差
    mae = np.mean(np.abs(y_true - y_pred))
    return mae


def calculate_rmse(y_true, y_pred):
    # 平均绝对误差
    rmse = np.sqrt(np.mean(np.power((y_pred - y_true), 2)))
    return rmse

def calculate_mape(y_true, y_pred):
    # 平均绝对误差
    mape = np.sum(np.abs((y_true - y_pred) / y_true))
    return 100 * mape / len(y_true)

def plot(y, pred, ind):
    # plot
    plt.figure(figsize=(6, 2))
    plt.plot(pred, color='blue', label='Actual value')
    plt.plot(y, color='red', label='predict value')
    plt.xlabel('Timestep',size=12)
    # plt.title('第' + str(ind) + '变量的预测示意图')
    plt.grid(True)
    plt.tick_params(labelsize=14)
    plt.legend(loc='lower right', ncol=4)
    plt.show()

def get_mape(x, y):
    """
    :param x: true value
    :param y: pred value
    :return: mape
    """
    return np.mean(np.abs((x - y) / x))


def get_r2(y, pred):
    return r2_score(y, pred)


def get_mae(y, pred):
    return mean_absolute_error(y, pred)


def get_mse(y, pred):
    return mean_squared_error(y, pred)


def get_rmse(y, pred):
    return np.sqrt(mean_squared_error(y, pred))


def create_dataloader(config, device, div):
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>创建数据加载器<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    df = pd.read_excel(config.data_path, parse_dates=['时间'])  # 解析时间列
    time_column = df['时间'].dt.strftime('%H:%M')  # 提取时间列并格式化为小时:分钟
    pre_len = config.pre_len  # 预测未来数据的长度
    train_window = config.window_size  # 观测窗口

    # 将特征列移到末尾
    target_data = df[config.target]
    df = df.drop(config.target, axis=1)
    df = pd.concat((df, target_data), axis=1)

    cols_data = df.columns[1:]
    df_data = df[cols_data]

    # 数据预处理
    true_data = df_data.values

    # 定义标准化优化器
    scaler = StandardScaler()
    scaler.fit(true_data)

    train_data = true_data[:int(div * len(true_data))]
    valid_data = true_data[int(0.7 * len(true_data)):int(0.9 * len(true_data))]
    test_data = true_data[int(div * len(true_data)):]
    print("训练集尺寸:", len(train_data), "测试集尺寸:", len(test_data), "验证集尺寸:", len(valid_data))

    # 进行标准化处理
    train_data_normalized = scaler.transform(train_data)
    test_data_normalized = scaler.transform(test_data)
    valid_data_normalized = scaler.transform(valid_data)

    # 转化为深度学习模型需要的类型Tensor
    train_data_normalized = torch.FloatTensor(train_data_normalized).to(device)
    test_data_normalized = torch.FloatTensor(test_data_normalized).to(device)
    valid_data_normalized = torch.FloatTensor(valid_data_normalized).to(device)

    # 定义训练器的的输入
    train_inout_seq = create_inout_sequences(train_data_normalized, train_window, pre_len, config)
    test_inout_seq = create_inout_sequences(test_data_normalized, train_window, pre_len, config)
    valid_inout_seq = create_inout_sequences(valid_data_normalized, train_window, pre_len, config)

    # 创建数据集
    train_dataset = TimeSeriesDataset(train_inout_seq)
    test_dataset = TimeSeriesDataset(test_inout_seq)
    valid_dataset = TimeSeriesDataset(valid_inout_seq)

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

    print("通过滑动窗口共有训练集数据：", len(train_inout_seq), "转化为批次数据:", len(train_loader))
    print("通过滑动窗口共有测试集数据：", len(test_inout_seq), "转化为批次数据:", len(test_loader))
    print("通过滑动窗口共有验证集数据：", len(valid_inout_seq), "转化为批次数据:", len(valid_loader))
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>创建数据加载器完成<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    return train_loader, test_loader, valid_loader, scaler, time_column


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_dim, pre_len):
        super(LSTM, self).__init__()
        self.pre_len = pre_len
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_directions = 1
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        h_0 = torch.zeros(self.num_directions * self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c_0 = torch.zeros(self.num_directions * self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h_0, c_0))  # output(5, 30, 64)
        out = self.dropout(out)

        # 取最后 pre_len 时间步的输出
        out = out[:, -self.pre_len:, :]

        out = self.relu(out)
        out = self.fc(out)

        # print(out.shape)

        return out

def train(model, args, train_loader, scaler, device):
    model = model.to(device)  # 将模型移动到指定设备
    model.train()  # 训练模式
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    epochs = args.epochs

    for epoch in tqdm(range(epochs)):
        losss = []
        for seq, labels in train_loader:
            seq, labels = seq.to(device), labels.to(device)  # 将数据移动到指定设备
            optimizer.zero_grad()
            y_pred = model(seq)  # 前向传播
            single_loss = loss_function(y_pred, labels)  # 计算损失
            single_loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            losss.append(single_loss.detach().cpu().numpy())

        # 打印每个 epoch 的损失
        tqdm.write(f"\t Epoch {epoch + 1} / {epochs}, Loss: {sum(losss) / len(losss)}")



def test(model, args, test_loader, scaler, time_column, model_name,div,start_point ,select=False):
    """
    测试函数，支持动态选择策略
    :param model: 模型实例
    :param args: 命令行参数
    :param test_loader: 测试数据加载器
    :param scaler: 标准化器
    :param time_column: 时间列
    :param model_name: 模型名称
    :param select: 是否启用动态选择策略
    """
    division = f'({args.div}_{1-args.div:.1f})'

    if select:
        model.load_state_dict(torch.load(fr'.\save\LSTM_policy_model{division}.pth'))
    else:
        model.load_state_dict(torch.load(fr'.\save\LSTM_model{division}.pth'))

    model.eval()  # 评估模式
    results = []  # 存储预测结果
    labels = []  # 存储真实标签

    with torch.no_grad():
        for seq, label in test_loader:
            seq, label = seq.to(device), label.to(device)  # 数据移动到设备

            if select:
                # 动态预测模式
                pred = model(seq, label, scaler, device, select=True)  # [batch, time_steps, 1]
            else:
                # 常规预测模式
                pred = model(seq)  # 直接使用模型预测

            # 反标准化（仅处理目标特征）
            pred_denorm = scaler.inverse_transform(pred.detach().cpu().numpy())  # [batch, time_steps, 1]
            label_denorm = scaler.inverse_transform(label.detach().cpu().numpy())  # [batch, time_steps, features]

            # 只取目标特征
            pred_denorm = pred_denorm[:, :, -1]  # [batch, time_steps]
            label_denorm = label_denorm[:, :, -1]  # [batch, time_steps]

            results.append(pred_denorm)  # 收集预测结果
            labels.append(label_denorm)  # 收集真实标签
    # 合并结果
    results = np.concatenate(results, axis=0)  # [total_samples, time_steps]
    labels = np.concatenate(labels, axis=0)  # [total_samples, time_steps]

    # 计算指标（仅目标特征）
    mae = get_mae(labels, results)
    rmse = get_rmse(labels, results)

    # 打印评估指标
    print(f"\n{model_name} 模型评估指标:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

    # 计算测试集的时间范围
    test_start = int(div * len(time_column))  # 测试集起始位置
    test_time = time_column[test_start:test_start + len(results)]  # 确保时间列与预测值长度一致

    # 创建 DataFrame 保存结果
    df = pd.DataFrame({
        '时间': test_time,
        '实际值': labels[:, 0],  # 取第一个时间步的真实值
        '预测值': results[:, 0]  # 取第一个时间步的预测值
    })

    # 保存预测结果
    os.makedirs('./data/result', exist_ok=True)
    result_path = f'./data/result/{args.target[0]}_{model_name}_预测结果.csv'
    df.to_csv(result_path, index=False)
    # 如果启用选择策略，保存策略触发记录
    if select:
        # 合并所有时间步的记录
        use_pred_records = np.concatenate(model.policy.use_pred_records, axis=0)  # [total_steps]
        df['策略触发'] = use_pred_records[:len(df)]  # 确保长度一致
    else:
        df['策略触发'] = False  # 未启用选择策略时，全部标记为 False

    df.to_csv(result_path, index=False)
    print(f"预测结果已保存到 {result_path}")

    # 计算策略生效比例（仅当启用选择策略时）
    if select:
        # 合并所有时间步的记录
        use_pred_records = np.concatenate(model.policy.use_pred_records, axis=0)  # [total_steps, batch_size]
        use_pred_records = use_pred_records.T  # [batch_size, total_steps]

        # 定义时间段（12h、24h、36h、48h）
        points_per_hour = 6  # 每10分钟一个数据点，每小时6个
        intervals = {
            '12h': 12 * points_per_hour,
            '24h': 24 * points_per_hour,
            '36h': 36 * points_per_hour,
            '48h': 48 * points_per_hour
        }
        # 设置起点
        metrics_data = []

        # 计算每个时间段的策略生效比例
        for name, max_step in intervals.items():
            if start_point + max_step > use_pred_records.shape[0]:
                print(f"警告：测试集长度不足，无法计算 {name} 的策略生效比例")
                continue

            # 从起点开始截取当前时间段的数据
            segment = use_pred_records[start_point:start_point + max_step]
            total_segment=use_pred_records
            origin_precent=100-np.mean(total_segment) * 100
            print(f"原始数据占比: {origin_precent:.2f}%")
            # 计算策略生效比例
            true_ratio = np.mean(segment) * 100  # 转换为百分比 生效比例=预测值比例
            pred_ratio = 100 - true_ratio  # 计算未生效比例=真实值比例

            # 输出结果
            print(f"{name} 真实值: {pred_ratio:.2f}%，预测值: {true_ratio:.2f}%")

            # 将结果添加到 metrics_data
            metrics_data.append({
                'Model': model_name,
                'Division': division,
                'Time Range (hours)': name,
                'True Value (%)': f"{pred_ratio:.2f}%",
                'Predicted Value (%)':f"{true_ratio:.2f}%"
            })

        output_path = f'./data/result/{args.target[0]}strategy_metrics.csv'
        # 保存到 CSV 文件
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_metrics = pd.DataFrame(metrics_data)
        df_metrics.to_csv(output_path, mode='a',index=False, encoding='utf-8-sig')
        print(f"策略生效比例已保存到 {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Time Series forecast')
    parser.add_argument('-model', type=str, default='LSTM', help="LSTM")
    parser.add_argument('-window_size', type=int, default=12, help="时间窗口大小, window_size > pre_len")
    parser.add_argument('-pre_len', type=int, default=1, help="预测未来数据长度")
    # data
    parser.add_argument('-shuffle', action='store_true', default=True, help="是否打乱数据顺序")
    parser.add_argument('-data_path', type=str, default=r"4G网络控制器 - 8路一号.xlsx", help="数据数据地址")
    parser.add_argument('-target', type=str, default=['湿度传感器一号'], help='需要预测的特征列，最后保存在csv文件里')
    parser.add_argument('-input_size', type=int, default=16, help='特征个数')
    parser.add_argument('-feature', type=str, default='MS', help='[M, S, MS],多元预测多元,单元预测单元,多元预测单元')

    # learning
    parser.add_argument('-lr', type=float, default=0.005, help="学习率")
    parser.add_argument('-drop_out', type=float, default=0.05, help="随机丢弃概率,防止过拟合")
    parser.add_argument('-epochs', type=int, default=30, help="训练轮次")
    parser.add_argument('-batch_size', type=int, default=24, help="批次大小")
    parser.add_argument('-save_path', type=str, default=r'.\save\LSTM.pth')

    # model
    parser.add_argument('-hidden_size', type=int, default=128, help="隐藏层单元数")
    parser.add_argument('-kernel_sizes', type=int, default=3)
    parser.add_argument('-layer_num', type=int, default=2)

    # device
    parser.add_argument('-use_gpu', type=bool, default=True)
    device = torch.device("cpu")

    # option
    parser.add_argument('-train', type=bool, default=True)
    parser.add_argument('-test', type=bool, default=True)
    parser.add_argument('-lr-scheduler', type=bool, default=True)

    # Policy 参数
    parser.add_argument('-threshold', type=float, default=0.6, help="选择策略的误差阈值")

    # div
    parser.add_argument('-div', type=float, default=0.6, help="训练集和测试集的划分比例")
    args = parser.parse_args()

    # 创建数据加载器
    train_loader, test_loader, valid_loader, scaler, time_column = create_dataloader(args, device,args.div)

    # 设置输出维度
    if args.feature == 'MS' or args.feature == 'S':
        args.output_size = 1
    else:
        args.output_size = args.input_size

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~·

    division = f'({args.div}_{1-args.div:.1f})'

    # 实例化
    model = LSTM(args.input_size, args.hidden_size, args.layer_num, args.output_size, args.pre_len).to(device)
    model_policy = modelPolicy(model,args.threshold).to(device)

    """
        训练完就注释掉↓
    """
    # # 训练
    # train(model, args, train_loader, scaler, device)
    # torch.save(model.state_dict(), f'./save/LSTM_model{division}.pth')
    #
    # train(model_policy, args, train_loader, scaler, device)
    # torch.save(model_policy.state_dict(), f'./save/LSTM_policy_model{division}.pth')


    # 测试
    start_point=0
    # # 原模型
    test(model, args, test_loader, scaler, time_column, 'LSTM',args.div,start_point, select=False)
    plot_results(args, 'LSTM',division,start_point)
    plot_single_model_violin(args, 'LSTM',division,start_point)
    # 策略模型
    test(model_policy, args, test_loader, scaler, time_column, 'LSTM_Policy',args.div,start_point, select=True)
    plot_results(args, 'LSTM_Policy',division,start_point)
    plot_single_model_violin(args, 'LSTM_Policy',division,start_point)

    plt.show()




