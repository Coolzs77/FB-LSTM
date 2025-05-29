import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.dates import DayLocator
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.colors as mcolors

'''科研图表全局设置'''
plt.rcParams.update({
    'font.family': 'Times New Roman',  # 统一字体
    'axes.linewidth': 1.5,  # 坐标轴线宽
    'xtick.major.width': 1.5,  # X轴刻度线宽
    'ytick.major.width': 1.5,  # Y轴刻度线宽
    'axes.edgecolor': '#2F2F2F',  # 坐标轴颜色
    'axes.labelweight': 'bold',  # 坐标标签加粗
    'axes.titleweight': 'bold',  # 标题加粗
    'figure.dpi': 600  # 输出分辨率
})

def plot_single_model_violin(args, model_name,division, start_point=0):
    """
    绘制单个模型多时间段误差分布小提琴图
    :param args: 命令行参数对象
    :param model_name: 模型名称
    """

    # ================== 数据准备 ==================
    # 读取预测结果文件
    result_path = f'./data/result/{args.target[0]}_{model_name}_预测结果.csv'
    if not os.path.exists(result_path):
        print(f"错误：{model_name} 预测结果文件不存在")
        return

    df = pd.read_csv(result_path)
    errors = (df['实际值'] - df['预测值']).abs().values  # 计算绝对误差

    # 按时间段分割数据
    time_intervals = ['12h', '24h', '36h', '48h']
    points_per_interval = 6 * 12  # 每12小时72个数据点（每10分钟一个点）

    plot_data = []
    for i, interval in enumerate(time_intervals):
        start = start_point + i * points_per_interval
        end = start + points_per_interval
        if end > len(errors):
            print(f"警告：{model_name} {interval} 数据不足，跳过该时间段")
            continue
        plot_data.extend([(interval, e) for e in errors[start:end]])

    if not plot_data:
        print("错误：没有足够的数据绘制小提琴图")
        return
    # 创建数据框
    df_plot = pd.DataFrame(plot_data, columns=['时间段', '误差值'])
    # print(df_plot.head())  # 打印前几行数据，检查是否正确

    # ================== 绘制小提琴图 ==================
    plt.figure(figsize=(10, 6))

    # 绘制小提琴图
    ax=sns.violinplot(
        x='时间段',
        y='误差值',
        data=df_plot,
        hue='时间段',  # 添加 hue 参数
        palette="Set2",
        inner="box",
        linewidth=1.5,
        legend=False  # 不显示图例
    )
    # 坐标轴优化
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color('#4F4F4F')
    ax.spines['left'].set_color('#4F4F4F')

    # ================== 统计标注 ==================
    for i, interval in enumerate(time_intervals):
        subset = df_plot[df_plot['时间段'] == interval]
        # 计算统计量
        min_val = subset['误差值'].min()
        q1 = subset['误差值'].quantile(0.25)
        median = subset['误差值'].median()
        q3 = subset['误差值'].quantile(0.75)
        max_val = subset['误差值'].max()

        # 标注最小值
        ax.text(i + 0.1, min_val-1, f'{min_val:.2f}',
                color='black', fontsize=12, va='center', font='Times New Roman')
        # 标注Q1
        ax.text(i + 0.1, q1, f'{q1:.2f}',
                color='black', fontsize=12, va='center', font='Times New Roman')

        # 标注Q3
        ax.text(i + 0.1, q3+0.8, f'{q3:.2f}',
                color='black', fontsize=12, va='center', font='Times New Roman')
        # 标注最大值
        ax.text(i + 0.1, max_val+2.3, f'{max_val:.2f}',
                color='black', fontsize=12, va='center', font='Times New Roman')

        # 配置传感器简称和单位
        sensor_config = {
            '温度传感器一号': {'abbr': 'IAT', 'unit': '℃', 'name': r"$\epsilon_{T}$"},
            '湿度传感器一号': {'abbr': 'IAH', 'unit': '%', 'name': r"$\epsilon_{H}$"}
        }

        # 获取当前传感器的配置
        current_config = sensor_config.get(args.target[0], {'abbr': 'Unknown', 'unit': 'Unit', 'name': 'Unknown'})
        current_abbr = current_config['abbr']  # 简称
        current_unit = current_config['unit']  # 单位
        current_name = current_config['name']  # 名称

    # ================== 设置标题和标签 ==================
    # plt.title(f'{model_name}{division} model({current_abbr})', fontsize=18)
    plt.xlabel('Time (h)', fontsize=20, font='Times New Roman')
    plt.ylabel(f'{current_name} ({current_unit})', fontsize=20, font='Times New Roman')

    # 显式设置刻度位置和标签
    ax.set_xticks(range(len(time_intervals)))  # 设置刻度位置
    ax.set_xticklabels(['12', '24', '36', '48'], fontsize=20, font='Times New Roman')  # 设置刻度标签
    if args.target[0]=='温度传感器一号':
        # 设置 y 轴刻度
        ax.set_ylim(-1, 3)  # 设置主刻度范围
        ax.yaxis.set_major_locator(MultipleLocator(0.5))  # 主刻度间隔
        ax.yaxis.set_minor_locator(MultipleLocator(0.25))  # 副刻度间隔
        ax.yaxis.set_minor_formatter(plt.NullFormatter())  # 不显示副刻度文字
    elif args.target[0]=='湿度传感器一号':
        # 设置 y 轴刻度
        ax.set_ylim(-8, 20)  # 设置主刻度范围
        ax.yaxis.set_major_locator(MultipleLocator(4))  # 主刻度间隔
        ax.yaxis.set_minor_locator(MultipleLocator(2))  # 副刻度间隔为2
        ax.yaxis.set_minor_formatter(plt.NullFormatter())  # 不显示副刻度文字
    # 设置刻度字体大小
    ax.tick_params(axis='both', which='major', labelsize=14)


    # plt.show()
    # ================== 保存图片 ==================
    os.makedirs('./data/plots', exist_ok=True)
    plot_path = f'./data/plots/{args.target[0]}_{model_name}{division}_误差分布.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"{model_name} 模型小提琴图已保存至：{plot_path}")

def plot_results(args, model_name, division, start_point=0):
    """
    绘制预测结果图并保存到本地，同时将每个时间段的 MAE 和 RMSE 保存到 CSV 文件中
    :param args: 命令行参数
    :param model_name: 模型名称（如 'GRU', 'LSTM' 等）
    :param division: 数据划分（如 'train', 'test' 等）
    """

    # 读取预测结果
    result_path = f'./data/result/{args.target[0]}_{model_name}_预测结果.csv'
    if not os.path.exists(result_path):
        print(f"预测结果文件不存在，请先运行 {model_name} 模型的测试代码！")
        return

    df = pd.read_csv(result_path)
    test_time = df['时间']
    ys = df['实际值']
    preds = df['预测值']
    triggered = df['策略触发']  # 读取策略触发记录

    # 定义时间间隔（10分钟）
    interval = 10  # 分钟
    points_per_hour = 6  # 每小时 6 个数据点
    time_ranges = [12, 24, 36, 48]  # 需要绘制的时间范围（小时）
    points_ranges = [t * points_per_hour for t in time_ranges]  # 转换为数据点数量

    # 用于存储每个时间段的 MAE 和 RMSE
    metrics_data = []

    # 绘制各时间段的图表并计算指标
    for points, time_range in zip(points_ranges, time_ranges):
        # 检查起点和终点是否超出数据范围
        if start_point + points > len(ys):
            print(f"警告：测试集长度不足，无法绘制 {time_range} 小时的图表")
            continue

        # 截取从起点开始的数据
        ys_subset = ys[start_point:start_point + points]
        preds_subset = preds[start_point:start_point + points]
        test_time_subset = test_time[start_point:start_point + points]
        triggered_subset = triggered[start_point:start_point + points]

        # 计算 MAE 和 RMSE
        mae = round(get_mae(ys_subset, preds_subset), 4)  # 保留 4 位小数
        rmse = round(get_rmse(ys_subset, preds_subset), 4)  # 保留 4 位小数

        # 将指标添加到 metrics_data
        metrics_data.append({
            'Model': model_name,
            'Division': division,
            'Time Range (hours)': time_range,
            'MAE': mae,
            'RMSE': rmse
        })

        # 绘制图表
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        x_indices = range(len(test_time_subset))
        # ================== 坐标轴优化 ==================
        # 坐标轴优化
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        ax.spines['bottom'].set_color('#4F4F4F')
        ax.spines['left'].set_color('#4F4F4F')
        # 设置刻度字体大小
        ax.tick_params(axis='both', which='major', labelsize=14)
        trigger_indices = [i for i, trigger in enumerate(triggered_subset) if trigger]
        # 显示副刻度但不显示文字提示
        ax.xaxis.set_minor_locator(DayLocator())
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        # 绘制实际值和预测值
        ax.plot(x_indices, ys_subset, 'r^-', label='Actual', markersize=4)  # 实际温度
        ax.plot(x_indices, preds_subset, 'bo-', label='Predict', markersize=4)  # 预测温度

        # 标注策略触发的时间点
        ax.scatter(trigger_indices, preds_subset.iloc[trigger_indices], color='green', label='Applied', zorder=5)
        # 设置图例位置为左上角并自定义样式
        ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=14, frameon=False)
        # 配置传感器简称和单位
        sensor_config = {
            '温度传感器一号': {'abbr': 'IAT', 'unit': '℃'},
            '湿度传感器一号': {'abbr': 'IAH', 'unit': '%'}
        }

        # 获取当前传感器的配置
        current_config = sensor_config.get(args.target[0], {'abbr': 'Unknown', 'unit': 'Unit'})
        current_abbr = current_config['abbr']  # 简称
        current_unit = current_config['unit']  # 单位

        # 添加标题和标签
        # ax.set_title(f'{model_name}{division} Model Performance ({time_range}hours)')
        ax.set_xlabel('Time',fontsize=18)
        ax.set_ylabel(f'{current_abbr} ({current_unit})',fontsize=18, font='Times New Roman')
        ax.legend()
        ax.grid(False)

        # 设置 x 轴刻度
        if time_range == 12:
            tick_spacing = 3  # 每半小时显示一个数据点（3个10分钟间隔）
        elif time_range == 24:
            tick_spacing = 6  # 每小时显示一个数据点（6个10分钟间隔）
        elif time_range == 36:
            tick_spacing = 9  # 每1.5小时显示一个数据点（9个10分钟间隔）
        elif time_range == 48:
            tick_spacing = 12  # 每2小时显示一个数据点（12个10分钟间隔）

        ax.set_xticks(range(0, len(test_time_subset), tick_spacing),
                   test_time_subset[::tick_spacing],
                   rotation=45)
        # plt.show()
        # 设置纵轴刻度
        if args.target[0] == '温度传感器一号':
            ax.set_ylim(20, 30)  # 设置主刻度范围
            ax.yaxis.set_major_locator(MultipleLocator(2))  # 主刻度间隔
            ax.yaxis.set_minor_locator(MultipleLocator(1))  # 副刻度间隔
            ax.yaxis.set_minor_formatter(plt.NullFormatter())  # 不显示副刻度文字

        elif args.target[0] == '湿度传感器一号':
            ax.set_ylim(60, 105)  # 设置主刻度范围
            ax.yaxis.set_major_locator(MultipleLocator(10))  # 主刻度间隔
            ax.yaxis.set_minor_locator(MultipleLocator(5))  # 副刻度间隔
            ax.yaxis.set_minor_formatter(plt.NullFormatter())  # 不显示副刻度文字

        # 保存图表到本地
        os.makedirs('./data/plots', exist_ok=True)
        plot_path = f'./data/plots/{model_name}{division}_{args.target[0]}_{time_range}小时预测结果.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"图表已保存到 {plot_path}")

        # 打印指标
        print('--------------------------------')
        print(f'{model_name}{division} 模型 {args.target[0]} 的预测指标（{time_range}小时）:')
        print('MAE:', mae)
        print('RMSE:', rmse)
        print('--------------------------------')

    # 将指标保存到 CSV 文件
    metrics_df = pd.DataFrame(metrics_data)
    metrics_path = f'./data/result/{args.target[0]}_metrics.csv'
    metrics_df.to_csv(metrics_path, mode='a', header=False, index=False)
    print(f"指标已保存到 {metrics_path}")


def get_mae(y, pred):
    return mean_absolute_error(y, pred)


def get_mse(y, pred):
    return mean_squared_error(y, pred)


def get_rmse(y, pred):
    return np.sqrt(mean_squared_error(y, pred))
