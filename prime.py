import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, DayLocator

# 专业科研图表配置
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'axes.linewidth': 1.5,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'axes.edgecolor': '#2F2F2F',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold'
})

# 参数配置系统（移除非数据列配置）
PARAM_CONFIG = {
    # 室内环境参数
    '温度传感器一号': {'abbr': 'IAT', 'unit': '°C'},    # IAT: Indoor Air Temperature
    '湿度传感器一号': {'abbr': 'IAH', 'unit': '%'},     # IAH: Indoor Air Humidity
    'CO2传感器一号': {'abbr': 'ICO2', 'unit': 'ppm'},  # ICO2: Indoor Carbon Dioxide Concentration

    '室外温度': {'abbr': 'OAT', 'unit': '°C'},        # OAT: Outdoor Air Temperature
    '室外湿度': {'abbr': 'OAH', 'unit': '%'},         # OAH: Outdoor Air Humidity
    '室外co2': {'abbr': 'OCO2', 'unit': 'ppm'},      # OCO2: Outdoor Carbon Dioxide Concentration

    '新温度': {'abbr': 'NT', 'unit': '°C'},           # NT: New Air Temperature
    '新湿度': {'abbr': 'NH', 'unit': '%'},            # NH: New Air Humidity
    '新co2': {'abbr': 'NCO2', 'unit': 'ppm'},        # NCO2: New Air Carbon Dioxide Concentration

}


def create_research_plot(data, config):
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)

    # 绘制数据
    data.plot(ax=ax, linewidth=2.0, color='#0000ff', alpha=0.9)

    # 设置坐标轴范围
    ax.set_xlim(data.index.min(), data.index.max())
    y_padding = (data.max() - data.min()) * 0.1
    ax.set_ylim(data.min() - y_padding, data.max() + y_padding)

    # 坐标轴标签设置
    ax.set_xlabel('Time', fontsize=14, labelpad=10)  # X轴标签
    ax.set_ylabel(config['unit'], fontsize=16, labelpad=10,  # Y轴单位作为标签
                  rotation=90, va='center')

    # 时间轴刻度设置
    xticks = pd.date_range(start=data.index.min(), end=data.index.max(), periods=4)
    ax.set_xticks(xticks)
    ax.set_xticklabels([tick.strftime('%Y/%m/%d') for tick in xticks],
                       ha='center', fontsize=12, rotation=0)  # X轴刻度字号加大
    # 移除副刻度
    ax.xaxis.set_minor_locator(DayLocator())
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    # Y轴刻度字号
    ax.tick_params(axis='y', labelsize=12)

    # 样式优化（保持不变）
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color('#4F4F4F')
    ax.spines['left'].set_color('#4F4F4F')

    ax.set_title(f"{config['abbr']}",  # 标题仅显示缩写
                 fontsize=18, loc='right')
    # 时间轴刻度设置
    ax.set_xticklabels(['2023/9/29', '2023/9/30', '2023/10/01', '2023/10/02'], ha='center', fontsize=12, rotation=0)
    plt.tight_layout()
    return fig



# 数据加载与预处理
df = pd.read_excel(r"4G网络控制器 - 8路一号.xlsx",
                   index_col='时间',
                   parse_dates=True)
df.index.name = 'Time'

# 生成图表
for col, config in PARAM_CONFIG.items():
    try:
        # 获取原始数据
        original = df[col]

        # 计算时间范围（精确到分钟的前三天）
        start_time = original.index.min()+pd.Timedelta(days=6.3)
        end_time = start_time + pd.Timedelta(days=3) - pd.Timedelta(minutes=10)

        # 截取前三日完整数据
        original_3days = original.loc[start_time:end_time]

        # 使用截取的数据绘图
        fig = create_research_plot(original_3days, config)
        fig.savefig(f"./origin_data/{config['abbr']}.png", dpi=600,
                    bbox_inches='tight', transparent=False)
        plt.close(fig)
    except KeyError:
        continue


