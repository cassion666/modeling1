import pandas as pd
from prophet import Prophet  # 需安装：pip install prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# 让图片中可以显示负号
# Matplotlib 默认对 Unicode 负号显示可能有问题，设置为 False 可正确显示负号（如 -1 不会显示异常）
plt.rcParams['axes.unicode_minus'] = False

# 让图片中可以显示中文
# 设置字体为 'SimHei'（宋体），这样绘图时遇到中文就能用该字体显示，避免中文显示为方框等乱码情况
plt.rcParams['font.sans-serif'] = 'SimHei'

# ====================== 1. 数据读取与预处理 ====================== #
# 注意：路径使用原始字符串（r""）避免转义问题
df = pd.read_csv(
    filepath_or_buffer=r"D:\Pythonwenjianjia\COVID-19-Data-master\csv格式\china_provincedata.csv",
    encoding='gbk'
)

# 筛选北京数据
beijing = df[df['provinceShortName'] == '北京'].copy()

# 转换日期格式（dateId -> datetime）
beijing['date'] = pd.to_datetime(beijing['dateId'].astype(str), format='%Y%m%d')
beijing = beijing.sort_values('date')  # 按日期排序

# 整理为 Prophet 要求的格式：列名必须为 'ds'（日期）和 'y'（目标值）
df_prophet = beijing[['date', 'confirmedCount']].rename(
    columns={'date': 'ds', 'confirmedCount': 'y'}
)


# ====================== 2. 拆分训练集 & 测试集 ====================== #
train_end = '2020-06-21'       # 训练集结束日期
test_start = '2020-06-22'      # 测试集开始日期
test_end = '2020-07-02'        # 测试集结束日期

train_prophet = df_prophet[df_prophet['ds'] <= train_end]
test_prophet = df_prophet[
    (df_prophet['ds'] > train_end) & (df_prophet['ds'] <= test_end)
]


# ====================== 3. 拟合 Prophet 模型 ====================== #
model = Prophet()  # 可添加参数：如seasonality_mode='multiplicative'（若有明显季节效应）
model.fit(train_prophet)


# ====================== 4. 预测测试集 ====================== #
# 构建未来日期（包含测试集时间范围，periods=测试集样本数）
future = model.make_future_dataframe(
    periods=len(test_prophet),  # 预测长度=测试集天数
    freq='D'  # 按天预测
)
forecast = model.predict(future)

# 提取测试集的预测结果（仅保留测试集时间范围内的数据）
forecast_test = forecast[forecast['ds'].isin(test_prophet['ds'])]


# ====================== 5. 模型评估（MAE、MSE、RMSE） ====================== #
mae = mean_absolute_error(test_prophet['y'], forecast_test['yhat'])
mse = mean_squared_error(test_prophet['y'], forecast_test['yhat'])
rmse = np.sqrt(mse)

print("=== 模型评估指标 ===")
print(f"MAE: {mae:.2f}")   # 平均绝对误差
print(f"MSE: {mse:.2f}")   # 均方误差
print(f"RMSE: {rmse:.2f}") # 均方根误差


# ====================== 6. 可视化 ====================== #
# (1) Prophet 内置的趋势+季节分解图
fig1 = model.plot(forecast)
plt.title("Prophet 整体预测结果（含趋势）")

fig2 = model.plot_components(forecast)
plt.suptitle("Prophet 成分分解（趋势+周/年周期，本例无明显周期，故成分简单）")

# (2) 手动绘制：训练集、测试集、预测值对比
plt.figure(figsize=(12, 6))
plt.plot(train_prophet['ds'], train_prophet['y'], label='训练集（实际累计确诊）', color='blue')
plt.plot(test_prophet['ds'], test_prophet['y'], label='测试集（实际累计确诊）', color='green')
plt.plot(forecast_test['ds'], forecast_test['yhat'], label='测试集（预测值）', color='red', linestyle='--')
plt.xlabel('日期')
plt.ylabel('累计确诊数')
plt.title('Prophet 模型预测北京新冠累计确诊数')
plt.legend()
plt.grid(True)
plt.show()
