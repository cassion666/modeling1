#  Project Name：modeling1 - a modeling and prediction project based on some epidemic data in Beijing

# Project Introduction   
>Based on some data sets during the Beijing epidemic period, this project uses Python to realize data modeling, prediction and evaluation, which can be used to analyze the general trend of urban epidemic, and output visual results and prediction reports.

# The Core Steps of Use 
 1. Environmental preparations(Developed based on Pycharm)
import pandas as pd
from prophet import Prophet  # 需安装：pip install prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np  

**Example (using a Python project)**：  
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
