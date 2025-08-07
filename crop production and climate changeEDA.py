#导包
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

#导数据
df = pd.read_csv(r"F:\Pythonwenjianjia\crop_production.csv\crop_production.csv")

#观察大致数据
all_locations = df.LOCATION.value_counts().index.tolist()
all_subjects = df.SUBJECT.value_counts().index.tolist()
all_measures = df.MEASURE.value_counts().index.tolist()
print(df.head())
print(df['LOCATION'].unique())
print(df['SUBJECT'].value_counts())

#根据观察结果清洗数据（移除几列）
df.drop(['INDICATOR','FREQUENCY','Flag Codes'], axis=1, inplace=True)
print(df.head())

select_location = 'CAN'
select_measure = 'THND_TONNE'
# extract corresponding sub data frame
df_select = df[(df.LOCATION==select_location) & (df.MEASURE==select_measure)]

# plot all 4 subjects
plt.figure(figsize=(12,6))
sns.lineplot(data=df_select, x='TIME', y='Value', hue='SUBJECT', hue_order=all_subjects)
plt.title(select_location + ' - '+ select_measure)
plt.grid()
plt.show()

for select_location in all_locations:
    df_select = df[(df.LOCATION==select_location) & (df.MEASURE==select_measure)]
    plt.figure(figsize=(12,5))
    sns.lineplot(data=df_select, x='TIME', y='Value',
                 hue='SUBJECT', hue_order=all_subjects)
    plt.title(select_location + ' - '+ select_measure)
    plt.grid()
    plt.show()