# 导入相关库
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

#导入数据
iris = pd.read_csv(r"F:\Iris Code -- Machine Learning 3\Iris Code -- Machine Learning 3\Iris.csv")

#查看数据基本情况
print(iris.head())
print(iris.describe())
print(iris.info())

#删除无用行，让代码看着更清爽
iris.drop(labels='Id',axis=1,inplace=True)

##花萼长宽
#绘制Setosa的散点图
fig = iris[iris.Species=='Iris-setosa'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='orange', label='Setosa')
#叠加Versicolor的散点图
iris[iris.Species=='Iris-versicolor'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='blue', label='versicolor',ax=fig)
#叠加Virginica的散点图
iris[iris.Species=='Iris-virginica'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='green', label='virginica', ax=fig)
#美化图表+显示
fig.set_xlabel("Sepal Length")   #设置横轴标题：花萼长度
fig.set_ylabel("Sepal Width")   #设置纵标题：花萼宽度
fig.set_title("Sepal Length VS Width")   #设置图表标题
fig=plt.gcf()   #获取当前画布
fig.set_size_inches(10,6)   #设置画布尺寸
plt.show()   #显示最终图

##花瓣长宽
#绘制Setosa的散点图
fig = iris[iris.Species=='Iris-setosa'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='orange', label='Setosa')
#叠加Versicolor的散点图
iris[iris.Species=='Iris-versicolor'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='blue', label='versicolor',ax=fig)
#叠加Virginica的散点图
iris[iris.Species=='Iris-virginica'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='green', label='virginica', ax=fig)
#美化图表+显示
fig.set_xlabel("Petal Length")   #设置横轴标题：花瓣长度
fig.set_ylabel("Petal Width")   #设置纵标题：花瓣宽度
fig.set_title(" Petal Length VS Width")
fig=plt.gcf()
fig.set_size_inches(10,6)
plt.show()

##全体特征的直方图（目标是看所有数值特征的整体分布）
#画直方图，让柱子带黑边更清晰
iris.hist(edgecolor='black', linewidth=1.2)
fig=plt.gcf()
fig.set_size_inches(12,6)
plt.show()

#分品种的小提琴图（4个子图），小提琴图展示分布形状，比箱线图更直观
plt.figure(figsize=(15,10))   #画布尺寸
plt.subplot(2,2,1)
sns.violinplot(x='Species',y='PetalLengthCm',data=iris)
plt.subplot(2,2,2)
sns.violinplot(x='Species',y='PetalWidthCm',data=iris)
plt.subplot(2,2,3)
sns.violinplot(x='Species',y='SepalLengthCm',data=iris)
plt.subplot(2,2,4)
sns.violinplot(x='Species',y='SepalWidthCm',data=iris)
plt.show()

#筛选数值列用于绘制热力图
numeric_iris=iris.select_dtypes(include=['float64', 'int64'])
#热力图分析相关性
plt.figure(figsize=(7,4))
sns.heatmap(numeric_iris.corr(),annot=True,cmap='cubehelix_r')
plt.show()

## 提取特征（X）和标签（y）
X = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris['Species']

# 拆分训练集和测试集（补充闭合括号）
train_X, test_X, train_y, test_y = train_test_split(
    X, y, test_size=0.3, random_state=42
)

### SVM
model = svm.SVC()
model.fit(train_X,train_y)
prediction=model.predict(test_X)
print('The accuracy of the SVM is:',metrics.accuracy_score(prediction,test_y))

### LR
model = LogisticRegression()
model.fit(train_X,train_y)
prediction=model.predict(test_X)
print('The accuracy of the Logistic Regression is',metrics.accuracy_score(prediction,test_y))

### DT
model=DecisionTreeClassifier()
model.fit(train_X,train_y)
prediction=model.predict(test_X)
print('The accuracy of the Decision Tree is',metrics.accuracy_score(prediction,test_y))

### KNN
model=KNeighborsClassifier(n_neighbors=3)
model.fit(train_X,train_y)
prediction=model.predict(test_X)
print('The accuracy of the KNN is',metrics.accuracy_score(prediction,test_y))

a_index = list(range(1,11))
a = pd.Series(dtype='float64')
x = [1,2,3,4,5,6,7,8,9,10]
for i in x:
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(train_X, train_y)
    prediction = model.predict(test_X)
    # 计算准确率
    acc = metrics.accuracy_score(prediction, test_y)
    # 关键：把 acc 追加到 a 里（用 pd.concat 替代 append）
    new_acc_series = pd.Series(acc)  # 构造新的 Series
    a = pd.concat([a, new_acc_series], ignore_index=True)  # 追加到 a 中

# 补充可视化细节
plt.plot(a_index, a, marker='o')  # 增加标记点，更易观察
plt.xticks(x)
plt.xlabel("Number of Neighbors (n_neighbors)")  # x轴标签
plt.ylabel("Classification Accuracy")  # y轴标签
plt.title("KNN Accuracy vs. Number of Neighbors")  # 图表标题
plt.grid(alpha=0.3)  # 增加网格线，便于读数
plt.show()  # 显示图像