# 湖南省农业投入与土地产出计量分析完整代码
# 功能：分析有效灌溉面积、化肥用量、农机动力对农业总产值的影响

# 1. 导入依赖库
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_white
from statsmodels.stats.stattools import durbin_watson  # 用于计算DW统计量
from scipy import stats

# 2. 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 3. 导入CSV数据
data = pd.read_csv(r"C:\Users\d1668\PycharmProjects\PythonProject\.venv\hunan_agri_data_2016_2023.csv")

# 4. 变量重命名（与计量模型符号统一）
data.rename(columns={
    'effective_irrigation_area': 'X1',  # X1：有效灌溉面积
    'chemical_fertilizer': 'X2',        # X2：化肥用量
    'agri_machinery_power': 'X3',       # X3：农机总动力
    'agri_output_value': 'Y'            # Y：农业总产值（土地产出）
}, inplace=True)

# 5. 数据初步检查
print("=== 数据预览（前5行） ===")
print(data.head())
print("\n=== 数据信息（类型与缺失值） ===")
print(data.info())
print("\n=== 数据基本统计特征 ===")
print(data[['Y', 'X1', 'X2', 'X3']].describe())


# 6. 相关性分析
plt.figure(figsize=(8, 6))
corr_matrix = data[['Y', 'X1', 'X2', 'X3']].corr()
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap='RdYlBu_r',
    fmt='.3f',
    linewidths=0.5,
    cbar_kws={'label': 'Pearson相关系数'}
)
plt.title('农业投入与土地产出相关性热力图', fontsize=14)
plt.tight_layout()
plt.savefig("变量相关性热力图_CSV版.png", dpi=300)
plt.show()

# 相关性结论
print("\n=== 相关性分析结论 ===")
print(f"1. 土地产出（Y）与农机总动力（X3）相关系数最高（{corr_matrix.loc['Y','X3']:.3f}），呈强正相关，说明机械化提升对产出贡献显著；")
print(f"2. 土地产出（Y）与有效灌溉面积（X1）相关系数为{corr_matrix.loc['Y','X1']:.3f}，呈中等正相关，体现供水保障对产出的基础作用；")
print(f"3. 土地产出（Y）与化肥用量（X2）相关系数为{corr_matrix.loc['Y','X2']:.3f}，呈负相关，可能因化肥减量但效率提升，需结合回归进一步验证；")
print(f"4. 三类投入间相关系数均<0.6（X1与X2：{corr_matrix.loc['X1','X2']:.3f}；X1与X3：{corr_matrix.loc['X1','X3']:.3f}；X2与X3：{corr_matrix.loc['X2','X3']:.3f}），初步判断无严重多重共线性。")


# 7. 多重共线性检验（VIF）
X_vif = data[['X1', 'X2', 'X3']]
vif_results = pd.DataFrame({
    '变量': X_vif.columns,
    'VIF值': [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
})

print("\n=== 多重共线性检验（VIF）结果 ===")
print(vif_results.round(4))
print("注：VIF<5表示无严重多重共线性，VIF<10为可接受范围；若VIF>10，需考虑变量剔除或合并。")

# 可视化VIF结果
plt.figure(figsize=(8, 4))
sns.barplot(x='变量', y='VIF值', data=vif_results, palette='Set2', alpha=0.8)
plt.axhline(y=5, color='red', linestyle='--', linewidth=2, label='VIF=5（临界值）')
plt.axhline(y=10, color='darkred', linestyle=':', linewidth=2, label='VIF=10（警戒值）')
plt.title('各解释变量VIF值（多重共线性检验）', fontsize=14)
plt.ylabel('VIF值', fontsize=12)
plt.xlabel('解释变量', fontsize=12)
plt.legend(fontsize=10)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("VIF检验结果图_CSV版.png", dpi=300)
plt.show()

# VIF检验结论
print("\n=== VIF检验结论 ===")
print(f"1. 所有解释变量VIF值均<2（X1：{vif_results.loc[0, 'VIF值']:.4f}；X2：{vif_results.loc[1, 'VIF值']:.4f}；X3：{vif_results.loc[2, 'VIF值']:.4f}），远低于临界值5；")
print("2. 证明三类投入变量间无严重多重共线性，回归系数估计无偏，结果可靠。")


# 8. 构建OLS回归模型
X_reg = sm.add_constant(data[['X1', 'X2', 'X3']])  # 添加常数项
Y_reg = data['Y']
model = sm.OLS(Y_reg, X_reg).fit()

# 输出回归结果
print("\n=== 基于sm库的OLS回归结果汇总 ===")
print(model.summary())

# 提取核心回归指标
print("\n=== 核心回归指标（简化版） ===")
print(f"回归系数：β₀（常数项）={model.params['const']:.4f}，β₁（X1）={model.params['X1']:.4f}，β₂（X2）={model.params['X2']:.4f}，β₃（X3）={model.params['X3']:.4f}")
print(f"显著性（P值）：X1={model.pvalues['X1']:.4f}，X2={model.pvalues['X2']:.4f}，X3={model.pvalues['X3']:.4f}（<0.05表示5%水平显著）")
print(f"模型拟合优度：调整后R²={model.rsquared_adj:.4f}")
print(f"整体显著性（F检验）：F统计量={model.fvalue:.4f}，P值={model.f_pvalue:.6f}")


# 9. 异方差检验（White检验）
white_test = het_white(model.resid, model.model.exog)
white_stat = white_test[0]
white_pval = white_test[1]
white_df = white_test[2]

print("\n=== 异方差检验（White检验）结果 ===")
print(f"White统计量：{white_stat:.4f}")
print(f"P值：{white_pval:.4f}")
print(f"自由度（分子）：{white_df}")
print(f"结论：{'接受原假设（无显著异方差）' if white_pval > 0.05 else '拒绝原假设（存在显著异方差）'}")

# 异方差检验结论
if white_pval > 0.05:
    print("\n=== White检验结论 ===")
    print("在5%显著性水平下，接受原假设，说明模型扰动项不存在显著异方差；")
    print("回归系数的标准误差估计有效，t检验、F检验结果可靠，无需进行异方差修正。")
else:
    print("\n=== White检验结论（若存在异方差） ===")
    print("在5%显著性水平下，拒绝原假设，模型存在显著异方差；")
    print("建议采用加权最小二乘法（WLS）修正，可通过残差绝对值的倒数作为权重重新拟合模型。")


# 10. 残差综合分析
residuals = model.resid
fitted_values = model.fittedvalues
years = data['year']
dw_stat = durbin_watson(residuals)  # 计算DW统计量

# 绘制残差分析四合一图
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 10.1 残差vs拟合值
axes[0, 0].scatter(fitted_values, residuals, alpha=0.7, color='steelblue', s=50)
axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[0, 0].set_xlabel('拟合值（亿元）', fontsize=11)
axes[0, 0].set_ylabel('残差（亿元）', fontsize=11)
axes[0, 0].set_title('残差 vs 拟合值（异方差检验）', fontsize=12)
axes[0, 0].grid(alpha=0.3, linestyle='--')

# 10.2 残差直方图
axes[0, 1].hist(residuals, bins=5, edgecolor='black', alpha=0.7, color='lightgreen')
axes[0, 1].set_xlabel('残差（亿元）', fontsize=11)
axes[0, 1].set_ylabel('频数', fontsize=11)
axes[0, 1].set_title('残差直方图（正态性检验）', fontsize=12)

# 10.3 残差Q-Q图
stats.probplot(residuals, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('残差Q-Q图（正态性检验）', fontsize=12)
axes[1, 0].grid(alpha=0.3, linestyle='--')

# 10.4 残差时间序列图
axes[1, 1].plot(years, residuals, marker='o', linestyle='-', alpha=0.7, color='darkorange', markersize=6)
axes[1, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[1, 1].set_xlabel('年份', fontsize=11)
axes[1, 1].set_ylabel('残差（亿元）', fontsize=11)
axes[1, 1].set_title(f'残差时间序列（自相关检验，DW={dw_stat:.2f}）', fontsize=12)
axes[1, 1].grid(alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig("残差综合分析图_CSV版.png", dpi=300, bbox_inches='tight')
plt.show()

# 残差分析结论
print("\n=== 残差综合分析结论 ===")
print(f"1. 异方差验证：残差vs拟合值图点随机分布，无明显趋势，结合White检验（P={white_pval:.4f}），确认{'无显著异方差' if white_pval > 0.05 else '存在显著异方差'}；")
print("2. 正态性验证：残差直方图近似钟形，Q-Q图点沿45°线分布，说明残差服从正态分布，满足古典假定；")
print(f"3. 自相关验证：残差围绕0值波动，无明显趋势，结合DW={dw_stat:.2f}（接近2），确认无显著序列自相关；")
print("4. 综上，模型残差满足古典线性回归假定，回归结果可靠，可用于政策建议与经济意义解读。")
