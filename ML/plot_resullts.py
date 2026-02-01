import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据
csv_path = 'error.xlsx'  # 修改为你的误差表格文件名
df = pd.read_excel(csv_path)

# 模型列名（假设从'DT'到'FinalModel'都是模型误差列）
model_cols = df.columns[4:]  # 跳过前4列: Well, Depth_measured, Pressure_measured, Nearest_Depth

# 1. 误差条形图（每个点多模型对比）
plt.figure(figsize=(12, 6))
bar_width = 0.1
x = range(len(df))
for i, model in enumerate(model_cols):
    plt.bar([xi + i*bar_width for xi in x], df[model], width=bar_width, label=model)
plt.xticks([xi + bar_width*(len(model_cols)/2) for xi in x], df['Well'] + '_' + df['Depth_measured'].astype(str), rotation=45)
plt.ylabel('Error')
plt.legend()
plt.tight_layout()
plt.savefig('barplot_per_point.png', dpi=300)
plt.close()

# 2. 箱线图（各模型误差分布）
plt.figure(figsize=(10, 7))
sns.boxplot(data=df[model_cols])
plt.ylabel('Error')
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid()
plt.savefig('boxplot_per_model.png', dpi=300)
plt.close()

# 3. 误差随深度变化折线图
plt.figure(figsize=(12, 6))
for model in model_cols:
    plt.plot(df['Depth_measured'], df[model], marker='o', label=model)
plt.xlabel('Depth(m)')
plt.ylabel('Error')
plt.legend()
plt.tight_layout()
plt.savefig('lineplot_error_vs_depth.png', dpi=300)
plt.close()

# 4. 热力图（所有点所有模型误差）
plt.figure(figsize=(12, 6))
sns.heatmap(df[model_cols], annot=True, fmt=".2f", cmap='YlOrRd', yticklabels=df['Well'] + '_' + df['Depth_measured'].astype(str))
plt.xlabel('Models')
plt.ylabel('Well & Depth')
plt.tight_layout()
plt.savefig('heatmap_error.png', dpi=300)
plt.close()

print('四种误差分析图已保存为：barplot_per_point.png, boxplot_per_model.png, lineplot_error_vs_depth.png, heatmap_error.png')
