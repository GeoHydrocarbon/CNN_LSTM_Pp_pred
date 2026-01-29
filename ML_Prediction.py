import numpy as np
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
from smoothing import smooth_predictions_dataframe

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def predict(file_path, model_path, output_path, thin_ratio=1, rho_water_g_cm3=1.03, smoothing_method="savgol", window_length=None, polyorder=2, ma_window=None, median_kernel=None):
    df = pd.read_csv(file_path)

    if "Depth" not in df.columns:
        raise ValueError("数据中缺少 'Depth' 列，无法绘图和保存")

    # 输入特征 X 为除了 Depth 和 pp 之外的列
    X = df.drop(columns=["Depth", "pp"], errors="ignore")
    depth = df["Depth"]
    true_pp = df["pp"] if "pp" in df.columns else None

    models = ["DecisionTree", "RandomForest", "MLP", "XGBoost", "CatBoost", "SVR"]
    predictions = {}

    for model_name in models:
        model_file = os.path.join(model_path, f"{model_name}.pkl")
        if os.path.exists(model_file):
            print(f"加载模型：{model_name}")
            model = joblib.load(model_file)
            try:
                y_pred = model.predict(X)
                predictions[model_name] = y_pred
            except Exception as e:
                print(f"{model_name} 预测失败: {e}")
        else:
            print(f"模型文件缺失：{model_name}")

    # 构建预测 DataFrame（包含深度、实际压力、每个模型预测）
    predictions_df = pd.DataFrame(predictions)
    predictions_df.insert(0, "Depth", depth.values)
    if true_pp is not None:
        predictions_df.insert(1, "Actual_Pressure", true_pp.values)

    # 按深度升序排序
    predictions_df = predictions_df.sort_values(by="Depth").reset_index(drop=True)

    # 保存为 CSV
    predictions_df.to_csv(output_path, index=False)
    print(f"预测结果保存至：{output_path}")

    # 平滑预测压力
    predictions_smoothed_df = smooth_predictions_dataframe(
        predictions_df,
        models,
        method=smoothing_method,
        window_length=window_length,
        polyorder=polyorder,
        ma_window=ma_window,
        median_kernel=median_kernel
    )

    # 保存平滑后的预测压力 CSV
    smoothed_csv_path = os.path.splitext(output_path)[0] + "_smoothed.csv"
    predictions_smoothed_df.to_csv(smoothed_csv_path, index=False)
    print(f"平滑后的预测结果保存至：{smoothed_csv_path}")

    # 计算静水压力(MPa)与压力系数
    rho_kg_m3 = rho_water_g_cm3 * 1000.0
    hydrostatic_mpa = compute_hydrostatic_pressure_mpa(predictions_smoothed_df["Depth"].values, rho_kg_m3)
    coef_df = predictions_smoothed_df.copy()
    coef_df.insert(1, "Hydrostatic_MPa", hydrostatic_mpa)
    for model_name in models:
        if model_name in coef_df.columns:
            coef_df[model_name] = coef_df[model_name] / coef_df["Hydrostatic_MPa"].replace(0, np.nan)
    if "Actual_Pressure" in coef_df.columns:
        coef_df["Actual_Coefficient"] = coef_df["Actual_Pressure"] / coef_df["Hydrostatic_MPa"].replace(0, np.nan)

    # 保存压力系数 CSV
    coef_csv_path = os.path.splitext(output_path)[0] + "_coef.csv"
    coef_df.to_csv(coef_csv_path, index=False)
    print(f"压力系数数据保存至：{coef_csv_path}")

    # 可视化（使用平滑后的压力）
    plot_predictions(predictions_smoothed_df, model_path, file_path, thin_ratio)
    # 绘制压力系数图
    plot_pressure_coefficients(coef_df, model_path, file_path, rho_kg_m3, thin_ratio)
    # 绘制原始与平滑叠加图
    plot_overlay_raw_vs_smoothed(predictions_df, predictions_smoothed_df, model_path, file_path, thin_ratio)


def save_measured_point_errors(predictions_df, measured_depths, measured_pressures, output_path):
    """
    统计每个实测点与预测结果中最近深度的每种算法预测压力值的绝对误差，并保存为CSV
    """
    model_names = [col for col in predictions_df.columns if col not in ["Depth", "Actual_Pressure"]]
    results = []
    for depth_measured, pressure_measured in zip(measured_depths, measured_pressures):
        # 找到最近的深度索引
        idx = (predictions_df["Depth"] - depth_measured).abs().idxmin()
        nearest_depth = predictions_df.loc[idx, "Depth"]
        row = {
            "Depth_measured": depth_measured,
            "Pressure_measured": pressure_measured,
            "Nearest_Depth": nearest_depth
        }
        for model in model_names:
            pred_pressure = predictions_df.loc[idx, model]
            error = abs(pred_pressure - pressure_measured)
            row[f"{model}_Error"] = error
        results.append(row)
    # 保存为CSV
    error_df = pd.DataFrame(results)
    # 生成输出路径
    error_csv_path = os.path.splitext(output_path)[0] + "_measured_point_errors.csv"
    error_df.to_csv(error_csv_path, index=False)
    print(f"实测点误差统计已保存至：{error_csv_path}")


def plot_predictions(predictions_df, model_path, input_filename, thin_ratio=1):
    model_names = [col for col in predictions_df.columns if col not in ["Depth", "Actual_Pressure"]]
    
    # 实测压力点（与CNN_LSTM_pred.py一致的输入方式）
    measured_depths = [3042.4, 3275.1]
    measured_pressures = [34.01, 48.16]
    
    # 统计并保存实测点误差
    save_measured_point_errors(predictions_df, measured_depths, measured_pressures, input_filename)
    
    # 创建子图，横向排列
    n_models = len(model_names)
    if n_models == 0:
        print("没有可用的模型预测结果")
        return
        
    fig, axes = plt.subplots(1, n_models, figsize=(3*n_models, 8), sharey=False)
    if n_models == 1:
        axes = [axes]
    
    # 抽稀
    df_thin = predictions_df.iloc[::thin_ratio, :]

    for i, model_name in enumerate(model_names):
        ax = axes[i]
        
        # 绘制预测结果
        ax.plot(df_thin[model_name], df_thin["Depth"], 
                label="Predicted")
        
        # 绘制实测压力点
        for depth, pressure in zip(measured_depths, measured_pressures):
            ax.scatter(pressure, depth, color='red', s=100, marker='o', 
                      label='Measured Points' if depth == measured_depths[0] else "", zorder=5)
        
        ax.set_xlabel("Pressure (MPa)")
        if i == 0:
            ax.set_ylabel("Depth (m)")
        ax.set_title(f"{model_name}")
        ax.legend()
        # 确保y轴反转并设置范围
        ax.invert_yaxis()
        # 明确设置y轴范围，确保深度从深到浅排列
        depth_min, depth_max = df_thin["Depth"].min(), df_thin["Depth"].max()
        ax.set_ylim(depth_max, depth_min)
        # ax.set_ylim(depth_max, 2150)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 根据输入文件名生成输出文件名
    base_name = os.path.splitext(os.path.basename(input_filename))[0]
    save_path = os.path.join(model_path, f"{base_name}_all_models_predictions.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"所有模型预测对比图已保存至：{save_path}")

def compute_hydrostatic_pressure_mpa(depth_array_m, rho_kg_m3, g=9.81):
    """
    静水压力(MPa) = rho(kg/m3) * g(m/s2) * depth(m) / 1e6
    """
    depth = np.asarray(depth_array_m)
    return rho_kg_m3 * g * depth / 1e6


def plot_pressure_coefficients(coef_df, model_path, input_filename, rho_kg_m3, thin_ratio=1):
    model_names = [col for col in coef_df.columns if col not in ["Depth", "Actual_Pressure", "Hydrostatic_MPa", "Actual_Coefficient"]]

    # 实测压力点（与 CNN_LSTM_pred.py 一致的输入方式）
    measured_depths = [3042.4, 3275.1]
    measured_pressures = [34.01, 48.16]

    # 计算实测压力系数
    measured_hydrostatic = compute_hydrostatic_pressure_mpa(np.array(measured_depths), rho_kg_m3)
    measured_coefficients = np.array(measured_pressures) / measured_hydrostatic

    n_models = len(model_names)
    if n_models == 0:
        print("没有可用的模型压力系数结果")
        return

    fig, axes = plt.subplots(1, n_models, figsize=(3*n_models, 8), sharey=False)
    if n_models == 1:
        axes = [axes]

    df_thin = coef_df.iloc[::thin_ratio, :]

    for i, model_name in enumerate(model_names):
        ax = axes[i]

        ax.plot(df_thin[model_name], df_thin["Depth"], label="Coefficient")

        for depth, coef_val in zip(measured_depths, measured_coefficients):
            ax.scatter(coef_val, depth, color='red', s=100, marker='o', label='Measured Coef' if depth == measured_depths[0] else "", zorder=5)

        ax.set_xlabel("Pressure Coefficient")
        if i == 0:
            ax.set_ylabel("Depth (m)")
        ax.set_title(f"{model_name} Coef")
        ax.legend()
        ax.invert_yaxis()
        depth_min, depth_max = df_thin["Depth"].min(), df_thin["Depth"].max()
        ax.set_ylim(depth_max, depth_min)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    base_name = os.path.splitext(os.path.basename(input_filename))[0]
    save_path = os.path.join(model_path, f"{base_name}_all_models_pressure_coefficient.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"所有模型压力系数对比图已保存至：{save_path}")


def plot_overlay_raw_vs_smoothed(raw_df, smoothed_df, model_path, input_filename, thin_ratio=1):
    """
    在同一张图上叠加展示：原始预测压力 vs 平滑后预测压力。
    每个模型一个子图，x=Pressure(MPa), y=Depth(m)。
    """
    model_names = [col for col in smoothed_df.columns if col not in ["Depth", "Actual_Pressure"]]

    n_models = len(model_names)
    if n_models == 0:
        print("没有可用的模型预测结果用于叠加图")
        return

    fig, axes = plt.subplots(1, n_models, figsize=(3*n_models, 8), sharey=False)
    if n_models == 1:
        axes = [axes]

    raw_thin = raw_df.iloc[::thin_ratio, :]
    sm_thin = smoothed_df.iloc[::thin_ratio, :]

    for i, model_name in enumerate(model_names):
        ax = axes[i]

        # 原始预测
        if model_name in raw_thin.columns:
            ax.plot(raw_thin[model_name], raw_thin["Depth"], label="Raw", color="#1f77b4", alpha=0.7)

        # 平滑预测
        ax.plot(sm_thin[model_name], sm_thin["Depth"], label="Smoothed", color="#ff7f0e", linewidth=2)

        ax.set_xlabel("Pressure (MPa)")
        if i == 0:
            ax.set_ylabel("Depth (m)")
        ax.set_title(f"{model_name} Raw vs Smoothed")
        ax.legend()
        ax.invert_yaxis()
        depth_min, depth_max = sm_thin["Depth"].min(), sm_thin["Depth"].max()
        ax.set_ylim(depth_max, depth_min)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    base_name = os.path.splitext(os.path.basename(input_filename))[0]
    save_path = os.path.join(model_path, f"{base_name}_raw_vs_smoothed.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"原始与平滑叠加图已保存至：{save_path}")

if __name__ == "__main__":
    file_path = r"E:\AAAA工作-研一\3济阳凹陷\5超压预测\2-机器学习方法\数据\NY1-6HF_forpred_filtered.csv"
    model_path = r"E:\AAAA工作-研一\3济阳凹陷\5超压预测\2-机器学习方法\数据\models_N873_ensemble2"
    output_path = r"E:\AAAA工作-研一\3济阳凹陷\5超压预测\2-机器学习方法\数据\models_N873_ensemble2\预测数据\NY1-6HF_forpred_filtered_predicted_ML.csv"

    predict(
        file_path,
        model_path,
        output_path,
        thin_ratio=20,
        rho_water_g_cm3=1.03,
        smoothing_method="ma", # 可选："savgol"、"ma"、"median"
        window_length=None,  # 自适应（savgol）
        polyorder=2,        # （savgol）
        ma_window=11,     # （ma）
        median_kernel=9  # （median）奇数窗口大小，如 5/7/9
    )
