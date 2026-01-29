import os
import numpy as np
import pandas as pd
import torch
import joblib
from torch.utils.data import Dataset, DataLoader
from CNN_LSTM_train import CNN_LSTM_Inception  # assumes you have the model class in model.py or same script
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# --- Configuration ---
WINDOW_SIZE = 30
FEATURES = ["AC", "DEN", "GR", "CN", "LLD"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "./CNN_LSTM_Multi_wells/第二次训练-替换了N873井密度-T/final_model.pth"
SCALER_PATH = "./CNN_LSTM_Multi_wells/第二次训练-替换了N873井密度-T/target_scaler.save"
FEATURE_SCALER_PATH = "./CNN_LSTM_Multi_wells/第二次训练-替换了N873井密度-T/feature_scaler.save"
OUTPUT_DIR = "./CNN_LSTM_Multi_wells/第二次训练-替换了N873井密度-T/CNN_LSTM_pred"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Dataset for prediction ---
class PredictDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq, depth = self.sequences[idx]
        return torch.FloatTensor(seq), torch.tensor(depth, dtype=torch.float32)

# --- Preprocessing ---
def preprocess_new_data(file_path, window_size=30):
    df = pd.read_csv(file_path)
    df.interpolate(method="linear", inplace=True)
    df.fillna(df.mean(numeric_only=True), inplace=True)

    full_depth = df["Depth"].values.copy()
    feat_scaler = joblib.load(FEATURE_SCALER_PATH)
    df[FEATURES] = feat_scaler.transform(df[FEATURES])

    sequences = []
    for i in range(len(df) - window_size):
        seq = df.iloc[i:i+window_size][FEATURES].values.astype(np.float32)
        depth = full_depth[i + window_size - 1]
        sequences.append((seq, depth))

    return sequences, full_depth

# --- 模型加载优化 ---
def load_models(model_paths, device):
    models = []
    for model_path in model_paths:
        model = CNN_LSTM_Inception().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        models.append(model)
    return models

# --- 内存优化：直接保存预测结果 ---
def save_predictions_to_csv(depths, predictions, file_path, output_dir):
    result_df = pd.DataFrame({"Depth": depths, "Predicted_Pressure": predictions})
    output_csv = os.path.join(output_dir, os.path.basename(file_path).replace(".csv", "_predicted.csv"))
    result_df.to_csv(output_csv, index=False)
    print(f"Predictions saved to: {output_csv}")

# --- 结果保存优化：提升图片质量 ---
def save_figure(fig, output_png):
    fig.tight_layout()
    fig.savefig(output_png, dpi=300)
    print(f"Figure saved to: {output_png}")

# --- Inference ---
def predict_all_folds_and_plot(file_path, xlim=None, ylim=None, thin_step=1):
    import glob
    model_dir = "./CNN_LSTM_Inception"
    model_paths = sorted(glob.glob(os.path.join(model_dir, "best_fold*.pth")))
    final_model_path = os.path.join(model_dir, "final_model.pth")
    model_paths.append(final_model_path)
    n_folds = len(model_paths)
    if n_folds == 0:
        print("No fold models found.")
        return
    sequences, full_depth = preprocess_new_data(file_path, WINDOW_SIZE)
    dataset = PredictDataset(sequences)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    tgt_scaler = joblib.load(SCALER_PATH)
    feat_scaler = joblib.load(FEATURE_SCALER_PATH)

    measured_depths = [2973.9,3249.65]
    measured_pressures = [30.4, 48.61]

    fig, axes = plt.subplots(1, n_folds, figsize=(3*n_folds, 10), sharey=True)
    if n_folds == 1:
        axes = [axes]

    all_predictions = []
    all_depths = None

    models = load_models(model_paths, DEVICE)  # 一次性加载所有模型

    for i, model in enumerate(models):
        predictions = []
        depths = []

        with torch.no_grad():
            for X, depth in loader:
                X = X.to(DEVICE)
                output = model(X).cpu().numpy()
                output = tgt_scaler.inverse_transform(output.reshape(-1, 1)).flatten()
                predictions.extend(output)
                depths.extend(depth.numpy())

        sorted_idx = np.argsort(depths)
        predictions = np.array(predictions)[sorted_idx]
        depths = np.array(depths)[sorted_idx]

        predictions_thin = predictions[::thin_step]
        depths_thin = depths[::thin_step]

        ax = axes[i]
        ax.plot(predictions_thin, depths_thin, label="Predicted Pressure")
        ax.scatter(measured_pressures, measured_depths, color='red', s=100, marker='o', label='Measured Points', zorder=5)
        ax.set_xlabel("Pressure (MPa)")
        if i == 0:
            ax.set_ylabel("Depth (m)")
        if i < n_folds - 1:
            ax.set_title(f"Fold {i+1}")
        else:
            ax.set_title("Final Model")
        ax.legend()
        if xlim is not None:
            ax.set_xlim(xlim)
        
        # 设置Y轴范围，优先使用传入的ylim参数
        if ylim is not None:
            # 如果传入了ylim，使用传入的值并反转
            ax.set_ylim(ylim[1], ylim[0])  # 反转Y轴
        else:
            # 否则使用数据的最大最小值并反转
            ax.set_ylim(max(depths), min(depths))  # Y轴反转

        if all_depths is None:
            all_depths = depths
        all_predictions.append(predictions)

    # 保存图片
    output_png = os.path.join(OUTPUT_DIR, os.path.basename(file_path).replace(".csv", "_allfolds_plot.png"))
    save_figure(fig, output_png)

    # 保存所有fold的预测到csv
    result_df = pd.DataFrame({"Depth": all_depths})
    for i, preds in enumerate(all_predictions):
        if i < n_folds - 1:
            result_df[f"Fold{i+1}"] = preds
        else:
            result_df["FinalModel"] = preds
    merged_csv = os.path.join(OUTPUT_DIR, os.path.basename(file_path).replace('.csv', '_allfolds_pred.csv'))
    result_df.to_csv(merged_csv, index=False)
    print(f"All folds predictions saved to: {merged_csv}")

    # 统计实测点误差并保存
    if measured_depths and measured_pressures and all_depths is not None:
        all_depths_np = np.array(all_depths)
        error_rows = []
        for md, mp in zip(measured_depths, measured_pressures):
            # 找到预测中与实测深度最近的索引
            idx = (np.abs(all_depths_np - md)).argmin()
            row = {
                "Measured_Depth": md,
                "Measured_Pressure": mp
            }
            for i, preds in enumerate(all_predictions):
                pred_val = preds[idx]
                error = abs(pred_val - mp)
                if i < n_folds - 1:
                    row[f"Fold{i+1}_Error"] = error
                else:
                    row["FinalModel_Error"] = error
            error_rows.append(row)
        error_df = pd.DataFrame(error_rows)
        error_csv = os.path.join(OUTPUT_DIR, os.path.basename(file_path).replace('.csv', '_measured_errors.csv'))
        error_df.to_csv(error_csv, index=False)
        print(f"Measured point errors saved to: {error_csv}")

# --- Entry point ---
if __name__ == "__main__":
    # 用户可自定义坐标范围，如predict(..., xlim=(20,70), ylim=(2500,3500))
    predict_all_folds_and_plot(
        r"E:\AAAA工作-研一\3济阳凹陷\5超压预测\2-机器学习方法\数据\N873_forpred_filtered.csv", 
        # xlim=(20, 70),  # 压力范围
        # ylim=(2150, 3223),  # 深度范围（注意：由于Y轴反转，这里应该是(浅层, 深层)）
        thin_step=20
    )
