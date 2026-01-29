"""
Updated version of train_pressure_model.py with the following improvements:
1. Multi‑branch (Inception‑style) CNN for multi‑scale feature extraction.
2. Two‑layer LSTM for deeper temporal modelling.
3. During final training on the full training set, no early stopping is used (or an optional new validation split can be supplied).
4. Training routine now records and plots both train_loss and val_loss curves.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib
from typing import List, Tuple
from typing import Optional
import seaborn as sns

# ---------------------------
# Globals & IO utils
# ---------------------------
OUTPUT_DIR = "./CNN_LSTM_Multi_wells/第二次训练-替换了N873井密度-T"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_plot(fig_name: str):
    """Save current matplotlib figure to OUTPUT_DIR then close."""
    plt.savefig(os.path.join(OUTPUT_DIR, fig_name))
    plt.close()

# ---------------------------
# Dataset
# ---------------------------
class WellLogDataset(Dataset):
    """Dataset wrapper for windowed well‑log sequences."""

    def __init__(self, sequences: List[Tuple[np.ndarray, float, float]]):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence, target, depth = self.sequences[idx]
        return (
            torch.FloatTensor(sequence),  # [T, F]
            torch.FloatTensor([target]),  # scalar
            torch.tensor(depth, dtype=torch.float32),
        )

# ---------------------------
# Data preparation
# ---------------------------
FEATURES = ["AC", "DEN", "GR", "CN", "LLD"]
TARGET = "pp"

def load_and_preprocess_data(file_path: str, window_size: int = 30, test_size: float = 0.2):
    """Load csv, interpolate, scale, create sliding windows."""
    df = pd.read_csv(file_path)
    # fill NaNs
    df.interpolate(method="linear", inplace=True)
    df.fillna(df.mean(numeric_only=True), inplace=True)

    full_depth = df["Depth"].values.copy()

    # scale
    feat_scaler = StandardScaler()
    df[FEATURES] = feat_scaler.fit_transform(df[FEATURES])
    tgt_scaler = StandardScaler()
    df[TARGET] = tgt_scaler.fit_transform(df[[TARGET]])

    sequences = []
    for i in range(len(df) - window_size):
        seq = df.iloc[i : i + window_size][FEATURES].values.astype(np.float32)
        label = df.iloc[i + window_size - 1][TARGET].astype(np.float32)
        depth = full_depth[i + window_size - 1]
        sequences.append((seq, label, depth))

    np.random.seed(42)
    np.random.shuffle(sequences)

    split_idx = int(len(sequences) * (1 - test_size))
    return sequences[:split_idx], sequences[split_idx:], tgt_scaler, feat_scaler

# ---------------------------
# Model components
# ---------------------------
class InceptionConv1D(nn.Module):
    """Three‑branch 1‑D Inception module (kernels 3, 5, 7)."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float):
        super().__init__()
        # each branch produces out_channels
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.branch5 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.branch7 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.out_channels = out_channels * 3

    def forward(self, x):  # x: [B, F, T]
        return torch.cat([self.branch3(x), self.branch5(x), self.branch7(x)], dim=1)

class CNN_LSTM_Inception(nn.Module):
    def __init__(
        self,
        input_dim: int = 5,
        cnn_channels: int = 32,
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.inception = InceptionConv1D(
            in_channels=input_dim, out_channels=cnn_channels, dropout=dropout_rate
        )
        lstm_input = cnn_channels * 3  # concat branches
        self.lstm = nn.LSTM(
            input_size=lstm_input,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout_rate if lstm_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(lstm_hidden, 1)

    def forward(self, x):  # x: [B, T, F]
        # Inception expects [B,F,T]
        x = x.permute(0, 2, 1)  # [B,F,T]
        x = self.inception(x)    # [B, C*3, T]
        x = x.permute(0, 2, 1)  # [B, T, C*3]
        lstm_out, _ = self.lstm(x)  # [B,T,H]
        out = lstm_out[:, -1, :]  # last timestep
        out = self.dropout(out)
        return self.fc(out).squeeze()

# ---------------------------
# Training helpers
# ---------------------------

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for X, y, _ in loader:
        X, y = X.to(device), y.to(device).squeeze()
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X, y, _ in loader:
            X, y = X.to(device), y.to(device).squeeze()
            pred = model(X)
            loss = criterion(pred, y)
            total_loss += loss.item()
    return total_loss / len(loader)


def train_with_early_stopping(
    model,
    train_loader,
    val_loader,
    device,
    epochs: int = 10,
    lr: float = 1e-3,
    patience: int = 10,
    save_path: str = "best_model.pth",
    fold_idx: Optional[int] = None,
):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val = float("inf")
    wait = 0
    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        print(
            f"Epoch {epoch}/{epochs} | Train {tr_loss:.4f} | Val {val_loss:.4f} | Best {best_val:.4f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            wait = 0
            torch.save(model.state_dict(), save_path)
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered!")
                break

    # Plot both losses
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    if fold_idx is not None:
        save_plot(f"loss_curve_fold{fold_idx}.png")
    else:
        save_plot("loss_curve.png")


# full training without early stopping

def train_full(model, train_loader, device, epochs: int = 50, lr: float = 1e-3):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        losses.append(loss)
        if epoch % 10 == 0 or epoch == 1:
            print(f"[Full] Epoch {epoch}/{epochs} | Loss {loss:.4f}")
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Full Training Loss (no val)")
    save_plot("full_training_loss.png")

# ---------------------------
# Evaluation helper (MAE & R2 + plot)
# ---------------------------

def plot_scatter_and_residual(y_true, y_pred, fold_idx, output_dir=OUTPUT_DIR):
    from sklearn.metrics import r2_score
    residuals = y_pred - y_true
    r2 = r2_score(y_true, y_pred)
    plt.figure(figsize=(12, 5))
    # Scatter plot
    plt.subplot(1, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r-', lw=2)
    plt.xlabel('Actual Pressure')
    plt.ylabel('Predicted Pressure')
    plt.title(f'Actual vs Predicted (R²={r2:.2f})')
    # Residual plot
    plt.subplot(1, 2, 2)
    sns.histplot(residuals, bins=40, kde=True, color='steelblue')
    plt.xlabel('Residual')
    plt.ylabel('Count')
    plt.title('Residual Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'scatter_residual_fold{fold_idx}.png'))
    plt.close()

def infer_and_report(
    model,
    loader,
    tgt_scaler,
    device,
    csv_out: str,
    fig_out: str,
    fold_idx: Optional[int] = None,
):
    model.eval()
    preds, gts, depths = [], [], []
    with torch.no_grad():
        for X, y, depth in loader:
            X = X.to(device)
            output = model(X).cpu().numpy()
            y = y.squeeze().cpu().numpy()
            output = tgt_scaler.inverse_transform(output.reshape(-1, 1)).flatten()
            y = tgt_scaler.inverse_transform(y.reshape(-1, 1)).flatten()
            preds.extend(output)
            gts.extend(y)
            depths.extend(depth.numpy())

    mae = mean_absolute_error(gts, preds)
    mse = np.mean((np.array(gts) - np.array(preds)) ** 2)  # 计算MSE
    r2 = r2_score(gts, preds)
    print(f"Test MAE {mae:.2f} | MSE {mse:.2f} | R² {r2:.4f}")

    df = pd.DataFrame({"Depth": depths, "Actual": gts, "Pred": preds})
    df.to_csv(csv_out, index=False)

    # plot
    sorted_idx = np.argsort(depths)
    plt.figure(figsize=(6, 12))
    plt.plot(np.array(gts)[sorted_idx], np.array(depths)[sorted_idx], color='blue', label="Actual")
    plt.plot(np.array(preds)[sorted_idx], np.array(depths)[sorted_idx], color='red', label="Pred", alpha=0.7)
    # 添加实测压力点
    measured_depths = [2973.9, 3249.65]
    measured_pressures = [30.4, 47.61]
    plt.scatter(measured_pressures, measured_depths, color='green', s=100, marker='o', label='Measured Points', zorder=5)
    plt.ylabel("Depth (m)")
    plt.xlabel("Pressure (MPa)")
    plt.gca().invert_yaxis()
    # plt.title("Pressure Prediction vs Depth")
    plt.legend()
    plt.text(
        0.02,
        0.98,
        f"MAE: {mae:.2f}\nMSE: {mse:.2f}\nR²: {r2:.4f}",
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    save_plot(fig_out)
    # 新增：每一折绘制交汇图和残差分布图
    if fold_idx is not None:
        plot_scatter_and_residual(np.array(gts), np.array(preds), fold_idx)
    return mae, mse, r2

# ---------------------------
# Main flow
# ---------------------------

# 新增：画所有折和full的深度-预测/真实曲线
def plot_all_depth_curves():
    files = [os.path.join(OUTPUT_DIR, f"val_fold{i+1}.csv") for i in range(5)]
    files.append(os.path.join(OUTPUT_DIR, "test_predictions.csv"))
    titles = [f"Fold {i+1}" for i in range(5)] + ["Full"]
    fig, axs = plt.subplots(1, 6, figsize=(18, 8), sharey=True)
    
    # 实测压力点数据
    measured_depths = [2973.9, 3249.65]
    measured_pressures = [30.4, 47.61]
    
    # 收集所有深度数据来确定Y轴范围
    all_depths = []
    for file in files:
        if os.path.exists(file):
            df = pd.read_csv(file)
            all_depths.extend(df["Depth"].values)
    
    min_depth = min(all_depths)
    max_depth = max(all_depths)
    
    for i, (file, title) in enumerate(zip(files, titles)):
        if not os.path.exists(file):
            continue
        df = pd.read_csv(file)
        sort_idx = np.argsort(df["Depth"].to_numpy())
        axs[i].plot(df["Actual"].values[sort_idx], df["Depth"].values[sort_idx], label="Actual", color="blue")
        axs[i].plot(df["Pred"].values[sort_idx], df["Depth"].values[sort_idx], label="Pred", color="red", alpha=0.7)
        
        # 添加实测压力点
        axs[i].scatter(measured_pressures, measured_depths, color='green', s=100, marker='o', label='Measured Points', zorder=5)
        
        axs[i].set_title(title)
        # 强制设置Y轴范围并反转
        axs[i].set_ylim(max_depth, min_depth)  # 反转Y轴
        if i == 0:
            axs[i].set_ylabel("Depth (m)")
        axs[i].set_xlabel("Pressure (MPa)")
        axs[i].legend()
    plt.tight_layout()
    save_plot("all_folds_and_full_depth_curve.png")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # hyper‑parameters
    WINDOW = 30
    BATCH = 32
    EPOCHS = 100
    LR = 1e-3

    data_path = r"E:\AAAA工作-研一\3济阳凹陷\5超压预测\2-机器学习方法\数据\multi_Eaton_index\原始数据\Resform_export_filtered_train2.csv"

    train_seqs, test_seqs, tgt_scaler, feat_scaler = load_and_preprocess_data(
        data_path, WINDOW
    )

    # ------------------- 5‑fold CV to tune model -------------------
    folds = 5
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)

    train_np = np.array(train_seqs, dtype=object)
    fold_mae, fold_r2 = [], []
    metrics_records = []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(train_np)):
        print(f"\n===== Fold {fold+1}/{folds} =====")
        tr_data = train_np[tr_idx].tolist()
        val_data = train_np[val_idx].tolist()

        tr_loader = DataLoader(WellLogDataset(tr_data), batch_size=BATCH, shuffle=True)
        val_loader = DataLoader(WellLogDataset(val_data), batch_size=BATCH, shuffle=False)

        model = CNN_LSTM_Inception().to(device)
        train_with_early_stopping(
            model,
            tr_loader,
            val_loader,
            device,
            epochs=EPOCHS,
            lr=LR,
            save_path=os.path.join(OUTPUT_DIR, f"best_fold{fold+1}.pth"),
            fold_idx=fold+1,
        )

        # load best & evaluate on val
        model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, f"best_fold{fold+1}.pth")))
        mae, mse, r2 = infer_and_report(
            model,
            val_loader,
            tgt_scaler,
            device,
            csv_out=os.path.join(OUTPUT_DIR, f"val_fold{fold+1}.csv"),
            fig_out=f"val_fold{fold+1}_plot.png",
            fold_idx=fold+1,
        )
        fold_mae.append(mae)
        fold_r2.append(r2)
        metrics_records.append({"Fold": f"Fold_{fold+1}", "MAE": mae, "MSE": mse, "R2": r2})

    print(
        f"\n5‑fold CV MAE {np.mean(fold_mae):.2f} ± {np.std(fold_mae):.2f} | R² {np.mean(fold_r2):.4f} ± {np.std(fold_r2):.4f}"
    )

    # ------------------- Final training on all training data -------------------
    full_loader = DataLoader(WellLogDataset(train_seqs), batch_size=BATCH, shuffle=True)
    test_loader = DataLoader(WellLogDataset(test_seqs), batch_size=BATCH, shuffle=False)

    final_model = CNN_LSTM_Inception().to(device)
    train_full(final_model, full_loader, device, epochs=EPOCHS // 2, lr=LR)

    # save scalers and model
    joblib.dump(tgt_scaler, os.path.join(OUTPUT_DIR, "target_scaler.save"))
    joblib.dump(feat_scaler, os.path.join(OUTPUT_DIR, "feature_scaler.save"))
    torch.save(final_model.state_dict(), os.path.join(OUTPUT_DIR, "final_model.pth"))

    full_mae, full_mse, full_r2 = infer_and_report(
        final_model,
        test_loader,
        tgt_scaler,
        device,
        csv_out=os.path.join(OUTPUT_DIR, "test_predictions.csv"),
        fig_out="test_pred_vs_actual.png",
        fold_idx=None  # full时不传fold_idx
    )
    # full scatter_residual单独画
    df_full = pd.read_csv(os.path.join(OUTPUT_DIR, "test_predictions.csv"))
    plot_scatter_and_residual(df_full["Actual"].values, df_full["Pred"].values, fold_idx="full")
    metrics_records.append({"Fold": "FullTest", "MAE": full_mae, "MSE": full_mse, "R2": full_r2})

    # 保存所有折和full test的MAE/R2到csv
    pd.DataFrame(metrics_records).to_csv(os.path.join(OUTPUT_DIR, "cv_test_metrics.csv"), index=False)

    # 新增：画所有折和full的深度-预测/真实曲线
    plot_all_depth_curves()

if __name__ == "__main__":
    main()
