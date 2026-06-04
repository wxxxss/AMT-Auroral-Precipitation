"""
infer_point.py  ·  按塔 per-head 散点评估 (1 行 × 4 列)
==============================================================
对比策略 (修正 bug):
    训练 target 是互斥的:
        target_diffuse    = ele_energy_flux  if aurora_type == 1 else 0
        target_mono       = ele_energy_flux  if aurora_type == 2 else 0
        target_broadband  = ele_energy_flux  if aurora_type == 3 else 0
        target_ion        = ion_energy_flux
    旧脚本用 sum(4 heads) vs (ele + ion) 做散点, 必然系统性高估.

    本脚本按塔分别评估: pred[k] vs target_k, 且只在 "target>0 或 pred>0" 的
    活跃/误报像素上计算 R / RMSE / PE.

输出: 1 × 4 子图, 全量 2014 测试集.
"""
import sys
import os
import torch
import torch_npu  # noqa: F401
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

BASE_DIR = "/home/docker/code/Aurora_MLP_final"
sys.path.append(BASE_DIR)

from infer_v4_utils import load_v4_model, build_v4_dataset, predict_v4_batched


HEAD_NAMES = ['Diffuse', 'Monoenergetic', 'Broadband', 'Ion']


# ================= 1. 指标 =================
def calculate_metrics(y_true, y_pred):
    if len(y_true) < 3:
        return np.nan, np.nan, np.nan
    r, _ = pearsonr(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mse_model = mean_squared_error(y_true, y_pred)
    mse_baseline = np.mean((y_true - np.mean(y_true)) ** 2)
    pe = 1.0 - (mse_model / mse_baseline) if mse_baseline > 0 else np.nan
    return r, rmse, pe


def filter_and_log(true_flux, pred_flux, threshold=1e-4):
    """剔除真实+预测都极低(双暗)的像素, log10 化."""
    active_mask = (true_flux > threshold) | (pred_flux > threshold)
    t = np.log10(np.clip(true_flux[active_mask], 1e-6, None))
    p = np.log10(np.clip(pred_flux[active_mask], 1e-6, None))
    return t, p


# ================= 2. 绘图: 1×4 按塔散点 =================
def plot_per_head_scatter(true_by_head, pred_by_head, save_path):
    fig, axs = plt.subplots(1, 4, figsize=(26, 6.5))
    fig.patch.set_facecolor('white')

    # bins 从 -6 开始, 让 target=-6 (背景) 的假阳性点可见.
    # 旧版 -4 起点会裁掉 log(1e-6)=-6 的背景像素, 散点图看不到 'pred 高但 target=-6' 的失败区域.
    min_val, max_val = -6.0, 2.5
    bins = np.linspace(min_val, max_val, 120)

    last_image = None
    for k in range(4):
        ax = axs[k]
        ax.set_facecolor('#f4f4f4')

        t_log, p_log = filter_and_log(true_by_head[k], pred_by_head[k])
        r, rmse, pe = calculate_metrics(t_log, p_log)

        h, xe, ye, image = ax.hist2d(
            t_log, p_log, bins=[bins, bins],
            cmap='viridis', norm=LogNorm(), cmin=1,
        )
        last_image = image

        ax.plot([min_val, max_val], [min_val, max_val],
                color='red', linestyle='--', linewidth=2, label='y = x')

        txt = (f"Active points: {len(t_log):,}\n"
               f"R = {r:.3f}\n"
               f"RMSE = {rmse:.3f}\n"
               f"PE = {pe:.3f}")
        ax.text(0.04, 0.96, txt, transform=ax.transAxes,
                fontsize=11, va='top',
                bbox=dict(boxstyle='round,pad=0.4',
                          facecolor='white', alpha=0.92, edgecolor='gray'))

        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.set_xlabel(r'Log$_{10}$  Observed flux  [ergs cm$^{-2}$ s$^{-1}$]',
                      fontsize=12)
        if k == 0:
            ax.set_ylabel(r'MLP prediction  Log$_{10}$ flux',
                          fontsize=12, weight='bold')
        ax.set_title(HEAD_NAMES[k], fontsize=15, weight='bold', pad=12)
        ax.legend(loc='lower right', framealpha=0.9, fontsize=10)
        ax.grid(True, linestyle='-', color='white', alpha=0.6, linewidth=0.5)

    # 共享 colorbar
    cbar = fig.colorbar(last_image, ax=axs, fraction=0.018, pad=0.02)
    cbar.set_label('Point density (count)', fontsize=11)

    fig.suptitle('Per-Head Scatter (MLP vs SSUSI, 2014 test set)',
                 fontsize=17, weight='bold', y=1.01)
    plt.savefig(save_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"✅ 1×4 per-head 散点图已保存: {save_path}")


# ================= 3. 主程序 =================
def main():
    # ============ V4 路径 ============
    TEST_DATA_PATH = "/home/docker/data/private/AuroraData/final_test_v4_phys.parquet"
    CKPT_DIR = "/home/docker/code/Aurora_MLP_final/ckpt/ckpt_v4_simple_different_10"
    MODEL_PATH = f"{CKPT_DIR}/aurora_v4_best.pth"
    SCALER_PATH = f"{CKPT_DIR}/scaler/sw_scaler_v4.pkl"
    OUTPUT_DIR = "/home/docker/code/Aurora_MLP_final/res/evaluation_point_v4_phys_different_10"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "npu:0"

    print("📥 加载 2014 年全量测试集...")
    df = pd.read_parquet(TEST_DATA_PATH)

    # ---- 构造 4 塔互斥 target (和 dataset._prepare_targets 口径一致) ----
    ele = df['ele_energy_flux'].fillna(0.0).values
    ion = df['ion_energy_flux'].fillna(0.0).values
    at = df['aurora_type'].fillna(0).values

    true_by_head = [
        np.where(at == 1, ele, 0.0),   # Diffuse
        np.where(at == 2, ele, 0.0),   # Mono
        np.where(at == 3, ele, 0.0),   # Broadband
        ion,                           # Ion
    ]

    # ---- V4 MLP 推理 (共享 backbone + 4 独立塔) ----
    print(f"\n⚡ V4 MLP 推理 (N={len(df):,}) ...")
    model = load_v4_model(model_path=MODEL_PATH, device=device)
    eval_ds = build_v4_dataset(df, scaler_path=SCALER_PATH)
    preds = predict_v4_batched(model, eval_ds, device=device, batch_size=32768)
    print("✅ 推理完成")

    pred_by_head = [preds[:, k] for k in range(4)]

    # ---- 打印每塔活跃样本 & 指标摘要 ----
    print("\n" + "=" * 76)
    print(f"{'Head':>14s} | {'Active N':>10s} | {'R':>6s} | {'RMSE':>6s} | {'PE':>6s}")
    print('-' * 76)
    for k, name in enumerate(HEAD_NAMES):
        t_log, p_log = filter_and_log(true_by_head[k], pred_by_head[k])
        r, rmse, pe = calculate_metrics(t_log, p_log)
        print(f"{name:>14s} | {len(t_log):>10,d} | {r:>6.3f} | {rmse:>6.3f} | {pe:>6.3f}")
    print("=" * 76)

    save_fig_path = os.path.join(OUTPUT_DIR, "mlp_per_head_scatter_1x4.png")
    plot_per_head_scatter(true_by_head, pred_by_head, save_fig_path)


if __name__ == "__main__":
    main()
