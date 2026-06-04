"""
infer_compare_ovation_point.py  (V4 改造版)
================================================================
论文图 8: 总散点对比 — V4 MLP vs OVATION-Prime
两个模型在 SSJ 真值上的 (Observed_total_flux, Predicted_total_flux) 散点决战.

布局 2 × 3:
    Row 1: OVATION-Prime    Storm | Quiet | All
    Row 2: V4 MLP (Ours)    Storm | Quiet | All

关键变化 (vs v2 版):
  * V4 模型: load_v4_model + run_mlp_4heads_v4 (4 head, 然后 sum 得总通量)
  * V4 测试集: final_test_v4_phys.parquet (mlat 已折叠, 已修复)
  * Storm/Quiet 阈值与 infer_compare_v4_3periods.py 统一:
      Storm: Bz < -10 nT  OR   P_dyn > 5 nPa
      Quiet: Bz >= -2 nT  AND  P_dyn <= 3 nPa
  * 仍用 sample_by_utc 限制 OVATION 推理代价 (SeasonalFluxEstimator 较慢)
  * R 计算与 infer_compare_ovation_radar_new.py 统一 (radar 风格):
      Filter (joint): pred_ovation > 1e-4  AND  pred_mlp_total > 1e-4
      → 两个模型在同一像素集上比较, N_OV == N_MLP

输入:
  /home/docker/data/private/AuroraData/final_test_v4_phys.parquet
  ckpt: /home/docker/code/Aurora_MLP_final/ckpt/ckpt_v4_simple_asym20/...

输出:
  res/evaluation_compare_point_v4/ssj_2x3_ovation_vs_v4mlp.png

用法:
  python infer_compare_ovation_point.py
"""
import sys
import os
import math
import torch
import torch_npu  # noqa: F401
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from scipy.interpolate import RegularGridInterpolator

BASE_DIR = "/home/docker/code/Aurora_MLP_final"
sys.path.append(BASE_DIR)

from infer_v4_utils import load_v4_model, run_mlp_4heads_v4
from auroramaps import ovation as ao


# ===================================================================
# 配置
# ===================================================================
TEST_DATA_PATH = "/home/docker/data/private/AuroraData/final_test_v4_phys_2.parquet"
CKPT_DIR = f"{BASE_DIR}/ckpt/ckpt_v4_simple_different_10"
MODEL_PATH = f"{CKPT_DIR}/aurora_v4_best.pth"
SCALER_PATH = f"{CKPT_DIR}/scaler/sw_scaler_v4.pkl"
OUTPUT_DIR = f"{BASE_DIR}/res/evaluation_compare_point_v4_different_10_v2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Storm/Quiet 阈值 (与 infer_compare_v4_3periods.py 一致)
QUIET_BZ_HI   = -2.0
QUIET_PDYN_HI =  3.0
STORM_BZ_LO   = -10.0
STORM_PDYN_LO =  5.0

# 抽样上限 (按独立 UTC 数控制 OVATION 计算量)
SAMPLE_MAX_UTCS_STORM = 30_000
SAMPLE_MAX_UTCS_QUIET = 30_000
SAMPLE_MAX_UTCS_ALL   = 60_000

PRED_THRESHOLD = 1e-4  # radar 风格 joint filter: 两模型预测都需 > 此阈值


# ===================================================================
# 1. 工具
# ===================================================================
def calculate_newell_coupling(by, bz, v):
    bt = math.sqrt(by**2 + bz**2)
    tc = math.atan2(by, 0.001 if bz == 0 else bz)
    if bt * math.cos(tc) * bz < 0:
        tc += math.pi
    return (v ** (4.0/3.0)) * (abs(math.sin(tc / 2.0)) ** (8.0/3.0)) * (bt ** (2.0/3.0))


def calculate_metrics(y_true, y_pred):
    if len(y_true) < 3:
        return np.nan, np.nan, np.nan
    r, _ = pearsonr(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mse_model = mean_squared_error(y_true, y_pred)
    mse_baseline = np.mean((y_true - np.mean(y_true)) ** 2)
    pe = 1.0 - (mse_model / mse_baseline) if mse_baseline > 0 else np.nan
    return r, rmse, pe


def log_clip(flux):
    return np.log10(np.clip(flux, 1e-6, None))


def joint_filter_mask(df, threshold=PRED_THRESHOLD):
    """radar 风格: 两模型预测都需 > threshold (统一像素集, 公平对比)."""
    return (df['pred_ovation'].values > threshold) & \
           (df['pred_mlp_total'].values > threshold)


def sample_by_utc(df, max_utcs, seed=42):
    unique_utcs = df['utc'].unique()
    if len(unique_utcs) > max_utcs:
        np.random.seed(seed)
        sampled = np.random.choice(unique_utcs, max_utcs, replace=False)
        return df[df['utc'].isin(sampled)].copy()
    return df.copy()


# ===================================================================
# 2. OVATION 推理 (按 UTC 分组 → 插值到测试点)
# ===================================================================
def run_ovation_on_df(df):
    """
    在 df 的每个 (utc, mlat, mlt) 上获取 OVATION 总通量预测.
    返回追加 'pred_ovation' 列的 df.

    规避 OVATION interp_wedge 副作用: 不信任返回的 mlt_2d, 用已知规格 [0, 24, 96].

    🔑 季节校正 (与 dataset_v4.py 保持一致):
      若 df 含 src_hemi 列 (fold 测试集), SH 样本 (src_hemi='S') 调用 OVATION
      时使用 utc + 182d → OVATION 给出"等效 NH 同季节" flux, 与 fold 后
      mlat=|mlat| 的真值在季节相位上对齐. NH 样本 utc 不变.
      ec (Newell coupling) 仍按真实 SW 量计算, 不参与翻转.
    """
    de = ao.FluxEstimator('diff', 'electron energy flux')
    me = ao.FluxEstimator('mono', 'electron energy flux')
    we = ao.FluxEstimator('wave', 'electron energy flux')
    ie = ao.FluxEstimator('ions', 'ion energy flux')

    MLT_AX_OV = np.linspace(0.0, 24.0, 96)

    df = df.copy()
    df['pred_ovation'] = 0.0

    # ---- 构造 OVATION 季节等效 utc 列 _utc_eq ----
    df['_utc_eq'] = pd.to_datetime(df['utc'])
    if 'src_hemi' in df.columns:
        is_south = (df['src_hemi'] == 'S').values
        if is_south.any():
            df.loc[is_south, '_utc_eq'] = (
                pd.to_datetime(df.loc[is_south, 'utc']) + pd.Timedelta(days=182)
            ).values
            print(f"   🔄 OVATION 季节校正: {int(is_south.sum()):,} 条 SH 样本 "
                  f"utc +182d (NH 等效)")
        else:
            print("   ℹ️  src_hemi 全为 N, 无需翻转")
    else:
        print("   ⚠️  数据无 src_hemi 列, OVATION 沿用真实 utc (季节相位可能与 MLP 不一致)")

    grouped = list(df.groupby('_utc_eq'))
    n_groups = len(grouped)
    print(f"   => OVATION: {n_groups:,} 个独立等效 UTC, 预计 ~{n_groups/200:.0f} 分钟...")

    for i, (utc_eq, group) in enumerate(grouped):
        if i > 0 and i % 200 == 0:
            print(f"      [{i:>6}/{n_groups}]")

        vx, vy, vz = group['Vx'].iloc[0], group['Vy'].iloc[0], group['Vz'].iloc[0]
        v = float(np.sqrt(vx*vx + vy*vy + vz*vz))
        bz = group['Bz'].iloc[0]
        by = group['By'].iloc[0]
        ec_val = calculate_newell_coupling(by, bz, v)

        # 用季节等效 utc 调用 OVATION (NH 样本 = 真实 utc, SH 样本 = +182d)
        _, _, f_d = de.get_flux_for_time(utc_eq, ec_val)
        _, _, f_m = me.get_flux_for_time(utc_eq, ec_val)
        _, _, f_w = we.get_flux_for_time(utc_eq, ec_val)
        mlat_2d, _, f_i = ie.get_flux_for_time(utc_eq, ec_val)

        total_flux_ov = f_d + f_m + f_w + f_i

        # mlat 取列, mlt 用已知规格 (规避 interp_wedge 污染 mlt_2d)
        mlat_ax = mlat_2d[:, 0]
        if not np.all(np.diff(mlat_ax) > 0):
            order = np.argsort(mlat_ax)
            mlat_ax = mlat_ax[order]
            total_flux_ov = total_flux_ov[order, :]

        interp = RegularGridInterpolator(
            (mlat_ax, MLT_AX_OV), total_flux_ov,
            bounds_error=False, fill_value=0.0,
        )

        pts = group[['mlat', 'mlt']].values.copy()
        pts[:, 0] = np.abs(pts[:, 0])     # fold 后 mlat 已正, 这里仅兜底
        pts[:, 1] = pts[:, 1] % 24.0
        ov_preds = interp(pts)

        df.loc[group.index, 'pred_ovation'] = ov_preds

    df = df.drop(columns=['_utc_eq'])
    return df


# ===================================================================
# 3. 绘图 (2×3: OVATION top, MLP bottom)
# ===================================================================
def plot_2d_histogram_2x3(df_storm, df_quiet, df_all, save_path):
    fig, axs = plt.subplots(2, 3, figsize=(22, 14))
    fig.patch.set_facecolor('white')

    min_val, max_val = -6.0, 2.5    # 含 1e-6 平台, 与 3periods_total 一致
    bins = np.linspace(min_val, max_val, 140)

    datasets = [df_storm, df_quiet, df_all]
    titles_col = [
        f"Storm  ($B_z<{STORM_BZ_LO}$ or $P_{{dyn}}>{STORM_PDYN_LO}$)",
        f"Quiet  ($B_z\\geq{QUIET_BZ_HI}$ and $P_{{dyn}}\\leq{QUIET_PDYN_HI}$)",
        "All 2014-2015 Test (Sampled)",
    ]
    model_names = ["OVATION-Prime (Baseline)", " AMT (Ours)"]
    pred_cols = ['pred_ovation', 'pred_mlp_total']

    for row in range(2):
        for col in range(3):
            ax = axs[row, col]
            ax.set_facecolor('#f4f4f4')

            df_target = datasets[col]
            mask = joint_filter_mask(df_target)
            y_true = log_clip(df_target['true_flux'].values[mask])
            y_pred = log_clip(df_target[pred_cols[row]].values[mask])

            _, _, _, image = ax.hist2d(
                y_true, y_pred,
                bins=[bins, bins], cmap='viridis',
                norm=LogNorm(), cmin=1,
            )

            ax.plot([min_val, max_val], [min_val, max_val],
                    color='red', linestyle='--', linewidth=2, label='y = x')

            r, rmse, pe = calculate_metrics(y_true, y_pred)
            txt = (f"N = {len(y_true):,}\n"
                   f"R = {r:.3f}\n"
                   f"RMSE = {rmse:.3f}\n"
                   f"PE = {pe:.3f}")
            ax.text(0.04, 0.96, txt, transform=ax.transAxes,
                    fontsize=12, va='top',
                    bbox=dict(boxstyle='round,pad=0.4',
                              facecolor='white', alpha=0.92, edgecolor='gray'))

            ax.set_xlim(min_val, max_val)
            ax.set_ylim(min_val, max_val)
            if row == 1:
                ax.set_xlabel(r'Log$_{10}$ Observed Total Flux',
                              fontsize=14)
            if col == 0:
                ax.set_ylabel(f'{model_names[row]}\nLog$_{{10}}$ Predicted',
                              fontsize=14, weight='bold')
            if row == 0:
                ax.set_title(titles_col[col],
                             fontsize=15, weight='bold', pad=15)
            ax.legend(loc='lower right', framealpha=0.9)
            ax.grid(True, linestyle='-', color='white', alpha=0.6, linewidth=0.5)

            if col == 2:
                cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('Point density', fontsize=12)

    fig.suptitle(
        ' AMT vs OVATION-Prime — Pixel-wise Total Flux Scatter',
        fontsize=18, weight='bold', y=1.00,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"\n✅ 散点对比图已保存: {save_path}")


# ===================================================================
# 4. 主程序
# ===================================================================
def main():
    device = "npu:0"

    # ---- 加载 V4 测试集 ----
    print(f"📥 加载 V4 测试集: {TEST_DATA_PATH}")
    all_df = pd.read_parquet(TEST_DATA_PATH)
    all_df['utc'] = pd.to_datetime(all_df['utc'])
    print(f"   样本数: {len(all_df):,}")

    n_neg = int((all_df['mlat'] < 0).sum())
    if n_neg > 0:
        print(f"⚠️  WARNING: 测试集仍有 {n_neg:,} 行 mlat<0 (OOD), "
              f"先运行 data/merge_test_v4_absmlat.py 修复.")

    # 真实总通量 (4 head 真值之和: ele(d/m/b 之一) + ion)
    all_df['true_flux'] = (
        all_df['ele_energy_flux'].fillna(0.0)
        + all_df['ion_energy_flux'].fillna(0.0)
    )

    # ---- 划分 Storm / Quiet / All ----
    bz = all_df['Bz'].values
    pdyn = all_df['P_dyn'].values
    storm_mask = (bz < STORM_BZ_LO) | (pdyn > STORM_PDYN_LO)
    quiet_mask = (bz >= QUIET_BZ_HI) & (pdyn <= QUIET_PDYN_HI)

    print(f"\n🌪️  时期划分:")
    print(f"   Storm:  {storm_mask.sum():>10,} ({100.*storm_mask.mean():.2f}%)")
    print(f"   Quiet:  {quiet_mask.sum():>10,} ({100.*quiet_mask.mean():.2f}%)")
    print(f"   All:    {len(all_df):>10,}")

    # ---- 按 UTC 抽样 (限 OVATION 计算量) ----
    print(f"\n🔍 按 UTC 抽样 (Storm≤{SAMPLE_MAX_UTCS_STORM:,}, "
          f"Quiet≤{SAMPLE_MAX_UTCS_QUIET:,}, All≤{SAMPLE_MAX_UTCS_ALL:,})...")
    df_storm = sample_by_utc(all_df[storm_mask], SAMPLE_MAX_UTCS_STORM)
    df_quiet = sample_by_utc(all_df[quiet_mask], SAMPLE_MAX_UTCS_QUIET)
    df_all_s = sample_by_utc(all_df,             SAMPLE_MAX_UTCS_ALL)

    # 合并去重 (避免 OVATION/MLP 重复推理)
    eval_df = pd.concat([df_storm, df_quiet, df_all_s])
    eval_df = eval_df[~eval_df.index.duplicated(keep='first')]
    print(f"   去重后总评估点: {len(eval_df):,}")

    # ---- 阶段 A: V4 MLP 推理 ----
    print(f"\n⚡ [1/2] V4 MLP 推理...")
    print(f"   model:  {MODEL_PATH}")
    print(f"   scaler: {SCALER_PATH}")
    model = load_v4_model(model_path=MODEL_PATH, device=device)
    eval_df = run_mlp_4heads_v4(
        eval_df, model, scaler_path=SCALER_PATH, device=device,
        pred_prefix='pred_mlp',
    )
    eval_df['pred_mlp_total'] = (
        eval_df['pred_mlp_d'] + eval_df['pred_mlp_m']
        + eval_df['pred_mlp_b'] + eval_df['pred_mlp_i']
    )
    del model
    if hasattr(torch, 'npu'):
        torch.npu.empty_cache()
    print(f"   ✅ MLP 完成. pred_mlp_total range: "
          f"[{eval_df['pred_mlp_total'].min():.4f}, {eval_df['pred_mlp_total'].max():.2f}]")

    # ---- 阶段 B: OVATION 推理 ----
    print(f"\n⏳ [2/2] OVATION 空间插值...")
    eval_df = run_ovation_on_df(eval_df)
    print(f"   ✅ OVATION 完成. pred_ovation range: "
          f"[{eval_df['pred_ovation'].min():.4f}, {eval_df['pred_ovation'].max():.2f}]")

    # ---- 把推理回灌 storm/quiet/all df ----
    df_storm['pred_mlp_total'] = eval_df.loc[df_storm.index, 'pred_mlp_total']
    df_storm['pred_ovation']  = eval_df.loc[df_storm.index, 'pred_ovation']
    df_quiet['pred_mlp_total'] = eval_df.loc[df_quiet.index, 'pred_mlp_total']
    df_quiet['pred_ovation']  = eval_df.loc[df_quiet.index, 'pred_ovation']
    df_all_s['pred_mlp_total'] = eval_df.loc[df_all_s.index, 'pred_mlp_total']
    df_all_s['pred_ovation']  = eval_df.loc[df_all_s.index, 'pred_ovation']

    # ---- 控制台简表 ----
    print("\n" + "=" * 80)
    print("📊 总通量散点指标 (log10 active points)")
    print("=" * 80)
    header = (f"{'Period':>8s} | {'N_act':>8s} | "
              f"{'OVATION R':>9s} {'OV RMSE':>8s} {'OV PE':>7s}  | "
              f"{'V4 R':>6s} {'V4 RMSE':>8s} {'V4 PE':>7s}")
    print(header)
    print("-" * 80)
    for label, df_ in [('Storm', df_storm), ('Quiet', df_quiet), ('All', df_all_s)]:
        mask = joint_filter_mask(df_)
        true_log = log_clip(df_['true_flux'].values[mask])
        ov_log   = log_clip(df_['pred_ovation'].values[mask])
        mlp_log  = log_clip(df_['pred_mlp_total'].values[mask])
        r_o, rm_o, pe_o = calculate_metrics(true_log, ov_log)
        r_m, rm_m, pe_m = calculate_metrics(true_log, mlp_log)
        print(f"{label:>8s} | {mask.sum():>8,d} | "
              f"{r_o:>+9.3f} {rm_o:>8.3f} {pe_o:>+7.3f}  | "
              f"{r_m:>+6.3f} {rm_m:>8.3f} {pe_m:>+7.3f}")
    print("=" * 80)

    # ---- 绘图 ----
    save_path = os.path.join(OUTPUT_DIR, "ssj_2x3_ovation_vs_v4mlp.png")
    plot_2d_histogram_2x3(df_storm, df_quiet, df_all_s, save_path)


if __name__ == "__main__":
    main()
