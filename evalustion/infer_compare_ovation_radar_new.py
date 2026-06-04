"""
infer_compare_ovation_radar_new.py  (V4 改造版)
================================================================
论文图 11: 多指标雷达图 — V4 MLP vs OVATION-Prime
6 个高阶地学指标 (3 连续 + 3 分类), 全局平均, 反映模型的综合优劣.

雷达图轴 (顺时针):
  1. ROC AUC          (active vs background 二分类区分能力)
  2. CSI              (Critical Success Index, threshold = 0.5 erg/cm²/s)
  3. Accuracy         (active vs background 分类精度)
  4. Pearson R Skill  (R 映射到 [0,1])
  5. KGE              (Kling-Gupta Efficiency, 替代 PE)
  6. Error Skill      (1 - NMedAE)

数据 = 与 point.py 相同的 V4 测试集采样, 复用 OVATION 插值结果.

输入:
  /home/docker/data/private/AuroraData/final_test_v4_phys.parquet

输出:
  res/evaluation_skill_radar_v4/v4_global_skill_radar.png

用法:
  python infer_compare_ovation_radar_new.py
"""
import sys
import os
import math
import torch
import torch_npu  # noqa: F401
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix, median_absolute_error,
)
from scipy.stats import pearsonr
from scipy.interpolate import RegularGridInterpolator
import warnings
warnings.filterwarnings("ignore")

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
OUTPUT_DIR = f"{BASE_DIR}/res/evaluation_skill_radar_v4_different_10_v2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLE_MAX_UTCS = 60_000      # 控制 OVATION 计算量 (~5 min)
ACTIVE_THRESHOLD = 0.5        # erg/cm²/s, ROC/CSI/Accuracy 用


# ===================================================================
# 1. 工具
# ===================================================================
def calculate_newell_coupling(by, bz, v):
    bt = math.sqrt(by**2 + bz**2)
    tc = math.atan2(by, 0.001 if bz == 0 else bz)
    if bt * math.cos(tc) * bz < 0:
        tc += math.pi
    return (v ** (4.0/3.0)) * (abs(math.sin(tc / 2.0)) ** (8.0/3.0)) * (bt ** (2.0/3.0))


def sample_by_utc(df, max_utcs, seed=42):
    unique_utcs = df['utc'].unique()
    if len(unique_utcs) > max_utcs:
        np.random.seed(seed)
        sampled = np.random.choice(unique_utcs, max_utcs, replace=False)
        return df[df['utc'].isin(sampled)].copy()
    return df.copy()


# ===================================================================
# 2. OVATION 推理 (按 UTC 分组)
# ===================================================================
def run_ovation_on_df(df):
    """
    给 df 追加 'pred_ovation' 列.

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
    print(f"   => OVATION: {n_groups:,} 个独立等效 UTC...")

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

        total = f_d + f_m + f_w + f_i
        mlat_ax = mlat_2d[:, 0]
        if not np.all(np.diff(mlat_ax) > 0):
            order = np.argsort(mlat_ax)
            mlat_ax = mlat_ax[order]
            total = total[order, :]

        interp = RegularGridInterpolator(
            (mlat_ax, MLT_AX_OV), total,
            bounds_error=False, fill_value=0.0,
        )
        pts = group[['mlat', 'mlt']].values.copy()
        pts[:, 0] = np.abs(pts[:, 0])     # fold 后 mlat 已正, 这里仅兜底
        pts[:, 1] = pts[:, 1] % 24.0
        df.loc[group.index, 'pred_ovation'] = interp(pts)

    df = df.drop(columns=['_utc_eq'])
    return df


# ===================================================================
# 3. 高阶指标计算
# ===================================================================
def compute_skill_metrics_v2(eval_df, threshold=ACTIVE_THRESHOLD):
    """
    计算 6 个高阶指标 (R / KGE / NMedAE / Accuracy / ROC AUC / CSI),
    返回原始值 + 雷达图 [0,1] 标度.
    """
    print("\n📐 计算全局平均指标...")
    df_clean = eval_df[
        (eval_df['pred_ovation'] > 1e-4) & (eval_df['pred_mlp_total'] > 1e-4)
    ].copy()
    print(f"   过滤后评估点: {len(df_clean):,}")

    models = ['ovation', 'mlp']
    metrics_raw = {m: {} for m in models}

    true_flux = df_clean['true_flux'].values
    true_log = np.log10(np.clip(true_flux, 1e-6, None))
    t_bin = (true_flux >= threshold).astype(int)
    has_two = len(np.unique(t_bin)) == 2

    data_range = max(true_log.max() - true_log.min(), 1e-6)

    for m in models:
        col = 'pred_ovation' if m == 'ovation' else 'pred_mlp_total'
        pred_flux = df_clean[col].values
        pred_log = np.log10(np.clip(pred_flux, 1e-6, None))
        pred_bin = (pred_flux >= threshold).astype(int)

        # --- 连续值 ---
        if np.std(pred_log) > 1e-6:
            r, _ = pearsonr(true_log, pred_log)
            metrics_raw[m]['R'] = r
        else:
            metrics_raw[m]['R'] = 0.0

        if np.std(true_log) > 1e-6 and np.mean(true_log) != 0:
            alpha = np.std(pred_log) / np.std(true_log)
            beta = np.mean(pred_log) / np.mean(true_log)
            kge = 1.0 - np.sqrt(
                (metrics_raw[m]['R'] - 1)**2 + (alpha - 1)**2 + (beta - 1)**2
            )
            metrics_raw[m]['KGE'] = kge
        else:
            metrics_raw[m]['KGE'] = 0.0

        medae = median_absolute_error(true_log, pred_log)
        metrics_raw[m]['NMedAE'] = medae / data_range

        # --- 分类 ---
        metrics_raw[m]['Accuracy'] = accuracy_score(t_bin, pred_bin)
        if has_two:
            metrics_raw[m]['ROC'] = roc_auc_score(t_bin, pred_flux)
            tn, fp, fn, tp = confusion_matrix(t_bin, pred_bin).ravel()
            metrics_raw[m]['CSI'] = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0.0
        else:
            metrics_raw[m]['ROC'] = 0.5
            metrics_raw[m]['CSI'] = 0.0

    # --- 雷达图 [0,1] 标度 ---
    transformed = {m: [] for m in models}
    categories = [
        'ROC AUC', 'CSI (Threat Score)', 'Accuracy',
        'Pearson R (Skill)', 'KGE (Efficiency)', 'Error Skill (1-NMedAE)',
    ]

    for m in models:
        skill_vec = [
            metrics_raw[m]['ROC'],
            metrics_raw[m]['CSI'],
            metrics_raw[m]['Accuracy'],
            max(0.0, (metrics_raw[m]['R'] + 1.0) / 2.0),
            min(1.0, max(0.0, metrics_raw[m]['KGE'])),
            max(0.0, 1.0 - metrics_raw[m]['NMedAE']),
        ]
        transformed[m] = skill_vec
        metrics_raw[m]['display_vals'] = [
            f"{metrics_raw[m]['ROC']:.3f}",
            f"{metrics_raw[m]['CSI']:.3f}",
            f"{metrics_raw[m]['Accuracy']:.3f}",
            f"{metrics_raw[m]['R']:.3f}",
            f"{metrics_raw[m]['KGE']:.3f}",
            f"NMedAE:\n{metrics_raw[m]['NMedAE']:.3f}",
        ]

    return transformed, categories, metrics_raw


# ===================================================================
# 4. 雷达图绘制
# ===================================================================
def plot_skill_radar(skills, categories, metrics_raw, save_path):
    print("\n🎨 绘制雷达图...")
    n = len(categories)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={'projection': 'polar'})
    fig.patch.set_facecolor('white')

    cfg = {
        'mlp':     {'color': '#ff7f0e', 'linewidth': 4, 'linestyle': '-',
                    'label': ' AMT (Ours)', 'fill_alpha': 0.15, 'text_offset':  0.08},
        'ovation': {'color': '#1f77b4', 'linewidth': 3, 'linestyle': '--',
                    'label': 'OVATION-Prime',    'fill_alpha': 0.0,  'text_offset': -0.10},
    }

    for m in ['ovation', 'mlp']:
        c = cfg[m]
        raw = skills[m]
        closed = raw + raw[:1]
        disp = metrics_raw[m]['display_vals']

        ax.plot(angles, closed, color=c['color'], linewidth=c['linewidth'],
                linestyle=c['linestyle'], label=c['label'], zorder=5)
        if c['fill_alpha'] > 0:
            ax.fill(angles, closed, color=c['color'], alpha=c['fill_alpha'], zorder=4)
        ax.scatter(angles[:-1], raw, color=c['color'], s=100,
                   edgecolors='white', linewidths=2, zorder=10)

        for i, ang in enumerate(angles[:-1]):
            r_pos = raw[i] + c['text_offset']
            if r_pos < 0.1:
                r_pos = 0.15
            ha = 'center'
            if abs(ang) < 1e-6: ha = 'left'
            elif abs(ang - np.pi) < 1e-6: ha = 'right'
            ax.text(ang, r_pos, disp[i], size=13, color=c['color'],
                    ha=ha, va='center', weight='bold', zorder=15,
                    bbox=dict(facecolor='white', alpha=0.6,
                              edgecolor='none', pad=1))

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1.25)
    ax.set_rticks(np.arange(0.2, 1.1, 0.2))
    ax.tick_params(axis='y', labelcolor='gray', labelsize=11)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=15, weight='bold')
    ax.spines['polar'].set_visible(False)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fontsize=15, ncol=2)
    fig.suptitle(' AMT vs OVATION — Global Skill Radar',
                 fontsize=22, weight='bold', y=0.92)
    plt.tight_layout(pad=4.0)
    plt.savefig(save_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"✅ 雷达图已保存: {save_path}")


# ===================================================================
# 5. 主程序
# ===================================================================
def main():
    device = "npu:0"

    print(f"📥 加载 V4 测试集: {TEST_DATA_PATH}")
    all_df = pd.read_parquet(TEST_DATA_PATH)
    all_df['utc'] = pd.to_datetime(all_df['utc'])
    print(f"   样本数: {len(all_df):,}")

    n_neg = int((all_df['mlat'] < 0).sum())
    if n_neg > 0:
        print(f"⚠️  WARNING: 测试集仍有 {n_neg:,} 行 mlat<0!")

    all_df['true_flux'] = (
        all_df['ele_energy_flux'].fillna(0.0)
        + all_df['ion_energy_flux'].fillna(0.0)
    )

    print(f"\n🔍 按 UTC 抽样 (max {SAMPLE_MAX_UTCS:,})...")
    eval_df = sample_by_utc(all_df, SAMPLE_MAX_UTCS)
    print(f"   抽样后: {len(eval_df):,} 点 (来自 {eval_df['utc'].nunique():,} UTC)")

    # ---- V4 MLP 推理 ----
    print(f"\n⚡ [1/2] V4 MLP 推理...")
    print(f"   model: {MODEL_PATH}")
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
    print(f"   ✅ MLP done.")

    # ---- OVATION 推理 ----
    print(f"\n⏳ [2/2] OVATION 推理...")
    eval_df = run_ovation_on_df(eval_df)
    print(f"   ✅ OVATION done.")

    # ---- 指标 ----
    skills, categories, metrics_raw = compute_skill_metrics_v2(eval_df)

    # 控制台简表
    print("\n" + "=" * 90)
    print("📊 全局指标 (raw)")
    print("=" * 90)
    print(f"{'Metric':>20s} | {'OVATION':>10s} | {'V4 MLP':>10s} | {'Δ (V4 - OV)':>12s}")
    print("-" * 90)
    for k in ['R', 'KGE', 'NMedAE', 'Accuracy', 'ROC', 'CSI']:
        ov = metrics_raw['ovation'][k]
        ml = metrics_raw['mlp'][k]
        delta = ml - ov
        print(f"{k:>20s} | {ov:>+10.4f} | {ml:>+10.4f} | {delta:>+12.4f}")
    print("=" * 90)

    # ---- 绘图 ----
    save_path = os.path.join(OUTPUT_DIR, "v4_global_skill_radar.png")
    plot_skill_radar(skills, categories, metrics_raw, save_path)


if __name__ == "__main__":
    main()
