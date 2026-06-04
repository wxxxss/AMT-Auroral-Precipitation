"""
infer_compare_ovation_HP.py  (V4 改造版)
================================================================
论文图 9: 磁暴时段 Hemispheric Power 演化对比 — V4 MLP vs OVATION.

布局: 4 行 × 1 列, 共享 X 轴 (时间)
  Panel 1: IMF Bz / By
  Panel 2: P_dyn (左) / V (右)        ← 把 P_dyn 与 V 合并到双 y 轴
  Panel 3: Newell coupling dΦ/dt        ← 新增: 太阳风—磁层耦合驱动量
  Panel 4: HP (GW) — V4 MLP vs OVATION

数据来源:
  /home/docker/data/private/AuroraData/omni_shortterm_2008_2014.parquet
    虽然名字带 _2014, 实际涵盖 2008-2015-12-31 (5-min 等间隔, 168 维 lag).
  *不读* V4 测试集, 因为 final_test_v4_phys.parquet 实际只覆盖 2014 年,
  而 St. Patrick's 2015-03-17 不在其中.

事件: 2015-03-17 St. Patrick's Day Storm (Dst min ≈ -222 nT)
  默认窗口 04:00 ~ 18:00 UT, 共 14 小时
  这是太阳周 24 最强地磁暴, 论文里非常有代表性. 2014 年没有可比量级事件.

预测策略:
  1. 在 OMNI parquet 中取窗口内所有 5-min 行 (≈ 168 步), 直接遍历 (天然 5-min 等间隔)
  2. 每个 UTC 取一行 SW (mlat/mlt 用统一网格覆盖), 经 predict_grid_v4 获得 4 头通量
  3. 4 头求和后乘 area_2d 即得 HP
  4. OVATION: Newell coupling + 4 estimator → 总通量 → ×area → HP

输出:
  res/evaluation_hp_v4_asym20/v4_hp_evolution_storm.png

用法:
  python infer_compare_ovation_HP.py
"""
import sys
import os
import math
from datetime import datetime
import torch
import torch_npu  # noqa: F401
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

BASE_DIR = "/home/docker/code/Aurora_MLP_final"
sys.path.append(BASE_DIR)

from infer_v4_utils import load_v4_model, predict_grid_v4
from auroramaps import ovation as ao


# ===================================================================
# 配置
# ===================================================================
OMNI_DATA_PATH = "/home/docker/data/private/AuroraData/omni_shortterm_2014_2015.parquet"
CKPT_DIR = f"{BASE_DIR}/ckpt/ckpt_v4_simple_different_10"
MODEL_PATH = f"{CKPT_DIR}/aurora_v4_best.pth"
SCALER_PATH = f"{CKPT_DIR}/scaler/sw_scaler_v4.pkl"
OUTPUT_DIR = f"{BASE_DIR}/res/evaluation_hp_v4_different_10"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 事件窗口
START_TIME = datetime(2015, 3, 17, 4, 0)
END_TIME   = datetime(2015, 3, 17, 18, 0)

# 网格 (与 infer_storm_snapshot.py 一致)
MLAT_1D = np.linspace(50, 90, 80)
MLT_1D  = np.linspace(0, 24, 144)


# ===================================================================
# 1. 工具
# ===================================================================
def calculate_newell_coupling(by, bz, v):
    bt = math.sqrt(by**2 + bz**2)
    tc = math.atan2(by, 0.001 if bz == 0 else bz)
    if bt * math.cos(tc) * bz < 0:
        tc += math.pi
    return (v ** (4.0/3.0)) * (abs(math.sin(tc / 2.0)) ** (8.0/3.0)) * (bt ** (2.0/3.0))


def area_2d_cm2(mlat_1d, mlt_1d):
    R_E = 6.3712e8
    d_mlat = np.deg2rad(mlat_1d[1] - mlat_1d[0])
    d_mlt = (mlt_1d[1] - mlt_1d[0]) * (2.0 * np.pi / 24.0)
    mlat_rad = np.deg2rad(mlat_1d)
    area_1d = (R_E ** 2) * np.cos(mlat_rad) * d_mlat * d_mlt
    return np.tile(area_1d.reshape(-1, 1), (1, len(mlt_1d)))


# ===================================================================
# 2. OVATION grid 推理 + HP
# ===================================================================
def ovation_predict_total(estimators, ref_utc, ec_val, mlat_1d, mlt_1d):
    """OVATION 4 head 求和插值到 (mlat_1d × mlt_1d) 上."""
    from scipy.interpolate import RegularGridInterpolator
    MLT_AX_OV = np.linspace(0.0, 24.0, 96)
    MLT_g, MLAT_g = np.meshgrid(mlt_1d, mlat_1d)
    target_pts = np.stack([MLAT_g.flatten(), MLT_g.flatten() % 24.0], axis=1)

    total_2d_ov_native = None
    mlat_ax_native = None
    for est in estimators:
        mlat_2d, _, f = est.get_flux_for_time(ref_utc, ec_val)
        mlat_ax = mlat_2d[:, 0]
        if not np.all(np.diff(mlat_ax) > 0):
            order = np.argsort(mlat_ax)
            mlat_ax = mlat_ax[order]
            f = f[order, :]
        if total_2d_ov_native is None:
            total_2d_ov_native = f.copy()
            mlat_ax_native = mlat_ax
        else:
            total_2d_ov_native = total_2d_ov_native + f

    interp = RegularGridInterpolator(
        (mlat_ax_native, MLT_AX_OV), total_2d_ov_native,
        bounds_error=False, fill_value=0.0,
    )
    return interp(target_pts).reshape(len(mlat_1d), len(mlt_1d))


# ===================================================================
# 3. 主程序
# ===================================================================
def main():
    device = "npu:0"

    # ---- 加载 OMNI 短期 parquet (含 St. Patrick's 2015-03-17) ----
    if not os.path.exists(OMNI_DATA_PATH):
        raise FileNotFoundError(f"OMNI parquet 不存在: {OMNI_DATA_PATH}")
    print(f"📥 加载 OMNI 5-min 短期序列: {OMNI_DATA_PATH}")
    df_all = pd.read_parquet(OMNI_DATA_PATH)
    df_all['utc'] = pd.to_datetime(df_all['utc'])
    print(f"   总样本: {len(df_all):,}  时间: {df_all['utc'].min()} ~ {df_all['utc'].max()}")

    # ---- 取磁暴窗口 (OMNI 已 5-min 等间隔, 直接 mask 即可) ----
    mask = (df_all['utc'] >= START_TIME) & (df_all['utc'] <= END_TIME)
    sw_rows = df_all[mask].sort_values('utc').reset_index(drop=True)
    sw_rows['utc_min'] = sw_rows['utc']    # 兼容下游 row['utc_min'] 访问
    print(f"\n📅 磁暴窗口 {START_TIME} ~ {END_TIME}")
    print(f"   5-min 步数: {len(sw_rows):,}")

    if len(sw_rows) == 0:
        print(f"❌ OMNI parquet 中该窗口无数据 "
              f"(参考: utc_max={df_all['utc'].max()}). 请检查时间或换事件.")
        return

    # ---- 加载 V4 MLP ----
    print(f"\n⚙️  加载 V4 MLP: {MODEL_PATH}")
    model = load_v4_model(model_path=MODEL_PATH, device=device)

    # ---- OVATION estimators ----
    print("⚙️  初始化 OVATION estimators...")
    estimators = (
        ao.FluxEstimator('diff', 'electron energy flux'),
        ao.FluxEstimator('mono', 'electron energy flux'),
        ao.FluxEstimator('wave', 'electron energy flux'),
        ao.FluxEstimator('ions', 'ion energy flux'),
    )

    area_2d = area_2d_cm2(MLAT_1D, MLT_1D)

    # ---- 逐分钟预测 ----
    times, bz_l, by_l, v_l, n_l, pdyn_l, newell_l = [], [], [], [], [], [], []
    hp_mlp_l, hp_ov_l = [], []

    print(f"\n🚀 逐分钟预测 (V4 MLP + OVATION) — 共 {len(sw_rows)} 步...")
    for idx, row in sw_rows.iterrows():
        utc = row['utc_min'].to_pydatetime() if hasattr(row['utc_min'], 'to_pydatetime') else row['utc_min']
        bx = row['Bx']; by = row['By']; bz = row['Bz']
        vx, vy, vz = row['Vx'], row['Vy'], row['Vz']
        v = float(np.sqrt(vx*vx + vy*vy + vz*vz))
        pdyn = row['P_dyn']
        n_val = float(row.get('N', np.nan))   # OMNI proton density 列名可能是 'N' 或 'P_density'

        times.append(utc)
        bz_l.append(bz); by_l.append(by); v_l.append(v)
        pdyn_l.append(pdyn); n_l.append(n_val)

        # --- A. V4 MLP ---
        flux_4heads = predict_grid_v4(
            model, sw_row=row, mlat_1d=MLAT_1D, mlt_1d=MLT_1D,
            scaler_path=SCALER_PATH, device=device,
        )    # (4, n_mlat, n_mlt)
        total_mlp = flux_4heads.sum(axis=0)
        hp_mlp = float(np.sum(total_mlp * area_2d) * 1e-16)
        hp_mlp_l.append(hp_mlp)

        # --- B. OVATION ---
        ec = calculate_newell_coupling(by, bz, v)
        newell_l.append(ec)
        total_ov = ovation_predict_total(estimators, utc, ec, MLAT_1D, MLT_1D)
        hp_ov = float(np.sum(total_ov * area_2d) * 1e-16)
        hp_ov_l.append(hp_ov)

        if (idx + 1) % 30 == 0:
            print(f"   [{utc.strftime('%H:%M')}] V4={hp_mlp:6.1f} GW | OV={hp_ov:6.1f} GW")

    # ---- 自动检测活跃区间 (HP > 阈值的连续时段) ----
    arr_hp = np.array(hp_mlp_l)
    hp_median = np.median(arr_hp)
    hp_std = np.std(arr_hp)
    hp_thresh = hp_median + 0.5 * hp_std   # 超过中位数 + 0.5σ 视为活跃
    active_mask = arr_hp > hp_thresh

    # 找连续 True 段的起止索引
    intervals = []
    in_span = False
    for j in range(len(active_mask)):
        if active_mask[j] and not in_span:
            start_j = j
            in_span = True
        elif not active_mask[j] and in_span:
            intervals.append((start_j, j - 1))
            in_span = False
    if in_span:
        intervals.append((start_j, len(active_mask) - 1))

    # 合并间距 < 3 步 (15 min) 的相邻区间, 避免碎片
    merged = []
    for s, e in intervals:
        if merged and s - merged[-1][1] <= 3:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))

    # 过滤太短区间 (< 4 步 = 20 min)
    merged = [(s, e) for s, e in merged if e - s >= 3]

    # 所有活跃区间统一使用同一颜色 (柔和金黄)
    SPAN_COLOR = '#FFD27F'
    span_labels = []
    for k, (s, e) in enumerate(merged):
        peak_hp = arr_hp[s:e+1].max()
        span_labels.append(f'Peak {peak_hp:.0f} GW')

    print(f"   检测到 {len(merged)} 个活跃区间 (阈值 {hp_thresh:.1f} GW)")

    # ---- 绘图 (4 panel) ----
    print("\n🎨 绘图...")
    fig, axs = plt.subplots(4, 1, figsize=(11, 14), sharex=True,
                            gridspec_kw={'height_ratios': [1, 1, 1, 2]})
    plt.subplots_adjust(hspace=0.10)
    fig.patch.set_facecolor('white')

    # 在所有面板上画垂直阴影带 (统一颜色)
    for ax in axs:
        for s, e in merged:
            ax.axvspan(times[s], times[e],
                       color=SPAN_COLOR, alpha=0.25, zorder=0)

    # Panel 1: IMF
    axs[0].plot(times, bz_l, color='blue',   linewidth=1.5, label='$B_z$')
    axs[0].plot(times, by_l, color='orange', linewidth=1.5, alpha=0.8, label='$B_y$')
    axs[0].axhline(0, color='black', linestyle='--', linewidth=1)
    axs[0].set_ylabel('IMF (nT)', fontsize=12, weight='bold')
    axs[0].legend(loc='upper right', ncol=2, fontsize=10)
    axs[0].grid(True, linestyle=':', alpha=0.6)

    # 在顶部面板标注区间标签 (统一颜色)
    for k, (s, e) in enumerate(merged):
        mid_t = times[s] + (times[e] - times[s]) / 2
        axs[0].text(mid_t, axs[0].get_ylim()[1] * 0.92,
                    span_labels[k], fontsize=9, weight='bold',
                    ha='center', va='top',
                    bbox=dict(boxstyle='round,pad=0.3',
                              facecolor=SPAN_COLOR,
                              alpha=0.7, edgecolor='gray'))

    # Panel 2: P_dyn (左) + V (右) 双 Y
    axs[1].plot(times, pdyn_l, color='purple', linewidth=1.5, label='$P_{dyn}$')
    axs[1].set_ylabel('$P_{dyn}$ (nPa)', fontsize=12, weight='bold',
                     color='purple')
    axs[1].tick_params(axis='y', labelcolor='purple')
    axs[1].grid(True, linestyle=':', alpha=0.6)
    ax1_v = axs[1].twinx()
    ax1_v.plot(times, v_l, color='green', linewidth=1.5, label='V')
    ax1_v.set_ylabel('V (km/s)', fontsize=12, weight='bold', color='green')
    ax1_v.tick_params(axis='y', labelcolor='green')

    # Panel 3: Newell coupling dΦ_MP/dt (太阳风—磁层耦合驱动量)
    axs[2].plot(times, newell_l, color='darkorange', linewidth=1.5)
    axs[2].set_ylabel(r'$d\Phi_{MP}/dt$ (Wb/s)',
                     fontsize=12, weight='bold')
    axs[2].grid(True, linestyle=':', alpha=0.6)

    # Panel 4: HP
    axs[3].plot(times, hp_ov_l, color='black', linestyle='--',
                linewidth=2, label='OVATION-Prime')
    axs[3].plot(times, hp_mlp_l, color='red', linewidth=2.5,
                label='AMT (Ours)')
    axs[3].fill_between(times, hp_mlp_l, alpha=0.15, color='red')
    axs[3].set_ylabel('Hemispheric Power (GW)', fontsize=12, weight='bold')
    axs[3].set_xlabel('Time (UT)', fontsize=12, weight='bold')
    axs[3].legend(loc='upper left', fontsize=11, framealpha=0.9)
    axs[3].grid(True, linestyle=':', alpha=0.6)

    axs[3].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    # 强制 x 轴范围覆盖整个事件窗口, 即使末端数据缺失 (如 OMNI 重建留下的尾部 NaN)
    axs[3].set_xlim(START_TIME, END_TIME)
    plt.xticks(rotation=0)

    fig.suptitle(
        f"Auroral HP Response — {START_TIME.strftime('%Y-%m-%d')} Storm Event",
        fontsize=18, weight='bold', y=0.92,
    )

    save_path = os.path.join(OUTPUT_DIR, "v4_hp_evolution_storm.png")
    plt.savefig(save_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"\n✅ HP 演化图已保存: {save_path}")

    # ---- 控制台简表 ----
    print("\n" + "=" * 72)
    print(f"📊 HP 统计 ({START_TIME} ~ {END_TIME})")
    print("=" * 72)
    print(f"{'Quantity':>20s} | {'V4 MLP':>10s} | {'OVATION':>10s} | {'Δ':>10s}")
    print("-" * 72)
    arr_m, arr_o = np.array(hp_mlp_l), np.array(hp_ov_l)
    print(f"{'mean (GW)':>20s} | {arr_m.mean():>10.2f} | {arr_o.mean():>10.2f} | {arr_m.mean()-arr_o.mean():>+10.2f}")
    print(f"{'peak (GW)':>20s} | {arr_m.max():>10.2f} | {arr_o.max():>10.2f} | {arr_m.max()-arr_o.max():>+10.2f}")
    if len(arr_m) > 2:
        from scipy.stats import pearsonr
        r, _ = pearsonr(arr_m, arr_o)
        print(f"{'corr V4 vs OV':>20s} | {r:>+10.3f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
