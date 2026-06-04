"""
infer_compare_ovation_boundary.py  (V4 改造版)
================================================================
论文图 10: 极光卵边界拟合对比 — V4 MLP vs OVATION vs IMAGE GT.

流程:
  1. 从 IMAGE-WIC 反演的 EALB/PALB txt 文件中挑出"最佳事件"
     (最多 MLT 桶, 平均赤道边界最低 → 最强磁暴典例)
  2. 在 OMNI 2000-2005 短期 parquet 中找到对应 UTC 的 SW 行 (含 168 维 lag)
     注: V4 测试集仅覆盖 2014-2015, 而 IMAGE-WIC 仅在 2000-2002 有数据,
         所以这里 *不读测试集*, 直接读单独构建的 OMNI parquet.
         需先运行: python data/build_omni_2000_2005.py
  3. V4 MLP 与 OVATION 在统一极坐标网格上预测总通量
  4. 用阈值法 (0.5 erg/cm²/s) 提取赤道/极向边界
  5. 单张 1×3 合成图输出:
       Col 1: OVATION 总通量热力图 + IMAGE GT 边界
       Col 2: V4 MLP 总通量热力图 + IMAGE GT 边界
       Col 3: 三方边界对比 (V4 MLP / OVATION / IMAGE GT)

输入:
  /home/docker/code/Aurora_MLP/EALB_wic_v2.txt  (IMAGE GT)
  /home/docker/code/Aurora_MLP/PALB_wic_v2.txt
  /home/docker/data/private/AuroraData/omni_shortterm_2000_2005.parquet

输出:
  res/evaluation_boundary_v4/fig_polar_flux_and_boundaries.png  (1×3 合成图)

用法:
  # 一次性构建 OMNI 2000-2005:
  python data/build_omni_2000_2005.py
  # 然后跑边界对比:
  python infer_compare_ovation_boundary.py
"""
import sys
import os
import math
import torch
import torch_npu  # noqa: F401
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = "/home/docker/code/Aurora_MLP_final"
sys.path.append(BASE_DIR)

from infer_v4_utils import load_v4_model, predict_grid_v4
from auroramaps import ovation as ao


# ===================================================================
# 配置
# ===================================================================
OMNI_DATA_PATH = "/home/docker/data/private/AuroraData/omni_shortterm_2000_2005.parquet"
CKPT_DIR = f"{BASE_DIR}/ckpt/ckpt_v4_simple_different_10"
MODEL_PATH = f"{CKPT_DIR}/aurora_v4_best.pth"
SCALER_PATH = f"{CKPT_DIR}/scaler/sw_scaler_v4.pkl"
OUTPUT_DIR = f"{BASE_DIR}/res/v4_different_3/ckpt_v4_simple_different_10"
os.makedirs(OUTPUT_DIR, exist_ok=True)

EALB_TXT = "/home/docker/code/Aurora_MLP/EALB_wic_v2.txt"
PALB_TXT = "/home/docker/code/Aurora_MLP/PALB_wic_v2.txt"

# 网格 (与 OVATION 原生分辨率匹配以便公平对比)
MLAT_1D = np.linspace(50, 90, 80)
MLT_1D  = np.linspace(0, 24, 144)
BOUNDARY_THRESHOLD = 0.5     # erg/cm²/s
TIME_TOLERANCE = pd.Timedelta(minutes=10)   # IMAGE 事件与 V4 SW 行的最大 UTC 偏差


# ===================================================================
# 1. IMAGE GT 加载
# ===================================================================
def load_and_find_best_event(ealb_path, palb_path):
    """读取 IMAGE-WIC EALB/PALB 文件, 返回最佳事件的 UTC + 24 MLT 桶赤道/极向纬度."""
    cols = ['Year', 'SOY'] + [f'MLT_{i}' for i in range(24)]
    df_e = pd.read_csv(ealb_path, sep=r'\s+', comment='#', names=cols)
    df_p = pd.read_csv(palb_path, sep=r'\s+', comment='#', names=cols)

    df_e['utc'] = pd.to_datetime(df_e['Year'].astype(str), format='%Y') \
                  + pd.to_timedelta(df_e['SOY'], unit='s')
    df_p['utc'] = pd.to_datetime(df_p['Year'].astype(str), format='%Y') \
                  + pd.to_timedelta(df_p['SOY'], unit='s')

    df_e[cols[2:]] = df_e[cols[2:]].replace(0.0, np.nan)
    df_p[cols[2:]] = df_p[cols[2:]].replace(0.0, np.nan)

    df_e['valid_counts'] = df_e[cols[2:]].count(axis=1)
    df_e['mean_eq_lat']  = df_e[cols[2:]].mean(axis=1)

    candidates = df_e[
        (df_e['valid_counts'] >= 18) & (df_e['mean_eq_lat'] < 62.0)
    ].sort_values('mean_eq_lat')
    if candidates.empty:
        raise ValueError("未找到符合条件的 IMAGE 事件 (valid≥18 且 mean_eq_lat<62°)")

    best = candidates.iloc[0]
    target_utc = best['utc']
    gt_e = best[cols[2:]].values.astype(float)
    gt_p = df_p[df_p['utc'] == target_utc][cols[2:]].values.astype(float)[0]
    return target_utc, gt_e, gt_p


# ===================================================================
# 2. OMNI SW 行匹配 (从 omni_shortterm_2000_2005.parquet)
# ===================================================================
def find_matching_sw_row(df_omni, target_utc, tolerance=TIME_TOLERANCE):
    """
    在 OMNI 短期 parquet 中找最近 UTC 的 SW 行.
    df_omni 应已 sort_values('utc'), 含 7 base + 168 lag = 176 列.
    """
    # 用 searchsorted 加速 (df_omni 已按 utc 排序)
    utc_arr = df_omni['utc'].values
    idx = np.searchsorted(utc_arr, np.datetime64(target_utc))
    candidates = []
    if idx > 0:
        candidates.append(idx - 1)
    if idx < len(utc_arr):
        candidates.append(idx)
    # 选距离最小的
    best_i = min(candidates,
                 key=lambda i: abs(utc_arr[i] - np.datetime64(target_utc)))
    closest = df_omni.iloc[best_i].copy()
    dt = pd.Timedelta(abs(utc_arr[best_i] - np.datetime64(target_utc)))
    if dt > tolerance:
        raise ValueError(
            f"OMNI 2000-2005 parquet 中无 UTC ±{tolerance} 内的数据 "
            f"(最近相差 {dt}). 请检查数据范围或换 IMAGE 事件."
        )
    closest['_dt'] = dt
    return closest


# ===================================================================
# 3. OVATION 总通量 (与 HP.py 一致)
# ===================================================================
def ovation_predict_total(estimators, ref_utc, ec_val, mlat_1d, mlt_1d):
    from scipy.interpolate import RegularGridInterpolator
    MLT_AX_OV = np.linspace(0.0, 24.0, 96)
    MLT_g, MLAT_g = np.meshgrid(mlt_1d, mlat_1d)
    target_pts = np.stack([MLAT_g.flatten(), MLT_g.flatten() % 24.0], axis=1)

    total_native = None
    mlat_ax_native = None
    for est in estimators:
        mlat_2d, _, f = est.get_flux_for_time(ref_utc, ec_val)
        mlat_ax = mlat_2d[:, 0]
        if not np.all(np.diff(mlat_ax) > 0):
            order = np.argsort(mlat_ax)
            mlat_ax = mlat_ax[order]
            f = f[order, :]
        if total_native is None:
            total_native = f.copy()
            mlat_ax_native = mlat_ax
        else:
            total_native = total_native + f

    interp = RegularGridInterpolator(
        (mlat_ax_native, MLT_AX_OV), total_native,
        bounds_error=False, fill_value=0.0,
    )
    return interp(target_pts).reshape(len(mlat_1d), len(mlt_1d))


def calculate_newell_coupling(by, bz, v):
    bt = math.sqrt(by**2 + bz**2)
    tc = math.atan2(by, 0.001 if bz == 0 else bz)
    if bt * math.cos(tc) * bz < 0:
        tc += math.pi
    return (v ** (4.0/3.0)) * (abs(math.sin(tc / 2.0)) ** (8.0/3.0)) * (bt ** (2.0/3.0))


# ===================================================================
# 4. 边界提取
# ===================================================================
def extract_boundaries(flux_grid, mlat_array, threshold=BOUNDARY_THRESHOLD):
    """
    每个 MLT 列: 找 flux≥threshold 的 mlat 范围.
    返回 eq_bnd (赤道边界, 最低 mlat), pol_bnd (极向边界, 最高 mlat).
    """
    n_mlt = flux_grid.shape[1]
    eq_b, pol_b = np.full(n_mlt, np.nan), np.full(n_mlt, np.nan)
    for i in range(n_mlt):
        active = np.where(flux_grid[:, i] >= threshold)[0]
        if len(active) > 0:
            eq_b[i]  = mlat_array[active[0]]
            pol_b[i] = mlat_array[active[-1]]
    if not np.isnan(eq_b).any():
        eq_b  = gaussian_filter1d(eq_b,  sigma=2, mode='wrap')
        pol_b = gaussian_filter1d(pol_b, sigma=2, mode='wrap')
    return eq_b, pol_b


# ===================================================================
# 5. 极坐标投影工具
# ===================================================================
def mlt_mlat_to_polar(mlt, mlat):
    """MLT/MLAT → 极坐标 (theta, r). MLT=0 在最下方."""
    theta = (mlt / 24.0) * 2.0 * np.pi
    r = 90.0 - mlat
    return theta, r


def process_gt_for_plotting(mlt_gt, lat_gt):
    """过滤 NaN, 排序, 强制首尾闭合."""
    valid = ~np.isnan(lat_gt)
    mlt_v = mlt_gt[valid]
    lat_v = lat_gt[valid]
    sort = np.argsort(mlt_v)
    mlt_v, lat_v = mlt_v[sort], lat_v[sort]
    if len(mlt_v) > 0:
        mlt_v = np.append(mlt_v, mlt_v[0])
        lat_v = np.append(lat_v, lat_v[0])
    return mlt_mlat_to_polar(mlt_v, lat_v)


# ===================================================================
# 6. 绘图: Fig 1 极坐标边界对比
# ===================================================================
def plot_polar_boundaries(mlt_grid, eq_ov, pol_ov, eq_mlp, pol_mlp,
                          mlt_gt, gt_e, gt_p, target_utc, bz, out_dir):
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.set_theta_zero_location("S")
    ax.set_theta_direction(1)

    theta_ov,  r_eq_ov  = mlt_mlat_to_polar(mlt_grid, eq_ov)
    _,         r_pol_ov = mlt_mlat_to_polar(mlt_grid, pol_ov)
    theta_mlp, r_eq_mlp = mlt_mlat_to_polar(mlt_grid, eq_mlp)
    _,         r_pol_mlp= mlt_mlat_to_polar(mlt_grid, pol_mlp)
    theta_gt_e, r_gt_e = process_gt_for_plotting(mlt_gt, gt_e)
    theta_gt_p, r_gt_p = process_gt_for_plotting(mlt_gt, gt_p)

    ax.plot(theta_ov, r_eq_ov,  color='gray', linewidth=2.5, label='OVATION')
    ax.plot(theta_ov, r_pol_ov, color='gray', linewidth=2.5)
    ax.fill_between(theta_ov, r_eq_ov, r_pol_ov, color='gray', alpha=0.15)

    ax.plot(theta_mlp, r_eq_mlp,  color='red', linewidth=3, label='V4 MLP (Ours)')
    ax.plot(theta_mlp, r_pol_mlp, color='red', linewidth=3)
    ax.fill_between(theta_mlp, r_eq_mlp, r_pol_mlp, color='red', alpha=0.20)

    ax.plot(theta_gt_e, r_gt_e, color='blue', marker='o', markersize=6,
            linewidth=1.5, markeredgecolor='white', label='IMAGE GT')
    ax.plot(theta_gt_p, r_gt_p, color='blue', marker='o', markersize=6,
            linewidth=1.5, markeredgecolor='white')

    ax.set_ylim(0, 40)
    lat_circles = [50, 60, 70, 80]
    ax.set_rticks([90 - lat for lat in lat_circles])
    ax.set_yticklabels([f'{lat}°' for lat in lat_circles],
                       fontsize=10, color='gray')
    hour_ticks = np.arange(0, 24, 3)
    ax.set_xticks(np.deg2rad((hour_ticks / 24.0) * 360))
    ax.set_xticklabels([f'{h:02d}' for h in hour_ticks], fontsize=12)

    plt.title(
        f"Polar Projection of Auroral Boundaries\n"
        f"Event: {target_utc.strftime('%Y-%m-%d %H:%M UT')} ($B_z$={bz:+.1f} nT)",
        fontsize=16, weight='bold', pad=20,
    )
    ax.legend(loc='lower right', bbox_to_anchor=(1.2, 0), fontsize=12)

    save_path = os.path.join(out_dir, "fig1_polar_boundary_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Fig 1 极坐标边界对比图: {save_path}")


# ===================================================================
# 7. 绘图: Fig 2 双面热力图
# ===================================================================
def plot_polar_flux_maps(mlat_2d, mlt_2d, flux_ov, flux_mlp,
                         mlt_gt, gt_e, gt_p, target_utc, out_dir):
    fig, axs = plt.subplots(1, 2, figsize=(14, 7), subplot_kw={'projection': 'polar'})
    fig.patch.set_facecolor('white')

    Theta_g, R_g = mlt_mlat_to_polar(mlt_2d, mlat_2d)
    theta_gt_e, r_gt_e = process_gt_for_plotting(mlt_gt, gt_e)
    theta_gt_p, r_gt_p = process_gt_for_plotting(mlt_gt, gt_p)

    flux_ov_safe  = np.nan_to_num(flux_ov,  nan=0.0)
    flux_mlp_safe = np.nan_to_num(flux_mlp, nan=0.0)

    colors = [(0, 0, 0), (0, 0, 0.5), (0, 0, 0.8), (0, 0.5, 1), (0, 1, 1),
              (0.5, 1, 0.5), (1, 1, 0), (1, 0.5, 0), (1, 0, 0), (0.8, 0.8, 0.8)]
    cmap = LinearSegmentedColormap.from_list('aurora_cmap', colors, N=256)

    titles = ["OVATION-Prime 2013", "V4 MLP (Ours)"]
    fluxes = [flux_ov_safe, flux_mlp_safe]

    c = None
    for i, ax in enumerate(axs):
        ax.set_theta_zero_location('S')
        ax.set_theta_direction(1)
        c = ax.pcolormesh(Theta_g, R_g, fluxes[i], cmap=cmap,
                          shading='nearest', vmin=0, vmax=5.0)
        ax.plot(theta_gt_e, r_gt_e, color='cyan', marker='o', markersize=4,
                linewidth=1.5, label='IMAGE GT' if i == 1 else "")
        ax.plot(theta_gt_p, r_gt_p, color='cyan', marker='o', markersize=4,
                linewidth=1.5)

        ax.set_ylim(0, 40)
        lat_circles = [50, 60, 70, 80]
        ax.set_rticks([90 - lat for lat in lat_circles])
        ax.set_yticklabels([f'{lat}°' for lat in lat_circles],
                           fontsize=10, color='white')
        hour_ticks = np.arange(0, 24, 3)
        ax.set_xticks(np.deg2rad((hour_ticks / 24.0) * 360))
        ax.set_xticklabels([f'{h:02d}' for h in hour_ticks], fontsize=12)
        ax.set_title(titles[i], fontsize=15, weight='bold', pad=20)

        if i == 1:
            ax.legend(loc='lower right', bbox_to_anchor=(1.2, 0),
                      facecolor='white', framealpha=0.8)

    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.03])
    cbar = fig.colorbar(c, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Auroral Energy Flux (ergs cm$^{-2}$ s$^{-1}$)',
                   fontsize=12, weight='bold')

    plt.suptitle(
        f"Auroral Energy Flux Polar Comparison\n"
        f"Event: {target_utc.strftime('%Y-%m-%d %H:%M UT')}",
        fontsize=18, weight='bold', y=1.05,
    )

    save_path = os.path.join(out_dir, "fig2_polar_flux_maps.png")
    plt.savefig(save_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"✅ Fig 2 极地通量热力图: {save_path}")


# ===================================================================
# 7'. 绘图: 1×3 合成图 (OVATION 通量 / V4 MLP 通量 / 三方边界对比)
# ===================================================================
def plot_combined_1x3(mlat_2d, mlt_2d,
                      flux_ov, flux_mlp,
                      mlt_grid, eq_ov, pol_ov, eq_mlp, pol_mlp,
                      mlt_gt, gt_e, gt_p,
                      target_utc, bz, out_dir):
    """单张 1×3 极坐标合成图.
    Col 1: OVATION-Prime 总通量热力图 + IMAGE GT 边界
    Col 2: V4 MLP 总通量热力图 + IMAGE GT 边界
    Col 3: 边界对比 (V4 MLP / OVATION / IMAGE GT)
    """
    fig, axs = plt.subplots(1, 3, figsize=(21, 7.5),
                            subplot_kw={'projection': 'polar'})
    fig.patch.set_facecolor('white')

    Theta_g, R_g = mlt_mlat_to_polar(mlt_2d, mlat_2d)
    theta_gt_e, r_gt_e = process_gt_for_plotting(mlt_gt, gt_e)
    theta_gt_p, r_gt_p = process_gt_for_plotting(mlt_gt, gt_p)

    flux_ov_safe  = np.nan_to_num(flux_ov,  nan=0.0)
    flux_mlp_safe = np.nan_to_num(flux_mlp, nan=0.0)

    colors = [(0, 0, 0), (0, 0, 0.5), (0, 0, 0.8), (0, 0.5, 1), (0, 1, 1),
              (0.5, 1, 0.5), (1, 1, 0), (1, 0.5, 0), (1, 0, 0), (0.8, 0.8, 0.8)]
    cmap = LinearSegmentedColormap.from_list('aurora_cmap', colors, N=256)

    # ---- Col 1 & 2: 通量热力图 ----
    titles_flux = ["OVATION-Prime 2013", "AMT (Ours)"]
    fluxes = [flux_ov_safe, flux_mlp_safe]
    c_flux = None
    for i in range(2):
        ax = axs[i]
        ax.set_theta_zero_location('S')
        ax.set_theta_direction(1)
        c_flux = ax.pcolormesh(Theta_g, R_g, fluxes[i], cmap=cmap,
                               shading='nearest', vmin=0, vmax=5.0)
        ax.plot(theta_gt_e, r_gt_e, color='cyan', marker='o', markersize=4,
                linewidth=1.5, label='IMAGE GT' if i == 1 else "")
        ax.plot(theta_gt_p, r_gt_p, color='cyan', marker='o', markersize=4,
                linewidth=1.5)

        ax.set_ylim(0, 40)
        lat_circles = [50, 60, 70, 80]
        ax.set_rticks([90 - lat for lat in lat_circles])
        ax.set_yticklabels([f'{lat}°' for lat in lat_circles],
                           fontsize=10, color='white')
        hour_ticks = np.arange(0, 24, 3)
        ax.set_xticks(np.deg2rad((hour_ticks / 24.0) * 360))
        ax.set_xticklabels([f'{h:02d}' for h in hour_ticks], fontsize=12)
        ax.set_title(titles_flux[i], fontsize=15, weight='bold', pad=20)
        if i == 1:
            ax.legend(loc='lower right', bbox_to_anchor=(1.18, -0.02),
                      facecolor='white', framealpha=0.85, fontsize=10)

    # ---- Col 3: 边界对比 ----
    ax = axs[2]
    ax.set_facecolor('white')
    ax.set_theta_zero_location('S')
    ax.set_theta_direction(1)

    theta_ov,  r_eq_ov  = mlt_mlat_to_polar(mlt_grid, eq_ov)
    _,         r_pol_ov = mlt_mlat_to_polar(mlt_grid, pol_ov)
    theta_mlp, r_eq_mlp = mlt_mlat_to_polar(mlt_grid, eq_mlp)
    _,         r_pol_mlp= mlt_mlat_to_polar(mlt_grid, pol_mlp)

    ax.plot(theta_ov, r_eq_ov,  color='gray', linewidth=2.5, label='OVATION')
    ax.plot(theta_ov, r_pol_ov, color='gray', linewidth=2.5)
    ax.fill_between(theta_ov, r_eq_ov, r_pol_ov, color='gray', alpha=0.15)

    ax.plot(theta_mlp, r_eq_mlp,  color='red', linewidth=3, label='V4 MLP (Ours)')
    ax.plot(theta_mlp, r_pol_mlp, color='red', linewidth=3)
    ax.fill_between(theta_mlp, r_eq_mlp, r_pol_mlp, color='red', alpha=0.20)

    ax.plot(theta_gt_e, r_gt_e, color='blue', marker='o', markersize=6,
            linewidth=1.5, markeredgecolor='white', label='IMAGE GT')
    ax.plot(theta_gt_p, r_gt_p, color='blue', marker='o', markersize=6,
            linewidth=1.5, markeredgecolor='white')

    ax.set_ylim(0, 40)
    lat_circles = [50, 60, 70, 80]
    ax.set_rticks([90 - lat for lat in lat_circles])
    ax.set_yticklabels([f'{lat}°' for lat in lat_circles],
                       fontsize=10, color='gray')
    hour_ticks = np.arange(0, 24, 3)
    ax.set_xticks(np.deg2rad((hour_ticks / 24.0) * 360))
    ax.set_xticklabels([f'{h:02d}' for h in hour_ticks], fontsize=12)
    ax.set_title("Boundary Comparison", fontsize=15, weight='bold', pad=20)
    ax.legend(loc='lower right', bbox_to_anchor=(1.22, -0.02), fontsize=10)

    # ---- 共享水平 colorbar (仅覆盖前 2 列通量图) ----
    cbar_ax = fig.add_axes([0.13, 0.06, 0.50, 0.022])
    cbar = fig.colorbar(c_flux, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Auroral Energy Flux (ergs cm$^{-2}$ s$^{-1}$)',
                   fontsize=12, weight='bold')

    plt.suptitle(
        f"Auroral Energy Flux & Boundary Polar Comparison\n"
        f"Event: {target_utc.strftime('%Y-%m-%d %H:%M UT')} "
        f"($B_z$={bz:+.1f} nT)",
        fontsize=18, weight='bold', y=1.02,
    )

    save_path = os.path.join(out_dir, "fig_polar_flux_and_boundaries.png")
    plt.savefig(save_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"✅ 1×3 极坐标合成图: {save_path}")


# ===================================================================
# 8. 主程序
# ===================================================================
def main():
    device = "npu:0"

    # ---- 1. 选最佳 IMAGE 事件 ----
    print("📥 加载 IMAGE-WIC EALB/PALB 边界 GT...")
    target_utc, gt_e, gt_p = load_and_find_best_event(EALB_TXT, PALB_TXT)
    target_utc_aligned = target_utc.round('min')
    print(f"   选中事件 UTC: {target_utc_aligned}")
    print(f"   赤道边界平均: {np.nanmean(gt_e):.2f}° MLAT")

    # ---- 2. 加载 OMNI 2000-2005 + 找匹配 SW 行 ----
    if not os.path.exists(OMNI_DATA_PATH):
        raise FileNotFoundError(
            f"OMNI 2000-2005 parquet 不存在: {OMNI_DATA_PATH}\n"
            f"请先运行: python {BASE_DIR}/data/build_omni_2000_2005.py"
        )
    print(f"\n📥 加载 OMNI 2000-2005: {OMNI_DATA_PATH}")
    df_omni = pd.read_parquet(OMNI_DATA_PATH)
    df_omni['utc'] = pd.to_datetime(df_omni['utc'])
    df_omni = df_omni.sort_values('utc').reset_index(drop=True)
    print(f"   OMNI 行数: {len(df_omni):,} | 列数: {len(df_omni.columns)} "
          f"| 时间: {df_omni['utc'].min()} ~ {df_omni['utc'].max()}")
    sw_row = find_matching_sw_row(df_omni, target_utc_aligned)
    print(f"   匹配到 SW 行 UTC: {sw_row['utc']}, Δt = {sw_row['_dt']}")

    bz   = float(sw_row['Bz'])
    by   = float(sw_row['By'])
    vx, vy, vz = float(sw_row['Vx']), float(sw_row['Vy']), float(sw_row['Vz'])
    v    = float(np.sqrt(vx*vx + vy*vy + vz*vz))
    pdyn = float(sw_row['P_dyn'])
    print(f"   Bz={bz:+.1f} nT | By={by:+.1f} nT | V={v:.0f} km/s | "
          f"Pdyn={pdyn:.1f} nPa")

    # ---- 3. V4 MLP 预测 ----
    print(f"\n⚙️  加载 V4 MLP: {MODEL_PATH}")
    model = load_v4_model(model_path=MODEL_PATH, device=device)

    print("🔮 V4 MLP 预测网格通量...")
    flux_4heads_mlp = predict_grid_v4(
        model, sw_row=sw_row, mlat_1d=MLAT_1D, mlt_1d=MLT_1D,
        scaler_path=SCALER_PATH, device=device,
    )
    total_mlp = flux_4heads_mlp.sum(axis=0)    # (n_mlat, n_mlt)
    print(f"   total_mlp range: [{total_mlp.min():.3f}, {total_mlp.max():.3f}]")

    # ---- 4. OVATION 预测 ----
    print("🔮 OVATION 预测...")
    ec = calculate_newell_coupling(by, bz, v)
    estimators = (
        ao.FluxEstimator('diff', 'electron energy flux'),
        ao.FluxEstimator('mono', 'electron energy flux'),
        ao.FluxEstimator('wave', 'electron energy flux'),
        ao.FluxEstimator('ions', 'ion energy flux'),
    )
    target_utc_py = (target_utc_aligned.to_pydatetime()
                     if hasattr(target_utc_aligned, 'to_pydatetime')
                     else target_utc_aligned)
    total_ov = ovation_predict_total(estimators, target_utc_py, ec, MLAT_1D, MLT_1D)
    print(f"   total_ov range: [{total_ov.min():.3f}, {total_ov.max():.3f}]")

    # ---- 5. 边界提取 ----
    eq_mlp, pol_mlp = extract_boundaries(total_mlp, MLAT_1D, BOUNDARY_THRESHOLD)
    eq_ov,  pol_ov  = extract_boundaries(total_ov,  MLAT_1D, BOUNDARY_THRESHOLD)
    mlt_gt = np.arange(0.5, 24.5, 1.0)

    # ---- 6. 绘图 (1×3 合成图: OVATION 通量 / V4 MLP 通量 / 三方边界对比) ----
    MLT_g, MLAT_g = np.meshgrid(MLT_1D, MLAT_1D)
    plot_combined_1x3(MLAT_g, MLT_g, total_ov, total_mlp,
                      MLT_1D, eq_ov, pol_ov, eq_mlp, pol_mlp,
                      mlt_gt, gt_e, gt_p, target_utc_aligned,
                      bz, OUTPUT_DIR)

    # ---- 7. 边界拟合数值 ----
    valid_e = ~np.isnan(gt_e)
    valid_p = ~np.isnan(gt_p)

    def boundary_at_mlt(mlt_target_arr, eq_grid, mlt_grid):
        return np.array([
            eq_grid[np.argmin(np.abs(mlt_grid - mt % 24.0))] for mt in mlt_target_arr
        ])

    eq_mlp_at_gt = boundary_at_mlt(mlt_gt[valid_e], eq_mlp, MLT_1D)
    eq_ov_at_gt  = boundary_at_mlt(mlt_gt[valid_e], eq_ov,  MLT_1D)
    pol_mlp_at_gt = boundary_at_mlt(mlt_gt[valid_p], pol_mlp, MLT_1D)
    pol_ov_at_gt  = boundary_at_mlt(mlt_gt[valid_p], pol_ov,  MLT_1D)

    print("\n" + "=" * 80)
    print(f"📊 边界拟合误差 vs IMAGE GT  (事件 {target_utc_aligned})")
    print("=" * 80)
    print(f"{'Boundary':>14s} | {'V4 MAE':>8s} | {'V4 RMSE':>9s} | "
          f"{'OV MAE':>8s} | {'OV RMSE':>9s}")
    print("-" * 80)

    for label, gt, mlp_b, ov_b in [
        ('Equatorward', gt_e[valid_e], eq_mlp_at_gt,  eq_ov_at_gt),
        ('Poleward',    gt_p[valid_p], pol_mlp_at_gt, pol_ov_at_gt),
    ]:
        mae_m = np.mean(np.abs(mlp_b - gt))
        rms_m = np.sqrt(np.mean((mlp_b - gt) ** 2))
        mae_o = np.mean(np.abs(ov_b - gt))
        rms_o = np.sqrt(np.mean((ov_b - gt) ** 2))
        print(f"{label:>14s} | {mae_m:>8.2f} | {rms_m:>9.2f} | "
              f"{mae_o:>8.2f} | {rms_o:>9.2f}  (deg)")
    print("=" * 80)


if __name__ == "__main__":
    main()
