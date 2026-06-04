"""
infer_season_compare_N.py  (V4 改造版 — 纯时间效应, 季节平均)
================================================================
论文图 7: 季节效应 — 固定太阳风条件, 仅切换日期 (DOY), 隔离纯时间项.

设计原则:
    * 唯一变量 = utc (DOY 不同, 12:00 UT 固定)
    * 完全固定 = Bx/By/Bz, Vx/Vy/Vz, P_dyn 当前 + 所有 24 步 lag (稳态)
    * 这样得到的 Δ flux = 纯 dipole_tilt + cos_sza 引起的季节响应,
      不混入太阳风变化带来的污染.

MODE 开关 (顶部 MODE 变量):
    'season_avg' (默认): 整夏季 vs 整冬季 平均 (与 audit_season_energy_bias 一致)
        Summer = DOY 135-225 (5/15-8/13, 91 天)
        Winter = DOY 315-365 ∪ 1-45 (11/11-2/14, 96 天)
        对每个 DOY (12:00 UT) 推理一次 grid, 最后取 mean → 减少单点偶然性
    'point':         单时间点 (Summer Solstice DOY 172 vs Winter Solstice DOY 355)
        快速但只反映夏至/冬至两点

布局 3 × 4:
    Row 1: Northern Summer (季节平均 或 DOY 172) × {D,M,B,I}
    Row 2: Northern Winter (季节平均 或 DOY 355) × {D,M,B,I}
    Row 3: Δ = Summer - Winter (RdBu_r divergent cmap)

输入: 无需测试 parquet (合成 sw_row).
输出: res/evaluation_season_N_v4_pure_time_different_4/season_pure_time_3x4.png

用法:
    python infer_season_compare_N.py
    # 改 MODE = 'point' 即可回到单时间点模式
    # 改 FIXED_SW 即可看不同太阳风状态下的纯时间响应 (quiet / moderate / strong)
"""
import sys
import os
from datetime import datetime, timedelta
import torch
import torch_npu  # noqa: F401
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

BASE_DIR = "/home/docker/code/Aurora_MLP_final"
sys.path.append(BASE_DIR)

from infer_v4_utils import load_v4_model, predict_grid_v4


# ===================================================================
# 配置
# ===================================================================
CKPT_DIR = f"{BASE_DIR}/ckpt/ckpt_v4_simple_different_10"
MODEL_PATH = f"{CKPT_DIR}/aurora_v4_best.pth"
SCALER_PATH = f"{CKPT_DIR}/scaler/sw_scaler_v4.pkl"
OUTPUT_DIR = f"{BASE_DIR}/res/evaluation_season_N_v4_pure_time_different_10"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- 模式: 'season_avg' (整季节平均) 或 'point' (单时间点, 原行为) ----
MODE = 'season_avg'

# ---- 固定的太阳风状态 (单变量实验, 用户可改) ----
# 默认: 中等活跃 — Bz=-5 nT, V=500 km/s, Pdyn=3 nPa
# 想看其他: 把整个 dict 改了即可 (注意 24 步 lag 也按相同值填稳态)
FIXED_SW = {
    'Bx':    0.0,
    'By':    0.0,
    'Bz':   -5.0,        # 中等南向 IMF
    'Vx':  -500.0,       # 500 km/s 反日向
    'Vy':    0.0,
    'Vz':    0.0,
    'P_dyn': 3.0,
}
SW_LABEL = "Moderate ($B_z=-5$ nT, $V=500$ km/s, $P_{dyn}=3$ nPa)"

# 夏至 / 冬至 (12:00 UT, 同一时刻保证 dipole tilt 周日变化相同) - 'point' 模式用
SUMMER_UTC = datetime(2014, 6, 21, 12, 0)
WINTER_UTC = datetime(2014, 12, 21, 12, 0)

# 季节定义 (与 data_draw.py: audit_season_energy_bias 一致) - 'season_avg' 模式用
# Summer: DOY 135-225 (5/15-8/13, 91 天)
# Winter: DOY 315-365 ∪ 1-45 (11/11 跨年到 2/14, 96 天)
SEASON_AVG_YEAR = 2014
SEASON_AVG_HOUR = 12       # 每天用 12:00 UT 采样
SUMMER_DOYS = list(range(135, 226))
WINTER_DOYS = list(range(315, 366)) + list(range(1, 46))

# 网格 (北半球极区)
MLAT_1D = np.linspace(50, 90, 80)
MLT_1D  = np.linspace(0, 24, 144)

# OMNI 24 步 lag (5..120 min)
LAG_MINUTES = list(range(5, 125, 5))


# ===================================================================
# 1. 工具
# ===================================================================
def calculate_hp_gw(flux_grid, mlat_1d, mlt_1d):
    """积分得 HP (GW)."""
    R_E = 6.3712e8
    d_mlat = np.deg2rad(mlat_1d[1] - mlat_1d[0])
    d_mlt = (mlt_1d[1] - mlt_1d[0]) * (2.0 * np.pi / 24.0)
    mlat_rad = np.deg2rad(mlat_1d)
    area_1d = (R_E ** 2) * np.cos(mlat_rad) * d_mlat * d_mlt
    area_2d = np.tile(area_1d.reshape(-1, 1), (1, len(mlt_1d)))
    return float(np.sum(flux_grid * area_2d) * 1e-16)


def build_steady_sw_row(utc, fixed_sw):
    """合成一个稳态 SW 行: 当前值 + 所有 24 步 lag 等于当前值."""
    row = {'utc': utc}
    row.update(fixed_sw)
    # 24 步 lag, 与训练 pipeline 命名一致 (Bx_lag_5 ... P_dyn_lag_120)
    for lag in LAG_MINUTES:
        for var, val in fixed_sw.items():
            row[f'{var}_lag_{lag}'] = val
    # 占位 (predict_grid_v4 内会被网格覆盖)
    row['mlat'] = 70.0
    row['mlt'] = 12.0
    row['aurora_type'] = 0
    row['ele_energy_flux'] = 0.0
    row['ion_energy_flux'] = 0.0
    return pd.Series(row)


def infer_season_avg(model, doy_list, fixed_sw, scaler_path, device,
                    year=SEASON_AVG_YEAR, hour=SEASON_AVG_HOUR, season_name=''):
    """对 doy_list 中每个 DOY 推理一次 grid, 返回 4-head 平均 flux 网格 + 平均 HP.

    Args:
        doy_list: list[int] DOY 列表 (1-365); 跨年的冬季已在调用方拼好 (e.g. [315..365, 1..45])
        year:    用于构造 datetime 的年份 (闰年/非闰年对 dipole_tilt 影响 < 0.1°, 可忽略)
        hour:    每天采样的 UT 小时 (默认 12)

    Returns:
        fluxes_avg: list of 4 ndarray (n_mlat, n_mlt), 每 head 网格平均 flux
        hps_avg:    list of 4 float, 每 head 平均 HP (GW)
    """
    n_mlat, n_mlt = len(MLAT_1D), len(MLT_1D)
    fluxes_sum = [np.zeros((n_mlat, n_mlt), dtype=np.float64) for _ in range(4)]
    hps_sum = [0.0 for _ in range(4)]
    n = len(doy_list)
    print(f"   {season_name}: 推理 {n} 个 DOY (year={year}, hour={hour:02d}:00 UT)")
    for i, doy in enumerate(doy_list):
        utc = datetime(year, 1, 1) + timedelta(days=int(doy) - 1, hours=int(hour))
        sw_row = build_steady_sw_row(utc, fixed_sw)
        fluxes = predict_grid_v4(model, sw_row=sw_row,
                                  mlat_1d=MLAT_1D, mlt_1d=MLT_1D,
                                  scaler_path=scaler_path, device=device)
        for k in range(4):
            fluxes_sum[k] += fluxes[k]
            hps_sum[k] += calculate_hp_gw(fluxes[k], MLAT_1D, MLT_1D)
        if (i + 1) % 10 == 0 or (i + 1) == n:
            print(f"     [{i+1:>3}/{n}] DOY {doy:>3} ({utc.strftime('%m-%d')}) ✓")
    fluxes_avg = [f / n for f in fluxes_sum]
    hps_avg = [h / n for h in hps_sum]
    return fluxes_avg, hps_avg


# ===================================================================
# 2. 绘图
# ===================================================================
AURORA_COLORS = [(0, 0, 0), (0, 0, 0.5), (0, 0, 0.8), (0, 0.5, 1), (0, 1, 1),
                 (0.5, 1, 0.5), (1, 1, 0), (1, 0.5, 0), (1, 0, 0), (0.75, 0.75, 0.75)]
AURORA_CMAP = LinearSegmentedColormap.from_list('aurora_cmap', AURORA_COLORS, N=256)

HEAD_NAMES = ['Diffuse', 'Monoenergetic', 'Broadband', 'Ion', 'Total']
HEAD_VMAX = [3.0, 5.0, 1.5, 1.0, 5.0]           # 各 head 物理量程
DIFF_VMAX = [1.5, 2.5, 0.75, 0.5, 2.5]           # Δ 量程 = 各 head abs vmax / 2


def _setup_polar_ax(ax):
    ax.set_theta_zero_location('S')
    ax.set_theta_direction(1)
    ax.set_ylim(0, 40)
    ax.set_yticklabels([])
    hour_ticks = np.arange(0, 24, 1)
    ax.set_xticks(np.deg2rad((hour_ticks / 24.0) * 360))
    ax.set_xticklabels([str(h) if h in [0, 6, 12, 18] else '' for h in hour_ticks],
                       fontsize=11)
    lat_circles = [60, 70, 80]
    ax.set_rticks([90 - lat for lat in lat_circles])
    ax.set_yticklabels([f'{lat}°' for lat in lat_circles],
                       fontsize=8, color='white')


def plot_pure_time_3x5(fluxes_s, hps_s, fluxes_w, hps_w,
                       summer_label, winter_label, sw_label, save_path):
    """3 行 (Summer / Winter / Δ) × 5 列 (D/M/B/I/Total) 极坐标图.

    summer_label / winter_label 既支持纯字符串, 也支持 (主名称, 时间)
    二元组. 元组形式下, 主名称放在最左列, 时间放在主名称右一列, 避免
    长时间串旋转 90° 后越界与相邻行的主名称重叠.
    """
    n_cols = 5
    MLT, MLAT = np.meshgrid(MLT_1D, MLAT_1D)
    theta = (MLT / 24.0) * 2 * np.pi
    r = 90.0 - MLAT

    fig, axs = plt.subplots(3, n_cols, figsize=(24, 15.5),
                            subplot_kw={'projection': 'polar'})
    fig.patch.set_facecolor('white')

    # ---- 扩展数据: 追加 Total (4 head 求和) ----
    def _append_total(fluxes_4, hps_4):
        total_flux = sum(fluxes_4)
        total_hp = sum(hps_4)
        return list(fluxes_4) + [total_flux], list(hps_4) + [total_hp]

    fluxes_s5, hps_s5 = _append_total(fluxes_s, hps_s)
    fluxes_w5, hps_w5 = _append_total(fluxes_w, hps_w)

    fluxes_d5 = [fluxes_s5[k] - fluxes_w5[k] for k in range(n_cols)]
    hps_d5 = [hps_s5[k] - hps_w5[k] for k in range(n_cols)]

    # 把行标签统一成 (主名称, 时间) 二元组, 字符串则视为只有主名称
    def _split_label(lbl):
        if isinstance(lbl, tuple):
            return lbl[0], (lbl[1] if len(lbl) > 1 else '')
        return lbl, ''

    summer_main, summer_time = _split_label(summer_label)
    winter_main, winter_time = _split_label(winter_label)

    row_data = [
        (summer_main, summer_time, fluxes_s5, hps_s5, AURORA_CMAP, HEAD_VMAX, False),
        (winter_main, winter_time, fluxes_w5, hps_w5, AURORA_CMAP, HEAD_VMAX, False),
        ('$\\Delta$  (Summer $-$ Winter)', '',
         fluxes_d5, hps_d5, 'RdBu_r', DIFF_VMAX, True),
    ]

    meshes = [[None] * n_cols for _ in range(3)]
    for row, (row_main, row_time, fluxes, hps, cmap, vmaxes, is_diff) in enumerate(row_data):
        for col in range(n_cols):
            ax = axs[row, col]
            _setup_polar_ax(ax)
            vmax = vmaxes[col]
            vmin = -vmax if is_diff else 0
            c = ax.pcolormesh(theta, r, fluxes[col], cmap=cmap,
                              shading='nearest', vmin=vmin, vmax=vmax)
            meshes[row][col] = c

            if row == 0:
                ax.set_title(HEAD_NAMES[col], fontsize=16, weight='bold', pad=14)

            # HP 标签
            hp_label = f'$\\Delta$HP = {hps[col]:+.1f} GW' if is_diff else f'HP = {hps[col]:.1f} GW'
            color = 'darkblue' if is_diff and hps[col] > 0 else ('darkred' if is_diff else 'darkred')
            ax.text(-0.05, 1.04, hp_label,
                    transform=ax.transAxes,
                    fontsize=11, weight='bold', color=color,
                    bbox=dict(boxstyle='round,pad=0.25',
                              facecolor='white', alpha=0.92,
                              edgecolor='gray'),
                    va='top', ha='left')

        # 行标签拆两列: 主名称 (大字) + 时间 (小字)
        y_centers = [0.81, 0.5, 0.19]
        fig.text(0.015, y_centers[row], row_main,
                 rotation=90, fontsize=14, weight='bold',
                 va='center', ha='center')
        if row_time:
            fig.text(0.040, y_centers[row], row_time,
                     rotation=90, fontsize=10,
                     va='center', ha='center')

    # 共享 colorbar (每列 2 条: 行 1-2 同色阶, 行 3 独立)
    plt.subplots_adjust(left=0.07, right=0.98, top=0.87, bottom=0.10,
                        wspace=0.35, hspace=0.30)

    for col in range(n_cols):
        bbox_top = axs[1, col].get_position()
        cax_top = fig.add_axes([bbox_top.x0 + 0.01, bbox_top.y0 - 0.025,
                                bbox_top.width - 0.02, 0.008])
        cb = fig.colorbar(meshes[0][col], cax=cax_top, orientation='horizontal')
        cb.ax.tick_params(labelsize=8)
        if col == 0:
            cb.set_label('Absolute flux  [erg cm$^{-2}$ s$^{-1}$]',
                         fontsize=9, labelpad=2)

        bbox_bot = axs[2, col].get_position()
        cax_bot = fig.add_axes([bbox_bot.x0 + 0.01, bbox_bot.y0 - 0.025,
                                bbox_bot.width - 0.02, 0.008])
        cb_d = fig.colorbar(meshes[2][col], cax=cax_bot, orientation='horizontal')
        cb_d.ax.tick_params(labelsize=8)
        if col == 0:
            cb_d.set_label(r'$\Delta$ flux (S$-$W)', fontsize=9, labelpad=2)

    fig.suptitle(
        f'Pure Seasonal Effect — Northern Hemisphere\n'
        f'Fixed SW: {sw_label}',
        fontsize=18, weight='bold', y=0.97,
    )
    plt.savefig(save_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"\n✅ 季节纯时间效应 3×5 图已保存: {save_path}")


# ===================================================================
# 3. 主程序
# ===================================================================
def main():
    device = "npu:0"

    # ---- 加载 V4 MLP ----
    print(f"⚙️  加载 V4 MLP: {MODEL_PATH}")
    model = load_v4_model(model_path=MODEL_PATH, device=device)

    print(f"\n   MODE: {MODE}")
    print(f"   固定 SW: {SW_LABEL}")

    if MODE == 'season_avg':
        print(f"\n🌞 整夏季平均 (DOY {SUMMER_DOYS[0]}-{SUMMER_DOYS[-1]}, n={len(SUMMER_DOYS)} 天)")
        fluxes_s, hps_s = infer_season_avg(
            model, SUMMER_DOYS, FIXED_SW,
            scaler_path=SCALER_PATH, device=device,
            season_name='🌞 Summer Avg')
        print(f"\n❄️  整冬季平均 (DOY {WINTER_DOYS[0]}-...-{WINTER_DOYS[-1]} 跨年, n={len(WINTER_DOYS)} 天)")
        fluxes_w, hps_w = infer_season_avg(
            model, WINTER_DOYS, FIXED_SW,
            scaler_path=SCALER_PATH, device=device,
            season_name='❄️ Winter Avg')
        summer_label = (
            'Summer Avg',
            f'(DOY {SUMMER_DOYS[0]}–{SUMMER_DOYS[-1]}, '
            f'{len(SUMMER_DOYS)} days, 12:00 UT)',
        )
        winter_label = (
            'Winter Avg',
            f'(DOY 315–45 wrap, {len(WINTER_DOYS)} days, 12:00 UT)',
        )
        title_suffix = 'Season-Averaged'
        out_tag = 'season_avg'
    elif MODE == 'point':
        print(f"\n🌞 合成 Summer SW 行 ({SUMMER_UTC})")
        sw_summer = build_steady_sw_row(SUMMER_UTC, FIXED_SW)
        print(f"❄️  合成 Winter SW 行 ({WINTER_UTC})")
        sw_winter = build_steady_sw_row(WINTER_UTC, FIXED_SW)
        print(f"   Summer DOY: {SUMMER_UTC.timetuple().tm_yday} (北半球夏至)")
        print(f"   Winter DOY: {WINTER_UTC.timetuple().tm_yday} (北半球冬至)")
        print(f"\n🔮 Summer 网格预测...")
        fluxes_s = predict_grid_v4(model, sw_row=sw_summer,
                                    mlat_1d=MLAT_1D, mlt_1d=MLT_1D,
                                    scaler_path=SCALER_PATH, device=device)
        print(f"🔮 Winter 网格预测...")
        fluxes_w = predict_grid_v4(model, sw_row=sw_winter,
                                    mlat_1d=MLAT_1D, mlt_1d=MLT_1D,
                                    scaler_path=SCALER_PATH, device=device)
        hps_s = [calculate_hp_gw(fluxes_s[k], MLAT_1D, MLT_1D) for k in range(4)]
        hps_w = [calculate_hp_gw(fluxes_w[k], MLAT_1D, MLT_1D) for k in range(4)]
        summer_label = (
            'Summer',
            f'(DOY {SUMMER_UTC.timetuple().tm_yday}, '
            f'{SUMMER_UTC.strftime("%m-%d %H:%M UT")})',
        )
        winter_label = (
            'Winter',
            f'(DOY {WINTER_UTC.timetuple().tm_yday}, '
            f'{WINTER_UTC.strftime("%m-%d %H:%M UT")})',
        )
        title_suffix = 'Single-Point'
        out_tag = 'point'
    else:
        raise ValueError(f"未知 MODE: {MODE!r}, 仅支持 'season_avg' 或 'point'")

    # ---- 控制台对比表 ----
    print("\n" + "=" * 78)
    print(f"📊 纯时间季节效应 [{title_suffix}]  ({SW_LABEL})")
    print("=" * 78)
    print(f"{'Head':>14s} | {'Summer HP':>10s} | {'Winter HP':>10s} | "
          f"{'ΔHP (S-W)':>10s} | {'Δ%':>8s}")
    print("-" * 78)
    for k in range(4):
        s, w = hps_s[k], hps_w[k]
        d = s - w
        dp = d / max(abs(w), 1e-6) * 100.0
        print(f"{HEAD_NAMES[k]:>14s} | {s:>10.2f} | {w:>10.2f} | "
              f"{d:>+10.2f} | {dp:>+7.1f}%")
    tot_s, tot_w = sum(hps_s), sum(hps_w)
    tot_d = tot_s - tot_w
    print("-" * 78)
    print(f"{'TOTAL':>14s} | {tot_s:>10.2f} | {tot_w:>10.2f} | "
          f"{tot_d:>+10.2f} | {tot_d / max(abs(tot_w), 1e-6) * 100:>+7.1f}%")
    print("=" * 78)
    print(f"   Winter / Summer = {tot_w / max(tot_s, 1e-6):.3f}")

    # ---- 绘图 ----
    save_path = os.path.join(OUTPUT_DIR, f"season_pure_time_3x5_{out_tag}.png")
    plot_pure_time_3x5(fluxes_s, hps_s, fluxes_w, hps_w,
                       summer_label, winter_label, SW_LABEL, save_path)

    # ---- 保存数值 csv ----
    summary_path = os.path.join(OUTPUT_DIR, f"season_pure_time_summary_{out_tag}.csv")
    pd.DataFrame({
        'head': HEAD_NAMES,
        'hp_summer_gw': list(hps_s) + [tot_s],
        'hp_winter_gw': list(hps_w) + [tot_w],
        'hp_diff_gw':   [s - w for s, w in zip(hps_s, hps_w)] + [tot_d],
    }).to_csv(summary_path, index=False, float_format='%.3f')
    print(f"✅ HP 汇总已保存: {summary_path}")


if __name__ == "__main__":
    main()
