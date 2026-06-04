"""
infer_climate.py — V4 气候态极光图 (Quiet / Moderate / Active)
=====================================================================
按太阳风 Bz / V 把测试集分成 3 个条件, 取各自 SW 均值, 在 V4 模型上
走一次 (mlat × mlt) 网格推理, 画出平均极光能量通量场 + HP 总能.
"""
import sys
import os
import torch
import torch_npu  # noqa: F401
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

BASE_DIR = "/home/docker/code/Aurora_MLP_final"
sys.path.append(BASE_DIR)

from infer_v4_utils import load_v4_model, predict_grid_v4

# ================= 新增：半球功率 (HP) 积分计算 =================
def calculate_hp_gw(flux_grid, mlat_1d, mlt_1d):
    """计算网格上的总半球功率 (Hemispheric Power)，单位：GW"""
    R_E = 6.3712e8  # 地球半径 (厘米)
    
    d_mlat = np.deg2rad(mlat_1d[1] - mlat_1d[0])
    d_mlt = (mlt_1d[1] - mlt_1d[0]) * (2.0 * np.pi / 24.0)
    
    mlat_rad = np.deg2rad(mlat_1d)
    area_1d = (R_E ** 2) * np.cos(mlat_rad) * d_mlat * d_mlt
    area_2d = np.tile(area_1d.reshape(-1, 1), (1, len(mlt_1d)))
    
    total_ergs_per_s = np.sum(flux_grid * area_2d)
    hp_gw = total_ergs_per_s * 1e-16
    return hp_gw

HEAD_NAMES = ['Diffuse', 'Monoenergetic', 'Broadband', 'Ion', 'Total']
# 每行单独的 vmax, 避免弱通道被强通道压平
HEAD_VMAX = [2.0, 3.0, 1.0, 0.5, 3.0]


def plot_climatology_3x5(flux_4heads_list, hp_head_list, save_path):
    """
    绘制 3×5 极地气候态对比图 (横版: 3 行 SW × 5 列 head).
    flux_4heads_list: 长度 3 的 list, 每个元素 shape (4, n_mlat, n_mlt)
                      顺序 [Quiet, Moderate, Storm]
    hp_head_list:     长度 3 的 list, 每个元素长度 5 [hp_d, hp_m, hp_b, hp_i, hp_total]
    """
    mlat = np.linspace(50, 90, 80)
    mlt = np.linspace(0, 24, 144)
    MLT, MLAT = np.meshgrid(mlt, mlat)
    theta = (MLT / 24.0) * 2 * np.pi
    r = 90.0 - MLAT

    colors = [(0, 0, 0), (0, 0, 0.5), (0, 0, 0.8), (0, 0.5, 1), (0, 1, 1),
              (0.5, 1, 0.5), (1, 1, 0), (1, 0.5, 0), (1, 0, 0), (0.8, 0.8, 0.8)]
    cmap = LinearSegmentedColormap.from_list('aurora_cmap', colors, N=256)

    row_titles = [
        r'Quiet ($B_z>0,\;V<400$)',
        r'Moderate ($-5<B_z\leq0$)',
        r'Storm ($B_z\leq-5,\;V\geq500$)',
    ]

    fig, axs = plt.subplots(3, 5, figsize=(30, 18),
                            subplot_kw={'projection': 'polar'})

    for row in range(3):
        for col in range(5):
            ax = axs[row, col]
            ax.set_theta_zero_location('S')
            ax.set_theta_direction(1)
            ax.set_ylim(0, 40)
            ax.set_yticklabels([])

            hour_ticks = np.arange(0, 24, 1)
            ax.set_xticks(np.deg2rad((hour_ticks / 24.0) * 360))
            ax.set_xticklabels(
                [str(h) if h in [0, 6, 12, 18] else '' for h in hour_ticks],
                fontsize=10,
            )

            lat_circles = [50, 60, 70, 80]
            ax.set_rticks([90 - lat for lat in lat_circles])
            ax.set_yticklabels([f'{lat}°' for lat in lat_circles],
                               fontsize=7, color='white')

            # 选数据: col 0-3 = 4 heads, col 4 = total
            flux_4 = flux_4heads_list[row]          # (4, mlat, mlt)
            if col < 4:
                data = flux_4[col]
            else:
                data = flux_4.sum(axis=0)

            vmax = HEAD_VMAX[col]
            c = ax.pcolormesh(theta, r, data, cmap=cmap,
                              shading='nearest', vmin=0, vmax=vmax)

            # HP 标注
            hp_val = hp_head_list[row][col]
            ax.text(0.0, 1.05, f"HP: {hp_val:.2f} GW",
                    transform=ax.transAxes, fontsize=11, weight='bold',
                    color='darkred',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              alpha=0.8, edgecolor='gray'),
                    va='top', ha='left')

            # 列标题 (仅首行) — head 名称
            if row == 0:
                ax.set_title(HEAD_NAMES[col], fontsize=15, weight='bold',
                             pad=18)

            # 行标题 (仅首列) — SW 条件
            if col == 0:
                ax.text(-0.22, 0.5, row_titles[row],
                        transform=ax.transAxes, fontsize=13, weight='bold',
                        rotation=90, va='center', ha='center')

    # 全局布局 (预留底部 colorbar 条带)
    plt.subplots_adjust(top=0.90, bottom=0.12, left=0.06, right=0.98,
                        hspace=0.30, wspace=0.30)

    # 每列底部加横向 colorbar (每个 head 独立 vmax)
    n_col = 5
    fig_left, fig_right = 0.06, 0.98
    col_w = (fig_right - fig_left) / n_col
    for col in range(n_col):
        col_center = fig_left + (col + 0.5) * col_w
        cbar_w = col_w * 0.55
        cbar_left = col_center - cbar_w / 2
        cbar_ax = fig.add_axes([cbar_left, 0.06, cbar_w, 0.012])
        sm = plt.cm.ScalarMappable(cmap=cmap,
                                   norm=plt.Normalize(0, HEAD_VMAX[col]))
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        cbar.ax.tick_params(labelsize=9)
        cbar.set_label(f'{HEAD_NAMES[col]}  (ergs cm$^{{-2}}$ s$^{{-1}}$)',
                       fontsize=9)

    fig.suptitle('Climatological Auroral Patterns by Solar Wind Drivers',
                 fontsize=20, weight='bold', y=0.98)
    plt.savefig(save_path, dpi=300, facecolor='white', bbox_inches='tight')
    print(f"✅ 3×5 极地气候态对比图已保存至: {save_path}")
    plt.close()

def main():
    # ============ V4 路径 ============
    TEST_DATA_PATH = "/home/docker/data/private/AuroraData/final_test_v4_phys.parquet"
    CKPT_DIR = "/home/docker/code/Aurora_MLP_final/ckpt/ckpt_v4_simple_different_10"
    MODEL_PATH = f"{CKPT_DIR}/aurora_v4_best.pth"
    SCALER_PATH = f"{CKPT_DIR}/scaler/sw_scaler_v4.pkl"
    OUTPUT_DIR = "/home/docker/code/Aurora_MLP_final/res/v4_different_10"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "npu:0"

    # ============ 2. 加载数据 ============
    print("📥 加载 V4 测试集...")
    test_df = pd.read_parquet(TEST_DATA_PATH)
    test_df['utc'] = pd.to_datetime(test_df['utc'])
    print(f"   测试集: {len(test_df):,}")

    # 计算 V 模 (V4 保留 Vx/Vy/Vz 分量)
    test_df['V_mag'] = np.sqrt(
        test_df['Vx']**2 + test_df['Vy']**2 + test_df['Vz']**2,
    )

    # ============ 3. 物理驱动分类 (用当前 Bz, V) ============
    print("🔍 物理驱动分类 (Bz, V_mag)...")
    quiet_df  = test_df[(test_df['Bz'] > 0) & (test_df['V_mag'] < 400)]
    mod_df    = test_df[(test_df['Bz'] <= 0) & (test_df['Bz'] > -5)
                        & (test_df['V_mag'] >= 400) & (test_df['V_mag'] < 500)]
    active_df = test_df[(test_df['Bz'] <= -5) & (test_df['V_mag'] >= 500)]
    print(f"   Quiet:    {len(quiet_df):,}")
    print(f"   Moderate: {len(mod_df):,}")
    print(f"   Active:   {len(active_df):,}")

    # 采样以限制计算 (不影响均值, 只加速)
    sample_size = min(200000, len(quiet_df), len(mod_df), len(active_df))
    if sample_size <= 0:
        raise RuntimeError("某类 SW 条件下样本为空, 请检查 V4 测试集范围.")
    quiet_df  = quiet_df.sample(n=sample_size, random_state=42)  if len(quiet_df) > sample_size else quiet_df
    mod_df    = mod_df.sample(n=sample_size, random_state=42)    if len(mod_df) > sample_size else mod_df
    active_df = active_df.sample(n=sample_size, random_state=42) if len(active_df) > sample_size else active_df
    print(f"   每类采样 -> {sample_size:,} 点")

    # ============ 4. 加载 V4 模型 ============
    print("⚙️ 加载 V4 MLP 模型...")
    model = load_v4_model(model_path=MODEL_PATH, device=device)

    mlat_1d = np.linspace(50, 90, 80)
    mlt_1d = np.linspace(0, 24, 144)

    def compute_average_climatology(target_df):
        """取 target_df 里所有数值列的均值作为代表 SW row,
        选中位数时间作为 utc, 走 V4 grid 推理.
        返回 flux_4heads (4, mlat, mlt) + hp_5 [d, m, b, i, total].
        """
        sw_row = target_df.select_dtypes(include=[np.number]).mean()
        sw_row['utc'] = target_df['utc'].sort_values().iloc[len(target_df) // 2]
        sw_row['mlat'] = 70.0
        sw_row['mlt'] = 12.0

        flux_4 = predict_grid_v4(
            model, sw_row=sw_row, mlat_1d=mlat_1d, mlt_1d=mlt_1d,
            scaler_path=SCALER_PATH, device=device,
        )   # (4, n_mlat, n_mlt)

        hp_per_head = [calculate_hp_gw(flux_4[h], mlat_1d, mlt_1d) for h in range(4)]
        hp_total = calculate_hp_gw(flux_4.sum(axis=0), mlat_1d, mlt_1d)
        hp_5 = hp_per_head + [hp_total]
        return flux_4, hp_5

    # ============ 5. 计算 ============
    flux_list, hp_list = [], []
    for label, sub_df in [('Quiet', quiet_df), ('Moderate', mod_df), ('Storm', active_df)]:
        print(f"🔮 推演 {label} 气候态...")
        f4, hp5 = compute_average_climatology(sub_df)
        flux_list.append(f4)
        hp_list.append(hp5)
        print(f"   HP: D={hp5[0]:.2f}  M={hp5[1]:.2f}  B={hp5[2]:.2f}  I={hp5[3]:.2f}  Total={hp5[4]:.2f} GW")

    save_fig_path = os.path.join(OUTPUT_DIR, "v4_climatological_polar_3x5.png")
    plot_climatology_3x5(flux_list, hp_list, save_fig_path)

if __name__ == "__main__":
    main()