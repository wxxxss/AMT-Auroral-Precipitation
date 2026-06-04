"""
  1. plot_spatial_occurrence     — 四大类极光的高分辨率空间发生频次图
  2. plot_energy_flux_distribution — 极光能量通量的长尾分布图
  3. plot_solar_wind_violin       — 太阳风驱动参数的统计分布 (小提琴图)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns

# 全局美化设置
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)


# ===================================================================
# 1. 空间发生频次图 (极坐标)
# ===================================================================
def plot_spatial_occurrence(df, save_path):
    """
    图1：四大类极光的高分辨率空间发生频次图 (Spatial Occurrence Distribution)
    """
    print("正在绘制极光高分辨率空间数量分布图...")
    
    # 物理过滤：只提取真正的"发光点"（通量大于 1e-4），过滤掉背景暗区
    active_thresh = 1e-4
    pts_diffuse = df[(df['aurora_type'] == 1) & (df['ele_energy_flux'] > active_thresh)]
    pts_mono = df[(df['aurora_type'] == 2) & (df['ele_energy_flux'] > active_thresh)]
    pts_bb = df[(df['aurora_type'] == 3) & (df['ele_energy_flux'] > active_thresh)]
    pts_ion = df[df['ion_energy_flux'] > active_thresh]
    
    datasets = [pts_diffuse, pts_mono, pts_bb, pts_ion]
    display_names = ['Diffuse Aurora', 'Monoenergetic Aurora', 'Broadband Aurora', 'Ion Aurora']
    
    fig, axs = plt.subplots(1, 4, figsize=(24, 6), subplot_kw={'projection': 'polar'})
    fig.patch.set_facecolor('white')
    
    # 高分辨率网格定义
    bins_theta = np.linspace(0, 2 * np.pi, 97) # 96 个 MLT 区间
    bins_r = np.linspace(0, 40, 81)            # 80 个 MLAT 区间 (50-90)
    
    # 使用高级的热力图配色 (Magma 非常适合展示频次和密度)
    cmap = plt.get_cmap('magma').copy()
    cmap.set_bad(color='#f0f0f0') # 没有数据的地方显示浅灰色
    
    for i, ax in enumerate(axs):
        ax.set_theta_zero_location("S")
        ax.set_theta_direction(1)
        ax.set_ylim(0, 40)
        ax.set_yticklabels([])
        
        hour_ticks = np.arange(0, 24, 3)
        ax.set_xticks(np.deg2rad((hour_ticks / 24.0) * 360))
        ax.set_xticklabels([f'{h:02d}' for h in hour_ticks], fontsize=12)
        
        lat_circles = [60, 70, 80]
        ax.set_rticks([90 - lat for lat in lat_circles])
        ax.set_yticklabels([f'{lat}°' for lat in lat_circles], fontsize=8, color='gray')
        
        # 提取坐标并转换
        target_df = datasets[i]
        theta = (target_df['mlt'] / 24.0) * 2 * np.pi
        r = 90.0 - target_df['mlat']
        
        # 直接使用 hist2d 统计"数量 (Count)"，LogNorm 突显空间结构
        h, xedges, yedges, image = ax.hist2d(
            theta, r, bins=[bins_theta, bins_r], 
            cmap=cmap, norm=LogNorm(), cmin=1
        )
        
        # 标题：名称 + 总数据点数量
        ax.set_title(f"{display_names[i]}\nTotal Points: {len(target_df):,}", 
                     fontsize=16, weight='bold', pad=15)
        
        # 独立 Colorbar (每个子图都加 label)
        cbar = plt.colorbar(image, ax=ax, fraction=0.046, pad=0.08, orientation='horizontal')
        cbar.set_label('Observation Counts (Log Scale)', fontsize=12, weight='bold')

    plt.suptitle("High-Resolution Spatial Occurrence Distribution of Auroral Types", 
                 fontsize=20, weight='bold', y=1.10)
    # 增加这一行：自动紧凑排版，但限制所有子图只能画在 [left, bottom, right, top] 的区域内
    plt.savefig(save_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"✅ 空间发生频次图已保存至: {save_path}")


# ===================================================================
# 2. 能量通量数值分布图
# ===================================================================
def plot_energy_flux_distribution(df_sampled, save_path):
    """图2：极光能量通量的长尾分布图 (恢复了高级的 KDE 平滑曲线)"""
    print("正在绘制极光能量通量密度分布图...")
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')
    
    types = {
        'Diffuse': df_sampled[df_sampled['aurora_type'] == 1]['ele_energy_flux'],
        'Monoenergetic': df_sampled[df_sampled['aurora_type'] == 2]['ele_energy_flux'],
        'Broadband': df_sampled[df_sampled['aurora_type'] == 3]['ele_energy_flux'],
        'Ion': df_sampled['ion_energy_flux']
    }
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for (name, flux), color in zip(types.items(), colors):
        valid_flux = flux.dropna()
        valid_flux = valid_flux[valid_flux > 1e-4]
        log_flux = np.log10(valid_flux)
        
        # 恢复 KDE (核密度估计曲线)，同时保留直方图的阶梯轮廓
        sns.histplot(log_flux, bins=80, color=color, label=name, 
                     kde=True, stat="density", element="step", fill=True, alpha=0.15, ax=ax)

    ax.set_title('Probability Density Distribution of Auroral Energy Fluxes', fontsize=16, weight='bold', pad=15)
    ax.set_xlabel('Log$_{10}$ Energy Flux [ergs cm$^{-2}$ s$^{-1}$]', fontsize=14, weight='bold')
    ax.set_ylabel('Density (Probability)', fontsize=14, weight='bold')
    
    ax.legend(title='Auroral Types', fontsize=12, title_fontsize=13, loc='upper left', framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, facecolor='white')
    plt.close()
    print(f"✅ 能量通量分布图已保存至: {save_path}")


# ===================================================================
# 3. 太阳风参数小提琴图 (已适配 V4 列名)
# ===================================================================
def plot_solar_wind_violin(df_sampled, save_path):
    """图3：太阳风驱动参数的统计分布 (V4 数据格式)"""
    print("正在绘制太阳风参数统计小提琴图...")
    fig, axs = plt.subplots(1, 5, figsize=(20, 5))
    fig.patch.set_facecolor('white')
    
    params = [
        ('Bz',    'Z-component of IMF ($B_z$)\n[nT]',       '#1f77b4'),
        ('By',    'Y-component of IMF ($B_y$)\n[nT]',       '#9467bd'),
        ('V_mag', 'Flow Speed ($V$)\n[km/s]',               '#2ca02c'),
        ('Bx',    'X-component of IMF ($B_x$)\n[nT]',       '#ff7f0e'),
        ('P_dyn', 'Flow Pressure ($P_{dyn}$)\n[nPa]',       '#d62728')
    ]
    
    for i, (col, title_label, color) in enumerate(params):
        ax = axs[i]
        
        q_low = df_sampled[col].quantile(0.01)
        q_hi  = df_sampled[col].quantile(0.99)
        filtered_data = df_sampled[(df_sampled[col] > q_low) & (df_sampled[col] < q_hi)][col]
        
        sns.violinplot(y=filtered_data, ax=ax, color=color, inner="quartile", alpha=0.7)
        
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.set_xticks([]) 
        ax.set_title(title_label, fontsize=14, weight='bold', pad=10, loc='center')
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)
        
        if 'B_z' in title_label or 'B_y' in title_label or 'B_x' in title_label:
            ax.axhline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)

    plt.subplots_adjust(wspace=0.3)
    plt.savefig(save_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"✅ 太阳风参数小提琴图已保存至: {save_path}")


# ===================================================================
# 主程序
# ===================================================================
def main():
    TRAIN_DATA_PATH = "/home/docker/data/private/AuroraData/ssj_2009_2013_classified_fold.parquet"
    OUTPUT_DIR = "/home/docker/code/Aurora_MLP_final/res/data_distribution"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"📥 正在加载数据: {TRAIN_DATA_PATH}")
    df_full = pd.read_parquet(TRAIN_DATA_PATH)
    print(f"   总样本: {len(df_full):,}")

    # V4 数据中无单独 V_mag 列, 需从 Vx/Vy/Vz 合成
    if 'V_mag' not in df_full.columns:
        df_full['V_mag'] = np.sqrt(
            df_full['Vx'] ** 2 + df_full['Vy'] ** 2 + df_full['Vz'] ** 2
        )

    # 图 1: 空间发生频次
    plot_spatial_occurrence(
        df_full, os.path.join(OUTPUT_DIR, "fig1_spatial_occurrence.png"))

    # # 图 2: 能量通量分布 (大数据集用抽样加速 KDE)
    if len(df_full) > 1_000_000:
        df_sampled = df_full.sample(n=1_000_000, random_state=42)
    else:
        df_sampled = df_full
    # plot_energy_flux_distribution(
    #     df_sampled, os.path.join(OUTPUT_DIR, "fig2_flux_distribution.png"))

    # 图 3: 太阳风参数小提琴图
    plot_solar_wind_violin(
        df_sampled, os.path.join(OUTPUT_DIR, "fig3_solar_wind_violin.png"))

    print("\n🎉 三张数据分布图全部生成完毕！")


if __name__ == "__main__":
    main()
