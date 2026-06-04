"""
infer_v4_utils.py — V4 推理共享工具
=====================================
所有 infer_*.py 推理脚本统一通过这里加载 V4 模型 / dataset / 推理.

主要 API:
  load_v4_model(model_path, sw_dim, skip_dim, device) -> nn.Module (eval 模式)
  build_v4_dataset(df, scaler_path) -> AuroraMultiTaskDataset_V4
  predict_v4_batched(model, ds, device, batch_size, return_log) -> ndarray (N,4)
  run_mlp_4heads_v4(df, model, scaler_path, device) -> df (追加 pred_mlp_{d,m,b,i})
  predict_grid_v4(model, scaler_path, sw_row, mlat_1d, mlt_1d, device) -> (4, n_mlat, n_mlt)

V4_DEFAULTS 提供默认 ckpt/scaler/parquet 路径, 调用方可全部 override.
"""
import os
import sys
import torch
import torch_npu  # noqa: F401  npu backend init
import numpy as np
import pandas as pd

BASE_DIR = "/home/docker/code/Aurora_MLP_final"
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from data.dataset_v4 import (
    AuroraMultiTaskDataset_V4,
    compute_derived_block,
    compute_dipole_tilt_rad,
    OMNI_BASE_VARS,
    LAG_MINUTES,
    DERIVED_SCALAR_NAMES,
)
from method.model import AMT


# ===================================================================
# 默认配置 (调用方 override 即可)
# ===================================================================
V4_DEFAULTS = {
    'ckpt_dir': f'{BASE_DIR}/ckpt/ckpt_v4_phys',
    'model_path': f'{BASE_DIR}/ckpt/ckpt_v4_phys/aurora_v4_best.pth',
    'scaler_path': f'{BASE_DIR}/ckpt/ckpt_v4_phys/scaler/sw_scaler_v4.pkl',
    'val_parquet': '/home/docker/data/private/AuroraData/final_val_v4_phys.parquet',
    'test_parquet': '/home/docker/data/private/AuroraData/final_test_v4_phys.parquet',
    'train_parquet': '/home/docker/data/private/AuroraData/final_train_v4_phys.parquet',
    # 模型架构超参 (与 train_v4*.py 一致)
    'sw_dim': 116,
    'skip_dim': 9,
    'hidden_wide': 1024,
    'hidden_mid': 512,
    'latent_dim': 256,
    'head_hidden': 128,
    'dropout': 0.2,
}


# ===================================================================
# 1) 模型加载
# ===================================================================
def load_v4_model(model_path=None,
                  sw_dim=None,
                  skip_dim=None,
                  device='npu:0',
                  cfg_overrides=None):
    """
    加载 V4 共享 backbone 模型.

    Args:
        model_path:  ckpt 路径, 默认 V4_DEFAULTS['model_path']
        sw_dim:      太阳风 + 衍生维度, 默认 116
        skip_dim:    skip (空间/时间/物理) 维度, 默认 9
        device:      'npu:0' / 'cuda:0' / 'cpu'
        cfg_overrides: dict, 覆盖 hidden_wide/hidden_mid/latent_dim/head_hidden/dropout

    Returns:
        model (nn.Module, eval mode)
    """
    cfg = dict(V4_DEFAULTS)
    if cfg_overrides:
        cfg.update(cfg_overrides)

    model_path = model_path or cfg['model_path']
    sw_dim = sw_dim if sw_dim is not None else cfg['sw_dim']
    skip_dim = skip_dim if skip_dim is not None else cfg['skip_dim']

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ V4 ckpt 不存在: {model_path}")

    model = AMT(
        sw_dim=sw_dim, skip_dim=skip_dim,
        hidden_wide=cfg['hidden_wide'], hidden_mid=cfg['hidden_mid'],
        latent_dim=cfg['latent_dim'], head_hidden=cfg['head_hidden'],
        dropout=cfg['dropout'],
    ).to(device)

    state = torch.load(model_path, map_location=device)
    # 兼容保存格式: 纯 state_dict 或 {'model_state_dict': ...}
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    model.load_state_dict(state)
    model.eval()
    return model


# ===================================================================
# 2) Dataset 构建 (容错: 自动补缺失的 target 列)
# ===================================================================
def build_v4_dataset(df, scaler_path=None):
    """
    把 df 包成 AuroraMultiTaskDataset_V4 (推理模式).

    df 必须包含:
      - utc, mlat, mlt
      - Bx, By, Bz, Vx, Vy, Vz, P_dyn (当前)
      - 所有 _lag_5, _lag_10, ..., _lag_120 列 (24 步 × 7 量 = 168 列)

    若 df 缺少 aurora_type / ele_energy_flux / ion_energy_flux,
    自动补 0 (推理时 target 用不到, 仅供 dataset 内部走 _prepare_targets 流程).
    """
    scaler_path = scaler_path or V4_DEFAULTS['scaler_path']
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"❌ V4 Scaler 不存在: {scaler_path}")

    df = df.copy()
    for col, default in [
        ('aurora_type', 0),
        ('ele_energy_flux', 0.0),
        ('ion_energy_flux', 0.0),
    ]:
        if col not in df.columns:
            df[col] = default
    return AuroraMultiTaskDataset_V4(
        df=df, is_train=False, scaler_path=scaler_path,
    )


# ===================================================================
# 3) 批量推理 (对一个已经构建好的 dataset)
# ===================================================================
def predict_v4_batched(model, ds,
                       device='npu:0',
                       batch_size=32768,
                       return_log=False):
    """
    对 V4 dataset 批量推理, 返回 (N, 4) ndarray.

    Args:
        return_log: True → log10 flux (模型直接输出); False → 线性 flux (默认).
                    线性公式: lin = 10^log_pred - 1e-6, 负值 clip 0
                    (与 train_v4 dataset 的 bias=1e-6 反推一致)
    """
    n = len(ds)
    out = np.zeros((n, 4), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for i in range(0, n, batch_size):
            X_sw = ds.X_sw_tensor[i:i + batch_size].to(device)
            X_skip = ds.X_skip_tensor[i:i + batch_size].to(device)
            log_pred = model(X_sw, X_skip).cpu().numpy()
            if return_log:
                out[i:i + len(log_pred)] = log_pred
            else:
                lin = (10.0 ** log_pred) - 1e-6
                lin[lin < 0] = 0.0
                out[i:i + len(lin)] = lin
    return out


# ===================================================================
# 4) 4 头预测追加到 df (与旧 run_mlp_4heads 等价)
# ===================================================================
def run_mlp_4heads_v4(df, model, scaler_path=None,
                     device='npu:0', batch_size=32768,
                     pred_prefix='pred_mlp'):
    """
    给 df 追加 {prefix}_d / _m / _b / _i 四列 (线性 flux, erg/cm²/s).
    返回追加后的 df (in-place + return).

    与旧版 (V1/V2) run_mlp_4heads 接口对齐, 调用方迁移代价最小.
    """
    ds = build_v4_dataset(df, scaler_path=scaler_path)
    out_lin = predict_v4_batched(model, ds, device=device, batch_size=batch_size)
    df[f'{pred_prefix}_d'] = out_lin[:, 0]
    df[f'{pred_prefix}_m'] = out_lin[:, 1]
    df[f'{pred_prefix}_b'] = out_lin[:, 2]
    df[f'{pred_prefix}_i'] = out_lin[:, 3]
    # 推理结束释放 dataset 占用 (parquet 大时尤其重要)
    del ds
    if hasattr(torch, 'npu'):
        torch.npu.empty_cache()
    return df


# ===================================================================
# 5) 单时刻在 (mlat × mlt) 网格上预测 (storm_snapshot / climate 等用)
# ===================================================================
def predict_grid_v4(model, sw_row, mlat_1d, mlt_1d,
                    scaler_path=None, device='npu:0', batch_size=65536):
    """
    给定单时刻 SW 行 (sw_row, pandas Series 或 dict-like, 含所有 OMNI + lag 列),
    在 mlat × mlt 网格上预测 4 通道 flux.

    Args:
        sw_row:    单时刻 SW (含 utc, Bx..P_dyn, 所有 _lag_*); mlat/mlt 列被忽略
        mlat_1d:   shape (n_mlat,) 磁纬度数组 (deg)
        mlt_1d:    shape (n_mlt,)  磁地方时数组 (hour, 0~24)
        scaler_path: V4 scaler pkl
        device:    npu:0

    Returns:
        flux_4heads: shape (4, n_mlat, n_mlt) 线性 flux
                     顺序 [Diff, Mono, BB, Ion]
    """
    n_mlat, n_mlt = len(mlat_1d), len(mlt_1d)
    MLT_g, MLAT_g = np.meshgrid(mlt_1d, mlat_1d)
    n_pts = n_mlat * n_mlt

    # 把 sw_row 复制到每个 grid 点; mlat/mlt 用 grid 坐标
    if hasattr(sw_row, 'to_dict'):
        sw_dict = sw_row.to_dict()
    else:
        sw_dict = dict(sw_row)
    # 移除将要被覆盖的列, 防 dataset 重复
    for k in ('mlat', 'mlt', 'aurora_type', 'ele_energy_flux', 'ion_energy_flux',
              '_dt'):    # 防 storm_snapshot 里临时加的辅助列污染
        sw_dict.pop(k, None)

    grid_data = {k: np.full(n_pts, v) for k, v in sw_dict.items()}
    grid_data['mlat'] = MLAT_g.flatten().astype(np.float32)
    grid_data['mlt'] = (MLT_g.flatten() % 24.0).astype(np.float32)
    # utc 必须是 datetime64, 上面 np.full 会广播 timestamp scalar
    grid_data['utc'] = pd.to_datetime(grid_data['utc'])
    grid_data['aurora_type'] = np.zeros(n_pts, dtype=np.int8)
    grid_data['ele_energy_flux'] = np.zeros(n_pts, dtype=np.float32)
    grid_data['ion_energy_flux'] = np.zeros(n_pts, dtype=np.float32)

    df_grid = pd.DataFrame(grid_data)

    ds = build_v4_dataset(df_grid, scaler_path=scaler_path)
    out_lin = predict_v4_batched(model, ds, device=device, batch_size=batch_size)
    # 返回 (4, n_mlat, n_mlt)
    return out_lin.T.reshape(4, n_mlat, n_mlt)


# ===================================================================
# 5b) 多时刻一次性网格预测 (batched, ~5× 加速)
# ===================================================================
def predict_grid_v4_multi(model, sw_rows, mlat_1d, mlt_1d,
                          scaler_path=None, device='npu:0',
                          batch_size=131072, verbose=False):
    """
    一次性对 K 个时刻在 (mlat × mlt) 网格上批量预测.
    内部把 K × n_pts 行拼成一个大 dataframe -> 一次 dataset 构造 -> 一次推理,
    避免 K 次 dataset 构造 / scaler 加载等开销, 推理 K=4-16 时实测 4-7× 加速.

    Args:
        sw_rows:    list[Series/dict] 或 DataFrame, K 个时刻的 SW 行 (含 utc, Bx.., 所有 lag)
        mlat_1d, mlt_1d: 共用网格
        scaler_path: V4 scaler pkl
        batch_size: NPU 推理每次最多 N 行 (8 时刻 × 11520 ≈ 92k, 给 131k 留余量)

    Returns:
        flux_4heads: shape (K, 4, n_mlat, n_mlt) 线性 flux
    """
    # 统一成 list of dict
    if isinstance(sw_rows, pd.DataFrame):
        rows_iter = (sw_rows.iloc[k] for k in range(len(sw_rows)))
        K = len(sw_rows)
    else:
        rows_iter = iter(sw_rows)
        K = len(sw_rows)
    if K == 0:
        raise ValueError("sw_rows 为空")

    n_mlat, n_mlt = len(mlat_1d), len(mlt_1d)
    MLT_g, MLAT_g = np.meshgrid(mlt_1d, mlat_1d)
    n_pts = n_mlat * n_mlt

    # ---- 1) 拼接 K × n_pts 的 grid_data ----
    # 列顺序由第一行决定; 后续行必须有相同列集合
    drop_cols = ('mlat', 'mlt', 'aurora_type', 'ele_energy_flux',
                 'ion_energy_flux', '_dt')
    chunks = []
    for k, sw_row in enumerate(rows_iter):
        if hasattr(sw_row, 'to_dict'):
            d = sw_row.to_dict()
        else:
            d = dict(sw_row)
        for c in drop_cols:
            d.pop(c, None)

        grid = {col: np.full(n_pts, v) for col, v in d.items()}
        grid['mlat'] = MLAT_g.flatten().astype(np.float32)
        grid['mlt']  = (MLT_g.flatten() % 24.0).astype(np.float32)
        grid['utc'] = pd.to_datetime(grid['utc'])
        grid['aurora_type'] = np.zeros(n_pts, dtype=np.int8)
        grid['ele_energy_flux'] = np.zeros(n_pts, dtype=np.float32)
        grid['ion_energy_flux'] = np.zeros(n_pts, dtype=np.float32)
        chunks.append(pd.DataFrame(grid))

    df_grid = pd.concat(chunks, ignore_index=True)
    if verbose:
        print(f"  [predict_grid_v4_multi] K={K}, total rows={len(df_grid)}")

    # ---- 2) 一次 dataset 构造 + 推理 ----
    ds = build_v4_dataset(df_grid, scaler_path=scaler_path)
    out_lin = predict_v4_batched(model, ds, device=device, batch_size=batch_size)
    # out_lin: (K * n_pts, 4)

    # ---- 3) reshape -> (K, 4, n_mlat, n_mlt) ----
    return (out_lin
            .reshape(K, n_pts, 4)
            .transpose(0, 2, 1)            # (K, 4, n_pts)
            .reshape(K, 4, n_mlat, n_mlt))


# ===================================================================
# 5c) FAST 多时刻网格预测 (绕开 dataset 冗余, ~10-20× 加速)
# ===================================================================
# X_sw 列顺序 (共 116 维), 必须严格匹配 dataset_v4._build_sw_cols
_SW_CURR_DERIVED = ['Newell_log', 'Ey_conv', 'sin_clock', 'cos_clock',
                    'Bz_south_log', 'Bt', 'Bmag', 'Vmag', 'Akasofu_log']
_SW_INTEG_NAMES = ['Newell_log', 'Ey_conv', 'Bz_south_log']

# X_skip 列顺序 (共 9 维)
_SKIP_COLS = ['mlat_scaled', 'sin_mlt', 'cos_mlt',
              'dipole_tilt', 'cos_sza',
              'sin_doy', 'cos_doy', 'sin_hour', 'cos_hour']

_SCALER_CACHE = {}     # path -> StandardScaler


def _get_scaler(scaler_path):
    """缓存加载 scaler, 避免每次重复 joblib.load."""
    import joblib
    sp = scaler_path or V4_DEFAULTS['scaler_path']
    if sp not in _SCALER_CACHE:
        if not os.path.exists(sp):
            raise FileNotFoundError(f"scaler not found: {sp}")
        _SCALER_CACHE[sp] = joblib.load(sp)
    return _SCALER_CACHE[sp]


def _compute_X_sw_per_time(df_K):
    """
    对 K 个时刻的 OMNI 行计算 X_sw, 返回 ndarray (K, 116) float32.
    列顺序与 dataset_v4 sw_cols 严格一致:
      7 OMNI base + 9 当前衍生 + 96 lag(4×24) + 3 累积积分(1h avg) + 1 dPdyn_dt
    """
    K = len(df_K)
    feats = []

    # a) 7 OMNI base 当前
    for c in OMNI_BASE_VARS:
        feats.append(df_K[c].values.astype(np.float32))

    # b) 9 当前衍生
    Bx0 = df_K['Bx'].values.astype(np.float32)
    By0 = df_K['By'].values.astype(np.float32)
    Bz0 = df_K['Bz'].values.astype(np.float32)
    Vx0 = df_K['Vx'].values.astype(np.float32)
    Vy0 = df_K['Vy'].values.astype(np.float32)
    Vz0 = df_K['Vz'].values.astype(np.float32)
    curr = compute_derived_block(Bx0, By0, Bz0, Vx0, Vy0, Vz0)
    for name in _SW_CURR_DERIVED:
        feats.append(curr[name])

    # c) 96 lag 衍生 (列顺序: 先 Newell_log 24 lag, 再 Ey_conv 24 lag, ...)
    lag_cache = {name: {} for name in DERIVED_SCALAR_NAMES}
    for m in LAG_MINUTES:
        Bx = df_K[f'Bx_lag_{m}'].values.astype(np.float32)
        By = df_K[f'By_lag_{m}'].values.astype(np.float32)
        Bz = df_K[f'Bz_lag_{m}'].values.astype(np.float32)
        Vx = df_K[f'Vx_lag_{m}'].values.astype(np.float32)
        Vy = df_K[f'Vy_lag_{m}'].values.astype(np.float32)
        Vz = df_K[f'Vz_lag_{m}'].values.astype(np.float32)
        block = compute_derived_block(Bx, By, Bz, Vx, Vy, Vz)
        for name in DERIVED_SCALAR_NAMES:
            lag_cache[name][m] = block[name]

    for name in DERIVED_SCALAR_NAMES:
        for m in LAG_MINUTES:
            feats.append(lag_cache[name][m])

    # d) 3 累积积分 (过去 1h: 当前 + lag 5..55 共 12 点平均)
    for name in _SW_INTEG_NAMES:
        # 当前值: name 必在 _SW_CURR_DERIVED, 取 curr[name]
        vals = [curr[name]]
        for m in range(5, 56, 5):
            vals.append(lag_cache[name][m])
        avg = np.mean(np.stack(vals, axis=0), axis=0).astype(np.float32)
        feats.append(avg)

    # e) 1 dPdyn_dt
    if 'P_dyn_lag_5' in df_K.columns:
        dP = ((df_K['P_dyn'].values - df_K['P_dyn_lag_5'].values) / 5.0).astype(np.float32)
    else:
        dP = np.zeros(K, dtype=np.float32)
    feats.append(dP)

    X_sw = np.stack(feats, axis=1).astype(np.float32)        # (K, 116)
    if X_sw.shape[1] != 116:
        raise ValueError(f"X_sw 维度 {X_sw.shape[1]} != 116, 列顺序错误?")
    return X_sw


def _compute_X_skip_per_grid(utc_K, mlat_1d, mlt_1d):
    """
    返回 X_skip ndarray (K, n_pts, 9) float32.
    列顺序: mlat_scaled, sin_mlt, cos_mlt, dipole_tilt, cos_sza,
             sin_doy, cos_doy, sin_hour, cos_hour
    """
    K = len(utc_K)
    n_mlat, n_mlt = len(mlat_1d), len(mlt_1d)
    n_pts = n_mlat * n_mlt
    MLT_g, MLAT_g = np.meshgrid(mlt_1d, mlat_1d)
    mlat_grid = MLAT_g.flatten().astype(np.float32)         # (n_pts,)
    mlt_grid = (MLT_g.flatten() % 24.0).astype(np.float32)

    # 与 K 无关 (n_pts,)
    mlat_scaled = (mlat_grid - 50.0) / 40.0
    sin_mlt = np.sin(mlt_grid * np.pi / 12.0).astype(np.float32)
    cos_mlt = np.cos(mlt_grid * np.pi / 12.0).astype(np.float32)
    mlat_rad = np.radians(mlat_grid)
    sin_mlat_g = np.sin(mlat_rad).astype(np.float32)        # (n_pts,)
    cos_mlat_g = np.cos(mlat_rad).astype(np.float32)
    mlt_angle = (2.0 * np.pi * (mlt_grid - 12.0) / 24.0).astype(np.float32)
    cos_mlt_angle = np.cos(mlt_angle).astype(np.float32)

    # 与 K 相关 (K,)
    utc_K_dt = pd.to_datetime(utc_K)
    DOY = utc_K_dt.dayofyear.values.astype(np.float32)
    DOY_phase = 2.0 * np.pi * (DOY - 172.0) / 365.25
    solar_decl_rad = (np.radians(23.44) * np.cos(DOY_phase)).astype(np.float32)
    sin_decl = np.sin(solar_decl_rad).astype(np.float32)    # (K,)
    cos_decl = np.cos(solar_decl_rad).astype(np.float32)
    dipole_tilt_K = compute_dipole_tilt_rad(utc_K_dt).astype(np.float32)  # (K,)
    sin_doy_K = np.sin(2.0 * np.pi * DOY / 365.25).astype(np.float32)
    cos_doy_K = np.cos(2.0 * np.pi * DOY / 365.25).astype(np.float32)
    hour_K = (utc_K_dt.hour.values.astype(np.float32)
              + utc_K_dt.minute.values.astype(np.float32) / 60.0)
    sin_hour_K = np.sin(2.0 * np.pi * hour_K / 24.0).astype(np.float32)
    cos_hour_K = np.cos(2.0 * np.pi * hour_K / 24.0).astype(np.float32)

    # cos_sza: (K, n_pts)
    cos_sza = (sin_mlat_g[None, :] * sin_decl[:, None]
               + cos_mlat_g[None, :] * cos_decl[:, None] * cos_mlt_angle[None, :])
    cos_sza = cos_sza.astype(np.float32)

    # 拼成 (K, n_pts, 9)
    X_skip = np.empty((K, n_pts, 9), dtype=np.float32)
    X_skip[:, :, 0] = mlat_scaled[None, :]
    X_skip[:, :, 1] = sin_mlt[None, :]
    X_skip[:, :, 2] = cos_mlt[None, :]
    X_skip[:, :, 3] = dipole_tilt_K[:, None]
    X_skip[:, :, 4] = cos_sza
    X_skip[:, :, 5] = sin_doy_K[:, None]
    X_skip[:, :, 6] = cos_doy_K[:, None]
    X_skip[:, :, 7] = sin_hour_K[:, None]
    X_skip[:, :, 8] = cos_hour_K[:, None]
    return X_skip


def predict_grid_v4_fast(model, sw_rows, mlat_1d, mlt_1d,
                         scaler_path=None, device='npu:0',
                         batch_size=131072):
    """
    高效多时刻网格推理:
      - sw 衍生只对 K 个时刻算 (而不是 K * n_pts 次)
      - skip 特征 (mlat/mlt/dipole_tilt/cos_sza/时间编码) 直接构造
      - 一次 model.forward 完成 K * n_pts 点

    与 predict_grid_v4 / predict_grid_v4_multi 输出严格一致 (验证 max abs diff < 1e-4).

    Args:
        sw_rows:    DataFrame, K 行, 含 utc, Bx..P_dyn, 所有 _lag_5..120 列
        mlat_1d, mlt_1d: 共用网格
        scaler_path: V4 scaler pkl

    Returns:
        flux_4heads: shape (K, 4, n_mlat, n_mlt) 线性 flux
    """
    if not isinstance(sw_rows, pd.DataFrame):
        sw_rows = pd.DataFrame(list(sw_rows))
    K = len(sw_rows)
    if K == 0:
        raise ValueError("sw_rows 为空")

    n_mlat, n_mlt = len(mlat_1d), len(mlt_1d)
    n_pts = n_mlat * n_mlt

    # 1) X_sw per time (K, 116) -> broadcast (K, n_pts, 116)
    X_sw_per_time = _compute_X_sw_per_time(sw_rows)         # (K, 116)
    scaler = _get_scaler(scaler_path)
    X_sw_scaled_per_time = scaler.transform(X_sw_per_time).astype(np.float32)
    # broadcast 到 (K, n_pts, 116) 再 flatten
    X_sw_flat = np.broadcast_to(
        X_sw_scaled_per_time[:, None, :], (K, n_pts, 116)
    ).reshape(K * n_pts, 116).copy()

    # 2) X_skip per grid (K, n_pts, 9) -> flatten
    X_skip = _compute_X_skip_per_grid(
        sw_rows['utc'].values, mlat_1d, mlt_1d)              # (K, n_pts, 9)
    X_skip_flat = X_skip.reshape(K * n_pts, 9)

    # 3) Tensor + batched forward
    X_sw_t = torch.from_numpy(X_sw_flat)
    X_skip_t = torch.from_numpy(X_skip_flat)

    n_total = X_sw_t.shape[0]
    out = np.zeros((n_total, 4), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for i in range(0, n_total, batch_size):
            sw_b = X_sw_t[i:i + batch_size].to(device)
            sk_b = X_skip_t[i:i + batch_size].to(device)
            log_pred = model(sw_b, sk_b).cpu().numpy()
            lin = (10.0 ** log_pred) - 1e-6
            lin[lin < 0] = 0.0
            out[i:i + len(lin)] = lin

    # 4) reshape -> (K, 4, n_mlat, n_mlt)
    return (out
            .reshape(K, n_pts, 4)
            .transpose(0, 2, 1)
            .reshape(K, 4, n_mlat, n_mlt))


# ===================================================================
# 6) 工具: 测试集真值按 head 拆分
# ===================================================================
def split_true_by_head(df):
    """
    把 df 中的 ele_energy_flux + ion_energy_flux + aurora_type
    拆成 4 个 head 各自的真值数组 (线性 flux).

    Returns:
        list[ndarray]  长度 4, 顺序 [Diff, Mono, BB, Ion]
    """
    ele = df['ele_energy_flux'].fillna(0.0).values
    ion = df['ion_energy_flux'].fillna(0.0).values
    at = df['aurora_type'].fillna(0).astype(int).values
    return [
        np.where(at == 1, ele, 0.0),    # Diff
        np.where(at == 2, ele, 0.0),    # Mono
        np.where(at == 3, ele, 0.0),    # BB
        ion,                            # Ion
    ]


HEAD_NAMES = ['Diffuse', 'Monoenergetic', 'Broadband', 'Ion']
HEAD_KEYS_OV = ['diff', 'mono', 'wave', 'ions']    # OVATION head 名称


def area_2d_cm2(mlat_1d, mlt_1d):
    R_E = 6.3712e8
    d_mlat = np.deg2rad(mlat_1d[1] - mlat_1d[0])
    d_mlt = (mlt_1d[1] - mlt_1d[0]) * (2.0 * np.pi / 24.0)
    mlat_rad = np.deg2rad(mlat_1d)
    area_1d = (R_E ** 2) * np.cos(mlat_rad) * d_mlat * d_mlt
    return np.tile(area_1d.reshape(-1, 1), (1, len(mlt_1d)))
