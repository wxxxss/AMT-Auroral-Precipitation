import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class AMT(nn.Module):
    """
      1) 物理一致性: 太阳风只从一个共享编码器进入"磁层状态隐空间",
         避免 4 个独立塔对同一物理状态产生自相矛盾的内部表征.
      2) 数据传递 (MTL): Diff 占 88% 产生稳定梯度, 把共享 backbone 训成
         强太阳风特征提取器; Mono/BB 少数类可在此基础上"白嫖"高级表征.
      3) 非线性时序解开: 先升维到 hidden_wide (~1024) 做 kernel-trick 式
         特征展开, 再压到 latent_dim (~256) 作磁层状态表示.

    输入:
      x_sw:   (B, sw_dim)   太阳风原始 + 物理衍生 (~116 维)
      x_skip: (B, skip_dim) 空间/时间/磁偶极倾角/SZA (~9 维)

    输出: (B, 4) log10 flux, 末端 clamp(-6.5, 4.0)
    """

    def __init__(self,
                 sw_dim=116,
                 skip_dim=9,
                 hidden_wide=1024,
                 hidden_mid=512,
                 latent_dim=256,
                 head_hidden=128,
                 dropout=0.2,
                 out_clamp=(-6.5, 4.0)):
        super().__init__()
        self.sw_dim = sw_dim
        self.skip_dim = skip_dim
        self.latent_dim = latent_dim
        self.out_clamp = out_clamp

        # ---- 共享 backbone  ----
        self.backbone = nn.Sequential(
            nn.Linear(sw_dim, hidden_wide),
            nn.BatchNorm1d(hidden_wide),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_wide, hidden_mid),
            nn.BatchNorm1d(hidden_mid),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_mid, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.GELU(),
        )

        # ---- 4 个轻量 head ----
        def build_head():
            return nn.Sequential(
                nn.Linear(latent_dim + skip_dim, head_hidden),
                nn.BatchNorm1d(head_hidden),
                nn.GELU(),
                nn.Dropout(dropout),

                nn.Linear(head_hidden, head_hidden // 2),
                nn.BatchNorm1d(head_hidden // 2),
                nn.GELU(),

                nn.Linear(head_hidden // 2, 1),
            )

        self.head_diffuse = build_head()
        self.head_mono = build_head()
        self.head_broadband = build_head()
        self.head_ion = build_head()

    def forward(self, x_sw, x_skip):
        sw_latent = self.backbone(x_sw)                 # (B, latent_dim)
        fused = torch.cat([sw_latent, x_skip], dim=1)   # (B, latent+skip)

        out_d = self.head_diffuse(fused)
        out_m = self.head_mono(fused)
        out_b = self.head_broadband(fused)
        out_i = self.head_ion(fused)

        pred = torch.cat([out_d, out_m, out_b, out_i], dim=1)   # (B, 4)
        if self.out_clamp is not None:
            pred = torch.clamp(pred, self.out_clamp[0], self.out_clamp[1])
        return pred