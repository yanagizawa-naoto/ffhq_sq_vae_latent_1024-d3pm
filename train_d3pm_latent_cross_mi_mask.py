#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SQ-VAE latent 用 D3PM (absorbing, MI スケジュール)
- 入力: VQ indices [0..K] （K が吸収 [MASK]）
- 出力: x0 logits [B, K, H, W] （クラス 0..K-1 のみ）
- forward: MI absorbing + mask NLL (cross スクリプトと同じロジック)
- 逆拡散: MI absorbing サンプリング（p(stay MASK|MASK) = (1 - a_{t-1}) / (1 - a_t)）

Backbone:
- CNN ベース U-Net + 一部スケールでの Self-Attention（Transformer 的表現）
- 各ブロックの頭で GroupNorm
- 最後の conv 前にも GroupNorm
- 勾配蓄積 / AMP (FP16/bfloat16 対応は単純版)
"""

import os
import math
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.utils as vutils
from tqdm.auto import tqdm

from torch.cuda.amp import autocast, GradScaler

# ★ あなたの SQ-VAE 実装に合わせて import を調整
from ffhq_sq_vae import SQVAEModel


# ===============================
# 1. Latent Dataset (npz)
# ===============================
class LatentDataset(Dataset):
    def __init__(self, latent_root: str):
        self.files = []
        for r, _, fs in os.walk(latent_root):
            for f in fs:
                if f.lower().endswith(".npz"):
                    self.files.append(os.path.join(r, f))
        if not self.files:
            raise RuntimeError(f"No .npz files found under {latent_root}")

        sample = np.load(self.files[0])
        idx = sample["indices"]
        self.H, self.W = idx.shape

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        data = np.load(path)
        indices = torch.from_numpy(data["indices"].astype("int64"))  # [H, W]
        return indices, path


# ===============================
# 2. Timestep embedding
# ===============================
def timestep_embedding(timesteps, dim, max_period=10000):
    """
    timesteps: [B]
    return: [B, dim]
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half
    ).to(timesteps.device)
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)  # [B, half]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def make_groupnorm(num_channels: int) -> nn.GroupNorm:
    """
    だいたい「チャンネル/8 グループ」を狙う簡単ヘルパ。
    割り切れなければ 1 グループ (LayerNorm 的) にフォールバック。
    """
    groups = max(1, num_channels // 8)
    if num_channels % groups != 0:
        groups = 1
    return nn.GroupNorm(groups, num_channels)


# ===============================
# 3. CNN ベース time-conditioned Block
# ===============================
class ResBlockTime(nn.Module):
    """
    [B, C, H, W] に対する ResBlock。
    t_proj: [B, C] を FiLM っぽく加算。
    """

    def __init__(self, channels, dropout=0.0):
        super().__init__()
        self.in_norm1 = make_groupnorm(channels)
        self.in_norm2 = make_groupnorm(channels)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x, t_proj):
        """
        x: [B, C, H, W]
        t_proj: [B, C]
        """
        B, C, H, W = x.shape
        # 時刻埋め込みを空間全体にブロードキャスト
        t_map = t_proj.view(B, C, 1, 1)

        h = self.in_norm1(x)
        h = self.act(h + t_map)   # 1 層目の前で t を加算
        h = self.conv1(h)

        h = self.in_norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return x + h  # residual


class CNNSpatialBlock(nn.Module):
    """
    ResBlockTime を depth 回適用するだけのブロック。
    """

    def __init__(self, channels, depth, dropout=0.0):
        super().__init__()
        self.blocks = nn.ModuleList(
            [ResBlockTime(channels, dropout=dropout) for _ in range(depth)]
        )

    def forward(self, x, t_proj):
        for blk in self.blocks:
            x = blk(x, t_proj)
        return x


# ===============================
# 3.5 Self-Attention Block
# ===============================
class SpatialSelfAttention2d(nn.Module):
    """
    [B, C, H, W] に対する Multi-Head Self-Attention。
    DDPM 系 2D Self-Attn 相当。時間埋め込みは畳み込み側で注入済みとみなし、ここでは純粋な空間 Self-Attn。
    """

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        assert channels % num_heads == 0, \
            f"channels ({channels}) must be divisible by num_heads ({num_heads})"
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm = make_groupnorm(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=True)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]
        """
        B, C, H, W = x.shape
        h = self.norm(x)

        qkv = self.qkv(h)  # [B, 3C, H, W]
        q, k, v = qkv.chunk(3, dim=1)  # 各 [B, C, H, W]

        # [B, C, H, W] -> [B, num_heads, N, head_dim]
        def reshape_heads(t):
            t = t.view(B, self.num_heads, self.head_dim, H * W)
            t = t.permute(0, 1, 3, 2)  # [B, num_heads, N, head_dim]
            return t

        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)

        # Attention weights: [B, num_heads, N, N]
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # Output: [B, num_heads, N, head_dim]
        out = torch.matmul(attn, v)

        # [B, num_heads, N, head_dim] -> [B, C, H, W]
        out = out.permute(0, 1, 3, 2).contiguous()  # [B, num_heads, head_dim, N]
        out = out.view(B, C, H, W)

        out = self.proj_out(out)
        return x + out  # residual


# ===============================
# 4. MultiScale CNN U-Net (+ Self-Attention)
# ===============================
class MultiScaleCNNUNet(nn.Module):
    """
    D3PM の x0_model 用 CNN+Attention バックボーン。
    入力: x_t (カテゴリ index, 0..K) と t
    出力: x0 の logits, shape = [B, K, H, W]  (クラス 0..K-1)
    """

    def __init__(
        self,
        vocab_size,      # = K+1 (0..K-1 + [MASK])
        data_classes,    # = K   (生成対象クラス数)
        base_channels=256,
        time_emb_dim=128,
        n_heads=8,              # Self-Attention 用のヘッド数
        depth=2,
        window_size=4,          # 互換のためのダミー（使わない）
        mlp_ratio=4.0,          # 使わない
        dropout=0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.data_classes = data_classes
        ch = base_channels
        self.n_heads = n_heads

        # token embedding
        self.token_emb = nn.Embedding(vocab_size, ch)

        # time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )
        self.time_proj0 = nn.Linear(time_emb_dim, ch)
        self.time_proj1 = nn.Linear(time_emb_dim, ch * 2)
        self.time_proj2 = nn.Linear(time_emb_dim, ch * 4)
        self.time_proj3 = nn.Linear(time_emb_dim, ch * 4)
        self.time_proj_mid = nn.Linear(time_emb_dim, ch * 4)

        # Down path
        self.down0 = CNNSpatialBlock(
            channels=ch,
            depth=depth,
            dropout=dropout,
        )

        self.downsample1 = nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1)
        self.down1 = CNNSpatialBlock(
            channels=ch * 2,
            depth=depth,
            dropout=dropout,
        )

        self.downsample2 = nn.Conv2d(ch * 2, ch * 4, 3, stride=2, padding=1)
        self.down2 = CNNSpatialBlock(
            channels=ch * 4,
            depth=depth,
            dropout=dropout,
        )
        # Self-Attention at H/4
        self.attn2 = SpatialSelfAttention2d(ch * 4, num_heads=n_heads)

        self.downsample3 = nn.Conv2d(ch * 4, ch * 4, 3, stride=2, padding=1)
        self.down3 = CNNSpatialBlock(
            channels=ch * 4,
            depth=depth,
            dropout=dropout,
        )
        # Self-Attention at H/8
        self.attn3 = SpatialSelfAttention2d(ch * 4, num_heads=n_heads)

        self.downsample4 = nn.Conv2d(ch * 4, ch * 4, 3, stride=2, padding=1)

        # Mid
        self.mid = CNNSpatialBlock(
            channels=ch * 4,
            depth=depth,
            dropout=dropout,
        )
        # Self-Attention at bottleneck H/16
        self.attn_mid = SpatialSelfAttention2d(ch * 4, num_heads=n_heads)

        # Up path
        self.upsample4 = nn.ConvTranspose2d(ch * 4, ch * 4, 4, stride=2, padding=1)
        self.up4_combine = nn.Conv2d(ch * 4 + ch * 4, ch * 4, 1)
        self.up4 = CNNSpatialBlock(
            channels=ch * 4,
            depth=depth,
            dropout=dropout,
        )
        # Attention on up4 (H/8)
        self.attn_up4 = SpatialSelfAttention2d(ch * 4, num_heads=n_heads)

        self.upsample3 = nn.ConvTranspose2d(ch * 4, ch * 4, 4, stride=2, padding=1)
        self.up3_combine = nn.Conv2d(ch * 4 + ch * 4, ch * 4, 1)
        self.up3 = CNNSpatialBlock(
            channels=ch * 4,
            depth=depth,
            dropout=dropout,
        )
        # Attention on up3 (H/4)
        self.attn_up3 = SpatialSelfAttention2d(ch * 4, num_heads=n_heads)

        self.upsample2 = nn.ConvTranspose2d(ch * 4, ch * 2, 4, stride=2, padding=1)
        self.up2_combine = nn.Conv2d(ch * 2 + ch * 2, ch * 2, 1)
        self.up2 = CNNSpatialBlock(
            channels=ch * 2,
            depth=depth,
            dropout=dropout,
        )

        self.upsample1 = nn.ConvTranspose2d(ch * 2, ch, 4, stride=2, padding=1)
        self.up1_combine = nn.Conv2d(ch + ch, ch, 1)
        self.up1 = CNNSpatialBlock(
            channels=ch,
            depth=depth,
            dropout=dropout,
        )

        self.out_norm = make_groupnorm(ch)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(ch, data_classes, 3, padding=1)

    def _match_shape(self, x, ref):
        if x.shape[-2:] == ref.shape[-2:]:
            return x, ref
        Hmin = min(x.shape[-2], ref.shape[-2])
        Wmin = min(x.shape[-1], ref.shape[-1])
        x = x[..., :Hmin, :Wmin]
        ref = ref[..., :Hmin, :Wmin]
        return x, ref

    def forward(self, x, t):
        """
        x: [B, H, W] (long, 0..vocab_size-1; K が [MASK])
        t: [B] or [B]-float
        return: logits [B, K, H, W] （クラス 0..K-1）
        """
        B, H, W = x.shape

        # token embedding
        h = self.token_emb(x)         # [B,H,W,C]
        h = h.permute(0, 3, 1, 2)     # [B,C,H,W]

        # time embedding
        if not torch.is_floating_point(t):
            t = t.float()
        t_emb_in = timestep_embedding(t, self.time_mlp[0].in_features)
        t_emb = self.time_mlp(t_emb_in)

        t0 = self.time_proj0(t_emb)
        t1 = self.time_proj1(t_emb)
        t2 = self.time_proj2(t_emb)
        t3 = self.time_proj3(t_emb)
        t_mid = self.time_proj_mid(t_emb)

        # Down
        h0 = self.down0(h, t0)              # [B,ch,H,W]

        h1 = self.downsample1(h0)           # [B,2ch,H/2,W/2]
        h1 = self.down1(h1, t1)

        h2 = self.downsample2(h1)           # [B,4ch,H/4,W/4]
        h2 = self.down2(h2, t2)
        h2 = self.attn2(h2)                 # Self-Attn at H/4

        h3 = self.downsample3(h2)           # [B,4ch,H/8,W/8]
        h3 = self.down3(h3, t3)
        h3 = self.attn3(h3)                 # Self-Attn at H/8

        h4 = self.downsample4(h3)           # [B,4ch,H/16,W/16]

        # Mid
        h_mid = self.mid(h4, t_mid)         # [B,4ch,H/16,W/16]
        h_mid = self.attn_mid(h_mid)        # Self-Attn at bottleneck

        # Up 4 -> 3
        h_up = self.upsample4(h_mid)        # [B,4ch,~H/8,~W/8]
        h_up, h3m = self._match_shape(h_up, h3)
        h_up = self.up4_combine(torch.cat([h_up, h3m], dim=1))
        h_up = self.up4(h_up, t3)
        h_up = self.attn_up4(h_up)          # Self-Attn at H/8 (decoder)

        # 3 -> 2
        h_up = self.upsample3(h_up)         # [B,4ch,~H/4,~W/4]
        h_up, h2m = self._match_shape(h_up, h2)
        h_up = self.up3_combine(torch.cat([h_up, h2m], dim=1))
        h_up = self.up3(h_up, t2)
        h_up = self.attn_up3(h_up)          # Self-Attn at H/4 (decoder)

        # 2 -> 1
        h_up = self.upsample2(h_up)         # [B,2ch,~H/2,~W/2]
        h_up, h1m = self._match_shape(h_up, h1)
        h_up = self.up2_combine(torch.cat([h_up, h1m], dim=1))
        h_up = self.up2(h_up, t1)

        # 1 -> 0
        h_up = self.upsample1(h_up)         # [B,ch,~H,~W]
        h_up, h0m = self._match_shape(h_up, h0)
        h_up = self.up1_combine(torch.cat([h_up, h0m], dim=1))
        h_up = self.up1(h_up, t0)

        h_out = self.out_norm(h_up)
        h_out = self.out_act(h_out)
        logits = self.out_conv(h_out)       # [B,K,H,W]
        return logits


# ===============================
# 5. D3PM (absorbing, MI マスク + mask NLL, cross 方式)
# ===============================
class D3PMAbsorbingLatentIO(nn.Module):
    """
    cross スクリプト D3PMAbsorbingCrossIO と同じロジックを、
    - コンテキストなし版
    - SQ-VAE latent の x_t を直接食う版
    として実装。

    model: x_t, t -> logits[x0] (B,K,H,W)  ※ K = num_data_classes
    """

    def __init__(
        self,
        model: nn.Module,
        num_data_classes: int,
        absorbing_index: int,
        timesteps: int,
        lambda_ce: float = 1e-4,
    ):
        super().__init__()
        self.model = model
        self.K = num_data_classes
        self.abs_idx = absorbing_index
        self.T = timesteps
        self.lambda_ce = lambda_ce

        # MI absorbing schedule (cross と同じ)
        betas = torch.zeros(timesteps + 1, dtype=torch.float32)
        for t in range(1, timesteps + 1):
            betas[t] = 1.0 / (timesteps - t + 1.0)
        alphas = 1.0 - betas
        abar = torch.ones_like(betas)
        for t in range(1, timesteps + 1):
            abar[t] = abar[t - 1] * alphas[t]

        self.register_buffer("betas", betas)    # (T+1,)
        self.register_buffer("alphas", alphas)  # (T+1,)
        self.register_buffer("abar", abar)      # (T+1,)

    # q(x_t | x_0): keep with \barα_t, else -> [MASK]
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x0: (B,H,W) in 0..K-1
        t : (B,)
        return: x_t (B,H,W) in 0..K or [MASK]=abs_idx
        """
        B, H, W = x0.shape
        abar_t = self.abar[t].reshape(B, 1, 1)              # (B,1,1)
        keep = torch.rand(B, H, W, device=x0.device) < abar_t
        xt = torch.full_like(x0, fill_value=self.abs_idx)
        xt[keep] = x0[keep]
        return xt

    def p_losses(self, x0: torch.Tensor, t: torch.Tensor, return_info: bool = False):
        """
        学習損失:
          main_term = - E_{xt|x0} [ log pθ(x0 | xt) ] （xt==MASK 部分だけ）
          ce_all    = x0 全画素 CE
          L = main_term + λ * ce_all
        """
        xt = self.q_sample(x0, t)                     # (B,H,W)
        logits = self.model(xt, t)                    # (B,K,H,W)
        log_probs = F.log_softmax(logits, dim=1)      # (B,K,H,W)

        # 主項: xt == MASK のところだけ NLL
        mask_abs = (xt == self.abs_idx)
        if mask_abs.any():
            lp = log_probs.gather(1, x0.unsqueeze(1)).squeeze(1)    # (B,H,W)
            main_term = -(lp[mask_abs]).mean()
        else:
            main_term = logits.new_tensor(0.0)

        # 補助 CE: 全画素
        ce_all = F.nll_loss(
            log_probs.permute(0, 2, 3, 1).reshape(-1, self.K),
            x0.reshape(-1),
            reduction="mean",
        )

        loss = main_term + self.lambda_ce * ce_all
        if return_info:
            info = {
                "main_loss": float(main_term.detach()),
                "ce_loss": float(ce_all.detach()),
            }
            return loss, info
        return loss

    def forward(self, x0: torch.Tensor):
        """
        x0: (B,H,W) 0..K-1
        return: (loss, info_dict)
        """
        B = x0.size(0)
        t = torch.randint(1, self.T + 1, (B,), device=x0.device, dtype=torch.long)
        return self.p_losses(x0, t, return_info=True)

    @torch.no_grad()
    def sample(self, batch: int, height: int, width: int, device: torch.device,
               steps: int = None) -> torch.Tensor:
        """
        x_T は全 [MASK] から開始。最終的に 0..K-1 を返す。
        cross の D3PMAbsorbingCrossIO.sample と同じロジック（コンテキスト抜き）。
        """
        T = self.T if steps is None else steps
        xt = torch.full(
            (batch, height, width),
            self.abs_idx,
            device=device,
            dtype=torch.long,
        )

        for ti in reversed(range(1, T + 1)):
            tb = torch.full((batch,), ti, device=device, dtype=torch.long)
            logits = self.model(xt, tb)                 # (B,K,H,W)
            probs_e = torch.softmax(logits, dim=1)      # (B,K,H,W)

            # MI吸収：p(stay MASK|MASK)= (t-1)/t = (1-abar_{t-1}) / (1-abar_t)
            abar_t = float(self.abar[ti].item())
            abar_prev = float(self.abar[ti - 1].item())
            p_stay_abs = (1.0 - abar_prev) / max(1e-8, (1.0 - abar_t))

            mask_abs = (xt == self.abs_idx)
            if mask_abs.any():
                b, h, w = mask_abs.nonzero(as_tuple=True)
                pe = probs_e[b, :, h, w]                           # (M, K)
                c = torch.multinomial(pe, num_samples=1).squeeze(1)  # (M,)
                go = torch.rand_like(c.float()) < (1.0 - p_stay_abs)
                xt[b[go], h[go], w[go]] = c[go]

        return xt  # (B,H,W) in 0..K-1


# ===============================
# 6. Args
# ===============================
def parse_args():
    p = argparse.ArgumentParser(
        description="D3PM (absorbing, MI schedule, CNN+Attention U-Net backbone) on VQ latents"
    )

    p.add_argument("--latent_root", type=str, default="latent_1024_npz")
    p.add_argument(
        "--sqvae_ckpt",
        type=str,
        default="/home/5/us05085/vae_train/sqvae_results_ffhq_1024/checkpoints/sqvae_latest_backup.pth",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="/home/5/us05085/vae_train/d3pm_latent_cnn_unet_mi",
    )

    p.add_argument("--resume", type=str, default=None)

    p.add_argument("--epochs", type=int, default=50000)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--num_embeddings", type=int, default=1024)
    p.add_argument("--embedding_dim", type=int, default=4)

    p.add_argument("--base_channels", type=int, default=128)
    p.add_argument("--num_heads", type=int, default=32)  # Self-Attention heads
    p.add_argument("--transformer_depth", type=int, default=4)  # CNN ブロック depth として使用
    p.add_argument("--window_size", type=int, default=4)  # ダミー（未使用）

    # cross-style の λ_CE
    p.add_argument("--lambda_ce", type=float, default=1e-4)

    # 勾配蓄積
    p.add_argument("--grad_accum_steps", type=int, default=16)

    # サンプル生成
    p.add_argument("--num_sample_images", type=int, default=16)
    p.add_argument("--sample_dir", type=str, default=None)

    # AMP
    p.add_argument("--no_amp", action="store_true", help="disable mixed precision")

    return p.parse_args()


# ===============================
# 7. メイン
# ===============================
def main():
    args = parse_args()

    # ==== Device 決定 & ログ ====
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[Device] CUDA not available, using CPU")

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    sample_dir = args.sample_dir or os.path.join(args.output_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)

    use_amp = (device.type == "cuda") and (not args.no_amp)
    scaler = GradScaler(enabled=use_amp)

    # Dataset
    dataset = LatentDataset(args.latent_root)
    H, W = dataset.H, dataset.W
    print(f"[Dataset] {len(dataset)} latents, shape=({H},{W})")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # SQ-VAE (decoder)
    sqvae = SQVAEModel(
        num_embeddings=args.num_embeddings,
        embedding_dim=args.embedding_dim,
    ).to(device)
    print(f"[SQVAE] load {args.sqvae_ckpt}")
    sq_ckpt = torch.load(args.sqvae_ckpt, map_location=device)
    sqvae.load_state_dict(sq_ckpt["model_state_dict"])
    sqvae.eval()
    for p in sqvae.parameters():
        p.requires_grad_(False)

    # D3PM + CNN+Attention U-Net (cross MI absorbing)
    K = args.num_embeddings          # data classes 0..K-1
    vocab_size = K + 1               # + [MASK]
    mask_id = K                      # absorbing index

    x0_model = MultiScaleCNNUNet(
        vocab_size=vocab_size,
        data_classes=K,
        base_channels=args.base_channels,
        time_emb_dim=128,
        n_heads=args.num_heads,
        depth=args.transformer_depth,
        window_size=args.window_size,
        mlp_ratio=4.0,
        dropout=0.0,
    ).to(device)

    diff = D3PMAbsorbingLatentIO(
        model=x0_model,
        num_data_classes=K,
        absorbing_index=mask_id,
        timesteps=args.timesteps,
        lambda_ce=args.lambda_ce,
    ).to(device)

    # ★ モデルのデバイス確認
    print(f"[Check] diff first param device: {next(diff.parameters()).device}")

    optim = torch.optim.AdamW(diff.parameters(), lr=args.lr)

    start_epoch = 0
    global_step = 0

    # ===== Resume =====
    if args.resume is not None and os.path.isfile(args.resume):
        print(f"[Resume] loading checkpoint from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        diff.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optim.load_state_dict(ckpt["optimizer_state_dict"])
        if "scaler_state_dict" in ckpt:
            try:
                scaler.load_state_dict(ckpt["scaler_state_dict"])
            except Exception:
                print("[Resume] scaler_state_dict load failed, ignore.")
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("global_step", 0)
        print(f"[Resume] start from epoch {start_epoch}, global_step {global_step}")
    elif args.resume is not None:
        print(f"[Resume] ckpt not found: {args.resume} (start from scratch)")

    # ===== Train loop =====
    for epoch in range(start_epoch, args.epochs):
        diff.train()
        running_loss = 0.0
        running_main = 0.0
        running_ce = 0.0
        steps = 0

        print(f"\n=== Epoch {epoch+1}/{args.epochs} ===")
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}", leave=False)

        optim.zero_grad(set_to_none=True)
        accum = 0

        for it, (x0, _) in enumerate(pbar):
            x0 = x0.to(device, non_blocking=True)  # [B,H,W], 0..K-1

            # ★ 念のため GPU チェック
            if device.type == "cuda":
                assert x0.is_cuda, "[Bug] x0 is not on CUDA"
                assert next(diff.parameters()).is_cuda, "[Bug] model is not on CUDA"

            with autocast(enabled=use_amp):
                loss, info = diff(x0)
                loss = loss / max(1, args.grad_accum_steps)

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accum += 1
            global_step += 1

            # 統計は unscaled の loss で
            running_loss += float(loss.item()) * max(1, args.grad_accum_steps)
            running_main += info["main_loss"]
            running_ce += info["ce_loss"]
            steps += 1

            # optimizer step (勾配蓄積)
            do_step = (accum % args.grad_accum_steps == 0)

            grad_norm = 0.0
            if do_step:
                if use_amp:
                    scaler.unscale_(optim)
                    grad_norm = float(
                        torch.nn.utils.clip_grad_norm_(diff.parameters(), 1.0)
                    )
                    scaler.step(optim)
                    scaler.update()
                else:
                    grad_norm = float(
                        torch.nn.utils.clip_grad_norm_(diff.parameters(), 1.0)
                    )
                    optim.step()
                optim.zero_grad(set_to_none=True)
                accum = 0

            mean_loss = running_loss / steps
            mean_main = running_main / steps
            mean_ce = running_ce / steps

            pbar.set_postfix(
                loss=f"{mean_loss:.4f}",
                main=f"{mean_main:.4f}",
                ce=f"{mean_ce:.4f}",
                grad=f"{grad_norm:.3f}",
            )

        # ===== エポック単位の平均ロスを表示 =====
        epoch_loss = running_loss / steps
        epoch_main = running_main / steps
        epoch_ce = running_ce / steps
        print(
            f"[Epoch {epoch+1}] avg_loss={epoch_loss:.4f}, "
            f"avg_main={epoch_main:.4f}, avg_ce={epoch_ce:.4f}"
        )

        # ===== checkpoint =====
        ckpt_path = os.path.join(ckpt_dir, f"d3pm_latent_epoch_{epoch+1:04d}.pth")
        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": diff.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "num_embeddings": args.num_embeddings,
                "timesteps": args.timesteps,
                "base_channels": args.base_channels,
                "num_heads": args.num_heads,
                "transformer_depth": args.transformer_depth,
                "window_size": args.window_size,
                "lambda_ce": args.lambda_ce,
            },
            ckpt_path,
        )
        print(f"[Checkpoint] saved to {ckpt_path}")

        # ===== サンプル生成 =====
        diff.eval()
        with torch.no_grad():
            num_samp = min(args.num_sample_images, 16)
            x_codes = diff.sample(
                batch=num_samp,
                height=H,
                width=W,
                device=device,          # torch.device をそのまま渡す
                steps=args.timesteps,
            )  # [B,H,W], 0..K-1

            # SQ-VAE decoder で画像デコード
            idx_flat = x_codes.view(-1)
            zq_flat = sqvae.vq.embedding(idx_flat)  # [B*H*W, D]
            D_emb = zq_flat.shape[-1]
            zq = zq_flat.view(num_samp, H, W, D_emb).permute(0, 3, 1, 2).contiguous()
            recon = sqvae.decoder(zq)               # [-1,1]
            recon_vis = (recon * 0.5 + 0.5).clamp(0.0, 1.0)

            out_path = os.path.join(sample_dir, f"epoch_{epoch+1:04d}_samples.png")
            vutils.save_image(recon_vis, out_path, nrow=int(math.sqrt(num_samp)))
            print(f"[Sample] saved decoded samples to {out_path}")


if __name__ == "__main__":
    main()
