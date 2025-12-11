import os
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# ★ 学習スクリプトのファイル名に合わせて修正してください
# 例: 学習コードを sqvae_train.py として保存しているならこれでOK
from ffhq_sq_vae import SQVAEModel


# ===============================
# Metrics（学習コードと同じもの）
# ===============================
def calculate_psnr_batch(img1, img2):
    """
    img1, img2: [-1,1], shape [B, C, H, W]
    画像ごとの PSNR を返す: Tensor [B]
    """
    # mse per image
    mse = ((img1.float() - img2.float()) ** 2).view(img1.size(0), -1).mean(dim=1)
    # avoid log(0)
    mse = torch.clamp(mse, min=1e-10)
    psnr = 20 * torch.log10(2.0 / torch.sqrt(mse))  # max=2.0 for [-1,1]
    return psnr


def gaussian_window(size, sigma):
    coords = torch.arange(size, dtype=torch.float)
    coords -= size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()
    return g.reshape(1, 1, 1, -1) * g.reshape(1, 1, -1, 1)


def ssim(img1, img2, window_size=11, size_average=True):
    """
    img1, img2: [-1,1], shape [B, C, H, W]
    size_average=True  -> スカラー
    size_average=False -> Tensor [B]
    """
    img1 = (img1.float() + 1) / 2
    img2 = (img2.float() + 1) / 2
    channel = img1.size(1)
    window = gaussian_window(window_size, 1.5).to(img1.device).type(img1.dtype)
    window = window.expand(channel, 1, window_size, window_size).contiguous()
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1 = 0.01**2
    C2 = 0.03**2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    if size_average:
        return ssim_map.mean()
    else:
        # 平均を (N,C,H,W) -> (N) になるように
        return ssim_map.mean(dim=[1, 2, 3])


# ===============================
# Dataset: latent(npz) + 元画像
# ===============================
class LatentWithImageDataset(Dataset):
    """
    latent_root 以下の .npz を全部読み、
    - indices: [H, W] (uint16)
    - image_path: 元画像パス
    を使って、(indices_tensor, image_tensor, path) を返す
    """

    def __init__(self, latent_root, image_size=512):
        self.latent_root = latent_root
        self.files = []
        for r, _, fs in os.walk(latent_root):
            for f in fs:
                if f.lower().endswith(".npz"):
                    self.files.append(os.path.join(r, f))
        if len(self.files) == 0:
            raise RuntimeError(f"No .npz files found under {latent_root}")

        self.transform = T.Compose(
            [
                T.Resize(image_size, interpolation=Image.BICUBIC),
                T.CenterCrop(image_size),
                T.ToTensor(),          # [0,1]
                T.Normalize(0.5, 0.5), # [-1,1]
            ]
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        latent_path = self.files[idx]
        data = np.load(latent_path)
        # indices: [H, W], uint16
        indices = data["indices"].astype("int64")  # Long に変換
        indices = torch.from_numpy(indices)        # [H, W]

        image_path = data["image_path"].item() if hasattr(data["image_path"], "item") else data["image_path"]
        image_path = str(image_path)

        img = Image.open(image_path).convert("RGB")
        img = self.transform(img)  # [3, H, W], [-1,1]

        return indices, img, image_path


# ===============================
# メイン処理
# ===============================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate SQ-VAE reconstruction quality from saved latents"
    )
    parser.add_argument(
        "--latent_root",
        type=str,
        required=True,
        help="Directory containing latent .npz files.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to trained SQ-VAE checkpoint (.pth).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=512,
    )

    # ★ 学習時と同じ num_embeddings / embedding_dim を指定してください
    parser.add_argument("--num_embeddings", type=int, default=1024)
    parser.add_argument("--embedding_dim", type=int, default=4)

    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset & DataLoader
    dataset = LatentWithImageDataset(args.latent_root, image_size=args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"[Info] Found {len(dataset)} latent files under {args.latent_root}")

    # モデル構築 & ckpt ロード
    model = SQVAEModel(
        num_embeddings=args.num_embeddings,
        embedding_dim=args.embedding_dim,
    ).to(device)

    print(f"[Info] Loading checkpoint from {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    total_images = 0
    sum_mse = 0.0
    sum_psnr = 0.0
    sum_ssim = 0.0

    with torch.no_grad():
        for step, (indices, imgs, paths) in enumerate(loader):
            # indices: [B, H, W], imgs: [B, 3, H, W]
            indices = indices.to(device, non_blocking=True).long()
            imgs = imgs.to(device, non_blocking=True)

            B, H, W = indices.shape

            # ===== latent -> z_q -> recon =====
            # indices_flat: [B*H*W]
            indices_flat = indices.view(-1)
            # embedding lookup: [B*H*W, D]
            z_q_flat = model.vq.embedding(indices_flat)
            D = z_q_flat.shape[-1]
            # reshape: [B, H, W, D] -> [B, D, H, W]
            z_q = z_q_flat.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()
            # decoder
            recons = model.decoder(z_q)  # [-1,1], [B, 3, H, W]

            # ===== metrics =====
            # per-image MSE
            mse_per = ((recons - imgs) ** 2).view(B, -1).mean(dim=1)  # [B]
            psnr_per = calculate_psnr_batch(recons, imgs)             # [B]
            ssim_per = ssim(recons, imgs, size_average=False)          # [B]

            sum_mse += mse_per.sum().item()
            sum_psnr += psnr_per.sum().item()
            sum_ssim += ssim_per.sum().item()
            total_images += B

            if (step + 1) % 10 == 0:
                print(
                    f"[{step+1}/{len(loader)}] "
                    f"curr mean PSNR: {(sum_psnr/total_images):.2f} dB, "
                    f"mean SSIM: {(sum_ssim/total_images):.4f}"
                )

    avg_mse = sum_mse / total_images
    avg_psnr = sum_psnr / total_images
    avg_ssim = sum_ssim / total_images

    print("======================================")
    print(f"Total images: {total_images}")
    print(f"Mean L2 (MSE): {avg_mse:.6f}")
    print(f"Mean PSNR:     {avg_psnr:.3f} dB")
    print(f"Mean SSIM:     {avg_ssim:.4f}")
    print("======================================")


if __name__ == "__main__":
    main()
