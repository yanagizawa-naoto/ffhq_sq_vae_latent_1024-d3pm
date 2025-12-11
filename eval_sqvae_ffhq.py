import os
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# === あなたのトレーニングスクリプトからインポート ===
# ファイル名が違う場合はここを変更してください。
from ffhq_sq_vae import SQVAEModel, gaussian_window


# ===============================
# 1. PSNR / SSIM（サンプルごと）
# ===============================
def calculate_psnr_per_image(img1, img2):
    """
    img1, img2: [-1, 1], shape [B, C, H, W]
    return: PSNR [dB] for each sample, shape [B]
    """
    mse = F.mse_loss(img1.float(), img2.float(), reduction="none")  # [B,C,H,W]
    mse = mse.view(mse.size(0), -1).mean(dim=1)  # [B]
    eps = 1e-10
    psnr = 20 * torch.log10(2.0 / torch.sqrt(mse + eps))  # max=2.0 for [-1,1]
    return psnr


def ssim_per_image(img1, img2, window_size=11):
    """
    img1, img2: [-1, 1], shape [B, C, H, W]
    return: SSIM for each sample, shape [B]
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

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2) /
                ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)))

    ssim_vals = ssim_map.view(ssim_map.size(0), -1).mean(dim=1)  # [B]
    return ssim_vals


# ===============================
# 2. 連続値の「最頻値」近似
# ===============================
def approximate_mode(values, num_bins=100):
    values = np.asarray(values)
    hist, bin_edges = np.histogram(values, bins=num_bins)
    max_bin = hist.argmax()
    mode_val = 0.5 * (bin_edges[max_bin] + bin_edges[max_bin + 1])
    return mode_val


# ===============================
# 3. 壊れ画像をスキップする Dataset
# ===============================
class ImageFolderAllRecursive(Dataset):
    """
    data_root 以下の全画像を再帰的に読み込み、壊れている画像は事前にスキップする Dataset
    - 対応拡張子: .jpg, .jpeg, .png, .webp
    - 読み込んだ画像は Resize(image_size) + CenterCrop(image_size)
      + ToTensor + [-1,1] 正規化
    """

    def __init__(self, root, image_size=512):
        super().__init__()
        self.root = root
        self.image_size = image_size

        self.exts = (".jpg", ".jpeg", ".png", ".webp")

        self.paths = []
        num_checked = 0
        num_bad = 0

        print(f"[Dataset] Scanning {root} ...")
        for r, _, files in os.walk(root):
            for f in files:
                if not f.lower().endswith(self.exts):
                    continue
                path = os.path.join(r, f)
                num_checked += 1

                # 壊れ画像チェック（verify）
                try:
                    with Image.open(path) as img:
                        img.verify()  # ここで decode/ヘッダチェック
                    # verify した後は必ず再度 open し直す必要があるので、
                    # __getitem__ では改めて open する
                    self.paths.append(path)
                except Exception as e:
                    num_bad += 1
                    print(f"[WARN] Skipping corrupt image: {path} ({e})")

        print(f"[Dataset] Total images checked : {num_checked}")
        print(f"[Dataset] Valid images        : {len(self.paths)}")
        print(f"[Dataset] Corrupt images skip : {num_bad}")

        if len(self.paths) == 0:
            raise RuntimeError("No valid images found in dataset.")

        self.transform = transforms.Compose([
            transforms.Resize(self.image_size, interpolation=InterpolationMode.LANCZOS),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),                             # [0,1]
            transforms.Lambda(lambda t: t * 2.0 - 1.0),        # [-1,1]
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        # verify 済みなので基本エラーは出ないはずだが、念のため try/except
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            # ここで失敗したら、適当な別サンプルを返すかエラーにするかだが、
            # 非常にレアケースなので RuntimeError にしてしまう
            raise RuntimeError(f"Failed to open image at {path}: {e}")

        img = self.transform(img)
        return img


# ===============================
# 4. 引数
# ===============================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate trained SQ-VAE on full dataset (L2/PSNR/SSIM stats & histograms, skip corrupt images)"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/dev/shm/ffhq_512",
        help="Dataset root directory (recursive).",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=512,
        help="Image size (should match training).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to trained checkpoint (sqvae_latest.pth etc.)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./sqvae_eval_results",
        help="Where to save stats and hist plots.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help='Device to use ("cuda" or "cpu").',
    )

    return parser.parse_args()


# ===============================
# 5. メイン評価ループ
# ===============================
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device)

    # ==== Dataset & DataLoader ====
    dataset = ImageFolderAllRecursive(args.data_root, image_size=args.image_size)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    num_samples = len(dataset)
    print(f"[Eval] Valid samples: {num_samples}")

    # ==== モデル構築 & 重みロード ====
    print(f"[Checkpoint] Loading from {args.ckpt}")
    checkpoint = torch.load(args.ckpt, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]

    # 埋め込みの形状から num_embeddings / embedding_dim を自動推定
    emb_weight = state_dict["vq.embedding.weight"]
    num_embeddings = emb_weight.shape[0]
    embedding_dim = emb_weight.shape[1]

    print(f"[Model] num_embeddings={num_embeddings}, embedding_dim={embedding_dim}")

    model = SQVAEModel(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
    ).to(device)
    model.load_state_dict(state_dict)

    # ★★★ 推論モード → ハード量子化 ★★★
    model.eval()

    # ==== 評価ループ ====
    l2_all = []
    psnr_all = []
    ssim_all = []

    start_time = time.time()
    processed = 0

    with torch.no_grad():
        for i, images in enumerate(dataloader):
            images = images.to(device, non_blocking=True)  # [-1,1], [B,3,H,W]

            # temp は eval/hard 分岐では実質使われない
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=torch.bfloat16):
                recon, _ = model(images, temp=1.0)

            recon = recon.float()
            images_f = images.float()

            B = images.size(0)

            # --- L2 (MSE) per image ---
            mse = F.mse_loss(recon, images_f, reduction="none")  # [B,C,H,W]
            mse = mse.view(B, -1).mean(dim=1)                    # [B]
            l2_batch = mse.cpu().numpy()

            # --- PSNR / SSIM per image ---
            psnr_batch = calculate_psnr_per_image(images_f, recon).cpu().numpy()
            ssim_batch = ssim_per_image(images_f, recon).cpu().numpy()

            l2_all.extend(l2_batch.tolist())
            psnr_all.extend(psnr_batch.tolist())
            ssim_all.extend(ssim_batch.tolist())

            processed += B
            if (i + 1) % 10 == 0:
                print(f"[Eval] Processed {processed}/{num_samples} images")

    duration = time.time() - start_time
    print(f"[Eval] Done in {duration:.2f} seconds.")

    l2_all = np.array(l2_all)
    psnr_all = np.array(psnr_all)
    ssim_all = np.array(ssim_all)

    # ===============================
    # 6. 統計量（平均 / 中央 / 最頻値）計算
    # ===============================
    def summarize(name, arr):
        mean = float(arr.mean())
        median = float(np.median(arr))
        mode = float(approximate_mode(arr))
        print(f"\n[{name}]")
        print(f"  mean   : {mean}")
        print(f"  median : {median}")
        print(f"  mode(~): {mode}")
        return mean, median, mode

    print("\n===== Summary statistics =====")
    l2_mean, l2_med, l2_mode = summarize("L2 (MSE)", l2_all)
    psnr_mean, psnr_med, psnr_mode = summarize("PSNR", psnr_all)
    ssim_mean, ssim_med, ssim_mode = summarize("SSIM", ssim_all)

    # テキストで保存
    summary_path = os.path.join(args.output_dir, "reconstruction_stats.txt")
    with open(summary_path, "w") as f:
        f.write("Reconstruction statistics over dataset\n")
        f.write(f"Num samples: {len(l2_all)}\n\n")

        f.write("[L2 (MSE)]\n")
        f.write(f"mean   : {l2_mean}\n")
        f.write(f"median : {l2_med}\n")
        f.write(f"mode(~): {l2_mode}\n\n")

        f.write("[PSNR]\n")
        f.write(f"mean   : {psnr_mean}\n")
        f.write(f"median : {psnr_med}\n")
        f.write(f"mode(~): {psnr_mode}\n\n")

        f.write("[SSIM]\n")
        f.write(f"mean   : {ssim_mean}\n")
        f.write(f"median : {ssim_med}\n")
        f.write(f"mode(~): {ssim_mode}\n")
    print(f"[Save] Summary written to {summary_path}")

    # ===============================
    # 7. ヒストグラムを保存
    # ===============================
    def save_hist(values, title, xlabel, filename, bins=100):
        plt.figure(figsize=(6, 4))
        plt.hist(values, bins=bins)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Count")
        plt.tight_layout()
        out_path = os.path.join(args.output_dir, filename)
        plt.savefig(out_path)
        plt.close()
        print(f"[Save] Histogram saved to {out_path}")

    save_hist(l2_all, "L2 Reconstruction Error Histogram", "L2 (MSE)", "hist_l2.png")
    save_hist(psnr_all, "PSNR Histogram", "PSNR [dB]", "hist_psnr.png")
    save_hist(ssim_all, "SSIM Histogram", "SSIM", "hist_ssim.png")

    # 必要なら生データも numpy で保存
    np.save(os.path.join(args.output_dir, "l2_all.npy"), l2_all)
    np.save(os.path.join(args.output_dir, "psnr_all.npy"), psnr_all)
    np.save(os.path.join(args.output_dir, "ssim_all.npy"), ssim_all)
    print("[Save] Raw metric arrays saved as .npy")


if __name__ == "__main__":
    main()
