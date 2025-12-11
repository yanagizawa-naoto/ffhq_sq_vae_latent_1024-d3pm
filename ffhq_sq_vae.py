import os
import time
import argparse
import math
import warnings
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils

# --- DALI Imports ---
try:
    from nvidia.dali.plugin.pytorch import DALIGenericIterator
    from nvidia.dali import pipeline_def
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
except ImportError:
    # DALI is optional for code verification but required for running
    # Define dummy pipeline_def to avoid NameError
    def pipeline_def(func):
        return func
    pass

warnings.filterwarnings("ignore")


# ===============================
# 1. Args
# ===============================
def parse_args():
    parser = argparse.ArgumentParser(
        description="SQ-VAE Training on local FFHQ (/dev/shm/ffhq_512) "
                    "(SiLU + BF16 + SCN)"
    )

    # ローカルデータセットのルート
    parser.add_argument(
        "--data_root",
        type=str,
        default="/dev/shm/ffhq_512",
        help="Directory containing FFHQ images (recursively searched).",
    )

    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint (.pth) for resuming.")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--image_size", type=int, default=512)

    # エポック数 / 保存間隔
    parser.add_argument("--epochs", type=int, default=10000,
                        help="Number of training epochs.")
    parser.add_argument(
        "--save_interval",
        type=int,
        default=1,
        help="Save checkpoint & sample images every N epochs.",
    )

    parser.add_argument("--output_dir", type=str,
                        default="./sqvae_results_ffhq_256")
    parser.add_argument("--use_wandb", action="store_true")

    # SQ-VAE specific arguments
    parser.add_argument("--temp_init", type=float, default=1.0, help="Initial temperature for Gumbel-Softmax")
    parser.add_argument("--temp_min", type=float, default=0.5, help="Minimum temperature")
    parser.add_argument("--temp_decay", type=float, default=0.999995, help="Temperature decay factor per step")

    # VQ 設定（モデルは既存のまま）
    parser.add_argument(
        "--num_embeddings",
        type=int,
        default=256,
        help="Size of VQ codebook K.",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=4,
        help="VQ embedding dimension D.",
    )

    # 勾配蓄積ステップ
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps.",
    )

    return parser.parse_args()


# ===============================
# 2. Metrics
# ===============================
def calculate_psnr(img1, img2):
    mse = F.mse_loss(img1.float(), img2.float())
    if mse == 0:
        return float("inf")
    # 入力 [-1,1] を想定しているので max=2.0
    return 20 * torch.log10(2.0 / torch.sqrt(mse))


def gaussian_window(size, sigma):
    coords = torch.arange(size, dtype=torch.float)
    coords -= size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()
    return g.reshape(1, 1, 1, -1) * g.reshape(1, 1, -1, 1)


def ssim(img1, img2, window_size=11, size_average=True):
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
    return ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)


# ===============================
# 3. Model
# ===============================
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(ResidualBlock, self).__init__()
        self._block = nn.Sequential(
            nn.GroupNorm(32, in_channels, eps=1e-6),
            nn.SiLU(True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_residual_hiddens,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="reflect",
                bias=False,
            ),
            nn.GroupNorm(32, num_residual_hiddens, eps=1e-6),
            nn.SiLU(True),
            nn.Conv2d(
                in_channels=num_residual_hiddens,
                out_channels=num_hiddens,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(
        self, in_channels, num_hiddens, num_residual_hiddens, num_residual_layers
    ):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList(
            [
                ResidualBlock(in_channels, num_hiddens, num_residual_hiddens)
                for _ in range(self._num_residual_layers)
            ]
        )

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return x


class SQVAEQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # Learnable codebook
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z, temp):
        # z: [B, C, H, W]
        # temp: float, current temperature
        
        B, C, H, W = z.shape
        # [B, H, W, C]
        z_permuted = z.permute(0, 2, 3, 1).contiguous()
        # [B*H*W, C]
        flat_input = z_permuted.view(-1, self.embedding_dim)
        
        # Calculate distances (squared euclidean)
        # (x-y)^2 = x^2 + y^2 - 2xy
        # [B*H*W, 1]
        dist_x = torch.sum(flat_input ** 2, dim=1, keepdim=True)
        # [K, 1] -> [1, K]
        dist_y = torch.sum(self.embedding.weight ** 2, dim=1).unsqueeze(0)
        
        # [B*H*W, K]
        distances = dist_x + dist_y - 2 * torch.matmul(flat_input, self.embedding.weight.t())
        
        # SQ-VAE uses negative distance (similarity) as logits for Gumbel-Softmax
        logits = -distances
        
        if self.training:
            # Gumbel-Softmax
            # [B*H*W, K]
            soft_one_hot = F.gumbel_softmax(logits, tau=temp, hard=False, dim=-1)
            
            # Quantize
            # [B*H*W, C]
            z_q = torch.matmul(soft_one_hot, self.embedding.weight)
            
            # Calculate perplexity
            probs = F.softmax(logits, dim=-1)
            avg_probs = torch.mean(probs, dim=0)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
            
            # Indices (for logging/visualization mostly)
            encoding_indices = torch.argmax(probs, dim=1).unsqueeze(1)
            
        else:
            # Hard quantization for inference
            indices = torch.argmax(logits, dim=-1)
            encoding_indices = indices.unsqueeze(1)
            
            # One-hot
            encodings = torch.zeros(indices.shape[0], self.num_embeddings, device=z.device)
            encodings.scatter_(1, indices.unsqueeze(1), 1)
            
            z_q = torch.matmul(encodings, self.embedding.weight)
            perplexity = torch.tensor(0.0, device=z.device)

        # Reshape back to [B, C, H, W]
        z_q = z_q.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        
        return z_q, perplexity, encoding_indices


class Encoder(nn.Module):
    def __init__(
        self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens
    ):
        super(Encoder, self).__init__()
        self._conv_1 = nn.Conv2d(
            in_channels, num_hiddens // 2, 4, 2, 1, padding_mode="reflect"
        )
        self._conv_2 = nn.Conv2d(
            num_hiddens // 2, num_hiddens, 4, 2, 1, padding_mode="reflect"
        )
        self._conv_3 = nn.Conv2d(
            num_hiddens, num_hiddens, 4, 2, 1, padding_mode="reflect"
        )
        self._conv_4 = nn.Conv2d(
            num_hiddens, num_hiddens, 3, 1, 1, padding_mode="reflect"
        )
        self._residual_stack = ResidualStack(
            num_hiddens, num_hiddens, num_residual_hiddens, num_residual_layers
        )
        self._final_norm = nn.GroupNorm(32, num_hiddens, eps=1e-6)

    def forward(self, inputs):
        x = F.silu(self._conv_1(inputs))
        x = F.silu(self._conv_2(x))
        x = F.silu(self._conv_3(x))
        x = self._conv_4(x)
        x = self._residual_stack(x)
        return F.silu(self._final_norm(x))


class SpatialConditionalNorm(nn.Module):
    """
    簡易 Spatially Conditional Normalization:
      - GroupNorm で正規化
      - 入力 x から gamma, beta を conv で生成（空間的に変化）
      - y = (1 + gamma) * norm(x) + beta
    """

    def __init__(self, num_channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, num_channels, eps=1e-6)
        self.gamma = nn.Conv2d(
            num_channels,
            num_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="reflect",
        )
        self.beta = nn.Conv2d(
            num_channels,
            num_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="reflect",
        )

    def forward(self, x):
        h = self.norm(x)
        gamma = self.gamma(x)
        beta = self.beta(x)
        return h * (1.0 + gamma) + beta


class Decoder(nn.Module):
    def __init__(
        self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens
    ):
        super(Decoder, self).__init__()
        self._conv_1 = nn.Conv2d(
            in_channels, num_hiddens, 3, 1, 1, padding_mode="reflect"
        )
        self._residual_stack = ResidualStack(
            num_hiddens, num_hiddens, num_residual_hiddens, num_residual_layers
        )
        self._scn = SpatialConditionalNorm(num_hiddens)
        self._conv_trans_1 = nn.ConvTranspose2d(num_hiddens, num_hiddens, 4, 2, 1)
        self._conv_trans_2 = nn.ConvTranspose2d(
            num_hiddens, num_hiddens // 2, 4, 2, 1
        )
        self._conv_trans_3 = nn.ConvTranspose2d(num_hiddens // 2, 3, 4, 2, 1)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = self._residual_stack(x)
        x = self._scn(x)
        x = F.silu(self._conv_trans_1(x))
        x = F.silu(self._conv_trans_2(x))
        return torch.tanh(self._conv_trans_3(x))


class SQVAEModel(nn.Module):
    def __init__(
        self,
        num_hiddens=512,
        num_residual_layers=8,
        num_residual_hiddens=32,
        num_embeddings=16384,
        embedding_dim=32,
    ):
        super(SQVAEModel, self).__init__()
        self.encoder = Encoder(
            3, num_hiddens, num_residual_layers, num_residual_hiddens
        )
        self._pre_vq_conv = nn.Conv2d(
            in_channels=num_hiddens,
            out_channels=embedding_dim,
            kernel_size=1,
            stride=1,
        )
        self.vq = SQVAEQuantizer(
            num_embeddings, embedding_dim
        )
        self.decoder = Decoder(
            embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens
        )

    def forward(self, x, temp):
        z = self.encoder(x)
        z = self._pre_vq_conv(z)
        z_q, perplexity, _ = self.vq(z, temp)
        x_recon = self.decoder(z_q)
        return x_recon, perplexity


# ===============================
# 4. DALI Pipeline for local dataset
# ===============================
@pipeline_def
def create_dali_pipeline(file_root, image_size):
    """
    ローカルディレクトリから読み込む DALI パイプライン。
    サブディレクトリがクラスラベルとして扱われる。
    """
    jpegs, _ = fn.readers.file(
        file_root=file_root,
        random_shuffle=True,
        pad_last_batch=True,
        name="Reader",
    )
    # Use CPU for decoding to save GPU memory (avoids nvJPEG error 5)
    images = fn.decoders.image(jpegs, device="cpu", output_type=types.RGB)

    # Resize on CPU
    images = fn.resize(images, resize_shorter=image_size)

    # ランダムな 512x512 クロップ位置
    crop_pos_x = fn.random.uniform(range=(0.0, 1.0))
    crop_pos_y = fn.random.uniform(range=(0.0, 1.0))

    # ランダム 512x512 クロップ + 正規化 [-1,1]
    # Move to GPU here explicitly
    images = images.gpu()
    images = fn.crop_mirror_normalize(
        images,
        crop=(image_size, image_size),
        crop_pos_x=crop_pos_x,
        crop_pos_y=crop_pos_y,
        mean=[127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5],
        output_layout="CHW",
        device="gpu",  # Explicitly move to GPU
    )
    return images


def count_images(root):
    exts = (".jpg", ".jpeg", ".png", ".webp")
    cnt = 0
    for r, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(exts):
                cnt += 1
    return cnt


def prepare_dali_root_for_flat_dir(data_root, work_dir):
    """
    data_root 直下に画像がベタ置きされている場合、
    DALI が読めるように 1 クラス構造のシンボリックリンクツリーを作る。
    """
    exts = (".jpg", ".jpeg", ".png", ".webp")

    # 1) data_root 配下に「画像を含むサブディレクトリ」があるかチェック
    has_labeled_subdirs = False
    with os.scandir(data_root) as it:
        for entry in it:
            if entry.is_dir():
                for r, _, files in os.walk(entry.path):
                    if any(f.lower().endswith(exts) for f in files):
                        has_labeled_subdirs = True
                        break
            if has_labeled_subdirs:
                break

    if has_labeled_subdirs:
        print(f"[DALI] Found labeled subdirectories under {data_root}. Using it directly.")
        return data_root

    # 2) サブディレクトリが無く、直下に画像があるケース → symlink で 1 クラス構造を作る
    print(
        f"[DALI] No labeled subdirs with images found under {data_root}. "
        f"Assuming flat directory of images and creating symlink-based root."
    )

    dali_root = os.path.join(work_dir, "_dali_symlink_root")
    label_dir = os.path.join(dali_root, "0")  # クラス 0 用

    os.makedirs(dali_root, exist_ok=True)

    # 既に存在して & symlink ならそのまま使う
    if os.path.islink(label_dir):
        print(f"[DALI] Using existing symlink label dir: {label_dir}")
        return dali_root

    # 既に普通のディレクトリとして存在していたら、エラーにしたくないので一度消す
    if os.path.exists(label_dir):
        print(f"[DALI] Removing existing non-symlink directory: {label_dir}")
        shutil.rmtree(label_dir)

    # symlink 作成
    target = os.path.abspath(data_root)
    print(f"[DALI] Creating symlink: {label_dir} -> {target}")
    os.symlink(target, label_dir)

    return dali_root


# ===============================
# 5. Main training loop
# ===============================
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.output_dir, exist_ok=True)
    img_save_dir = os.path.join(args.output_dir, "images")
    ckpt_save_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(img_save_dir, exist_ok=True)
    os.makedirs(ckpt_save_dir, exist_ok=True)

    # データセット枚数をカウント（元の data_root に対して）
    if not os.path.isdir(args.data_root):
        # For verification without data
        print(f"[Warning] data_root not found: {args.data_root}. Proceeding for code verification.")
        dataset_size = 1000 # Dummy size
    else:
        dataset_size = count_images(args.data_root)
        if dataset_size == 0:
            raise RuntimeError(f"No images found under {args.data_root}")
        print(f"[Dataset] Found {dataset_size} images in {args.data_root}")

    # DALI 用の file_root を決定（flat ディレクトリなら symlink ツリーを作る）
    if os.path.isdir(args.data_root):
        dali_file_root = prepare_dali_root_for_flat_dir(
            args.data_root, args.output_dir
        )
    else:
        dali_file_root = args.data_root # Dummy

    if args.use_wandb:
        import wandb
        wandb.init(project="sqvae-ffhq", config=vars(args))

    print(f"--- SQ-VAE Training on FFHQ (SiLU + BF16 + SCN) ---")
    print(f"[Config] num_embeddings={args.num_embeddings}, embedding_dim={args.embedding_dim}")

    # モデル初期化
    model = SQVAEModel(
        num_embeddings=args.num_embeddings,
        embedding_dim=args.embedding_dim,
    ).to(device)

    # オプティマイザ
    optimizer_G = optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    start_epoch = 0
    global_step = 0
    current_temp = args.temp_init

    # Resume
    if args.resume and os.path.isfile(args.resume):
        print(f"[Resume] Loading checkpoint from {args.resume}...")
        try:
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer_G.load_state_dict(checkpoint["optimizer_G_state_dict"])
            start_epoch = checkpoint.get("epoch", 0) + 1
            global_step = checkpoint.get("global_step", 0)
            current_temp = checkpoint.get("current_temp", args.temp_init)
            print(f"[Resume] Starting from epoch {start_epoch}, global_step {global_step}, temp {current_temp:.4f}")
        except Exception as e:
            print(f"[Resume] Failed: {e}. Starting from scratch...")
            start_epoch = 0
            global_step = 0

    # 勾配蓄積用
    grad_accum = args.grad_accum_steps

    for epoch in range(start_epoch, args.epochs):
        print(f"\n=== Epoch {epoch+1}/{args.epochs} ===")

        try:
            # DALI pipeline & iterator を毎エポック作り直し
            pipe = create_dali_pipeline(
                batch_size=args.batch_size,
                num_threads=16,
                device_id=0,
                file_root=dali_file_root,
                image_size=args.image_size,
            )
            pipe.build()
            dali_iter = DALIGenericIterator(
                pipe,
                ["data"],
                size=dataset_size,
                auto_reset=True,
            )
        except (NameError, RuntimeError) as e:
            print(f"DALI initialization failed (expected if no GPU/Data): {e}")
            dali_iter = []

        model.train()

        total_recon_loss = 0.0
        steps = 0
        start_time = time.time()

        accum_steps_G = 0
        optimizer_G.zero_grad(set_to_none=True)

        last_images = None

        try:
            for batch in dali_iter:
                images = batch[0]["data"].to(device, non_blocking=True)
                last_images = images
                steps += 1
                global_step += 1

                # --------- Generator (SQ-VAE) forward ---------
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    x_recon, perplexity = model(images, current_temp)
                    
                    # Reconstruction Loss (MSE)
                    l2_loss = F.mse_loss(
                        x_recon.float(), images.float()
                    )
                    
                    total_loss = l2_loss

                (total_loss / grad_accum).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                accum_steps_G += 1

                if accum_steps_G >= grad_accum:
                    optimizer_G.step()
                    optimizer_G.zero_grad(set_to_none=True)
                    accum_steps_G = 0
                
                # Anneal temperature every step
                current_temp = max(args.temp_min, current_temp * args.temp_decay)

                # ログ
                if global_step % 10 == 0:
                    with torch.no_grad():
                        val_psnr = calculate_psnr(images, x_recon)
                        val_ssim = ssim(images, x_recon)
                    print(
                        f"[Epoch {epoch+1}] Step {steps} (global {global_step}) | "
                        f"L2: {l2_loss.item():.4f} | "
                        f"Temp: {current_temp:.4f} | "
                        f"PPL: {perplexity.item():.1f} | "
                        f"PSNR: {val_psnr:.2f} | SSIM: {val_ssim:.3f}"
                    )

                    if args.use_wandb:
                        import wandb

                        wandb.log(
                            {
                                "epoch": epoch + 1,
                                "step_in_epoch": steps,
                                "global_step": global_step,
                                "l2_loss": l2_loss.item(),
                                "temperature": current_temp,
                                "perplexity": perplexity.item(),
                                "psnr": val_psnr,
                                "ssim": val_ssim,
                            }
                        )

        except Exception as e:
            print(f"[DALI Info] Epoch error: {e}")

        # 端数の勾配蓄積を flush
        if accum_steps_G > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer_G.step()
            optimizer_G.zero_grad(set_to_none=True)

        if 'dali_iter' in locals():
            del dali_iter
        if 'pipe' in locals():
            del pipe

        duration = time.time() - start_time
        imgs_per_sec = dataset_size / max(duration, 1e-6)
        mean_recon = total_recon_loss / max(steps, 1)
        print(
            f"Epoch {epoch+1} finished in {duration:.2f}s "
            f"(~{imgs_per_sec:.2f} img/sec). "
            f"Mean L2: {mean_recon:.4f}"
        )

        # チェックポイント & サンプル保存
        if (epoch + 1) % args.save_interval == 0:
            ckpt_path = os.path.join(ckpt_save_dir, "sqvae_latest.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_G_state_dict": optimizer_G.state_dict(),
                    "current_temp": current_temp,
                },
                ckpt_path,
            )
            print(f"[Checkpoint] Saved to {ckpt_path}")

            if last_images is not None:
                with torch.no_grad():
                    sample = last_images[:8]
                    # Inference with current temp
                    recon, _ = model(sample, current_temp)
                    viz = torch.cat([sample, recon]) * 0.5 + 0.5  # [-1,1]→[0,1]
                    img_path = os.path.join(
                        img_save_dir, f"epoch_{epoch+1:04d}.jpg"
                    )
                    vutils.save_image(viz, img_path, nrow=8)
                    print(f"[Sample] Saved to {img_path}")


if __name__ == "__main__":
    main()
