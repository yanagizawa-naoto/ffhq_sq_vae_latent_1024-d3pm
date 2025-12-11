import os
import argparse
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
import torchvision.transforms as T
import numpy as np

# 壊れた画像を無理やり読み込まない方針
ImageFile.LOAD_TRUNCATED_IMAGES = False

# ★ 学習コードのファイル名に合わせて import を修正してください
from ffhq_sq_vae import SQVAEModel


class ImageFolderFlat(Dataset):
    def __init__(self, root, image_size=512):
        self.root = root
        exts = (".jpg", ".jpeg", ".png", ".webp")

        # まず拡張子で全部集める
        all_files = []
        for r, _, fs in os.walk(root):
            for f in fs:
                if f.lower().endswith(exts):
                    all_files.append(os.path.join(r, f))

        if len(all_files) == 0:
            raise RuntimeError(f"No images found under {root}")

        # ★ ここで壊れた画像を除外する
        self.files = []
        for p in all_files:
            if self._is_valid_image(p):
                self.files.append(p)
            else:
                print(f"[Warn] Skipping corrupted image: {p}")

        if len(self.files) == 0:
            raise RuntimeError(f"All images under {root} are corrupted or unreadable.")

        self.transform = T.Compose(
            [
                T.Resize(image_size, interpolation=Image.BICUBIC),
                T.CenterCrop(image_size),
                T.ToTensor(),          # [0,1]
                T.Normalize(0.5, 0.5), # [-1,1]
            ]
        )

    @staticmethod
    def _is_valid_image(path: str) -> bool:
        """画像が壊れていないか簡易チェック"""
        try:
            # verify() は実際に decode まではしないが、
            # ヘッダやチャンクがおかしいとここで例外が出る
            with Image.open(path) as img:
                img.verify()
            return True
        except Exception:
            return False

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """念のためここでも try/except して壊れた画像を避ける"""
        path = self.files[idx]
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                img = self.transform(img)
        except OSError as e:
            # 実行中にファイルが消えた/壊れたなどレアケース用
            print(
                f"[Error] Failed to load image at runtime: {path} ({e}), "
                "this sample will be replaced by another index."
            )
            # 別のインデックスを返す（極端に壊れたデータセットでなければ OK）
            new_idx = (idx + 1) % len(self.files)
            return self.__getitem__(new_idx)

        return img, path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Encode images to discrete SQ-VAE latents (compressed)"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/dev/shm/ffhq_512",
        help="Directory containing images (recursively searched).",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to trained SQ-VAE checkpoint (.pth).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save latent files.",
    )
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)

    # ★ 学習時と同じ値にすること
    parser.add_argument("--num_embeddings", type=int, default=256)
    parser.add_argument("--embedding_dim", type=int, default=4)

    # 保存形式の選択
    parser.add_argument(
        "--save_format",
        type=str,
        default="npz",
        choices=["npz", "pt_gzip"],
        help="npz: numpy compressed, pt_gzip: torch.save with gzip",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = ImageFolderFlat(args.data_root, image_size=args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"[Info] Found {len(dataset)} images under {args.data_root}")

    model = SQVAEModel(
        num_embeddings=args.num_embeddings,
        embedding_dim=args.embedding_dim,
    ).to(device)

    print(f"[Info] Loading checkpoint from {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    import gzip  # pt_gzip 用

    with torch.no_grad():
        for step, (images, paths) in enumerate(loader):
            images = images.to(device, non_blocking=True)

            # ---- encode → quantize ----
            z = model.encoder(images)
            z = model._pre_vq_conv(z)  # [B, C, H, W]
            z_q, perplexity, encoding_indices = model.vq(z, 1.0)

            B, C, H, W = z.shape
            # encoding_indices: [B*H*W, 1] → [B, H, W]
            indices = encoding_indices.view(B, H, W).cpu()  # long

            # ★ uint16 にキャスト（num_embeddings <= 65535 前提）
            indices_u16 = indices.to(torch.uint16).numpy()  # [B, H, W], uint16

            for b in range(B):
                img_path = paths[b]
                rel = os.path.relpath(img_path, args.data_root)
                rel_no_ext = os.path.splitext(rel)[0]

                if args.save_format == "npz":
                    out_path = os.path.join(args.output_dir, rel_no_ext + ".npz")
                else:  # pt_gzip
                    out_path = os.path.join(args.output_dir, rel_no_ext + ".pt.gz")

                Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)

                if args.save_format == "npz":
                    # ★ zlib 圧縮された npz に保存（indices は uint16）
                    np.savez_compressed(
                        out_path,
                        indices=indices_u16[b],  # [H, W], uint16
                        h=H,
                        w=W,
                        image_path=img_path,
                    )
                else:
                    # ★ .pt を gzip 圧縮して保存
                    latent_dict = {
                        "indices": torch.from_numpy(indices_u16[b]),  # uint16 tensor
                        "image_path": img_path,
                        "height": H,
                        "width": W,
                    }
                    with gzip.open(out_path, "wb") as f:
                        torch.save(latent_dict, f)

            if (step + 1) % 10 == 0:
                print(f"[{step+1}/{len(loader)}] saved up to batch {step+1}")

    print("[Done] All latents saved to:", args.output_dir)


if __name__ == "__main__":
    main()
