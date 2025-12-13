# FFHQ SQ-VAE Latent (1024) + D3PM

This repository contains an experimental implementation of **SQ-VAE** (a derivative of Sony’s VQ-VAE research) combined with a **Discrete Diffusion Probabilistic Model (D3PM)** to generate face images from highly compressed discrete latent representations.

本プロジェクトは、**技術力の証明および研究実装の再現性確認**を目的として作成しました。

---

## Overview

本実装では以下の構成を採用しています。

- **Encoder / Decoder**: SQ-VAE (Sony の VQ-VAE 派生論文)
- **Latent Space**: 離散潜在表現（1024 codebook）
- **Generative Model**: Discrete Diffusion Probabilistic Model (D3PM)
- **Dataset**: FFHQ

SQ-VAE によって得られた離散潜在表現を用いて D3PM を訓練し、  
最終的に **顔画像を生成可能なモデル**を構築しました。

---

## Motivation

### なぜ SQ-VAE を選択したか

SQ-VAE は Sony の VQ-VAE 系研究の中で、

- **視覚品質をほとんど劣化させず**
- **非常に高い圧縮率**を実現できる

という特徴を持ちます。

本実験では、

- 画像を **JPEG 比で約 1/400** まで圧縮
- それにも関わらず顔画像としての視覚的品質を維持

できることを確認しました。

これは **生成モデルの学習対象を大幅に単純化できる**可能性を示します。

---

### なぜ D3PM を選択したか

D3PM を選定した理由は以下です。

- SQ-VAE の出力は **離散潜在表現**
- 連続潜在表現よりも **情報量が少ない**
- ⇒ **モデル容量を削減できるのではないか**と仮説を立てた

この仮説を検証するため、  
連続潜在ではなく **離散拡散モデル (D3PM)** を用いた生成を試みました。

---

## Results

- SQ-VAE により **高圧縮かつ高品質な潜在表現**の生成に成功
- D3PM により、潜在空間上での拡散生成が可能
- 潜在からのデコードによって **顔画像生成を確認**

一方で、

- **離散潜在表現を用いても、期待したほどモデルサイズの削減はできなかった**
- D3PM の構造上、カテゴリ数とステップ数により計算量が増加する傾向がある

という結果も得られました。

---

## Discussion

本実験から得られた知見は以下です。

- 離散潜在表現は **情報圧縮として非常に有効**
- しかし、
  - 離散化 = モデル軽量化  
    という単純な関係にはならない
- モデルサイズや計算量は、
  - 潜在の次元
  - 拡散ステップ数
  - カテゴリ数
  に強く依存する

この結果は、  
**「潜在表現の情報量」と「生成モデルの複雑さ」は独立に評価すべき**  
であることを示唆しています。

---

## Purpose of This Repository

- 最新研究（SQ-VAE / D3PM）の理解と再実装
- 圧縮 × 生成の設計トレードオフの検証
- **研究・実装能力のポートフォリオとしての提示**

---

## References

- VQ-VAE (van den Oord et al.)
- SQ-VAE (Sony R&D)
- Discrete Diffusion Probabilistic Models (D3PM)
- FFHQ Dataset

