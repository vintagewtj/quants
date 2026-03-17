# train_v1.py
# 完整替换版：使用"训练期全局统计量"做标准化（避免数据泄漏）
# 适配你当前 akshare 数据目录：data_akshare/raw/*.parquet（或 *.csv）
#
# 关键点：
# - 每只股票先做特征工程（不做单股票 zscore）
# - 只用训练期 [train_start, train_end] 计算全局 mean/std
# - train / valid 都用同一组 mean/std 进行标准化
# - 严格时间切分：train <= train_end；valid in (train_end, valid_end]

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import get_root_logger
from data_utils import StockWindowDataset, compute_global_stats, load_all_frames
from model import ReturnPredictor


# -------------------------
# Config
# -------------------------
@dataclass
class CFG:
    lookback: int = 60
    horizon: int = 5
    batch_size: int = 256
    lr: float = 3e-4
    epochs: int = 10
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    clip_grad: float = 1.0

    # 时间切分
    train_start: str = "2018-01-01"
    train_end: str = "2024-12-31"
    valid_end: str = "2025-06-30"

    # 数据路径（适配 fetch_akshare_a_daily.py）
    data_glob_parquet: str = "data_akshare/raw/*.parquet"
    data_glob_csv: str = "data_akshare/raw/*.csv"

    # 输出
    ckpt_dir: str = "checkpoints"
    best_ckpt: str = "checkpoints/best.pt"


cfg = CFG()


# -------------------------
# Train / Eval
# -------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    losses = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = (pred - y).pow(2).mean()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else 1e9


def main():
    logger = get_root_logger()

    # 1) 读数据 & 特征工程（无标准化）
    frames = load_all_frames(
        data_glob_parquet=cfg.data_glob_parquet,
        data_glob_csv=cfg.data_glob_csv,
        lookback=cfg.lookback,
        horizon=cfg.horizon,
    )

    # 2) 全局统计量（仅训练期）
    feat_cols = [c for c in frames[0].columns if c not in ("date", "y_raw")]
    global_mu, global_sd = compute_global_stats(frames, cfg.train_start, cfg.train_end, feat_cols)
    logger.info("全局统计量已计算，训练期: %s -> %s", cfg.train_start, cfg.train_end)

    # 3) 严格时间切分
    train_start = cfg.train_start
    train_end = cfg.train_end
    valid_start = (pd.to_datetime(train_end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    valid_end = cfg.valid_end

    train_ds = StockWindowDataset(frames, train_start, train_end, global_mu, global_sd, cfg.lookback)
    valid_ds = StockWindowDataset(frames, valid_start, valid_end, global_mu, global_sd, cfg.lookback)

    if len(train_ds) == 0:
        raise RuntimeError("Train dataset is empty. Check train_start/train_end and your data range.")
    if len(valid_ds) == 0:
        raise RuntimeError("Valid dataset is empty. Check valid_end or adjust split dates.")

    logger.info("train samples: %d | valid samples: %d", len(train_ds), len(valid_ds))

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device.startswith("cuda")),
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device.startswith("cuda")),
        drop_last=False,
    )

    # 4) 模型
    num_features = train_ds.X.shape[-1]
    model = ReturnPredictor(num_features).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-3)

    best = 1e9
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses = []

        for x, y in train_loader:
            x, y = x.to(cfg.device), y.to(cfg.device)
            pred = model(x)
            loss = (pred - y).pow(2).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
            opt.step()

            losses.append(loss.item())

        train_mse = float(np.mean(losses)) if losses else 1e9
        val_mse = evaluate(model, valid_loader, cfg.device)
        logger.info("epoch=%d  train_mse=%.6f  valid_mse=%.6f", epoch, train_mse, val_mse)

        if val_mse < best:
            best = val_mse
            torch.save(
                {
                    "model": model.state_dict(),
                    "cfg": cfg.__dict__,
                    "global_mu": global_mu.to_dict(),
                    "global_sd": global_sd.to_dict(),
                },
                cfg.best_ckpt,
            )
            logger.info("已保存 checkpoint: %s", cfg.best_ckpt)


if __name__ == "__main__":
    main()
