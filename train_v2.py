# train_v2.py
# 下一版训练脚本（更像量化）：
# ✅ label 改为"横截面去均值"的相对收益 y_cs（按 date）
# ✅ 评估输出：MSE + IC(Pearson) + RankIC(Spearman)（按日横截面再平均）
# ✅ 损失：HuberLoss（更稳）
# ✅ 早停 + ReduceLROnPlateau 均以 -RankIC 为监控指标（RankIC 越高越好）
# ✅ 仍用训练期全局统计量对特征做标准化（避免泄漏）
# ✅ 加入 Test Set，训练结束后独立评估（不参与超参数调优）
#
# 数据目录：data_akshare/raw/*.parquet（或 csv）
# 直接运行：python train_v2.py

import os
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import get_root_logger
from data_utils import (
    StockWindowDatasetCS,
    attach_cross_sectional_label,
    collate_with_dates,
    compute_global_stats,
    load_all_frames,
    pearson_corr,
    spearman_corr,
)
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
    epochs: int = 30
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    clip_grad: float = 1.0

    train_start: str = "2018-01-01"
    train_end: str = "2024-06-30"
    valid_end: str = "2024-12-31"
    test_end: str = "2025-06-30"    # 独立测试集结束日期（不参与超参数调优）

    # 数据路径
    data_glob_parquet: str = "data_akshare/raw/*.parquet"
    data_glob_csv: str = "data_akshare/raw/*.csv"

    # 早停
    early_stop_patience: int = 5
    min_delta: float = 1e-5

    # label 横截面处理：winsorize
    winsor_p: float = 0.01  # 1% / 99%

    # 输出
    ckpt_dir: str = "checkpoints_v2"
    best_ckpt: str = "checkpoints_v2/best.pt"


cfg = CFG()


# -------------------------
# Eval: MSE + IC + RankIC (per date)
# -------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, float]:
    model.eval()

    all_pred = []
    all_y = []
    all_d = []
    losses = []

    for x, y, d in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)

        loss = nn.functional.mse_loss(pred, y)
        losses.append(loss.item())

        all_pred.append(pred.detach().cpu().numpy().reshape(-1))
        all_y.append(y.detach().cpu().numpy().reshape(-1))
        all_d.append(d)

    if not losses:
        return {"mse": 1e9, "ic": 0.0, "rankic": 0.0}

    pred_arr = np.concatenate(all_pred)
    y_arr    = np.concatenate(all_y)
    d_arr    = np.concatenate(all_d)

    # 按 date 分组算 IC/RankIC，再平均
    ic_list  = []
    ric_list = []
    for dt in np.unique(d_arr):
        m = (d_arr == dt)
        if m.sum() < 30:  # 横截面太小就跳过
            continue
        ic_list.append(pearson_corr(pred_arr[m], y_arr[m]))
        ric_list.append(spearman_corr(pred_arr[m], y_arr[m]))

    ic     = float(np.nanmean(ic_list))  if ic_list  else 0.0
    rankic = float(np.nanmean(ric_list)) if ric_list else 0.0
    mse    = float(np.mean(losses))

    return {"mse": mse, "ic": ic, "rankic": rankic}


# -------------------------
# Train
# -------------------------
def main():
    logger = get_root_logger()

    frames = load_all_frames(
        data_glob_parquet=cfg.data_glob_parquet,
        data_glob_csv=cfg.data_glob_csv,
        lookback=cfg.lookback,
        horizon=cfg.horizon,
    )

    # 先做横截面标签变换（得到 y_cs）
    attach_cross_sectional_label(frames, cfg.winsor_p)

    feat_cols = [c for c in frames[0].columns if c not in ("date", "y_raw", "y_cs")]
    global_mu, global_sd = compute_global_stats(frames, cfg.train_start, cfg.train_end, feat_cols)
    logger.info("全局统计量已计算，训练期: %s -> %s", cfg.train_start, cfg.train_end)

    train_start  = cfg.train_start
    train_end    = cfg.train_end
    valid_start  = (pd.to_datetime(train_end)  + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    valid_end    = cfg.valid_end
    test_start   = (pd.to_datetime(valid_end)  + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    test_end     = cfg.test_end

    train_ds = StockWindowDatasetCS(frames, train_start, train_end,  global_mu, global_sd, cfg.lookback)
    valid_ds = StockWindowDatasetCS(frames, valid_start, valid_end,  global_mu, global_sd, cfg.lookback)
    test_ds  = StockWindowDatasetCS(frames, test_start,  test_end,   global_mu, global_sd, cfg.lookback)

    if len(train_ds) == 0:
        raise RuntimeError("Train dataset empty.")
    if len(valid_ds) == 0:
        raise RuntimeError("Valid dataset empty.")

    logger.info(
        "train samples: %d | valid samples: %d | test samples: %d",
        len(train_ds), len(valid_ds), len(test_ds),
    )

    dl_kwargs = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device.startswith("cuda")),
        collate_fn=collate_with_dates,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  drop_last=True,  **dl_kwargs)
    valid_loader = DataLoader(valid_ds, shuffle=False, drop_last=False, **dl_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, drop_last=False, **dl_kwargs)

    num_features = train_ds.X.shape[-1]
    model = ReturnPredictor(num_features).to(cfg.device)

    opt       = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-3)
    # 监控 -RankIC（mode="min"，因为 RankIC 越大越好，取负号后越小越好）
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=2,
    )
    loss_fn = nn.HuberLoss(delta=0.02)

    best       = 1e9   # 监控 -RankIC，越小越好
    bad_epochs = 0
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses = []

        for x, y, _d in train_loader:
            x, y = x.to(cfg.device), y.to(cfg.device)
            pred = model(x)
            loss = loss_fn(pred, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
            opt.step()
            losses.append(loss.item())

        train_loss  = float(np.mean(losses)) if losses else 1e9
        val_metrics = evaluate(model, valid_loader, cfg.device)

        # 早停监控指标：-RankIC（RankIC 越大越好，取负号）
        monitor_val = -val_metrics["rankic"]
        scheduler.step(monitor_val)

        lr_now = opt.param_groups[0]["lr"]
        logger.info(
            "epoch=%d  lr=%.2e  train_huber=%.6f  "
            "valid_mse=%.6f  valid_ic=%.4f  ★valid_rankic=%.4f",
            epoch, lr_now, train_loss,
            val_metrics["mse"], val_metrics["ic"], val_metrics["rankic"],
        )

        # 早停依据：-RankIC 越小越好（即 RankIC 越大越好）
        if monitor_val + cfg.min_delta < best:
            best       = monitor_val
            bad_epochs = 0
            torch.save(
                {
                    "model":     model.state_dict(),
                    "cfg":       cfg.__dict__,
                    "global_mu": global_mu.to_dict(),
                    "global_sd": global_sd.to_dict(),
                },
                cfg.best_ckpt,
            )
            logger.info("★ 保存最佳 checkpoint (valid_rankic=%.4f): %s", val_metrics["rankic"], cfg.best_ckpt)
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.early_stop_patience:
                logger.info(
                    "早停触发，epoch=%d，best valid_rankic=%.4f",
                    epoch, -best,
                )
                break

    # -------------------------
    # Test Set 独立评估（不参与任何超参数调优）
    # -------------------------
    if len(test_ds) > 0:
        logger.info("=" * 60)
        logger.info("加载最佳 checkpoint 进行 Test Set 独立评估...")
        state = torch.load(cfg.best_ckpt, map_location=cfg.device)
        model.load_state_dict(state["model"])
        test_metrics = evaluate(model, test_loader, cfg.device)
        logger.info(
            "【Test Set 独立评估】 mse=%.6f  ic=%.4f  rankic=%.4f",
            test_metrics["mse"], test_metrics["ic"], test_metrics["rankic"],
        )
        logger.info("=" * 60)
    else:
        logger.warning("Test Set 为空，跳过独立评估（检查 test_end 配置）。")


if __name__ == "__main__":
    main()
