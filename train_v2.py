# train_v2.py
# 下一版训练脚本（更像量化）：
# ✅ label 改为“横截面去均值”的相对收益 y_cs（按 date）
# ✅ 评估输出：MSE + IC(Pearson) + RankIC(Spearman)（按日横截面再平均）
# ✅ 损失：HuberLoss（更稳）
# ✅ Early Stopping + ReduceLROnPlateau
# ✅ 仍用训练期全局统计量对特征做标准化（避免泄漏）
#
# 数据目录：data_akshare/raw/*.parquet（或 csv）
# 直接运行：python train_v2.py

import gc
import os
import math
import glob
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


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
    train_end: str = "2024-12-31"
    valid_end: str = "2025-06-30"

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
# Utils
# -------------------------
def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman = Pearson(rank(x), rank(y))，用 pandas rank 简化实现"""
    sx = pd.Series(x).rank(method="average").to_numpy()
    sy = pd.Series(y).rank(method="average").to_numpy()
    if np.std(sx) < 1e-12 or np.std(sy) < 1e-12:
        return np.nan
    return float(np.corrcoef(sx, sy)[0, 1])


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


# -------------------------
# Feature engineering
# -------------------------
def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    输出列：
      date(datetime64), feat_cols..., y_raw(未来N日收益)
    不做标准化；标准化在训练期统计量里做
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["r1"] = df["close"].pct_change(1)
    df["r5"] = df["close"].pct_change(5)
    df["r10"] = df["close"].pct_change(10)

    def ma(x, w): return x.rolling(w).mean()
    def std(x, w): return x.rolling(w).std()

    df["ma20"] = ma(df["close"], 20)
    df["ma60"] = ma(df["close"], 60)
    df["d_ma20"] = df["close"] / df["ma20"] - 1.0
    df["d_ma60"] = df["close"] / df["ma60"] - 1.0

    df["vol_ma20"] = ma(df["volume"], 20)
    df["d_vol20"] = df["volume"] / df["vol_ma20"] - 1.0

    df["hl_range"] = (df["high"] - df["low"]) / df["close"]
    df["vol10"] = std(df["r1"], 10)
    df["vol20"] = std(df["r1"], 20)
    df["vol60"] = std(df["r1"], 60)

    # raw label
    df["y_raw"] = df["close"].shift(-cfg.horizon) / df["close"] - 1.0

    feat_cols = ["r1", "r5", "r10", "d_ma20", "d_ma60", "d_vol20", "hl_range", "vol10", "vol20", "vol60"]
    df = df.dropna(subset=feat_cols + ["y_raw"]).reset_index(drop=True)

    return df[["date"] + feat_cols + ["y_raw"]]


# -------------------------
# Global normalization (train-period stats)
# -------------------------
def compute_global_stats(frames: List[pd.DataFrame], train_start: str, train_end: str, feat_cols: List[str]) -> Tuple[pd.Series, pd.Series]:
    start_dt = pd.to_datetime(train_start)
    end_dt = pd.to_datetime(train_end)

    buf = []
    for df in frames:
        dff = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]
        if len(dff) == 0:
            continue
        buf.append(dff[feat_cols])

    if not buf:
        raise RuntimeError("No training-period data available to compute global stats. Check train_start/train_end.")

    all_feats = pd.concat(buf, axis=0)
    mu = all_feats.mean()
    sd = all_feats.std()
    sd = sd.replace(0, 1e-8).fillna(1e-8)
    return mu, sd


# -------------------------
# Cross-sectional label transform
# -------------------------
def attach_cross_sectional_label(frames: List[pd.DataFrame], winsor_p: float) -> None:
    """
    原地修改 frames：增加 y_cs
    y_cs = winsorize(y_raw within date) then demean by date

    只 concat ["date", "y_raw"] 两列来计算横截面统计，减少内存占用；
    计算出每个 date 的 winsorize 边界和均值后，逐 frame 回写结果。
    """
    def _winsor(s: pd.Series) -> pd.Series:
        lo = s.quantile(winsor_p)
        hi = s.quantile(1 - winsor_p)
        return s.clip(lo, hi)

    # 仅 concat 必要的两列，大幅减少峰值内存
    slim = pd.concat(
        [df[["date", "y_raw"]] for df in frames],
        axis=0,
        ignore_index=True,
    )
    slim["y_w"] = slim.groupby("date")["y_raw"].transform(_winsor)

    # 计算每个 date 的均值，存为字典便于快速查找
    date_mean_dict = slim.groupby("date")["y_w"].mean().to_dict()

    # 拆回每个 frame，逐帧回写 y_cs，避免持有整个大 DataFrame
    idx = 0
    for i, df in enumerate(frames):
        L = len(df)
        y_w = slim["y_w"].iloc[idx: idx + L].values
        dates = slim["date"].iloc[idx: idx + L].values
        means = np.array([date_mean_dict[d] for d in dates], dtype=np.float32)
        frames[i]["y_cs"] = (y_w - means).astype(np.float32)
        idx += L

    del slim, date_mean_dict
    gc.collect()


# -------------------------
# Dataset
# -------------------------
class StockWindowDatasetCS(Dataset):
    """
    懒加载 Dataset：不在 __init__ 中预构建全量滑窗 tensor，
    改为存储原始 numpy 数组 + 索引列表，在 __getitem__ 中按需切窗口。
    峰值内存从 ~37 GB 降至 ~1-2 GB。
    """
    def __init__(self, frames: List[pd.DataFrame], start_date: str, end_date: str, mu: pd.Series, sd: pd.Series):
        self.samples = []       # list of (frame_idx, row_idx)
        self.frames_data = []   # list of (feats_np, labels_np, dates_np)
        self.lookback = cfg.lookback

        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        feat_cols = [c for c in frames[0].columns if c not in ["date", "y_raw", "y_cs"]]
        mu = mu[feat_cols]
        sd = sd[feat_cols]

        for df in frames:
            dff = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)].reset_index(drop=True)
            if len(dff) < cfg.lookback + 1:
                continue

            dff = dff.copy()
            dff[feat_cols] = (dff[feat_cols] - mu) / sd

            feats = dff[feat_cols].values.astype(np.float32)
            labels = dff["y_cs"].values.astype(np.float32)
            dates = dff["date"].values  # numpy datetime64

            fi = len(self.frames_data)
            self.frames_data.append((feats, labels, dates))

            for i in range(cfg.lookback - 1, len(dff)):
                self.samples.append((fi, i))

        self.num_features = len(feat_cols)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fi, ri = self.samples[idx]
        feats, labels, dates = self.frames_data[fi]
        x = torch.from_numpy(feats[ri - self.lookback + 1: ri + 1].copy())
        y = torch.tensor([labels[ri]], dtype=torch.float32)
        d = dates[ri]
        return x, y, d


def collate_with_dates(batch):
    xs, ys, ds = zip(*batch)
    x = torch.stack(xs, dim=0)
    y = torch.stack(ys, dim=0)
    # dates keep as numpy array/object list -> return as numpy array
    d = np.array(ds)
    return x, y, d


# -------------------------
# Model
# -------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class ReturnPredictor(nn.Module):
    def __init__(self, num_features: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(num_features, d_model)
        self.pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        h = self.pos(h)
        h = self.enc(h)
        return self.head(h[:, -1, :])


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

    pred = np.concatenate(all_pred)
    y = np.concatenate(all_y)
    d = np.concatenate(all_d)

    # 按 date 分组算 IC/RankIC，再平均
    ic_list = []
    ric_list = []
    for dt in np.unique(d):
        m = (d == dt)
        if m.sum() < 30:  # 横截面太小就跳过
            continue
        ic_list.append(pearson_corr(pred[m], y[m]))
        ric_list.append(spearman_corr(pred[m], y[m]))

    ic = float(np.nanmean(ic_list)) if ic_list else 0.0
    rankic = float(np.nanmean(ric_list)) if ric_list else 0.0
    mse = float(np.mean(losses))

    return {"mse": mse, "ic": ic, "rankic": rankic}


# -------------------------
# Data loading
# -------------------------
def load_all_frames() -> List[pd.DataFrame]:
    raw_files = glob.glob(cfg.data_glob_parquet)
    read_parquet = True
    if not raw_files:
        raw_files = glob.glob(cfg.data_glob_csv)
        read_parquet = False

    if not raw_files:
        raise RuntimeError(
            f"No data found.\nTried: {cfg.data_glob_parquet}\n   and: {cfg.data_glob_csv}\n"
            f"Put your files under data_akshare/raw/"
        )

    frames = []
    usable = 0
    for fp in raw_files:
        try:
            df = pd.read_parquet(fp) if read_parquet else pd.read_csv(fp)
            need = {"date", "open", "high", "low", "close", "volume"}
            if not need.issubset(set(df.columns)):
                continue
            feats = make_features(df)
            if len(feats) >= cfg.lookback + 10:
                frames.append(feats)
                usable += 1
        except Exception:
            continue

    if not frames:
        raise RuntimeError("All files failed to produce usable features.")
    print(f"loaded files: {len(raw_files)} | usable frames: {usable}")
    return frames


# -------------------------
# Train
# -------------------------
def main():
    frames = load_all_frames()

    # 先做横截面标签变换（得到 y_cs）
    attach_cross_sectional_label(frames, cfg.winsor_p)
    gc.collect()

    feat_cols = [c for c in frames[0].columns if c not in ["date", "y_raw", "y_cs"]]
    global_mu, global_sd = compute_global_stats(frames, cfg.train_start, cfg.train_end, feat_cols)
    print("global stats computed on train period:", cfg.train_start, "->", cfg.train_end)
    gc.collect()

    train_start = cfg.train_start
    train_end = cfg.train_end
    valid_start = (pd.to_datetime(train_end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    valid_end = cfg.valid_end

    train_ds = StockWindowDatasetCS(frames, train_start, train_end, global_mu, global_sd)
    gc.collect()
    valid_ds = StockWindowDatasetCS(frames, valid_start, valid_end, global_mu, global_sd)
    gc.collect()

    del frames
    gc.collect()

    if len(train_ds) == 0:
        raise RuntimeError("Train dataset empty.")
    if len(valid_ds) == 0:
        raise RuntimeError("Valid dataset empty.")

    print(f"train samples: {len(train_ds)} | valid samples: {len(valid_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device.startswith("cuda")),
        drop_last=True,
        collate_fn=collate_with_dates,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device.startswith("cuda")),
        drop_last=False,
        collate_fn=collate_with_dates,
    )

    num_features = train_ds.num_features
    model = ReturnPredictor(num_features).to(cfg.device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2, )

    loss_fn = nn.HuberLoss(delta=0.02)  # delta 可调；0.02 对5日收益比较常用

    best = 1e9
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

        train_loss = float(np.mean(losses)) if losses else 1e9
        val_metrics = evaluate(model, valid_loader, cfg.device)

        # scheduler 根据 valid mse 调整学习率
        scheduler.step(val_metrics["mse"])

        lr_now = opt.param_groups[0]["lr"]
        print(
            f"epoch={epoch} lr={lr_now:.2e} "
            f"train_huber={train_loss:.6f} "
            f"valid_mse={val_metrics['mse']:.6f} "
            f"valid_ic={val_metrics['ic']:.4f} "
            f"valid_rankic={val_metrics['rankic']:.4f}"
        )

        # 早停依据：valid_mse
        if val_metrics["mse"] + cfg.min_delta < best:
            best = val_metrics["mse"]
            bad_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "cfg": cfg.__dict__,
                    "global_mu": global_mu.to_dict(),
                    "global_sd": global_sd.to_dict(),
                },
                cfg.best_ckpt,
            )
            print(f"saved {cfg.best_ckpt}")
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.early_stop_patience:
                print(f"Early stopping triggered at epoch={epoch}. best_valid_mse={best:.6f}")
                break


if __name__ == "__main__":
    main()
