# train.py
# 完整替换版：使用“训练期全局统计量”做标准化（避免数据泄漏）
# 适配你当前 akshare 数据目录：data_akshare/raw/*.parquet（或 *.csv）
#
# 关键点：
# - 每只股票先做特征工程（不做单股票 zscore）
# - 只用训练期 [train_start, train_end] 计算全局 mean/std
# - train / valid 都用同一组 mean/std 进行标准化
# - 严格时间切分：train <= train_end；valid in (train_end, valid_end]

import os
import math
import glob
from dataclasses import dataclass
from typing import List, Tuple, Optional

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
# Feature engineering
# -------------------------
def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    输入 df 至少要有：date, open, high, low, close, volume
    输出：date + feat_cols + y
    这里不做标准化（由训练期全局统计量统一标准化）
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 收益率特征
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

    # label: future N-day return
    df["y"] = df["close"].shift(-cfg.horizon) / df["close"] - 1.0

    feat_cols = ["r1", "r5", "r10", "d_ma20", "d_ma60", "d_vol20", "hl_range", "vol10", "vol20", "vol60"]
    df = df.dropna(subset=feat_cols + ["y"]).reset_index(drop=True)

    return df[["date"] + feat_cols + ["y"]]


# -------------------------
# Global normalization (train-period stats)
# -------------------------
def compute_global_stats(
    frames: List[pd.DataFrame],
    train_start: str,
    train_end: str,
    feat_cols: List[str]
) -> Tuple[pd.Series, pd.Series]:
    """
    只用训练期的数据计算全局 mean/std（跨股票、跨日期）
    """
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

    # 防止除 0
    sd = sd.replace(0, 1e-8).fillna(1e-8)

    return mu, sd


# -------------------------
# Dataset
# -------------------------
class StockWindowDataset(Dataset):
    """
    从多个股票的特征序列中，按滑窗生成样本：
      x: [L, F], y: [1]
    标准化：使用传入的全局 mu/sd（训练期统计量）
    """
    def __init__(
        self,
        frames: List[pd.DataFrame],
        start_date: str,
        end_date: str,
        mu: pd.Series,
        sd: pd.Series
    ):
        self.X = []
        self.y = []

        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        feat_cols = [c for c in frames[0].columns if c not in ["date", "y"]]

        # 对齐 mu/sd 的列顺序
        mu = mu[feat_cols]
        sd = sd[feat_cols]

        for df in frames:
            dff = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)].reset_index(drop=True)
            if len(dff) < cfg.lookback + 1:
                continue

            dff = dff.copy()
            dff[feat_cols] = (dff[feat_cols] - mu) / sd

            feats = dff[feat_cols].values.astype(np.float32)
            labels = dff["y"].values.astype(np.float32)

            for i in range(cfg.lookback - 1, len(dff)):
                x = feats[i - cfg.lookback + 1: i + 1]   # [L, F]
                t = labels[i]
                self.X.append(x)
                self.y.append([t])

        if not self.X:
            self.X = torch.empty((0, cfg.lookback, len(feat_cols)), dtype=torch.float32)
            self.y = torch.empty((0, 1), dtype=torch.float32)
        else:
            self.X = torch.tensor(np.stack(self.X), dtype=torch.float32)
            self.y = torch.tensor(np.stack(self.y), dtype=torch.float32)

    def __len__(self):
        return int(self.X.shape[0])

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# -------------------------
# Model (Transformer Encoder)
# -------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

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


def load_all_frames() -> List[pd.DataFrame]:
    # 优先 parquet
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
            if read_parquet:
                df = pd.read_parquet(fp)
            else:
                df = pd.read_csv(fp)

            need = {"date", "open", "high", "low", "close", "volume"}
            if not need.issubset(set(df.columns)):
                continue

            feats = make_features(df)

            # 至少要能形成一些窗口
            if len(feats) >= cfg.lookback + 10:
                frames.append(feats)
                usable += 1
        except Exception:
            continue

    if not frames:
        raise RuntimeError("All files failed to produce usable features. Check raw data columns/dates.")

    print(f"loaded files: {len(raw_files)} | usable frames: {usable}")
    return frames


def main():
    # 1) 读数据 & 特征工程（无标准化）
    frames = load_all_frames()

    # 2) 全局统计量（仅训练期）
    feat_cols = [c for c in frames[0].columns if c not in ["date", "y"]]
    global_mu, global_sd = compute_global_stats(frames, cfg.train_start, cfg.train_end, feat_cols)
    print("global stats computed on train period:", cfg.train_start, "->", cfg.train_end)

    # 3) 严格时间切分
    train_start = cfg.train_start
    train_end = cfg.train_end
    valid_start = (pd.to_datetime(train_end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    valid_end = cfg.valid_end

    train_ds = StockWindowDataset(frames, train_start, train_end, global_mu, global_sd)
    valid_ds = StockWindowDataset(frames, valid_start, valid_end, global_mu, global_sd)

    if len(train_ds) == 0:
        raise RuntimeError("Train dataset is empty. Check train_start/train_end and your data range.")
    if len(valid_ds) == 0:
        raise RuntimeError("Valid dataset is empty. Check valid_end or adjust split dates.")

    print(f"train samples: {len(train_ds)} | valid samples: {len(valid_ds)}")

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
        print(f"epoch={epoch} train_mse={train_mse:.6f} valid_mse={val_mse:.6f}")

        if val_mse < best:
            best = val_mse
            torch.save(
                {"model": model.state_dict(), "cfg": cfg.__dict__, "global_mu": global_mu.to_dict(), "global_sd": global_sd.to_dict()},
                cfg.best_ckpt
            )
            print(f"saved {cfg.best_ckpt}")


if __name__ == "__main__":
    main()
