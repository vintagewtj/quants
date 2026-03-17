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
    lr: float = 5e-5
    weight_decay: float = 5e-3
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
    容错：如果原始数据缺少 turnover/amount/outstanding_share，跳过对应特征
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    def ma(x, w): return x.rolling(w).mean()
    def std(x, w): return x.rolling(w).std()

    # --- 原有特征 ---
    df["r1"] = df["close"].pct_change(1)
    df["r5"] = df["close"].pct_change(5)
    df["r10"] = df["close"].pct_change(10)
    df["r20"] = df["close"].pct_change(20)
    df["r60"] = df["close"].pct_change(60)

    df["ma5"] = ma(df["close"], 5)
    df["ma10"] = ma(df["close"], 10)
    df["ma20"] = ma(df["close"], 20)
    df["ma60"] = ma(df["close"], 60)
    df["d_ma5"] = df["close"] / df["ma5"] - 1.0
    df["d_ma10"] = df["close"] / df["ma10"] - 1.0
    df["d_ma20"] = df["close"] / df["ma20"] - 1.0
    df["d_ma60"] = df["close"] / df["ma60"] - 1.0

    df["vol_ma5"] = ma(df["volume"], 5)
    df["vol_ma20"] = ma(df["volume"], 20)
    df["d_vol20"] = df["volume"] / df["vol_ma20"] - 1.0
    df["vol_ratio"] = df["volume"] / df["vol_ma5"] - 1.0

    df["hl_range"] = (df["high"] - df["low"]) / df["close"]
    df["vol10"] = std(df["r1"], 10)
    df["vol20"] = std(df["r1"], 20)
    df["vol60"] = std(df["r1"], 60)
    df["vol5"] = std(df["r1"], 5)

    # --- K线形态特征 ---
    hl = df["high"] - df["low"] + 1e-8
    df["oc_range"] = (df["close"] - df["open"]) / df["close"]
    df["upper_shadow"] = (df["high"] - df[["open", "close"]].max(axis=1)) / hl
    df["lower_shadow"] = (df[["open", "close"]].min(axis=1) - df["low"]) / hl
    df["close_pos"] = (df["close"] - df["low"]) / hl

    # --- KDJ RSV ---
    low9 = df["low"].rolling(9).min()
    high9 = df["high"].rolling(9).max()
    df["rsv"] = (df["close"] - low9) / (high9 - low9 + 1e-8)

    # --- 资金流量代理 ---
    df["mfi_proxy"] = df["r1"] * df["volume"]

    # --- 换手率特征（容错） ---
    if "turnover" in df.columns:
        df["turnover"] = pd.to_numeric(df["turnover"], errors="coerce")
        df["turn"] = df["turnover"]
        df["turn_ma5"] = ma(df["turnover"], 5)
        df["turn_ma20"] = ma(df["turnover"], 20)
        df["d_turn20"] = df["turnover"] / df["turn_ma20"] - 1.0

    # --- 成交额特征（容错） ---
    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
        df["amt_ma5"] = ma(df["amount"], 5)
        df["amt_ma20"] = ma(df["amount"], 20)
        df["d_amt20"] = df["amount"] / df["amt_ma20"] - 1.0
        df["amt_vol_ratio"] = df["amount"] / (df["volume"] + 1e-8)

    # --- 对数市值代理（容错） ---
    if "outstanding_share" in df.columns:
        df["outstanding_share"] = pd.to_numeric(df["outstanding_share"], errors="coerce")
        df["ln_mcap"] = np.log(df["close"] * df["outstanding_share"] + 1)

    # raw label
    df["y_raw"] = df["close"].shift(-cfg.horizon) / df["close"] - 1.0

    # 动态构建 feat_cols（只保留 DataFrame 中实际存在的列）
    candidate_cols = [
        "r1", "r5", "r10", "r20", "r60",
        "d_ma5", "d_ma10", "d_ma20", "d_ma60",
        "d_vol20", "vol_ratio",
        "hl_range", "vol5", "vol10", "vol20", "vol60",
        "oc_range", "upper_shadow", "lower_shadow", "close_pos",
        "rsv", "mfi_proxy",
        # 容错特征
        "turn", "turn_ma5", "turn_ma20", "d_turn20",
        "amt_ma5", "amt_ma20", "d_amt20", "amt_vol_ratio",
        "ln_mcap",
    ]
    feat_cols = [c for c in candidate_cols if c in df.columns]

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
    """
    # concat 做一次横截面处理，再拆回去（比逐日逐股循环快）
    lengths = [len(df) for df in frames]
    big = pd.concat(frames, axis=0, ignore_index=True)

    # winsorize per date
    def _winsor(s: pd.Series) -> pd.Series:
        lo = s.quantile(winsor_p)
        hi = s.quantile(1 - winsor_p)
        return s.clip(lo, hi)

    big["y_w"] = big.groupby("date")["y_raw"].transform(_winsor)
    big["y_cs"] = big["y_w"] - big.groupby("date")["y_w"].transform("mean")
    big = big.drop(columns=["y_w"])

    # 拆回 frames
    idx = 0
    for i, L in enumerate(lengths):
        frames[i] = big.iloc[idx: idx + L].reset_index(drop=True)
        idx += L


# -------------------------
# Dataset
# -------------------------
class StockWindowDatasetCS(Dataset):
    """
    生成样本，同时保留每条样本的 date（用于按日算 IC/RankIC）
    """
    def __init__(self, frames: List[pd.DataFrame], start_date: str, end_date: str, mu: pd.Series, sd: pd.Series):
        self.X = []
        self.y = []
        self.dates = []

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

            for i in range(cfg.lookback - 1, len(dff)):
                x = feats[i - cfg.lookback + 1: i + 1]
                t = labels[i]
                self.X.append(x)
                self.y.append([t])
                self.dates.append(dates[i])

        if not self.X:
            self.X = torch.empty((0, cfg.lookback, len(feat_cols)), dtype=torch.float32)
            self.y = torch.empty((0, 1), dtype=torch.float32)
            self.dates = np.array([], dtype="datetime64[ns]")
        else:
            self.X = torch.tensor(np.stack(self.X), dtype=torch.float32)
            self.y = torch.tensor(np.stack(self.y), dtype=torch.float32)
            self.dates = np.array(self.dates)

    def __len__(self):
        return int(self.X.shape[0])

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.dates[idx]


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
    def __init__(self, num_features: int, d_model: int = 32, nhead: int = 4, num_layers: int = 1, dropout: float = 0.3, dim_feedforward: int = 128):
        super().__init__()
        self.proj = nn.Linear(num_features, d_model)
        self.pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
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

    feat_cols = [c for c in frames[0].columns if c not in ["date", "y_raw", "y_cs"]]
    global_mu, global_sd = compute_global_stats(frames, cfg.train_start, cfg.train_end, feat_cols)
    print("global stats computed on train period:", cfg.train_start, "->", cfg.train_end)

    train_start = cfg.train_start
    train_end = cfg.train_end
    valid_start = (pd.to_datetime(train_end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    valid_end = cfg.valid_end

    train_ds = StockWindowDatasetCS(frames, train_start, train_end, global_mu, global_sd)
    valid_ds = StockWindowDatasetCS(frames, valid_start, valid_end, global_mu, global_sd)

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

    num_features = train_ds.X.shape[-1]
    model = ReturnPredictor(num_features).to(cfg.device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # patience=3 < early_stop_patience=5，让 LR 先降一次再判断是否早停
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)

    loss_fn = nn.HuberLoss(delta=0.02)  # delta 可调；0.02 对5日收益比较常用

    best_rankic = -1e9
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

        # scheduler 根据 valid_rankic 调整学习率
        scheduler.step(val_metrics["rankic"])

        lr_now = opt.param_groups[0]["lr"]
        print(
            f"epoch={epoch} lr={lr_now:.2e} "
            f"train_huber={train_loss:.6f} "
            f"valid_mse={val_metrics['mse']:.6f} "
            f"valid_ic={val_metrics['ic']:.4f} "
            f"★valid_rankic={val_metrics['rankic']:.4f}"
        )

        # 早停依据：valid_rankic（越高越好）
        if val_metrics["rankic"] > best_rankic + cfg.min_delta:
            best_rankic = val_metrics["rankic"]
            bad_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "cfg": cfg.__dict__,
                    "global_mu": global_mu.to_dict(),
                    "global_sd": global_sd.to_dict(),
                    "feat_cols": feat_cols,
                },
                cfg.best_ckpt,
            )
            print(f"★ 保存最佳 checkpoint (valid_rankic={best_rankic:.4f}): {cfg.best_ckpt}")
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.early_stop_patience:
                print(f"早停触发，epoch={epoch}，best valid_rankic={best_rankic:.4f}")
                break


if __name__ == "__main__":
    main()
