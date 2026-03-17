"""data_utils.py
数据工具模块：包含数据加载、全局统计量计算、Dataset 定义等。

导出：
  - pearson_corr / spearman_corr      相关性工具函数
  - load_all_frames()                 加载所有股票 DataFrame
  - compute_global_stats()            计算训练期全局 mean/std
  - attach_cross_sectional_label()    横截面标签变换（in-place）
  - StockWindowDataset                v1 数据集（绝对收益）
  - StockWindowDatasetCS              v2 数据集（横截面相对收益）
  - collate_with_dates()              带日期的 DataLoader collate 函数
"""

import glob
import logging
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from features import make_features

logger = logging.getLogger(__name__)


# -------------------------
# 相关性工具函数
# -------------------------

def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson 相关系数"""
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman 相关系数 = Pearson(rank(x), rank(y))"""
    sx = pd.Series(x).rank(method="average").to_numpy()
    sy = pd.Series(y).rank(method="average").to_numpy()
    if np.std(sx) < 1e-12 or np.std(sy) < 1e-12:
        return np.nan
    return float(np.corrcoef(sx, sy)[0, 1])


# -------------------------
# 数据加载
# -------------------------

def load_all_frames(
    data_glob_parquet: str,
    data_glob_csv: str,
    lookback: int,
    horizon: int,
    make_features_fn: Optional[Callable] = None,
) -> List[pd.DataFrame]:
    """
    加载所有股票数据文件，进行特征工程，返回可用的 DataFrame 列表。

    Args:
        data_glob_parquet: parquet 文件 glob 路径
        data_glob_csv:     CSV 文件 glob 路径
        lookback:          滑窗长度（用于过滤太短的数据）
        horizon:           预测期（天），传给 make_features_fn
        make_features_fn:  特征构建函数，默认使用 features.make_features
    """
    if make_features_fn is None:
        make_features_fn = make_features

    raw_files = glob.glob(data_glob_parquet)
    read_parquet = True
    if not raw_files:
        raw_files = glob.glob(data_glob_csv)
        read_parquet = False

    if not raw_files:
        raise RuntimeError(
            f"No data found.\nTried: {data_glob_parquet}\n   and: {data_glob_csv}\n"
            f"Put your files under data_akshare/raw/"
        )

    frames: List[pd.DataFrame] = []
    usable = 0

    for fp in raw_files:
        try:
            df = pd.read_parquet(fp) if read_parquet else pd.read_csv(fp)
            need = {"date", "open", "high", "low", "close", "volume"}
            if not need.issubset(set(df.columns)):
                logger.debug("跳过文件 %s：缺少必要列", fp)
                continue
            feats = make_features_fn(df, horizon=horizon)
            # 至少能形成一些窗口
            if len(feats) >= lookback + 10:
                frames.append(feats)
                usable += 1
        except Exception as e:
            logger.warning("加载文件 %s 失败：%s", fp, repr(e))

    if not frames:
        raise RuntimeError("All files failed to produce usable features. Check raw data columns/dates.")

    logger.info("已加载文件数: %d | 可用 frames: %d", len(raw_files), usable)
    return frames


# -------------------------
# 全局统计量
# -------------------------

def compute_global_stats(
    frames: List[pd.DataFrame],
    train_start: str,
    train_end: str,
    feat_cols: List[str],
) -> Tuple[pd.Series, pd.Series]:
    """
    只用训练期的数据计算全局 mean/std（跨股票、跨日期），避免数据泄漏。

    Returns:
        (mu, sd) — 两个 pd.Series，index 为 feat_cols
    """
    start_dt = pd.to_datetime(train_start)
    end_dt   = pd.to_datetime(train_end)

    buf: List[pd.DataFrame] = []
    for df in frames:
        dff = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]
        if len(dff) == 0:
            continue
        buf.append(dff[feat_cols])

    if not buf:
        raise RuntimeError(
            "No training-period data available to compute global stats. "
            "Check train_start/train_end."
        )

    all_feats = pd.concat(buf, axis=0)
    mu = all_feats.mean()
    sd = all_feats.std()
    # 防止除 0
    sd = sd.replace(0, 1e-8).fillna(1e-8)
    return mu, sd


# -------------------------
# 横截面标签变换
# -------------------------

def attach_cross_sectional_label(frames: List[pd.DataFrame], winsor_p: float) -> None:
    """
    原地修改 frames：增加列 y_cs（横截面去均值的相对收益）。

    处理步骤：
      1. 按 date 对 y_raw 做 winsorize（去掉首尾 winsor_p 分位的极端值）
      2. 按 date 对 winsorize 后的值去均值
    """
    lengths = [len(df) for df in frames]
    big = pd.concat(frames, axis=0, ignore_index=True)

    def _winsor(s: pd.Series) -> pd.Series:
        lo = s.quantile(winsor_p)
        hi = s.quantile(1 - winsor_p)
        return s.clip(lo, hi)

    big["y_w"]  = big.groupby("date")["y_raw"].transform(_winsor)
    big["y_cs"] = big["y_w"] - big.groupby("date")["y_w"].transform("mean")
    big = big.drop(columns=["y_w"])

    # 拆回各个 frame
    idx = 0
    for i, L in enumerate(lengths):
        frames[i] = big.iloc[idx: idx + L].reset_index(drop=True)
        idx += L


# -------------------------
# Dataset: v1（绝对收益）
# -------------------------

class StockWindowDataset(Dataset):
    """
    从多个股票的特征序列中，按滑窗生成样本：
      x: [L, F], y: [1]
    标准化：使用传入的全局 mu/sd（训练期统计量）。

    label_col: 标签列名（默认 "y_raw"，即未来 N 日原始收益率）。
      - v1 训练：使用 "y_raw"（绝对收益率）
      - 若需要横截面相对收益，请改用 StockWindowDatasetCS
    """

    def __init__(
        self,
        frames: List[pd.DataFrame],
        start_date: str,
        end_date: str,
        mu: pd.Series,
        sd: pd.Series,
        lookback: int,
        label_col: str = "y_raw",
    ):
        self.X: torch.Tensor
        self.y: torch.Tensor

        start_dt = pd.to_datetime(start_date)
        end_dt   = pd.to_datetime(end_date)

        # 特征列：排除 date 和所有标签列
        feat_cols = [c for c in frames[0].columns if c not in ("date", "y_raw", "y_cs")]
        mu = mu[feat_cols]
        sd = sd[feat_cols]

        X_list: List[np.ndarray] = []
        y_list: List[List[float]] = []

        for df in frames:
            dff = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)].reset_index(drop=True)
            if len(dff) < lookback + 1:
                continue

            dff = dff.copy()
            dff[feat_cols] = (dff[feat_cols] - mu) / sd

            feats  = dff[feat_cols].values.astype(np.float32)
            labels = dff[label_col].values.astype(np.float32)

            for i in range(lookback - 1, len(dff)):
                X_list.append(feats[i - lookback + 1: i + 1])
                y_list.append([labels[i]])

        if not X_list:
            self.X = torch.empty((0, lookback, len(feat_cols)), dtype=torch.float32)
            self.y = torch.empty((0, 1), dtype=torch.float32)
        else:
            self.X = torch.tensor(np.stack(X_list), dtype=torch.float32)
            self.y = torch.tensor(np.array(y_list), dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# -------------------------
# Dataset: v2（横截面相对收益）
# -------------------------

class StockWindowDatasetCS(Dataset):
    """
    生成样本时同时保留每条样本的 date（用于按日计算 IC/RankIC）。
    label_col 默认为 "y_cs"（横截面去均值后的相对收益）。
    """

    def __init__(
        self,
        frames: List[pd.DataFrame],
        start_date: str,
        end_date: str,
        mu: pd.Series,
        sd: pd.Series,
        lookback: int,
        label_col: str = "y_cs",
    ):
        self.X: torch.Tensor
        self.y: torch.Tensor
        self.dates: np.ndarray

        start_dt = pd.to_datetime(start_date)
        end_dt   = pd.to_datetime(end_date)

        feat_cols = [c for c in frames[0].columns if c not in ("date", "y_raw", "y_cs")]
        mu = mu[feat_cols]
        sd = sd[feat_cols]

        X_list: List[np.ndarray] = []
        y_list: List[List[float]] = []
        d_list: List = []

        for df in frames:
            dff = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)].reset_index(drop=True)
            if len(dff) < lookback + 1:
                continue

            dff = dff.copy()
            dff[feat_cols] = (dff[feat_cols] - mu) / sd

            feats  = dff[feat_cols].values.astype(np.float32)
            labels = dff[label_col].values.astype(np.float32)
            dates  = dff["date"].values  # numpy datetime64

            for i in range(lookback - 1, len(dff)):
                X_list.append(feats[i - lookback + 1: i + 1])
                y_list.append([labels[i]])
                d_list.append(dates[i])

        if not X_list:
            self.X     = torch.empty((0, lookback, len(feat_cols)), dtype=torch.float32)
            self.y     = torch.empty((0, 1), dtype=torch.float32)
            self.dates = np.array([], dtype="datetime64[ns]")
        else:
            self.X     = torch.tensor(np.stack(X_list), dtype=torch.float32)
            self.y     = torch.tensor(np.array(y_list), dtype=torch.float32)
            self.dates = np.array(d_list)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx], self.dates[idx]


def collate_with_dates(batch):
    """自定义 collate 函数，支持 (x, y, date) 格式的 batch"""
    xs, ys, ds = zip(*batch)
    x = torch.stack(xs, dim=0)
    y = torch.stack(ys, dim=0)
    d = np.array(ds)
    return x, y, d
