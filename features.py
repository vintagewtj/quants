"""features.py
特征工程模块：包含特征列定义和特征构建函数。

- FEAT_COLS:    原始 10 个特征（v1/v2 向后兼容）
- FEAT_COLS_V2: 扩展特征集（含动量/波动率/流动性/市场状态等）
- make_features():    构建 FEAT_COLS
- make_features_v2():  构建 FEAT_COLS_V2
"""

from typing import List, Optional

import numpy as np
import pandas as pd


# -------------------------
# 特征列定义
# -------------------------

# v1/v2 通用的原始特征集（向后兼容，保持不变）
FEAT_COLS: List[str] = [
    "r1", "r5", "r10",
    "d_ma20", "d_ma60", "d_vol20",
    "hl_range",
    "vol10", "vol20", "vol60",
]

# 扩展特征集（含动量/波动率/流动性/市场状态）
FEAT_COLS_V2: List[str] = [
    # 收益率动量
    "r1", "r5", "r10", "r20", "r60",
    # 均线偏离
    "d_ma20", "d_ma60",
    # 量比
    "d_vol20",
    # 振幅
    "hl_range",
    # 波动率（收益率标准差）
    "vol10", "vol20", "vol60",
    # 量价高阶：VWAP 偏离
    "vwap_dev",
    # 动量扩展：最大回撤（20日）、连涨/连跌天数
    "drawdown_20", "up_streak", "down_streak",
    # 波动率改进：Garman-Klass、Parkinson
    "gk_vol20", "pk_vol20",
    # 流动性：Amihud 非流动性指标
    "amihud_20",
    # 市场状态：距离 52 周高低点
    "dist_52w_high", "dist_52w_low",
]


# -------------------------
# 特征构建
# -------------------------

def make_features(df: pd.DataFrame, horizon: Optional[int] = None) -> pd.DataFrame:
    """
    构建 FEAT_COLS 特征（v1/v2 通用的原始 10 个特征）。

    Args:
        df:      原始股价 DataFrame，至少含 date, open, high, low, close, volume
        horizon: 预测期（天）。非 None 时计算标签列 y_raw；None 时不计算（用于预测）

    Returns:
        DataFrame，列为 ["date"] + FEAT_COLS [+ "y_raw"]
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # ---- 收益率特征 ----
    df["r1"]  = df["close"].pct_change(1)
    df["r5"]  = df["close"].pct_change(5)
    df["r10"] = df["close"].pct_change(10)

    def ma(x, w):  return x.rolling(w).mean()
    def std(x, w): return x.rolling(w).std()

    # ---- 均线偏离 ----
    df["ma20"]  = ma(df["close"], 20)
    df["ma60"]  = ma(df["close"], 60)
    df["d_ma20"] = df["close"] / df["ma20"] - 1.0
    df["d_ma60"] = df["close"] / df["ma60"] - 1.0

    # ---- 量比 ----
    df["vol_ma20"] = ma(df["volume"], 20)
    df["d_vol20"]  = df["volume"] / df["vol_ma20"] - 1.0

    # ---- 振幅 ----
    df["hl_range"] = (df["high"] - df["low"]) / df["close"]

    # ---- 收益率波动率 ----
    df["vol10"] = std(df["r1"], 10)
    df["vol20"] = std(df["r1"], 20)
    df["vol60"] = std(df["r1"], 60)

    # ---- 标签（仅训练时） ----
    if horizon is not None:
        df["y_raw"] = df["close"].shift(-horizon) / df["close"] - 1.0
        required = FEAT_COLS + ["y_raw"]
    else:
        required = FEAT_COLS

    df = df.dropna(subset=required).reset_index(drop=True)
    return df[["date"] + required]


def make_features_v2(df: pd.DataFrame, horizon: Optional[int] = None) -> pd.DataFrame:
    """
    构建 FEAT_COLS_V2 扩展特征集（约 22 个因子）。

    新增特征：
      - 动量：r20, r60
      - VWAP 偏离
      - 最大回撤（20日）、连涨/连跌天数
      - Garman-Klass / Parkinson 20日波动率
      - Amihud 非流动性指标（20日均值）
      - 距离 52 周高低点百分比
    对所有特征做 winsorize（1%/99%）防止极端值。

    Args:
        df:      原始股价 DataFrame，至少含 date, open, high, low, close, volume
        horizon: 预测期（天）。非 None 时计算标签 y_raw；None 时不计算（用于预测）

    Returns:
        DataFrame，列为 ["date"] + FEAT_COLS_V2 [+ "y_raw"]
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    def ma(x, w):  return x.rolling(w).mean()
    def std(x, w): return x.rolling(w).std()

    # ---- 收益率动量 ----
    df["r1"]  = df["close"].pct_change(1)
    df["r5"]  = df["close"].pct_change(5)
    df["r10"] = df["close"].pct_change(10)
    df["r20"] = df["close"].pct_change(20)
    df["r60"] = df["close"].pct_change(60)

    # ---- 均线偏离 ----
    df["ma20"]   = ma(df["close"], 20)
    df["ma60"]   = ma(df["close"], 60)
    df["d_ma20"] = df["close"] / df["ma20"] - 1.0
    df["d_ma60"] = df["close"] / df["ma60"] - 1.0

    # ---- 量比 ----
    df["vol_ma20"] = ma(df["volume"], 20)
    df["d_vol20"]  = df["volume"] / df["vol_ma20"] - 1.0

    # ---- 振幅 ----
    df["hl_range"] = (df["high"] - df["low"]) / df["close"]

    # ---- 收益率波动率 ----
    df["vol10"] = std(df["r1"], 10)
    df["vol20"] = std(df["r1"], 20)
    df["vol60"] = std(df["r1"], 60)

    # ---- VWAP 偏离（典型价格代理 VWAP） ----
    # 典型价格 = (H + L + C) / 3
    df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3.0
    df["vwap_dev"] = df["close"] / df["typical_price"] - 1.0

    # ---- 最大回撤（过去 20 日） ----
    rolling_max20 = df["close"].rolling(20).max()
    df["drawdown_20"] = df["close"] / rolling_max20 - 1.0  # 负值，越小越差

    # ---- 连涨/连跌天数 ----
    streak_up = []
    streak_dn = []
    cnt_u, cnt_d = 0, 0
    for r1_val in df["r1"]:
        if pd.isna(r1_val):
            cnt_u = 0
            cnt_d = 0
        elif r1_val > 0:
            cnt_u += 1
            cnt_d = 0
        elif r1_val < 0:
            cnt_u = 0
            cnt_d += 1
        else:
            cnt_u = 0
            cnt_d = 0
        streak_up.append(cnt_u)
        streak_dn.append(cnt_d)
    df["up_streak"]   = streak_up
    df["down_streak"] = streak_dn

    # ---- Garman-Klass 波动率（20日） ----
    # GK_daily = 0.5 * ln(H/L)^2 - (2ln2 - 1) * ln(C/O)^2
    eps = 1e-10
    log_hl = np.log((df["high"] + eps) / (df["low"] + eps))
    log_co = np.log((df["close"] + eps) / (df["open"] + eps))
    gk_daily = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2
    gk_daily = gk_daily.clip(lower=0)
    df["gk_vol20"] = np.sqrt(gk_daily.rolling(20).mean())

    # ---- Parkinson 波动率（20日） ----
    # PK_daily = 1/(4*ln2) * ln(H/L)^2
    pk_daily = (1.0 / (4.0 * np.log(2))) * log_hl ** 2
    df["pk_vol20"] = np.sqrt(pk_daily.rolling(20).mean())

    # ---- Amihud 非流动性指标（20日均值） ----
    # Amihud = |return| / volume（成交量为 0 时跳过）
    amihud_daily = df["r1"].abs() / df["volume"].replace(0, np.nan)
    df["amihud_20"] = amihud_daily.rolling(20).mean()

    # ---- 距离 52 周高低点 ----
    high_52w = df["close"].rolling(252).max()
    low_52w  = df["close"].rolling(252).min()
    df["dist_52w_high"] = df["close"] / high_52w - 1.0   # 负值，越接近 0 越靠近新高
    df["dist_52w_low"]  = df["close"] / low_52w  - 1.0   # 正值，越接近 0 越靠近新低

    # ---- winsorize 防极端值（按 1%/99% clip） ----
    winsor_p = 0.01
    for col in FEAT_COLS_V2:
        if col in df.columns:
            lo = df[col].quantile(winsor_p)
            hi = df[col].quantile(1 - winsor_p)
            df[col] = df[col].clip(lo, hi)

    # ---- 标签（仅训练时） ----
    if horizon is not None:
        df["y_raw"] = df["close"].shift(-horizon) / df["close"] - 1.0
        required = FEAT_COLS_V2 + ["y_raw"]
    else:
        required = FEAT_COLS_V2

    df = df.dropna(subset=required).reset_index(drop=True)
    return df[["date"] + required]
