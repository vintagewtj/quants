# predict_v2.py
# 适配：train.py(全局标准化) / train_v2.py(横截面y_cs + 全局标准化)
# 用法：python predict_v2.py
import os
import glob
import math
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

# =========================
# Config
# =========================
LOOKBACK = 60
TOPK = 10

BASE_DIR = os.path.dirname(__file__)
RAW_DIR = os.path.join(BASE_DIR, "data_akshare", "raw")   # ✅ 你的数据目录
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# 你训练产物（v1 / v2 任选其一）
# - v1: checkpoints/best.pt
# - v2: checkpoints_v2/best.pt
CKPT_CANDIDATES = [
    os.path.join(BASE_DIR, "checkpoints_v2", "best.pt"),
    os.path.join(BASE_DIR, "checkpoints", "best.pt"),
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# Model (必须与训练一致)
# =========================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class ReturnPredictor(nn.Module):
    def __init__(self, num_features, d_model=32, nhead=4, num_layers=1, dropout=0.3, dim_feedforward=128):
        super().__init__()
        self.proj = nn.Linear(num_features, d_model)
        self.pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        h = self.proj(x)
        h = self.pos(h)
        h = self.enc(h)
        return self.head(h[:, -1, :])


# =========================
# Feature engineering (与训练一致：不做单股zscore)
# 容错：如果原始数据缺少 turnover/amount/outstanding_share，跳过对应特征
# =========================
_DEFAULT_FEAT_COLS = [
    "r1", "r5", "r10", "r20", "r60",
    "d_ma5", "d_ma10", "d_ma20", "d_ma60",
    "d_vol20", "vol_ratio",
    "hl_range", "vol5", "vol10", "vol20", "vol60",
    "oc_range", "upper_shadow", "lower_shadow", "close_pos",
    "rsv", "mfi_proxy",
]

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    def ma(x, w): return x.rolling(w).mean()
    def std(x, w): return x.rolling(w).std()

    df["r1"]  = df["close"].pct_change(1)
    df["r5"]  = df["close"].pct_change(5)
    df["r10"] = df["close"].pct_change(10)
    df["r20"] = df["close"].pct_change(20)
    df["r60"] = df["close"].pct_change(60)

    df["ma5"]  = ma(df["close"], 5)
    df["ma10"] = ma(df["close"], 10)
    df["ma20"] = ma(df["close"], 20)
    df["ma60"] = ma(df["close"], 60)
    df["d_ma5"]  = df["close"] / df["ma5"]  - 1.0
    df["d_ma10"] = df["close"] / df["ma10"] - 1.0
    df["d_ma20"] = df["close"] / df["ma20"] - 1.0
    df["d_ma60"] = df["close"] / df["ma60"] - 1.0

    df["vol_ma5"]  = ma(df["volume"], 5)
    df["vol_ma20"] = ma(df["volume"], 20)
    df["d_vol20"]  = df["volume"] / df["vol_ma20"] - 1.0
    df["vol_ratio"] = df["volume"] / df["vol_ma5"] - 1.0

    df["hl_range"] = (df["high"] - df["low"]) / df["close"]
    df["vol5"]  = std(df["r1"], 5)
    df["vol10"] = std(df["r1"], 10)
    df["vol20"] = std(df["r1"], 20)
    df["vol60"] = std(df["r1"], 60)

    hl = df["high"] - df["low"] + 1e-8
    df["oc_range"]     = (df["close"] - df["open"]) / df["close"]
    df["upper_shadow"] = (df["high"] - df[["open", "close"]].max(axis=1)) / hl
    df["lower_shadow"] = (df[["open", "close"]].min(axis=1) - df["low"]) / hl
    df["close_pos"]    = (df["close"] - df["low"]) / hl

    low9  = df["low"].rolling(9).min()
    high9 = df["high"].rolling(9).max()
    df["rsv"] = (df["close"] - low9) / (high9 - low9 + 1e-8)

    df["mfi_proxy"] = df["r1"] * df["volume"]

    if "turnover" in df.columns:
        df["turnover"] = pd.to_numeric(df["turnover"], errors="coerce")
        df["turn"]      = df["turnover"]
        df["turn_ma5"]  = ma(df["turnover"], 5)
        df["turn_ma20"] = ma(df["turnover"], 20)
        df["d_turn20"]  = df["turnover"] / df["turn_ma20"] - 1.0

    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
        df["amt_ma5"]      = ma(df["amount"], 5)
        df["amt_ma20"]     = ma(df["amount"], 20)
        df["d_amt20"]      = df["amount"] / df["amt_ma20"] - 1.0
        df["amt_vol_ratio"] = df["amount"] / (df["volume"] + 1e-8)

    if "outstanding_share" in df.columns:
        df["outstanding_share"] = pd.to_numeric(df["outstanding_share"], errors="coerce")
        df["ln_mcap"] = np.log(df["close"] * df["outstanding_share"] + 1)

    return df


def load_ckpt() -> tuple[str, dict]:
    for p in CKPT_CANDIDATES:
        if os.path.exists(p):
            state = torch.load(p, map_location="cpu")
            return p, state
    raise FileNotFoundError(f"no checkpoint found in: {CKPT_CANDIDATES}")


def get_global_stats(state: dict, feat_cols: list) -> tuple[pd.Series, pd.Series]:
    if "global_mu" not in state or "global_sd" not in state:
        raise RuntimeError("Checkpoint missing global_mu/global_sd. Use the train.py / train_v2.py that saves them.")
    mu = pd.Series(state["global_mu"])
    sd = pd.Series(state["global_sd"])
    # 对齐列顺序
    mu = mu.reindex(feat_cols)
    sd = sd.reindex(feat_cols).replace(0, 1e-8).fillna(1e-8)
    if mu.isna().any():
        raise RuntimeError(f"global_mu missing some columns: {mu[mu.isna()].index.tolist()}")
    return mu, sd


@torch.no_grad()
def predict_one(model: nn.Module, df_feat: pd.DataFrame, mu: pd.Series, sd: pd.Series, feat_cols: list) -> float | None:
    if len(df_feat) < LOOKBACK:
        return None

    x = df_feat[feat_cols].copy()
    x = (x - mu) / sd
    x = x.values.astype(np.float32)[-LOOKBACK:]  # [L, F]

    xt = torch.tensor(x).unsqueeze(0).to(DEVICE)  # [1, L, F]
    y = model(xt).item()
    return float(y)


def main():
    ckpt_path, state = load_ckpt()
    print("using ckpt:", ckpt_path)

    # 从 checkpoint 读取 feat_cols（兼容旧版 checkpoint）
    feat_cols = state.get("feat_cols", _DEFAULT_FEAT_COLS)

    mu, sd = get_global_stats(state, feat_cols)

    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.parquet")))
    if not files:
        files = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))
    if not files:
        raise RuntimeError(f"No parquet/csv files in {RAW_DIR}")

    num_features = len(feat_cols)

    model = ReturnPredictor(num_features).to(DEVICE)
    model.load_state_dict(state["model"])
    model.eval()

    rows = []
    for fp in files:
        code = os.path.splitext(os.path.basename(fp))[0]
        try:
            df = pd.read_parquet(fp) if fp.endswith(".parquet") else pd.read_csv(fp)
            if "date" not in df.columns:
                continue
            df_feat = make_features(df)
            # 确认所有 feat_cols 均存在于 df_feat，缺列则跳过该文件
            if not all(c in df_feat.columns for c in feat_cols):
                continue
            df_feat = df_feat.dropna(subset=feat_cols).reset_index(drop=True)
            pred = predict_one(model, df_feat, mu, sd, feat_cols)
            if pred is None:
                continue

            last_date = pd.to_datetime(df_feat["date"].iloc[-1]).date()
            rows.append({
                "stock_code": code,
                "asof_date": str(last_date),
                # 注意：如果是 train_v2.py（y_cs），pred 是“相对收益信号”，适合排序，不等同绝对收益%
                "pred_signal": pred,
            })
        except Exception:
            continue

    if not rows:
        raise RuntimeError("No predictions generated. Check raw data / feature columns.")

    out = pd.DataFrame(rows).sort_values("pred_signal", ascending=False).reset_index(drop=True)
    top = out.head(TOPK).copy()

    today = pd.Timestamp.now().strftime("%Y%m%d")
    out_path_all = os.path.join(OUT_DIR, f"pred_all_{today}.csv")
    out_path_top = os.path.join(OUT_DIR, f"recommend_top{TOPK}_{today}.csv")

    out.to_csv(out_path_all, index=False, encoding="utf-8-sig")
    top.to_csv(out_path_top, index=False, encoding="utf-8-sig")

    print("Saved:")
    print("  ", out_path_all)
    print("  ", out_path_top)
    print("\nTopK:")
    print(top[["stock_code", "asof_date", "pred_signal"]])


if __name__ == "__main__":
    main()
