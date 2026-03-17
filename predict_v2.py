# predict_v2.py
# 适配：train_v1.py(全局标准化) / train_v2.py(横截面y_cs + 全局标准化)
# 用法：python predict_v2.py

import glob
import os
from datetime import date

import numpy as np
import pandas as pd
import torch

from config import get_root_logger
from features import FEAT_COLS, make_features
from model import ReturnPredictor

# =========================
# Config
# =========================
LOOKBACK = 60
TOPK     = 10

# 数据新鲜度警告阈值（交易日，约 3 个自然日对应 2-3 个交易日）
STALE_DAYS_THRESHOLD = 5

BASE_DIR = os.path.dirname(__file__)
RAW_DIR  = os.path.join(BASE_DIR, "data_akshare", "raw")
OUT_DIR  = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# 你训练产物（v1 / v2 任选其一）
CKPT_CANDIDATES = [
    os.path.join(BASE_DIR, "checkpoints_v2", "best.pt"),
    os.path.join(BASE_DIR, "checkpoints",    "best.pt"),
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# 工具函数
# =========================
def load_ckpt() -> tuple[str, dict]:
    for p in CKPT_CANDIDATES:
        if os.path.exists(p):
            state = torch.load(p, map_location="cpu")
            return p, state
    raise FileNotFoundError(f"no checkpoint found in: {CKPT_CANDIDATES}")


def get_global_stats(state: dict) -> tuple[pd.Series, pd.Series]:
    if "global_mu" not in state or "global_sd" not in state:
        raise RuntimeError(
            "Checkpoint missing global_mu/global_sd. Use train_v1.py / train_v2.py that saves them."
        )
    mu = pd.Series(state["global_mu"]).reindex(FEAT_COLS)
    sd = pd.Series(state["global_sd"]).reindex(FEAT_COLS).replace(0, 1e-8).fillna(1e-8)
    if mu.isna().any():
        raise RuntimeError(f"global_mu missing some columns: {mu[mu.isna()].index.tolist()}")
    return mu, sd


@torch.no_grad()
def predict_one(model, df_feat: pd.DataFrame, mu: pd.Series, sd: pd.Series):
    if len(df_feat) < LOOKBACK:
        return None

    x = df_feat[FEAT_COLS].copy()
    x = (x - mu) / sd
    x = x.values.astype(np.float32)[-LOOKBACK:]  # [L, F]

    xt = torch.tensor(x).unsqueeze(0).to(DEVICE)  # [1, L, F]
    return float(model(xt).item())


def check_data_freshness(last_date: date, logger) -> None:
    """校验数据新鲜度，若数据超过阈值天数则发出警告"""
    today = pd.Timestamp.now().date()
    delta = (today - last_date).days
    if delta > STALE_DAYS_THRESHOLD:
        logger.warning(
            "数据新鲜度警告：最新数据日期为 %s，距今已 %d 天（>%d），"
            "建议先运行 update_recent_data.py 更新数据后再预测。",
            last_date, delta, STALE_DAYS_THRESHOLD,
        )


def main():
    logger = get_root_logger()

    ckpt_path, state = load_ckpt()
    logger.info("使用 checkpoint: %s", ckpt_path)

    mu, sd = get_global_stats(state)

    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.parquet")))
    if not files:
        files = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))
    if not files:
        raise RuntimeError(f"No parquet/csv files in {RAW_DIR}")

    num_features = len(FEAT_COLS)
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
            df_feat = make_features(df)   # 预测时不传 horizon，不计算标签
            pred = predict_one(model, df_feat, mu, sd)
            if pred is None:
                continue

            last_date = pd.to_datetime(df_feat["date"].iloc[-1]).date()

            # 数据新鲜度校验（只在第一只股票触发一次警告）
            if not rows:
                check_data_freshness(last_date, logger)

            # 近期波动率（vol20）
            vol20 = float(df_feat["vol20"].iloc[-1]) if "vol20" in df_feat.columns else np.nan

            rows.append({
                "stock_code":  code,
                "asof_date":   str(last_date),
                # 注意：如果是 train_v2.py（y_cs），pred 是"相对收益信号"，适合排序，不等同绝对收益%
                "pred_signal": pred,
                "vol20":       round(vol20, 6) if not np.isnan(vol20) else None,
            })
        except Exception as e:
            logger.warning("股票 %s 预测失败：%s", code, repr(e))

    if not rows:
        raise RuntimeError("No predictions generated. Check raw data / feature columns.")

    out = pd.DataFrame(rows).sort_values("pred_signal", ascending=False).reset_index(drop=True)

    # 信号排名百分位
    n = len(out)
    out["signal_rank"]       = range(1, n + 1)
    out["signal_percentile"] = (1 - (out["signal_rank"] - 1) / max(n - 1, 1)).round(4)

    top = out.head(TOPK).copy()

    today = pd.Timestamp.now().strftime("%Y%m%d")
    out_path_all = os.path.join(OUT_DIR, f"pred_all_{today}.csv")
    out_path_top = os.path.join(OUT_DIR, f"recommend_top{TOPK}_{today}.csv")

    out.to_csv(out_path_all, index=False, encoding="utf-8-sig")
    top.to_csv(out_path_top, index=False, encoding="utf-8-sig")

    logger.info("已保存预测结果:")
    logger.info("  全量 (%d 只): %s", len(out), out_path_all)
    logger.info("  Top%d: %s", TOPK, out_path_top)
    print("\nTopK 推荐:")
    print(top[["stock_code", "asof_date", "pred_signal", "vol20", "signal_percentile"]])


if __name__ == "__main__":
    main()
