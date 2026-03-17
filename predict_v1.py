# predict_v1.py
# 适配 v1 train_v1.py（训练期全局统计量标准化，label=未来N日收益 y_raw）
# 用法：python predict_v1.py

import glob
import os

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

BASE_DIR = os.path.dirname(__file__)

# 你的数据目录（按你当前 akshare 抓取脚本）
RAW_DIR = os.path.join(BASE_DIR, "data_akshare", "raw")

# v1 checkpoint
CKPT = os.path.join(BASE_DIR, "checkpoints", "best.pt")

OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# 工具函数
# =========================
def load_global_stats(state: dict) -> tuple[pd.Series, pd.Series]:
    if "global_mu" not in state or "global_sd" not in state:
        raise RuntimeError("checkpoint missing global_mu/global_sd (train_v1.py should have saved them)")
    mu = pd.Series(state["global_mu"]).reindex(FEAT_COLS)
    sd = pd.Series(state["global_sd"]).reindex(FEAT_COLS)
    sd = sd.replace(0, 1e-8).fillna(1e-8)
    if mu.isna().any():
        raise RuntimeError(f"global_mu missing cols: {mu[mu.isna()].index.tolist()}")
    return mu, sd


@torch.no_grad()
def predict_one(model, df_feat: pd.DataFrame, mu: pd.Series, sd: pd.Series):
    if len(df_feat) < LOOKBACK:
        return None

    x = df_feat[FEAT_COLS].copy()
    x = (x - mu) / sd
    x = x.values.astype(np.float32)[-LOOKBACK:]   # [L, F]
    xt = torch.tensor(x).unsqueeze(0).to(DEVICE)  # [1, L, F]
    return float(model(xt).item())


def main():
    logger = get_root_logger()

    if not os.path.exists(CKPT):
        raise FileNotFoundError(f"checkpoint not found: {CKPT}")

    # 读 ckpt
    state = torch.load(CKPT, map_location="cpu")
    mu, sd = load_global_stats(state)

    # 找数据
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.parquet")))
    if not files:
        files = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))
    if not files:
        raise RuntimeError(f"No parquet/csv files in {RAW_DIR}")

    # load model
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
            rows.append({
                "stock_code":       code,
                "asof_date":        str(last_date),
                "pred_return_Nd":   pred,
                "pred_return_pct":  pred * 100.0,
            })
        except Exception as e:
            logger.warning("股票 %s 预测失败：%s", code, repr(e))

    if not rows:
        raise RuntimeError("No predictions generated.")

    out = pd.DataFrame(rows).sort_values("pred_return_Nd", ascending=False).reset_index(drop=True)
    top = out.head(TOPK).copy()

    today = pd.Timestamp.now().strftime("%Y%m%d")
    out_path_all = os.path.join(OUT_DIR, f"pred_all_{today}.csv")
    out_path_top = os.path.join(OUT_DIR, f"recommend_top{TOPK}_{today}.csv")

    out.to_csv(out_path_all, index=False, encoding="utf-8-sig")
    top.to_csv(out_path_top, index=False, encoding="utf-8-sig")

    logger.info("已保存预测结果:")
    logger.info("  全量: %s", out_path_all)
    logger.info("  Top%d: %s", TOPK, out_path_top)
    print("\nTopK:")
    print(top[["stock_code", "asof_date", "pred_return_pct"]])


if __name__ == "__main__":
    main()
