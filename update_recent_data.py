# update_recent_data.py
# 增量更新 data/raw/*.parquet（只拉最近几天或者从最后日期续拉）
# 数据源：akshare.stock_zh_a_daily（你已验证可用）
#
# 用法：
#   python update_recent_data.py --days 10
# 或者只更新某个代码：
#   python update_recent_data.py --days 10 --codes 000001,600078
#
# 逻辑：
# - 遍历 data/raw 中已有的股票文件（或你指定 codes）
# - 读取每只股票最后一个 date
# - 从 last_date-3 天开始拉（留buffer防止缺口），合并去重后写回
#
# 注意：
# - akshare 接口需要 symbol 形如 "sh600000" / "sz000001" / "bjxxxxxx"
# - 我这里用规则：6开头->sh，0/3开头->sz，8/4/9->bj（不覆盖的会跳过）

import os
import glob
import time
import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import akshare as ak


def code_to_symbol(code: str) -> str | None:
    c = str(code).strip()
    if len(c) != 6 or not c.isdigit():
        return None
    if c.startswith("6"):
        return "sh" + c
    if c.startswith(("0", "3")):
        return "sz" + c
    if c.startswith(("4", "8", "9")):
        return "bj" + c
    return None


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    把 akshare 输出统一为你训练需要的列：
      date open high low close volume
    """
    if df is None or len(df) == 0:
        return pd.DataFrame()

    # stock_zh_a_daily 通常列：date, open, high, low, close, volume, amount, ...
    if "date" not in df.columns:
        # 有些接口返回 "日期"
        if "日期" in df.columns:
            df = df.rename(columns={"日期": "date"})
        else:
            return pd.DataFrame()

    # 统一列名（中文列名兼容）
    rename_map = {}
    cn_map = {
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",
        "成交额": "amount",
    }
    for k, v in cn_map.items():
        if k in df.columns and v not in df.columns:
            rename_map[k] = v
    df = df.rename(columns=rename_map)

    need = ["date", "open", "high", "low", "close", "volume"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        # 缺少关键列就不要写入，避免污染
        return pd.DataFrame()

    df = df[need].copy()
    df["date"] = pd.to_datetime(df["date"])
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    return df


def fetch_daily_ak(code: str) -> pd.DataFrame:
    symbol = code_to_symbol(code)
    if symbol is None:
        return pd.DataFrame()
    # 你 probe 过：ak.stock_zh_a_daily("sz000001") 能成功
    df = ak.stock_zh_a_daily(symbol=symbol)
    return normalize_df(df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default=os.path.join(os.path.dirname(__file__), "data_akshare", "raw"))
    ap.add_argument("--days", type=int, default=10, help="update recent N days (used as a fallback range)")
    ap.add_argument("--codes", default="", help="comma separated stock codes; if empty, use existing files in raw_dir")
    ap.add_argument("--sleep", type=float, default=0.08, help="sleep seconds between requests")
    ap.add_argument("--max_retry", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(args.raw_dir, exist_ok=True)

    if args.codes.strip():
        codes = [c.strip() for c in args.codes.split(",") if c.strip()]
    else:
        files = sorted(glob.glob(os.path.join(args.raw_dir, "*.parquet")))
        codes = [os.path.splitext(os.path.basename(fp))[0] for fp in files]

    if not codes:
        raise RuntimeError(f"No codes found. Put parquet files into {args.raw_dir} or pass --codes")

    today = pd.Timestamp.now().normalize()
    fallback_start = today - pd.Timedelta(days=args.days + 5)

    ok = 0
    fail = 0

    for i, code in enumerate(codes, 1):
        fp = os.path.join(args.raw_dir, f"{code}.parquet")
        last_date = None
        if os.path.exists(fp):
            try:
                old = pd.read_parquet(fp)
                if "date" in old.columns and len(old) > 0:
                    old["date"] = pd.to_datetime(old["date"])
                    last_date = old["date"].max().normalize()
            except Exception:
                last_date = None

        # 从最后日期往后更新；留 3 天 buffer 防止缺口
        if last_date is not None:
            start_date = max(fallback_start, last_date - pd.Timedelta(days=3))
        else:
            start_date = fallback_start

        # 拉全量（接口不支持 start/end），再截取近段；但只写 merge 后增量，成本可接受
        # 如果你后面要更快，可以改成“只更新今天/最近两周”的缓存源
        err_last = None
        df_new = None
        for t in range(1, args.max_retry + 1):
            try:
                df_all = fetch_daily_ak(code)
                if df_all is None or len(df_all) == 0:
                    df_new = pd.DataFrame()
                else:
                    df_new = df_all[df_all["date"] >= start_date].copy()
                err_last = None
                break
            except Exception as e:
                err_last = repr(e)
                time.sleep(min(2.0, 0.2 * (2 ** (t - 1))))

        if err_last is not None:
            print(f"[{i}/{len(codes)}] {code} ❌ fetch_failed: {err_last}")
            fail += 1
            continue

        if df_new is None or len(df_new) == 0:
            print(f"[{i}/{len(codes)}] {code} ⛔ no_new_data (last_date={last_date})")
            time.sleep(args.sleep)
            continue

        # merge 写回
        if os.path.exists(fp):
            try:
                old = pd.read_parquet(fp)
                old["date"] = pd.to_datetime(old["date"])
                # old 也规范一下列
                keep_cols = ["date", "open", "high", "low", "close", "volume"]
                old = old[[c for c in keep_cols if c in old.columns]].copy()
                for c in ["open", "high", "low", "close", "volume"]:
                    if c in old.columns:
                        old[c] = pd.to_numeric(old[c], errors="coerce")
                merged = pd.concat([old, df_new], ignore_index=True)
            except Exception:
                merged = df_new.copy()
        else:
            merged = df_new.copy()

        merged = merged.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

        # 写 parquet（你环境 pyarrow 已装）
        merged.to_parquet(fp, index=False)

        ok += 1
        print(f"[{i}/{len(codes)}] {code} ✅ updated: +{len(df_new)} rows | last={merged['date'].iloc[-1].date()} | file={os.path.basename(fp)}")
        time.sleep(args.sleep)

    print(f"DONE. ok={ok}, fail={fail}, total={len(codes)}")


if __name__ == "__main__":
    main()
