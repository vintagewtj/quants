# fetch_akshare_a_daily.py
# 路线B：akshare + 本地存储 + 增量更新（在你实例上可用的 stock_zh_a_daily）
# 特点：
# - codes：akshare stock_info_a_code_name，过滤 ST
# - 行情：ak.stock_zh_a_daily(symbol="sz000001"/"sh600000")（注意：该接口常返回全历史，本地切片实现增量）
# - 存储：默认 Parquet（需要 pyarrow），否则自动降级 CSV
# - 断点续跑：done.txt
# - 失败记录：fail.txt
# - 过滤：停牌/低流动性/太短等

import os
import re
import time
import random
from datetime import datetime
from typing import Optional, Tuple

import pandas as pd
import akshare as ak

# -------------------------
# Storage / dirs
# -------------------------
DATA_DIR = "data_akshare"
RAW_DIR = os.path.join(DATA_DIR, "raw")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

DONE_FILE = os.path.join(CACHE_DIR, "done.txt")
FAIL_FILE = os.path.join(CACHE_DIR, "fail.txt")

# -------------------------
# Config
# -------------------------
DEFAULT_START = "20180101"  # 首次拉取起始（yyyymmdd）
END_DATE = datetime.now().strftime("%Y%m%d")

# 限速/重试
SLEEP_OK = 0.2
RETRIES = 4
BASE_BACKOFF = 1.0
MAX_BACKOFF = 60.0
BATCH_COOLDOWN_EVERY = 200
BATCH_COOLDOWN_RANGE = (3, 9)

# 质量过滤
MIN_BARS_TOTAL = 200
MAX_ZERO_VOL_RATIO = 0.25
RECENT_N = 60
RECENT_MAX_ZERO = 15
MIN_MEDIAN_VOL = 1e6  # 成交量（股）中位数阈值，可按你需要调

# 保存格式：优先 parquet
try:
    import pyarrow  # noqa: F401
    HAS_PARQUET = True
except Exception:
    HAS_PARQUET = False

SAVE_AS_PARQUET = True and HAS_PARQUET


# -------------------------
# Helpers
# -------------------------
def is_st_name(name: str) -> bool:
    if not name:
        return False
    n = str(name).upper()
    return ("*ST" in n) or ("ST" in n) or ("退" in n)


def to_ak_symbol(code: str) -> str:
    """A股 6位代码 -> akshare symbol: sh600000 / sz000001"""
    code = str(code).zfill(6)
    if code.startswith(("6", "9")):
        return f"sh{code}"
    return f"sz{code}"


def load_set(path: str) -> set:
    if not os.path.exists(path):
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return set(x.strip() for x in f if x.strip())


def append_line(path: str, s: str):
    with open(path, "a", encoding="utf-8") as f:
        f.write(s + "\n")


def get_codes_akshare() -> pd.DataFrame:
    """
    ak.stock_info_a_code_name() 常见返回列: code, name
    """
    df = ak.stock_info_a_code_name()
    df = df.rename(columns={"code": "stock_code", "name": "short_name"})
    df["stock_code"] = df["stock_code"].astype(str).str.strip()
    df["short_name"] = df["short_name"].astype(str).str.replace(r"\s+", "", regex=True)

    # 只保留 6 位数字
    df = df[df["stock_code"].apply(lambda x: bool(re.fullmatch(r"\d{6}", str(x))))].copy()
    # 剔除 ST / 退
    df = df[~df["short_name"].apply(is_st_name)].copy()
    return df


def read_existing(code: str) -> Optional[pd.DataFrame]:
    fp_pq = os.path.join(RAW_DIR, f"{code}.parquet")
    fp_csv = os.path.join(RAW_DIR, f"{code}.csv")

    if SAVE_AS_PARQUET and os.path.exists(fp_pq):
        return pd.read_parquet(fp_pq)
    if os.path.exists(fp_csv):
        return pd.read_csv(fp_csv)

    # 若之前存过 parquet 但当前环境没 pyarrow，读不了就算
    if os.path.exists(fp_pq):
        try:
            return pd.read_parquet(fp_pq)
        except Exception:
            return None
    return None


def last_date_of_existing(df: pd.DataFrame) -> Optional[str]:
    """返回本地数据中最后日期（yyyymmdd）"""
    if df is None or df.empty or "date" not in df.columns:
        return None
    d = pd.to_datetime(df["date"], errors="coerce").dropna()
    if d.empty:
        return None
    return d.max().strftime("%Y%m%d")


def pass_filters(df: pd.DataFrame) -> Tuple[bool, str]:
    if df is None or df.empty:
        return False, "empty"
    if len(df) < MIN_BARS_TOTAL:
        return False, f"too_short(len={len(df)})"

    need = {"date", "open", "high", "low", "close", "volume"}
    miss = need - set(df.columns)
    if miss:
        return False, f"missing_cols({sorted(list(miss))})"

    vol = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    zero_ratio = float((vol <= 0).mean())
    if zero_ratio > MAX_ZERO_VOL_RATIO:
        return False, f"too_many_zero_vol(ratio={zero_ratio:.2f})"

    med = float(vol[vol > 0].median()) if (vol > 0).any() else 0.0
    if med < MIN_MEDIAN_VOL:
        return False, f"median_vol_low(med={med:.0f})"

    recent = vol.tail(RECENT_N)
    if int((recent <= 0).sum()) > RECENT_MAX_ZERO:
        return False, f"recent_zero_vol_too_many({int((recent<=0).sum())}/{RECENT_N})"

    return True, "ok"


def fetch_daily_full_then_slice(code: str, start_yyyymmdd: str, end_yyyymmdd: str) -> Optional[pd.DataFrame]:
    """
    使用 ak.stock_zh_a_daily(symbol=sh/szxxxxxx)
    注意：该接口往往返回全历史，因此这里在本地做 start/end 切片，达到“增量”效果。
    返回列示例：
      date open high low close volume amount outstanding_share turnover
    """
    symbol = to_ak_symbol(code)
    df = ak.stock_zh_a_daily(symbol=symbol)
    if df is None or df.empty:
        return None

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    start_dt = pd.to_datetime(start_yyyymmdd, format="%Y%m%d", errors="coerce")
    end_dt = pd.to_datetime(end_yyyymmdd, format="%Y%m%d", errors="coerce")

    if pd.notna(start_dt):
        df = df[df["date"] >= start_dt]
    if pd.notna(end_dt):
        df = df[df["date"] <= end_dt]

    if df.empty:
        return None

    # 统一 date 存字符串
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    # 确保核心列为数值
    for c in ["open", "high", "low", "close", "volume", "amount", "turnover"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df.insert(0, "stock_code", code)
    return df


def save(code: str, df: pd.DataFrame):
    fp_pq = os.path.join(RAW_DIR, f"{code}.parquet")
    fp_csv = os.path.join(RAW_DIR, f"{code}.csv")
    if SAVE_AS_PARQUET:
        df.to_parquet(fp_pq, index=False)
    else:
        df.to_csv(fp_csv, index=False, encoding="utf-8")


def update_one(code: str, end_date: str) -> Tuple[bool, str]:
    existing = read_existing(code)
    last = last_date_of_existing(existing)

    if last is None:
        start = DEFAULT_START
    else:
        start = (pd.to_datetime(last) + pd.Timedelta(days=1)).strftime("%Y%m%d")

    # 已经最新
    if start > end_date:
        return True, "up_to_date"

    last_err = None
    for r in range(1, RETRIES + 1):
        try:
            df_new = fetch_daily_full_then_slice(code, start, end_date)
            if df_new is None or df_new.empty:
                raise RuntimeError("empty_return")

            if existing is not None and not existing.empty:
                df_all = pd.concat([existing, df_new], ignore_index=True)
                df_all = df_all.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
            else:
                df_all = df_new

            keep, reason = pass_filters(df_all)
            if not keep:
                return False, f"filtered:{reason}"

            save(code, df_all)
            return True, f"saved(len={len(df_all)})"

        except Exception as e:
            last_err = e
            backoff = min(MAX_BACKOFF, BASE_BACKOFF * (2 ** (r - 1)) + random.uniform(0, 0.6))
            print(f"    retry {r}/{RETRIES} code={code} err={repr(e)} sleep={backoff:.2f}s")
            time.sleep(backoff)

    return False, f"failed_after_retries(last_err={repr(last_err)})"


# -------------------------
# Main
# -------------------------
def main():
    print("akshare version:", ak.__version__)
    codes_df = get_codes_akshare()
    codes = codes_df["stock_code"].tolist()
    # codes = ["000001"]

    print("total codes(after ST filter):", len(codes))
    print("SAVE_AS_PARQUET:", SAVE_AS_PARQUET, "(pyarrow installed:", HAS_PARQUET, ")")

    done = load_set(DONE_FILE)
    print("already done:", len(done))

    ok = fail = 0
    time.sleep(1.5)

    for i, code in enumerate(codes, 1):
        if code in done:
            continue

        success, msg = update_one(code, END_DATE)
        if success:
            ok += 1
            append_line(DONE_FILE, code)
            print(f"[{i}/{len(codes)}] {code} ✅ {msg}")
        else:
            fail += 1
            append_line(FAIL_FILE, f"{code}\t{msg}")
            print(f"[{i}/{len(codes)}] {code} ❌ {msg}")

        time.sleep(SLEEP_OK)

        if i % BATCH_COOLDOWN_EVERY == 0:
            cool = random.uniform(*BATCH_COOLDOWN_RANGE)
            print(f"== batch cooldown {cool:.1f}s at i={i} ==")
            time.sleep(cool)

    print(f"DONE. ok={ok}, fail={fail}")


if __name__ == "__main__":
    main()
