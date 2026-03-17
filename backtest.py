"""backtest.py
回测框架：基于预测信号的 Top-K 选股策略回测。

使用方法：
  python backtest.py --pred_dir outputs/ --price_dir data_akshare/raw/
  python backtest.py --help

功能：
  - 读取预测信号文件（CSV），按日期生成 Top-K 持仓
  - 读取历史价格数据，计算每日 PnL
  - 考虑交易成本（佣金 + 滑点）
  - 计算核心指标：累积收益、年化收益、年化波动率、Sharpe Ratio、最大回撤、
                   Calmar Ratio、胜率、换手率
  - 等权基准对比
  - 输出 CSV + 打印摘要
"""

import argparse
import glob
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# -------------------------
# 默认参数
# -------------------------
TOPK        = 10        # 每期持有的股票数量
COMMISSION  = 0.001     # 单边佣金率（0.1%）
SLIPPAGE    = 0.001     # 单边滑点率（0.1%）
COST_RATE   = COMMISSION + SLIPPAGE  # 单边总交易成本


# -------------------------
# 指标计算
# -------------------------

def calc_metrics(daily_ret: pd.Series, rf: float = 0.0) -> Dict[str, float]:
    """
    计算策略核心指标。

    Args:
        daily_ret: 每日收益率 Series（index 为日期）
        rf:        无风险收益率（日化，默认 0）

    Returns:
        包含各项指标的字典
    """
    if len(daily_ret) == 0:
        return {}

    cum_ret = (1 + daily_ret).cumprod()
    total_ret   = float(cum_ret.iloc[-1] - 1)
    n_days      = len(daily_ret)
    ann_ret     = float((1 + total_ret) ** (252 / n_days) - 1)
    ann_vol     = float(daily_ret.std() * np.sqrt(252))
    sharpe      = float((ann_ret - rf) / ann_vol) if ann_vol > 1e-12 else np.nan

    # 最大回撤
    roll_max = cum_ret.cummax()
    drawdown = cum_ret / roll_max - 1
    max_dd   = float(drawdown.min())

    calmar = float(ann_ret / abs(max_dd)) if abs(max_dd) > 1e-12 else np.nan

    # 胜率（日度）
    win_rate = float((daily_ret > 0).mean())

    return {
        "total_ret":  round(total_ret,  4),
        "ann_ret":    round(ann_ret,    4),
        "ann_vol":    round(ann_vol,    4),
        "sharpe":     round(sharpe,     4) if not np.isnan(sharpe)  else np.nan,
        "max_dd":     round(max_dd,     4),
        "calmar":     round(calmar,     4) if not np.isnan(calmar)  else np.nan,
        "win_rate":   round(win_rate,   4),
        "n_days":     n_days,
    }


def print_metrics(name: str, metrics: Dict[str, float]) -> None:
    """打印指标摘要"""
    print(f"\n{'='*50}")
    print(f"  策略: {name}")
    print(f"{'='*50}")
    print(f"  累积收益率:   {metrics.get('total_ret', 'N/A'):.2%}")
    print(f"  年化收益率:   {metrics.get('ann_ret',   'N/A'):.2%}")
    print(f"  年化波动率:   {metrics.get('ann_vol',   'N/A'):.2%}")
    print(f"  Sharpe Ratio: {metrics.get('sharpe',    'N/A'):.4f}")
    print(f"  最大回撤:     {metrics.get('max_dd',    'N/A'):.2%}")
    print(f"  Calmar Ratio: {metrics.get('calmar',    'N/A'):.4f}")
    print(f"  胜率（日度）: {metrics.get('win_rate',  'N/A'):.2%}")
    print(f"  回测天数:     {metrics.get('n_days',    'N/A')}")
    print(f"{'='*50}\n")


# -------------------------
# 价格数据加载
# -------------------------

def load_price_data(price_dir: str) -> Dict[str, pd.DataFrame]:
    """
    加载价格目录下所有 parquet/csv 文件，返回 {stock_code: df} 字典。
    每个 df 含 date, close（至少）。
    """
    files = sorted(glob.glob(os.path.join(price_dir, "*.parquet")))
    use_parquet = True
    if not files:
        files = sorted(glob.glob(os.path.join(price_dir, "*.csv")))
        use_parquet = False

    if not files:
        raise RuntimeError(f"No price data files found in: {price_dir}")

    prices: Dict[str, pd.DataFrame] = {}
    for fp in files:
        code = os.path.splitext(os.path.basename(fp))[0]
        try:
            df = pd.read_parquet(fp) if use_parquet else pd.read_csv(fp)
            df["date"]  = pd.to_datetime(df["date"], errors="coerce")
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
            df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
            if len(df) > 0:
                prices[code] = df[["date", "close"]]
        except Exception as e:
            logger.warning("加载价格文件 %s 失败：%s", fp, repr(e))

    logger.info("已加载价格数据: %d 只股票", len(prices))
    return prices


# -------------------------
# 预测信号加载
# -------------------------

def load_signals(pred_dir: str, signal_col: str = "pred_signal") -> pd.DataFrame:
    """
    加载预测信号目录下所有 CSV，合并为宽表。
    支持两种格式：
      1. 含 [stock_code, asof_date, pred_signal] 的长表
      2. 单次预测 CSV（只有一个 asof_date）

    Returns:
        DataFrame，列为 stock_code，index 为 asof_date（日期）
    """
    csv_files = sorted(glob.glob(os.path.join(pred_dir, "pred_all_*.csv")))
    if not csv_files:
        csv_files = sorted(glob.glob(os.path.join(pred_dir, "*.csv")))
    if not csv_files:
        raise RuntimeError(f"No prediction CSV files found in: {pred_dir}")

    dfs: List[pd.DataFrame] = []
    for fp in csv_files:
        try:
            df = pd.read_csv(fp, encoding="utf-8-sig")
            if signal_col not in df.columns:
                # 兼容 v1 格式（pred_return_Nd）
                for alt in ["pred_return_Nd", "pred_return_pct"]:
                    if alt in df.columns:
                        df = df.rename(columns={alt: signal_col})
                        break
            if "asof_date" not in df.columns or "stock_code" not in df.columns:
                continue
            if signal_col not in df.columns:
                continue
            dfs.append(df[["stock_code", "asof_date", signal_col]])
        except Exception as e:
            logger.warning("加载信号文件 %s 失败：%s", fp, repr(e))

    if not dfs:
        raise RuntimeError("No valid signal CSVs loaded.")

    long = pd.concat(dfs, axis=0, ignore_index=True)
    long["asof_date"] = pd.to_datetime(long["asof_date"], errors="coerce")
    long = long.dropna(subset=["asof_date"]).sort_values("asof_date")

    # 转为宽表：行=日期, 列=stock_code, 值=信号
    wide = long.pivot_table(index="asof_date", columns="stock_code", values=signal_col, aggfunc="last")
    logger.info("信号覆盖日期: %d 个, 股票: %d 只", len(wide), len(wide.columns))
    return wide


# -------------------------
# 构建持仓
# -------------------------

def build_holdings(
    signal_wide: pd.DataFrame,
    topk: int,
) -> pd.DataFrame:
    """
    根据信号宽表，每个日期选择 Top-K 股票等权持仓。

    Returns:
        DataFrame，index=date, columns=stock_code, 值=持仓权重（等权 1/K 或 0）
    """
    weights = signal_wide.apply(
        lambda row: _topk_weights(row, topk),
        axis=1,
    )
    return weights


def _topk_weights(row: pd.Series, topk: int) -> pd.Series:
    valid = row.dropna()
    if len(valid) == 0:
        return pd.Series(0.0, index=row.index)
    top_codes = valid.nlargest(topk).index
    w = pd.Series(0.0, index=row.index)
    w[top_codes] = 1.0 / min(topk, len(top_codes))
    return w


# -------------------------
# 日度收益计算
# -------------------------

def compute_daily_returns(
    holdings: pd.DataFrame,
    prices: Dict[str, pd.DataFrame],
    cost_rate: float = COST_RATE,
    rebalance_lag: int = 1,
) -> pd.DataFrame:
    """
    根据每日持仓权重 + 价格数据，计算策略每日收益。

    Args:
        holdings:      持仓权重宽表（index=signal_date, columns=stock_code）
        prices:        {stock_code: df(date, close)} 字典
        cost_rate:     单边交易成本（佣金+滑点）
        rebalance_lag: 信号日 -> 实际买入的滞后天数（默认 1，即 T+1 执行）

    Returns:
        DataFrame，含 [date, strategy_ret, benchmark_ret, turnover]
    """
    # 构建价格宽表
    price_list: List[pd.DataFrame] = []
    for code, df in prices.items():
        tmp = df[["date", "close"]].copy()
        tmp = tmp.rename(columns={"close": code})
        price_list.append(tmp.set_index("date"))

    if not price_list:
        raise RuntimeError("No price data available for return calculation.")

    price_wide = pd.concat(price_list, axis=1).sort_index()

    # 日度收益率宽表
    ret_wide = price_wide.pct_change()

    # 对齐持仓日期与价格日期
    all_dates = price_wide.index
    signal_dates = sorted(holdings.index)

    rows = []
    prev_weights: Optional[pd.Series] = None

    for i, sig_date in enumerate(signal_dates):
        # 实际执行日（信号日 + rebalance_lag 个交易日）
        future_dates = all_dates[all_dates > sig_date]
        if len(future_dates) < rebalance_lag:
            continue
        exec_date = future_dates[rebalance_lag - 1]

        # 下一个信号日的执行日（用于计算持仓期收益）
        if i + 1 < len(signal_dates):
            next_exec_dates = all_dates[all_dates > signal_dates[i + 1]]
            if len(next_exec_dates) < rebalance_lag:
                continue
            next_exec_date = next_exec_dates[rebalance_lag - 1]
        else:
            # 最后一个信号，持仓到最后一个价格日
            next_exec_date = all_dates[-1]

        # 持仓期内每日收益
        hold_period = all_dates[(all_dates >= exec_date) & (all_dates < next_exec_date)]
        if len(hold_period) == 0:
            continue

        w = holdings.loc[sig_date]
        w = w.reindex(ret_wide.columns).fillna(0.0)

        # 换手率（与上一期相比）
        if prev_weights is not None:
            turnover = float((w - prev_weights).abs().sum() / 2)
        else:
            turnover = float(w.abs().sum() / 2)
        prev_weights = w.copy()

        # 单边交易成本（换手率 × 单边成本率 × 2 = 双边）
        cost = turnover * cost_rate * 2

        for dt in hold_period:
            if dt not in ret_wide.index:
                continue
            day_ret = float((w * ret_wide.loc[dt].fillna(0.0)).sum())
            # 只在换仓日扣除交易成本
            actual_ret = day_ret - (cost if dt == exec_date else 0.0)

            # 等权基准：持仓期所有股票等权持有
            n_valid = (ret_wide.loc[dt].notna()).sum()
            bm_ret  = float(ret_wide.loc[dt].mean()) if n_valid > 0 else 0.0

            rows.append({
                "date":          dt,
                "strategy_ret":  actual_ret,
                "benchmark_ret": bm_ret,
                "turnover":      turnover if dt == exec_date else 0.0,
            })

    if not rows:
        raise RuntimeError("No return data computed. Check signal dates vs price data dates.")

    result = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    # 去重（同日期取均值，防止日期重复）
    result = result.groupby("date").mean().reset_index()
    return result


# -------------------------
# 主回测流程
# -------------------------

def run_backtest(
    pred_dir: str,
    price_dir: str,
    output_dir: str,
    topk: int = TOPK,
    cost_rate: float = COST_RATE,
    rebalance_lag: int = 1,
    signal_col: str = "pred_signal",
) -> pd.DataFrame:
    """
    完整回测流程：加载信号 → 构建持仓 → 计算收益 → 计算指标 → 保存结果。

    Returns:
        每日收益 DataFrame
    """
    os.makedirs(output_dir, exist_ok=True)

    logger.info("加载价格数据...")
    prices = load_price_data(price_dir)

    logger.info("加载预测信号...")
    signal_wide = load_signals(pred_dir, signal_col=signal_col)

    logger.info("构建 Top-%d 持仓...", topk)
    holdings = build_holdings(signal_wide, topk=topk)

    logger.info("计算每日收益（含交易成本 %.2f%%）...", cost_rate * 100)
    daily = compute_daily_returns(holdings, prices, cost_rate=cost_rate, rebalance_lag=rebalance_lag)

    # ---- 计算策略指标 ----
    strat_metrics = calc_metrics(daily["strategy_ret"])
    bm_metrics    = calc_metrics(daily["benchmark_ret"])

    # 换手率（年化）
    avg_turnover     = float(daily["turnover"].mean())
    ann_turnover     = avg_turnover * 252
    strat_metrics["avg_daily_turnover"] = round(avg_turnover,  4)
    strat_metrics["ann_turnover"]       = round(ann_turnover,  4)

    print_metrics(f"Top-{topk} 策略（含成本）", strat_metrics)
    print_metrics("等权基准", bm_metrics)

    # 超额收益
    daily["excess_ret"]    = daily["strategy_ret"] - daily["benchmark_ret"]
    excess_metrics         = calc_metrics(daily["excess_ret"])
    print_metrics("超额收益", excess_metrics)

    # ---- 保存结果 ----
    out_daily  = os.path.join(output_dir, "backtest_daily.csv")
    out_cumret = os.path.join(output_dir, "backtest_cumret.csv")
    out_metric = os.path.join(output_dir, "backtest_metrics.csv")

    daily.to_csv(out_daily, index=False, encoding="utf-8-sig")

    # 累积收益曲线
    cum = daily[["date"]].copy()
    cum["strategy_cum"]  = (1 + daily["strategy_ret"]).cumprod()
    cum["benchmark_cum"] = (1 + daily["benchmark_ret"]).cumprod()
    cum["excess_cum"]    = (1 + daily["excess_ret"]).cumprod()
    cum.to_csv(out_cumret, index=False, encoding="utf-8-sig")

    # 指标对比
    metrics_df = pd.DataFrame({
        "strategy": strat_metrics,
        "benchmark": bm_metrics,
        "excess": excess_metrics,
    })
    metrics_df.to_csv(out_metric, encoding="utf-8-sig")

    logger.info("回测结果已保存到: %s", output_dir)
    logger.info("  每日收益: %s", out_daily)
    logger.info("  累积曲线: %s", out_cumret)
    logger.info("  指标摘要: %s", out_metric)

    return daily


# -------------------------
# CLI 入口
# -------------------------

def main():
    parser = argparse.ArgumentParser(description="量化策略回测框架")
    parser.add_argument("--pred_dir",      default="outputs",                help="预测信号 CSV 目录（含 pred_all_*.csv）")
    parser.add_argument("--price_dir",     default="data_akshare/raw",       help="价格数据目录（含 *.parquet 或 *.csv）")
    parser.add_argument("--output_dir",    default="outputs/backtest",        help="回测结果输出目录")
    parser.add_argument("--topk",          type=int,   default=TOPK,          help="每期持有的 Top-K 股票数量")
    parser.add_argument("--commission",    type=float, default=COMMISSION,     help="单边佣金率")
    parser.add_argument("--slippage",      type=float, default=SLIPPAGE,       help="单边滑点率")
    parser.add_argument("--rebalance_lag", type=int,   default=1,              help="信号执行滞后天数（默认 T+1）")
    parser.add_argument("--signal_col",    default="pred_signal",              help="信号列名")
    args = parser.parse_args()

    cost_rate = args.commission + args.slippage

    run_backtest(
        pred_dir      = args.pred_dir,
        price_dir     = args.price_dir,
        output_dir    = args.output_dir,
        topk          = args.topk,
        cost_rate     = cost_rate,
        rebalance_lag = args.rebalance_lag,
        signal_col    = args.signal_col,
    )


if __name__ == "__main__":
    main()
