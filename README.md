# quants — AI 量化选股系统

基于 Transformer Encoder 的 A 股量化选股方案，支持数据获取、特征工程、模型训练、预测选股和历史回测。

---

## 项目结构

```
quants/
├── features.py               # 特征工程（FEAT_COLS / FEAT_COLS_V2 / make_features）
├── model.py                  # 模型定义（PositionalEncoding / ReturnPredictor）
├── data_utils.py             # 数据工具（load_all_frames / Dataset / 统计量等）
├── config.py                 # 日志配置工具
├── backtest.py               # 回测框架（Top-K 选股 → PnL → 指标 → CSV）
│
├── fetch_akshare_a_daily.py  # 全量数据获取（akshare）
├── update_recent_data.py     # 增量数据更新
│
├── train_v1.py               # 训练脚本 v1（绝对收益标签）
├── train_v2.py               # 训练脚本 v2（横截面去均值标签 + IC/RankIC 早停 + Test Set）
├── predict_v1.py             # 预测脚本 v1
├── predict_v2.py             # 预测脚本 v2（含数据新鲜度校验 + 丰富输出）
│
├── requirements.txt          # Python 依赖
├── .gitignore
└── README.md
```

---

## 安装

```bash
# 1. 克隆仓库
git clone <repo_url>
cd quants

# 2. 创建虚拟环境（推荐）
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt
```

---

## 使用流程

### 1. 获取数据

**全量获取**（首次运行，耗时较长）：
```bash
python fetch_akshare_a_daily.py
```

**增量更新**（每日盘后运行）：
```bash
python update_recent_data.py --days 10
```

数据保存在 `data_akshare/raw/` 目录下，每只股票一个 parquet 文件。

---

### 2. 训练模型

**v1 训练**（绝对收益标签）：
```bash
python train_v1.py
```

**v2 训练**（横截面去均值标签，含 IC/RankIC 早停 + Test Set 独立评估）：
```bash
python train_v2.py
```

训练完成后，checkpoint 保存在：
- v1: `checkpoints/best.pt`
- v2: `checkpoints_v2/best.pt`

---

### 3. 预测选股

**v1 预测**：
```bash
python predict_v1.py
```

**v2 预测**（含数据新鲜度警告、波动率、信号百分位）：
```bash
python predict_v2.py
```

输出文件保存在 `outputs/` 目录：
- `pred_all_YYYYMMDD.csv`：全量股票预测信号
- `recommend_topK_YYYYMMDD.csv`：Top-K 推荐股票

---

### 4. 历史回测

```bash
python backtest.py \
    --pred_dir outputs/ \
    --price_dir data_akshare/raw/ \
    --topk 10 \
    --commission 0.001 \
    --slippage 0.001
```

回测结果保存在 `outputs/backtest/`：
- `backtest_daily.csv`：每日收益
- `backtest_cumret.csv`：累积收益曲线
- `backtest_metrics.csv`：指标摘要（Sharpe、最大回撤、Calmar 等）

---

## 主要特性

| 模块 | 功能 |
|------|------|
| **特征工程** | 原始 10 个因子（FEAT_COLS）+ 扩展 22 个因子（FEAT_COLS_V2，含动量/GK波动率/Amihud流动性/市场状态等） |
| **训练流程** | 横截面去均值标签、全局 z-score 标准化、早停监控 RankIC、ReduceLROnPlateau、Test Set 独立评估 |
| **预测** | 数据新鲜度校验、个股近期波动率、信号排名百分位 |
| **回测** | Top-K 等权持仓、T+1 执行、双边交易成本、Sharpe/MaxDD/Calmar/胜率/换手率、等权基准对比 |
| **工程质量** | 共享模块（features/model/data_utils）消除重复代码、logging 替代 print、完善异常处理 |

---

## 配置说明

各训练/预测脚本均通过顶部的 `CFG` dataclass 配置关键参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `lookback` | 60 | 输入序列长度（交易日） |
| `horizon` | 5 | 预测未来 N 日收益 |
| `train_start` | 2018-01-01 | 训练集开始日期 |
| `train_end` | 2024-06-30 | 训练集结束日期（v2） |
| `valid_end` | 2024-12-31 | 验证集结束日期（v2） |
| `test_end` | 2025-06-30 | 测试集结束日期（v2，样本外） |
| `early_stop_patience` | 5 | 早停耐心轮数 |

---

## 注意事项

- 数据目录 `data_akshare/` 已加入 `.gitignore`，不会提交到 Git
- 训练/推理必须使用相同的特征集（FEAT_COLS 或 FEAT_COLS_V2）和 checkpoint 配对
- v2 的预测信号是横截面相对信号，适合排序选股，不等同于绝对收益百分比
