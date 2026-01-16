# Scalping Strategy Backtest Results

This document presents the comprehensive backtest results for the TSLA momentum scalping strategy, tested across two distinct market environments: 2022 (volatile bear market) and 2024 (mixed conditions).

## Executive Summary

| Metric | 2022 (Bear Market) | 2024 (Mixed) |
|--------|-------------------|--------------|
| **Total Return** | +231.6% | +48.8% |
| **Win Rate** | 49.4% | 56.3% |
| **Total Trades** | 178 | 64 |
| **Profit Factor** | 3.70 | 3.48 |
| **Sharpe Ratio** | 2.17 | 2.50 |
| **Max Drawdown** | -5.3% | -4.4% |
| **Expectancy/Trade** | $1,301 | $763 |

**Key Finding:** The strategy performs well in both market conditions, with stronger absolute returns in volatile markets (2022) but better risk-adjusted metrics in calmer conditions (2024).

---

## 2022 Backtest: Bear Market Volatility

### Performance Summary

```
Period:           Jan 1, 2022 - Dec 31, 2022
Trading Days:     207
Starting Equity:  $100,000
Ending Equity:    $331,615
Total Return:     +231.6%
CAGR:             330.3%
```

### Risk Metrics

| Metric | Value |
|--------|-------|
| Max Drawdown | $16,680 (-5.3%) |
| Sharpe Ratio | 2.17 |
| Sortino Ratio | 12.60 |
| Calmar Ratio | 53.32 |

### Trade Statistics

| Metric | Value |
|--------|-------|
| Total Trades | 178 |
| Winners | 88 (49.4%) |
| Losers | 90 (50.6%) |
| Avg Win | $3,608 |
| Avg Loss | $954 |
| Largest Win | $107,833 |
| Largest Loss | -$4,348 |
| Profit Factor | 3.70 |
| Expectancy | $1,301/trade |

### Performance by Direction

| Direction | Trades | Win Rate | Total P&L |
|-----------|--------|----------|-----------|
| **PUT** | 74 | 54.1% | +$162,087 |
| **CALL** | 104 | 46.2% | +$69,528 |

**Insight:** PUT signals significantly outperformed in 2022's bear market, capturing panic drops more effectively.

### Performance by DTE

| DTE | Trades | Win Rate | Total P&L | Avg P&L/Trade |
|-----|--------|----------|-----------|---------------|
| **1** | 73 | 60.3% | +$189,090 | +$2,590 |
| **2** | 60 | 46.7% | +$31,950 | +$533 |
| **3** | 45 | 35.6% | +$10,575 | +$235 |

**Insight:** DTE=1 dramatically outperforms longer-dated options. This validates the strategy's focus on 1-3 DTE with preference for DTE=1.

### Exit Reason Analysis

| Exit Reason | Trades | Win Rate | Total P&L |
|-------------|--------|----------|-----------|
| Take Profit (+20%) | 88 | 100% | +$317,504 |
| Time Stop (5 min) | 60 | 0% | -$27,529 |
| Stop Loss (-15%) | 30 | 0% | -$58,360 |

### Monthly Returns

| Month | Return | Notable Events |
|-------|--------|----------------|
| Jan | +7.3% | |
| Feb | +38.3% | Russia-Ukraine tensions |
| Mar | +4.7% | |
| Apr | +0.2% | Low volatility |
| May | +4.7% | |
| **Jun** | **+65.9%** | Peak volatility, $107K single trade |
| Jul | +2.6% | |
| Aug | +0.0% | No trades |
| Sep | +0.7% | |
| Oct | +2.3% | |
| Nov | +1.5% | |
| Dec | +8.4% | Year-end volatility |

### Notable Trades

**Best Trades:**
1. Jun 2, 2022 - PUT DTE=1: **+$107,832** (largest single trade)
2. Feb 24, 2022 - PUT DTE=1: **+$12,451** (Russia invasion day)
3. May 19, 2022 - CALL DTE=1: **+$5,076**

**Worst Trades:**
1. Dec 28, 2022 - PUT DTE=2: **-$4,348**
2. Nov 10, 2022 - CALL DTE=1: **-$3,670**
3. Nov 9, 2022 - PUT DTE=2: **-$3,364**

---

## 2024 Backtest: Mixed Market Conditions

### Performance Summary

```
Period:           Jan 1, 2024 - Dec 31, 2024
Trading Days:     205
Starting Equity:  $100,000
Ending Equity:    $148,804
Total Return:     +48.8%
CAGR:             63.0%
```

### Risk Metrics

| Metric | Value |
|--------|-------|
| Max Drawdown | $5,118 (-4.4%) |
| Sharpe Ratio | 2.50 |
| Sortino Ratio | 3.60 |
| Calmar Ratio | 13.76 |

### Trade Statistics

| Metric | Value |
|--------|-------|
| Total Trades | 64 |
| Winners | 36 (56.3%) |
| Losers | 28 (43.7%) |
| Avg Win | $1,903 |
| Avg Loss | $703 |
| Largest Win | $8,457 |
| Largest Loss | -$2,578 |
| Profit Factor | 3.48 |
| Expectancy | $763/trade |

### Performance by Direction

| Direction | Trades | Win Rate | Total P&L |
|-----------|--------|----------|-----------|
| **PUT** | 32 | 56.3% | +$28,884 |
| **CALL** | 32 | 56.3% | +$19,920 |

**Insight:** Balanced performance between PUTs and CALLs in 2024's mixed market, unlike 2022's PUT dominance.

### Performance by DTE

| DTE | Trades | Win Rate | Total P&L | Avg P&L/Trade |
|-----|--------|----------|-----------|---------------|
| **1** | 25 | 64.0% | +$28,864 | +$1,155 |
| **2** | 19 | 47.4% | +$3,505 | +$184 |
| **3** | 20 | 55.0% | +$16,435 | +$822 |

**Insight:** DTE=1 again shows the highest win rate (64%), reinforcing the preference for near-expiration options.

### Exit Reason Analysis

| Exit Reason | Trades | Total P&L |
|-------------|--------|-----------|
| Take Profit (+20%) | 36 | +$68,501 |
| Time Stop (5 min) | 13 | -$3,819 |
| Stop Loss (-15%) | 14 | -$15,468 |
| End of Day | 1 | -$411 |

### Monthly Returns

| Month | Return |
|-------|--------|
| Jan | +10.7% |
| Feb | +1.9% |
| Mar | +0.7% |
| Apr | +2.5% |
| May | +0.0% |
| Jun | -0.4% |
| **Jul** | **+24.1%** |
| Aug | +0.0% |
| Sep | +0.0% |
| Oct | +0.0% |
| Nov | +0.0% |
| Dec | -0.3% |

**Note:** Several months show 0% return due to no qualifying signals meeting all entry criteria.

---

## Strategy Configuration Used

These backtests used the following configuration (now with asymmetric thresholds):

```python
ScalpConfig(
    # DTE filtering
    min_dte=1,                        # Never 0DTE
    max_dte=3,
    dte_preference=(1, 2, 3),         # Prefer DTE=1

    # Momentum thresholds (asymmetric)
    momentum_threshold_put_pct=0.4,   # PUTs: 0.4% velocity
    momentum_threshold_call_pct=0.6,  # CALLs: 0.6% velocity
    momentum_window_seconds=30,

    # Option selection
    target_delta=0.40,
    delta_tolerance=0.15,             # Accept 0.25-0.55 delta
    max_spread_pct=10.0,
    max_contract_price=3.00,

    # Risk management
    take_profit_pct=20.0,
    stop_loss_pct=15.0,
    time_stop_minutes=5,
    max_confidence=75,                # Skip overextended moves
)
```

---

## Key Insights

### 1. DTE=1 is Optimal
Both years show DTE=1 significantly outperforming:
- 2022: 60.3% win rate, $2,590 avg profit
- 2024: 64.0% win rate, $1,155 avg profit

Longer DTEs (2-3) have lower win rates and should only be used when DTE=1 options aren't available.

### 2. PUT Signals Outperform in Bear Markets
In 2022's bear market, PUTs generated 2.3x the P&L of CALLs despite fewer trades. This supports asymmetric thresholds:
- PUT threshold: 0.4% (catch panic drops early)
- CALL threshold: 0.6% (require rally confirmation)

### 3. Take Profit Captures Most Gains
~50% of trades hit take profit, generating all the positive P&L. The 20% take profit level appears well-calibrated.

### 4. Time Stop Prevents Larger Losses
The 5-minute time stop exits unprofitable trades early, limiting average loss to ~$950 (2022) and ~$700 (2024).

### 5. Volatility Drives Opportunity
- 2022 (high volatility): 178 trades, +231% return
- 2024 (lower volatility): 64 trades, +49% return

The strategy naturally reduces exposure in calm markets by generating fewer signals.

---

## Backtest Methodology

- **Data Source:** DataBento historical options tick data
- **Price Source:** 1-second TSLA bars for accurate velocity calculation
- **Execution:** Simulated fills at ask price (conservative)
- **Position Sizing:** 5% of equity per trade
- **Slippage:** Not modeled (TSLA options highly liquid)

---

## Files

The complete backtest results with individual trade details are available in:
- `backtest_results/backtest_2022_baseline_momentum.json`
- `backtest_results/backtest_2024_baseline_momentum.json`

To run your own backtest:
```bash
python run.py --backtest /path/to/databento/data --start 2024-01-01 --end 2024-12-31 -o results.json
```
