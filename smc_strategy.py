"""
SMC Concepts Used:
  - Break of Structure (BOS) / Change of Character (CHOCH)
  - Order Blocks (OB)
  - Fair Value Gaps (FVG / Imbalance)
  - Liquidity Sweeps (Equal Highs/Lows)
  - Market Structure (HH, HL, LH, LL)
  - Premium / Discount Zones (Fibonacci 50% level)

Usage:
  pip install pandas numpy matplotlib yfinance ta-lib scipy
  python smc_strategy.py

  Or with custom settings:
  python smc_strategy.py --symbol EURUSD=X --interval 1h --start 2023-01-01 --end 2024-01-01
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import warnings
import argparse
import sys
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  CONFIG  (edit these or pass via CLI args)
# ─────────────────────────────────────────────
DEFAULT_SYMBOL   = "BTC-USD"      # Yahoo Finance ticker
DEFAULT_INTERVAL = "30m"           # 1m, 5m, 15m, 30m, 1h, 1d
DEFAULT_END      = datetime.now().strftime("%Y-%m-%d")
DEFAULT_START    = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

RISK_PER_TRADE   = 0.01            # 1% risk per trade
ACCOUNT_SIZE     = 30_000          # Starting account balance (USD)
MAX_TRADES_DAY   = 3               # Hard cap – never more than 3 trades/day
RR_TARGET        = 2.0             # Reward-to-Risk ratio
SWING_LOOKBACK   = 10              # Candles for swing high/low detection
OB_LOOKBACK      = 3               # Candles to look back for order block
FVG_MIN_SIZE_PCT = 0.0001          # Minimum FVG size (% of price) to be valid
LIQ_SWEEP_MARGIN = 0.0002          # How close price must get to sweep a level
SESSION_FILTER   = True            # Only trade during London/NY sessions (for intraday)


# ─────────────────────────────────────────────
#  DATA LOADER
# ─────────────────────────────────────────────
def load_data(symbol: str, interval: str, start: str, end: str) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError:
        print("❌  yfinance not found. Run: pip install yfinance")
        sys.exit(1)

    print(f"📥  Downloading {symbol} | {interval} | {start} → {end}")
    df = yf.download(symbol, start=start, end=end, interval=interval, auto_adjust=True, progress=False)

    if df.empty:
        print("❌  No data returned. Check symbol / date range.")
        sys.exit(1)

    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df.dropna(inplace=True)
    df.index = pd.to_datetime(df.index)
    print(f"✅  Loaded {len(df)} candles")
    return df


# ─────────────────────────────────────────────
#  SMC INDICATORS
# ─────────────────────────────────────────────

def detect_swing_highs_lows(df: pd.DataFrame, lookback: int = SWING_LOOKBACK):
    """Detect swing highs and lows using local maxima/minima."""
    highs = df["high"].values
    lows  = df["low"].values
    n = len(df)

    swing_high = np.zeros(n, dtype=bool)
    swing_low  = np.zeros(n, dtype=bool)

    for i in range(lookback, n - lookback):
        if highs[i] == max(highs[i - lookback: i + lookback + 1]):
            swing_high[i] = True
        if lows[i] == min(lows[i - lookback: i + lookback + 1]):
            swing_low[i] = True

    df["swing_high"] = swing_high
    df["swing_low"]  = swing_low
    return df


def detect_market_structure(df: pd.DataFrame):
    """
    Label each candle's market structure.
    HH = Higher High, HL = Higher Low, LH = Lower High, LL = Lower Low
    BOS = Break of Structure, CHOCH = Change of Character
    """
    df["structure"] = ""
    df["bos"]       = False   # Bullish BOS
    df["choch"]     = False   # Bearish CHOCH (reversal signal)

    swing_highs = df.index[df["swing_high"]].tolist()
    swing_lows  = df.index[df["swing_low"]].tolist()

    prev_sh_price = None
    prev_sl_price = None
    trend = "neutral"

    for i in range(len(df)):
        idx = df.index[i]

        if df.at[idx, "swing_high"]:
            if prev_sh_price is not None:
                if df.at[idx, "high"] > prev_sh_price:
                    df.at[idx, "structure"] = "HH"
                    if trend == "bearish":
                        df.at[idx, "choch"] = True   # CHOCH: bearish → bullish flip
                    trend = "bullish"
                else:
                    df.at[idx, "structure"] = "LH"
                    if trend == "bullish":
                        df.at[idx, "choch"] = True   # CHOCH: bullish → bearish flip
                    trend = "bearish"
            prev_sh_price = df.at[idx, "high"]

        if df.at[idx, "swing_low"]:
            if prev_sl_price is not None:
                if df.at[idx, "low"] > prev_sl_price:
                    df.at[idx, "structure"] = "HL"
                    if trend == "bullish":
                        df.at[idx, "bos"] = True     # BOS: continuation
                else:
                    df.at[idx, "structure"] = "LL"
            prev_sl_price = df.at[idx, "low"]

    return df


def detect_order_blocks(df: pd.DataFrame, lookback: int = OB_LOOKBACK):
    """
    Bullish OB: Last bearish candle before a strong bullish move (BOS up).
    Bearish OB: Last bullish candle before a strong bearish move (BOS down).
    """
    df["bull_ob"]     = False
    df["bear_ob"]     = False
    df["bull_ob_top"] = np.nan
    df["bull_ob_bot"] = np.nan
    df["bear_ob_top"] = np.nan
    df["bear_ob_bot"] = np.nan

    closes = df["close"].values
    opens  = df["open"].values
    highs  = df["high"].values
    lows   = df["low"].values
    n = len(df)

    for i in range(lookback + 1, n):
        # Bullish OB: bearish candle followed by strong bullish momentum
        if closes[i] > highs[i - lookback]:   # Price breaks above recent high
            for j in range(i - 1, max(i - lookback - 1, 0), -1):
                if closes[j] < opens[j]:       # Bearish candle = potential bull OB
                    df.iloc[j, df.columns.get_loc("bull_ob")]     = True
                    df.iloc[j, df.columns.get_loc("bull_ob_top")] = opens[j]
                    df.iloc[j, df.columns.get_loc("bull_ob_bot")] = lows[j]
                    break

        # Bearish OB: bullish candle followed by strong bearish momentum
        if closes[i] < lows[i - lookback]:    # Price breaks below recent low
            for j in range(i - 1, max(i - lookback - 1, 0), -1):
                if closes[j] > opens[j]:       # Bullish candle = potential bear OB
                    df.iloc[j, df.columns.get_loc("bear_ob")]     = True
                    df.iloc[j, df.columns.get_loc("bear_ob_top")] = highs[j]
                    df.iloc[j, df.columns.get_loc("bear_ob_bot")] = closes[j]
                    break

    return df


def detect_fvg(df: pd.DataFrame, min_size_pct: float = FVG_MIN_SIZE_PCT):
    """
    Fair Value Gap (3-candle imbalance):
    Bullish FVG: candle[i-2].high < candle[i].low  (gap up)
    Bearish FVG: candle[i-2].low  > candle[i].high (gap down)
    """
    df["bull_fvg"]     = False
    df["bear_fvg"]     = False
    df["bull_fvg_top"] = np.nan
    df["bull_fvg_bot"] = np.nan
    df["bear_fvg_top"] = np.nan
    df["bear_fvg_bot"] = np.nan

    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values
    n = len(df)

    for i in range(2, n):
        mid_price = closes[i]

        # Bullish FVG
        gap_size = lows[i] - highs[i - 2]
        if gap_size > 0 and (gap_size / mid_price) >= min_size_pct:
            df.iloc[i - 1, df.columns.get_loc("bull_fvg")]     = True
            df.iloc[i - 1, df.columns.get_loc("bull_fvg_top")] = lows[i]
            df.iloc[i - 1, df.columns.get_loc("bull_fvg_bot")] = highs[i - 2]

        # Bearish FVG
        gap_size = lows[i - 2] - highs[i]
        if gap_size > 0 and (gap_size / mid_price) >= min_size_pct:
            df.iloc[i - 1, df.columns.get_loc("bear_fvg")]     = True
            df.iloc[i - 1, df.columns.get_loc("bear_fvg_top")] = lows[i - 2]
            df.iloc[i - 1, df.columns.get_loc("bear_fvg_bot")] = highs[i]

    return df


def detect_liquidity_levels(df: pd.DataFrame, margin: float = LIQ_SWEEP_MARGIN):
    """
    Detect Equal Highs (sell-side liquidity) and Equal Lows (buy-side liquidity).
    A sweep occurs when price goes just beyond these levels and reverses.
    """
    df["liq_sweep_bull"] = False   # Swept buy-side liq (price dipped below EQL then reversed)
    df["liq_sweep_bear"] = False   # Swept sell-side liq (price spiked above EQH then reversed)

    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values
    n = len(df)

    for i in range(20, n):
        window_highs = highs[i - 20: i]
        window_lows  = lows[i - 20: i]

        # Equal lows (buy-side liquidity below): look for recent similar lows
        min_low = np.min(window_lows)
        if (lows[i] <= min_low * (1 + margin)) and (closes[i] > min_low):
            df.iloc[i, df.columns.get_loc("liq_sweep_bull")] = True

        # Equal highs (sell-side liquidity above): look for recent similar highs
        max_high = np.max(window_highs)
        if (highs[i] >= max_high * (1 - margin)) and (closes[i] < max_high):
            df.iloc[i, df.columns.get_loc("liq_sweep_bear")] = True

    return df


def is_session_active(timestamp, interval: str) -> bool:
    """Filter to London (07:00-12:00 UTC) and NY (13:00-17:00 UTC) sessions."""
    if not SESSION_FILTER or interval in ["1d", "1wk"]:
        return True
    hour = timestamp.hour
    return (7 <= hour < 12) or (13 <= hour < 17)


# ─────────────────────────────────────────────
#  SIGNAL GENERATION
# ─────────────────────────────────────────────

def generate_signals(df: pd.DataFrame, interval: str):
    """
    Entry Conditions:
    ──────────────────────────────────────────────
    LONG:
      1. Bullish market structure (CHOCH up or BOS confirmed)
      2. Price retraces into a Bullish OB or Bullish FVG
      3. (Bonus) Liquidity sweep to the downside just occurred
      4. Must be in Premium/Discount zone (below 50% of last swing)
      5. Session filter (London or NY)

    SHORT:
      1. Bearish market structure (CHOCH down)
      2. Price retraces into a Bearish OB or Bearish FVG
      3. (Bonus) Liquidity sweep to the upside just occurred
      4. Must be in Premium/Discount zone (above 50% of last swing)
      5. Session filter (London or NY)
    ──────────────────────────────────────────────
    """
    signals = []
    trades_per_day: dict = {}
    
    # Track last known OBs and FVGs
    active_bull_ob = None  # (top, bot)
    active_bear_ob = None
    active_bull_fvg = None
    active_bear_fvg = None
    last_choch_bull = False
    last_choch_bear = False
    trend_bias = "neutral"

    highs = df["high"].values
    lows  = df["low"].values
    n = len(df)

    for i in range(30, n):
        idx  = df.index[i]
        date = idx.date()

        # ── Day trade cap ──────────────────────────────────
        day_count = trades_per_day.get(date, 0)
        if day_count >= MAX_TRADES_DAY:
            continue

        # ── Session filter ─────────────────────────────────
        if not is_session_active(idx, interval):
            continue

        row = df.iloc[i]
        price = row["close"]

        # ── Update active OBs / FVGs ───────────────────────
        if row["bull_ob"] and not pd.isna(row["bull_ob_top"]):
            active_bull_ob = (row["bull_ob_top"], row["bull_ob_bot"])
        if row["bear_ob"] and not pd.isna(row["bear_ob_top"]):
            active_bear_ob = (row["bear_ob_top"], row["bear_ob_bot"])
        if row["bull_fvg"] and not pd.isna(row["bull_fvg_top"]):
            active_bull_fvg = (row["bull_fvg_top"], row["bull_fvg_bot"])
        if row["bear_fvg"] and not pd.isna(row["bear_fvg_top"]):
            active_bear_fvg = (row["bear_fvg_top"], row["bear_fvg_bot"])

        # ── Track CHOCH / trend bias ───────────────────────
        if row["choch"]:
            if row["structure"] in ("HH",):
                last_choch_bull = True
                last_choch_bear = False
                trend_bias = "bullish"
            elif row["structure"] in ("LH",):
                last_choch_bear = True
                last_choch_bull = False
                trend_bias = "bearish"

        # ── Recent swing for Premium/Discount zone ─────────
        recent_high = max(highs[max(0, i - 20): i])
        recent_low  = min(lows[max(0, i - 20): i])
        mid_level   = (recent_high + recent_low) / 2
        in_discount = price < mid_level   # Good for longs
        in_premium  = price > mid_level   # Good for shorts

        # ═══════════════════════════════════════════════════
        #  LONG SIGNAL
        # ═══════════════════════════════════════════════════
        long_score = 0

        if trend_bias == "bullish":
            long_score += 2

        # Price in OB zone
        if active_bull_ob:
            ob_top, ob_bot = active_bull_ob
            if ob_bot <= price <= ob_top:
                long_score += 3

        # Price in FVG zone
        if active_bull_fvg:
            fvg_top, fvg_bot = active_bull_fvg
            if fvg_bot <= price <= fvg_top:
                long_score += 2

        # Liquidity sweep bonus
        if row["liq_sweep_bull"]:
            long_score += 2

        # Discount zone
        if in_discount:
            long_score += 1

        # ═══════════════════════════════════════════════════
        #  SHORT SIGNAL
        # ═══════════════════════════════════════════════════
        short_score = 0

        if trend_bias == "bearish":
            short_score += 2

        if active_bear_ob:
            ob_top, ob_bot = active_bear_ob
            if ob_bot <= price <= ob_top:
                short_score += 3

        if active_bear_fvg:
            fvg_top, fvg_bot = active_bear_fvg
            if fvg_bot <= price <= fvg_top:
                short_score += 2

        if row["liq_sweep_bear"]:
            short_score += 2

        if in_premium:
            short_score += 1

        # ── Decide signal (min score = 5) ──────────────────
        direction = None
        if long_score >= 5 and long_score > short_score:
            direction = "LONG"
        elif short_score >= 5 and short_score > long_score:
            direction = "SHORT"

        if direction:
            # ── Calculate SL / TP ──────────────────────────
            atr = df["high"].iloc[max(0,i-14):i].values - df["low"].iloc[max(0,i-14):i].values
            atr_val = np.mean(atr) if len(atr) else price * 0.001

            if direction == "LONG":
                stop_loss   = price - (1.5 * atr_val)
                take_profit = price + (RR_TARGET * 1.5 * atr_val)
            else:
                stop_loss   = price + (1.5 * atr_val)
                take_profit = price - (RR_TARGET * 1.5 * atr_val)

            # ── Position size ──────────────────────────────
            risk_amount   = ACCOUNT_SIZE * RISK_PER_TRADE
            risk_per_unit = abs(price - stop_loss)
            position_size = risk_amount / risk_per_unit if risk_per_unit > 0 else 0

            signals.append({
                "entry_time":    idx,
                "entry_price":   price,
                "direction":     direction,
                "stop_loss":     stop_loss,
                "take_profit":   take_profit,
                "position_size": position_size,
                "score":         long_score if direction == "LONG" else short_score,
                "atr":           atr_val,
                "trend_bias":    trend_bias,
            })

            trades_per_day[date] = day_count + 1

    return pd.DataFrame(signals)


# ─────────────────────────────────────────────
#  BACKTEST ENGINE
# ─────────────────────────────────────────────

def run_backtest(df: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate trade outcomes using OHLC data after signal entry.
    Exit logic: price hits TP or SL on a candle-by-candle basis.
    """
    if signals.empty:
        print("⚠️  No signals generated.")
        return pd.DataFrame()

    results = []
    equity  = ACCOUNT_SIZE

    for _, sig in signals.iterrows():
        entry_time  = sig["entry_time"]
        entry_price = sig["entry_price"]
        direction   = sig["direction"]
        sl          = sig["stop_loss"]
        tp          = sig["take_profit"]
        size        = sig["position_size"]

        # Find candles after entry
        future = df[df.index > entry_time]

        outcome    = "OPEN"
        exit_price = entry_price
        exit_time  = entry_time
        pnl        = 0

        for idx, candle in future.iterrows():
            if direction == "LONG":
                if candle["low"] <= sl:
                    outcome    = "LOSS"
                    exit_price = sl
                    exit_time  = idx
                    break
                if candle["high"] >= tp:
                    outcome    = "WIN"
                    exit_price = tp
                    exit_time  = idx
                    break
            else:  # SHORT
                if candle["high"] >= sl:
                    outcome    = "LOSS"
                    exit_price = sl
                    exit_time  = idx
                    break
                if candle["low"] <= tp:
                    outcome    = "WIN"
                    exit_price = tp
                    exit_time  = idx
                    break

        if outcome == "OPEN":
            exit_price = future["close"].iloc[-1] if not future.empty else entry_price
            exit_time  = future.index[-1] if not future.empty else entry_time

        if direction == "LONG":
            pnl = (exit_price - entry_price) * size
        else:
            pnl = (entry_price - exit_price) * size

        equity += pnl

        results.append({
            **sig.to_dict(),
            "exit_time":   exit_time,
            "exit_price":  exit_price,
            "outcome":     outcome,
            "pnl":         round(pnl, 2),
            "equity":      round(equity, 2),
        })

    return pd.DataFrame(results)


# ─────────────────────────────────────────────
#  PERFORMANCE METRICS
# ─────────────────────────────────────────────

def compute_metrics(results: pd.DataFrame) -> dict:
    if results.empty:
        return {}

    closed = results[results["outcome"].isin(["WIN", "LOSS"])]
    wins   = closed[closed["outcome"] == "WIN"]
    losses = closed[closed["outcome"] == "LOSS"]

    total_trades  = len(closed)
    win_rate      = len(wins) / total_trades * 100 if total_trades else 0
    net_pnl       = closed["pnl"].sum()
    gross_profit  = wins["pnl"].sum()
    gross_loss    = abs(losses["pnl"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss else float("inf")
    avg_win       = wins["pnl"].mean() if len(wins) else 0
    avg_loss      = losses["pnl"].mean() if len(losses) else 0

    # Drawdown
    equity_curve = results["equity"].values
    peak = equity_curve[0]
    max_dd = 0
    for eq in equity_curve:
        peak  = max(peak, eq)
        dd    = (peak - eq) / peak * 100
        max_dd = max(max_dd, dd)

    # Sharpe (simplified daily returns)
    if len(results) > 1:
        returns = results["pnl"] / ACCOUNT_SIZE
        sharpe  = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() != 0 else 0
    else:
        sharpe = 0

    return {
        "Total Trades":       total_trades,
        "Win Rate (%)":       round(win_rate, 1),
        "Net P&L ($)":        round(net_pnl, 2),
        "Profit Factor":      round(profit_factor, 2),
        "Avg Win ($)":        round(avg_win, 2),
        "Avg Loss ($)":       round(avg_loss, 2),
        "Max Drawdown (%)":   round(max_dd, 2),
        "Sharpe Ratio":       round(sharpe, 2),
        "Final Equity ($)":   round(results["equity"].iloc[-1], 2),
        "Return (%)":         round((results["equity"].iloc[-1] - ACCOUNT_SIZE) / ACCOUNT_SIZE * 100, 2),
    }


# ─────────────────────────────────────────────
#  VISUALISATION
# ─────────────────────────────────────────────

def plot_results(df: pd.DataFrame, results: pd.DataFrame, metrics: dict, symbol: str):
    """Generate a comprehensive backtest report chart."""
    fig = plt.figure(figsize=(20, 14), facecolor="#0d1117")
    fig.suptitle(f"SMC Strategy Backtest — {symbol}", 
                 fontsize=18, color="#e6edf3", fontweight="bold", y=0.98)

    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3,
                          left=0.05, right=0.97, top=0.93, bottom=0.05)

    ax_price  = fig.add_subplot(gs[0, :2])   # Price chart
    ax_equity = fig.add_subplot(gs[1, :2])   # Equity curve
    ax_pnl    = fig.add_subplot(gs[2, :2])   # P&L distribution
    ax_stats  = fig.add_subplot(gs[:, 2])    # Stats panel

    DARK   = "#0d1117"
    GRID   = "#21262d"
    TEXT   = "#e6edf3"
    GREEN  = "#3fb950"
    RED    = "#f85149"
    YELLOW = "#d29922"
    BLUE   = "#58a6ff"
    PURPLE = "#bc8cff"

    for ax in [ax_price, ax_equity, ax_pnl, ax_stats]:
        ax.set_facecolor(DARK)
        ax.tick_params(colors=TEXT, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)

    # ── Price chart ────────────────────────────────────────
    recent = df.tail(500)
    ax_price.plot(recent.index, recent["close"], color="#8b949e", linewidth=0.8, alpha=0.6)

    # Plot swing highs/lows
    sh_idx = recent[recent["swing_high"]].index
    sl_idx = recent[recent["swing_low"]].index
    ax_price.scatter(sh_idx, recent.loc[sh_idx, "high"] * 1.001,
                     marker="v", color=RED, s=30, zorder=5, label="Swing High")
    ax_price.scatter(sl_idx, recent.loc[sl_idx, "low"] * 0.999,
                     marker="^", color=GREEN, s=30, zorder=5, label="Swing Low")

    # Plot trade entries
    if not results.empty:
        for _, t in results.iterrows():
            if t["entry_time"] not in recent.index:
                continue
            color  = GREEN if t["direction"] == "LONG" else RED
            marker = "^" if t["direction"] == "LONG" else "v"
            ax_price.scatter(t["entry_time"], t["entry_price"],
                             marker=marker, color=color, s=80, zorder=10)

    ax_price.set_title("Price + Trade Entries", color=TEXT, fontsize=10, pad=6)
    ax_price.yaxis.label.set_color(TEXT)
    ax_price.legend(fontsize=7, facecolor=DARK, labelcolor=TEXT, loc="upper left")
    ax_price.grid(True, color=GRID, linewidth=0.4)

    # ── Equity curve ───────────────────────────────────────
    if not results.empty:
        eq_times = results["exit_time"]
        eq_vals  = results["equity"]
        ax_equity.plot(eq_times, eq_vals, color=BLUE, linewidth=1.5)
        ax_equity.fill_between(eq_times, ACCOUNT_SIZE, eq_vals,
                               where=(eq_vals >= ACCOUNT_SIZE),
                               color=GREEN, alpha=0.15)
        ax_equity.fill_between(eq_times, ACCOUNT_SIZE, eq_vals,
                               where=(eq_vals < ACCOUNT_SIZE),
                               color=RED, alpha=0.15)
        ax_equity.axhline(ACCOUNT_SIZE, color=YELLOW, linewidth=0.8, linestyle="--", alpha=0.7)

    ax_equity.set_title("Equity Curve", color=TEXT, fontsize=10, pad=6)
    ax_equity.grid(True, color=GRID, linewidth=0.4)

    # ── P&L distribution ───────────────────────────────────
    if not results.empty:
        pnl_vals = results["pnl"].values
        colors_  = [GREEN if p >= 0 else RED for p in pnl_vals]
        ax_pnl.bar(range(len(pnl_vals)), pnl_vals, color=colors_, width=0.8)
        ax_pnl.axhline(0, color=YELLOW, linewidth=0.8, linestyle="--")

    ax_pnl.set_title("Trade P&L", color=TEXT, fontsize=10, pad=6)
    ax_pnl.set_xlabel("Trade #", color=TEXT, fontsize=8)
    ax_pnl.set_ylabel("P&L ($)", color=TEXT, fontsize=8)
    ax_pnl.grid(True, color=GRID, linewidth=0.4)

    # ── Stats panel ────────────────────────────────────────
    ax_stats.axis("off")
    ax_stats.set_title("Performance Summary", color=TEXT, fontsize=10, pad=6)

    y_pos = 0.95
    for key, val in metrics.items():
        is_positive = True
        if "P&L" in key or "Return" in key or "Equity" in key:
            is_positive = float(str(val).replace("$", "").replace("%", "")) >= 0
        elif "Drawdown" in key:
            is_positive = float(str(val).replace("%", "")) < 15
        elif "Win Rate" in key:
            is_positive = float(str(val).replace("%", "")) >= 50
        elif "Profit Factor" in key:
            is_positive = float(str(val)) >= 1.0

        val_color = GREEN if is_positive else RED
        if "Sharpe" in key:
            val_color = GREEN if float(str(val)) > 0.5 else RED

        ax_stats.text(0.05, y_pos, f"{key}:", transform=ax_stats.transAxes,
                      fontsize=9, color="#8b949e", va="top")
        ax_stats.text(0.95, y_pos, str(val), transform=ax_stats.transAxes,
                      fontsize=9, color=val_color, va="top", ha="right", fontweight="bold")
        y_pos -= 0.085

    # Config info
    ax_stats.text(0.05, 0.08, f"Risk/Trade: {RISK_PER_TRADE*100:.1f}%\nRR Target: 1:{RR_TARGET}\nMax Trades/Day: {MAX_TRADES_DAY}",
                  transform=ax_stats.transAxes, fontsize=8, color="#8b949e", va="bottom")

    plt.savefig("smc_backtest_report.png", dpi=150, bbox_inches="tight",
                facecolor=DARK, edgecolor="none")
    print("📊  Chart saved → smc_backtest_report.png")
    plt.show()


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SMC Strategy Backtester")
    parser.add_argument("--symbol",   default=DEFAULT_SYMBOL,   help="Yahoo Finance ticker")
    parser.add_argument("--interval", default=DEFAULT_INTERVAL, help="Candle interval")
    parser.add_argument("--start",    default=DEFAULT_START,    help="Start date YYYY-MM-DD")
    parser.add_argument("--end",      default=DEFAULT_END,      help="End date YYYY-MM-DD")
    args = parser.parse_args()

    print("\n" + "═"*60)
    print("  SMC Backtesting Engine — Smart Money Concepts")
    print("═"*60)

    # 1. Load data
    df = load_data(args.symbol, args.interval, args.start, args.end)

    # 2. Run SMC indicators
    print("🔍  Detecting SMC structure...")
    df = detect_swing_highs_lows(df)
    df = detect_market_structure(df)
    df = detect_order_blocks(df)
    df = detect_fvg(df)
    df = detect_liquidity_levels(df)

    bull_obs  = df["bull_ob"].sum()
    bear_obs  = df["bear_ob"].sum()
    bull_fvgs = df["bull_fvg"].sum()
    bear_fvgs = df["bear_fvg"].sum()
    print(f"   ✔ Bull OBs: {bull_obs}  Bear OBs: {bear_obs}")
    print(f"   ✔ Bull FVGs: {bull_fvgs}  Bear FVGs: {bear_fvgs}")
    print(f"   ✔ Liq sweeps ↑: {df['liq_sweep_bull'].sum()}  Liq sweeps ↓: {df['liq_sweep_bear'].sum()}")

    # 3. Generate signals
    print("📡  Generating signals (max 3/day)...")
    signals = generate_signals(df, args.interval)
    print(f"   ✔ {len(signals)} signals generated")

    if signals.empty:
        print("\n⚠️  No signals found. Try a wider date range or different timeframe.")
        return

    # 4. Backtest
    print("⚙️  Running backtest...")
    results = run_backtest(df, signals)

    # 5. Metrics
    metrics = compute_metrics(results)
    print("\n" + "─"*40)
    print("  📈  BACKTEST RESULTS")
    print("─"*40)
    for k, v in metrics.items():
        print(f"  {k:<22} {v}")
    print("─"*40)

    # 6. Save CSV
    results.to_csv("smc_backtest_trades.csv", index=False)
    print("\n💾  Trades saved → smc_backtest_trades.csv")

    # 7. Plot
    plot_results(df, results, metrics, args.symbol)
    print("\n✅  Backtest complete!\n")


if __name__ == "__main__":
    main()
