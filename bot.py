"""
=============================================================
  SOL/USDT  |  9 EMA × 9 SMA(EMA) Crossover Backtest
  Timeframe  : 1-minute
  Data       : Binance (1 year, auto-chunked)
  Signal     : BUY when 9-EMA crosses above its 9-SMA
  SL         : Below trigger-candle low + buffer
  TP         : 2× SL distance  (Risk:Reward = 2:1)
=============================================================

EDITABLE PARAMETERS — change anything in the CONFIG block below
"""

# ─────────────────────── CONFIG ───────────────────────────── #

SYMBOL          = "SOLUSDT"     # Binance symbol
INTERVAL        = "1m"          # Candle timeframe
LOOKBACK_DAYS   = 365           # Days of history to fetch

EMA_PERIOD      = 9             # Period for the base EMA
SMA_PERIOD      = 9             # Period for SMA applied ON TOP of the EMA

SL_BUFFER_PCT   = 0.20          # Buffer on SL distance (% of raw SL dist)
RISK_REWARD     = 2.0           # TP = entry + SL_dist × RISK_REWARD

# ── Position Sizing ──────────────────────────────────────── #
#
#   RISK_MODE = "fixed"    → risk a flat USD amount every trade (realistic)
#   RISK_MODE = "percent"  → risk % of current capital every trade (compounds)
#
RISK_MODE       = "fixed"       # "fixed" or "percent"
RISK_FIXED_USD  = 100           # USD risked per trade  (used when RISK_MODE = "fixed")
RISK_PCT        = 1.0           # % of capital risked   (used when RISK_MODE = "percent")

CAPITAL         = 10_000        # Starting capital in USDT

# ── Output ───────────────────────────────────────────────── #
PLOT_CHART       = True         # Show charts (requires matplotlib)
SAVE_TRADES_CSV  = True         # Save trade log → trades_log.csv
PRINT_EACH_TRADE = False        # Print every trade to console

# ──────────────────────────────────────────────────────────── #

import requests, time, math
from datetime import datetime, timezone

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.gridspec as gridspec
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False
    print("[WARN] matplotlib not installed — charts disabled. Run: pip install matplotlib")


# ═══════════════════════ DATA FETCH ═══════════════════════════

BINANCE_BASE = "https://api.binance.com"

def fetch_klines_chunk(symbol, interval, start_ms, end_ms, limit=1000):
    url = f"{BINANCE_BASE}/api/v3/klines"
    params = dict(symbol=symbol, interval=interval,
                  startTime=start_ms, endTime=end_ms, limit=limit)
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def fetch_all_klines(symbol, interval, days):
    end_ms   = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = end_ms - days * 24 * 3600 * 1000

    interval_ms = {
        "1m": 60_000, "3m": 180_000, "5m": 300_000,
        "15m": 900_000, "30m": 1_800_000, "1h": 3_600_000,
    }[interval]

    total_candles = (end_ms - start_ms) // interval_ms
    num_chunks    = math.ceil(total_candles / 1000)

    print(f"\n{'='*60}")
    print(f"  Symbol   : {symbol}")
    print(f"  Interval : {interval}")
    print(f"  Period   : {days} days  (~{total_candles:,} candles)")
    print(f"  Requests : {num_chunks} API calls needed")
    print(f"{'='*60}\n")

    all_rows  = []
    cur_start = start_ms
    chunk     = 0

    while cur_start < end_ms:
        chunk += 1
        pct = chunk / num_chunks * 100
        print(f"\r  Downloading... {pct:5.1f}%  ({chunk}/{num_chunks})", end="", flush=True)

        rows = fetch_klines_chunk(symbol, interval, cur_start, end_ms, limit=1000)
        if not rows:
            break

        all_rows.extend(rows)
        cur_start = rows[-1][0] + interval_ms

        if chunk % 10 == 0:
            time.sleep(0.1)

    print(f"\r  Downloaded {len(all_rows):,} candles total.{' '*30}")

    cols = ["open_time","open","high","low","close","volume",
            "close_time","qav","num_trades","tbbav","tbqav","ignore"]
    df = pd.DataFrame(all_rows, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    df = df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    return df


# ═══════════════════════ INDICATORS ════════════════════════════

def calc_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def calc_sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema"]     = calc_ema(df["close"], EMA_PERIOD)
    df["sma_ema"] = calc_sma(df["ema"], SMA_PERIOD)

    df["ema_above"]      = df["ema"] > df["sma_ema"]
    # shift(1) produces NaN on first row → fill False, cast to bool
    df["ema_above_prev"] = df["ema_above"].shift(1).fillna(False).astype(bool)

    # BUY: EMA just crossed ABOVE its SMA
    df["signal_buy"] = (~df["ema_above_prev"]) & df["ema_above"]
    return df


# ════════════════════════ BACKTEST ═════════════════════════════

def run_backtest(df: pd.DataFrame):
    warmup  = EMA_PERIOD + SMA_PERIOD + 5
    trades  = []
    capital = float(CAPITAL)

    for i in range(warmup, len(df) - 2):
        if not df.at[i, "signal_buy"]:
            continue

        trigger = df.iloc[i]
        entry_c = df.iloc[i + 1]

        entry    = entry_c["open"]
        trig_low = trigger["low"]

        raw_sl_dist = entry - trig_low
        if raw_sl_dist <= 0:
            continue

        # SL with buffer
        sl_dist = raw_sl_dist * (1 + SL_BUFFER_PCT / 100)
        sl      = entry - sl_dist
        tp      = entry + sl_dist * RISK_REWARD

        # ── Position sizing ──────────────────────────────────
        if RISK_MODE == "fixed":
            risk_usd = RISK_FIXED_USD                       # flat $100 every trade
        else:
            risk_usd = capital * (RISK_PCT / 100)           # % of current capital

        qty = risk_usd / sl_dist                            # SOL units

        # ── Scan for exit ────────────────────────────────────
        exit_price = exit_reason = exit_idx = None
        for j in range(i + 2, len(df)):
            c = df.iloc[j]
            if c["low"] <= sl:
                exit_price, exit_reason, exit_idx = sl, "SL", j
                break
            if c["high"] >= tp:
                exit_price, exit_reason, exit_idx = tp, "TP", j
                break

        if exit_price is None:
            continue    # still open at end of data

        pnl_usd  = (exit_price - entry) * qty
        pnl_r    = pnl_usd / risk_usd
        capital += pnl_usd

        trade = {
            "entry_time" : entry_c["open_time"].strftime("%Y-%m-%d %H:%M"),
            "exit_time"  : df.iloc[exit_idx]["open_time"].strftime("%Y-%m-%d %H:%M"),
            "entry"      : round(entry, 4),
            "sl"         : round(sl, 4),
            "tp"         : round(tp, 4),
            "exit_price" : round(exit_price, 4),
            "sl_dist"    : round(sl_dist, 4),
            "qty"        : round(qty, 4),
            "risk_usd"   : round(risk_usd, 2),
            "pnl_usd"    : round(pnl_usd, 2),
            "pnl_r"      : round(pnl_r, 3),
            "result"     : exit_reason,
            "capital"    : round(capital, 2),
        }
        trades.append(trade)

        if PRINT_EACH_TRADE:
            print(f"  {trade['entry_time']}  {exit_reason}  "
                  f"Entry ${entry:.3f}  SL ${sl:.3f}  TP ${tp:.3f}  "
                  f"Exit ${exit_price:.3f}  {pnl_r:+.2f}R / ${pnl_usd:+.2f}")

    return pd.DataFrame(trades), capital


# ════════════════════════ STATISTICS ═══════════════════════════

def print_stats(trades: pd.DataFrame, final_capital: float):
    if trades.empty:
        print("\n  No completed trades found in this period.\n")
        return

    wins   = trades[trades["result"] == "TP"]
    losses = trades[trades["result"] == "SL"]

    total     = len(trades)
    win_count = len(wins)
    win_rate  = win_count / total * 100

    gross_win  = wins["pnl_usd"].sum()
    gross_loss = abs(losses["pnl_usd"].sum())
    net_pnl    = trades["pnl_usd"].sum()
    net_r      = trades["pnl_r"].sum()

    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")

    avg_win_usd  = wins["pnl_usd"].mean()   if not wins.empty   else 0
    avg_loss_usd = losses["pnl_usd"].mean() if not losses.empty else 0
    avg_win_r    = wins["pnl_r"].mean()     if not wins.empty   else 0
    avg_loss_r   = losses["pnl_r"].mean()   if not losses.empty else 0

    # Expectancy per trade
    expectancy_r   = (win_rate/100 * avg_win_r) + ((1 - win_rate/100) * avg_loss_r)
    expectancy_usd = (win_rate/100 * avg_win_usd) + ((1 - win_rate/100) * avg_loss_usd)

    # Max drawdown on capital curve
    cap_curve = [CAPITAL] + list(trades["capital"])
    peak, max_dd = cap_curve[0], 0.0
    for c in cap_curve:
        if c > peak:
            peak = c
        dd = (peak - c) / peak * 100
        if dd > max_dd:
            max_dd = dd

    # Max consecutive losses
    streak, max_streak = 0, 0
    for r in trades["result"]:
        if r == "SL":
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    # Breakeven win rate for this R:R
    breakeven_wr = 1 / (1 + RISK_REWARD) * 100

    risk_label = (f"${RISK_FIXED_USD} fixed per trade"
                  if RISK_MODE == "fixed"
                  else f"{RISK_PCT}% of capital (compounding)")

    sep = "─" * 54
    print(f"\n{'═'*54}")
    print(f"  BACKTEST RESULTS  —  {SYMBOL}  {INTERVAL}")
    print(f"{'═'*54}")
    print(f"  Strategy        :  {EMA_PERIOD} EMA × {SMA_PERIOD} SMA(EMA) crossover")
    print(f"  SL Buffer       :  {SL_BUFFER_PCT}% of SL distance")
    print(f"  Risk : Reward   :  1 : {RISK_REWARD}")
    print(f"  Risk / Trade    :  {risk_label}")
    print(f"{sep}")
    print(f"  Total Trades    :  {total:,}")
    print(f"  Wins / Losses   :  {win_count:,} / {len(losses):,}")
    print(f"  Win Rate        :  {win_rate:.1f}%  (breakeven: {breakeven_wr:.1f}%)")
    print(f"{sep}")
    print(f"  Net P&L (USD)   :  ${net_pnl:+,.2f}")
    print(f"  Net P&L (R)     :  {net_r:+.2f} R")
    print(f"  Profit Factor   :  {pf:.3f}")
    print(f"  Return on Cap   :  {(final_capital - CAPITAL) / CAPITAL * 100:+.2f}%")
    print(f"{sep}")
    print(f"  Avg Win         :  ${avg_win_usd:+,.2f}  /  {avg_win_r:+.3f} R")
    print(f"  Avg Loss        :  ${avg_loss_usd:+,.2f}  /  {avg_loss_r:+.3f} R")
    print(f"  Expectancy/Trade:  ${expectancy_usd:+,.2f}  /  {expectancy_r:+.4f} R")
    print(f"  Max Drawdown    :  {max_dd:.2f}%")
    print(f"  Max Consec Loss :  {max_streak}")
    print(f"{sep}")
    print(f"  Start Capital   :  ${CAPITAL:,.2f}")
    print(f"  End Capital     :  ${final_capital:,.2f}")
    print(f"{'═'*54}\n")

    # Quick health check
    print("  HEALTH CHECK:")
    print(f"    Win rate {win_rate:.1f}% vs breakeven {breakeven_wr:.1f}% → "
          + ("✓ Above breakeven" if win_rate > breakeven_wr else "✗ Below breakeven"))
    print(f"    Profit Factor {pf:.3f} → "
          + ("✓ Positive edge" if pf > 1 else "✗ Losing edge"))
    print(f"    Expectancy {expectancy_r:+.4f} R/trade → "
          + ("✓ Positive" if expectancy_r > 0 else "✗ Negative"))
    print(f"    Max Drawdown {max_dd:.1f}% → "
          + ("✓ Manageable" if max_dd < 30 else "⚠ High — review position sizing"))
    print()


# ════════════════════════ CHARTS ═══════════════════════════════

def plot_results(df: pd.DataFrame, trades: pd.DataFrame):
    if not MPL_AVAILABLE or not PLOT_CHART:
        return
    if trades.empty:
        print("  [INFO] No trades to plot.")
        return

    fig = plt.figure(figsize=(16, 14), facecolor="#0e1117")
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.30)

    ax_price = fig.add_subplot(gs[0, :])
    ax_pnl   = fig.add_subplot(gs[1, :])
    ax_dist  = fig.add_subplot(gs[2, 0])
    ax_dd    = fig.add_subplot(gs[2, 1])

    clr = {
        "bg"    : "#0e1117", "fg"   : "#e0e0e0", "grid" : "#1e2533",
        "price" : "#4a90d9", "ema"  : "#f0c040", "sma"  : "#e07090",
        "win"   : "#2ecc71", "loss" : "#e74c3c", "pnl"  : "#3498db",
        "dd"    : "#e74c3c",
    }

    def style_ax(ax):
        ax.set_facecolor(clr["bg"])
        ax.tick_params(colors=clr["fg"], labelsize=8)
        ax.xaxis.label.set_color(clr["fg"])
        ax.yaxis.label.set_color(clr["fg"])
        ax.title.set_color(clr["fg"])
        for spine in ax.spines.values():
            spine.set_edgecolor(clr["grid"])
        ax.grid(True, color=clr["grid"], linewidth=0.5, alpha=0.7)

    # ── Price + Indicators (last 2000 candles) ──────────────
    tail = df.tail(2000).copy()
    ax_price.plot(tail["open_time"], tail["close"],   color=clr["price"], lw=0.6, label="Close", alpha=0.7)
    ax_price.plot(tail["open_time"], tail["ema"],     color=clr["ema"],   lw=1.3, label=f"{EMA_PERIOD} EMA")
    ax_price.plot(tail["open_time"], tail["sma_ema"], color=clr["sma"],   lw=1.3, label=f"{SMA_PERIOD} SMA(EMA)", linestyle="--")
    sig_tail = tail[tail["signal_buy"]]
    if not sig_tail.empty:
        ax_price.scatter(sig_tail["open_time"], sig_tail["close"],
                         marker="^", color=clr["win"], s=20, zorder=5, alpha=0.75, label="Signal")
    ax_price.set_title(f"{SYMBOL} {INTERVAL} — Price & Indicators (last 2000 candles)", fontsize=10)
    ax_price.set_ylabel("Price (USDT)", fontsize=8)
    ax_price.legend(fontsize=7, facecolor="#1a1f2e", labelcolor=clr["fg"])
    ax_price.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    style_ax(ax_price)

    # ── Cumulative P&L ──────────────────────────────────────
    cum = trades["pnl_usd"].cumsum()
    ax_pnl.plot(range(len(trades)), cum, color=clr["pnl"], lw=1.5)
    ax_pnl.fill_between(range(len(trades)), 0, cum,
                         where=cum >= 0, alpha=0.15, color=clr["win"])
    ax_pnl.fill_between(range(len(trades)), 0, cum,
                         where=cum <  0, alpha=0.15, color=clr["loss"])
    ax_pnl.axhline(0, color=clr["fg"], lw=0.5, linestyle="--", alpha=0.4)
    risk_label_short = (f"Fixed ${RISK_FIXED_USD}/trade"
                        if RISK_MODE == "fixed" else f"{RISK_PCT}% compounding")
    ax_pnl.set_title(f"Cumulative P&L (USD)  —  {risk_label_short}", fontsize=10)
    ax_pnl.set_xlabel("Trade #", fontsize=8)
    ax_pnl.set_ylabel("USD", fontsize=8)
    style_ax(ax_pnl)

    # ── Individual trade P&L bars ────────────────────────────
    pnl_vals   = trades["pnl_usd"].values
    bar_colors = [clr["win"] if v >= 0 else clr["loss"] for v in pnl_vals]
    ax_dist.bar(range(len(pnl_vals)), pnl_vals, color=bar_colors, width=0.8, alpha=0.85)
    ax_dist.axhline(0, color=clr["fg"], lw=0.5, linestyle="--", alpha=0.4)
    ax_dist.set_title("Individual Trade P&L (USD)", fontsize=10)
    ax_dist.set_xlabel("Trade #", fontsize=8)
    ax_dist.set_ylabel("USD", fontsize=8)
    style_ax(ax_dist)

    # ── Drawdown curve ───────────────────────────────────────
    cap_curve = [CAPITAL] + list(trades["capital"].values)
    peak, dd_pct = cap_curve[0], []
    for c in cap_curve[1:]:
        if c > peak:
            peak = c
        dd_pct.append((peak - c) / peak * 100)
    ax_dd.fill_between(range(len(dd_pct)), 0, [-d for d in dd_pct],
                        color=clr["dd"], alpha=0.5)
    ax_dd.plot(range(len(dd_pct)), [-d for d in dd_pct], color=clr["dd"], lw=1.2)
    ax_dd.set_title("Drawdown (%)", fontsize=10)
    ax_dd.set_xlabel("Trade #", fontsize=8)
    ax_dd.set_ylabel("%", fontsize=8)
    style_ax(ax_dd)

    fig.suptitle(
        f"9 EMA × 9 SMA(EMA) Backtest  |  {SYMBOL} {INTERVAL}  |  "
        f"RR {RISK_REWARD}:1  |  SL buf {SL_BUFFER_PCT}%  |  {risk_label_short}",
        color=clr["fg"], fontsize=10, y=0.99
    )

    plt.savefig("backtest_chart.png", dpi=150, bbox_inches="tight", facecolor=clr["bg"])
    print("  Chart saved → backtest_chart.png")
    plt.show()


# ════════════════════════ SAVE CSV ═════════════════════════════

def save_csv(trades: pd.DataFrame):
    if not SAVE_TRADES_CSV or trades.empty:
        return
    path = "trades_log.csv"
    trades.to_csv(path, index=False)
    print(f"  Trade log saved → {path}")


# ════════════════════════ MAIN ═════════════════════════════════

def main():
    print("\n  SOL/USDT  9 EMA × 9 SMA(EMA) Crossover Backtester")
    print("  " + "─" * 50)

    df = fetch_all_klines(SYMBOL, INTERVAL, LOOKBACK_DAYS)
    print(f"  Date range : {df['open_time'].iloc[0].strftime('%Y-%m-%d')} "
          f"→ {df['open_time'].iloc[-1].strftime('%Y-%m-%d')}")

    print("\n  Computing indicators...")
    df = add_indicators(df)
    print(f"  Crossover signals found : {df['signal_buy'].sum():,}")

    print("\n  Running backtest...\n")
    trades, final_capital = run_backtest(df)

    print_stats(trades, final_capital)
    save_csv(trades)
    plot_results(df, trades)

    print("  Done.\n")


if __name__ == "__main__":
    main()
