"""
╔══════════════════════════════════════════════════════════════╗
║   SOL/USDT  |  9 EMA × 9 SMA(EMA)  |  LIVE SIGNAL TRACKER  ║
║   Sends Telegram alerts on every signal + trade close        ║
╚══════════════════════════════════════════════════════════════╝

HOW IT WORKS:
  1. Every 60 seconds fetches latest 1m candles from Binance
  2. Recalculates 9 EMA and its 9-period SMA
  3. Detects crossover → sends BUY SIGNAL alert on Telegram
  4. Tracks the open trade for SL / TP hit on every tick
  5. Sends TRADE CLOSED alert with P&L + running totals

SETUP:
  pip install requests pandas

EDIT THE CONFIG BLOCK BELOW, THEN RUN:
  python sol_live_tracker.py
"""

# ─────────────────────── CONFIG ───────────────────────────── #

SYMBOL         = "SOLUSDT"          # Binance pair
INTERVAL       = "1m"               # Candle interval (keep 1m)
WARMUP_CANDLES = 100                # Candles to fetch on startup (min: EMA+SMA period)

EMA_PERIOD     = 9                  # Base EMA period
SMA_PERIOD     = 9                  # SMA applied ON TOP of the EMA

SL_BUFFER_PCT  = 0.20               # Buffer % added to raw SL distance
RISK_REWARD    = 2.0                # TP = entry + SL_dist × RISK_REWARD

# ── Position Sizing ──────────────────────────────────────── #
RISK_MODE      = "fixed"            # "fixed" → flat USD | "percent" → % of capital
RISK_FIXED_USD = 100                # USD risked per trade (if RISK_MODE = "fixed")
RISK_PCT       = 1.0                # % of capital risked  (if RISK_MODE = "percent")
CAPITAL        = 10_000             # Starting capital (used for % mode & display)

# ── Telegram ─────────────────────────────────────────────── #
TG_BOT_TOKEN   = "8551296586:AAFX631OEnFIX0L1uVoc4Ysv3-KpTsqkqHA"
TG_CHAT_IDS    = ["1950462171"]     # Add more chat IDs if needed: ["id1", "id2"]

# ── Behaviour ────────────────────────────────────────────── #
POLL_SECONDS   = 60                 # How often to check for new candle (seconds)
SEND_STARTUP   = True               # Send a startup message when bot launches
SAVE_LOG_CSV   = True               # Append each closed trade to trades_live.csv

# ──────────────────────────────────────────────────────────── #

import requests
import time
import csv
import os
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


# ═══════════════════════ TELEGRAM ══════════════════════════════

TG_API = f"https://api.telegram.org/bot{TG_BOT_TOKEN}"

def tg_send(text: str, silent: bool = False):
    """Send a message to all configured Telegram chat IDs."""
    for chat_id in TG_CHAT_IDS:
        try:
            resp = requests.post(
                f"{TG_API}/sendMessage",
                json={
                    "chat_id": chat_id,
                    "text": text,
                    "parse_mode": "HTML",
                    "disable_notification": silent,
                },
                timeout=10,
            )
            if not resp.ok:
                print(f"  [TG WARN] {resp.status_code} → {resp.text[:120]}")
        except Exception as e:
            print(f"  [TG ERROR] {e}")


# ═══════════════════════ DATA FETCH ════════════════════════════

BINANCE_BASE = "https://api.binance.com"

def fetch_candles(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    """Fetch the latest `limit` candles from Binance."""
    url = f"{BINANCE_BASE}/api/v3/klines"
    resp = requests.get(
        url,
        params={"symbol": symbol, "interval": interval, "limit": limit},
        timeout=10,
    )
    resp.raise_for_status()
    raw = resp.json()

    cols = ["open_time","open","high","low","close","volume",
            "close_time","qav","num_trades","tbbav","tbqav","ignore"]
    df = pd.DataFrame(raw, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for c in ["open","high","low","close"]:
        df[c] = df[c].astype(float)
    return df.reset_index(drop=True)


# ═══════════════════════ INDICATORS ════════════════════════════

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema"]          = df["close"].ewm(span=EMA_PERIOD, adjust=False).mean()
    df["sma_ema"]      = df["ema"].rolling(SMA_PERIOD).mean()
    df["ema_above"]    = df["ema"] > df["sma_ema"]
    df["ema_above_prev"] = df["ema_above"].shift(1).fillna(False).astype(bool)
    df["signal_buy"]   = (~df["ema_above_prev"]) & df["ema_above"]
    return df


# ═══════════════════════ STATE ═════════════════════════════════

@dataclass
class Trade:
    entry_time  : str
    entry       : float
    sl          : float
    tp          : float
    sl_dist     : float
    risk_usd    : float
    qty         : float
    trigger_low : float

@dataclass
class Stats:
    total   : int   = 0
    wins    : int   = 0
    losses  : int   = 0
    net_pnl : float = 0.0
    capital : float = CAPITAL

    @property
    def win_rate(self) -> float:
        return self.wins / self.total * 100 if self.total else 0.0

    @property
    def net_r(self) -> float:
        return self.wins * RISK_REWARD - self.losses * 1.0


# ═══════════════════════ ALERTS ════════════════════════════════

def fmt_time() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

def alert_startup(stats: Stats):
    msg = (
        "🚀 <b>SOL/USDT Live Signal Bot Started</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 Strategy  : {EMA_PERIOD} EMA × {SMA_PERIOD} SMA(EMA)\n"
        f"⏱ Timeframe : {INTERVAL}\n"
        f"🎯 R:R       : 1 : {RISK_REWARD}\n"
        f"🛡 SL Buffer : {SL_BUFFER_PCT}%\n"
        f"💰 Risk/Trade: "
        + (f"${RISK_FIXED_USD} fixed" if RISK_MODE == "fixed" else f"{RISK_PCT}% of capital")
        + f"\n⏰ Started   : {fmt_time()}\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "Watching for crossover signals..."
    )
    tg_send(msg)


def alert_signal(trade: Trade, stats: Stats):
    msg = (
        "🟢 <b>BUY SIGNAL — SOLUSDT 1m</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"⏰ Time     : {trade.entry_time}\n"
        f"📈 Entry    : <b>${trade.entry:.4f}</b>\n"
        f"🛑 SL       : ${trade.sl:.4f}  "
        f"({((trade.entry - trade.sl) / trade.entry * 100):.3f}% below)\n"
        f"🎯 TP       : ${trade.tp:.4f}  "
        f"({((trade.tp - trade.entry) / trade.entry * 100):.3f}% above)\n"
        f"📐 SL Dist  : ${trade.sl_dist:.4f}\n"
        f"💲 Risk     : ${trade.risk_usd:.2f}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 Session  : {stats.total} trades  "
        f"| {stats.wins}W / {stats.losses}L"
    )
    tg_send(msg)


def alert_trade_closed(trade: Trade, exit_price: float,
                        exit_reason: str, pnl_usd: float,
                        pnl_r: float, stats: Stats):
    emoji  = "✅" if exit_reason == "TP" else "❌"
    result = "WIN  (TP HIT)" if exit_reason == "TP" else "LOSS (SL HIT)"
    pnl_sign = "+" if pnl_usd >= 0 else ""

    msg = (
        f"{emoji} <b>TRADE CLOSED — {result}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"⏰ Closed   : {fmt_time()}\n"
        f"📈 Entry    : ${trade.entry:.4f}\n"
        f"📉 Exit     : ${exit_price:.4f}\n"
        f"💵 P&L      : <b>{pnl_sign}${pnl_usd:.2f}  ({pnl_sign}{pnl_r:.2f}R)</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 <b>Running Stats</b>\n"
        f"   Total Trades : {stats.total}\n"
        f"   Wins / Losses: {stats.wins} / {stats.losses}\n"
        f"   Win Rate     : {stats.win_rate:.1f}%\n"
        f"   Net P&L      : {'+' if stats.net_pnl >= 0 else ''}${stats.net_pnl:.2f}\n"
        f"   Net R        : {'+' if stats.net_r >= 0 else ''}{stats.net_r:.1f} R\n"
        f"   Capital      : ${stats.capital:.2f}"
    )
    tg_send(msg)


# ═══════════════════════ CSV LOG ═══════════════════════════════

CSV_PATH = "trades_live.csv"
CSV_HEADERS = [
    "entry_time","exit_time","entry","sl","tp","exit_price",
    "sl_dist","qty","risk_usd","pnl_usd","pnl_r","result","capital"
]

def log_trade_csv(trade: Trade, exit_price: float, exit_reason: str,
                  pnl_usd: float, pnl_r: float, capital: float):
    if not SAVE_LOG_CSV:
        return
    write_header = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        if write_header:
            w.writeheader()
        w.writerow({
            "entry_time"  : trade.entry_time,
            "exit_time"   : fmt_time(),
            "entry"       : round(trade.entry, 4),
            "sl"          : round(trade.sl, 4),
            "tp"          : round(trade.tp, 4),
            "exit_price"  : round(exit_price, 4),
            "sl_dist"     : round(trade.sl_dist, 4),
            "qty"         : round(trade.qty, 4),
            "risk_usd"    : round(trade.risk_usd, 2),
            "pnl_usd"     : round(pnl_usd, 2),
            "pnl_r"       : round(pnl_r, 3),
            "result"      : exit_reason,
            "capital"     : round(capital, 2),
        })


# ═══════════════════════ MAIN LOOP ═════════════════════════════

def compute_risk(capital: float) -> float:
    if RISK_MODE == "fixed":
        return RISK_FIXED_USD
    return capital * (RISK_PCT / 100)


def main():
    print("╔══════════════════════════════════════════════╗")
    print("║  SOL/USDT  9EMA × 9SMA(EMA)  Live Tracker   ║")
    print("╚══════════════════════════════════════════════╝")
    print(f"  Telegram chat(s) : {TG_CHAT_IDS}")
    print(f"  Risk mode        : {RISK_MODE}")
    print(f"  Poll interval    : {POLL_SECONDS}s\n")

    stats        : Stats         = Stats()
    open_trade   : Optional[Trade] = None
    last_candle_time              = None  # track last processed candle

    # Send startup ping
    if SEND_STARTUP:
        print("  Sending startup message to Telegram...")
        alert_startup(stats)
        print("  ✓ Startup message sent.\n")

    print("  Entering live loop. Press Ctrl+C to stop.\n")

    while True:
        try:
            # ── 1. Fetch latest candles ──────────────────────
            needed = max(WARMUP_CANDLES, EMA_PERIOD + SMA_PERIOD + 10)
            df = fetch_candles(SYMBOL, INTERVAL, needed)
            df = add_indicators(df)

            # Use the LAST CLOSED candle (second to last row — last may still be forming)
            closed = df.iloc[:-1].copy().reset_index(drop=True)
            latest = closed.iloc[-1]
            candle_time = latest["open_time"]

            # ── 2. Check open trade for SL / TP ─────────────
            if open_trade is not None:
                low  = latest["low"]
                high = latest["high"]

                exit_price = exit_reason = None

                if low <= open_trade.sl:
                    exit_price  = open_trade.sl
                    exit_reason = "SL"
                elif high >= open_trade.tp:
                    exit_price  = open_trade.tp
                    exit_reason = "TP"

                if exit_reason:
                    pnl_usd  = (exit_price - open_trade.entry) * open_trade.qty
                    pnl_r    = pnl_usd / open_trade.risk_usd
                    stats.total   += 1
                    stats.net_pnl += pnl_usd
                    stats.capital += pnl_usd
                    if exit_reason == "TP":
                        stats.wins += 1
                    else:
                        stats.losses += 1

                    print(f"  [{fmt_time()}] Trade CLOSED → {exit_reason}  "
                          f"P&L: {'+' if pnl_usd >= 0 else ''}${pnl_usd:.2f} "
                          f"({'+' if pnl_r >= 0 else ''}{pnl_r:.2f}R)")

                    alert_trade_closed(open_trade, exit_price, exit_reason,
                                       pnl_usd, pnl_r, stats)
                    log_trade_csv(open_trade, exit_price, exit_reason,
                                  pnl_usd, pnl_r, stats.capital)
                    open_trade = None

            # ── 3. Skip if we already processed this candle ──
            if candle_time == last_candle_time:
                time.sleep(POLL_SECONDS)
                continue
            last_candle_time = candle_time

            # ── 4. Check for new BUY signal ──────────────────
            if open_trade is None and latest["signal_buy"]:
                # Entry on NEXT candle open — which is the current (still-forming) candle
                next_open = df.iloc[-1]["open"]    # current candle's open price
                trig_low  = latest["low"]

                raw_sl_dist = next_open - trig_low
                if raw_sl_dist > 0:
                    sl_dist  = raw_sl_dist * (1 + SL_BUFFER_PCT / 100)
                    sl       = next_open - sl_dist
                    tp       = next_open + sl_dist * RISK_REWARD
                    risk_usd = compute_risk(stats.capital)
                    qty      = risk_usd / sl_dist

                    open_trade = Trade(
                        entry_time  = fmt_time(),
                        entry       = next_open,
                        sl          = sl,
                        tp          = tp,
                        sl_dist     = sl_dist,
                        risk_usd    = risk_usd,
                        qty         = qty,
                        trigger_low = trig_low,
                    )

                    print(f"  [{fmt_time()}] BUY SIGNAL  "
                          f"Entry ${next_open:.4f}  SL ${sl:.4f}  TP ${tp:.4f}")
                    alert_signal(open_trade, stats)

            else:
                ema_val  = latest["ema"]
                sma_val  = latest["sma_ema"]
                cur_px   = latest["close"]
                in_trade = "IN TRADE" if open_trade else "watching"
                print(f"  [{candle_time.strftime('%H:%M')}]  "
                      f"Price ${cur_px:.3f}  "
                      f"EMA {ema_val:.3f}  SMA {sma_val:.3f}  "
                      f"| {in_trade}")

        except KeyboardInterrupt:
            print("\n\n  Stopped by user.")
            summary = (
                "🛑 <b>Live Bot Stopped</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"📊 Total Trades : {stats.total}\n"
                f"✅ Wins         : {stats.wins}\n"
                f"❌ Losses       : {stats.losses}\n"
                f"📈 Win Rate     : {stats.win_rate:.1f}%\n"
                f"💵 Net P&L      : {'+' if stats.net_pnl >= 0 else ''}${stats.net_pnl:.2f}\n"
                f"📐 Net R        : {'+' if stats.net_r >= 0 else ''}{stats.net_r:.1f} R\n"
                f"💰 Capital      : ${stats.capital:.2f}"
            )
            tg_send(summary)
            print("  Final summary sent to Telegram. Goodbye.\n")
            break

        except requests.exceptions.RequestException as e:
            print(f"  [NET ERROR] {e} — retrying in {POLL_SECONDS}s")
            time.sleep(POLL_SECONDS)
            continue

        except Exception as e:
            print(f"  [ERROR] {e}")
            time.sleep(POLL_SECONDS)
            continue

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
