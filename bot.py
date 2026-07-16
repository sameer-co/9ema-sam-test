import datetime
import time
import warnings
import pandas as pd
import numpy as np
import pandas_ta as ta
import requests

# 1. CLEAN DEPLOYMENT LOGS: Suppress backtesting engine margin warnings
# This stops your cloud console from being spammed by internal broker rejections
warnings.filterwarnings("ignore", category=UserWarning, module="backtesting")

from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# =====================================================================
# 2. DECOUPLED INDICATOR HELPER FUNCTIONS
# =====================================================================
def compute_ema(series, length):
    return ta.ema(series, length=length)

def compute_sma_of_ema(series, ema_length, sma_length):
    ema = ta.ema(series, length=ema_length)
    return ta.sma(ema, length=sma_length)


# =====================================================================
# 3. HISTORICAL BINANCE DATA EXTRACTION
# =====================================================================
def fetch_binance_1m_data(symbol="SOLUSDT", limit_days=365):
    print(f"Fetching {limit_days} days of 1-minute data for {symbol}...")
    
    base_url = "https://api.binance.com/api/v3/klines"
    end_time = int(time.time() * 1000)
    start_time = end_time - (limit_days * 24 * 60 * 60 * 1000)
    
    all_candles = []
    current_start = start_time
    
    while current_start < end_time:
        params = {
            "symbol": symbol,
            "interval": "5m",
            "startTime": current_start,
            "endTime": end_time,
            "limit": 1000
        }
        
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"Error fetching data batch: {e}")
            break
            
        if not data:
            break
            
        all_candles.extend(data)
        current_start = data[-1][0] + 60000 
        
        # Simple progress tracker
        progress_pct = min(100.0, ((current_start - start_time) / (end_time - start_time)) * 100)
        print(f"Downloaded... {progress_pct:.2f}%", end="\r")
        time.sleep(0.1)
        
    print(f"\nSuccessfully downloaded {len(all_candles)} candles.")
    
    df = pd.DataFrame(all_candles, columns=[
        'OpenTime', 'Open', 'High', 'Low', 'Close', 'Volume',
        'CloseTime', 'QuoteVolume', 'Trades', 'TakerBase', 'TakerQuote', 'Ignore'
    ])
    
    df['Date'] = pd.to_datetime(df['OpenTime'], unit='ms')
    df.set_index('Date', inplace=True)
    
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = df[col].astype(float)
        
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]


# =====================================================================
# 4. STRATEGY ENGINE DEFINITION
# =====================================================================
class SolEmaSmaStrategy(Strategy):
    def init(self):
        close_series = pd.Self.colon(self.data.Close.s)
        self.ema9 = self.I(compute_ema, close_series, 9)
        self.sma9_of_ema = self.I(compute_sma_of_ema, close_series, 9, 9)

    def next(self):
        if len(self.data) < 18:
            return

        # 2. FIX: Check that there are NO active positions AND NO pending orders
        # This prevents duplicate orders from eating up remaining fractional margin
        if not self.position and not self.orders:
            
            if crossover(self.ema9, self.sma9_of_ema):
                entry_price = self.data.Close[-1]
                entry_low = self.data.Low[-1]
                
                stop_loss = entry_low
                risk = entry_price - entry_low
                
                if risk <= 0:
                    risk = entry_price * 0.001
                    stop_loss = entry_price - risk
                
                take_profit = entry_price + (2 * risk)
                
                # Use 95% of equity to leave buffer for fees and price gaps
                self.buy(size=0.95, sl=stop_loss, tp=take_profit)


# =====================================================================
# 5. EXECUTION PIPELINE
# =====================================================================
if __name__ == "__main__":
    # Fetch data (Set to 30 days for faster deployments, scale up as needed)
    df = fetch_binance_1m_data(symbol="SOLUSDT", limit_days=30)
    
    # 3. FIX: Increased starting cash to $100k to ensure fractional sizing 
    # doesn't round down to 0 units on high-price assets
    bt = Backtest(
        df, 
        SolEmaSmaStrategy, 
        cash=100000, 
        commission=0.00075,
        hedging=False, 
        exclusive_orders=True
    )
    
    stats = bt.run()
    
    # Print Exact Metrics
    print("\n" + "="*40)
    print("             BACKTEST METRICS            ")
    print("="*40)
    print(f"Total Trades Taken : {stats['# Trades']}")
    print(f"Winning Trades (%) : {stats['Win Rate [%]']:.2f}%" if not pd.isna(stats['Win Rate [%]']) else "Winning Trades (%) : 0.00%")
    print(f"Starting Capital   : ${stats['Equity Start [$]']:.2f}")
    print(f"Ending Capital     : ${stats['Equity Final [$]']:.2f}")
    print(f"Net Profit/Loss ($): ${stats['Return [$]']:.2f}")
    print(f"Total Return (%)   : {stats['Return [%]']:.2f}%")
    print(f"Max Drawdown (%)   : {stats['Max. Drawdown [%]']:.2f}%")
    print("="*40)
    
    print("\n--- Trade Execution Log (Buy Only) ---")
    trades = stats['_trades']
    if not trades.empty:
        pd.set_option('display.max_rows', 100)
        print(trades[['EntryTime', 'ExitTime', 'EntryPrice', 'ExitPrice', 'PnL', 'ReturnPct']])
    else:
        print("No completed trades matched your strategy's entry execution rules.")
