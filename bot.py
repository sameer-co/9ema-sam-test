import datetime
import time
import pandas as pd
import numpy as np
import pandas_ta as ta
import requests
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# =====================================================================
# 1. DECOUPLED INDICATOR HELPER FUNCTIONS
# =====================================================================
def compute_ema(series, length):
    """Calculates clean standard EMA series."""
    return ta.ema(series, length=length)

def compute_sma_of_ema(series, ema_length, sma_length):
    """Calculates EMA first, then applies SMA smoothing to it cleanly."""
    ema = ta.ema(series, length=ema_length)
    return ta.sma(ema, length=sma_length)


# =====================================================================
# 2. HISTORICAL BINANCE DATA EXTRACTION (NO API KEYS REQUIRED)
# =====================================================================
def fetch_binance_1m_data(symbol="SOLUSDT", limit_days=365):
    """
    Fetches historical 1-minute candle data directly from the public Binance API 
    for the specified number of days up to the current timestamp.
    """
    print(f"Fetching {limit_days} days of 1-minute historical data for {symbol} from Binance...")
    
    base_url = "https://api.binance.com/api/v3/klines"
    
    # Define time windows in milliseconds
    end_time = int(time.time() * 1000)
    start_time = end_time - (limit_days * 24 * 60 * 60 * 1000)
    
    all_candles = []
    current_start = start_time
    
    while current_start < end_time:
        params = {
            "symbol": symbol,
            "interval": "1m",
            "startTime": current_start,
            "endTime": end_time,
            "limit": 1000  # Max chunk size supported by Binance public klines
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
        
        # Shift time context forward using the open time of the final candle in the array
        last_candle_time = data[-1][0]
        current_start = last_candle_time + 60000  # Add 1 minute offset
        
        # Display explicit download track status
        progress_pct = min(100.0, ((current_start - start_time) / (end_time - start_time)) * 100)
        print(f"Downloaded up to {datetime.datetime.fromtimestamp(last_candle_time/1000).strftime('%Y-%m-%d %H:%M:%S')} ({progress_pct:.2f}%)", end="\r")
        
        # Anti-rate-limiting pause
        time.sleep(0.1)
        
    print(f"\nSuccessfully downloaded {len(all_candles)} candles.")
    
    # Map fields to match raw Binance structural arrays
    df = pd.DataFrame(all_candles, columns=[
        'OpenTime', 'Open', 'High', 'Low', 'Close', 'Volume',
        'CloseTime', 'QuoteVolume', 'Trades', 'TakerBase', 'TakerQuote', 'Ignore'
    ])
    
    # Set readable Timestamp indexing
    df['Date'] = pd.to_datetime(df['OpenTime'], unit='ms')
    df.set_index('Date', inplace=True)
    
    # Cast target pricing tracks into float structures
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = df[col].astype(float)
        
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]


# =====================================================================
# 3. STRATEGY ENGINE DEFINITION (9 EMA / 9 SMA SYSTEM)
# =====================================================================
class SolEmaSmaStrategy(Strategy):
    def init(self):
        # Safely capture the primary raw series array layout using the internal .s wrapper
        close_series = pd.Series(self.data.Close.s)
        
        # Register indicators safely via standard element arrays to prevent internal indexing conflicts
        self.ema9 = self.I(compute_ema, close_series, 9)
        self.sma9_of_ema = self.I(compute_sma_of_ema, close_series, 9, 9)

    def next(self):
        # Warmup gap check to accommodate indicator resolution ranges
        if len(self.data) < 18:
            return

        # Buy-Only Strategy Model execution rules
        if not self.position:
            # Entry Logic: Trigger order if the 9 EMA crosses above the smoothed SMA line
            if crossover(self.ema9, self.sma9_of_ema):
                entry_price = self.data.Close[-1]
                entry_low = self.data.Low[-1]
                
                # Dynamic Bracket setup using candle metrics
                stop_loss = entry_low
                risk = entry_price - entry_low
                
                # Protection fallback logic if low matches current close pricing
                if risk <= 0:
                    risk = entry_price * 0.001
                    stop_loss = entry_price - risk
                
                # Take profit set at standard double risk scale
                take_profit = entry_price + (2 * risk)
                
                # Sizing set to 0.95 allocates 95% of equity, leaving a 5% margin safety 
                # cushion to naturally pay for exchange taker commission fees without order rejections
                self.buy(size=0.95, sl=stop_loss, tp=take_profit)


# =====================================================================
# 4. EXECUTION PIPELINE
# =====================================================================
if __name__ == "__main__":
    # Fetch 1 year of historical data. (For debugging/quick speed tests, drop limit_days to 10 or 30)
    df = fetch_binance_1m_data(symbol="SOLUSDT", limit_days=365)
    
    # Initialize the backtest engine matching Binance futures parameters
    bt = Backtest(
        df, 
        SolEmaSmaStrategy, 
        cash=10000, 
        commission=0.00075,  # 0.075% standard VIP0 Taker fee scaling
        hedging=False, 
        exclusive_orders=True
    )
    
    # Run the backtest simulation
    stats = bt.run()
    
    # Print clean strategy metrics summary
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
    
    # Print detailed execution audit log
    print("\n--- Trade Execution Log (Buy Only) ---")
    trades = stats['_trades']
    if not trades.empty:
        pd.set_option('display.max_rows', 100)
        # Uses standard internal native fields 'PnL' and 'ReturnPct' to avoid formatting exceptions
        print(trades[['EntryTime', 'ExitTime', 'EntryPrice', 'ExitPrice', 'PnL', 'ReturnPct']])
    else:
        print("No completed trades matched your strategy's entry execution rules.")
