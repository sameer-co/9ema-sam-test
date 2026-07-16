import datetime
import time
import pandas as pd
import numpy as np
import pandas_ta as ta
import requests
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# =====================================================================
# 1. LIVE BINANCE DATA FETCH FUNCTION (NO API KEY REQUIRED)
# =====================================================================
def fetch_binance_1m_data(symbol="SOLUSDT", limit_days=365):
    """
    Fetches historical 1-minute candle data from the public Binance API 
    for the specified number of days up to the current moment.
    """
    print(f"Fetching {limit_days} days of 1-minute historical data for {symbol} from Binance...")
    
    base_url = "https://api.binance.com/api/v3/klines"
    
    # Define start and end times in milliseconds
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
            "limit": 1000 # Maximum candles allowed per single request
        }
        
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"Error fetching data: {e}")
            break
            
        if not data:
            break
            
        all_candles.extend(data)
        
        # The last candle's open time is used to offset the next batch forward
        last_candle_time = data[-1][0]
        current_start = last_candle_time + 60000 # Add 1 minute in milliseconds
        
        # Provide user feedback to track download progress
        progress_pct = min(100.0, ((current_start - start_time) / (end_time - start_time)) * 100)
        print(f"Downloaded up to {datetime.datetime.fromtimestamp(last_candle_time/1000).strftime('%Y-%m-%d %H:%M:%S')} ({progress_pct:.2f}%)", end="\r")
        
        # Slight pause to respect public endpoint rate limiting
        time.sleep(0.1)
        
    print(f"\nSuccessfully downloaded {len(all_candles)} candles.")
    
    # Construct DataFrame from raw Binance structure
    # Binance index map: 0=Open Time, 1=Open, 2=High, 3=Low, 4=Close, 5=Volume
    df = pd.DataFrame(all_candles, columns=[
        'OpenTime', 'Open', 'High', 'Low', 'Close', 'Volume',
        'CloseTime', 'QuoteVolume', 'Trades', 'TakerBase', 'TakerQuote', 'Ignore'
    ])
    
    # Process types and convert timestamps to readable Index
    df['Date'] = pd.to_datetime(df['OpenTime'], unit='ms')
    df.set_index('Date', inplace=True)
    
    # Convert numerical columns from strings to float format
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = df[col].astype(float)
        
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]


# =====================================================================
# 2. DEFINE THE 9 EMA / 9 SMA CROSSOVER STRATEGY (BUY-ONLY)
# =====================================================================
class SolEmaSmaStrategy(Strategy):
    def init(self):
        close_series = pd.Series(self.data.Close)
        
        # 1. Calculate 9-period EMA
        self.ema9 = self.I(ta.ema, close_series, length=9)
        
        # 2. Calculate the Smoothed 9 SMA (SMA of the 9 EMA)
        self.sma9_of_ema = self.I(lambda x: ta.sma(pd.Series(x), length=9), self.ema9)

    def next(self):
        if len(self.data) < 18:
            return

        # Buy-only execution model
        if not self.position:
            # Entry condition: 9 EMA crosses above its 9 SMA smoothed line
            if crossover(self.ema9, self.sma9_of_ema):
                entry_price = self.data.Close[-1]
                entry_low = self.data.Low[-1]
                
                # Stop Loss is set below the low of the entry candle
                stop_loss = entry_low
                
                risk = entry_price - entry_low
                
                # Dynamic fallback buffer if entry low is equal to entry close
                if risk <= 0:
                    risk = entry_price * 0.001
                    stop_loss = entry_price - risk
                
                # Target is exactly 2x the distance of the risk profile
                take_profit = entry_price + (2 * risk)
                
                self.buy(sl=stop_loss, tp=take_profit)


# =====================================================================
# 3. RUN STRATEGY PIPELINE
# =====================================================================
if __name__ == "__main__":
    # Fetch 1 year of 1-minute historical data (365 days)
    # Note: For faster testing runs you can reduce limit_days to 30 or 90.
    df = fetch_binance_1m_data(symbol="SOLUSDT", limit_days=365)
    
    # Execute Backtest starting with $10,000 cash and 0.075% taker fee/commission
    bt = Backtest(df, SolEmaSmaStrategy, cash=10000, commission=0.00075)
    stats = bt.run()
    
    # Print Exact Requested Metrics
    print("\n" + "="*40)
    print("             BACKTEST METRICS            ")
    print("="*40)
    print(f"Total Trades Taken : {stats['# Trades']}")
    print(f"Winning Trades (%) : {stats['Win Rate [%]']:.2f}%")
    print(f"Starting Capital   : ${stats['Equity Start [$]']:.2f}")
    print(f"Ending Capital     : ${stats['Equity Final [$]']:.2f}")
    print(f"Net Profit/Loss ($): ${stats['Return [$]']:.2f}")
    print(f"Total Return (%)   : {stats['Return [%]']:.2f}%")
    print(f"Max Drawdown (%)   : {stats['Max. Drawdown [%]']:.2f}%")
    print("="*40)
    
    # Print individual historical trade list with absolute P&L breakdown
    print("\n--- Trade Execution Log (Buy Only) ---")
    trades = stats['_trades']
    if not trades.empty:
        trades['P&L ($)'] = (trades['ExitPrice'] - trades['EntryPrice']) * trades['Size']
        # Show key trade properties
        pd.set_option('display.max_rows', 100)
        print(trades[['EntryTime', 'ExitTime', 'EntryPrice', 'ExitPrice', 'P&L ($)', 'ReturnPct']])
    else:
        print("No completed trades matched your strategy's entry execution rules.")
