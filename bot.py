import os
import time
import numpy as np
import pandas as pd
import ccxt
from backtesting import Backtest, Strategy

# ==========================================
# 1. Indicator Calculations
# ==========================================
def RSI(series, period=14):
    """Wilder's Relative Strength Index"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def WMA(series, period=14):
    """Weighted Moving Average"""
    weights = np.arange(1, period + 1)
    return series.rolling(period).apply(
        lambda prices: np.dot(prices, weights) / weights.sum(), 
        raw=True
    )

# ==========================================
# 2. Strategy Definition
# ==========================================
class ScalperStrategy(Strategy):
    rsi_period = 14
    wma_period = 20
    
    def init(self):
        # Calculate indicators safely within the backtesting context
        self.rsi = self.I(RSI, self.data.Close, self.rsi_period)
        self.wma = self.I(WMA, self.data.Close, self.wma_period)
        
    def next(self):
        price = self.data.Close[-1]
        
        # --- Entry / Exit signals (Replace with your custom rules) ---
        if not self.position:
            # Example: Buy when RSI is oversold and price is above WMA
            if self.rsi[-1] < 30 and price > self.wma[-1]:
                self.buy()
        else:
            # Example: Exit when RSI gets overbought
            if self.rsi[-1] > 70 or price < self.wma[-1]:
                self.position.close()

# ==========================================
# 3. Safe Historical Data Downloader
# ==========================================
def fetch_historical_data(symbol="SOL/USDT", timeframe="1m", days=30):
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}  # Uses Binance Futures
    })
    
    clean_symbol = symbol.replace("/", "")
    print(f"Fetching {days} days of {timeframe} data for {clean_symbol}...")
    
    # Target 30 days ago
    since = exchange.milliseconds() - (days * 24 * 60 * 60 * 1000)
    all_candles = []
    total_needed = days * 24 * 60  # Exactly 43,200 candles for 30 days of 1-minute bars
    
    while len(all_candles) < total_needed:
        try:
            # Fetch batch (max 1000 items per request)
            candles = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            if not candles:
                break
            
            all_candles.extend(candles)
            since = candles[-1][0] + 60000  # Shift forward by 1 minute
            
            # Progress calculation matching your exact container logs
            percentage = (len(all_candles) / total_needed) * 100
            if percentage > 100.0:
                percentage = 100.0
            print(f"Downloaded... {percentage:.2f}%")
            
            # Avoid hitting rate limits
            time.sleep(exchange.rateLimit / 1000)
            
        except Exception as e:
            print(f"Error downloading data: {e}. Retrying in 5 seconds...")
            time.sleep(5)
            
    print(f"Successfully downloaded {len(all_candles)} candles.")
    
    # Structure into the format backtesting.py expects
    df = pd.DataFrame(all_candles, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df.set_index('Timestamp', inplace=True)
    
    # Ensure standard capitalization
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    return df

# ==========================================
# 4. Main execution and Stats print
# ==========================================
if __name__ == "__main__":
    initial_cash = 10000.0
    
    # Get the raw historical data
    df = fetch_historical_data(symbol="SOL/USDT", timeframe="1m", days=30)
    
    # Initialize Backtest
    bt = Backtest(df, ScalperStrategy, cash=initial_cash, commission=0.0006, exclusive_orders=True)
    stats = bt.run()
    
    # CRITICAL: Safe extract of Starting Capital from the actual equity curve first entry
    starting_capital = (
        stats['_equity_curve']['Equity'].iloc[0] 
        if '_equity_curve' in stats and not stats['_equity_curve'].empty 
        else initial_cash
    )
    
    # Output to console using .get() to prevent unexpected KeyErrors
    print("========================================")
    print("               BACKTEST METRICS            ")
    print("========================================")
    print(f"Total Trades Taken : {int(stats.get('# Trades', 0))}")
    print(f"Winning Trades (%) : {stats.get('Win Rate [%]', 0.0):.2f}%")
    print(f"Starting Capital   : ${starting_capital:.2f}")
    print(f"Ending Capital     : ${stats.get('Equity Final [$]', 0.0):.2f}")
    print(f"Peak Capital       : ${stats.get('Equity Peak [$]', 0.0):.2f}")
    print(f"Total Return       : {stats.get('Return [%]', 0.0):.2f}%")
    print(f"Max Drawdown       : {stats.get('Max. Drawdown [%]', 0.0):.2f}%")
    print("========================================")
