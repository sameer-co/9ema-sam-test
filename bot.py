import os
import time
import numpy as np
import pandas as pd
import ccxt
from backtesting import Backtest, Strategy

# ==========================================
# 1. Crash-Proof Indicator Calculations
# ==========================================
def RSI(series, period=14):
    """Wilder's Relative Strength Index (Pandas Compatible)"""
    # Force conversion of backtesting's _Array to a Pandas Series
    series = pd.Series(series)
    
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    # Standard Wilder's EMA smoothing
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def WMA(series, period=14):
    """Weighted Moving Average (Pandas Compatible)"""
    # Force conversion of backtesting's _Array to a Pandas Series
    series = pd.Series(series)
    
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
        # self.I safely wraps indicators for backtesting.py
        self.rsi = self.I(RSI, self.data.Close, self.rsi_period)
        self.wma = self.I(WMA, self.data.Close, self.wma_period)
        
    def next(self):
        price = self.data.Close[-1]
        
        # Simple trading logic (Adjust as needed)
        if not self.position:
            if self.rsi[-1] < 30 and price > self.wma[-1]:
                self.buy()
        else:
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
    total_needed = days * 24 * 60  # 43,200 candles for 30 days of 1-minute bars
    
    while len(all_candles) < total_needed:
        try:
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
            
            # Rate limit buffer
            time.sleep(exchange.rateLimit / 1000)
            
        except Exception as e:
            print(f"Error downloading data: {e}. Retrying in 5 seconds...")
            time.sleep(5)
            
    print(f"Successfully downloaded {len(all_candles)} candles.")
    
    df = pd.DataFrame(all_candles, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df.set_index('Timestamp', inplace=True)
    
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    return df

# ==========================================
# 4. Main Execution Block
# ==========================================
if __name__ == "__main__":
    initial_cash = 10000.0
    
    # Get the raw historical data
    df = fetch_historical_data(symbol="SOL/USDT", timeframe="1m", days=30)
    
    # Initialize Backtest
    bt = Backtest(df, ScalperStrategy, cash=initial_cash, commission=0.0006, exclusive_orders=True)
    stats = bt.run()
    
    # Safe extraction of Starting Capital from equity curve to prevent crashes
    starting_capital = (
        stats['_equity_curve']['Equity'].iloc[0] 
        if '_equity_curve' in stats and not stats['_equity_curve'].empty 
        else initial_cash
    )
    
    # Safe terminal logging block
    print("\n" + "="*40)
    print("             BACKTEST RESULTS             ")
    print("="*40)
    print(f"Total Trades Taken : {int(stats.get('# Trades', 0))}")
    print(f"Winning Trades (%) : {stats.get('Win Rate [%]', 0.0):.2f}%")
    print(f"Starting Capital   : ${starting_capital:.2f}")
    print(f"Ending Capital     : ${stats.get('Equity Final [$]', 0.0):.2f}")
    print(f"Peak Capital       : ${stats.get('Equity Peak [$]', 0.0):.2f}")
    print(f"Total Return       : {stats.get('Return [%]', 0.0):.2f}%")
    print(f"Max Drawdown       : {stats.get('Max. Drawdown [%]', 0.0):.2f}%")
    print("="*40 + "\n")
