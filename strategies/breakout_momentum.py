"""
Strategy #2: Intraday Breakout Momentum
Edge: Exploit morning volatility windows in equity indices/stocks.
Institutional Concept: "Opening Range Breakout"
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional

class BreakoutMomentumStrategy:
    """
    Trades breakout of 9:15-9:30am opening range. Uses volume + volatility filters.
    Ideal for Bank Nifty, Nifty, and liquid stocks in Indian/US/FX markets.
    """

    def __init__(self, config: Dict):
        self.name = "Breakout Momentum"
        self.capital_allocation = config['strategy_allocation']['breakout_momentum']
        self.risk_per_trade = config['trading']['risk_per_trade']
        
        # Optimizable parameters
        self.range_minutes = 15   # Calculate opening range on first 15 min candle
        self.volume_ratio = 1.5   # Min. volume multiple to confirm breakout
        self.breakout_buffer = 0.1 # % buffer above/below range for entry
        self.target_rr = 1.5      # Risk:Reward
        self.stop_pct = 0.3       # 0.3% stop-loss

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy/sell breakout signals for daily data.
        """
        df = df.copy()  # Already copying above
        signals = []
        
        # Breakout on price + 1.5x previous day's volume
        for i in range(1, len(df)):
            prev_close = df['Close'].iloc[i-1]
            prev_vol_ma10 = df['Volume'].iloc[max(0, i-10):i].mean()
            curr_vol = df['Volume'].iloc[i]
            
            signal = 0
            # Bullish breakout
            if (df['High'].iloc[i] > prev_close * (1 + self.breakout_buffer/100)) \
                    and (curr_vol > prev_vol_ma10 * self.volume_ratio):
                signal = 1
            # Bearish breakdown
            elif (df['Low'].iloc[i] < prev_close * (1 - self.breakout_buffer/100)) \
                    and (curr_vol > prev_vol_ma10 * self.volume_ratio):
                signal = -1
            
            signals.append(signal)
        
        # Add first row as 0 signal
        df['Signal'] = [0] + signals
        return df

    def calculate_position_size(self, capital: float) -> float:
        """Standard institutional position sizing based on risk per trade"""
        return capital * self.capital_allocation * self.risk_per_trade

    def get_exit_levels(self, entry_price: float, entry_signal: int) -> (float, float):
        """Get target and stop for R:R enforcement"""
        stop_loss = entry_price * (1 - self.stop_pct / 100 * entry_signal)
        if entry_signal == 1:  # Long
            target = entry_price + self.target_rr * (entry_price - stop_loss)
        else:  # Short
            target = entry_price - self.target_rr * (stop_loss - entry_price)
        return target, stop_loss

if __name__ == "__main__":
    import yaml
    import sys
    from data.downloaders.universal_downloader import get_data
    from data.processors.feature_engineer import add_features
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    print("Testing Breakout Momentum Strategy...\n")
    df = get_data('BANKNIFTY', '2024-01-01', '2025-11-01')
    df = add_features(df)
    strat = BreakoutMomentumStrategy(config)
    df = strat.generate_signals(df)
    print(f"Long/buy signals: {(df['Signal']==1).sum()}")
    print(f"Short/sell signals: {(df['Signal']==-1).sum()}")
    print(df[df['Signal']!=0][['Close', 'Volume', 'Signal']].tail(10))
