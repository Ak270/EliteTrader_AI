"""
Strategy #3: Currency/Stock Mean Reversion (Pairs or Range Trading)
Edge: Profits from price oscillations around a mean.
"""

import pandas as pd
import numpy as np
from typing import Dict

class MeanReversionStrategy:
    """
    Market-neutral mean reversion system for currencies and stocks.
    Sells at upper band, buys at lower, exits on mean touch.
    """

    def __init__(self, config: Dict):
        self.name = "Mean Reversion"
        self.capital_allocation = config['strategy_allocation']['mean_reversion']
        self.risk_per_trade = config['trading']['risk_per_trade']

        # Optimizable params
        self.window = 20              # Lookback period
        self.entry_z = 1.25           # Z-score entry
        self.exit_z = 0.2             # Z-score exit
        self.stop_pct = 0.6           # 0.6% SL for FX

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Signal: 1 = Long (buy mean reversion), -1 = Short (sell mean reversion), 0 = No trade
        """
        df = df.copy()
        df['Signal'] = 0
        rolling_mean = df['Close'].rolling(self.window).mean()
        rolling_std = df['Close'].rolling(self.window).std()
        zscore = (df['Close'] - rolling_mean) / rolling_std

        # Buy when price is > entry_z std below mean, Sell when > entry_z above
        df.loc[zscore <= -self.entry_z, 'Signal'] = 1
        df.loc[zscore >= self.entry_z, 'Signal'] = -1

        # Exit criteria: revert near mean (|z| < exit_z)
        df['Exit'] = (zscore.abs() < self.exit_z).astype(int)

        return df

    def calculate_position_size(self, capital: float) -> float:
        """Risk 1% of capital per mean reversion position"""
        return capital * self.capital_allocation * self.risk_per_trade

    def get_exit_levels(self, entry_price: float, entry_signal: int) -> (float, float):
        stop_loss = entry_price * (1 - self.stop_pct / 100 * entry_signal)
        exit_target = entry_price  # Exit at mean reversion
        return exit_target, stop_loss

if __name__ == "__main__":
    import yaml
    import sys
    sys.path.append('..')
    from downloaders.universal_downloader import get_data
    from processors.feature_engineer import add_features
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    print("Testing Mean Reversion Strategy...\n")
    df = get_data('USDINR', '2024-01-01', '2025-11-01')
    df = add_features(df)
    strat = MeanReversionStrategy(config)
    df = strat.generate_signals(df)
    print(f"Long signals: {(df['Signal']==1).sum()}")
    print(f"Short signals: {(df['Signal']==-1).sum()}")
    print(df[df['Signal']!=0][['Close', 'Signal', 'Exit']].tail(10))
