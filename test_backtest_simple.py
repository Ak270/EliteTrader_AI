import pandas as pd
import numpy as np
import yaml

from data.downloaders.universal_downloader import get_data
from data.processors.feature_engineer import add_features
from strategies.options_selling import OptionsPremiumStrategy
from backtesting.backtest_engine import BacktestEngine

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

print("Simple Backtest Test\n")

# Get data
df = get_data('BANKNIFTY', '2024-01-01', '2025-11-01')
df = add_features(df)

# Generate signals
strategy = OptionsPremiumStrategy(config)
df = strategy.generate_signals(df)

print(f"Signals generated: {df['Signal'].abs().sum()}")
print(f"Buy signals: {(df['Signal'] == 1).sum()}")
print(f"Sell signals: {(df['Signal'] == -1).sum()}\n")

# Run backtest
engine = BacktestEngine(config)
pos_sizes = pd.Series([config['trading']['capital']] * len(df))
metrics = engine.run(df, df[['Signal']], pos_sizes, "Options Premium Test")

print(f"\n✓ Backtest complete!")
print(f"Total trades executed: {metrics['total_trades']}")
print(f"Final capital: ₹{metrics['ending_capital']:,.0f}")
print(f"Return: {metrics['return_pct']:+.2f}%")
