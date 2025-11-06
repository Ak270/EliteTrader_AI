import pandas as pd
import yaml
import sys
sys.path.append('.')

from data.downloaders.universal_downloader import get_data
from data.processors.feature_engineer import add_features
from strategies.options_selling import OptionsPremiumStrategy
from strategies.breakout_momentum import BreakoutMomentumStrategy
from strategies.mean_reversion import MeanReversionStrategy

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

print("Testing Signal Generation...\n")

# Get data
df = get_data('BANKNIFTY', '2024-01-01', '2025-11-01')
df = add_features(df)

# Test each strategy
strategies = {
    'Options Premium': OptionsPremiumStrategy(config),
    'Breakout Momentum': BreakoutMomentumStrategy(config),
    'Mean Reversion': MeanReversionStrategy(config),
}

for name, strategy in strategies.items():
    df_test = df.copy()
    df_test = strategy.generate_signals(df_test)
    
    buy_signals = (df_test['Signal'] == 1).sum()
    sell_signals = (df_test['Signal'] == -1).sum()
    total_signals = buy_signals + sell_signals
    
    print(f"\n{name}")
    print(f"  Buy signals:  {buy_signals}")
    print(f"  Sell signals: {sell_signals}")
    print(f"  Total:        {total_signals}")
    
    if total_signals > 0:
        print(f"  ✓ Strategy generates signals!")
    else:
        print(f"  ✗ WARNING: No signals generated")
    
    print(f"\n  Sample signals (last 10):")
    print(df_test[['Close', 'RSI_14', 'Signal']].tail(10))

