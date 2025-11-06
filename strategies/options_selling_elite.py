"""
ELITE OPTIONS SELLING STRATEGY
Based on institutional practices:
- Sell OTM options when IV rank > 70 (overpriced)
- Take profits at 50% of collected premium
- Stop loss at 100% of collected premium
- Target: 70% win rate, 2:1 profit factor
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple

class EliteOptionsSelling:
    """Professional options writing system"""
    
    def __init__(self, config: Dict):
        self.name = "Elite Options Selling"
        self.capital_allocation = 0.40  # 40% for options
        self.risk_per_trade = 0.02  # 2% max loss
        
        # Professional parameters
        self.iv_rank_entry = 70  # Enter when IV > 70th percentile
        self.profit_target_pct = 0.50  # Take profit at 50% of premium
        self.stop_loss_pct = 1.0  # Stop at 100% loss (2:1 R:R)
        self.dte_min = 3  # Minimum days to expiry
        self.win_rate_target = 0.70
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate professional options selling signals
        
        Buy Put Spread (bullish): Sell OTM puts when IV rank > 70
        Sell Call Spread (bearish): Sell OTM calls when IV rank > 70 AND price overbought
        """
        df = df.copy()
        df['Signal'] = 0
        df['Trade_Type'] = ''
        df['Entry_Premium'] = 0
        
        for i in range(1, len(df)):
            iv_rank = df['IV_Rank'].iloc[i]
            underlying = df['Underlying'].iloc[i]
            put_premium = df['OTM_Put_Premium'].iloc[i]
            call_premium = df['OTM_Call_Premium'].iloc[i]
            
            # Entry conditions (ONLY enter when IV is HIGH - overpriced)
            if iv_rank > self.iv_rank_entry:
                
                # Sell Put Spread (BULLISH): Collect put premium
                if i > 5:  # Enough history for RSI/trend
                    # Check if underlying is in downtrend (good for selling puts)
                    sma_20 = df['Underlying'].iloc[max(0, i-20):i].mean()
                    if underlying < sma_20:  # Price below MA = oversold potential
                        df.loc[df.index[i], 'Signal'] = 1
                        df.loc[df.index[i], 'Trade_Type'] = 'SELL_PUT'
                        df.loc[df.index[i], 'Entry_Premium'] = put_premium
                
                # Sell Call Spread (BEARISH): Collect call premium
                elif underlying > df['Underlying'].iloc[max(0, i-20):i].mean():
                    df.loc[df.index[i], 'Signal'] = -1
                    df.loc[df.index[i], 'Trade_Type'] = 'SELL_CALL'
                    df.loc[df.index[i], 'Entry_Premium'] = call_premium
        
        return df
    
    def get_exit_levels(self, entry_premium: float) -> Tuple[float, float]:
        """
        Calculate professional exit levels
        
        Profit: 50% of collected premium
        Stop: 100% of collected premium (2:1 R:R)
        """
        profit_target = entry_premium * self.profit_target_pct
        stop_loss = entry_premium * self.stop_loss_pct
        
        return profit_target, stop_loss


# Also create a composite signal that requires multi-confirmation
class CompositeSignalEngine:
    """Combines multiple strategies for higher-confidence signals"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.elite_options = EliteOptionsSelling(config)
        self.min_confirmations = 2  # Need 2+ strategies to agree
    
    def generate_composite_signal(self, df_options: pd.DataFrame, 
                                 df_underlying: pd.DataFrame) -> pd.DataFrame:
        """Generate high-confidence composite signals"""
        
        df = df_options.copy()
        df['Elite_Options_Signal'] = 0
        
        # Get elite options signal
        df_elite = self.elite_options.generate_signals(df)
        df['Elite_Options_Signal'] = df_elite['Signal']
        
        # Multi-confirmation: RSI + IV Rank + Price Action
        df['Signal_Strength'] = 0
        
        for i in range(1, len(df)):
            strength = 0
            
            # Signal 1: IV Rank high (institutional premium is expensive)
            if df['IV_Rank'].iloc[i] > 70:
                strength += 1
            
            # Signal 2: Price moving (ATR high) - good for option selling
            atr = df['Underlying'].iloc[max(0, i-14):i].std() * 2
            if atr > df['Underlying'].iloc[i] * 0.005:  # ATR > 0.5%
                strength += 1
            
            # Signal 3: Days to expiry suitable
            if df['DTE'].iloc[i] >= self.min_confirmations:
                strength += 1
            
            df.loc[df.index[i], 'Signal_Strength'] = strength
        
        # Only generate signals when strength >= min_confirmations
        df['Composite_Signal'] = 0
        df.loc[df['Signal_Strength'] >= self.min_confirmations, 'Composite_Signal'] = \
            df.loc[df['Signal_Strength'] >= self.min_confirmations, 'Elite_Options_Signal']
        
        return df


if __name__ == "__main__":
    import yaml
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    from data.downloaders.options_downloader import OptionsDataDownloader
    
    print("Testing Elite Options Strategy\n")
    
    # Download options data
    downloader = OptionsDataDownloader()
    df_options = downloader.download_banknifty_options('2024-01-01', '2025-11-01')
    
    # Generate signals
    strategy = EliteOptionsSelling(config)
    df_options = strategy.generate_signals(df_options)
    
    # Count signals
    sell_puts = (df_options['Trade_Type'] == 'SELL_PUT').sum()
    sell_calls = (df_options['Trade_Type'] == 'SELL_CALL').sum()
    
    print(f"Sell Put signals: {sell_puts}")
    print(f"Sell Call signals: {sell_calls}")
    print(f"Total signals: {sell_puts + sell_calls}\n")
    
    print("Sample signals (last 10):")
    print(df_options[['Date', 'IV_Rank', 'Trade_Type', 'Entry_Premium']].tail(10))
