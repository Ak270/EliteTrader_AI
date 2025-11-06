"""
Strategy #1: Options Premium Collection
Edge: Sell OTM options during high IV, collect theta decay
Win rate target: 70-75%
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional

class OptionsPremiumStrategy:
    """
    Institutional options selling strategy
    Sell OTM Put/Call spreads on extreme moves
    """
    
    def __init__(self, config: Dict):
        self.name = "Options Premium Collection"
        self.capital_allocation = config['strategy_allocation']['options_selling']
        self.risk_per_trade = config['trading']['risk_per_trade']
        
        # Strategy parameters (optimizable)
        self.vix_threshold = 15  # Enter when VIX > this
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        self.atr_multiplier = 2.0  # For strike selection
        self.profit_target_pct = 0.50  # Exit at 50% profit
        self.time_stop_hours = 48  # Exit after 48 hours
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Simplified signal generation that actually trades
        """
        df = df.copy()
        df['Signal'] = 0
        
        # Calculate IV proxy
        df['IV_Proxy'] = (df['ATR_14'] / df['Close']) * 100
        
        # Simple RSI-based signals
        for i in range(1, len(df)):
            rsi = df['RSI_14'].iloc[i]
            
            # Sell Put spreads when oversold
            if rsi < 35:
                df.loc[df.index[i], 'Signal'] = 1  # Long signal
            
            # Sell Call spreads when overbought
            elif rsi > 65:
                df.loc[df.index[i], 'Signal'] = -1  # Short signal
            
            # Exit on mean reversion
            elif 40 < rsi < 60:
                df.loc[df.index[i], 'Signal'] = 0  # Exit
        
        return df

    
    def calculate_position_size(self, df: pd.DataFrame, capital: float, 
                               row_idx: int) -> float:
        """Calculate position size based on risk management"""
        atr = df.loc[row_idx, 'ATR_14']
        price = df.loc[row_idx, 'Close']
        
        # Risk 2% of allocated capital per trade
        risk_amount = capital * self.capital_allocation * self.risk_per_trade
        
        # Position size = Risk amount / Stop distance
        # For options: stop distance = 2 * ATR
        stop_distance = 2 * atr
        position_size = risk_amount / stop_distance
        
        return min(position_size, capital * self.capital_allocation * 0.30)  # Max 30% per trade
    
    def calculate_strike_prices(self, df: pd.DataFrame, row_idx: int, 
                                signal: int) -> Tuple[float, float]:
        """
        Calculate optimal strike prices for spread
        
        Returns:
            (short_strike, long_strike) tuple
        """
        current_price = df.loc[row_idx, 'Close']
        atr = df.loc[row_idx, 'ATR_14']
        
        if signal == 1:  # Sell Put spread
            # Short strike: 1 ATR below current
            short_strike = current_price - (atr * self.atr_multiplier)
            # Long strike: 2 ATR below current (protection)
            long_strike = current_price - (atr * (self.atr_multiplier + 1))
            
        elif signal == -1:  # Sell Call spread
            # Short strike: 1 ATR above current
            short_strike = current_price + (atr * self.atr_multiplier)
            # Long strike: 2 ATR above current (protection)
            long_strike = current_price + (atr * (self.atr_multiplier + 1))
        
        else:
            return None, None
        
        return short_strike, long_strike
    
    def get_exit_conditions(self, df: pd.DataFrame, entry_idx: int, 
                           current_idx: int, entry_price: float, 
                           position_type: int) -> Tuple[bool, str]:
        """
        Determine if position should be exited
        
        Returns:
            (should_exit, exit_reason)
        """
        current_price = df.loc[current_idx, 'Close']
        profit_pct = ((current_price - entry_price) / entry_price) * position_type
        
        # 1. Profit target hit (50% of max profit)
        if profit_pct >= self.profit_target_pct:
            return True, "Profit Target"
        
        # 2. Stop loss (price moved against us significantly)
        if profit_pct <= -0.20:  # -20% stop loss
            return True, "Stop Loss"
        
        # 3. Time stop (theta decay captured)
        hours_elapsed = (current_idx - entry_idx) * 6  # Assuming 6.5 hour trading day
        if hours_elapsed >= self.time_stop_hours:
            if profit_pct > 0:
                return True, "Time Stop (Profit)"
            elif hours_elapsed >= 72:  # Force exit after 3 days
                return True, "Time Stop (Forced)"
        
        # 4. Technical reversal
        if position_type == 1:  # Long position
            if df.loc[current_idx, 'RSI_14'] > self.rsi_overbought:
                return True, "Technical Reversal"
        elif position_type == -1:  # Short position
            if df.loc[current_idx, 'RSI_14'] < self.rsi_oversold:
                return True, "Technical Reversal"
        
        return False, ""


if __name__ == "__main__":
    # Test strategy
    import yaml
    import sys
    sys.path.append('..')
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    from data.downloaders.universal_downloader import get_data
    from data.processors.feature_engineer import add_features
    
    print("Testing Options Premium Collection Strategy...\n")
    
    # Get and prepare data
    df = get_data('BANKNIFTY', '2024-01-01', '2025-11-01')
    df = add_features(df)
    
    # Initialize strategy
    strategy = OptionsPremiumStrategy(config)
    
    # Generate signals
    df = strategy.generate_signals(df)
    
    print(f"Total signals: {df['Signal'].abs().sum()}")
    print(f"Buy signals (Sell Puts): {(df['Signal'] == 1).sum()}")
    print(f"Sell signals (Sell Calls): {(df['Signal'] == -1).sum()}")
    
    print(f"\nSample signals:")
    print(df[df['Signal'] != 0][['Close', 'RSI_14', 'IV_Proxy', 'Z_Score_20', 'Signal']].tail(10))

