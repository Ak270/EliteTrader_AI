"""
Feature Engineering - NATIVE implementation (No pandas_ta dependency)
All indicators built from scratch using NumPy/Pandas
Perfect for Mac M2 + Python 3.11
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import sys
from ..downloaders.universal_downloader import get_data

class FeatureEngineer:
    """Professional feature engineering using native NumPy/Pandas"""
    
    def __init__(self):
        self.features_created = []
    
    def add_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all institutional-grade features to dataframe"""
        print("\n" + "="*70)
        print("FEATURE ENGINEERING (NATIVE IMPLEMENTATION)")
        print("="*70)
        
        df = df.copy()
        
        # 1. Trend Indicators
        print("Adding trend indicators...")
        df = self._add_trend_features(df)
        
        # 2. Momentum Indicators
        print("Adding momentum indicators...")
        df = self._add_momentum_features(df)
        
        # 3. Volatility Indicators
        print("Adding volatility indicators...")
        df = self._add_volatility_features(df)
        
        # 4. Volume Indicators
        print("Adding volume indicators...")
        df = self._add_volume_features(df)
        
        # 5. Market Structure
        print("Adding market structure features...")
        df = self._add_structure_features(df)
        
        # 6. Price Action Patterns
        print("Adding price action patterns...")
        df = self._add_pattern_features(df)
        
        # 7. Statistical Features
        print("Adding statistical features...")
        df = self._add_statistical_features(df)
        
        # Drop NaN rows
        initial_rows = len(df)
        # Only drop rows where essential core features are NaN
        core_features = ['Close', 'SMA_20', 'RSI_14', 'MACD', 'ATR_14', 'BB_Middle']
        df = df.dropna(subset=[f for f in core_features if f in df.columns])
        dropped_rows = initial_rows - len(df)
        
        print(f"\n✓ Feature engineering complete!")
        print(f"  Total features: {len(self.features_created)}")
        print(f"  Valid rows: {len(df)} (dropped {dropped_rows} NaN rows)")
        print("="*70 + "\n")
        
        return df
    
    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trend-following indicators"""
        # Simple Moving Averages
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'SMA_{period}'] = df['Close'].rolling(period).mean()
            self.features_created.append(f'SMA_{period}')
        
        # Exponential Moving Averages
        for period in [8, 13, 21, 55]:
            df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
            self.features_created.append(f'EMA_{period}')
        
        # ADX (Average Directional Index)
        df = self._calculate_adx(df)
        
        # Supertrend
        df = self._calculate_supertrend(df)
        
        return df
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate ADX (Average Directional Index)"""
        high_diff = df['High'].diff()
        low_diff = -df['Low'].diff()
        
        # True Range
        tr1 = df['High'] - df['Low']
        tr2 = abs(df['High'] - df['Close'].shift())
        tr3 = abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        
        # Directional Movement
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        
        # Directional Indicators
        plus_di = 100 * (pd.Series(plus_dm).rolling(period).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm).rolling(period).mean() / atr)
        
        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        
        df['ADX'] = adx
        df['DI_Plus'] = plus_di
        df['DI_Minus'] = minus_di
        self.features_created.extend(['ADX', 'DI_Plus', 'DI_Minus'])
        
        return df
    
    def _calculate_supertrend(self, df: pd.DataFrame, period: int = 10, 
                             multiplier: float = 3.0) -> pd.DataFrame:
        """Calculate Supertrend indicator"""
        hl_avg = (df['High'] + df['Low']) / 2
        atr = self._calculate_atr(df, period)
        
        upper_band = hl_avg + (multiplier * atr)
        lower_band = hl_avg - (multiplier * atr)
        
        supertrend = pd.Series(index=df.index, dtype='float64')
        direction = pd.Series(index=df.index, dtype='int64')
        
        for i in range(len(df)):
            if i == 0:
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = -1
            else:
                if df['Close'].iloc[i] <= upper_band.iloc[i]:
                    supertrend.iloc[i] = upper_band.iloc[i]
                    direction.iloc[i] = -1
                else:
                    supertrend.iloc[i] = lower_band.iloc[i]
                    direction.iloc[i] = 1
        
        df['Supertrend'] = supertrend
        df['Supertrend_Dir'] = direction
        self.features_created.extend(['Supertrend', 'Supertrend_Dir'])
        
        return df
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        tr1 = df['High'] - df['Low']
        tr2 = abs(df['High'] - df['Close'].shift())
        tr3 = abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Momentum oscillators"""
        # RSI (Relative Strength Index)
        for period in [9, 14, 21]:
            df[f'RSI_{period}'] = self._calculate_rsi(df['Close'], period)
            self.features_created.append(f'RSI_{period}')
        
        # MACD (Moving Average Convergence Divergence)
        df = self._calculate_macd(df)
        
        # Stochastic
        df = self._calculate_stochastic(df)
        
        # Rate of Change
        for period in [5, 10, 20]:
            df[f'ROC_{period}'] = df['Close'].pct_change(period) * 100
            self.features_created.append(f'ROC_{period}')
        
        # Williams %R
        df['Williams_R'] = self._calculate_williams_r(df)
        self.features_created.append('Williams_R')
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        deltas = prices.diff()
        gains = deltas.where(deltas > 0, 0)
        losses = -deltas.where(deltas < 0, 0)
        
        avg_gain = gains.rolling(period).mean()
        avg_loss = losses.rolling(period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD"""
        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        
        df['MACD'] = ema_12 - ema_26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        self.features_created.extend(['MACD', 'MACD_Signal', 'MACD_Hist'])
        
        return df
    
    def _calculate_stochastic(self, df: pd.DataFrame, period: int = 14, 
                             smooth_k: int = 3, smooth_d: int = 3) -> pd.DataFrame:
        """Calculate Stochastic Oscillator"""
        low_min = df['Low'].rolling(period).min()
        high_max = df['High'].rolling(period).max()
        
        stoch_k = ((df['Close'] - low_min) / (high_max - low_min)) * 100
        df['Stoch_K'] = stoch_k.rolling(smooth_k).mean()
        df['Stoch_D'] = df['Stoch_K'].rolling(smooth_d).mean()
        
        self.features_created.extend(['Stoch_K', 'Stoch_D'])
        
        return df
    
    def _calculate_williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R"""
        high_max = df['High'].rolling(period).max()
        low_min = df['Low'].rolling(period).min()
        
        wr = -100 * (high_max - df['Close']) / (high_max - low_min)
        
        return wr
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volatility indicators"""
        # ATR
        for period in [7, 14, 21]:
            df[f'ATR_{period}'] = self._calculate_atr(df, period)
            self.features_created.append(f'ATR_{period}')
        
        # Bollinger Bands
        df = self._calculate_bollinger_bands(df)
        
        # Historical Volatility
        df['HV_20'] = df['Close'].pct_change().rolling(20).std() * np.sqrt(252)
        self.features_created.append('HV_20')
        
        return df
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20, 
                                   std_dev: float = 2.0) -> pd.DataFrame:
        """Calculate Bollinger Bands"""
        df['BB_Middle'] = df['Close'].rolling(period).mean()
        bb_std = df['Close'].rolling(period).std()
        
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * std_dev)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * std_dev)
        
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        self.features_created.extend(['BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_Width', 'BB_Position'])
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volume-based indicators"""
        # Volume Moving Averages
        for period in [10, 20, 50]:
            df[f'Vol_MA_{period}'] = df['Volume'].rolling(period).mean()
            df[f'Vol_Ratio_{period}'] = df['Volume'] / df[f'Vol_MA_{period}']
            self.features_created.extend([f'Vol_MA_{period}', f'Vol_Ratio_{period}'])
        
        # OBV (On Balance Volume)
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        self.features_created.append('OBV')
        
        # VWAP (Volume Weighted Average Price)
        df['VWAP'] = (df['Close'] * df['Volume']).rolling(20).sum() / df['Volume'].rolling(20).sum()
        self.features_created.append('VWAP')
        
        # Money Flow Index
        df['MFI'] = self._calculate_mfi(df)
        self.features_created.append('MFI')
        
        return df
    
    def _calculate_mfi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Money Flow Index"""
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        money_flow = typical_price * df['Volume']
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(), 0).rolling(period).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(), 0).rolling(period).sum()
        
        mfi = 100 * positive_flow / (positive_flow + negative_flow)
        
        return mfi
    
    def _add_structure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Market structure features"""
        # Pivot Points
        df['Pivot'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['R1'] = 2 * df['Pivot'] - df['Low']
        df['S1'] = 2 * df['Pivot'] - df['High']
        df['R2'] = df['Pivot'] + (df['High'] - df['Low'])
        df['S2'] = df['Pivot'] - (df['High'] - df['Low'])
        self.features_created.extend(['Pivot', 'R1', 'S1', 'R2', 'S2'])
        
        # Distance from MA
        df['Dist_SMA20'] = ((df['Close'] - df['SMA_20']) / df['SMA_20']) * 100
        df['Dist_SMA50'] = ((df['Close'] - df['SMA_50']) / df['SMA_50']) * 100
        self.features_created.extend(['Dist_SMA20', 'Dist_SMA50'])
        
        return df
    
    def _add_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Price action patterns"""
        # Candle body
        df['Body'] = abs(df['Close'] - df['Open'])
        df['Body_Pct'] = (df['Body'] / df['Close']) * 100
        
        # Shadows
        df['Upper_Shadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)
        df['Lower_Shadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']
        
        # Direction
        df['Bullish'] = (df['Close'] > df['Open']).astype(int)
        df['Bearish'] = (df['Close'] < df['Open']).astype(int)
        
        # Patterns
        df['Doji'] = (df['Body_Pct'] < 0.1).astype(int)
        df['Hammer'] = ((df['Lower_Shadow'] > 2 * df['Body']) & (df['Upper_Shadow'] < df['Body'])).astype(int)
        df['Shooting_Star'] = ((df['Upper_Shadow'] > 2 * df['Body']) & (df['Lower_Shadow'] < df['Body'])).astype(int)
        
        self.features_created.extend(['Body', 'Body_Pct', 'Upper_Shadow', 'Lower_Shadow',
                                     'Bullish', 'Bearish', 'Doji', 'Hammer', 'Shooting_Star'])
        
        return df
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Statistical features"""
        # Returns
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Rolling stats
        for period in [5, 10, 20]:
            df[f'Returns_Mean_{period}'] = df['Returns'].rolling(period).mean()
            df[f'Returns_Std_{period}'] = df['Returns'].rolling(period).std()
        
        # Z-Score
        df['Z_Score_20'] = (df['Close'] - df['SMA_20']) / df['Close'].rolling(20).std()
        
        # Momentum
        for period in [3, 5, 10]:
            df[f'Momentum_{period}'] = df['Close'] - df['Close'].shift(period)
        
        self.features_created.extend(['Returns', 'Log_Returns', 'Z_Score_20'])
        
        return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience function"""
    engineer = FeatureEngineer()
    return engineer.add_all_features(df)


if __name__ == "__main__":
    
    print("Testing Native Feature Engineering...\n")
    
    df = get_data('BANKNIFTY', '2024-01-01', '2025-11-01')
    
    if df is not None:
        print(f"\nRaw data shape: {df.shape}")
        df_features = add_features(df)
        print(f"\nFeature-engineered shape: {df_features.shape}")
        print(f"\nSample output:")
        print(df_features.tail())
