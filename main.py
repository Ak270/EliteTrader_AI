"""
EliteTrader AI - Master Control Script
Coordinates data, strategies, backtesting, optimization, and execution
"""

import yaml
import pandas as pd
from datetime import datetime
import sys
import os
from typing import Dict


# Import components
from data.downloaders.universal_downloader import UniversalDataDownloader
from data.processors.feature_engineer import FeatureEngineer
from strategies.options_selling import OptionsPremiumStrategy
from strategies.breakout_momentum import BreakoutMomentumStrategy
from strategies.mean_reversion import MeanReversionStrategy
from backtesting.backtest_engine import BacktestEngine
from backtesting.optimizer import StrategyOptimizer

class EliteTrader:
    """Main orchestrator for entire trading system"""
    
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.downloader = UniversalDataDownloader(config_path)
        self.engineer = FeatureEngineer()
        self.strategies = {
            'options_premium': OptionsPremiumStrategy(self.config),
            'breakout_momentum': BreakoutMomentumStrategy(self.config),
            'mean_reversion': MeanReversionStrategy(self.config),
        }
        self.backtest_engine = BacktestEngine(self.config)
    
    def run_full_pipeline(self, symbol: str, start_date: str, end_date: str):
        """Complete workflow: Download → Features → Backtest → Optimize"""
        
        print("\n" + "="*70)
        print(f"ELITETRADER AI - FULL PIPELINE")
        print(f"Symbol: {symbol} | Period: {start_date} to {end_date}")
        print("="*70)
        
        # 1. Download data
        print("\n[1/5] Downloading data...")
        df = self.downloader.download(symbol, start_date, end_date)
        if df is None:
            print("❌ Data download failed")
            return None
        
        # 2. Add features
        print("\n[2/5] Engineering features...")
        df = self.engineer.add_all_features(df)
        print(f"✓ {len(self.engineer.features_created)} features created")
        
        # 3. Backtest all strategies
        print("\n[3/5] Backtesting strategies...")
        results = {}
        for strat_name, strategy in self.strategies.items():
            print(f"\n  Testing {strat_name}...")
            df_signals = strategy.generate_signals(df.copy())
            pos_sizes = pd.Series([self.config['trading']['capital'] * 0.25] * len(df))
            metrics = self.backtest_engine.run(df_signals, df_signals[['Signal']], pos_sizes, strat_name)
            results[strat_name] = metrics
        
        # 4. Optimize top strategy
        print("\n[4/5] Optimizing best strategy...")
        best_strategy = max(results.items(), key=lambda x: x[1]['sharpe_ratio'])[0]
        print(f"Best strategy: {best_strategy} (Sharpe: {results[best_strategy]['sharpe_ratio']:.2f})")
        
        # 5. Save results
        print("\n[5/5] Saving results...")
        self._save_results(symbol, results, df)
        
        print("\n" + "="*70)
        print("✓ PIPELINE COMPLETE")
        print("="*70 + "\n")
        
        return results
    
    def _save_results(self, symbol: str, results: Dict, df: pd.DataFrame):
        """Save backtesting results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save metrics
        metrics_df = pd.DataFrame(results).T
        metrics_file = f"backtesting/reports/{symbol}_metrics_{timestamp}.csv"
        metrics_df.to_csv(metrics_file)
        print(f"✓ Metrics saved: {metrics_file}")
        
        # Save equity curve
        equity_file = f"backtesting/reports/{symbol}_equity_{timestamp}.csv"
        equity_df = pd.DataFrame(self.backtest_engine.equity_curve)
        equity_df.to_csv(equity_file)
        print(f"✓ Equity curve saved: {equity_file}")
        
        # Save trades
        trades_file = f"backtesting/reports/{symbol}_trades_{timestamp}.csv"
        trades_df = pd.DataFrame(self.backtest_engine.trades)
        trades_df.to_csv(trades_file)
        print(f"✓ Trades saved: {trades_file}")


if __name__ == "__main__":
    # Initialize EliteTrader
    trader = EliteTrader('config.yaml')
    
    # Run complete pipeline
    results = trader.run_full_pipeline(
        symbol='BANKNIFTY',
        start_date='2024-01-01',
        end_date='2025-11-01'
    )
    
    if results:
        print("\n✓ System ready for paper trading!")

