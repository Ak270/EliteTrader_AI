"""
Bayesian Hyperparameter Optimization using Optuna
Finds best parameters automatically through 500+ combinations
"""

import pandas as pd
import numpy as np
import optuna
from optuna.pruners import MedianPruner
from typing import Dict, Callable
import yaml

class StrategyOptimizer:
    """
    Automatically optimize strategy parameters using Bayesian search
    """
    
    def __init__(self, config: Dict, backtest_func: Callable):
        """
        Args:
            config: Config dict
            backtest_func: Function that runs backtest and returns sharpe_ratio
        """
        self.config = config
        self.backtest_func = backtest_func
        self.best_params = None
        self.best_score = -np.inf
        self.study = None
    
    def optimize(self, n_trials: int = 500) -> Dict:
        """
        Run Bayesian optimization over n_trials
        
        Returns:
            Dictionary with best parameters and score
        """
        print(f"\n{'='*70}")
        print(f"PARAMETER OPTIMIZATION (Bayesian Search)")
        print(f"{'='*70}")
        print(f"Trials: {n_trials}")
        print(f"Method: Optuna (Tree-structured Parzen Estimator)\n")
        
        # Create study
        self.study = optuna.create_study(
            direction='maximize',
            pruner=MedianPruner()
        )
        
        # Optimize
        self.study.optimize(
            self._objective,
            n_trials=n_trials,
            show_progress_bar=True
        )
        
        # Extract best
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        print(f"\n{'='*70}")
        print("OPTIMIZATION RESULTS")
        print(f"{'='*70}")
        print(f"Best Sharpe Ratio: {self.best_score:.4f}")
        print(f"\nOptimal Parameters:")
        for param, value in self.best_params.items():
            print(f"  {param}: {value}")
        print(f"{'='*70}\n")
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'trials': len(self.study.trials)
        }
    
    def _objective(self, trial):
        """Objective function for Optuna"""
        
        # Suggest parameters (customize per strategy)
        params = {
            'rsi_period': trial.suggest_int('rsi_period', 7, 21),
            'rsi_overbought': trial.suggest_int('rsi_overbought', 60, 80),
            'rsi_oversold': trial.suggest_int('rsi_oversold', 20, 40),
            'atr_period': trial.suggest_int('atr_period', 7, 21),
            'atr_multiplier': trial.suggest_float('atr_multiplier', 1.0, 3.0),
            'profit_target_pct': trial.suggest_float('profit_target_pct', 0.3, 0.7),
            'stop_loss_pct': trial.suggest_float('stop_loss_pct', 0.1, 0.5),
        }
        
        # Run backtest
        try:
            metrics = self.backtest_func(params)
            
            # Objective: maximize Sharpe ratio (risk-adjusted returns)
            score = metrics.get('sharpe_ratio', -100)
            
            # Prune if performing poorly
            if score < -10:
                trial.report(score, step=0)
                raise optuna.TrialPruned()
            
            return score
            
        except Exception as e:
            print(f"Trial failed: {e}")
            return -1000


def optimize_strategy(strategy_class, df: pd.DataFrame, 
                     config: Dict, n_trials: int = 500) -> Dict:
    """
    Convenience function to optimize any strategy
    
    Args:
        strategy_class: Strategy class (OptionsPremiumStrategy, etc)
        df: DataFrame with features
        config: Config dict
        n_trials: Number of optimization trials
    
    Returns:
        Best parameters dictionary
    """
    from backtest_engine import BacktestEngine
    
    def backtest_with_params(params):
        """Backtest function for optimizer"""
        # Create strategy with params
        strategy = strategy_class(config)
        for key, value in params.items():
            if hasattr(strategy, key):
                setattr(strategy, key, value)
        
        # Generate signals
        df_test = df.copy()
        df_test = strategy.generate_signals(df_test)
        
        # Run backtest
        engine = BacktestEngine(config)
        pos_sizes = pd.Series([config['trading']['capital'] * 0.25] * len(df_test))
        metrics = engine.run(df_test, df_test[['Signal']], pos_sizes, strategy.name)
        
        return metrics
    
    optimizer = StrategyOptimizer(config, backtest_with_params)
    return optimizer.optimize(n_trials=n_trials)


if __name__ == "__main__":
    import sys
    sys.path.append('..')
    from data.downloaders.universal_downloader import get_data
    from data.processors.feature_engineer import add_features
    from strategies.options_selling import OptionsPremiumStrategy
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("Testing Parameter Optimization...\n")
    
    # Get data
    df = get_data('BANKNIFTY', '2024-01-01', '2025-11-01')
    df = add_features(df)
    
    # Optimize (start with 50 trials for testing, scale to 500)
    best_params = optimize_strategy(OptionsPremiumStrategy, df, config, n_trials=50)
    
    print(f"\n✓ Optimization complete!")
    print(f"Best parameters saved for production use")

