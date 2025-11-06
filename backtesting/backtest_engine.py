"""
Simplified Working Backtest Engine
Fixed to properly execute trades from signals
"""

import pandas as pd
import numpy as np
from typing import Dict
from datetime import datetime

class BacktestEngine:
    """Working backtester that executes signals properly"""

    def __init__(self, config: Dict):
        self.config = config
        self.capital = config['trading']['capital']
        self.commission = config['backtest']['commission']
        self.trades = []
        self.equity_curve = []
        
    def run(self, df: pd.DataFrame, signals: pd.DataFrame, 
            position_sizes: pd.Series, strategy_name: str) -> Dict:
        """
        Run backtest with proper trade execution
        """
        print(f"\n{'='*70}")
        print(f"BACKTESTING: {strategy_name}")
        print(f"{'='*70}")
        print(f"Period: {df.index[0].date()} to {df.index[-1].date()}")
        print(f"Initial Capital: ₹{self.capital:,.0f}\n")
        
        # Initialize tracking
        current_capital = self.capital
        position = 0  # 0=flat, 1=long, -1=short
        entry_price = 0
        entry_date = None
        entry_capital = 0
        
        # Process each day
        for i in range(len(df)):
            date = df.index[i]
            price = df['Close'].iloc[i]
            signal = signals['Signal'].iloc[i]
            
            # Entry logic
            if position == 0 and signal != 0:
                # Open new position
                position = signal
                entry_price = price
                entry_date = date
                entry_capital = current_capital * 0.95  # Use 95% of capital
                
                self.equity_curve.append({
                    'date': date,
                    'equity': current_capital,
                    'position': position
                })
                
            # Exit logic
            elif position != 0 and (signal == -position or signal == 0):
                # Close position
                exit_price = price
                
                # Calculate P&L
                if position == 1:  # Long
                    pnl_pct = (exit_price / entry_price - 1)
                else:  # Short
                    pnl_pct = (entry_price / exit_price - 1)
                
                pnl_gross = entry_capital * pnl_pct
                pnl_after_commission = pnl_gross - (entry_capital * self.commission * 2)
                pnl_after_tax = pnl_after_commission * 0.7 if pnl_after_commission > 0 else pnl_after_commission
                
                # Record trade
                trade = {
                    'entry_date': entry_date,
                    'exit_date': date,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'direction': 'LONG' if position == 1 else 'SHORT',
                    'pnl_gross': pnl_gross,
                    'pnl_after_tax': pnl_after_tax,
                    'return_pct': pnl_pct * 100,
                    'hold_days': (date - entry_date).days
                }
                self.trades.append(trade)
                
                # Update capital
                current_capital += pnl_after_tax
                
                # Reset position
                position = 0
                entry_price = 0
                entry_date = None
                
                self.equity_curve.append({
                    'date': date,
                    'equity': current_capital,
                    'position': position
                })
        
        # Calculate metrics
        metrics = self._calculate_metrics(current_capital)
        self._print_results(metrics)
        
        return metrics
    
    def _calculate_metrics(self, final_capital: float) -> Dict:
        """Calculate performance metrics"""
        metrics = {
            'starting_capital': self.capital,
            'ending_capital': final_capital,
            'net_profit': final_capital - self.capital,
            'return_pct': ((final_capital - self.capital) / self.capital) * 100,
            'total_trades': len(self.trades),
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'max_drawdown_pct': 0,
            'sharpe_ratio': 0
        }
        
        if len(self.trades) > 0:
            trades_df = pd.DataFrame(self.trades)
            
            # Win/Loss stats
            metrics['winning_trades'] = len(trades_df[trades_df['pnl_after_tax'] > 0])
            metrics['losing_trades'] = len(trades_df[trades_df['pnl_after_tax'] < 0])
            metrics['win_rate'] = (metrics['winning_trades'] / len(self.trades)) * 100
            
            # Profit factor
            total_wins = trades_df[trades_df['pnl_after_tax'] > 0]['pnl_after_tax'].sum()
            total_losses = abs(trades_df[trades_df['pnl_after_tax'] < 0]['pnl_after_tax'].sum())
            metrics['profit_factor'] = total_wins / total_losses if total_losses > 0 else 0
            
            # Averages
            if metrics['winning_trades'] > 0:
                metrics['avg_win'] = trades_df[trades_df['pnl_after_tax'] > 0]['pnl_after_tax'].mean()
            if metrics['losing_trades'] > 0:
                metrics['avg_loss'] = trades_df[trades_df['pnl_after_tax'] < 0]['pnl_after_tax'].mean()
            
            # Drawdown
            if len(self.equity_curve) > 0:
                equity_series = pd.Series([e['equity'] for e in self.equity_curve])
                running_max = equity_series.cummax()
                drawdown = (equity_series - running_max) / running_max
                metrics['max_drawdown_pct'] = abs(drawdown.min()) * 100
            
            # Sharpe ratio
            returns = trades_df['return_pct'].values / 100
            if len(returns) > 1 and returns.std() > 0:
                metrics['sharpe_ratio'] = (returns.mean() / returns.std()) * np.sqrt(252)
        
        return metrics
    
    def _print_results(self, metrics: Dict):
        """Print results"""
        print(f"\n{'='*70}")
        print("RESULTS")
        print(f"{'='*70}\n")
        
        print(f"💰 CAPITAL")
        print(f"  Starting:        ₹{metrics['starting_capital']:,.0f}")
        print(f"  Ending:          ₹{metrics['ending_capital']:,.0f}")
        print(f"  Net Profit:      ₹{metrics['net_profit']:+,.0f}")
        print(f"  Return:          {metrics['return_pct']:+.2f}%\n")
        
        print(f"📊 TRADES")
        print(f"  Total:           {metrics['total_trades']}")
        print(f"  Winners:         {metrics['winning_trades']}")
        print(f"  Losers:          {metrics['losing_trades']}")
        print(f"  Win Rate:        {metrics['win_rate']:.1f}%\n")
        
        if metrics['total_trades'] > 0:
            print(f"📈 RISK/REWARD")
            print(f"  Profit Factor:   {metrics['profit_factor']:.2f}")
            print(f"  Avg Win:         ₹{metrics['avg_win']:+,.0f}")
            print(f"  Avg Loss:        ₹{metrics['avg_loss']:+,.0f}")
            print(f"  Sharpe Ratio:    {metrics['sharpe_ratio']:.2f}\n")
            
            print(f"⚠️  RISK")
            print(f"  Max Drawdown:    {metrics['max_drawdown_pct']:.2f}%")
        
        print(f"\n{'='*70}\n")
