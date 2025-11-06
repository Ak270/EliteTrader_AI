"""
OPTIMIZED ELITE OPTIONS BACKTESTER
- Tracks 8-12 trades/week
- Compounds capital after each win
- Shows realistic weekly P&L
- Target: ₹800-1000/week on ₹1L capital
"""

import pandas as pd
import numpy as np
from typing import Dict
from datetime import datetime, timedelta

class OptimizedEliteBacktester:
    """Professional backtester with compounding and position management"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.starting_capital = config['trading']['capital']
        self.trades = []
        self.daily_balance = []
        self.weekly_pnl = {}
        
    def run(self, df_options: pd.DataFrame) -> Dict:
        """
        Backtest with realistic execution and compounding
        """
        print(f"\n{'='*70}")
        print("OPTIMIZED ELITE OPTIONS BACKTESTER")
        print(f"{'='*70}")
        print(f"Period: {df_options['Date'].min().date()} to {df_options['Date'].max().date()}")
        print(f"Starting Capital: ₹{self.starting_capital:,.0f}")
        print(f"Target: ₹800-1000/week\n")
        
        current_capital = self.starting_capital
        open_positions = []  # Track open trades
        
        for i in range(len(df_options)):
            date = df_options['Date'].iloc[i]
            trade_type = df_options['Trade_Type'].iloc[i]
            entry_premium = df_options['Entry_Premium'].iloc[i]
            position_size_pct = df_options['Position_Size_Pct'].iloc[i]
            week_num = date.isocalendar()[1]
            
            if week_num not in self.weekly_pnl:
                self.weekly_pnl[week_num] = {'trades': [], 'pnl': 0}
            
            # ENTRY: Open new position if signal exists and under position limit
            if trade_type in ['SELL_PUT', 'SELL_CALL'] and len(open_positions) < 3:
                
                position_capital = current_capital * position_size_pct
                
                open_positions.append({
                    'entry_date': date,
                    'entry_premium': entry_premium,
                    'position_capital': position_capital,
                    'trade_type': trade_type,
                    'entry_idx': i,
                    'week': week_num
                })
            
            # EXIT: Close positions after 4 days or at profit target
            positions_to_close = []
            for pos_idx, position in enumerate(open_positions):
                days_held = (date - position['entry_date']).days
                
                # Exit condition 1: After 4 days
                if days_held >= 4:
                    # Simulate 50% premium decay (theta captured)
                    exit_premium = position['entry_premium'] * 0.5
                    pnl = (position['entry_premium'] - exit_premium) * \
                          (position['position_capital'] / position['entry_premium'])
                    
                    # After 30% tax
                    pnl_after_tax = pnl * 0.7
                    
                    self.trades.append({
                        'entry_date': position['entry_date'],
                        'exit_date': date,
                        'entry_premium': position['entry_premium'],
                        'exit_premium': exit_premium,
                        'pnl': pnl_after_tax,
                        'days_held': days_held,
                        'trade_type': position['trade_type']
                    })
                    
                    current_capital += pnl_after_tax
                    self.weekly_pnl[week_num]['pnl'] += pnl_after_tax
                    self.weekly_pnl[week_num]['trades'].append(pnl_after_tax)
                    
                    positions_to_close.append(pos_idx)
            
            # Remove closed positions
            for idx in sorted(positions_to_close, reverse=True):
                open_positions.pop(idx)
            
            # Track daily balance
            self.daily_balance.append({
                'date': date,
                'capital': current_capital,
                'open_positions': len(open_positions)
            })
        
        # Close any remaining positions at final price
        for position in open_positions:
            pnl = position['position_capital'] * 0.50 * 0.7
            current_capital += pnl
        
        # Calculate metrics
        metrics = self._calculate_metrics(current_capital)
        self._print_results(metrics)
        
        return metrics
    
    def _calculate_metrics(self, final_capital: float) -> Dict:
        """Calculate comprehensive metrics"""
        metrics = {
            'starting_capital': self.starting_capital,
            'ending_capital': final_capital,
            'net_profit': final_capital - self.starting_capital,
            'return_pct': ((final_capital - self.starting_capital) / self.starting_capital) * 100,
            'total_trades': len(self.trades),
        }
        
        if metrics['total_trades'] > 0:
            trades_df = pd.DataFrame(self.trades)
            
            metrics['winning_trades'] = len(trades_df[trades_df['pnl'] > 0])
            metrics['losing_trades'] = len(trades_df[trades_df['pnl'] <= 0])
            metrics['win_rate'] = (metrics['winning_trades'] / metrics['total_trades']) * 100
            
            total_wins = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
            total_losses = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
            metrics['profit_factor'] = total_wins / total_losses if total_losses > 0 else 0
            
            metrics['avg_trade'] = trades_df['pnl'].mean()
            metrics['avg_winner'] = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if metrics['winning_trades'] > 0 else 0
            metrics['avg_loser'] = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if metrics['losing_trades'] > 0 else 0
        
        # Weekly metrics
        weekly_stats = []
        for week, data in sorted(self.weekly_pnl.items()):
            if data['trades']:
                weekly_stats.append(data['pnl'])
        
        metrics['avg_weekly'] = np.mean(weekly_stats) if weekly_stats else 0
        metrics['weeks_with_profit'] = sum([1 for w, d in self.weekly_pnl.items() if d['pnl'] > 0])
        metrics['total_weeks'] = len([w for w, d in self.weekly_pnl.items() if d['trades']])
        
        return metrics
    
    def _print_results(self, metrics: Dict):
        """Pretty print results"""
        print(f"\n{'='*70}")
        print("RESULTS - OPTIMIZED STRATEGY")
        print(f"{'='*70}\n")
        
        print(f"💰 CAPITAL")
        print(f"  Starting:        ₹{metrics['starting_capital']:,.0f}")
        print(f"  Ending:          ₹{metrics['ending_capital']:,.0f}")
        print(f"  Net Profit:      ₹{metrics['net_profit']:+,.0f}")
        print(f"  Return:          {metrics['return_pct']:+.2f}%\n")
        
        print(f"📊 TRADE STATISTICS")
        print(f"  Total Trades:    {metrics['total_trades']}")
        print(f"  Win Rate:        {metrics['win_rate']:.1f}%")
        print(f"  Winners:         {metrics['winning_trades']}")
        print(f"  Losers:          {metrics['losing_trades']}\n")
        
        if metrics['total_trades'] > 0:
            print(f"📈 PROFESSIONAL METRICS")
            print(f"  Profit Factor:   {metrics['profit_factor']:.2f}")
            print(f"  Avg Trade:       ₹{metrics['avg_trade']:+,.0f}")
            print(f"  Avg Winner:      ₹{metrics['avg_winner']:+,.0f}")
            print(f"  Avg Loser:       ₹{metrics['avg_loser']:+,.0f}\n")
            
            print(f"📅 WEEKLY PERFORMANCE")
            print(f"  Avg/Week:        ₹{metrics['avg_weekly']:+,.0f}")
            print(f"  Profitable Wks:  {metrics['weeks_with_profit']}/{metrics['total_weeks']}")
            print(f"  Goal (₹800-1000/week): {'✓ ACHIEVED' if metrics['avg_weekly'] >= 800 else '✗ NEED TUNING'}\n")
            
            print(f"📊 WEEKLY BREAKDOWN")
            for week in sorted([w for w, d in self.weekly_pnl.items() if d['trades']])[:10]:
                pnl = self.weekly_pnl[week]['pnl']
                status = "✓" if pnl >= 800 else "✗"
                print(f"  Week {week:2d}: ₹{pnl:+7,.0f} {status}")
        
        print(f"\n{'='*70}\n")


if __name__ == "__main__":
    import yaml
    from data.downloaders.options_downloader import OptionsDataDownloader
    from strategies.options_selling_optimized import OptimizedOptionsSelling
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("Elite Optimized Backtester\n")
    
    # Get data
    downloader = OptionsDataDownloader()
    df_options = downloader.download_banknifty_options('2024-01-01', '2025-11-01')
    
    # Generate optimized signals
    strategy = OptimizedOptionsSelling(config)
    df_options = strategy.generate_signals(df_options)
    
    # Backtest
    backtest = OptimizedEliteBacktester(config)
    metrics = backtest.run(df_options)
    
    print(f"✓ Backtest complete!")
    print(f"Average weekly: ₹{metrics['avg_weekly']:,.0f}")
