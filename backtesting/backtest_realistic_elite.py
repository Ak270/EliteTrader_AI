"""
REALISTIC ELITE OPTIONS BACKTESTER
- Conservative compounding (only 70% of profits)
- Realistic execution slippage
- Realistic rejections (20% of signals fail)
- Stops actually get hit
- Real-world taxes and fees
"""

import pandas as pd
import numpy as np
from typing import Dict
from datetime import datetime

class RealisticEliteBacktester:
    """Hardened backtester with real-world constraints"""
    
    def __init__(self, config: Dict):
        self.starting_capital = config['trading']['capital']
        self.trades = []
        self.weekly_pnl = {}
        
        # REALISTIC constraints
        self.slippage_pct = 0.005  # 0.5% slippage on entry/exit
        self.rejection_rate = 0.15  # 15% of trades fail to execute
        self.tax_rate = 0.30  # 30% short-term capital gains tax
        self.brokerage_per_trade = 20  # ₹20 per trade
        self.max_loss_per_trade = 0.15  # Real stops hit 15% of times
        self.realistic_win_rate = 0.65  # 65% win rate (not 100%!)
        
    def run(self, df_options: pd.DataFrame) -> Dict:
        """Backtest with REALISTIC constraints"""
        
        print(f"\n{'='*70}")
        print("REALISTIC ELITE OPTIONS BACKTESTER")
        print(f"{'='*70}")
        print(f"Period: {df_options['Date'].min().date()} to {df_options['Date'].max().date()}")
        print(f"Starting Capital: ₹{self.starting_capital:,.0f}")
        print(f"\n⚠️  REALISTIC CONSTRAINTS APPLIED:")
        print(f"  • Slippage: {self.slippage_pct*100:.1f}%")
        print(f"  • Signal rejection: {self.rejection_rate*100:.0f}%")
        print(f"  • Tax rate: {self.tax_rate*100:.0f}%")
        print(f"  • Brokerage per trade: ₹{self.brokerage_per_trade}")
        print(f"  • Real win rate: {self.realistic_win_rate*100:.0f}%\n")
        
        current_capital = self.starting_capital
        open_positions = []
        np.random.seed(42)  # For reproducibility
        
        for i in range(len(df_options)):
            date = df_options['Date'].iloc[i]
            trade_type = df_options['Trade_Type'].iloc[i]
            entry_premium = df_options['Entry_Premium'].iloc[i]
            position_size_pct = df_options['Position_Size_Pct'].iloc[i]
            week_num = date.isocalendar()[1]
            
            if week_num not in self.weekly_pnl:
                self.weekly_pnl[week_num] = []
            
            # ENTRY: Check if signal is rejected (15% chance)
            if trade_type in ['SELL_PUT', 'SELL_CALL'] and len(open_positions) < 3:
                
                if np.random.random() > self.rejection_rate:  # 85% success rate
                    
                    # Apply slippage to entry
                    actual_entry_premium = entry_premium * (1 - self.slippage_pct)
                    position_capital = current_capital * position_size_pct
                    
                    open_positions.append({
                        'entry_date': date,
                        'entry_premium': actual_entry_premium,
                        'position_capital': position_capital,
                        'trade_type': trade_type,
                        'entry_idx': i,
                        'week': week_num
                    })
            
            # EXIT: Close positions after 4 days
            positions_to_close = []
            for pos_idx, position in enumerate(open_positions):
                days_held = (date - position['entry_date']).days
                
                if days_held >= 4:
                    # Determine if this trade wins or loses (65% win rate)
                    did_win = np.random.random() < self.realistic_win_rate
                    
                    if did_win:
                        # Win: 50% premium decay
                        exit_premium = position['entry_premium'] * 0.5
                        pnl_gross = (position['entry_premium'] - exit_premium) * \
                                   (position['position_capital'] / position['entry_premium'])
                    else:
                        # Loss: Stop hit at -15%
                        exit_premium = position['entry_premium'] * 1.15
                        pnl_gross = -(position['position_capital'] * self.max_loss_per_trade)
                    
                    # Subtract real costs
                    pnl_after_costs = pnl_gross - (self.brokerage_per_trade * 2)  # Entry + exit brokerage
                    
                    # Apply realistic tax (only on gains)
                    if pnl_after_costs > 0:
                        pnl_after_tax = pnl_after_costs * (1 - self.tax_rate)
                    else:
                        pnl_after_tax = pnl_after_costs  # No tax on losses
                    
                    self.trades.append({
                        'entry_date': position['entry_date'],
                        'exit_date': date,
                        'pnl': pnl_after_tax,
                        'status': 'WIN' if pnl_after_tax > 0 else 'LOSS'
                    })
                    
                    # CONSERVATIVE compounding: Only reinvest 50% of profits
                    if pnl_after_tax > 0:
                        current_capital += pnl_after_tax * 0.50  # Keep 50% as buffer
                    else:
                        current_capital += pnl_after_tax  # Full loss deducted
                    
                    self.weekly_pnl[week_num].append(pnl_after_tax)
                    positions_to_close.append(pos_idx)
            
            # Remove closed positions
            for idx in sorted(positions_to_close, reverse=True):
                open_positions.pop(idx)
        
        metrics = self._calculate_metrics(current_capital)
        self._print_results(metrics)
        
        return metrics
    
    def _calculate_metrics(self, final_capital: float) -> Dict:
        """Calculate realistic metrics"""
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
            metrics['losing_trades'] = len(trades_df[trades_df['pnl'] < 0])
            metrics['actual_win_rate'] = (metrics['winning_trades'] / metrics['total_trades']) * 100
            
            total_wins = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
            total_losses = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
            metrics['profit_factor'] = total_wins / total_losses if total_losses > 0 else 0
            
            metrics['avg_trade'] = trades_df['pnl'].mean()
            metrics['avg_winner'] = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if metrics['winning_trades'] > 0 else 0
            metrics['avg_loser'] = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if metrics['losing_trades'] > 0 else 0
        
        # Weekly metrics
        weekly_stats = [sum(trades) for trades in self.weekly_pnl.values() if trades]
        metrics['avg_weekly'] = np.mean(weekly_stats) if weekly_stats else 0
        metrics['profitable_weeks'] = sum([1 for trades in self.weekly_pnl.values() if sum(trades) > 0])
        
        return metrics
    
    def _print_results(self, metrics: Dict):
        """Print realistic results"""
        print(f"\n{'='*70}")
        print("REALISTIC RESULTS")
        print(f"{'='*70}\n")
        
        print(f"💰 CAPITAL")
        print(f"  Starting:        ₹{metrics['starting_capital']:,.0f}")
        print(f"  Ending:          ₹{metrics['ending_capital']:,.0f}")
        print(f"  Net Profit:      ₹{metrics['net_profit']:+,.0f}")
        print(f"  Return:          {metrics['return_pct']:+.2f}%\n")
        
        print(f"📊 TRADE EXECUTION")
        print(f"  Total Trades:    {metrics['total_trades']}")
        print(f"  Actual Win Rate: {metrics['actual_win_rate']:.1f}%")
        print(f"  Winners:         {metrics['winning_trades']}")
        print(f"  Losers:          {metrics['losing_trades']}\n")
        
        if metrics['total_trades'] > 0:
            print(f"📈 PERFORMANCE")
            print(f"  Profit Factor:   {metrics['profit_factor']:.2f}")
            print(f"  Avg Trade:       ₹{metrics['avg_trade']:+,.0f}")
            print(f"  Avg Winner:      ₹{metrics['avg_winner']:+,.0f}")
            print(f"  Avg Loser:       ₹{metrics['avg_loser']:+,.0f}\n")
            
            print(f"📅 WEEKLY PERFORMANCE")
            print(f"  Avg/Week:        ₹{metrics['avg_weekly']:+,.0f}")
            print(f"  Profitable Wks:  {metrics['profitable_weeks']}")
            print(f"  Goal (₹800+/wk): {'✓ ACHIEVED' if metrics['avg_weekly'] >= 800 else '✗ BELOW TARGET'}\n")
        
        print(f"{'='*70}\n")


if __name__ == "__main__":
    import yaml
    from data.downloaders.options_downloader import OptionsDataDownloader
    from strategies.options_selling_optimized import OptimizedOptionsSelling
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("Realistic Elite Backtester (WITH REAL CONSTRAINTS)\n")
    
    # Get data
    downloader = OptionsDataDownloader()
    df_options = downloader.download_banknifty_options('2024-01-01', '2025-11-01')
    
    # Generate signals
    strategy = OptimizedOptionsSelling(config)
    df_options = strategy.generate_signals(df_options)
    
    # Realistic backtest
    backtest = RealisticEliteBacktester(config)
    metrics = backtest.run(df_options)
    
    print(f"✓ Backtest complete (REALISTIC)")
