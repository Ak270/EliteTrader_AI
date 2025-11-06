"""
Universal Data Downloader - Multi-source with failover
Supports: yfinance, yahooquery, NSE official
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from yahooquery import Ticker
import requests
import time
from typing import Optional, Dict, List
import yaml

class UniversalDataDownloader:
    """Professional-grade data downloader with multiple sources"""
    
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.sources = self.config['data']['sources']
        self.cache_days = self.config['data']['cache_days']
        
    def download(self, symbol: str, start_date: str, end_date: str, 
                 interval: str = '1d') -> Optional[pd.DataFrame]:
        """
        Download data with automatic failover across sources
        
        Args:
            symbol: Trading symbol (e.g., 'BANKNIFTY', 'NIFTY', 'USDINR')
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD'
            interval: Data interval ('1d', '1h', '15m', etc.)
        
        Returns:
            DataFrame with OHLCV data or None if all sources fail
        """
        print(f"\n{'='*70}")
        print(f"DOWNLOADING {symbol} DATA")
        print(f"{'='*70}")
        print(f"Period: {start_date} to {end_date}")
        print(f"Interval: {interval}\n")
        
        # Convert Indian symbols to Yahoo format
        yahoo_symbol = self._convert_symbol(symbol)
        
        # Try each source
        for source in self.sources:
            try:
                print(f"Trying {source}...")
                
                if source == 'yfinance':
                    df = self._download_yfinance(yahoo_symbol, start_date, end_date, interval)
                elif source == 'yahooquery':
                    df = self._download_yahooquery(yahoo_symbol, start_date, end_date)
                elif source == 'nse_official':
                    df = self._download_nse(symbol, start_date, end_date)
                else:
                    continue
                
                if df is not None and len(df) > 0:
                    print(f"✓ {source} success! Got {len(df)} rows")
                    return self._clean_data(df)
                else:
                    print(f"✗ {source} returned no data")
                    
            except Exception as e:
                print(f"✗ {source} failed: {str(e)[:100]}")
            
            time.sleep(0.5)  # Rate limiting
        
        print(f"\n❌ All sources failed for {symbol}")
        return None
    
    def _convert_symbol(self, symbol: str) -> str:
        """Convert Indian symbols to Yahoo Finance format"""
        symbol_map = {
            'BANKNIFTY': '^NSEBANK',
            'NIFTY': '^NSEI',
            'USDINR': 'USDINR=X',
            'EURINR': 'EURINR=X',
        }
        return symbol_map.get(symbol, symbol)
    
    def _download_yfinance(self, symbol: str, start: str, end: str, 
                           interval: str) -> Optional[pd.DataFrame]:
        """Download from yfinance"""
        df = yf.download(symbol, start=start, end=end, interval=interval,
                        progress=False, auto_adjust=True)
        
        # Handle multi-index columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        
        return df if len(df) > 0 else None
    
    def _download_yahooquery(self, symbol: str, start: str, 
                            end: str) -> Optional[pd.DataFrame]:
        """Download from yahooquery"""
        ticker = Ticker(symbol)
        df = ticker.history(start=start, end=end)
        
        if df is not None and len(df) > 0:
            # Reset multi-index
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index(level=0, drop=True)
            
            # Ensure datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            # Rename columns
            df = df.rename(columns={
                'open': 'Open', 'high': 'High', 'low': 'Low',
                'close': 'Close', 'volume': 'Volume'
            })
            
            return df
        
        return None
    
    def _download_nse(self, symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
        """Download from NSE (backup for Indian indices)"""
        # TODO: Implement NSE official API
        # This requires scraping or official API access
        return None
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize data"""
        # Ensure required columns
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Remove NaN rows
        df = df.dropna()
        
        # Sort by date
        df = df.sort_index()
        
        # Add Adj Close if missing
        if 'Adj Close' not in df.columns:
            df['Adj Close'] = df['Close']
        
        return df
    
    def download_multiple(self, symbols: List[str], start: str, 
                         end: str) -> Dict[str, pd.DataFrame]:
        """Download data for multiple symbols"""
        results = {}
        
        for symbol in symbols:
            print(f"\n{'='*70}")
            print(f"Processing: {symbol}")
            df = self.download(symbol, start, end)
            if df is not None:
                results[symbol] = df
            time.sleep(1)  # Rate limiting between symbols
        
        return results


# Convenience function
def get_data(symbol: str, start: str, end: str, 
             config_path='config.yaml') -> Optional[pd.DataFrame]:
    """Quick data download function"""
    downloader = UniversalDataDownloader(config_path)
    return downloader.download(symbol, start, end)


if __name__ == "__main__":
    # Test the downloader
    print("Testing Universal Data Downloader...\n")
    
    # Test Bank Nifty
    df = get_data('BANKNIFTY', '2024-01-01', '2025-11-01')
    
    if df is not None:
        print(f"\n✓ SUCCESS!")
        print(f"Shape: {df.shape}")
        print(f"\nFirst 5 rows:")
        print(df.head())
        print(f"\nLast 5 rows:")
        print(df.tail())
    else:
        print("\n✗ FAILED - No data retrieved")

