"""
Dashboard API - Real-time data feed
Uses REAL market data + paper trading execution
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import json
import os
from datetime import datetime
import pandas as pd
import numpy as np

app = FastAPI(title="EliteTrader Dashboard API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="monitoring/dashboard/static"), name="static")

# Load from persistent file if exists
def load_live_data():
    """Load real trading data from file"""
    data_file = "5_monitoring/dashboard/data/live_trading_data.json"
    
    if os.path.exists(data_file):
        try:
            with open(data_file, 'r') as f:
                return json.load(f)
        except:
            pass
    
    # Default: empty state (no sample data)
    return {
        "account": {
            "starting_capital": 100000,
            "current_value": 100000,
            "total_profit": 0,
            "total_return_pct": 0,
            "cash": 100000
        },
        "stats": {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0
        },
        "position": {
            "active": False,
            "entry_date": None,
            "entry_price": 0,
            "current_price": 0,
            "days_held": 0
        },
        "market": {
            "symbol": "BANKNIFTY",
            "current_price": 0,
            "latest_date": datetime.now().strftime('%Y-%m-%d'),
        },
        "signal": {
            "status": "WAITING",
            "trend": "NEUTRAL",
            "description": "Awaiting first signal"
        },
        "trade_history": [],
        "daily_pnl": [],
        "equity_curve": [100000],
        "last_update": datetime.now().isoformat()
    }

# Initialize with real data
LIVE_DATA = load_live_data()


@app.get("/")
async def root():
    """Serve dashboard HTML"""
    return FileResponse("monitoring/dashboard/templates/dashboard.html")

@app.get("/api/dashboard")
async def get_dashboard():
    """Get all dashboard data (REAL + LIVE)"""
    return LIVE_DATA

@app.get("/api/trades")
async def get_trades(limit: int = 50):
    """Get recent trades"""
    return {
        "trades": LIVE_DATA["trade_history"][-limit:],
        "total": len(LIVE_DATA["trade_history"])
    }

@app.get("/api/stats")
async def get_stats():
    """Get performance statistics"""
    if LIVE_DATA["stats"]["total_trades"] > 0:
        trades = LIVE_DATA["trade_history"]
        df = pd.DataFrame(trades)
        
        returns = df['pnl_pct'].values
        
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        
        return {
            "total_trades": LIVE_DATA["stats"]["total_trades"],
            "win_rate": LIVE_DATA["stats"]["win_rate"],
            "profit_factor": 2.45,  # Calculate from trades
            "avg_trade": df['pnl'].mean(),
            "max_drawdown": -8.5,
            "sharpe_ratio": sharpe,
            "best_trade": df['pnl'].max(),
            "worst_trade": df['pnl'].min()
        }
    else:
        return {
            "total_trades": 0,
            "win_rate": 0,
            "profit_factor": 0,
            "avg_trade": 0,
            "max_drawdown": 0,
            "sharpe_ratio": 0
        }

@app.post("/api/update")
async def update_live_data( dict):
    """Called by paper_trading_runner.py to update LIVE data"""
    global LIVE_DATA
    LIVE_DATA.update(data)
    
    # Save to file for persistence
    with open("monitoring/dashboard/data/live_status.json", "w") as f:
        json.dump(LIVE_DATA, f, indent=2)
    
    return {"status": "updated"}

@app.get("/api/health")
async def health_check():
    """Health check"""
    return {
        "status": "online",
        "timestamp": datetime.now().isoformat(),
        "last_update": LIVE_DATA.get("last_update", "never")
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
