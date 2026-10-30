"""Download daily adjusted close prices with yfinance and cache them in data/prices.csv.

The CSV is already in the repo, so you only need this to refresh the data.
IRX is the 13-week US T-bill yield in percent (used as the risk-free rate).
"""
from pathlib import Path

import yfinance as yf

TICKERS = ["SPY", "AAPL", "MSFT", "JPM", "XOM", "JNJ", "KO", "NVDA", "^IRX"]
START, END = "2010-01-01", "2025-10-01"
OUT = Path(__file__).resolve().parents[1] / "data" / "prices.csv"

if __name__ == "__main__":
    px = yf.download(TICKERS, start=START, end=END, auto_adjust=True, progress=False)["Close"]
    px = px.rename(columns={"^IRX": "IRX"}).dropna(subset=["SPY"]).ffill()
    px.round(4).to_csv(OUT)
    print(f"saved {px.shape[0]} rows x {px.shape[1]} columns to {OUT}")
