# ============================
# Imports
# ============================
import yfinance as yf
import pandas as pd
from datetime import date

"""
API Tools for Portfolio Visualization
-------------------------------------
Functions to retrieve historical and current stock data for tickers.

"""

# -------------------------------
# Get historical daily prices for a list of tickers
# -------------------------------
def get_historical_prices(tickers, start_date, end_date):
    """
    Download historical daily close prices for the given tickers.

    Parameters:
        tickers (list of str): list of ticker symbols
        start_date (str): 'YYYY-MM-DD'
        end_date (str): 'YYYY-MM-DD'

    Returns:
        pandas DataFrame:
            index = dates
            columns = tickers
            values = closing prices
    """
    if not tickers:
        return pd.DataFrame()
    
    # yfinance supports multiple tickers at once
    data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=False)['Close']
    
    # If only one ticker, ensure DataFrame with proper column name
    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers[0])
    
    # Sort + Fill missing prices (forward-fill)
    data = data.sort_index()
    data.ffill(inplace=True)
    
    return data

# -------------------------------
# Get the latest price for a list of tickers
# -------------------------------
def get_latest_close_prices(tickers):
    """
    Get the latest available price (usually yesterday close) for each ticker.

    Parameters:
        tickers (list of str): list of ticker symbols

    Returns:
        dict: {ticker: latest_price}
    """
    latest_prices = {}
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        hist = stock.history(period='5d')  # last 5 days to ensure at least yesterday
        if not hist.empty:
            latest_prices[ticker] = hist['Close'].iloc[-1]
        else:
            latest_prices[ticker] = None
    return latest_prices



"""
API Tools for Portfolio Database

"""

def get_market_transaction_data(ticker, quantity, name=None, sector=None):
    """
    Fetches current market price and metadata for a ticker.

    Parameters:
        - ticker (str)
        - quantity (float): positive for buy, negative for sell
        - name (str, optional)
        - sector (str, optional)

    Returns:
        dict: {date, ticker, name, sector, quantity, price}
    """
    stock = yf.Ticker(ticker)
    hist = stock.history(period="5d")
    
    if hist.empty:
        raise ValueError(f"Market price not available for ticker {ticker}")
    
    market_price = round(hist["Close"].iloc[-1], 2)
    info = stock.info

    if name is None:
        name = info.get("shortName", ticker)
    if sector is None:
        sector = info.get("sector", "Unknown")
    
    transaction_date = date.today().isoformat()
    
    return {
        "date": transaction_date,
        "ticker": ticker,
        "name": name,
        "sector": sector,
        "quantity": quantity,
        "price": market_price
    }

from tools.database.db_tools import insert_transaction

def buy_stock_flow(ticker, quantity):
    """
     
    1. Download data from Yahoo Finance
    2. Saves them in the Database
    """

    print(f"[*] Buying {ticker}...")
    market_data = get_market_transaction_data(ticker, quantity)
    
    # 2. Salvataggio nel DB
    if market_data:
        db_res = insert_transaction(market_data)
        if db_res["status"] == "ok":
            return {
                "message": f"Transaction completed! I have added {quantity} actions of {market_data['name']} to the database.",
                "data": market_data
            }
    return {"error": "Something went wrong during the purchase."}

