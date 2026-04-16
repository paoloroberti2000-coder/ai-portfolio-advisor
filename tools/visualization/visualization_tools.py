"""
Visualization tools for AI Portfolio Manager.
--------------------------------------------
This module contains functions to visualize:
- portfolio composition
- sector allocation
- portfolio value and performance
- comparisons vs benchmarks or other assets
"""

# ============================
# Imports
# ============================
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from datetime import datetime
from tools.database.db_tools import ( 
    get_current_portfolio,  
    get_transactions_by_date,
    get_sector_allocation
)
from tools.api.api_tools import (
    get_historical_prices,
    get_latest_close_prices
)
from tools.analysis.analysis_tools import (
    unwrap_db_response, 
    portfolio_value_over_time,
    tool_optimize_markowitz_target,
    tool_sentiment_analysis
)

# ============================
# Portfolio Composition
# ============================
def plot_portfolio_composition(save_path=None):
    """
    Plot the portfolio composition showing the invested value per asset.
    Bars are colored by sector.
    """

    portfolio_data = unwrap_db_response(get_current_portfolio())

    if not portfolio_data:
        print("Portfolio is empty. No data to plot.")
        return {'data': pd.DataFrame(), 'image_path': None}

    df = pd.DataFrame(portfolio_data)

    tickers = df['ticker'].tolist()
    values = df['invested_value'].tolist()
    sectors = df['sector'].tolist()

    # Create a color map for sectors
    unique_sectors = sorted(set(sectors))
    cmap = plt.get_cmap("tab10")
    sector_colors = {sector: cmap(i % cmap.N) for i, sector in enumerate(unique_sectors)}
    bar_colors = [sector_colors[sector] for sector in sectors]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(tickers, values, color=bar_colors)
    plt.xlabel("Asset")
    plt.ylabel("Invested Value")
    plt.title("Portfolio Composition by Asset")
    plt.xticks(rotation=45)

    # Legend
    legend_elements = [
        Patch(facecolor=sector_colors[sector], label=sector)
        for sector in unique_sectors
    ]
    plt.legend(handles=legend_elements, title="Sector")
    plt.tight_layout()

    image_path = None
    if save_path:
        base, ext = os.path.splitext(save_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path_ts = f"{base}_{timestamp}{ext}"

        os.makedirs(os.path.dirname(save_path_ts), exist_ok=True)
        plt.savefig(save_path_ts)
        image_path = save_path_ts
        plt.close()
    else:
        plt.show()

    return {'data': df, 'image_path': image_path}



# ============================
# Portfolio Sector Allocation
# ============================
def plot_sector_allocation(save_path=None):
    """
    Plot in a pie chart the sector allocation of the current portfolio.
    """

    sector_data = unwrap_db_response(get_sector_allocation())

    if not sector_data:
        print("No sector allocation data available.")
        return {'data': pd.DataFrame(), 'image_path': None}

    df = pd.DataFrame(sector_data)  # ['sector', 'total_invested']

    if df.empty:
        print("Sector allocation is empty.")
        return {'data': df, 'image_path': None}

    # Calcolo percentuali
    total = df['total_invested'].sum()
    df['percentage'] = df['total_invested'] / total * 100

    # Pie chart
    plt.figure(figsize=(8, 6))
    plt.pie(
        df['percentage'],
        labels=df['sector'],
        autopct='%1.1f%%',
        startangle=90
    )
    plt.title("Asset Allocation per Sector")
    plt.tight_layout()

    image_path = None
    if save_path:
        base, ext = os.path.splitext(save_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path_ts = f"{base}_{timestamp}{ext}"
        os.makedirs(os.path.dirname(save_path_ts), exist_ok=True)
        plt.savefig(save_path_ts)
        image_path = save_path_ts
        plt.close()
    else:
        plt.show()

    return {'data': df, 'image_path': image_path}



# ============================
# Portfolio Performance over time
# ============================

def plot_portfolio_value_over_time(end_date=None, save_path=None):
    """
    Plot the portfolio value over time usando le transazioni storiche.

    Parameters:
        start_date (str, optional): 'YYYY-MM-DD', default '2025-01-02'
        end_date (str, optional): 'YYYY-MM-DD', default today
        save_path (str, optional): path to save the plot image (e.g., 'plots/portfolio_value.png').
                                   Se None, il grafico viene mostrato.

    Returns:
        dict: {'data': pandas Series of portfolio value, 'image_path': str or None}
    """
    # --- Default dates ---
    start_date = '2025-01-02'
    if end_date is None:
        end_date = pd.Timestamp.today().strftime('%Y-%m-%d')

    # --- Recupera transazioni ---
    transactions_resp = get_transactions_by_date(start_date, end_date)
    transactions_rows = unwrap_db_response(transactions_resp)

    if not transactions_rows:
        print("No transactions in the specified date range.")
        return {'data': pd.Series(dtype=float), 'image_path': None}

    transactions_df = pd.DataFrame(
        transactions_rows,
        columns=['id','date','ticker','name','sector','quantity','price']
    )
    transactions_df['date'] = pd.to_datetime(transactions_df['date'])
    transactions_df = transactions_df[['date','ticker','quantity']]

    tickers = transactions_df['ticker'].unique().tolist()
    price_df = get_historical_prices(tickers, start_date, end_date)

    if price_df.empty:
        print("No price data available.")
        return {'data': pd.Series(dtype=float), 'image_path': None}

    # --- Calcola portafoglio giorno per giorno ---
    portfolio_series = portfolio_value_over_time(transactions_df, price_df)

    # --- Plot ---
    plt.figure(figsize=(11,6))
    plt.plot(portfolio_series.index, portfolio_series.values, label='Portfolio Value', color='tab:blue')

    # --- Marker buy/sell/mixed ---
    daily_trades = transactions_df.groupby('date')['quantity'].sum().reset_index()
    buy_days = daily_trades[daily_trades['quantity'] > 0]['date']
    sell_days = daily_trades[daily_trades['quantity'] < 0]['date']
    mixed_days = daily_trades[daily_trades['quantity'] == 0]['date']

    for days, color, marker, label in [
        (buy_days, 'green', '^', 'Net Buy'),
        (sell_days, 'red', 'v', 'Net Sell'),
        (mixed_days, 'orange', 'o', 'Buy & Sell')
    ]:
        values = portfolio_series.loc[portfolio_series.index.isin(days)]
        if not values.empty:
            plt.scatter(values.index, values.values, color=color, marker=marker, s=80 if marker=='o' else 50,
                        edgecolors='k' if marker=='o' else None, label=label, zorder=3)

    # --- Styling ---
    plt.title("Portfolio Value Over Time")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    # --- Save or show ---
    image_path = None
    if save_path:
        base, ext = os.path.splitext(save_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path_ts = f"{base}_{timestamp}{ext}"
        os.makedirs(os.path.dirname(save_path_ts), exist_ok=True)
        plt.savefig(save_path_ts)
        image_path = save_path_ts
        plt.close()
    else:
        plt.show()

    return {'data': portfolio_series, 'image_path': image_path}


# ============================
# Portfolio's Assets Comparison
# ============================
def plot_portfolio_performance(save_path=None):
    """
    Compare invested value vs current value per asset
    using latest available market prices.

    Returns:
        dict: {
            'data': pandas DataFrame,
            'image_path': str or None
        }
    """

    portfolio_data = unwrap_db_response(get_current_portfolio())

    if not portfolio_data:
        print("Portfolio is empty.")
        return {'data': pd.DataFrame(), 'image_path': None}

    df = pd.DataFrame(portfolio_data)

    # Campi minimi richiesti
    required_cols = {'ticker', 'total_quantity', 'avg_price', 'invested_value'}
    if not required_cols.issubset(df.columns):
        print("Missing required portfolio fields.")
        return {'data': pd.DataFrame(), 'image_path': None}

    tickers = df['ticker'].tolist()
    quantities = df['total_quantity'].astype(float)
    invested_values = df['invested_value'].astype(float)

    # Prezzi di mercato
    latest_prices = get_latest_close_prices(tickers)

    df['current_price'] = df['ticker'].map(latest_prices)
    df['current_price'] = df['current_price'].fillna(df['avg_price'])

    df['current_value'] = quantities * df['current_price']
    df['pnl_pct'] = (
        (df['current_value'] - invested_values) / invested_values * 100
    ).replace([np.inf, -np.inf], 0).fillna(0)

    # Plot
    x = np.arange(len(df))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, invested_values, width, label="Invested Value")
    plt.bar(x + width/2, df['current_value'], width, label="Current Value")

    for i, pct in enumerate(df['pnl_pct']):
        plt.text(
            x[i],
            max(invested_values.iloc[i], df['current_value'].iloc[i]) * 1.02,
            f"{pct:+.1f}%",
            ha="center",
            fontsize=9,
            fontweight="bold"
        )

    plt.xticks(x, tickers, rotation=45)
    plt.ylabel("Value")
    plt.title("Portfolio Performance by Asset")
    plt.legend()
    plt.tight_layout()

    image_path = None
    if save_path:
        base, ext = os.path.splitext(save_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path_ts = f"{base}_{timestamp}{ext}"
        os.makedirs(os.path.dirname(save_path_ts), exist_ok=True)
        plt.savefig(save_path_ts)
        image_path = save_path_ts
        plt.close()
    else:
        plt.show()

    return {
        'data': df[['ticker', 'invested_value', 'current_value', 'pnl_pct']],
        'image_path': image_path
    }



# ============================
# Sector Comparison
# ============================
def plot_sector_performance(save_path=None):
    """
    Compare invested vs current value aggregated by sector.

    Returns:
        dict: {
            'data': pandas DataFrame,
            'image_path': str or None
        }
    """

    portfolio_data = unwrap_db_response(get_current_portfolio())

    if not portfolio_data:
        print("Portfolio is empty.")
        return {'data': pd.DataFrame(), 'image_path': None}

    df = pd.DataFrame(portfolio_data)

    # Campi minimi richiesti
    required_cols = {'ticker', 'sector', 'total_quantity', 'avg_price', 'invested_value'}
    if not required_cols.issubset(df.columns):
        print("Missing required portfolio fields.")
        return {'data': pd.DataFrame(), 'image_path': None}

    df['quantity'] = df['total_quantity'].astype(float)

    tickers = df['ticker'].tolist()
    latest_prices = get_latest_close_prices(tickers)

    df['current_price'] = df['ticker'].map(latest_prices)
    df['current_price'] = df['current_price'].fillna(df['avg_price'])

    df['current_value'] = df['quantity'] * df['current_price']

    # Aggregazione per settore
    sector_df = (
        df.groupby('sector')[['invested_value', 'current_value']]
        .sum()
        .reset_index()
    )

    sector_df['pnl_pct'] = (
        (sector_df['current_value'] - sector_df['invested_value'])
        / sector_df['invested_value'] * 100
    ).replace([np.inf, -np.inf], 0).fillna(0)

    # Plot
    x = np.arange(len(sector_df))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, sector_df['invested_value'], width, label="Invested Value")
    plt.bar(x + width/2, sector_df['current_value'], width, label="Current Value")

    for i, pct in enumerate(sector_df['pnl_pct']):
        plt.text(
            x[i],
            max(
                sector_df.loc[i, 'invested_value'],
                sector_df.loc[i, 'current_value']
            ) * 1.02,
            f"{pct:+.1f}%",
            ha="center",
            fontsize=9,
            fontweight="bold"
        )

    plt.xticks(x, sector_df['sector'], rotation=45)
    plt.ylabel("Value")
    plt.title("Portfolio Performance by Sector")
    plt.legend()
    plt.tight_layout()

    image_path = None
    if save_path:
        base, ext = os.path.splitext(save_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path_ts = f"{base}_{timestamp}{ext}"
        os.makedirs(os.path.dirname(save_path_ts), exist_ok=True)
        plt.savefig(save_path_ts)
        image_path = save_path_ts
        plt.close()
    else:
        plt.show()

    return {
        'data': sector_df,
        'image_path': image_path
    }



# ============================
# Portfolio vs Benchmark
# ============================
# ============================
# Portfolio vs Benchmark
# ============================
def plot_portfolio_vs_benchmark(
    benchmark_ticker='^GSPC',
    start_date=None,
    end_date=None,
    save_path=None
):
    """
    Plot portfolio value over time compared to a benchmark,
    normalized to 100 at portfolio start date.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import os
    from datetime import datetime

    if start_date is None:
        start_date = '2025-06-02'
    if end_date is None:
        end_date = pd.Timestamp.today().strftime('%Y-%m-%d')
    
    # 1. Transactions
    
    transactions_rows = unwrap_db_response(
        get_transactions_by_date(start_date, end_date)
    )

    if not transactions_rows:
        print("No transactions available.")
        return None

    transactions_df = pd.DataFrame(
        transactions_rows,
        columns=['id','date','ticker','name','sector','quantity','price']
    )[['date','ticker','quantity']]

    transactions_df['date'] = pd.to_datetime(transactions_df['date'])

    tickers = transactions_df['ticker'].unique().tolist()

    # 2. Prices

    price_df = get_historical_prices(
        tickers,
        start_date,
        end_date
    )

    if price_df.empty:
        print("No price data available.")
        return None

    
    # 3. Portfolio series
    
    portfolio_series = portfolio_value_over_time(
        transactions_df,
        price_df
    )

    if portfolio_series.empty:
        print("Portfolio series empty.")
        return None

    # Normalize to 100
    portfolio_series = portfolio_series / portfolio_series.iloc[0] * 100

    
    # 4. Benchmark
    
    benchmark_series = None
    try:
        benchmark_df = get_historical_prices(
            [benchmark_ticker],
            start_date,
            end_date
        )
        benchmark_series = benchmark_df[benchmark_ticker]
        benchmark_series = benchmark_series / benchmark_series.iloc[0] * 100
    except Exception as e:
        print(f"Warning: benchmark error {e}")

    
    # 5. Plot
    
    plt.figure(figsize=(12,6))
    plt.plot(portfolio_series.index, portfolio_series.values, label="Portfolio", linewidth=2)

    if benchmark_series is not None:
        plt.plot(
            benchmark_series.index,
            benchmark_series.values,
            linestyle='--',
            label=f"Benchmark ({benchmark_ticker})"
        )

    plt.title("Portfolio vs Benchmark")
    plt.xlabel("Date")
    plt.ylabel("Normalized Value (Base = 100)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    image_path = None
    if save_path:
        base, ext = os.path.splitext(save_path)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        image_path = f"{base}_{ts}{ext}"
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        plt.savefig(image_path)
        plt.close()
    else:
        plt.show()

    return {
        "portfolio_series": portfolio_series,
        "benchmark_series": benchmark_series,
        "image_path": image_path
    }





# ============================
# Comparison between Assets
# ============================
def plot_normalized_comparison(tickers=None, start_date=None, end_date=None, save_path=None):
    """
    Plot normalized price comparison for two or more tickers, starting from 100.
    Uses api_tools.get_historical_prices to fetch data.

    Parameters:
        tickers (list of str, optional): tickers to compare. If None, takes top portfolio tickers.
        start_date (str, optional): "YYYY-MM-DD", default '2025-01-02'
        end_date (str, optional): "YYYY-MM-DD", default today
        save_path (str, optional): path to save plot image
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    # --- Default tickers from portfolio if not provided ---
    if tickers is None:
        portfolio = unwrap_db_response(get_current_portfolio())
        if not portfolio:
            print("No tickers available in portfolio.")
            return {'data': pd.DataFrame(), 'image_path': None}
        tickers = [p['ticker'] for p in portfolio]

    if not tickers:
        print("No tickers provided.")
        return {'data': pd.DataFrame(), 'image_path': None}

    # --- Default dates ---
    if start_date is None:
        start_date = '2025-01-02'
    if end_date is None:
        end_date = pd.Timestamp.today().strftime('%Y-%m-%d')

    # --- Fetch historical prices ---
    price_df = get_historical_prices(tickers, start_date, end_date)
    if price_df.empty:
        print("No historical prices available for the specified tickers/dates.")
        return {'data': pd.DataFrame(), 'image_path': None}

    # Forward-fill missing values
    price_df.ffill(inplace=True)

    # Normalize to 100 at the first available date
    normalized = price_df / price_df.iloc[0] * 100

    # --- Plot ---
    plt.figure(figsize=(12,6))
    for col in normalized.columns:
        plt.plot(normalized.index, normalized[col], label=col)

    plt.title("Normalized Price Comparison")
    plt.xlabel("Date")
    plt.ylabel("Normalized Value (Start=100)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    # --- Save or show ---
    image_path = None
    if save_path:
        base, ext = os.path.splitext(save_path)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        image_path = f"{base}_{ts}{ext}"
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        plt.savefig(image_path)
        plt.close()
    else:
        plt.show()


    return {'data': normalized, 'image_path': image_path}



# ============================
# Stock Price over time 
# ============================
def plot_stock_price(ticker, start_date='2025-01-02', end_date=None, save_path=None):
    """
    Plot absolute stock price over time.

    Parameters:
        ticker (str): stock ticker
        start_date (str): 'YYYY-MM-DD', default portfolio start
        end_date (str): 'YYYY-MM-DD', defaults today
        save_path (str, optional): path to save the plot image

    Returns:
        dict: {'data': pandas Series of prices, 'image_path': str or None}
    """
    if end_date is None:
        end_date = pd.Timestamp.today().strftime('%Y-%m-%d')

    # Get historical prices
    df = get_historical_prices([ticker], start_date, end_date)
    if df.empty or ticker not in df:
        print(f"No data found for {ticker}")
        return {'data': pd.Series(dtype=float), 'image_path': None}

    plt.figure(figsize=(12,6))
    plt.plot(df.index, df[ticker], label=ticker)
    plt.title(f"{ticker} Price Over Time")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    image_path = None
    if save_path:
        base, ext = os.path.splitext(save_path)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        image_path = f"{base}_{ts}{ext}"
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        plt.savefig(image_path)
        plt.close()
    else:
        plt.show()

    return {'data': df[ticker], 'image_path': image_path}


# ============================
# Asset Correlation Heatmap
# ============================
def plot_asset_correlation_heatmap(start_date='2025-01-02', end_date=None, save_path=None):
    """
    Plot a heatmap of correlations between portfolio assets based on daily returns.

    Parameters:
        start_date (str): 'YYYY-MM-DD' start of historical data
        end_date (str): 'YYYY-MM-DD' end of historical data, defaults today
        save_path (str): path to save image, if None shows plot

    Returns:
        dict: {'data': correlation DataFrame, 'image_path': str or None}
    """
    if end_date is None:
        end_date = pd.Timestamp.today().strftime('%Y-%m-%d')

    # 1. Get portfolio tickers
    portfolio = unwrap_db_response(get_current_portfolio())
    if not portfolio:
        print("Portfolio empty.")
        return {'data': pd.DataFrame(), 'image_path': None}

    tickers = [p['ticker'] for p in portfolio]
    if len(tickers) < 2:
        print("At least 2 assets needed for correlation.")
        return {'data': pd.DataFrame(), 'image_path': None}

    # 2. Fetch historical prices
    price_df = get_historical_prices(tickers, start_date, end_date)
    if price_df.empty:
        print("No historical price data.")
        return {'data': pd.DataFrame(), 'image_path': None}

    # 3. Compute daily returns
    returns_df = price_df.pct_change().dropna()

    # 4. Compute correlation matrix
    corr_matrix = returns_df.corr()

    # 5. Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap='coolwarm',
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        cbar_kws={'label': 'Correlation'}
    )
    plt.title("Asset Correlation Heatmap")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()

    # 6. Save or show
    image_path = None
    if save_path:
        base, ext = os.path.splitext(save_path)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        image_path = f"{base}_{ts}{ext}"
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        plt.savefig(image_path)
        plt.close()
    else:
        plt.show()

    return {'data': corr_matrix, 'image_path': image_path}



# ============================
# Sector Returns Correlation Heatmap
# ============================
def plot_sector_correlation_heatmap(start_date='2025-01-02', end_date=None, save_path=None):
    """
    Plot a heatmap showing the correlation of daily returns between sectors.
    
    Parameters:
        start_date (str): 'YYYY-MM-DD' start date for historical prices
        end_date (str): 'YYYY-MM-DD' end date, defaults today
        save_path (str, optional): path to save the image
    
    Returns:
        dict: {
            'correlation_matrix': pandas DataFrame,
            'image_path': str or None
        }
    """

    if end_date is None:
        end_date = pd.Timestamp.today().strftime('%Y-%m-%d')
    
    # 1. Recupera portafoglio
    portfolio = unwrap_db_response(get_current_portfolio())
    if not portfolio:
        print("Portfolio vuoto.")
        return {'correlation_matrix': pd.DataFrame(), 'image_path': None}

    df_portfolio = pd.DataFrame(portfolio)
    if df_portfolio.empty or 'sector' not in df_portfolio.columns:
        print("Dati portafoglio insufficienti per settore.")
        return {'correlation_matrix': pd.DataFrame(), 'image_path': None}

    # 2. Recupera prezzi storici
    tickers = df_portfolio['ticker'].tolist()
    price_df = get_historical_prices(tickers, start_date, end_date)
    if price_df.empty:
        print("Prezzi storici non disponibili.")
        return {'correlation_matrix': pd.DataFrame(), 'image_path': None}

    # 3. Calcola ritorni giornalieri
    returns_df = price_df.pct_change().dropna()

    # 4. Aggiunge colonna settore per aggregazione
    sector_map = df_portfolio.set_index('ticker')['sector'].to_dict()
    returns_df_sector = returns_df.rename(columns=sector_map)

    # 5. Aggregazione per settore (media dei ticker)
    sector_returns = returns_df_sector.groupby(axis=1, level=0).mean()

    # 6. Matrice di correlazione
    corr_matrix = sector_returns.corr()

    if corr_matrix.empty:
        print("Nessuna correlazione calcolabile.")
        return {'correlation_matrix': pd.DataFrame(), 'image_path': None}

    # 7. Heatmap
    plt.figure(figsize=(10,8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Sector Returns Correlation Heatmap")
    plt.tight_layout()

    image_path = None
    if save_path:
        base, ext = os.path.splitext(save_path)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        image_path = f"{base}_{ts}{ext}"
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        plt.savefig(image_path)
        plt.close()
    else:
        plt.show()

    return {'correlation_matrix': corr_matrix, 'image_path': image_path}


# ============================
# Current allocation vs Markowitz optimal allocation
# ============================
def plot_allocation_vs_markowitz(target_return=0.10, save_path=None):
    """
    Confronta allocazione attuale del portafoglio con pesi ottimizzati
    secondo Markowitz per un target di rendimento annualizzato.

    Parameters:
        target_return (float): target annualizzato per Markowitz (es. 0.10 = 10%)
        save_path (str, optional): path per salvare il grafico

    Returns:
        dict: {
            'data': pandas DataFrame con tickers, pesi attuali e ottimizzati,
            'image_path': str o None
        }
    """

    # --- Dati portafoglio attuale ---
    portfolio = unwrap_db_response(get_current_portfolio())
    if not portfolio:
        print("Portfolio vuoto.")
        return {'data': pd.DataFrame(), 'image_path': None}

    df = pd.DataFrame(portfolio)
    df['quantity'] = df['total_quantity'].astype(float)
    df['invested_value'] = df['quantity'] * df['avg_price']

    total_value = df['invested_value'].sum()
    df['current_weight_pct'] = df['invested_value'] / total_value * 100

    # --- Ottimizzazione Markowitz ---
    markowitz_res = tool_optimize_markowitz_target(target_return)
    if 'error' in markowitz_res:
        print(f"Markowitz optimization failed: {markowitz_res['error']}")
        return {'data': df, 'image_path': None}

    weights_dict = markowitz_res['optimized_weights']
    df['markowitz_weight_pct'] = df['ticker'].map(weights_dict).fillna(0) * 100

    # --- Plot side-by-side ---
    x = np.arange(len(df))
    width = 0.35

    plt.figure(figsize=(12,6))
    plt.bar(x - width/2, df['current_weight_pct'], width, color='tab:blue', label='Current Allocation')
    plt.bar(x + width/2, df['markowitz_weight_pct'], width, color='tab:orange', label='Markowitz Allocation')

    plt.xticks(x, df['ticker'], rotation=45)
    plt.ylabel("Weight (%)")
    plt.title(f"Current vs Markowitz Allocation (Target Return: {target_return*100:.1f}%)")
    plt.legend()
    plt.tight_layout()

    image_path = None
    if save_path:
        base, ext = os.path.splitext(save_path)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        image_path = f"{base}_{ts}{ext}"
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        plt.savefig(image_path)
        plt.close()
    else:
        plt.show()

    return {'data': df[['ticker','current_weight_pct','markowitz_weight_pct']], 'image_path': image_path}




def plot_sentiment_analysis(tickers=None, top_n=5, save_path=None):
    """
    Plot sentiment analysis for a list of tickers or top portfolio assets.
    
    Parameters:
        tickers (list of str, optional): tickers da analizzare; default = top portfolio tickers
        top_n (int): numero massimo di ticker da visualizzare (se tickers None)
        save_path (str, optional): percorso per salvare il grafico
     
    Returns:
        dict: {
            'data': DataFrame con ticker, avg_score, label, article_count
            'image_path': path immagine salvata o None
        }
    """
    # --- Preleva ticker dal portafoglio se non forniti ---
    if tickers is None:
        from tools.database.db_tools import get_current_portfolio
        portfolio = unwrap_db_response(get_current_portfolio())
        if not portfolio:
            print("Portafoglio vuoto, nessun ticker disponibile.")
            return {'data': pd.DataFrame(), 'image_path': None}
        tickers = [p['ticker'] for p in sorted(portfolio, key=lambda x: x['total_quantity'], reverse=True)[:top_n]]

    results = []
    for t in tickers:
        try:
            sentiment = tool_sentiment_analysis(t)
            if 'error' not in sentiment:
                results.append({
                    'ticker': t,
                    'average_score': sentiment['average_score'],
                    'label': sentiment['sentiment_label'],
                    'article_count': sentiment.get('article_count', 0)
                })
        except Exception as e:
            print(f"Errore sentiment per {t}: {e}")

    if not results:
        print("Nessun dato sentiment disponibile.")
        return {'data': pd.DataFrame(), 'image_path': None}

    df = pd.DataFrame(results)

    # --- Colori in base all'etichetta sentiment ---
    color_map = {'BULLISH': 'green', 'NEUTRAL': 'gray', 'BEARISH': 'red'}
    bar_colors = [color_map.get(l, 'blue') for l in df['label']]

    # --- Plot ---
    plt.figure(figsize=(10,6))
    bars = plt.bar(df['ticker'], df['average_score'], color=bar_colors)

    # Annotazioni sopra le barre
    for bar, lbl, count in zip(bars, df['label'], df['article_count']):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.01, f"{lbl}\n({count})",
                 ha='center', fontsize=9, fontweight='bold')

    plt.axhline(0, color='black', linewidth=0.8)
    plt.title("Sentiment Analysis per Asset")
    plt.ylabel("Average Sentiment Score")
    plt.xlabel("Ticker")
    plt.xticks(rotation=45)
    plt.tight_layout()

    # --- Save or show ---
    image_path = None
    if save_path:
        base, ext = os.path.splitext(save_path)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        image_path = f"{base}_{ts}{ext}"
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        plt.savefig(image_path)
        plt.close()
    else:
        plt.show()


    return {'data': df, 'image_path': image_path}