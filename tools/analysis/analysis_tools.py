import pandas as pd
import numpy as np
from scipy.optimize import minimize
from datetime import datetime, timedelta
import os
import warnings
from newsapi import NewsApiClient
from textblob import TextBlob

# Import dai tuoi file tools esistenti
from tools.database.db_tools import get_current_portfolio, get_sector_allocation
from tools.api.api_tools import get_latest_close_prices, get_historical_prices

warnings.simplefilter(action='ignore', category=FutureWarning)

# --- CONFIGURAZIONE ---
NEWS_API_KEY = "f478f541562347d38b316ef6a2d19cac"

# ============================
# HELPER FUNCTIONS
# ============================

def unwrap_db_response(resp):
    """
    Estrae in sicurezza la lista dei dati dalla risposta del DB.
    Gestisce il formato { 'status': 'ok', 'data': [...] } di db_tools.py.
    """
    if resp and isinstance(resp, dict) and resp.get("status") == "ok":
        return resp.get("data", [])
    return []

# ============================
# FUNZIONI ANALISI CORE
# ============================

# 1. COMPUTE RETURNS
def tool_compute_returns():
    """
    Calcola il ROI del portafoglio incrociando i dati del DB con i prezzi API.
    """
    # Recupera i dati puliti usando l'helper
    portfolio = unwrap_db_response(get_current_portfolio())
    
    if not portfolio: 
        return {"error": "Portafoglio vuoto o errore DB"}

    tickers = [p['ticker'] for p in portfolio]
    current_prices = get_latest_close_prices(tickers)
      
    total_purchase_cost = 0.0
    total_current_value = 0.0

    for p in portfolio:
        ticker = p['ticker']
        # Conversione in float per sicurezza
        qty = float(p['total_quantity'])
        avg_price = float(p['avg_price'])
        
        # Prezzo corrente o fallback sul prezzo medio se API fallisce
        current_price = current_prices.get(ticker, avg_price)
        if current_price is None:
            current_price = avg_price
        
        total_purchase_cost += qty * avg_price
        total_current_value += qty * float(current_price)
    
    if total_purchase_cost == 0: 
        return {"error": "Il costo totale di acquisto è zero."}
    
    roi = ((total_current_value - total_purchase_cost) / total_purchase_cost) * 100
    
    return {
        "roi_percentage": round(roi, 2),
        "total_current_value": round(total_current_value, 2),
        "total_invested_cost": round(total_purchase_cost, 2),
        "currency": "USD"
    }

# 2. BEST RETURNS
def get_best_returns_data():
    """
    Restituisce una lista di asset ordinati per performance.
    """
    portfolio_rows = unwrap_db_response(get_current_portfolio())

    if not portfolio_rows:
        return []

    tickers = [row['ticker'] for row in portfolio_rows]
    current_prices = get_latest_close_prices(tickers)

    results = []

    for row in portfolio_rows:
        ticker = row['ticker']
        quantity = float(row['total_quantity'])
        avg_price = float(row['avg_price'])
        
        current_price = current_prices.get(ticker, avg_price)
        if current_price is None: current_price = avg_price
        current_price = float(current_price)

        invested_value = quantity * avg_price
        current_value = quantity * current_price
        profit_loss = current_value - invested_value
        
        return_pct = ((current_price - avg_price) / avg_price) * 100 if avg_price > 0 else 0.0

        results.append({
            "ticker": ticker,
            "name": row['name'],
            "sector": row['sector'],
            "quantity": int(quantity),
            "avg_purchase_price": round(avg_price, 2),
            "current_market_price": round(current_price, 2),
            "return_percentage": round(return_pct, 2),
            "profit_loss_usd": round(profit_loss, 2)
        })

    # Ordina dal migliore al peggiore
    results.sort(key=lambda x: x['return_percentage'], reverse=True)
    return results


# COMPACT SUMMARY OF BEST RETURNS
def get_best_returns_summary(top_n: int = 5) -> dict:
    """
    Returns a compact summary of best returns: only ticker + return_percentage.
    """
    rows = get_best_returns_data()  # <-- se da te si chiama diversamente, cambia qui
    if not rows:
        return {"status": "error", "message": "No portfolio data available.", "top": []}

    # ensure sorted by return_percentage desc
    rows_sorted = sorted(rows, key=lambda r: r.get("return_percentage", float("-inf")), reverse=True)
    top = rows_sorted[: max(1, int(top_n))]

    compact = [
        {"ticker": r.get("ticker"), "return_percentage": r.get("return_percentage")}
        for r in top
    ]

    return {"status": "ok", "top": compact}


# 3. SECTOR DIVERSIFICATION
def tool_sector_diversification_comparison():
    """
    Confronta allocazione iniziale (Costo) vs Attuale (Mercato).
    Restituisce un DataFrame Pandas.
    """
    # 1. Allocazione Iniziale (dal DB - usa invested_value)
    raw_allocation = unwrap_db_response(get_sector_allocation())
    
    # Calcoliamo il totale investito per derivare le percentuali iniziali
    total_invested_db = sum(float(item['total_invested']) for item in raw_allocation)
    initial_dict = {}
    if total_invested_db > 0:
        for item in raw_allocation:
            initial_dict[item['sector']] = (float(item['total_invested']) / total_invested_db) * 100
    
    # 2. Allocazione Corrente (dal DB + API - Valore Mercato)
    portfolio_data = unwrap_db_response(get_current_portfolio())

    if not portfolio_data: 
        return pd.DataFrame()
    
    # Convertiamo la lista di dict in DataFrame
    df_portfolio = pd.DataFrame(portfolio_data)
    
    tickers = df_portfolio['ticker'].tolist()
    current_prices = get_latest_close_prices(tickers)
    
    # Mappiamo i prezzi e calcoliamo il valore attuale
    df_portfolio['current_price'] = df_portfolio['ticker'].map(current_prices).fillna(0)
    df_portfolio['current_value'] = df_portfolio['total_quantity'] * df_portfolio['current_price']

    # Raggruppamento per settore
    current_sector_values = df_portfolio.groupby('sector')['current_value'].sum()
    total_market_value = current_sector_values.sum()
    
    results_data = []
    all_sectors = set(initial_dict.keys()).union(set(current_sector_values.index))

    for sector in sorted(all_sectors):
        init_p = float(initial_dict.get(sector, 0.0))
        curr_val = float(current_sector_values.get(sector, 0.0))
        
        curr_p = (curr_val / total_market_value * 100) if total_market_value > 0 else 0.0
        drift = curr_p - init_p
        
        results_data.append({
            "sector": sector,
            "initial_weight_pct": round(init_p, 2),
            "current_weight_pct": round(curr_p, 2),
            "drift_pct": round(drift, 2)
        })

    return pd.DataFrame(results_data)

# 4. MARKOWITZ OPTIMIZATION
def tool_optimize_markowitz_target(target_return_annualized=0.10):
    rows = unwrap_db_response(get_current_portfolio())
    tickers = [row['ticker'] for row in rows]
    
    if len(tickers) < 2: 
        return {"error": "Servono almeno 2 asset per l'ottimizzazione."}

    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    data = get_historical_prices(tickers, start_date, end_date)
    if data.empty: return {"error": "Dati storici non disponibili."}
      
    daily_returns = data.pct_change().dropna()
    expected_returns = daily_returns.mean() * 252
    cov_matrix = daily_returns.cov() * 252 

    max_possible = expected_returns.max()
    if target_return_annualized > max_possible:
        return {"error": "Target troppo alto", "max_possible_return": round(max_possible, 4)}

    def portfolio_variance(weights):
        return np.dot(weights.T, np.dot(cov_matrix, weights))

    cons = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        {'type': 'eq', 'fun': lambda x: np.dot(x, expected_returns) - target_return_annualized}
    ]
    bounds = tuple((0, 1) for _ in range(len(tickers)))
    init_guess = [1/len(tickers)] * len(tickers)

    try:
        optimized = minimize(portfolio_variance, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
    except Exception:
        return {"error": "Ottimizzazione matematica fallita."}

    if not optimized.success: return {"error": "Il solver non ha trovato soluzioni."}

    volatility = np.sqrt(optimized.fun)
    weights_dict = {tickers[i]: round(optimized.x[i], 4) for i in range(len(tickers)) if optimized.x[i] > 0.001}

    return {
        "target_return": target_return_annualized,
        "estimated_volatility": round(volatility, 4),
        "optimized_weights": weights_dict
    }

# 5. SENTIMENT ANALYSIS
def tool_sentiment_analysis(ticker=None):
    # Se non c'è ticker, prendiamo il più grande nel portafoglio
    if not ticker:
        portfolio = unwrap_db_response(get_current_portfolio())
        if portfolio:
            # Ordiniamo la lista in Python
            top_asset = sorted(portfolio, key=lambda x: x['total_quantity'], reverse=True)[0]
            ticker = top_asset['ticker']
        else:
            return {"error": "Nessun ticker trovato o portafoglio vuoto."}

    if not NEWS_API_KEY:
        return {"error": "API Key mancante. Esegui export NEWS_API_KEY='...'"}

    try:
        newsapi = NewsApiClient(api_key=NEWS_API_KEY)
        headlines = newsapi.get_everything(q=ticker, language='en', sort_by='publishedAt', page_size=5)
        articles = headlines.get('articles', [])
        
        if not articles:
            return {"ticker": ticker, "score": 0, "label": "No Data", "articles": []}

        scores = []
        articles_data = []
        
        for art in articles:
            title = art['title']
            description = art['description'] if art['description'] else "" # Prendiamo anche la descrizione
            
            # Uniamo titolo e descrizione per avere più testo da analizzare
            full_text = f"{title}. {description}"
            
            # Analizziamo il testo completo
            score = TextBlob(full_text).sentiment.polarity
            scores.append(score)
            
            articles_data.append({
                "title": title, 
                "score": round(score, 2), 
                "source": art['source']['name']
            })

        avg_score = sum(scores) / len(scores)
        label = "BULLISH" if avg_score > 0.1 else "BEARISH" if avg_score < -0.1 else "NEUTRAL"

        return {
            "ticker": ticker,
            "average_score": round(avg_score, 2),
            "sentiment_label": label,
            "article_count": len(articles),
            "articles": articles_data
        }

    except Exception as e:
        return {"error": f"Errore API News: {str(e)}"}

# ============================
# DATAFRAME HELPERS (TIME SERIES)
# ============================

# 6. PORTFOLIO VALUE OVER TIME
def portfolio_value_over_time(transactions_df, price_df):
    """
    Calcola il valore giornaliero del portafoglio basandosi sulle transazioni storiche.

    Parameters:
        transactions_df (DataFrame): ['date','ticker','quantity']
        price_df (DataFrame): index=dates, columns=tickers, valori=prezzi chiusura

    Returns:
        pandas Series: index=dates, values=portafoglio giornaliero
    """
    transactions_df['date'] = pd.to_datetime(transactions_df['date'])
    all_dates = price_df.index
    tickers = price_df.columns
    qty_df = pd.DataFrame(0, index=all_dates, columns=tickers)

    for ticker in tickers:
        df_t = transactions_df[transactions_df['ticker'] == ticker]
        df_t = df_t.groupby('date')['quantity'].sum().cumsum()
        df_t = df_t.reindex(all_dates, method='ffill').fillna(0)
        qty_df[ticker] = df_t

    portfolio_values = (qty_df * price_df).sum(axis=1)
    return portfolio_values


# ============================
# MAIN TEST (Se eseguito come script)
# ============================
if __name__ == "__main__":
    print("=== Testing Integration ===")
    
    # Test 1: Returns
    print("\n1. Compute Returns:")
    print(tool_compute_returns())


    # Test 2: Best Returns
    print("\n3. Best Returns Data:")
    best_returns = get_best_returns_data()
    for item in best_returns:
        print(item) 

    # Test 3: Sector Diversification
    print("\n4. Sector Diversification Comparison:")
    sector_df = tool_sector_diversification_comparison()
    if not sector_df.empty: 
        print(sector_df)
    else:        
        print("Nessun dato disponibile.")       

    # Test 4: Markowitz Optimization
    print("\n5. Markowitz Optimization:")
    optimization_result = tool_optimize_markowitz_target(0.10)
    print(optimization_result)  
    
    # Test 5: Sentiment Analysis
    print("\n6. Sentiment Analysis:")  
    sentiment_result = tool_sentiment_analysis()
    print(sentiment_result)
    print("=== End of Tests ===")