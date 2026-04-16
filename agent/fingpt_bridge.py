import yfinance as yf
import json
from agent.llm_client import LLMClient
from tools.analysis.analysis_tools import tool_sentiment_analysis 

def get_fingpt_opinion(ticker):
    """
    Simulates 'FinGPT': A data-driven Market Analyst.
    It fetches advanced metrics (PE, Target Price, Volatility) 
    to provide a motivated technical/fundamental opinion.
    """
    ticker = ticker.upper()
    
    # 1. Fetch Advanced Data (Directly via yfinance to avoid touching api_tools)
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Extract key metrics with safety defaults
        current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
        target_price = info.get('targetMeanPrice', 0)
        pe_ratio = info.get('trailingPE', 'N/A')
        high_52 = info.get('fiftyTwoWeekHigh', 0)
        recommendation = info.get('recommendationKey', 'none').upper()
        
        # Calculate Upside Potential if data exists
        upside = 0
        if current_price and target_price:
            upside = ((target_price - current_price) / current_price) * 100
            
    except Exception as e:
        return f"FinGPT Error: Could not fetch market data ({str(e)})."

    # 2. Get Sentiment (News)
    try:
        sentiment_data = tool_sentiment_analysis(ticker)
        sentiment_label = sentiment_data.get('sentiment_label', 'Neutral')
        avg_score = sentiment_data.get('average_score', 0)
    except:
        sentiment_label = "Neutral"
        avg_score = 0

    # 3. CONSTRUCT THE ANALYST PROMPT
    # We force the LLM to act like a Senior Equity Analyst using the specific numbers.
    fingpt_prompt = f"""
    Role: You are a Senior Equity Analyst at a top investment bank.
    Tone: Professional, incisive, data-driven, slightly aggressive (growth-focused).
    
    FINANCIAL DATA for {ticker}:
    - Price: ${current_price}
    - Analyst Target: ${target_price} (Implied Upside: {upside:.2f}%)
    - PE Ratio: {pe_ratio}
    - 52-Week High: ${high_52}
    - Wall St. Consensus: {recommendation}
    - News Sentiment: {sentiment_label} (Score: {avg_score})

    TASK:
    Write a concise market opinion (max 2 sentences).
    - JUSTIFY your view using the specific numbers above (e.g., "Trading at 20% discount to target", "PE is too high").
    - Combine technicals (price vs 52w high) with fundamentals (PE/Target).
    
    OUTPUT FORMAT (JSON ONLY):
    {{ 
        "opinion": "Your data-backed reasoning here.", 
        "signal": "BUY / SELL / HOLD" 
    }}
    """

    llm = LLMClient() 
    
    messages = [
        {"role": "system", "content": "You are a financial analyst engine. Output valid JSON only."},
        {"role": "user", "content": fingpt_prompt}
    ]
    
    try:
        response_str = llm.chat(messages)
        data = json.loads(response_str)
        
        opinion = data.get("opinion", "Analysis unavailable.")
        signal = data.get("signal", "HOLD").upper()
        
        # Choose icon based on signal
        icon = "🟢" if "BUY" in signal else "🔴" if "SELL" in signal else "⚪"
        
        return (f"{icon} **FinGPT Analysis ({signal}):**\n"
                f"\"{opinion}\"\n"
                f"*(Target: ${target_price} | PE: {pe_ratio} | Sentiment: {sentiment_label})*")
                
    except Exception as e:
        return f"FinGPT could not formulate an opinion. Error: {e}"
    

import pandas as pd
import numpy as np

def get_fingpt_technical_view(ticker):
    """
    Analyzes Technical Indicators (RSI, SMA) to give a TRADING signal.
    """
    ticker = ticker.upper()
    try:
        # 1. Fetch 3 months of history for technicals
        stock = yf.Ticker(ticker)
        hist = stock.history(period="3mo")
        
        if hist.empty:
            return "FinGPT could not retrieve technical data."

        current_price = hist['Close'].iloc[-1]
        
        # 2. Calculate RSI (14-day)
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        
        # 3. Calculate Trend (Simple Moving Average 50)
        # (We use a short window since we only fetched 3mo)
        sma_50 = hist['Close'].rolling(window=50).mean().iloc[-1]
        trend = "BULLISH" if current_price > sma_50 else "BEARISH"

    except Exception as e:
        return f"Technical Analysis Error: {str(e)}"

    # 4. Prompt for the "Technical Trader" Persona
    fingpt_prompt = f"""
    Role: You are a Technical Analysis Chart Expert.
    Context: The user asks "Should I sell {ticker}?".
    
    TECHNICAL DATA:
    - Price: ${current_price:.2f}
    - Trend (vs SMA50): {trend}
    - RSI (14): {rsi:.1f} (Over 70=Overbought, Under 30=Oversold)
    
    TASK:
    Give a 'SELL' or 'HOLD' recommendation based ONLY on the chart.
    - If RSI is < 30, warn about a "bounce" (Don't sell low).
    - If Trend is BEARISH, warn that support is broken.
    
    OUTPUT JSON ONLY:
    {{ "analysis": "Short technical commentary", "action": "SELL / HOLD / BUY_DIP" }}
    """

    llm = LLMClient() 
    messages = [
        {"role": "system", "content": "Output valid JSON only."},
        {"role": "user", "content": fingpt_prompt}
    ]
    
    try:
        response = llm.chat(messages)
        data = json.loads(response)
        action = data.get("action", "HOLD")
        
        icon = "🛑" if "SELL" in action else "✋" if "HOLD" in action else "🛒"
        
        return (f"{icon} **FinGPT Technicals ({action}):**\n"
                f"\"{data.get('analysis')}\"\n"
                f"*(RSI: {rsi:.1f} | Trend: {trend})*")
    except:
        return "FinGPT is confused by the chart."
    

def get_fingpt_trending_sectors():
    """
    Scans S&P 500 Sector ETFs to find the top momentum winner.
    Generates a hype-driven explanation.
    """
    # Map of major US Sector ETFs
    SECTOR_MAP = {
        "XLK": "Technology",
        "XLE": "Energy",
        "XLV": "Healthcare",
        "XLF": "Financials",
        "XLI": "Industrials",
        "XLY": "Consumer Discretionary",
        "XLP": "Consumer Staples",
        "XLB": "Materials",
        "XLU": "Utilities",
        "XLRE": "Real Estate",
        "XLC": "Communication Services"
    }
    
    tickers = list(SECTOR_MAP.keys())
    
    try:
        # 1. Fetch data for all sectors (1 month history)
        # We download in bulk for speed
        data = yf.download(tickers, period="1mo", progress=False)['Close']
        
        # 2. Calculate % Return (Last price vs Price 1 month ago)
        returns = {}
        for ticker in tickers:
            if ticker in data.columns:
                start_price = data[ticker].iloc[0]
                end_price = data[ticker].iloc[-1]
                pct_change = ((end_price - start_price) / start_price) * 100
                returns[ticker] = pct_change
        
        # 3. Find the Winner
        best_ticker = max(returns, key=returns.get)
        best_sector = SECTOR_MAP[best_ticker]
        best_return = returns[best_ticker]
        
    except Exception as e:
        return {"error": f"Market Scan Failed: {str(e)}"}

    # 4. Generate the "Hype" Message via LLM
    fingpt_prompt = f"""
    Role: You are a Momentum Trader. 
    Task: The hottest sector right now is **{best_sector}** ({best_ticker}), up {best_return:.2f}% this month.
    
    Output a short, high-energy comment about why money is flowing into {best_sector}. 
    Use phrases like "Sector Rotation", "Breakout", "Smart Money".
    
    OUTPUT JSON ONLY: {{ "comment": "Your text here" }}
    """

    llm = LLMClient()
    messages = [
        {"role": "system", "content": "Output valid JSON only."},
        {"role": "user", "content": fingpt_prompt}
    ]
    
    try:
        resp = llm.chat(messages)
        json_resp = json.loads(resp)
        comment = json_resp.get("comment", "Sector is moving fast!")
        
        return {
            "sector": best_sector,
            "ticker": best_ticker,
            "return": best_return,
            "message": f"🚀 **FinGPT Scanner:** The hot money is in **{best_sector}** ({best_ticker})!\n\"{comment}\"\n*(Performance: +{best_return:.2f}% in 30 days)*"
        }
    except:
        return {"error": "FinGPT could not generate hype."}