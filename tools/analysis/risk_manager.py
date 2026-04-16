import pandas as pd
from tools.database.db_tools import get_sector_allocation, get_current_portfolio
from tools.api.api_tools import get_market_transaction_data

def analyze_portfolio_risk(ticker):
    """
    Checks the user's portfolio and returns a structured risk report.
    """
    # 1. Get Data
    try:
        stock_info = get_market_transaction_data(ticker, 0) 
        new_sector = stock_info['sector']
        name = stock_info['name']
    except:
        return "Risk Analysis Failed: Could not fetch stock data."

    # 2. Check Holdings
    portfolio = get_current_portfolio()
    current_holding_qty = 0
    if portfolio['status'] == 'ok':
        for asset in portfolio['data']:
            if asset['ticker'] == ticker:
                current_holding_qty = asset['total_quantity']

    # 3. Check Sector
    allocation = get_sector_allocation()
    sector_exposure_pct = 0.0
    total_invested = 0.0
    
    if allocation['status'] == 'ok' and allocation['data']:
        df = pd.DataFrame(allocation['data'])
        total_invested = df['total_invested'].sum()
        sector_row = df[df['sector'] == new_sector]
        if not sector_row.empty:
            val = sector_row.iloc[0]['total_invested']
            sector_exposure_pct = (val / total_invested) * 100

    # 4. IMPROVED OUTPUT FORMATTING
    warnings = []
    THRESHOLD = 30.0
    
    risk_level = "LOW"
    
    if current_holding_qty > 0:
        warnings.append(f"• **Concentration:** You already own {current_holding_qty} shares.")
        risk_level = "MEDIUM"
        
    if sector_exposure_pct > THRESHOLD:
        warnings.append(f"• **Sector Risk:** {new_sector} is {sector_exposure_pct:.1f}% of your portfolio (Target: <{THRESHOLD}%).")
        risk_level = "HIGH"

    # Generate the formatted text
    header = f"🛡️ **Agent's Reality Check** (Risk Level: {risk_level})"
    
    if not warnings:
        return (f"{header}\n"
                f"✅ **Safe to Buy.**\n"
                f"• Sector: {new_sector} (Currently {sector_exposure_pct:.1f}%)\n"
                f"• Diversification looks good.")
    else:
        warning_text = "\n".join(warnings)
        return (f"{header}\n"
                f"⚠️ **Caution Suggested:**\n"
                f"{warning_text}\n"
                f"👉 *Recommendation:* Consider buying a smaller amount or diversifying.")


def analyze_exit_strategy(ticker):
    """
    Analyzes the user's specific position (P&L, Weight) to give selling advice.
    """
    portfolio = get_current_portfolio()
    
    # 1. Find the asset
    asset = None
    if portfolio['status'] == 'ok':
        for item in portfolio['data']:
            if item['ticker'] == ticker:
                asset = item
                break
    
    if not asset:
        return "⚠️ **Agent Error:** You don't own this stock, so you can't sell it!"

    # 2. Calculate Metrics
    qty = asset['total_quantity']
    avg_price = asset['avg_price']
    
    # Get real-time price for accurate P&L
    try:
        current_data = get_market_transaction_data(ticker, 0)
        curr_price = current_data['price']
    except:
        curr_price = avg_price # Fallback

    pnl_pct = ((curr_price - avg_price) / avg_price) * 100
    pnl_usd = (curr_price - avg_price) * qty

    # 3. Formulate "Accountant" Advice
    advice_header = f"💼 **Agent's P&L Check**"
    
    status = "PROFIT" if pnl_pct > 0 else "LOSS"
    
    if pnl_pct > 20:
        recommendation = "You have a nice profit. Locking in some gains (taking partial profit) is never a bad idea."
    elif pnl_pct < -10:
        recommendation = "You are down significantly. Review your long-term thesis. Selling now locks in a realized loss (tax deductible)."
    elif abs(pnl_pct) < 5:
        recommendation = "You are basically flat (breakeven). Transaction costs might outweigh the move."
    else:
        recommendation = "Review your conviction."

    return (f"{advice_header}\n"
            f"• **Your Entry:** ${avg_price:.2f} (Current: ${curr_price:.2f})\n"
            f"• **Unrealized P&L:** {status} of {pnl_pct:.1f}% (${pnl_usd:.2f})\n"
            f"👉 *Prudent View:* {recommendation}")


def analyze_sector_fit(sector_name):
    """
    Checks if the user has room for a specific sector in their portfolio.
    """
    # 1. Get current allocation
    allocation = get_sector_allocation()
    
    user_exposure = 0.0
    if allocation['status'] == 'ok':
        df = pd.DataFrame(allocation['data'])
        if not df.empty:
            total_val = df['total_invested'].sum()
            # Loose string matching (e.g., "Technology" vs "Information Technology")
            for _, row in df.iterrows():
                if sector_name.lower() in row['sector'].lower() or row['sector'].lower() in sector_name.lower():
                    user_exposure = (row['total_invested'] / total_val) * 100
                    break
    
    # 2. Formulate Advice
    header = f"🧩 **Agent's Portfolio Fit**"
    
    if user_exposure == 0:
        recommendation = (f"You have **0% exposure** to {sector_name}. "
                          f"This is a perfect opportunity to diversify your portfolio.")
        status = "✅ **High Priority**"
    elif user_exposure < 15:
        recommendation = (f"You have **{user_exposure:.1f}%** in this sector. "
                          f"You have room to add more to ride the trend.")
        status = "⚖️ **Balanced**"
    else:
        recommendation = (f"You are already heavy here (**{user_exposure:.1f}%**). "
                          f"Chasing the trend might overexpose you to sector rotation risk.")
        status = "⚠️ **Low Priority**"

    return (f"{header}\n"
            f"• **Current Exposure:** {user_exposure:.1f}%\n"
            f"• **Fit Verdict:** {status}\n"
            f"👉 *Advice:* {recommendation}")