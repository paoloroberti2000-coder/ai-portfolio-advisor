from agent.fingpt_bridge import get_fingpt_opinion
from tools.analysis.risk_manager import analyze_portfolio_risk
from agent.fingpt_bridge import get_fingpt_technical_view
from tools.analysis.risk_manager import analyze_exit_strategy
from agent.fingpt_bridge import get_fingpt_trending_sectors
from tools.analysis.risk_manager import analyze_sector_fit


def tool_compare_fingpt_vs_portfolio(ticker):
    """
    Orchestrates the debate between FinGPT and the Agent.
    """
    ticker = ticker.upper()
    
    # 1. Fetch Opinions
    fingpt_voice = get_fingpt_opinion(ticker)
    agent_voice = analyze_portfolio_risk(ticker)
    
    # 2. Format as a Report
    result_message = (
        f"\n{'='*40}\n"
        f"⚔️  **MARKET vs. PORTFOLIO DEBATE: {ticker}** ⚔️\n"
        f"{'='*40}\n\n"
        f"{fingpt_voice}\n\n"
        f"{'-'*40}\n\n"
        f"{agent_voice}\n"
        f"\n{'='*40}\n"
    )
    
    return {"message": result_message}

def tool_should_i_sell(ticker):
    """
    Scenario 1: The Panic Button.
    Compares technical breakdown (FinGPT) vs Portfolio impact (Agent).
    """
    ticker = ticker.upper()
    
    # 1. Get Technical View
    technical_view = get_fingpt_technical_view(ticker)
    
    # 2. Get P&L View
    pnl_view = analyze_exit_strategy(ticker)
    
    # 3. Combine
    result_message = (
        f"\n{'='*40}\n"
        f"🚨 **EXIT STRATEGY ANALYSIS: {ticker}** 🚨\n"
        f"{'='*40}\n\n"
        f"{technical_view}\n\n"
        f"{'-'*40}\n\n"
        f"{pnl_view}\n"
        f"\n{'='*40}\n"
    )
    
    return {"message": result_message}

def tool_scan_market_trends():
    """
    Scenario 2: The Trend Hunter.
    Finds the top performing sector and checks if it fits the user's portfolio.
    """
    # 1. Ask FinGPT for the hot sector
    trend_data = get_fingpt_trending_sectors()
    
    if "error" in trend_data:
        return {"message": trend_data["error"]}
    
    hot_sector = trend_data['sector']
    fingpt_msg = trend_data['message']
    
    # 2. Ask Agent if it fits
    agent_msg = analyze_sector_fit(hot_sector)
    
    # 3. Combine
    result_message = (
        f"\n{'='*40}\n"
        f"🔭 **MARKET TREND DISCOVERY** 🔭\n"
        f"{'='*40}\n\n"
        f"{fingpt_msg}\n\n"
        f"{'-'*40}\n\n"
        f"{agent_msg}\n"
        f"\n{'='*40}\n"
    )
    
    return {"message": result_message}