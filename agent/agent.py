import json
from .prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
import webbrowser
import os

# --- 1. REAL TOOLS IMPORT ---

# Database tools
from tools.database.db_tools import (
    insert_transaction, 
    get_sector_allocation, 
    get_current_portfolio,
    get_transactions_by_ticker,
    get_transactions_by_date,
    get_historical_portfolio,
    delete_transaction,
    update_transaction,
    get_portfolio_summary,
    get_best_avg_price
)

# API tools
from tools.api.api_tools import (
    get_market_transaction_data, 
    buy_stock_flow
)

# Analysis tools
from tools.analysis.analysis_tools import (
    tool_compute_returns,
    tool_optimize_markowitz_target,
    tool_sentiment_analysis,
    get_best_returns_data,
    get_best_returns_summary,
    tool_sector_diversification_comparison
)

# Visualization tools
from tools.visualization.visualization_tools import (
    plot_portfolio_composition,
    plot_sector_allocation,
    plot_portfolio_value_over_time,
    plot_portfolio_performance,
    plot_sector_performance,
    plot_portfolio_vs_benchmark,
    plot_normalized_comparison,
    plot_stock_price,
    plot_asset_correlation_heatmap,
    plot_allocation_vs_markowitz,
    plot_sentiment_analysis
)

# Consultant tool
from tools.analysis.consultant import (
    tool_compare_fingpt_vs_portfolio, 
    tool_should_i_sell,
    tool_scan_market_trends
)

# Reporting tool
from tools.reporting.reporting_tools import generate_portfolio_report, generate_risk_optimization_report


def show_asset_performance_chart(save_path=None, **kwargs):
    # Ignore unsupported kwargs safely
    result = plot_portfolio_performance(save_path=save_path)

    # Ensure we always return something usable
    if result is None:
        return {"status": "warning", "image_path": None, "message": "Chart function returned None unexpectedly."}

    if isinstance(result, dict):
        return result

    return {"status": "ok", "result": result}


# --- 2. TOOL REGISTRY ---
# Mapping string names from the LLM to actual Python functions
TOOLS = {
    # DATABASE & API
    "get_current_portfolio": get_current_portfolio,
    "buy_stock_flow": buy_stock_flow,
    "get_transactions_by_ticker": get_transactions_by_ticker,
    "get_transactions_by_date": get_transactions_by_date,
    "get_historical_portfolio": get_historical_portfolio,
    "delete_transaction": delete_transaction,
    "update_transaction": update_transaction,
    "get_portfolio_summary": get_portfolio_summary,
    "get_best_avg_price": get_best_avg_price,
    "get_market_transaction_data": get_market_transaction_data,
    "insert_transaction": insert_transaction,
    
    # FINANCIAL ANALYSIS
    "compute_roi": tool_compute_returns,
    "get_best_returns": get_best_returns_data,
    "get_best_returns_summary": get_best_returns_summary,
    "compare_sector_drift": tool_sector_diversification_comparison,
    "optimize_portfolio": tool_optimize_markowitz_target,
    "analyze_sentiment": tool_sentiment_analysis,
    
    # VISUALIZATIONS
    "show_composition_chart": plot_portfolio_composition,
    "show_sector_chart": plot_sector_allocation,
    "show_performance_chart": plot_portfolio_value_over_time,
    "show_asset_performance_chart": show_asset_performance_chart,
    "show_sector_performance_chart": plot_sector_performance,
    "show_benchmark_chart": plot_portfolio_vs_benchmark,
    "show_normalized_comparison_chart": plot_normalized_comparison,
    "show_price_chart": plot_stock_price,
    "show_correlation_chart": plot_asset_correlation_heatmap,
    "show_advice_chart": plot_allocation_vs_markowitz,
    "show_sentiment_chart": plot_sentiment_analysis,

    # CONSULTANT
    "consult_fingpt": tool_compare_fingpt_vs_portfolio,
    "analyze_sell_decision": tool_should_i_sell,
    "scan_market_trends": tool_scan_market_trends,
    
    # REPORTS
    "generate_portfolio_report": generate_portfolio_report,
    "generate_risk_optimization_report": generate_risk_optimization_report
}


class Agent:
    def __init__(self, llm_client):
        self.llm = llm_client
        self.system_prompt = SYSTEM_PROMPT

    def run(self, user_input):
        prompt = USER_PROMPT_TEMPLATE.format(user_input=user_input)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]

        # LLM Call
        response = self.llm.chat(messages)

        # JSON Parsing
        try:
            if isinstance(response, dict):
                decision = response
            else:
                decision = json.loads(response)
        except json.JSONDecodeError:
            return "Error: LLM did not return valid JSON."

        tool_name = decision.get("tool")
        args = decision.get("args", {})

        # --- 3. TOOL EXECUTION ---
        if tool_name and tool_name in TOOLS:
            try:
                if tool_name.startswith("show_") and "save_path" not in args:
                    args["save_path"] = f"plots/{tool_name}.png"

                result = TOOLS[tool_name](**args)
                
                if isinstance(result, dict) and 'image_path' in result and result['image_path']:
                    
                    full_path = os.path.abspath(result['image_path'])
                    webbrowser.open(f"file://{full_path}")
                # ---------------------------------

                return {
                    "tool": tool_name,
                    "result": result,
                    "args": args
                }
            except Exception as e:
                return f"Error running tool '{tool_name}': {e}"
            
        
