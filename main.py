# main.py
from agent.agent import Agent
from agent.llm_client import LLMClient
import os
import pandas as pd
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


def main():
    # Initialize the "brain" (Fake LLM) and the "orchestrator" (Agent)
    llm_client = LLMClient()
    portfolio_agent = Agent(llm_client)

    # Ensure plot directory exists
    if not os.path.exists('plots'):
        os.makedirs('plots')

    print("\n" + "="*50)
    print("=== FINANCIAL AI PORTFOLIO ADVISOR ===")
    print("="*50)
    print("- Database: Portfolio (current/historical), Transactions (ticker/dates), Summary")
    print("- Analysis: performance, sector diversification, optimize, news")
    print("- Charts: pie chart, benchmark chart, correlation chart, advice chart")
    print("- Reporting: PDF portfolio overview, PDF risk & optimization")
    print("- FinGPT: consultant & risk manager (optional insights)")
    print("-" * 50)
    print("Type 'exit' to close.\n")

    while True:
        user_input = input("User: ")
        
        if user_input.lower() in ["exit", "quit"]:
            print("Closing assistant. Goodbye!")
            break

        if not user_input.strip():
            continue

        try:
            # The Agent analyzes the input and runs the tools
            agent_output = portfolio_agent.run(user_input)
            
            # If the agent gives back a string (error or text), print it
            if isinstance(agent_output, str):
                print(f"\nAgent: {agent_output}")
                continue

            # extract data from the package
            tool_name = agent_output.get("tool")
            response = agent_output.get("result")
            decision = agent_output # This gets the 'args' in the CASE C

            
            # --- 1. Handle Dictionary Responses (DB Data, Tools, Charts) ---
            if isinstance(response, dict):
                # CASE A: Standard Tool Message
                if 'message' in response:
                    print(f"\nAgent: {response['message']}")
                
                # CASE B: Chart/Visualization Output
                if 'image_path' in response and response['image_path']:
                    print(f"📊 Success: Plot saved to: {response['image_path']}")
                
                # CASE C: Portfolio List (Current, Historical, or Transaction Log)
                if 'data' in response and isinstance(response['data'], list):
                    data = response['data']
                    if not data:
                        print("\nAgent: The list is empty.")
                    else:
                        # 1. Case transaction history
                        if 'date' in data[0]:
                            print(f"\nAgent: I found the following transaction history:")
                            
                            # to cut the list if AI forgets the limit
                            req_limit = decision.get('args', {}).get('limit')
                            if req_limit and len(data) > int(req_limit):
                                data = data[-int(req_limit):]

                            for item in data:
                                t_id = item.get('id', '?')  # added for deleting transactions
                                date = item.get('date', 'N/A')
                                ticker = item.get('ticker', 'N/A')
                                qty = item.get('quantity', 0)
                                price = item.get('price', 0)
                                print(f" • [#{t_id:<3}] {date:10} | {ticker:5} | Qty: {qty:5} | Price: ${price:.2f}")
                        
                        # 2. Case historical portfolio
                        elif tool_name == "get_historical_portfolio":
                            hist_date = decision.get('args', {}).get('date', 'specified date')
                            print(f"\nAgent: Historical Portfolio Snapshot as of {hist_date}:")
                            for item in data:
                                ticker = item.get('ticker', 'N/A')
                                name = item.get('name', 'N/A')
                                qty = item.get('total_quantity', item.get('quantity', 0))
                                price = item.get('avg_price', item.get('price', 0))
                                print(f" • {ticker:5} | {name:20} | Qty: {qty:4} | Avg Price: ${price:.2f}")
                        
                        # 3. Case current portfolio
                        else:
                            print(f"\nAgent: I found the following current holdings:")
                            for item in data:
                                ticker = item.get('ticker', 'N/A')
                                name = item.get('name', 'N/A')
                                qty = item.get('total_quantity', item.get('quantity', 0))
                                price = item.get('avg_price', item.get('price', 0))
                                print(f" • {ticker:5} | {name:20} | Qty: {qty:4} | Avg Price: ${price:.2f}")
                
                
                # CASE D: Optimization or Sentiment Results (Nested Dict)
                elif 'optimized_weights' in response:
                    print("\nAgent: Markowitz Optimization Results:")
                    print(f" - Est. Volatility: {response.get('estimated_volatility', 0)*100:.2f}%")
                    for t, w in response['optimized_weights'].items():
                        print(f" • {t}: {w*100:.2f}%")
                
                elif 'sentiment_label' in response:
                    print(f"\nAgent: Sentiment for {response.get('ticker')}: {response.get('sentiment_label')}")
                    print(f" - Average Score: {response.get('average_score')} (based on {response.get('article_count')} articles)")

                # Default fallback for dict
                elif not any(k in response for k in ['data', 'image_path', 'message']):
                    print(f"\nAgent Data: {response}")
                    
                # CASE E: Portfolio Summary 
                elif tool_name == "get_portfolio_summary":
                    summary_data = response.get('data', {})
                    if summary_data:
                        print(f"\nAgent: [PORTFOLIO SUMMARY]")
                        print(f" • Total Invested:  ${summary_data.get('total_invested', 0):,.2f}")
                        print(f" • Total Quantity:  {summary_data.get('total_quantity', 0)} units")
                        print(f" • Weighted Avg:    ${summary_data.get('avg_price', 0):.2f}")
                    else:
                        print("\nAgent: Summary data is currently unavailable.")
                
                # CASE F: Best Average Price (Single Object)
                elif tool_name == "get_best_avg_price":
                    best_data = response.get('data', {})
                    if best_data:
                        ticker = best_data.get('ticker', 'N/A')
                        name = best_data.get('name', 'N/A')
                        price = best_data.get('avg_price', 0)
                        print(f"\nAgent: The stock with the highest average purchase price is:")
                        print(f" • {name} ({ticker}) at ${price:.2f}")
                    else:
                        print("\nAgent: No data available for this calculation.")

            # --- 2. Handle DataFrame Responses (Comparison tables, Drift, etc.) ---
            elif isinstance(response, pd.DataFrame):
                print(f"\nAgent: Analysis Table:")
                if response.empty:
                    print("No data available for this analysis.")
                else:
                    print(response.to_string(index=False))

            # --- 3. Handle Simple Text Responses (Thoughts/Greetings) ---
            else:
                if response is None:
                    print("\nAgent: I processed the request but have no specific data to show.")
                elif isinstance(response, str):
                    print(f"\nAgent: {response}")
                else:
                    print(f"\nAgent (Note): {str(response)}")

        except Exception as e:
            print(f"\n[SYSTEM ERROR]: {e}")
        
    
        print("-" * 50)

if __name__ == "__main__":
    main()