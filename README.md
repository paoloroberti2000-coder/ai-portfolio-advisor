# **AI Portfolio Advisor Project**
### **Course**: Advanced Programming in Python
### **Authors**: *Matteo Beschi, Cristina Saramondi, Paolo Roberti*


AI Portfolio Advisor is a **Python-based financial analysis system** that combines portfolio analytics, visualization, optimization, and an AI agent to support portfolio performance evaluation and decision-making.

### Key Capabilities

**Dynamic SQL database**
- Unique table showing user's transactions
- AI agent can insert and delete transactions via natural language prompts

**API tools**
- Fetch historical and latest close prices from yfinance
 
 **Portfolio Analysis**
- Portfolio returns (ROI)
- Best and worst performing assets
- Sector diversification
- Mean-variance (Markowitz) optimization
- Sentiment analysis using NewsApi

**Visualization Tools**
- Portfolio value over time
- Invested vs current value per asset
- Sector allocation and sector performance
- Normalized price comparison between assets
- Normalized comparison portfolio vs benchmark (^GSPC)
- Correlation heatmaps
- Markowitz allocation comparison

**AI Agent**
- Natural language interface (terminal-based)
- Automatic tool selection
- AI-generated insights and recommendations using FinGpt

**Reporting**
- PDF portfolio overview report
- PDF risk & optimization report
- Both include relevant charts, tables, and AI insights

### Project Structure
agent/ # AI agent, LLM client, prompts

tools/

├─ analysis/ # Financial analysis tools

├─ api/ # yfinance connection

├─ visualization/ # Chart generation

├─ reporting/ # PDF reporting

└─ database/ # Portfolio storage (SQLite)

plots/ # Generated charts

reports/ # Generated PDF reports

main.py # Entry point

### Example Prompts

```text
What is the best performing asset?
Show asset performance invested vs current value
Compare AAPL vs GOOGL normalized
Generate a pdf report of my portfolio overall performance
Generate a pdf report of Risk & Optimization of my portfolio
```
### Notes
- The AI agent uses Groq LLM (llama-3.1-8b) for reasoning and commentary.
- Financial insights are for educational purposes only.
