import sqlite3
import os


# Database path - Versione corretta per puntare a tools/database
# os.path.dirname(__file__) è la cartella 'scripts'
# Il primo os.path.dirname(...) risale alla cartella principale 'AI_Portfolio'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Ora costruiamo il percorso verso la cartella corretta
db_path = os.path.join(BASE_DIR, "tools", "database", "portfolio_manager.db")


# 1. Connect to database
conn = sqlite3.connect(db_path)
cur = conn.cursor()


# 2. Creation of table transactions
# This table stores ALL historical transactions (buy & sell); if quantity > 0  → buy; otherwise → sell

cur.execute("""
CREATE TABLE IF NOT EXISTS transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ticker TEXT NOT NULL,
    name TEXT NOT NULL,
    sector TEXT NOT NULL,
    quantity REAL NOT NULL,
    price REAL NOT NULL
)
""")


# 3. Initial naive portfolio
transactions = [
    ("2025-01-02", "AAPL", "Apple Inc.", "Technology", 10, 242.75),
    ("2025-01-02", "MSFT", "Microsoft Corp.", "Technology", 8, 415.51),
    ("2025-01-02", "GOOGL", "Alphabet Inc.", "Technology", 5, 188.69),
    ("2025-01-02", "AMZN", "Amazon.com Inc.", "Consumer", 4, 220.22),
    ("2025-01-02", "NVDA", "NVIDIA Corp.", "Technology", 3, 138.27),
    ("2025-01-02", "JPM", "JPMorgan Chase & Co.", "Finance", 12, 235.02),
    ("2025-01-02", "JNJ", "Johnson & Johnson", "Healthcare", 6, 139.74),
    ("2025-01-02", "XOM", "Exxon Mobil Corp.", "Energy", 10, 103.51),
    ("2025-01-02", "KO", "Coca-Cola Co.", "Consumer", 15, 60.07),
    ("2025-01-02", "PG", "Procter & Gamble", "Consumer", 7, 161.71),
]

sell_transactions = [
    ("2025-06-02", "AAPL", "Apple Inc.", "Technology", -3, 201.28),
    ("2025-06-02", "NVDA", "NVIDIA Corp.", "Technology", -1, 137.36),
    ("2025-06-02", "KO", "Coca-Cola Co.", "Consumer", -5, 70.45),
]


# 4. Insert data
cur.executemany(
    "INSERT INTO transactions (date, ticker, name, sector, quantity, price) VALUES (?, ?, ?, ?, ?, ?)",
    transactions
)

cur.executemany(
    "INSERT INTO transactions (date, ticker, name, sector, quantity, price) VALUES (?, ?, ?, ?, ?, ?)",
    sell_transactions
)

# 5. Creation of the view portfolio
#This VIEW represents the CURRENT STATE of the portfolio:
# - total_quantity: net position
# - avg_price: weighted average buy price
# - invested_value: capital invested in the open position
#
# Titles with total_quantity = 0 are excluded

cur.execute("""
CREATE VIEW IF NOT EXISTS current_portfolio AS
SELECT
    ticker,
    MAX(name) AS name,
    sector,
    SUM(quantity) AS total_quantity,
    ROUND(
        SUM(CASE WHEN quantity > 0 THEN quantity * price ELSE 0 END)
        / NULLIF(SUM(CASE WHEN quantity > 0 THEN quantity ELSE 0 END), 0),
        2
    ) AS avg_price,
    ROUND(
        SUM(quantity) *
        (
            SUM(CASE WHEN quantity > 0 THEN quantity * price ELSE 0 END)
            / NULLIF(SUM(CASE WHEN quantity > 0 THEN quantity ELSE 0 END), 0)
        ),
        2
    ) AS invested_value
FROM transactions
GROUP BY ticker
HAVING total_quantity > 0
""")

conn.commit()

# 5. Check inserted data
print("Transactions in database:\n")
cur.execute("SELECT * FROM transactions")
for row in cur.fetchall():
    print(row)

# 5.1 Test: print the current portfolio
cur.execute("SELECT * FROM current_portfolio")
print("Current Portfolio:")
for row in cur.fetchall():
    print(row)

conn.commit()
conn.close()

print(f"\nDatabase created successfully at:\n{db_path}")