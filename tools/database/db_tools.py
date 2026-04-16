"""
Database tools for portfolio management
Functions to manage transactions and portfolio data
"""

import sqlite3
import os
import logging

# Logging 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_connection():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(base_dir, "portfolio_manager.db")
        
        #DEBUG_DB = False

        #if DEBUG_DB:
            #print(f"DEBUG: Sto cercando il DB in: {os.path.abspath(db_path)}")

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        logger.error(f"DB connection error: {e}")
        return None
    
# --- Internal Helper ---

def _dict_from_cursor(cur):
    """Converts cursor.fetchall() into list of dicts."""
    return [dict(row) for row in cur.fetchall()]


def _dict_from_row(row):
    """Converts single row to dict or returns None."""
    return dict(row) if row else None


# --- Main Functions ---

def insert_transaction(transaction_data):
    """
    Insert a new transaction.
    Required fields: date, ticker, quantity, price
    """
    required_fields = ["date", "ticker", "quantity", "price"]
    for field in required_fields:
        if field not in transaction_data:
            return {"status": "error", "message": f"Missing field {field}"}

    conn = get_connection()
    if not conn:
        return {"status": "error", "message": "DB connection failed"}

    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO transactions (date, ticker, name, sector, quantity, price)
            VALUES (:date, :ticker, :name, :sector, :quantity, :price)
        """, transaction_data)
        conn.commit()
        logger.info(f"Inserted transaction for {transaction_data['ticker']}")
        return {"status": "ok", "data": {"transaction_id": cur.lastrowid}}
    except sqlite3.Error as e:
        return {"status": "error", "message": str(e)}
    finally:
        conn.close()


def get_current_portfolio():
    conn = get_connection()
    if not conn:
        return {"status": "error", "data": None}

    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM current_portfolio")
        data = _dict_from_cursor(cur)
        return {"status": "ok", "data": data} if data else {"status": "empty", "data": []}
    except sqlite3.Error as e:
        return {"status": "error", "data": None, "message": str(e)}
    finally:
        conn.close()


def get_historical_portfolio(date):
    conn = get_connection()
    if not conn:
        return {"status": "error", "data": None}

    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT 
                ticker,
                MAX(name) AS name,
                SUM(quantity) AS total_quantity,
                ROUND(
                    SUM(CASE WHEN quantity > 0 THEN quantity * price ELSE 0 END)
                    / NULLIF(SUM(CASE WHEN quantity > 0 THEN quantity ELSE 0 END), 0), 2
                ) AS avg_price,
                ROUND(
                    SUM(quantity) *
                    (SUM(CASE WHEN quantity > 0 THEN quantity * price ELSE 0 END)
                    / NULLIF(SUM(CASE WHEN quantity > 0 THEN quantity ELSE 0 END), 0)), 2
                ) AS total_invested
            FROM transactions
            WHERE date <= ?
            GROUP BY ticker
            HAVING total_quantity > 0
        """, (date,))
        data = _dict_from_cursor(cur)
        return {"status": "ok", "data": data} if data else {"status": "empty", "data": []}
    except sqlite3.Error as e:
        return {"status": "error", "data": None, "message": str(e)}
    finally:
        conn.close()


def get_best_avg_price():
    conn = get_connection()
    if not conn:
        return {"status": "error", "data": None}

    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT ticker, name, sector, avg_price
            FROM current_portfolio
            ORDER BY avg_price DESC
            LIMIT 1
        """)
        row = cur.fetchone()
        data = _dict_from_row(row)
        return {"status": "ok", "data": data} if data else {"status": "empty", "data": None}
    except sqlite3.Error as e:
        return {"status": "error", "data": None, "message": str(e)}
    finally:
        conn.close()


def get_transactions_by_ticker(ticker):
    conn = get_connection()
    if not conn:
        return {"status": "error", "data": []}

    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM transactions WHERE ticker = ?", (ticker,))
        data = _dict_from_cursor(cur)
        return {"status": "ok", "data": data} if data else {"status": "empty", "data": []}
    except sqlite3.Error as e:
        return {"status": "error", "data": [], "message": str(e)}
    finally:
        conn.close()


def get_transactions_by_date(start_date, end_date, limit=None):
    conn = get_connection()
    if not conn:
        return {"status": "error", "data": []}

    try:
        cur = conn.cursor()
        sql = "SELECT * FROM transactions WHERE date BETWEEN ? AND ?"
        params = [start_date, end_date]
        if limit:
            sql += " LIMIT ?"
            params.append(limit)
        cur.execute(sql, tuple(params))
        data = _dict_from_cursor(cur)
        return {"status": "ok", "data": data} if data else {"status": "empty", "data": []}
    except sqlite3.Error as e:
        return {"status": "error", "data": [], "message": str(e)}
    finally:
        conn.close()


def delete_transaction(transaction_id=None, confirm=False):
    conn = get_connection()
    if not conn:
        return {"status": "error", "data": None}

    try:
        cur = conn.cursor()
        # Se non specificato, prendi l'ultima
        if transaction_id is None:
            cur.execute("SELECT * FROM transactions ORDER BY id DESC LIMIT 1")
        else:
            cur.execute("SELECT * FROM transactions WHERE id = ?", (transaction_id,))
        row = cur.fetchone()
        if not row:
            return {"status": "not_found", "data": None}

        transaction = dict(row)
        transaction_id = transaction["id"]

        if not confirm:
            return {"status": "confirmation_required", "data": transaction}

        cur.execute("DELETE FROM transactions WHERE id = ?", (transaction_id,))
        conn.commit()
        return {"status": "deleted", "data": {"transaction_id": transaction_id}}
    except sqlite3.Error as e:
        return {"status": "error", "data": None, "message": str(e)}
    finally:
        conn.close()

def update_transaction(transaction_id=None, **kwargs): 
    
    # If AI send a 'id', we take it manually
    if transaction_id is None:
        transaction_id = kwargs.pop('transaction_id', kwargs.pop('id', None))

    if transaction_id is None:
        return {"status": "error", "message": "Missing required argument: transaction_id"}

    required_fields = ["date", "ticker", "quantity", "price"]
    for field in required_fields:
        if field in kwargs and kwargs[field] is None:
            return {"status": "error", "message": f"Field {field} cannot be None"}

    conn = get_connection()
    if not conn: return {"status": "error", "data": None}

    try:
        cur = conn.cursor()
        fields = []
        values = []
        for key, value in kwargs.items():
            fields.append(f"{key} = ?")
            values.append(value)
        
        if not fields:
            return {"status": "error", "message": "No fields to update"}

        values.append(transaction_id)
        sql = f"UPDATE transactions SET {', '.join(fields)} WHERE id = ?"
        cur.execute(sql, values)
        conn.commit()
        return {"status": "ok", "message": f"Transaction {transaction_id} updated successfully", "data": {"transaction_id": transaction_id}}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        conn.close()

def get_sector_allocation():
    """
    Returns the total invested value per sector from the current portfolio.
    No percentages are calculated here; percentages can be computed in analysis_tools.
    """
    conn = get_connection()
    if not conn:
        return {"status": "error", "data": []}

    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT sector, SUM(invested_value) AS total_invested
            FROM current_portfolio
            GROUP BY sector
        """)
        data = _dict_from_cursor(cur)
        return {"status": "ok", "data": data} if data else {"status": "empty", "data": []}
    except sqlite3.Error as e:
        return {"status": "error", "data": [], "message": str(e)}
    finally:
        conn.close()


def get_portfolio_summary():
    conn = get_connection()
    if not conn:
        return {"status": "error", "data": None}

    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT 
                SUM(total_quantity) AS total_quantity,
                ROUND(SUM(invested_value),2) AS total_invested,
                ROUND(SUM(total_quantity * avg_price) / NULLIF(SUM(total_quantity),0),2) AS avg_price
            FROM current_portfolio
        """)
        row = cur.fetchone()
        data = dict(row) if row else None
        return {"status": "ok", "data": data} if data else {"status": "empty", "data": None}
    except sqlite3.Error as e:
        return {"status": "error", "data": None, "message": str(e)}
    finally:
        conn.close()
        