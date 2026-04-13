import pandas as pd
import numpy as np
import os
import json
import time
import dotenv
import ast
from sqlalchemy.sql import text
from datetime import datetime, timedelta
from typing import Dict, List, Union
from sqlalchemy import create_engine, Engine
from openai import OpenAI

# Load environment variables
dotenv.load_dotenv()

# Create an SQLite database
db_engine = create_engine("sqlite:///munder_difflin.db")

# List containing the different kinds of papers 
paper_supplies = [
    # Paper Types (priced per sheet unless specified)
    {"item_name": "A4 paper",                         "category": "paper",        "unit_price": 0.05},
    {"item_name": "Letter-sized paper",              "category": "paper",        "unit_price": 0.06},
    {"item_name": "Cardstock",                        "category": "paper",        "unit_price": 0.15},
    {"item_name": "Colored paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Glossy paper",                     "category": "paper",        "unit_price": 0.20},
    {"item_name": "Matte paper",                      "category": "paper",        "unit_price": 0.18},
    {"item_name": "Recycled paper",                   "category": "paper",        "unit_price": 0.08},
    {"item_name": "Eco-friendly paper",               "category": "paper",        "unit_price": 0.12},
    {"item_name": "Poster paper",                     "category": "paper",        "unit_price": 0.25},
    {"item_name": "Banner paper",                     "category": "paper",        "unit_price": 0.30},
    {"item_name": "Kraft paper",                      "category": "paper",        "unit_price": 0.10},
    {"item_name": "Construction paper",               "category": "paper",        "unit_price": 0.07},
    {"item_name": "Wrapping paper",                   "category": "paper",        "unit_price": 0.15},
    {"item_name": "Glitter paper",                    "category": "paper",        "unit_price": 0.22},
    {"item_name": "Decorative paper",                 "category": "paper",        "unit_price": 0.18},
    {"item_name": "Letterhead paper",                 "category": "paper",        "unit_price": 0.12},
    {"item_name": "Legal-size paper",                 "category": "paper",        "unit_price": 0.08},
    {"item_name": "Crepe paper",                      "category": "paper",        "unit_price": 0.05},
    {"item_name": "Photo paper",                      "category": "paper",        "unit_price": 0.25},
    {"item_name": "Uncoated paper",                   "category": "paper",        "unit_price": 0.06},
    {"item_name": "Butcher paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Heavyweight paper",                "category": "paper",        "unit_price": 0.20},
    {"item_name": "Standard copy paper",              "category": "paper",        "unit_price": 0.04},
    {"item_name": "Bright-colored paper",             "category": "paper",        "unit_price": 0.12},
    {"item_name": "Patterned paper",                  "category": "paper",        "unit_price": 0.15},

    # Product Types (priced per unit)
    {"item_name": "Paper plates",                     "category": "product",      "unit_price": 0.10},
    {"item_name": "Paper cups",                       "category": "product",      "unit_price": 0.08},
    {"item_name": "Paper napkins",                    "category": "product",      "unit_price": 0.02},
    {"item_name": "Disposable cups",                  "category": "product",      "unit_price": 0.10},
    {"item_name": "Table covers",                     "category": "product",      "unit_price": 1.50},
    {"item_name": "Envelopes",                        "category": "product",      "unit_price": 0.05},
    {"item_name": "Sticky notes",                     "category": "product",      "unit_price": 0.03},
    {"item_name": "Notepads",                         "category": "product",      "unit_price": 2.00},
    {"item_name": "Invitation cards",                 "category": "product",      "unit_price": 0.50},
    {"item_name": "Flyers",                           "category": "product",      "unit_price": 0.15},
    {"item_name": "Party streamers",                  "category": "product",      "unit_price": 0.05},
    {"item_name": "Decorative adhesive tape (washi tape)", "category": "product", "unit_price": 0.20},
    {"item_name": "Paper party bags",                 "category": "product",      "unit_price": 0.25},
    {"item_name": "Name tags with lanyards",          "category": "product",      "unit_price": 0.75},
    {"item_name": "Presentation folders",             "category": "product",      "unit_price": 0.50},

    # Large-format items (priced per unit)
    {"item_name": "Large poster paper (24x36 inches)", "category": "large_format", "unit_price": 1.00},
    {"item_name": "Rolls of banner paper (36-inch width)", "category": "large_format", "unit_price": 2.50},

    # Specialty papers
    {"item_name": "100 lb cover stock",               "category": "specialty",    "unit_price": 0.50},
    {"item_name": "80 lb text paper",                 "category": "specialty",    "unit_price": 0.40},
    {"item_name": "250 gsm cardstock",                "category": "specialty",    "unit_price": 0.30},
    {"item_name": "220 gsm poster paper",             "category": "specialty",    "unit_price": 0.35},
]

# Build a lookup dictionary for item prices
ITEM_PRICE_LOOKUP = {item["item_name"]: item["unit_price"] for item in paper_supplies}

# ============================================================
# UTILITY / DATABASE FUNCTIONS
# ============================================================

def generate_sample_inventory(paper_supplies: list, coverage: float = 0.4, seed: int = 137) -> pd.DataFrame:
    """Generate inventory for a specified fraction of items."""
    np.random.seed(seed)
    num_items = int(len(paper_supplies) * coverage)
    selected_indices = np.random.choice(range(len(paper_supplies)), size=num_items, replace=False)
    selected_items = [paper_supplies[i] for i in selected_indices]
    inventory = []
    for item in selected_items:
        inventory.append({
            "item_name": item["item_name"],
            "category": item["category"],
            "unit_price": item["unit_price"],
            "current_stock": np.random.randint(200, 800),
            "min_stock_level": np.random.randint(50, 150)
        })
    return pd.DataFrame(inventory)

def init_database(engine: Engine = None, seed: int = 137) -> Engine:
    """Set up the Munder Difflin database with all required tables and initial records."""
    if engine is None:
        engine = db_engine
    try:
        transactions_schema = pd.DataFrame({
            "id": [], "item_name": [], "transaction_type": [],
            "units": [], "price": [], "transaction_date": [],
        })
        transactions_schema.to_sql("transactions", engine, if_exists="replace", index=False)

        initial_date = datetime(2025, 1, 1).isoformat()

        quote_requests_df = pd.read_csv("quote_requests.csv")
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql("quote_requests", engine, if_exists="replace", index=False)

        quotes_df = pd.read_csv("quotes.csv")
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date

        if "request_metadata" in quotes_df.columns:
            quotes_df["request_metadata"] = quotes_df["request_metadata"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            quotes_df["job_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("job_type", ""))
            quotes_df["order_size"] = quotes_df["request_metadata"].apply(lambda x: x.get("order_size", ""))
            quotes_df["event_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("event_type", ""))

        quotes_df = quotes_df[[
            "request_id", "total_amount", "quote_explanation",
            "order_date", "job_type", "order_size", "event_type"
        ]]
        quotes_df.to_sql("quotes", engine, if_exists="replace", index=False)

        inventory_df = generate_sample_inventory(paper_supplies, seed=seed)

        initial_transactions = []
        initial_transactions.append({
            "item_name": None, "transaction_type": "sales",
            "units": None, "price": 50000.0, "transaction_date": initial_date,
        })

        for _, item in inventory_df.iterrows():
            initial_transactions.append({
                "item_name": item["item_name"], "transaction_type": "stock_orders",
                "units": item["current_stock"],
                "price": item["current_stock"] * item["unit_price"],
                "transaction_date": initial_date,
            })

        pd.DataFrame(initial_transactions).to_sql("transactions", engine, if_exists="append", index=False)
        inventory_df.to_sql("inventory", engine, if_exists="replace", index=False)

        return engine

    except Exception as e:
        print(f"Error initializing database: {e}")
        raise

def create_transaction(item_name, transaction_type, quantity, price, date):
    """Record a transaction of type 'stock_orders' or 'sales'."""
    try:
        date_str = date.isoformat() if isinstance(date, datetime) else date
        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError("Transaction type must be 'stock_orders' or 'sales'")
        transaction = pd.DataFrame([{
            "item_name": item_name, "transaction_type": transaction_type,
            "units": quantity, "price": price, "transaction_date": date_str,
        }])
        transaction.to_sql("transactions", db_engine, if_exists="append", index=False)
        result = pd.read_sql("SELECT last_insert_rowid() as id", db_engine)
        return int(result.iloc[0]["id"])
    except Exception as e:
        print(f"Error creating transaction: {e}")
        raise

def get_all_inventory(as_of_date: str) -> Dict[str, int]:
    """Retrieve a snapshot of available inventory as of a specific date."""
    query = """
        SELECT item_name,
            SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END) as stock
        FROM transactions
        WHERE item_name IS NOT NULL AND transaction_date <= :as_of_date
        GROUP BY item_name HAVING stock > 0
    """
    result = pd.read_sql(query, db_engine, params={"as_of_date": as_of_date})
    return dict(zip(result["item_name"], result["stock"]))

def get_stock_level(item_name: str, as_of_date) -> pd.DataFrame:
    """Retrieve the stock level of a specific item as of a given date."""
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()
    stock_query = """
        SELECT item_name,
            COALESCE(SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END), 0) AS current_stock
        FROM transactions
        WHERE item_name = :item_name AND transaction_date <= :as_of_date
    """
    return pd.read_sql(stock_query, db_engine, params={"item_name": item_name, "as_of_date": as_of_date})

def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """Estimate the supplier delivery date based on order quantity."""
    print(f"FUNC (get_supplier_delivery_date): Calculating for qty {quantity} from date string '{input_date_str}'")
    try:
        input_date_dt = datetime.fromisoformat(input_date_str.split("T")[0])
    except (ValueError, TypeError):
        print(f"WARN (get_supplier_delivery_date): Invalid date format '{input_date_str}', using today as base.")
        input_date_dt = datetime.now()
    if quantity <= 10:
        days = 0
    elif quantity <= 100:
        days = 1
    elif quantity <= 1000:
        days = 4
    else:
        days = 7
    delivery_date_dt = input_date_dt + timedelta(days=days)
    return delivery_date_dt.strftime("%Y-%m-%d")

def get_cash_balance(as_of_date) -> float:
    """Calculate the current cash balance as of a specified date."""
    try:
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.isoformat()
        transactions = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_date <= :as_of_date",
            db_engine, params={"as_of_date": as_of_date},
        )
        if not transactions.empty:
            total_sales = transactions.loc[transactions["transaction_type"] == "sales", "price"].sum()
            total_purchases = transactions.loc[transactions["transaction_type"] == "stock_orders", "price"].sum()
            return float(total_sales - total_purchases)
        return 0.0
    except Exception as e:
        print(f"Error getting cash balance: {e}")
        return 0.0

def generate_financial_report(as_of_date) -> Dict:
    """Generate a complete financial report for the company as of a specific date."""
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()
    cash = get_cash_balance(as_of_date)
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inventory_value = 0.0
    inventory_summary = []
    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(item["item_name"], as_of_date)
        stock = stock_info["current_stock"].iloc[0]
        item_value = stock * item["unit_price"]
        inventory_value += item_value
        inventory_summary.append({
            "item_name": item["item_name"],
            "stock": stock,
            "unit_price": item["unit_price"],
            "value": item_value,
        })
    top_sales_query = """
        SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
        FROM transactions
        WHERE transaction_type = 'sales' AND transaction_date <= :date
        GROUP BY item_name ORDER BY total_revenue DESC LIMIT 5
    """
    top_sales = pd.read_sql(top_sales_query, db_engine, params={"date": as_of_date})
    top_selling_products = top_sales.to_dict(orient="records")
    return {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary,
        "top_selling_products": top_selling_products,
    }

def search_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """Retrieve historical quotes matching any of the provided search terms."""
    conditions = []
    params = {}
    for i, term in enumerate(search_terms):
        param_name = f"term_{i}"
        conditions.append(
            f"(LOWER(qr.response) LIKE :{param_name} OR LOWER(q.quote_explanation) LIKE :{param_name})"
        )
        params[param_name] = f"%{term.lower()}%"
    where_clause = " AND ".join(conditions) if conditions else "1=1"
    query = f"""
        SELECT qr.response AS original_request, q.total_amount, q.quote_explanation,
            q.job_type, q.order_size, q.event_type, q.order_date
        FROM quotes q
        JOIN quote_requests qr ON q.request_id = qr.id
        WHERE {where_clause}
        ORDER BY q.order_date DESC LIMIT {limit}
    """
    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [dict(row._mapping) for row in result]


# ============================================================
# MULTI-AGENT SYSTEM
# ============================================================

# Initialize OpenAI client with the Vocareum proxy
client = OpenAI(
    api_key=os.getenv("UDACITY_OPENAI_API_KEY", ""),
    base_url="https://openai.vocareum.com/v1",
)
MODEL = "gpt-4o-mini"

# ---- TOOL IMPLEMENTATIONS ----

def tool_check_inventory(as_of_date: str) -> str:
    """Returns a JSON summary of current inventory levels."""
    inventory = get_all_inventory(as_of_date)
    if not inventory:
        return json.dumps({"status": "empty", "items": {}})
    # Also get min_stock levels
    inv_df = pd.read_sql("SELECT item_name, min_stock_level, unit_price FROM inventory", db_engine)
    result = {}
    for item_name, stock in inventory.items():
        row = inv_df[inv_df["item_name"] == item_name]
        min_stock = int(row["min_stock_level"].iloc[0]) if not row.empty else 100
        unit_price = float(row["unit_price"].iloc[0]) if not row.empty else 0.0
        result[item_name] = {
            "current_stock": int(stock),
            "min_stock_level": min_stock,
            "unit_price": unit_price,
            "needs_reorder": int(stock) < min_stock
        }
    return json.dumps({"status": "ok", "items": result})

def tool_check_item_stock(item_name: str, as_of_date: str) -> str:
    """Returns the stock level of a specific item."""
    df = get_stock_level(item_name, as_of_date)
    stock = int(df["current_stock"].iloc[0]) if not df.empty else 0
    # Get unit price from inventory
    inv_df = pd.read_sql(
        "SELECT unit_price, min_stock_level FROM inventory WHERE item_name = :name",
        db_engine, params={"name": item_name}
    )
    if inv_df.empty:
        # Item not in our inventory at all – check paper_supplies master list
        unit_price = ITEM_PRICE_LOOKUP.get(item_name, None)
        return json.dumps({
            "item_name": item_name,
            "current_stock": stock,
            "unit_price": unit_price,
            "in_catalog": unit_price is not None,
            "min_stock_level": 100
        })
    unit_price = float(inv_df["unit_price"].iloc[0])
    min_stock = int(inv_df["min_stock_level"].iloc[0])
    return json.dumps({
        "item_name": item_name,
        "current_stock": stock,
        "unit_price": unit_price,
        "in_catalog": True,
        "min_stock_level": min_stock
    })

def tool_get_quote_history(search_terms: List[str]) -> str:
    """Retrieves similar past quotes to inform pricing decisions."""
    history = search_quote_history(search_terms, limit=5)
    if not history:
        return json.dumps({"status": "no_history", "quotes": []})
    return json.dumps({"status": "ok", "quotes": history})

def tool_get_delivery_date(request_date: str, quantity: int) -> str:
    """Estimates supplier delivery date for a given quantity."""
    delivery_date = get_supplier_delivery_date(request_date, quantity)
    return json.dumps({"delivery_date": delivery_date, "quantity": quantity})

def tool_restock_item(item_name: str, quantity: int, request_date: str) -> str:
    """Orders stock from supplier and records the transaction."""
    # Get the unit price
    unit_price = ITEM_PRICE_LOOKUP.get(item_name)
    if unit_price is None:
        return json.dumps({"status": "error", "message": f"Item '{item_name}' not found in catalog"})

    cash = get_cash_balance(request_date)
    total_cost = unit_price * quantity
    if total_cost > cash:
        # Order what we can afford
        max_qty = int(cash / unit_price)
        if max_qty <= 0:
            return json.dumps({"status": "error", "message": "Insufficient cash to restock"})
        quantity = max_qty
        total_cost = unit_price * quantity

    delivery_date = get_supplier_delivery_date(request_date, quantity)
    tx_id = create_transaction(item_name, "stock_orders", quantity, total_cost, delivery_date)
    return json.dumps({
        "status": "ok",
        "item_name": item_name,
        "quantity_ordered": quantity,
        "total_cost": round(total_cost, 2),
        "delivery_date": delivery_date,
        "transaction_id": tx_id
    })

def tool_fulfill_order(item_name: str, quantity: int, unit_price: float, request_date: str) -> str:
    """Records a sale transaction and deducts from inventory."""
    # Check available stock
    df = get_stock_level(item_name, request_date)
    available = int(df["current_stock"].iloc[0]) if not df.empty else 0

    if available < quantity:
        return json.dumps({
            "status": "partial_or_unavailable",
            "item_name": item_name,
            "requested": quantity,
            "available": available,
            "message": f"Only {available} units available"
        })

    total_revenue = unit_price * quantity
    tx_id = create_transaction(item_name, "sales", quantity, total_revenue, request_date)
    return json.dumps({
        "status": "ok",
        "item_name": item_name,
        "quantity_sold": quantity,
        "unit_price": round(unit_price, 4),
        "total_revenue": round(total_revenue, 2),
        "transaction_id": tx_id
    })

def tool_get_cash_balance(as_of_date: str) -> str:
    """Returns the current cash balance."""
    balance = get_cash_balance(as_of_date)
    return json.dumps({"cash_balance": round(balance, 2), "as_of_date": as_of_date})

# ---- TOOL SCHEMAS for OpenAI function calling ----

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "check_inventory",
            "description": "Get a full snapshot of current inventory levels for all items, including which items need restocking.",
            "parameters": {
                "type": "object",
                "properties": {
                    "as_of_date": {"type": "string", "description": "Date in YYYY-MM-DD format"}
                },
                "required": ["as_of_date"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_item_stock",
            "description": "Check the stock level and unit price of a specific item by its exact name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_name": {"type": "string", "description": "Exact item name from the catalog"},
                    "as_of_date": {"type": "string", "description": "Date in YYYY-MM-DD format"}
                },
                "required": ["item_name", "as_of_date"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_quote_history",
            "description": "Search past quote history for similar orders to inform pricing. Returns similar historical quotes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_terms": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Keywords to search (e.g. item names, event types, job types)"
                    }
                },
                "required": ["search_terms"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_delivery_date",
            "description": "Estimate when a supplier order would be delivered based on quantity and request date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "request_date": {"type": "string", "description": "Request date in YYYY-MM-DD"},
                    "quantity": {"type": "integer", "description": "Number of units to order"}
                },
                "required": ["request_date", "quantity"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "restock_item",
            "description": "Order additional stock from supplier for a specific item. Use this when stock is low.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_name": {"type": "string", "description": "Exact item name"},
                    "quantity": {"type": "integer", "description": "Quantity to order"},
                    "request_date": {"type": "string", "description": "Date of request YYYY-MM-DD"}
                },
                "required": ["item_name", "quantity", "request_date"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fulfill_order",
            "description": "Record a sale for an item. This deducts from inventory and adds to revenue.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_name": {"type": "string", "description": "Exact item name"},
                    "quantity": {"type": "integer", "description": "Quantity sold"},
                    "unit_price": {"type": "number", "description": "Price per unit charged to the customer"},
                    "request_date": {"type": "string", "description": "Date of sale YYYY-MM-DD"}
                },
                "required": ["item_name", "quantity", "unit_price", "request_date"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_cash_balance",
            "description": "Get the current cash balance as of a given date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "as_of_date": {"type": "string", "description": "Date in YYYY-MM-DD format"}
                },
                "required": ["as_of_date"]
            }
        }
    }
]

# ---- TOOL DISPATCHER ----

def dispatch_tool(tool_name: str, args: dict) -> str:
    """Route tool calls to the correct Python function."""
    if tool_name == "check_inventory":
        return tool_check_inventory(**args)
    elif tool_name == "check_item_stock":
        return tool_check_item_stock(**args)
    elif tool_name == "get_quote_history":
        return tool_get_quote_history(**args)
    elif tool_name == "get_delivery_date":
        return tool_get_delivery_date(**args)
    elif tool_name == "restock_item":
        return tool_restock_item(**args)
    elif tool_name == "fulfill_order":
        return tool_fulfill_order(**args)
    elif tool_name == "get_cash_balance":
        return tool_get_cash_balance(**args)
    else:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

# ---- ORCHESTRATOR AGENT ----

SYSTEM_PROMPT = """You are the orchestrator of Munder Difflin Paper Company's multi-agent system.
Your job is to handle customer quote requests end-to-end by coordinating these sub-tasks:

1. INVENTORY CHECK: Before quoting, check if we have the requested items in stock.
2. QUOTE GENERATION: Generate competitive quotes using historical data. Apply bulk discounts:
   - 5% for orders over 500 units
   - 10% for orders over 1000 units
   - 15% for orders over 5000 units
   Use historical quotes to calibrate pricing when available.
3. RESTOCK: If stock is insufficient, order from supplier BEFORE fulfilling the order.
4. ORDER FULFILLMENT: Record the sale transaction after the customer agrees to the quote.

IMPORTANT RULES:
- Always map customer requests to the EXACT item names in our catalog.
  Known items include: "A4 paper", "Cardstock", "Colored paper", "Glossy paper", "Matte paper",
  "Recycled paper", "Construction paper", "Poster paper", "Flyers", "Paper napkins",
  "Paper cups", "Paper plates", "Heavyweight paper", "Standard copy paper", etc.
- Items NOT in our catalog (e.g. "balloons", "A3 paper", "A5 paper", "tickets") cannot be fulfilled.
  Politely note they are unavailable and skip them.
- Only fulfill items if we actually have stock (or have just restocked).
- Always confirm the delivery date feasibility.
- Be professional and customer-friendly in your final response.

Your final answer to the customer should:
1. List each requested item with: availability, unit price, quantity, discount applied, subtotal
2. State the total order amount
3. Confirm expected delivery date
4. Note any items that could not be fulfilled
"""

def run_orchestrator(customer_request: str, max_iterations: int = 20) -> str:
    """
    Run the multi-agent orchestrator for a single customer request.
    Uses OpenAI function calling in an agentic loop.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": customer_request}
    ]

    for _ in range(max_iterations):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
        )

        assistant_message = response.choices[0].message

        # Add assistant message to history
        messages.append({
            "role": "assistant",
            "content": assistant_message.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                }
                for tc in (assistant_message.tool_calls or [])
            ] if assistant_message.tool_calls else None
        })

        # If no tool calls, we have the final answer
        if not assistant_message.tool_calls:
            return assistant_message.content or "Request processed."

        # Execute all tool calls
        for tool_call in assistant_message.tool_calls:
            tool_name = tool_call.function.name
            try:
                tool_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                tool_args = {}

            print(f"  [TOOL] {tool_name}({tool_args})")
            tool_result = dispatch_tool(tool_name, tool_args)
            print(f"  [RESULT] {tool_result[:200]}...")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result
            })

    return "Request processed (max iterations reached)."


# ============================================================
# TEST SCENARIOS
# ============================================================

def run_test_scenarios():
    print("Initializing Database...")
    init_database()

    try:
        quote_requests_sample = pd.read_csv("quote_requests_sample.csv")
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"], format="%m/%d/%y", errors="coerce"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date")
    except Exception as e:
        print(f"FATAL: Error loading test data: {e}")
        return

    # Get initial state
    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    results = []
    for idx, row in quote_requests_sample.iterrows():
        request_date = row["request_date"].strftime("%Y-%m-%d")

        print(f"\n{'='*60}")
        print(f"=== Request {idx+1} ===")
        print(f"Context: {row['job']} organizing {row['event']}")
        print(f"Request Date: {request_date}")
        print(f"Cash Balance: ${current_cash:.2f}")
        print(f"Inventory Value: ${current_inventory:.2f}")

        request_with_date = f"{row['request']} (Date of request: {request_date})"
        print(f"Request: {request_with_date[:200]}...")

        # Call the multi-agent system
        try:
            response = run_orchestrator(request_with_date)
        except Exception as e:
            print(f"ERROR processing request: {e}")
            response = f"Error: {str(e)}"

        # Update state
        report = generate_financial_report(request_date)
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]

        print(f"\nResponse: {response}")
        print(f"Updated Cash: ${current_cash:.2f}")
        print(f"Updated Inventory: ${current_inventory:.2f}")

        results.append({
            "request_id": idx + 1,
            "request_date": request_date,
            "job": row["job"],
            "event": row["event"],
            "cash_balance": current_cash,
            "inventory_value": current_inventory,
            "response": response,
        })

        time.sleep(1)

    # Final report
    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    print("\n" + "="*60)
    print("===== FINAL FINANCIAL REPORT =====")
    print(f"Final Cash: ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")
    print(f"Total Assets: ${final_report['total_assets']:.2f}")
    print("\nTop Selling Products:")
    for p in final_report["top_selling_products"]:
        print(f"  - {p.get('item_name', 'N/A')}: {p.get('total_units', 0)} units, ${p.get('total_revenue', 0):.2f} revenue")

    # Save results
    pd.DataFrame(results).to_csv("test_results.csv", index=False)
    print("\nResults saved to test_results.csv")
    return results


if __name__ == "__main__":
    results = run_test_scenarios()