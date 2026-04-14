"""
Munder Difflin Paper Company — Multi-Agent System
==================================================
Framework: smolagents (v1.24+)

Architecture:
  OrchestratorAgent (ToolCallingAgent)
      ├── InventoryAgent  (ToolCallingAgent) — checks stock, triggers restock
      ├── QuotingAgent    (ToolCallingAgent) — prices items, applies discounts
      └── FulfillmentAgent (ToolCallingAgent) — records sales transactions

Each worker agent is registered as a managed sub-agent via the
`managed_agents` parameter, giving the orchestrator a natural-language interface
to each specialist.
"""

import pandas as pd
import numpy as np
import os
import ast
import time
import dotenv
from sqlalchemy import create_engine, Engine
from sqlalchemy.sql import text
from datetime import datetime, timedelta
from typing import Dict, List, Union

from smolagents import ToolCallingAgent, Tool, OpenAIServerModel

# ---------------------------------------------------------------------------
# Environment & DB setup
# ---------------------------------------------------------------------------
dotenv.load_dotenv()

db_engine = create_engine("sqlite:///munder_difflin.db")

paper_supplies = [
    {"item_name": "A4 paper",                              "category": "paper",        "unit_price": 0.05},
    {"item_name": "Letter-sized paper",                    "category": "paper",        "unit_price": 0.06},
    {"item_name": "Cardstock",                             "category": "paper",        "unit_price": 0.15},
    {"item_name": "Colored paper",                         "category": "paper",        "unit_price": 0.10},
    {"item_name": "Glossy paper",                          "category": "paper",        "unit_price": 0.20},
    {"item_name": "Matte paper",                           "category": "paper",        "unit_price": 0.18},
    {"item_name": "Recycled paper",                        "category": "paper",        "unit_price": 0.08},
    {"item_name": "Eco-friendly paper",                    "category": "paper",        "unit_price": 0.12},
    {"item_name": "Poster paper",                          "category": "paper",        "unit_price": 0.25},
    {"item_name": "Banner paper",                          "category": "paper",        "unit_price": 0.30},
    {"item_name": "Kraft paper",                           "category": "paper",        "unit_price": 0.10},
    {"item_name": "Construction paper",                    "category": "paper",        "unit_price": 0.07},
    {"item_name": "Wrapping paper",                        "category": "paper",        "unit_price": 0.15},
    {"item_name": "Glitter paper",                         "category": "paper",        "unit_price": 0.22},
    {"item_name": "Decorative paper",                      "category": "paper",        "unit_price": 0.18},
    {"item_name": "Letterhead paper",                      "category": "paper",        "unit_price": 0.12},
    {"item_name": "Legal-size paper",                      "category": "paper",        "unit_price": 0.08},
    {"item_name": "Crepe paper",                           "category": "paper",        "unit_price": 0.05},
    {"item_name": "Photo paper",                           "category": "paper",        "unit_price": 0.25},
    {"item_name": "Uncoated paper",                        "category": "paper",        "unit_price": 0.06},
    {"item_name": "Butcher paper",                         "category": "paper",        "unit_price": 0.10},
    {"item_name": "Heavyweight paper",                     "category": "paper",        "unit_price": 0.20},
    {"item_name": "Standard copy paper",                   "category": "paper",        "unit_price": 0.04},
    {"item_name": "Bright-colored paper",                  "category": "paper",        "unit_price": 0.12},
    {"item_name": "Patterned paper",                       "category": "paper",        "unit_price": 0.15},
    {"item_name": "Paper plates",                          "category": "product",      "unit_price": 0.10},
    {"item_name": "Paper cups",                            "category": "product",      "unit_price": 0.08},
    {"item_name": "Paper napkins",                         "category": "product",      "unit_price": 0.02},
    {"item_name": "Disposable cups",                       "category": "product",      "unit_price": 0.10},
    {"item_name": "Table covers",                          "category": "product",      "unit_price": 1.50},
    {"item_name": "Envelopes",                             "category": "product",      "unit_price": 0.05},
    {"item_name": "Sticky notes",                          "category": "product",      "unit_price": 0.03},
    {"item_name": "Notepads",                              "category": "product",      "unit_price": 2.00},
    {"item_name": "Invitation cards",                      "category": "product",      "unit_price": 0.50},
    {"item_name": "Flyers",                                "category": "product",      "unit_price": 0.15},
    {"item_name": "Party streamers",                       "category": "product",      "unit_price": 0.05},
    {"item_name": "Decorative adhesive tape (washi tape)", "category": "product",      "unit_price": 0.20},
    {"item_name": "Paper party bags",                      "category": "product",      "unit_price": 0.25},
    {"item_name": "Name tags with lanyards",               "category": "product",      "unit_price": 0.75},
    {"item_name": "Presentation folders",                  "category": "product",      "unit_price": 0.50},
    {"item_name": "Large poster paper (24x36 inches)",     "category": "large_format", "unit_price": 1.00},
    {"item_name": "Rolls of banner paper (36-inch width)", "category": "large_format", "unit_price": 2.50},
    {"item_name": "100 lb cover stock",                    "category": "specialty",    "unit_price": 0.50},
    {"item_name": "80 lb text paper",                      "category": "specialty",    "unit_price": 0.40},
    {"item_name": "250 gsm cardstock",                     "category": "specialty",    "unit_price": 0.30},
    {"item_name": "220 gsm poster paper",                  "category": "specialty",    "unit_price": 0.35},
]

ITEM_PRICE_LOOKUP = {item["item_name"]: item["unit_price"] for item in paper_supplies}


# ---------------------------------------------------------------------------
# Database helper functions (from starter code)
# ---------------------------------------------------------------------------

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
            "min_stock_level": np.random.randint(50, 150),
        })
    return pd.DataFrame(inventory)


def init_database(engine: Engine = None, seed: int = 137) -> Engine:
    """Set up the Munder Difflin database with all required tables and initial records."""
    if engine is None:
        engine = db_engine
    try:
        pd.DataFrame({
            "id": [], "item_name": [], "transaction_type": [],
            "units": [], "price": [], "transaction_date": [],
        }).to_sql("transactions", engine, if_exists="replace", index=False)

        initial_date = datetime(2025, 1, 1).isoformat()

        qr = pd.read_csv("quote_requests.csv")
        qr["id"] = range(1, len(qr) + 1)
        qr.to_sql("quote_requests", engine, if_exists="replace", index=False)

        quotes_df = pd.read_csv("quotes.csv")
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date
        if "request_metadata" in quotes_df.columns:
            quotes_df["request_metadata"] = quotes_df["request_metadata"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            quotes_df["job_type"]   = quotes_df["request_metadata"].apply(lambda x: x.get("job_type", ""))
            quotes_df["order_size"] = quotes_df["request_metadata"].apply(lambda x: x.get("order_size", ""))
            quotes_df["event_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("event_type", ""))
        quotes_df[["request_id", "total_amount", "quote_explanation",
                   "order_date", "job_type", "order_size", "event_type"]
                  ].to_sql("quotes", engine, if_exists="replace", index=False)

        inv_df = generate_sample_inventory(paper_supplies, seed=seed)
        txns = [{"item_name": None, "transaction_type": "sales",
                 "units": None, "price": 50000.0, "transaction_date": initial_date}]
        for _, row in inv_df.iterrows():
            txns.append({
                "item_name": row["item_name"], "transaction_type": "stock_orders",
                "units": row["current_stock"],
                "price": row["current_stock"] * row["unit_price"],
                "transaction_date": initial_date,
            })
        pd.DataFrame(txns).to_sql("transactions", engine, if_exists="append", index=False)
        inv_df.to_sql("inventory", engine, if_exists="replace", index=False)
        return engine

    except Exception as e:
        print(f"Error initializing database: {e}")
        raise


def create_transaction(item_name, transaction_type, quantity, price, date):
    """Record a transaction of type 'stock_orders' or 'sales'."""
    date_str = date.isoformat() if isinstance(date, datetime) else date
    if transaction_type not in {"stock_orders", "sales"}:
        raise ValueError("transaction_type must be 'stock_orders' or 'sales'")
    pd.DataFrame([{
        "item_name": item_name, "transaction_type": transaction_type,
        "units": quantity, "price": price, "transaction_date": date_str,
    }]).to_sql("transactions", db_engine, if_exists="append", index=False)
    result = pd.read_sql("SELECT last_insert_rowid() as id", db_engine)
    return int(result.iloc[0]["id"])


def get_all_inventory(as_of_date: str) -> Dict[str, int]:
    """Retrieve a snapshot of available inventory as of a specific date."""
    query = """
        SELECT item_name,
            SUM(CASE WHEN transaction_type='stock_orders' THEN units
                     WHEN transaction_type='sales' THEN -units ELSE 0 END) AS stock
        FROM transactions
        WHERE item_name IS NOT NULL AND transaction_date <= :d
        GROUP BY item_name HAVING stock > 0
    """
    result = pd.read_sql(query, db_engine, params={"d": as_of_date})
    return dict(zip(result["item_name"], result["stock"]))


def get_stock_level(item_name: str, as_of_date) -> pd.DataFrame:
    """Retrieve the stock level of a specific item as of a given date."""
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()
    return pd.read_sql("""
        SELECT item_name,
            COALESCE(SUM(CASE WHEN transaction_type='stock_orders' THEN units
                              WHEN transaction_type='sales' THEN -units ELSE 0 END), 0) AS current_stock
        FROM transactions WHERE item_name=:n AND transaction_date<=:d
    """, db_engine, params={"n": item_name, "d": as_of_date})


def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """Estimate the supplier delivery date based on order quantity."""
    try:
        base = datetime.fromisoformat(input_date_str.split("T")[0])
    except (ValueError, TypeError):
        base = datetime.now()
    days = 0 if quantity <= 10 else 1 if quantity <= 100 else 4 if quantity <= 1000 else 7
    return (base + timedelta(days=days)).strftime("%Y-%m-%d")


def get_cash_balance(as_of_date) -> float:
    """Calculate the current cash balance as of a specified date."""
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()
    try:
        txns = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_date <= :d",
            db_engine, params={"d": as_of_date})
        if not txns.empty:
            return float(
                txns.loc[txns["transaction_type"] == "sales", "price"].sum() -
                txns.loc[txns["transaction_type"] == "stock_orders", "price"].sum()
            )
        return 0.0
    except Exception:
        return 0.0


def generate_financial_report(as_of_date) -> Dict:
    """Generate a complete financial report for the company as of a specific date."""
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()
    cash = get_cash_balance(as_of_date)
    inv_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inv_value = 0.0
    inv_summary = []
    for _, item in inv_df.iterrows():
        stock = get_stock_level(item["item_name"], as_of_date)["current_stock"].iloc[0]
        val = stock * item["unit_price"]
        inv_value += val
        inv_summary.append({"item_name": item["item_name"], "stock": stock,
                             "unit_price": item["unit_price"], "value": val})
    top = pd.read_sql("""
        SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
        FROM transactions WHERE transaction_type='sales' AND transaction_date<=:d
        GROUP BY item_name ORDER BY total_revenue DESC LIMIT 5
    """, db_engine, params={"d": as_of_date}).to_dict(orient="records")
    return {
        "as_of_date": as_of_date, "cash_balance": cash,
        "inventory_value": inv_value, "total_assets": cash + inv_value,
        "inventory_summary": inv_summary, "top_selling_products": top,
    }


def search_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """Retrieve historical quotes matching any of the provided search terms."""
    conditions, params = [], {}
    for i, term in enumerate(search_terms):
        k = f"t{i}"
        conditions.append(f"(LOWER(qr.response) LIKE :{k} OR LOWER(q.quote_explanation) LIKE :{k})")
        params[k] = f"%{term.lower()}%"
    where = " AND ".join(conditions) if conditions else "1=1"
    query = f"""
        SELECT qr.response AS original_request, q.total_amount, q.quote_explanation,
               q.job_type, q.order_size, q.event_type, q.order_date
        FROM quotes q JOIN quote_requests qr ON q.request_id=qr.id
        WHERE {where} ORDER BY q.order_date DESC LIMIT {limit}
    """
    with db_engine.connect() as conn:
        return [dict(r._mapping) for r in conn.execute(text(query), params)]


# ===========================================================================
# SMOLAGENTS TOOL DEFINITIONS
# Each tool subclasses smolagents.Tool and implements forward().
# ===========================================================================

class CheckInventoryTool(Tool):
    name = "check_inventory"
    description = (
        "Returns current stock levels for ALL items in the warehouse as of a given date. "
        "Also flags items that are below their minimum stock threshold and need restocking."
    )
    inputs = {
        "as_of_date": {
            "type": "string",
            "description": "Date in YYYY-MM-DD format to check inventory as of."
        }
    }
    output_type = "string"

    def forward(self, as_of_date: str) -> str:
        inventory = get_all_inventory(as_of_date)
        inv_df = pd.read_sql("SELECT item_name, min_stock_level, unit_price FROM inventory", db_engine)
        rows = []
        for item_name, stock in inventory.items():
            row = inv_df[inv_df["item_name"] == item_name]
            min_stock = int(row["min_stock_level"].iloc[0]) if not row.empty else 100
            unit_price = float(row["unit_price"].iloc[0]) if not row.empty else 0.0
            flag = " [NEEDS RESTOCK]" if int(stock) < min_stock else ""
            rows.append(f"- {item_name}: {int(stock)} units (min: {min_stock}, price: ${unit_price:.4f}){flag}")
        if not rows:
            return "Inventory is empty."
        return "Inventory as of " + as_of_date + ":\n" + "\n".join(rows)


class CheckItemStockTool(Tool):
    name = "check_item_stock"
    description = (
        "Returns the stock level and unit price for a SPECIFIC item by its exact catalog name."
    )
    inputs = {
        "item_name": {
            "type": "string",
            "description": "Exact item name as it appears in the product catalog."
        },
        "as_of_date": {
            "type": "string",
            "description": "Date in YYYY-MM-DD format."
        }
    }
    output_type = "string"

    def forward(self, item_name: str, as_of_date: str) -> str:
        df = get_stock_level(item_name, as_of_date)
        stock = int(df["current_stock"].iloc[0]) if not df.empty else 0
        inv_row = pd.read_sql(
            "SELECT unit_price, min_stock_level FROM inventory WHERE item_name=:n",
            db_engine, params={"n": item_name})
        if inv_row.empty:
            unit_price = ITEM_PRICE_LOOKUP.get(item_name)
            if unit_price is None:
                return f"'{item_name}' is NOT in our catalog and cannot be supplied."
            return (f"'{item_name}': {stock} units in stock, unit price ${unit_price:.4f}. "
                    f"Item is in catalog but not yet stocked in warehouse.")
        up = float(inv_row["unit_price"].iloc[0])
        ms = int(inv_row["min_stock_level"].iloc[0])
        status = "NEEDS RESTOCK" if stock < ms else "OK"
        return (f"'{item_name}': {stock} units in stock, unit price ${up:.4f}, "
                f"min level {ms} [{status}]")


class RestockItemTool(Tool):
    name = "restock_item"
    description = (
        "Orders additional stock from the supplier for a specific item. "
        "Records a stock_orders transaction and returns the estimated delivery date. "
        "Use this when stock is below the minimum level or insufficient for an order."
    )
    inputs = {
        "item_name": {
            "type": "string",
            "description": "Exact item name to restock."
        },
        "quantity": {
            "type": "integer",
            "description": "Number of units to order from the supplier."
        },
        "request_date": {
            "type": "string",
            "description": "Date of the restock request in YYYY-MM-DD format."
        }
    }
    output_type = "string"

    def forward(self, item_name: str, quantity: int, request_date: str) -> str:
        unit_price = ITEM_PRICE_LOOKUP.get(item_name)
        if unit_price is None:
            return f"Cannot restock '{item_name}': item not found in catalog."
        cash = get_cash_balance(request_date)
        total_cost = unit_price * quantity
        if total_cost > cash:
            max_qty = int(cash / unit_price)
            if max_qty <= 0:
                return f"Insufficient cash (${cash:.2f}) to restock '{item_name}'."
            quantity = max_qty
            total_cost = unit_price * quantity
        delivery_date = get_supplier_delivery_date(request_date, quantity)
        tx_id = create_transaction(item_name, "stock_orders", quantity, total_cost, delivery_date)
        return (f"Restocked '{item_name}': {quantity} units ordered, cost ${total_cost:.2f}, "
                f"delivery by {delivery_date} (tx #{tx_id}).")


class GetDeliveryDateTool(Tool):
    name = "get_delivery_date"
    description = (
        "Estimates the supplier delivery date for a given order quantity and start date. "
        "Lead times: <=10 units = same day, <=100 = 1 day, <=1000 = 4 days, >1000 = 7 days."
    )
    inputs = {
        "request_date": {
            "type": "string",
            "description": "Order start date in YYYY-MM-DD format."
        },
        "quantity": {
            "type": "integer",
            "description": "Total number of units to order."
        }
    }
    output_type = "string"

    def forward(self, request_date: str, quantity: int) -> str:
        delivery = get_supplier_delivery_date(request_date, quantity)
        return f"Estimated delivery for {quantity} units ordered on {request_date}: {delivery}."


class GetQuoteHistoryTool(Tool):
    name = "get_quote_history"
    description = (
        "Searches past quote history for similar orders to inform pricing decisions. "
        "Returns up to 5 historical quotes matching the given keywords."
    )
    inputs = {
        "search_terms": {
            "type": "string",
            "description": "Comma-separated keywords to search (e.g. 'glossy,cardstock,ceremony')."
        }
    }
    output_type = "string"

    def forward(self, search_terms: str) -> str:
        terms = [t.strip() for t in search_terms.split(",") if t.strip()]
        history = search_quote_history(terms, limit=5)
        if not history:
            return "No matching quote history found."
        lines = ["Relevant past quotes:"]
        for q in history:
            lines.append(
                f"- [{q.get('job_type', '')} / {q.get('event_type', '')}] "
                f"${q.get('total_amount', 0):.2f} | {q.get('quote_explanation', '')[:120]}"
            )
        return "\n".join(lines)


class FulfillOrderTool(Tool):
    name = "fulfill_order"
    description = (
        "Records a confirmed sale for one item. Deducts from inventory and adds revenue. "
        "Call this once per item after quoting and confirming stock availability."
    )
    inputs = {
        "item_name": {
            "type": "string",
            "description": "Exact item name being sold."
        },
        "quantity": {
            "type": "integer",
            "description": "Number of units sold."
        },
        "unit_price": {
            "type": "number",
            "description": "Final price per unit charged to the customer (after any discounts)."
        },
        "sale_date": {
            "type": "string",
            "description": "Date of the sale in YYYY-MM-DD format."
        }
    }
    output_type = "string"

    def forward(self, item_name: str, quantity: int, unit_price: float, sale_date: str) -> str:
        df = get_stock_level(item_name, sale_date)
        available = int(df["current_stock"].iloc[0]) if not df.empty else 0
        if available < quantity:
            return (f"Cannot fulfill '{item_name}': requested {quantity} units "
                    f"but only {available} available.")
        total = unit_price * quantity
        tx_id = create_transaction(item_name, "sales", quantity, total, sale_date)
        return (f"Sale recorded: {quantity}x '{item_name}' @ ${unit_price:.4f} = "
                f"${total:.2f} revenue (tx #{tx_id}).")


class GetCashBalanceTool(Tool):
    name = "get_cash_balance"
    description = "Returns the current cash balance of the company as of a given date."
    inputs = {
        "as_of_date": {
            "type": "string",
            "description": "Date in YYYY-MM-DD format."
        }
    }
    output_type = "string"

    def forward(self, as_of_date: str) -> str:
        balance = get_cash_balance(as_of_date)
        return f"Cash balance as of {as_of_date}: ${balance:.2f}"


class GenerateFinancialReportTool(Tool):
    name = "generate_financial_report"
    description = (
        "Generates a full financial snapshot of Munder Difflin as of a given date. "
        "Returns cash balance, total inventory value, total assets, and the top 5 "
        "best-selling products. Use this to assess financial health before quoting "
        "or to confirm the impact of a completed order."
    )
    inputs = {
        "as_of_date": {
            "type": "string",
            "description": "Date in YYYY-MM-DD format for the financial snapshot."
        }
    }
    output_type = "string"

    def forward(self, as_of_date: str) -> str:
        report = generate_financial_report(as_of_date)
        top = "\n".join(
            f"  - {p['item_name']}: {p['total_units']} units, ${p['total_revenue']:.2f} revenue"
            for p in report["top_selling_products"]
        ) or "  (no sales yet)"
        return (
            f"Financial Report as of {as_of_date}:\n"
            f"  Cash balance   : ${report['cash_balance']:.2f}\n"
            f"  Inventory value: ${report['inventory_value']:.2f}\n"
            f"  Total assets   : ${report['total_assets']:.2f}\n"
            f"Top selling products:\n{top}"
        )


# ===========================================================================
# MODEL — OpenAI-compatible proxy provided by Udacity
# ===========================================================================

model = OpenAIServerModel(
    model_id="gpt-4o-mini",
    api_base="https://openai.vocareum.com/v1",
    api_key=os.getenv("OPENAI_API_KEY", ""),
)

# ===========================================================================
# AGENT DEFINITIONS
# ===========================================================================

# --- Inventory Agent -------------------------------------------------------
# Responsible for: checking stock levels, triggering restocking when needed,
# confirming supplier delivery dates.

inventory_agent = ToolCallingAgent(
    tools=[
        CheckInventoryTool(),
        CheckItemStockTool(),
        RestockItemTool(),
        GetDeliveryDateTool(),
    ],
    model=model,
    name="inventory_agent",
    description=(
        "Specialist for warehouse inventory management. "
        "Can check stock levels for any item, identify items that need restocking, "
        "place restock orders with the supplier, and estimate delivery dates. "
        "Call this agent with questions like: 'Is item X available in quantity Y on date Z?', "
        "'Restock item Y for date Z', 'When will N units of X arrive?'"
    ),
    max_steps=10,
)

# --- Quoting Agent ---------------------------------------------------------
# Responsible for: retrieving historical quote data, calculating prices,
# applying bulk discounts based on order quantity.

quoting_agent = ToolCallingAgent(
    tools=[
        GetQuoteHistoryTool(),
        GetCashBalanceTool(),
        GenerateFinancialReportTool(),
    ],
    model=model,
    name="quoting_agent",
    description=(
        "Specialist for pricing and quote generation. "
        "Searches historical quote data to calibrate competitive prices. "
        "Applies bulk discounts: 5% for 500-999 units, 10% for 1000-4999 units, "
        "15% for 5000+ units. "
        "Call this agent to generate a price quote for a list of items and quantities."
    ),
    max_steps=6,
)

# --- Fulfillment Agent -----------------------------------------------------
# Responsible for: recording confirmed sale transactions in the database.

fulfillment_agent = ToolCallingAgent(
    tools=[
        FulfillOrderTool(),
        GetCashBalanceTool(),
    ],
    model=model,
    name="fulfillment_agent",
    description=(
        "Specialist for order fulfillment. "
        "Records confirmed sales transactions in the database, updating inventory and revenue. "
        "Call this agent once a quote is accepted and stock is confirmed, "
        "passing: item name, quantity, final unit price (after discounts), and sale date."
    ),
    max_steps=10,
)

# --- Orchestrator ----------------------------------------------------------
# Coordinates the three specialist agents end-to-end for each customer request.

ORCHESTRATOR_INSTRUCTIONS = """
You are the Munder Difflin Paper Company sales orchestrator.
For each incoming customer request you MUST follow this workflow in order:

STEP 1 — INVENTORY CHECK
  Delegate to inventory_agent: check whether each requested item is in stock in
  the required quantity. If stock is insufficient for any item, ask inventory_agent
  to restock it first (order at least the requested quantity + 20% buffer).

STEP 2 — QUOTING
  Delegate to quoting_agent: look up historical quotes for similar items and events,
  then generate a price for each item. Apply bulk discounts:
    - 500-999 units: 5% discount
    - 1000-4999 units: 10% discount
    - 5000+ units: 15% discount

STEP 3 — FULFILLMENT
  Delegate to fulfillment_agent: record the sale for each item that can be fulfilled,
  using the final discounted unit price and the request date.

STEP 4 — FINAL RESPONSE
  Compose a professional customer-facing response containing:
  - Each item: quantity, base unit price, discount applied (%), discounted price, subtotal
  - Total order amount
  - Expected delivery date
  - Any items that could NOT be supplied (not in catalog or insufficient stock)

IMPORTANT RULES:
- Map customer descriptions to EXACT catalog names. Examples:
    "printer paper" or "A4 white paper"  → "A4 paper" or "Standard copy paper"
    "glossy paper" / "glossy A4"         → "Glossy paper"
    "matte paper" / "matte A3"           → "Matte paper"
    "heavy cardstock" / "cardstock"      → "Cardstock" or "Heavyweight paper"
    "colored paper" / "assorted colors"  → "Colored paper"
    "recycled paper" / "eco paper"       → "Recycled paper"
    "construction paper"                 → "Construction paper"
    "poster boards" / "24x36 boards"     → "Large poster paper (24x36 inches)"
    "poster paper"                       → "Poster paper"
    "streamers"                          → "Party streamers"
    "washi tape"                         → "Decorative adhesive tape (washi tape)"
    "napkins" / "table napkins"          → "Paper napkins"
    "paper cups" / "biodegradable cups"  → "Paper cups"
    "paper plates" / "biodegradable plates" → "Paper plates"
    "flyers"                             → "Flyers"
    "envelopes"                          → "Envelopes"
- Items NOT in catalog (e.g. "balloons", "A3 paper", "A5 paper", "tickets",
  "ream of printer paper") cannot be fulfilled — note them politely and skip.
- Always pass the request date when calling agents.
- Do not fulfill the same item twice.
"""

orchestrator = ToolCallingAgent(
    tools=[],
    model=model,
    managed_agents=[inventory_agent, quoting_agent, fulfillment_agent],
    instructions=ORCHESTRATOR_INSTRUCTIONS,
    max_steps=25,
)


# ===========================================================================
# TEST SCENARIOS
# ===========================================================================

def run_test_scenarios():
    print("Initializing Database...")
    init_database()

    try:
        sample = pd.read_csv("quote_requests_sample.csv")
        sample["request_date"] = pd.to_datetime(
            sample["request_date"], format="%m/%d/%y", errors="coerce")
        sample.dropna(subset=["request_date"], inplace=True)
        sample = sample.sort_values("request_date")
    except Exception as e:
        print(f"FATAL: Error loading test data: {e}")
        return

    initial_date = sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    results = []
    for idx, row in sample.iterrows():
        request_date = row["request_date"].strftime("%Y-%m-%d")

        print(f"\n{'='*60}")
        print(f"=== Request {idx+1} ===")
        print(f"Context : {row['job']} organizing {row['event']}")
        print(f"Date    : {request_date}")
        print(f"Cash    : ${current_cash:.2f}  |  Inventory: ${current_inventory:.2f}")

        request_with_date = f"{row['request']} (Date of request: {request_date})"

        try:
            response = orchestrator.run(request_with_date)
        except Exception as e:
            print(f"ERROR: {e}")
            response = f"Error: {str(e)}"

        report = generate_financial_report(request_date)
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]

        print(f"\nResponse: {response}")
        print(f"Updated Cash: ${current_cash:.2f}  |  Inventory: ${current_inventory:.2f}")

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

    final_date = sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    print("\n" + "="*60)
    print("===== FINAL FINANCIAL REPORT =====")
    print(f"Cash      : ${final_report['cash_balance']:.2f}")
    print(f"Inventory : ${final_report['inventory_value']:.2f}")
    print(f"Total     : ${final_report['total_assets']:.2f}")
    print("\nTop Selling Products:")
    for p in final_report["top_selling_products"]:
        print(f"  - {p.get('item_name', 'N/A')}: "
              f"{p.get('total_units', 0)} units, ${p.get('total_revenue', 0):.2f} revenue")

    pd.DataFrame(results).to_csv("test_results.csv", index=False)
    print("\nSaved to test_results.csv")
    return results


if __name__ == "__main__":
    results = run_test_scenarios()