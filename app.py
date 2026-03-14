# streamlit_papertrading_maker_bot.py
import streamlit as st
import threading
import json
from websocket import WebSocketApp
import pandas as pd

# =======================
# CONFIG
# =======================
RANGE_SIZE = 100
LEVELS = 3
LEVEL_SPACING = 1
ORDER_SIZE = 0.001
MAX_POSITION = 0.01
MAKER_FEE = -0.00002
SYMBOL = "BTCUSDT"
WS_URL = f"wss://stream.binance.com:9443/ws/{SYMBOL.lower()}@trade"

# =======================
# INIT SESSION STATE
# =======================
if "price" not in st.session_state:
    st.session_state.price = None
if "best_bid" not in st.session_state:
    st.session_state.best_bid = None
if "best_ask" not in st.session_state:
    st.session_state.best_ask = None
if "range_open" not in st.session_state:
    st.session_state.range_open = None
if "range_high" not in st.session_state:
    st.session_state.range_high = None
if "range_low" not in st.session_state:
    st.session_state.range_low = None
if "range_dir" not in st.session_state:
    st.session_state.range_dir = None
if "target" not in st.session_state:
    st.session_state.target = None
if "position" not in st.session_state:
    st.session_state.position = 0.0
if "orders" not in st.session_state:
    st.session_state.orders = []
if "trades" not in st.session_state:
    st.session_state.trades = []
if "equity" not in st.session_state:
    st.session_state.equity = 10000.0
if "pnl" not in st.session_state:
    st.session_state.pnl = 0.0
if "range_history" not in st.session_state:
    st.session_state.range_history = []
if "bot_running" not in st.session_state:
    st.session_state.bot_running = False
if "ws_thread" not in st.session_state:
    st.session_state.ws_thread = None

# =======================
# RANGE ENGINE
# =======================
def update_range(price):
    if st.session_state.range_open is None:
        st.session_state.range_open = price
        st.session_state.range_high = price
        st.session_state.range_low = price
        return

    st.session_state.range_high = max(st.session_state.range_high, price)
    st.session_state.range_low = min(st.session_state.range_low, price)

    if st.session_state.range_high - st.session_state.range_low >= RANGE_SIZE:
        new_dir = "UP" if price > st.session_state.range_open else "DOWN"

        st.session_state.range_history.append({
            "open": st.session_state.range_open,
            "high": st.session_state.range_high,
            "low": st.session_state.range_low,
            "close": price,
            "dir": new_dir
        })

        if st.session_state.range_dir and new_dir != st.session_state.range_dir:
            st.session_state.target = "LONG" if new_dir == "UP" else "SHORT"

        st.session_state.range_dir = new_dir
        st.session_state.range_open = price
        st.session_state.range_high = price
        st.session_state.range_low = price

# =======================
# ORDER MANAGEMENT
# =======================
def cancel_all():
    st.session_state.orders = []

def place_liquidity():
    bid = st.session_state.best_bid
    ask = st.session_state.best_ask
    if bid is None or ask is None:
        return
    cancel_all()

    if st.session_state.target == "LONG":
        for i in range(LEVELS):
            price = bid - i * LEVEL_SPACING
            st.session_state.orders.append({
                "side": "BUY",
                "price": price,
                "size": ORDER_SIZE
            })
    elif st.session_state.target == "SHORT":
        for i in range(LEVELS):
            price = ask + i * LEVEL_SPACING
            st.session_state.orders.append({
                "side": "SELL",
                "price": price,
                "size": ORDER_SIZE
            })

# =======================
# PNL UPDATE
# =======================
def update_pnl(price):
    pos = st.session_state.position
    st.session_state.pnl = pos * price

# =======================
# SIMULATE FILLS
# =======================
def simulate_fills(price):
    filled = []
    for order in st.session_state.orders:
        if order["side"] == "BUY" and price <= order["price"]:
            if st.session_state.position < MAX_POSITION:
                fee = order["price"] * order["size"] * MAKER_FEE
                st.session_state.position += order["size"]
                st.session_state.equity += fee
                st.session_state.trades.append({
                    "side": "BUY",
                    "price": order["price"],
                    "size": order["size"]
                })
                filled.append(order)
        elif order["side"] == "SELL" and price >= order["price"]:
            if st.session_state.position > -MAX_POSITION:
                fee = order["price"] * order["size"] * MAKER_FEE
                st.session_state.position -= order["size"]
                st.session_state.equity += fee
                st.session_state.trades.append({
                    "side": "SELL",
                    "price": order["price"],
                    "size": order["size"]
                })
                filled.append(order)
    for f in filled:
        st.session_state.orders.remove(f)

# =======================
# WEBSOCKET HANDLER
# =======================
def run_ws():
    def on_message(ws, message):
        data = json.loads(message)
        price = float(data["p"])

        st.session_state.price = price
        st.session_state.best_bid = price - 0.5
        st.session_state.best_ask = price + 0.5

        update_range(price)
        place_liquidity()
        simulate_fills(price)
        update_pnl(price)

    ws = WebSocketApp(
        WS_URL,
        on_message=on_message
    )
    ws.run_forever()

# =======================
# STREAMLIT UI
# =======================
st.title("BTC Range Maker Paper Trading Bot - LIVE per tick")

col1, col2, col3, col4 = st.columns(4)
col1.metric("BTC Price", st.session_state.price)
col2.metric("Position BTC", round(st.session_state.position,6))
col3.metric("Equity", round(st.session_state.equity,2))
col4.metric("Unrealized PnL", round(st.session_state.pnl,2))

start_col, stop_col = st.columns(2)
if start_col.button("Start bot") and not st.session_state.bot_running:
    st.session_state.bot_running = True
    thread = threading.Thread(target=run_ws)
    thread.daemon = True
    thread.start()
    st.session_state.ws_thread = thread

if stop_col.button("Stop bot"):
    st.session_state.bot_running = False
    st.session_state.ws_thread = None

# =======================
# RANGE CHART
# =======================
st.subheader("Range Bars")
if len(st.session_state.range_history) > 0:
    df = pd.DataFrame(st.session_state.range_history)
    st.line_chart(df['close'])

# =======================
# ACTIVE ORDERS
# =======================
st.subheader("Active Orders")
st.write(pd.DataFrame(st.session_state.orders))

# =======================
# TRADES
# =======================
st.subheader("Trades")
st.write(pd.DataFrame(st.session_state.trades))
