import streamlit as st
import websocket
import json
import threading
import time
import pandas as pd

# ============================================================
# CONFIG
# ============================================================
RANGE_SIZE = 100
LEVELS = 3
LEVEL_SPACING = 1
ORDER_SIZE = 0.001
MAX_POSITION = 0.01
MAKER_FEE = -0.00002
TAKER_FEE = 0.0004

SYMBOL = "btcusdt"
WS_URL = "wss://stream.binance.com:9443/ws/btcusdt@trade"

# ============================================================
# SESSION STATE INIT
# ============================================================

def init_state():

    defaults = {
        "price": None,
        "best_bid": None,
        "best_ask": None,
        "range_open": None,
        "range_high": None,
        "range_low": None,
        "range_dir": None,
        "target": None,
        "position": 0.0,
        "orders": [],
        "trades": [],
        "equity": 10000.0,
        "pnl": 0.0,
        "range_history": []
    }

    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()

# ============================================================
# RANGE ENGINE
# ============================================================

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

# ============================================================
# ORDER MANAGEMENT
# ============================================================

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
            price = bid - (i * LEVEL_SPACING)

            st.session_state.orders.append({
                "side": "BUY",
                "price": price,
                "size": ORDER_SIZE
            })

    if st.session_state.target == "SHORT":

        for i in range(LEVELS):
            price = ask + (i * LEVEL_SPACING)

            st.session_state.orders.append({
                "side": "SELL",
                "price": price,
                "size": ORDER_SIZE
            })

# ============================================================
# PNL
# ============================================================

def update_pnl(price):

    pos = st.session_state.position

    unrealized = pos * price

    st.session_state.pnl = unrealized

# ============================================================
# FILL SIMULATION
# ============================================================

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

        if order["side"] == "SELL" and price >= order["price"]:

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

# ============================================================
# PRICE STREAM
# ============================================================

def on_message(ws, message):

    data = json.loads(message)

    price = float(data['p'])

    st.session_state.price = price

    # simple synthetic spread
    st.session_state.best_bid = price - 0.5
    st.session_state.best_ask = price + 0.5

    update_range(price)

    place_liquidity()

    simulate_fills(price)

    update_pnl(price)

# ============================================================
# WEBSOCKET
# ============================================================

def run_ws():

    ws = websocket.WebSocketApp(
        WS_URL,
        on_message=on_message
    )

    ws.run_forever()

# ============================================================
# UI
# ============================================================

st.title("BTC Range Maker Paper Trading Bot PRO")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Price", st.session_state.price)
col2.metric("Position BTC", st.session_state.position)
col3.metric("Equity", round(st.session_state.equity,2))
col4.metric("Unrealized PnL", round(st.session_state.pnl,2))

if st.button("Start bot"):

    thread = threading.Thread(target=run_ws)
    thread.daemon = True
    thread.start()

# ============================================================
# RANGE CHART
# ============================================================

st.subheader("Range Bars")

if len(st.session_state.range_history) > 0:

    df = pd.DataFrame(st.session_state.range_history)

    st.line_chart(df['close'])

# ============================================================
# ORDERS
# ============================================================

st.subheader("Active Orders")

st.write(pd.DataFrame(st.session_state.orders))

# ============================================================
# TRADES
# ============================================================

st.subheader("Trades")

st.write(pd.DataFrame(st.session_state.trades))

# ============================================================
# AUTO REFRESH
# ============================================================

time.sleep(1)
st.experimental_rerun()
