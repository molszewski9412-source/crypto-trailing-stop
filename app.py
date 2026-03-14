# streamlit_range_maker_pro.py
import streamlit as st
import pandas as pd
import requests
import time
import random

# =======================
# CONFIG
# =======================
RANGE_SIZE = 100
LEVELS = 3
LEVEL_SPACING = 1
ORDER_SIZE = 0.001
MAX_POSITION = 0.01
MAKER_FEE = -0.00002
TICK_INTERVAL = 1  # seconds
SYMBOL = "BTCUSDT"

# =======================
# SESSION STATE INIT
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

# =======================
# FETCH PRICE
# =======================
def get_price():
    url = f"https://api.binance.com/api/v3/ticker/price?symbol={SYMBOL}"
    try:
        r = requests.get(url, timeout=1)
        data = r.json()
        return float(data["price"])
    except:
        return st.session_state.price  # keep last price if fail

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

        # flip LONG/SHORT
        if st.session_state.range_dir and new_dir != st.session_state.range_dir:
            st.session_state.target = "LONG" if new_dir == "UP" else "SHORT"

        st.session_state.range_dir = new_dir
        st.session_state.range_open = price
        st.session_state.range_high = price
        st.session_state.range_low = price

# =======================
# PLACE LIQUIDITY
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
# SIMULATE FILLS
# =======================
def simulate_fills(price):
    filled = []
    for order in st.session_state.orders:
        # fill probability simulation
        fill_prob = random.uniform(0,1)
        if fill_prob > 0.5:  # ~50% chance fill per tick
            if order["side"] == "BUY" and st.session_state.position < MAX_POSITION:
                fee = order["price"] * order["size"] * MAKER_FEE
                st.session_state.position += order["size"]
                st.session_state.equity += fee
                st.session_state.trades.append({
                    "side": "BUY",
                    "price": order["price"],
                    "size": order["size"]
                })
                filled.append(order)
            elif order["side"] == "SELL" and st.session_state.position > -MAX_POSITION:
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
# PNL UPDATE
# =======================
def update_pnl(price):
    st.session_state.pnl = st.session_state.position * price

# =======================
# BOT TICK LOOP
# =======================
placeholder = st.empty()

start_col, stop_col = st.columns(2)
if start_col.button("Start bot"):
    st.session_state.bot_running = True
if stop_col.button("Stop bot"):
    st.session_state.bot_running = False

while st.session_state.bot_running:
    price = get_price()
if price is None:
    st.warning("Price not available, retrying...")
else:
    st.session_state.price = price
    st.session_state.best_bid = price - 0.5
    st.session_state.best_ask = price + 0.5

    # dalsze funkcje bota
    update_range(price)
    place_liquidity()
    simulate_fills(price)
    update_pnl(price)
    st.session_state.price = price
    st.session_state.best_bid = price - 0.5
    st.session_state.best_ask = price + 0.5

    update_range(price)
    place_liquidity()
    simulate_fills(price)
    update_pnl(price)

    with placeholder.container():
        st.title("BTC Range Maker Bot - LIVE per tick")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("BTC Price", st.session_state.price)
        col2.metric("Position BTC", round(st.session_state.position,6))
        col3.metric("Equity", round(st.session_state.equity,2))
        col4.metric("Unrealized PnL", round(st.session_state.pnl,2))

        st.subheader("Range Bars")
        if st.session_state.range_history:
            df = pd.DataFrame(st.session_state.range_history)
            st.line_chart(df['close'])

        st.subheader("Active Orders")
        st.write(pd.DataFrame(st.session_state.orders))

        st.subheader("Trades")
        st.write(pd.DataFrame(st.session_state.trades))

    time.sleep(TICK_INTERVAL)
