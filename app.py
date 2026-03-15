# file: range_bars_ha_paper_ws.py
import streamlit as st
import pandas as pd
import asyncio
from mexc import MEXC

st.set_page_config(page_title="Range Bars HA Bot WS", layout="wide")
st.title("Range Bars Bot - BTC/USDT 1m Heikin Ashi (Paper Trading)")

# ===== INPUTY =====
range_size = st.number_input("Range Size ($)", min_value=1, value=100)
refresh_sec = st.number_input("Refresh every (seconds)", min_value=1, value=5)
initial_equity = 1000.0
position_pct = 0.98  # 98% equity

# ===== PAPER TRADING STATE =====
if "equity" not in st.session_state:
    st.session_state.equity = initial_equity
if "position" not in st.session_state:
    st.session_state.position = None
if "position_price" not in st.session_state:
    st.session_state.position_price = 0.0
if "last_high" not in st.session_state:
    st.session_state.last_high = None
if "last_low" not in st.session_state:
    st.session_state.last_low = None
if "last_direction" not in st.session_state:
    st.session_state.last_direction = 0

# ===== HEIKIN ASHI =====
def heikin_ashi(df):
    ha = pd.DataFrame(index=df.index, columns=["open","high","low","close"])
    ha["close"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    ha["open"] = 0.0
    ha["high"] = 0.0
    ha["low"] = 0.0
    for i in range(len(df)):
        if i == 0:
            ha["open"].iloc[i] = (df["open"].iloc[i] + df["close"].iloc[i]) / 2
        else:
            ha["open"].iloc[i] = (ha["open"].iloc[i-1] + ha["close"].iloc[i-1]) / 2
        ha["high"].iloc[i] = max(df["high"].iloc[i], ha["open"].iloc[i], ha["close"].iloc[i])
        ha["low"].iloc[i] = min(df["low"].iloc[i], ha["open"].iloc[i], ha["close"].iloc[i])
    return ha

# ===== RANGE BARS =====
def compute_signals(df, range_size, last_high, last_low, last_direction):
    signals = []
    for i in range(len(df)):
        current_high = df["high"].iloc[i]
        current_low = df["low"].iloc[i]
        current_direction = last_direction

        if current_high >= last_high + range_size:
            last_high += range_size
            last_low = last_high - range_size
            current_direction = 1  # LONG
        elif current_low <= last_low - range_size:
            last_low -= range_size
            last_high = last_low + range_size
            current_direction = -1  # SHORT

        if current_direction != last_direction:
            if current_direction == 1:
                signals.append(("LONG", current_high))
            elif current_direction == -1:
                signals.append(("SHORT", current_low))

        last_direction = current_direction

    return signals, last_high, last_low, last_direction

# ===== ASYNC STREAMING =====
placeholder = st.empty()

async def main():
    async with MEXC.new() as client:
        async for candle in client.spot.streams.candles('BTCUSDT', interval='Min1'):
            # candle = {'open', 'high', 'low', 'close', 'volume', 'timestamp'}
            df = pd.DataFrame([candle])
            df = df[['open','high','low','close']].astype(float)
            df = heikin_ashi(df)

            # initialize last_high/low
            if st.session_state.last_high is None:
                st.session_state.last_high = df["high"].iloc[0]
            if st.session_state.last_low is None:
                st.session_state.last_low = df["low"].iloc[0]

            # compute signals
            signals, st.session_state.last_high, st.session_state.last_low, st.session_state.last_direction = \
                compute_signals(df, range_size, st.session_state.last_high, st.session_state.last_low, st.session_state.last_direction)

            # PAPER TRADING LOGIC
            for sig in signals:
                side, price = sig
                position_size = st.session_state.equity * position_pct
                if st.session_state.position is None:
                    st.session_state.position = side
                    st.session_state.position_price = price
                elif st.session_state.position != side:
                    pnl = (price - st.session_state.position_price) if st.session_state.position=="LONG" else (st.session_state.position_price - price)
                    st.session_state.equity += pnl
                    st.session_state.position = side
                    st.session_state.position_price = price

            # DISPLAY
            with placeholder.container():
                st.subheader(f"Equity: {st.session_state.equity:.2f} USDT | Current Position: {st.session_state.position if st.session_state.position else 'None'}")
                st.subheader("Last Signals:")
                for sig in signals[-10:]:
                    st.write(f"{sig[0]} @ {sig[1]:.2f} USDT")
                st.line_chart(df[["close","high","low"]])

# RUN ASYNC LOOP
asyncio.run(main())
