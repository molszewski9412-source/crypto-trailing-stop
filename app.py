# range_bars_ha_bot.py
import streamlit as st
import pandas as pd
import requests
import time

st.set_page_config(page_title="Range Bars Heikin Ashi Bot", layout="wide")
st.title("Range Bars Bot - BTC/USDT 1m Heikin Ashi (Paper Trading)")

# ===== INPUTY =====
range_size = st.number_input("Range Size ($)", min_value=1, value=100)
refresh_sec = st.number_input("Refresh every (seconds)", min_value=1, value=5)
initial_equity = 1000.0

# ===== PAPER TRADING =====
equity = initial_equity
position = None
position_price = 0.0

# ===== FUNKCJA POBIERANIA DANYCH MEXC =====
def get_btc_candles(limit=100):
    url = f"https://www.mexc.com/api/v3/klines?symbol=BTCUSDT&interval=1m&limit={limit}"
    resp = requests.get(url)
    data = resp.json()
    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    df = df.astype({
        "open": float, "high": float, "low": float, "close": float
    })
    return df

# ===== FUNKCJA HEIKIN ASHI =====
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

# ===== LOGIKA RANGE BARS =====
def compute_signals(df, range_size, last_high, last_low, last_direction):
    signals = []
    for i in range(1, len(df)):
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

# ===== PĘTLA STREAMLIT =====
placeholder = st.empty()
last_high, last_low, last_direction = None, None, 0

while True:
    df_raw = get_btc_candles(limit=100)
    df = heikin_ashi(df_raw)

    if last_high is None or last_low is None:
        last_high = df["high"].iloc[0]
        last_low = df["low"].iloc[0]

    signals, last_high, last_low, last_direction = compute_signals(df, range_size, last_high, last_low, last_direction)

    # ===== PAPER TRADING =====
    for sig in signals:
        side, price = sig
        if position is None:
            position = side
            position_price = price
        elif position != side:
            # zamykamy poprzednią pozycję
            pnl = (price - position_price) if position=="LONG" else (position_price - price)
            equity += pnl
            position = side
            position_price = price

    # ===== WYŚWIETLANIE =====
    with placeholder.container():
        st.subheader(f"Equity: {equity:.2f} USDT | Current Position: {position if position else 'None'}")
        st.subheader("Ostatnie 10 sygnałów:")
        for sig in signals[-10:]:
            st.write(f"{sig[0]} @ {sig[1]:.2f} USDT")
        st.line_chart(df[["close","high","low"]])

    time.sleep(refresh_sec)
