import streamlit as st
import pandas as pd
import requests
from datetime import datetime
from typing import Dict
import hmac
import hashlib
import urllib.parse
import time
from streamlit_autorefresh import st_autorefresh

# ================== Konfiguracja strony ==================
st.set_page_config(
    page_title="Crypto Swap Matrix - Auto",
    page_icon="🔄",
    layout="wide"
)

# Odświeżanie co 3 sekundy
st_autorefresh(interval=3000, key="refresh")

class CryptoSwapMatrix:
    def __init__(self):
        self.fee_rate = 0.00025
        self.swap_threshold = 0.5
        self.tokens_to_track = [
            'BTC','ETH','BNB','ADA','SOL','XRP','DOT','DOGE','AVAX','LTC',
            'LINK','ATOM','XLM','BCH','ALGO','FIL','ETC','XTZ','AAVE','COMP',
            'UNI','CRV','SUSHI','YFI','SNX','1INCH','ZRX','TRX','VET','ONE',
            'CELO','RSR','NKN','STORJ','DODO','KAVA','RUNE','SAND','MANA','ENJ',
            'CHZ','ALICE','NEAR','ARB','OP','APT','SUI','SEI','INJ','RENDER','MX','USDT'
        ]

    # ================== Pobieranie cen ==================
    def get_prices(self) -> Dict[str, Dict]:
        prices = {}
        try:
            response = requests.get("https://api.mexc.com/api/v3/ticker/bookTicker", timeout=10)
            if response.status_code == 200:
                data = response.json()
                usdt_pairs = {item['symbol']: item for item in data if item['symbol'].endswith('USDT')}
                for token in self.tokens_to_track:
                    if token == 'USDT':
                        prices[token] = {'bid': 1.0, 'ask': 1.0, 'last_update': datetime.now()}
                        continue
                    symbol = f"{token}USDT"
                    if symbol in usdt_pairs:
                        try:
                            bid_price = float(usdt_pairs[symbol]['bidPrice'])
                            ask_price = float(usdt_pairs[symbol]['askPrice'])
                            if bid_price > 0 and ask_price > 0:
                                prices[token] = {'bid': bid_price, 'ask': ask_price, 'last_update': datetime.now()}
                        except:
                            continue
            return prices
        except:
            return {}

    # ================== Obliczanie ekwiwalentów ==================
    def calculate_equivalent(self, from_token: str, to_token: str, quantity: float) -> float:
        if from_token == to_token:
            return quantity * (1 - self.fee_rate)
        prices = st.session_state.prices
        if from_token == 'USDT':
            return (quantity / prices[to_token]['ask']) * (1 - self.fee_rate)
        elif to_token == 'USDT':
            return quantity * prices[from_token]['bid'] * (1 - self.fee_rate)
        else:
            usdt_value = quantity * prices[from_token]['bid'] * (1 - self.fee_rate)
            return (usdt_value / prices[to_token]['ask']) * (1 - self.fee_rate)

    # ================== Wybór głównego tokena ==================
    def find_main_token(self, portfolio: Dict) -> str:
        max_value = 0
        main_token = None
        prices = st.session_state.prices
        for asset, balance in portfolio.items():
            if asset in prices:
                value = balance['total'] * prices[asset]['bid']
                if value > max_value:
                    max_value = value
                    main_token = asset
        return main_token

    # ================== Inicjalizacja Session State ==================
    def init_session_state(self):
        if 'prices' not in st.session_state:
            st.session_state.prices = self.get_prices()
        if 'portfolio' not in st.session_state:
            st.session_state.portfolio = {}
        if 'baseline_data' not in st.session_state:
            st.session_state.baseline_data = {}
        if 'top_equivalents' not in st.session_state:
            st.session_state.top_equivalents = {}
        if 'round_active' not in st.session_state:
            st.session_state.round_active = False
        if 'last_main_token' not in st.session_state:
            st.session_state.last_main_token = None
        if 'tracking' not in st.session_state:
            st.session_state.tracking = False
        if 'api_key' not in st.session_state:
            st.session_state.api_key = ""
        if 'secret_key' not in st.session_state:
            st.session_state.secret_key = ""
        if 'last_price_update' not in st.session_state:
            st.session_state.last_price_update = datetime.now()

    # ================== Aktualizacja cen ==================
    def update_prices(self):
        now = datetime.now()
        if (now - st.session_state.last_price_update).seconds >= 3:
            new_prices = self.get_prices()
            if new_prices:
                st.session_state.prices = new_prices
                st.session_state.last_price_update = now

    # ================== Pobranie portfolio z MEXC ==================
    def get_real_portfolio(self, api_key, secret_key):
        base_url = "https://api.mexc.com"
        endpoint = "/api/v3/account"
        timestamp = int(time.time() * 1000)
        params = {"recvWindow": 5000, "timestamp": timestamp}
        query_string = urllib.parse.urlencode(params)
        signature = hmac.new(secret_key.encode(), query_string.encode(), hashlib.sha256).hexdigest()
        url = f"{base_url}{endpoint}?{query_string}&signature={signature}"
        headers = {"X-MEXC-APIKEY": api_key}
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                portfolio = {}
                for item in data['balances']:
                    total = float(item['free']) + float(item['locked'])
                    if total > 0:
                        portfolio[item['asset']] = {'total': total, 'free': float(item['free']), 'locked': float(item['locked'])}
                return portfolio
            else:
                st.error("❌ Failed to fetch portfolio from MEXC")
                return {}
        except:
            st.error("❌ Connection error")
            return {}

    # ================== Baseline ==================
    def set_baseline(self, main_token):
        portfolio = st.session_state.portfolio
        amount = portfolio[main_token]['total']
        equivalents = {}
        for target_token in self.tokens_to_track:
            if target_token != main_token:
                equivalents[target_token] = self.calculate_equivalent(main_token, target_token, amount)
        usdt_value = amount if main_token == 'USDT' else amount * st.session_state.prices[main_token]['bid']
        st.session_state.baseline_data = {
            'main_token': main_token,
            'equivalents': equivalents,
            'usdt_value': usdt_value,
            'timestamp': datetime.now()
        }
        st.session_state.round_active = True

    # ================== Top equivalents ==================
    def update_top_after_swap(self, new_token):
        amount = st.session_state.portfolio[new_token]['total']
        for target_token in self.tokens_to_track:
            if target_token != new_token:
                current = self.calculate_equivalent(new_token, target_token, amount)
                current_top = st.session_state.top_equivalents.get(target_token, 0)
                if current > current_top:
                    st.session_state.top_equivalents[target_token] = current

    # ================== Detekcja akcji ==================
    def detect_action(self):
        portfolio = st.session_state.portfolio
        old_main = st.session_state.last_main_token
        new_main = self.find_main_token(portfolio)
        st.session_state.last_main_token = new_main

        usdt_value = portfolio.get('USDT', {}).get('total', 0)
        baseline_usdt = st.session_state.baseline_data.get('usdt_value', usdt_value)

        if st.session_state.round_active and new_main != old_main and new_main != 'USDT':
            self.update_top_after_swap(new_main)
            return "swap"

        if st.session_state.round_active and new_main == "USDT" and usdt_value >= 1.02 * baseline_usdt:
            st.session_state.round_active = False
            return "end_round"

        return "hold"

    # ================== Panel API ==================
    def setup_api_credentials(self):
        st.sidebar.header("🔐 API Configuration")
        with st.sidebar.form("api_config"):
            api_key = st.text_input("MEXC API Key", value=st.session_state.api_key, type="password")
            secret_key = st.text_input("MEXC Secret Key", value=st.session_state.secret_key, type="password")
            if st.form_submit_button("🔗 Connect to MEXC"):
                if api_key and secret_key:
                    st.session_state.api_key = api_key
                    st.session_state.secret_key = secret_key
                    portfolio = self.get_real_portfolio(api_key, secret_key)
                    if portfolio:
                        st.session_state.portfolio = portfolio
                        main_token = self.find_main_token(portfolio)
                        self.set_baseline(main_token)
                        st.session_state.tracking = True
                        st.success("✅ Connected and portfolio loaded")
                        st.experimental_rerun()

    # ================== Render portfolio ==================
    def render_portfolio(self):
        st.header("💰 Portfolio")
        if not st.session_state.portfolio:
            st.info("No portfolio data")
            return
        prices = st.session_state.prices
        data = []
        total_value = 0
        for asset, balance in st.session_state.portfolio.items():
            if asset in prices:
                value = balance['total'] * prices[asset]['bid']
                total_value += value
                data.append({
                    'Asset': asset,
                    'Amount': f"{balance['total']:.6f}",
                    'Value USDT': f"${value:,.2f}",
                    'Price': f"{prices[asset]['bid']:.4f}"
                })
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)
        st.metric("Total Value USDT", f"${total_value:,.2f}")

    # ================== Render matrix with Δ USDT % ==================
    def render_matrix(self):
        if not st.session_state.round_active or not st.session_state.baseline_data:
            st.info("💡 Start a round with USDT to see matrix")
            return

        main_token = st.session_state.last_main_token
        if main_token not in st.session_state.portfolio:
            st.info("No main token in portfolio")
            return

        amount = st.session_state.portfolio[main_token]['total']
        baseline_eq = st.session_state.baseline_data['equivalents']
        baseline_usdt = st.session_state.baseline_data['usdt_value']
        top_eq = st.session_state.top_equivalents

        matrix = []

        for token in self.tokens_to_track:
            if token == main_token:
                continue
            current_tokens = self.calculate_equivalent(main_token, token, amount)
            base_tokens = baseline_eq.get(token, current_tokens)
            top_tokens = top_eq.get(token, current_tokens)
            # procentowe zmiany
            delta_base = (current_tokens - base_tokens) / base_tokens * 100 if base_tokens > 0 else 0
            delta_top = (current_tokens - top_tokens) / top_tokens * 100 if top_tokens > 0 else 0
            value_usdt = current_tokens * st.session_state.prices[token]['bid'] if token != 'USDT' else current_tokens
            delta_usdt = (value_usdt - baseline_usdt) / baseline_usdt * 100 if baseline_usdt > 0 else 0

            matrix.append({
                'Target': token,
                'Current Tokens': current_tokens,
                'Value USDT': value_usdt,
                'Δ Baseline %': delta_base,
                'Top Tokens': top_tokens,
                'Δ Top %': delta_top,
                'Δ USDT %': delta_usdt
            })

        df = pd.DataFrame(matrix).sort_values('Δ USDT %', ascending=False)

        # Kolorowanie Δ
        def color_vals(val):
            if val >= self.swap_threshold:
                return 'background-color: #b7e4c7'  # zielony
            elif val >= 0:
                return 'background-color: #fff3b0'  # żółty
            else:
                return 'background-color: #f5cac3'  # czerwony

        styled_df = df.style.applymap(color_vals, subset=['Δ Baseline %','Δ Top %','Δ USDT %'])
        st.header(f"🎯 Swap Matrix - {main_token}")
        st.dataframe(styled_df, use_container_width=True)

    # ================== Render control panel ==================
    def render_control_panel(self):
        st.sidebar.header("🎮 Control Panel")
        if st.session_state.tracking:
            st.sidebar.metric("Status", "🟢 LIVE")
            if st.sidebar.button("🔄 Refresh Prices"):
                self.update_prices()
                st.experimental_rerun()

    # ================== Run ==================
    def run(self):
        st.title("🔄 Crypto Swap Matrix - Auto")
        st.markdown("---")
        self.init_session_state()
        self.update_prices()
        col1, col2 = st.columns([3,1])
        with col1:
            if not st.session_state.tracking:
                st.info("🔐 Configure MEXC API in sidebar")
            self.render_portfolio()
            st.markdown("---")
            self.render_matrix()
        with col2:
            self.setup_api_credentials()
            if st.session_state.tracking:
                self.render_control_panel()

        # Detekcja akcji
        if st.session_state.tracking and st.session_state.portfolio:
            action = self.detect_action()
            if action == "swap":
                st.info("🔄 Swap detected")
            elif action == "end_round":
                st.success("🏁 Round ended! Target reached")

# ================== Uruchomienie ==================
if __name__ == "__main__":
    app = CryptoSwapMatrix()
    app.run()
