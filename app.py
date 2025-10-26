import streamlit as st
import pandas as pd
import requests
from datetime import datetime
from typing import Dict
import hmac
import hashlib
import urllib.parse
import time

# ================== Konfiguracja strony ==================
st.set_page_config(
    page_title="Crypto Swap Matrix - Auto",
    page_icon="🔄",
    layout="wide"
)

class CryptoSwapMatrix:
    def __init__(self):
        self.fee_rate = 0.00025
        self.swap_threshold = 0.5
        self.tokens_to_track = [
            'BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'XRP', 'DOT', 'DOGE', 'AVAX', 'LTC',
            'LINK', 'ATOM', 'XLM', 'BCH', 'ALGO', 'FIL', 'ETC', 'XTZ', 'AAVE', 'COMP',
            'UNI', 'CRV', 'SUSHI', 'YFI', 'SNX', '1INCH', 'ZRX', 'TRX', 'VET', 'ONE',
            'CELO', 'RSR', 'NKN', 'STORJ', 'DODO', 'KAVA', 'RUNE', 'SAND', 'MANA', 'ENJ',
            'CHZ', 'ALICE', 'NEAR', 'ARB', 'OP', 'APT', 'SUI', 'SEI', 'INJ', 'RENDER', 'MX', 'USDT'
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
        # Włącz rundę jeśli baseline ustalony dla USDT
        st.session_state.round_active = True

    # ================== Top equivalents ==================
    def update_top_after_swap(self, new_token):
        amount = st.session_state.portfolio[new_token]['total']
        current_equivalents = {}
        for target_token in self.tokens_to_track:
            if target_token != new_token:
                current_equivalents[target_token] = self.calculate_equivalent(new_token, target_token, amount)
        for token, value in current_equivalents.items():
            current_top = st.session_state.top_equivalents.get(token, 0)
            if value > current_top:
                st.session_state.top_equivalents[token] = value

    # ================== Detekcja akcji ==================
    def detect_action(self):
        portfolio = st.session_state.portfolio
        prices = st.session_state.prices
        old_main = st.session_state.last_main_token
        new_main = self.find_main_token(portfolio)
        st.session_state.last_main_token = new_main

        usdt_value = portfolio.get('USDT', {}).get('total', 0)
        baseline_usdt = st.session_state.baseline_data.get('usdt_value', usdt_value)

        # Swap między tokenami
        if st.session_state.round_active and new_main != old_main and new_main != 'USDT':
            self.update_top_after_swap(new_main)
            return "swap"

        # Zakończenie rundy: USDT >= 102% baseline
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
                        st.rerun()

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
        st.metric("Total Value", f"${total_value:,.2f}")

    # ================== Render matryca ==================
    def render_matrix(self):
        if 'baseline_data' not in st.session_state or not st.session_state.baseline_data:
            st.info("💡 Start a round with USDT to see matrix")
            return

        main_token = st.session_state.baseline_data['main_token']
        amount = st.session_state.portfolio[main_token]['total']
        baseline = st.session_state.baseline_data['equivalents']
        top = st.session_state.top_equivalents
        matrix = []

        for token in self.tokens_to_track:
            if token == main_token:
                continue
            current = self.calculate_equivalent(main_token, token, amount)
            base = baseline.get(token, current)
            top_val = top.get(token, current)
            change_base = ((current - base)/base*100) if base > 0 else 0
            change_top = ((current - top_val)/top_val*100) if top_val > 0 else 0
            matrix.append({
                'Target': token,
                'Current': current,
                'Baseline': base,
                'Δ Baseline %': change_base,
                'Top': top_val,
                'Δ Top %': change_top
            })

        df = pd.DataFrame(matrix).sort_values('Δ Top %', ascending=False)
        st.dataframe(df, use_container_width=True)

    # ================== Render kontrolny panel ==================
    def render_control_panel(self):
        st.sidebar.header("🎮 Control Panel")
        if st.session_state.tracking:
            st.sidebar.metric("Status", "🟢 LIVE")
            if st.sidebar.button("🔄 Refresh Prices"):
                st.session_state.prices = self.get_prices()
                st.rerun()

    # ================== Run ==================
    def run(self):
        st.title("🔄 Crypto Swap Matrix - Auto")
        st.markdown("---")
        self.init_session_state()
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
