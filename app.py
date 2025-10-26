import streamlit as st
import pandas as pd
import requests
import time
from datetime import datetime
from typing import Dict

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

    # ================== Detekcja akcji ==================
    def detect_action(self, new_portfolio):
        old_main = st.session_state.last_main_token
        new_main = self.find_main_token(new_portfolio)
        st.session_state.last_main_token = new_main
        prices = st.session_state.prices

        usdt_value = new_portfolio.get('USDT', {}).get('total', 0)
        total_value = sum(balance['total']*prices[a]['bid'] for a, balance in new_portfolio.items() if a in prices)

        baseline_usdt = st.session_state.baseline_data.get('usdt_value', total_value)

        # start rundy
        if not st.session_state.round_active and old_main == "USDT" and new_main != "USDT":
            st.session_state.round_active = True
            self.set_baseline(new_main)
            return "start_round"

        # swap między tokenami
        if st.session_state.round_active and new_main != "USDT" and old_main != new_main:
            self.update_top_after_swap(new_main)
            return "swap"

        # zakończenie rundy: wartość USDT >= 102% baseline
        if st.session_state.round_active and new_main == "USDT" and usdt_value >= 1.02 * baseline_usdt:
            st.session_state.round_active = False
            return "end_round"

        return "hold"

    # ================== Ustawianie baseline ==================
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

    # ================== Aktualizacja top equivalents ==================
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

    # ================== Panel API ==================
    def setup_api_credentials(self):
        st.sidebar.header("🔐 API Configuration")
        with st.sidebar.form("api_config"):
            api_key = st.text_input("MEXC API Key", type="password")
            secret_key = st.text_input("MEXC Secret Key", type="password")
            if st.form_submit_button("🔗 Connect to MEXC"):
                if api_key and secret_key:
                    try:
                        test_response = requests.get("https://api.mexc.com/api/v3/ping", timeout=10)
                        if test_response.status_code == 200:
                            st.session_state.tracking = True
                            st.session_state.portfolio = {
                                'BTC': {'total': 0.1, 'free': 0.1, 'locked': 0},
                                'USDT': {'total': 1000, 'free': 1000, 'locked': 0},
                                'MX': {'total': 100, 'free': 100, 'locked': 0}
                            }
                            st.success("✅ Connected to MEXC")
                            st.experimental_rerun()
                        else:
                            st.error("❌ API connection failed")
                    except:
                        st.error("❌ Connection error")

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

    # ================== Render matrix ==================
    def render_matrix(self):
        if not st.session_state.round_active:
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
        st.dataframe(df)

    # ================== Render Control Panel ==================
    def render_control_panel(self):
        st.sidebar.header("🎮 Control Panel")
        if st.session_state.tracking:
            st.sidebar.metric("Status", "🟢 LIVE" if st.session_state.tracking else "🔴 STOPPED")
            if st.sidebar.button("🔄 Refresh Now"):
                st.session_state.prices = self.get_prices()
                st.experimental_rerun()

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
        if st.session_state.portfolio:
            action = self.detect_action(st.session_state.portfolio)
            if action == "start_round":
                st.success("🎯 Round started")
            elif action == "swap":
                st.info("🔄 Swap detected")
            elif action == "end_round":
                st.success("🏁 Round ended! Target reached")
        # Auto refresh co 3 sekundy
        if st.session_state.tracking:
            time.sleep(3)
            st.experimental_rerun()

# ================== Uruchomienie ==================
if __name__ == "__main__":
    app = CryptoSwapMatrix()
    app.run()
