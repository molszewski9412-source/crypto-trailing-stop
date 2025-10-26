import streamlit as st
import pandas as pd
import requests
import time
from datetime import datetime
import json
import hmac
import hashlib
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
        self.data_file = "auto_swap_data.json"
        self.tokens_to_track = [
            'BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'XRP', 'DOT', 'DOGE', 'AVAX', 'LTC',
            'LINK', 'ATOM', 'XLM', 'BCH', 'ALGO', 'FIL', 'ETC', 'XTZ', 'AAVE', 'COMP',
            'UNI', 'CRV', 'SUSHI', 'YFI', 'SNX', '1INCH', 'ZRX', 'TRX', 'VET', 'ONE',
            'CELO', 'RSR', 'NKN', 'STORJ', 'DODO', 'KAVA', 'RUNE', 'SAND', 'MANA', 'ENJ',
            'CHZ', 'ALICE', 'NEAR', 'ARB', 'OP', 'APT', 'SUI', 'SEI', 'INJ', 'RENDER', 'MX', 'USDT'
        ]

    # ================== API - POBIERANIE CEN ==================
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
                                prices[token] = {
                                    'bid': bid_price,
                                    'ask': ask_price,
                                    'last_update': datetime.now()
                                }
                        except:
                            continue
            return prices
        except Exception:
            return {}

    # ================== API - PORTFEL ==================
    def get_real_portfolio(self, api_key: str, secret_key: str):
        """Pobiera prawdziwe saldo z MEXC API"""
        try:
            timestamp = int(time.time() * 1000)
            query_string = f"timestamp={timestamp}"
            signature = hmac.new(secret_key.encode(), query_string.encode(), hashlib.sha256).hexdigest()

            headers = {"X-MEXC-APIKEY": api_key}
            url = f"https://api.mexc.com/api/v3/account?{query_string}&signature={signature}"

            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                st.error(f"❌ API error: {response.text}")
                return {}

            data = response.json()
            portfolio = {}
            for asset in data.get("balances", []):
                total = float(asset["free"]) + float(asset["locked"])
                if total > 0:
                    portfolio[asset["asset"]] = {
                        "total": total,
                        "free": float(asset["free"]),
                        "locked": float(asset["locked"])
                    }

            return portfolio

        except Exception as e:
            st.error(f"⚠️ Error fetching portfolio: {e}")
            return {}

    # ================== OBLICZENIA ==================
    def calculate_equivalent(self, from_token: str, to_token: str, quantity: float) -> float:
        if from_token == to_token:
            return quantity * (1 - self.fee_rate)
        
        prices = st.session_state.prices
        if not prices or from_token not in prices or to_token not in prices:
            return 0.0
            
        if from_token == 'USDT':
            ask_price = prices[to_token]['ask']
            return (quantity / ask_price) * (1 - self.fee_rate)
        
        elif to_token == 'USDT':
            bid_price = prices[from_token]['bid']
            return quantity * bid_price * (1 - self.fee_rate)
        
        else:
            bid_price = prices[from_token]['bid']
            ask_price = prices[to_token]['ask']
            usdt_value = quantity * bid_price * (1 - self.fee_rate)
            return (usdt_value / ask_price) * (1 - self.fee_rate)

    def find_main_token(self, portfolio: Dict) -> str:
        max_value = 0
        main_token = None
        prices = st.session_state.prices
        
        for asset, balance_info in portfolio.items():
            if asset in prices:
                total = balance_info.get('total', 0)
                value = total * prices[asset]['bid']
                if value > max_value:
                    max_value = value
                    main_token = asset
        return main_token

    def set_baseline(self, token: str):
        if token not in st.session_state.portfolio:
            return
            
        amount = st.session_state.portfolio[token]['total']
        prices = st.session_state.prices
        
        equivalents = {}
        for target_token in self.tokens_to_track:
            if target_token != token:
                equivalent = self.calculate_equivalent(token, target_token, amount)
                equivalents[target_token] = equivalent
        
        usdt_value = amount if token == 'USDT' else amount * prices[token]['bid']
        
        st.session_state.baseline_data[token] = {
            'timestamp': datetime.now(),
            'equivalents': equivalents,
            'usdt_value': usdt_value
        }

    def update_top_after_swap(self, new_token: str):
        if new_token not in st.session_state.portfolio:
            return
            
        amount = st.session_state.portfolio[new_token]['total']
        
        current_equivalents = {}
        for target_token in self.tokens_to_track:
            if target_token != new_token:
                equivalent = self.calculate_equivalent(new_token, target_token, amount)
                current_equivalents[target_token] = equivalent
        
        st.session_state.top_equivalents[new_token] = amount
        
        for target_token, equivalent in current_equivalents.items():
            current_top = st.session_state.top_equivalents.get(target_token, 0)
            if equivalent > current_top:
                st.session_state.top_equivalents[target_token] = equivalent

    def detect_swap_and_update_top(self):
        current_main_token = self.find_main_token(st.session_state.portfolio)
        
        if st.session_state.main_token is None and current_main_token:
            st.session_state.main_token = current_main_token
            self.set_baseline(current_main_token)
            return False
            
        if current_main_token and current_main_token != st.session_state.main_token:
            old_token = st.session_state.main_token
            new_token = current_main_token
            
            st.session_state.main_token = new_token
            self.update_top_after_swap(new_token)
            self.set_baseline(new_token)
            
            st.success(f"🔄 Swap detected: {old_token} → {new_token}")
            return True
        
        return False

    # ================== INICJALIZACJA ==================
    def init_session_state(self):
        if 'prices' not in st.session_state:
            st.session_state.prices = self.get_prices()
        if 'portfolio' not in st.session_state:
            st.session_state.portfolio = {}
        if 'main_token' not in st.session_state:
            st.session_state.main_token = None
        if 'baseline_data' not in st.session_state:
            st.session_state.baseline_data = {}
        if 'top_equivalents' not in st.session_state:
            st.session_state.top_equivalents = {}
        if 'tracking' not in st.session_state:
            st.session_state.tracking = False
        if 'last_price_update' not in st.session_state:
            st.session_state.last_price_update = datetime.now()

    # ================== WIDOKI ==================
    def render_portfolio(self):
        st.header("💰 Portfolio")
        
        if not st.session_state.portfolio:
            st.info("No portfolio data")
            return
        
        prices = st.session_state.prices
        portfolio_data = []
        total_value = 0
        
        for asset, balance in st.session_state.portfolio.items():
            if asset in prices:
                value = balance['total'] * prices[asset]['bid']
                total_value += value
                portfolio_data.append({
                    'Asset': asset,
                    'Amount': f"{balance['total']:.6f}",
                    'Value USDT': f"${value:,.2f}",
                    'Price': f"{prices[asset]['bid']:.4f}"
                })
        
        if portfolio_data:
            df = pd.DataFrame(portfolio_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Value", f"${total_value:,.2f}")
            with col2:
                st.metric("Assets", len(portfolio_data))
            with col3:
                if st.session_state.main_token:
                    main_value = st.session_state.portfolio[st.session_state.main_token]['total'] * prices[st.session_state.main_token]['bid']
                    st.metric("Main Token", st.session_state.main_token, f"${main_value:,.2f}")

    def render_matrix(self):
        if not st.session_state.main_token:
            st.info("💡 Connect to MEXC to see matrix")
            return
            
        main_token = st.session_state.main_token
        amount = st.session_state.portfolio[main_token]['total']
        baseline = st.session_state.baseline_data.get(main_token, {})
        
        st.header(f"🎯 Swap Matrix - {main_token}")
        
        if baseline:
            baseline_time = baseline['timestamp'].strftime("%H:%M:%S")
            st.info(f"Baseline: {baseline_time} | Top: Updated on swap")
        
        matrix_data = []
        
        for target_token in self.tokens_to_track:
            if target_token == main_token:
                continue
                
            current_equivalent = self.calculate_equivalent(main_token, target_token, amount)
            baseline_equivalent = baseline.get('equivalents', {}).get(target_token, current_equivalent)
            top_equivalent = st.session_state.top_equivalents.get(target_token, current_equivalent)
            
            change_from_baseline = ((current_equivalent - baseline_equivalent) / baseline_equivalent * 100) if baseline_equivalent > 0 else 0
            change_from_top = ((current_equivalent - top_equivalent) / top_equivalent * 100) if top_equivalent > 0 else 0
            
            status = "🔴"
            if change_from_top >= self.swap_threshold:
                status = "🟢 SWAP"
            elif change_from_baseline >= 0:
                status = "🟡"
            
            matrix_data.append({
                'Target': target_token,
                'Current': current_equivalent,
                'Baseline': baseline_equivalent,
                'Δ Baseline': change_from_baseline,
                'Top': top_equivalent,
                'Δ Top': change_from_top,
                'Status': status
            })
        
        if matrix_data:
            df = pd.DataFrame(matrix_data).sort_values('Δ Top', ascending=False)
            st.dataframe(df, use_container_width=True, hide_index=True)

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
                            st.session_state.api_key = api_key
                            st.session_state.secret_key = secret_key
                            st.session_state.portfolio = self.get_real_portfolio(api_key, secret_key)
                            st.success("✅ Connected to MEXC")
                            st.rerun()
                        else:
                            st.error("❌ API connection failed")
                    except Exception as e:
                        st.error(f"❌ Connection error: {e}")

    def render_control_panel(self):
        st.sidebar.header("🎮 Control Panel")
        if st.session_state.tracking:
            status = "🟢 LIVE"
            st.sidebar.metric("Status", status)
            
            if st.sidebar.button("🔄 Refresh Now"):
                st.session_state.prices = self.get_prices()
                st.session_state.portfolio = self.get_real_portfolio(st.session_state.api_key, st.session_state.secret_key)
                st.rerun()

    # ================== PĘTLA GŁÓWNA ==================
    def run(self):
        st.title("🔄 Crypto Swap Matrix - Auto")
        st.markdown("---")
        
        self.init_session_state()
        
        col1, col2 = st.columns([3, 1])
        
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
        
        # Auto refresh
        if st.session_state.tracking:
            self.update_prices()
            st.session_state.portfolio = self.get_real_portfolio(st.session_state.api_key, st.session_state.secret_key)
            self.detect_swap_and_update_top()
            time.sleep(3)
            st.rerun()

    # ================== CENY AUTO-REFRESH ==================
    def update_prices(self):
        if hasattr(st.session_state, 'last_price_update'):
            if (datetime.now() - st.session_state.last_price_update).seconds < 3:
                return
        new_prices = self.get_prices()
        if new_prices:
            st.session_state.prices = new_prices
            st.session_state.last_price_update = datetime.now()


# ================== URUCHOMIENIE ==================
if __name__ == "__main__":
    app = CryptoSwapMatrix()
    app.run()
