import streamlit as st
import pandas as pd
import requests
import hmac
import hashlib
import time
import json
import urllib.parse
from datetime import datetime
from typing import Dict, List, Optional

# ================== Konfiguracja strony ==================
st.set_page_config(
    page_title="Crypto Swap Matrix",
    page_icon="🔄",
    layout="wide"
)

class MexcPrivateAPI:
    def __init__(self, api_key: str, secret_key: str):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = "https://api.mexc.com"
        
    def _sign_request(self, params: dict) -> str:
        """Generuje podpis HMAC SHA256"""
        try:
            query_string = urllib.parse.urlencode(params, doseq=True)
            return hmac.new(
                self.secret_key.encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
        except:
            return ""
    
    def _make_private_request(self, endpoint: str, params: dict = None) -> Optional[dict]:
        """Wysyła autoryzowany request do MEXC"""
        try:
            timestamp = int(time.time() * 1000)
            params = params or {}
            params.update({'timestamp': timestamp, 'recvWindow': 5000})
            params = {k: v for k, v in params.items() if v is not None}
            
            signature = self._sign_request(params)
            if not signature:
                return None
                
            params['signature'] = signature
            
            headers = {'X-MEXC-APIKEY': self.api_key, 'Content-Type': 'application/json'}
            
            response = requests.get(f"{self.base_url}{endpoint}", params=params, headers=headers, timeout=10)
            
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return None
    
    def get_account_balance(self) -> Dict[str, float]:
        """Pobiera portfolio z MEXC"""
        data = self._make_private_request('/api/v3/account')
        balances = {}
        
        if data and 'balances' in data:
            for balance in data['balances']:
                asset = balance['asset']
                free = float(balance['free'])
                locked = float(balance['locked'])
                total = free + locked
                
                if total > 0:
                    balances[asset] = {
                        'free': free,
                        'locked': locked,
                        'total': total
                    }
        return balances
    
    def test_connection(self) -> bool:
        """Testuje połączenie z API"""
        return self.get_account_balance() is not None

class CryptoMatrix:
    def __init__(self):
        self.fee_rate = 0.00025
        self.tokens_to_track = [
            'BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'XRP', 'DOT', 'DOGE', 'AVAX', 'LTC',
            'LINK', 'ATOM', 'XLM', 'BCH', 'ALGO', 'FIL', 'ETC', 'XTZ', 'AAVE', 'COMP',
            'UNI', 'CRV', 'SUSHI', 'YFI', 'SNX', '1INCH', 'ZRX', 'TRX', 'VET', 'ONE',
            'CELO', 'RSR', 'NKN', 'STORJ', 'DODO', 'KAVA', 'RUNE', 'SAND', 'MANA', 'ENJ',
            'CHZ', 'ALICE', 'NEAR', 'ARB', 'OP', 'APT', 'SUI', 'SEI', 'INJ', 'RENDER', 'MX', 'USDT'
        ]
        
    def init_session_state(self):
        """Inicjalizacja session state"""
        if 'api_initialized' not in st.session_state:
            st.session_state.api_initialized = False
        if 'mexc_api' not in st.session_state:
            st.session_state.mexc_api = None
        if 'portfolio' not in st.session_state:
            st.session_state.portfolio = {}
        if 'prices' not in st.session_state:
            st.session_state.prices = {}
        if 'main_token' not in st.session_state:
            st.session_state.main_token = None
        if 'baseline_data' not in st.session_state:
            st.session_state.baseline_data = {}  # {token: {timestamp, equivalents, usdt_value}}
        if 'top_equivalents' not in st.session_state:
            st.session_state.top_equivalents = {}  # {target_token: max_equivalent}
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = datetime.now()
        
    def get_current_prices(self) -> Dict[str, Dict]:
        """Pobiera aktualne ceny bid/ask"""
        prices = {}
        try:
            response = requests.get("https://api.mexc.com/api/v3/ticker/bookTicker", timeout=10)
            if response.status_code == 200:
                data = response.json()
                usdt_pairs = {item['symbol']: item for item in data if item['symbol'].endswith('USDT')}
                
                for token in self.tokens_to_track:
                    if token == 'USDT':
                        prices[token] = {'bid': 1.0, 'ask': 1.0, 'timestamp': datetime.now()}
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
                                    'timestamp': datetime.now()
                                }
                        except:
                            continue
        except:
            pass
        return prices
    
    def calculate_equivalent(self, from_token: str, to_token: str, quantity: float) -> float:
        """Oblicza ekwiwalent między tokenami"""
        if from_token == to_token:
            return quantity * (1 - self.fee_rate)
        
        prices = st.session_state.prices
        if not prices or from_token not in prices or to_token not in prices:
            return 0.0
            
        if from_token == 'USDT':
            ask_price = prices[to_token]['ask']
            equivalent = (quantity / ask_price) * (1 - self.fee_rate)
            return equivalent
        
        elif to_token == 'USDT':
            bid_price = prices[from_token]['bid']
            equivalent = quantity * bid_price * (1 - self.fee_rate)
            return equivalent
        
        else:
            bid_price = prices[from_token]['bid']
            ask_price = prices[to_token]['ask']
            usdt_value = quantity * bid_price * (1 - self.fee_rate)
            equivalent = (usdt_value / ask_price) * (1 - self.fee_rate)
            return equivalent

    def find_main_token(self):
        """Znajduje token z najwyższą wartością w portfolio"""
        max_value = 0
        main_token = None
        prices = st.session_state.prices
        
        for asset, balance_info in st.session_state.portfolio.items():
            if asset in prices:
                value = balance_info['total'] * prices[asset]['bid']
                if value > max_value:
                    max_value = value
                    main_token = asset
        
        return main_token

    def detect_token_change(self):
        """Wykrywa zmianę głównego tokena"""
        current_main_token = self.find_main_token()
        
        if current_main_token != st.session_state.main_token:
            old_token = st.session_state.main_token
            st.session_state.main_token = current_main_token
            
            if current_main_token:
                # Nowy token - ustaw baseline
                self.set_baseline(current_main_token)
                return True
        
        return False

    def set_baseline(self, token: str):
        """Ustawia baseline dla nowego tokena"""
        if token not in st.session_state.portfolio:
            return
            
        amount = st.session_state.portfolio[token]['total']
        prices = st.session_state.prices
        
        # Oblicz ekwiwalenty
        equivalents = {}
        for target_token in self.tokens_to_track:
            if target_token != token:
                equivalent = self.calculate_equivalent(token, target_token, amount)
                equivalents[target_token] = equivalent
        
        # Oblicz wartość USDT
        usdt_value = amount if token == 'USDT' else amount * prices[token]['bid']
        
        # Zapisz baseline
        st.session_state.baseline_data[token] = {
            'timestamp': datetime.now(),
            'equivalents': equivalents,
            'usdt_value': usdt_value
        }
        
        # Zaktualizuj top equivalents
        self.update_top_equivalents(token, equivalents)

    def update_top_equivalents(self, from_token: str, new_equivalents: Dict[str, float]):
        """Aktualizuje top equivalents po swapie"""
        for target_token, equivalent in new_equivalents.items():
            current_top = st.session_state.top_equivalents.get(target_token, 0)
            if equivalent > current_top:
                st.session_state.top_equivalents[target_token] = equivalent

    def setup_api_credentials(self):
        """Setup API credentials"""
        st.sidebar.header("🔐 API Configuration")
        
        with st.sidebar.form("api_config"):
            api_key = st.text_input("MEXC API Key", type="password")
            secret_key = st.text_input("MEXC Secret Key", type="password")
            
            if st.form_submit_button("🔗 Connect to MEXC"):
                if api_key and secret_key:
                    with st.spinner("Connecting..."):
                        try:
                            test_api = MexcPrivateAPI(api_key, secret_key)
                            if test_api.test_connection():
                                st.session_state.mexc_api = test_api
                                st.session_state.api_initialized = True
                                
                                # Load initial data
                                balance = st.session_state.mexc_api.get_account_balance()
                                if balance:
                                    st.session_state.portfolio = balance
                                    st.session_state.prices = self.get_current_prices()
                                    
                                    # Find main token
                                    main_token = self.find_main_token()
                                    if main_token:
                                        st.session_state.main_token = main_token
                                        self.set_baseline(main_token)
                                    
                                    st.success("✅ Connected to MEXC API")
                                    st.rerun()
                        except:
                            st.error("❌ Connection failed")

    def render_portfolio(self):
        """Wyświetla portfolio"""
        st.header("💰 Portfolio")
        
        if not st.session_state.portfolio:
            st.info("No portfolio data")
            return
        
        prices = st.session_state.prices
        portfolio_data = []
        
        for asset, balance in st.session_state.portfolio.items():
            if asset in prices:
                value = balance['total'] * prices[asset]['bid']
                portfolio_data.append({
                    'Asset': asset,
                    'Amount': f"{balance['total']:.6f}",
                    'Value USDT': f"${value:,.2f}",
                    'Price': f"{prices[asset]['bid']:.4f}"
                })
        
        if portfolio_data:
            df = pd.DataFrame(portfolio_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            if st.session_state.main_token:
                main_value = 0
                if st.session_state.main_token in prices:
                    main_value = st.session_state.portfolio[st.session_state.main_token]['total'] * prices[st.session_state.main_token]['bid']
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Main Token", st.session_state.main_token)
                with col2:
                    st.metric("Main Token Value", f"${main_value:,.2f}")

    def render_matrix(self):
        """Renderuje matrycę ekwiwalentów"""
        if not st.session_state.main_token:
            st.info("Connect to MEXC to see matrix")
            return
            
        main_token = st.session_state.main_token
        amount = st.session_state.portfolio[main_token]['total']
        baseline = st.session_state.baseline_data.get(main_token, {})
        
        st.header(f"🎯 Swap Matrix - {main_token}")
        
        if baseline:
            baseline_time = baseline['timestamp'].strftime("%H:%M:%S")
            st.info(f"Baseline: {baseline_time} | Top: Historical max")
        
        matrix_data = []
        
        for target_token in self.tokens_to_track:
            if target_token == main_token:
                continue
                
            # Oblicz aktualny ekwiwalent
            current_equivalent = self.calculate_equivalent(main_token, target_token, amount)
            
            # Pobierz baseline equivalent
            baseline_equivalent = baseline.get('equivalents', {}).get(target_token, current_equivalent)
            
            # Pobierz top equivalent
            top_equivalent = st.session_state.top_equivalents.get(target_token, current_equivalent)
            
            # Oblicz zmiany %
            change_from_baseline = ((current_equivalent - baseline_equivalent) / baseline_equivalent * 100) if baseline_equivalent > 0 else 0
            change_from_top = ((current_equivalent - top_equivalent) / top_equivalent * 100) if top_equivalent > 0 else 0
            
            # Status
            status = "🔴"
            if change_from_top >= 0.5:
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
            df = pd.DataFrame(matrix_data)
            df = df.sort_values('Δ Top', ascending=False)
            
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    'Current': st.column_config.NumberColumn(format="%.6f"),
                    'Baseline': st.column_config.NumberColumn(format="%.6f"),
                    'Δ Baseline': st.column_config.NumberColumn(format="%+.2f%%"),
                    'Top': st.column_config.NumberColumn(format="%.6f"),
                    'Δ Top': st.column_config.NumberColumn(format="%+.2f%%"),
                    'Status': st.column_config.TextColumn()
                }
            )

    def auto_refresh(self):
        """Automatyczne odświeżanie"""
        if st.session_state.api_initialized:
            current_time = datetime.now()
            
            # Odśwież co 3 sekundy
            if (current_time - st.session_state.last_refresh).seconds >= 3:
                # Odśwież ceny
                st.session_state.prices = self.get_current_prices()
                
                # Odśwież portfolio
                if st.session_state.mexc_api:
                    new_portfolio = st.session_state.mexc_api.get_account_balance()
                    if new_portfolio:
                        st.session_state.portfolio = new_portfolio
                        
                        # Sprawdź zmianę tokena
                        token_changed = self.detect_token_change()
                        
                        # Jeśli zmiana tokena, wymuś rerun
                        if token_changed:
                            return True
                
                st.session_state.last_refresh = current_time
                
        return False

    def render_control_panel(self):
        """Panel kontrolny"""
        st.sidebar.header("🎮 Control Panel")
        
        if st.session_state.api_initialized:
            if st.session_state.main_token:
                st.sidebar.metric("Main Token", st.session_state.main_token)
            
            if st.session_state.prices:
                price_values = list(st.session_state.prices.values())
                if price_values and 'timestamp' in price_values[0]:
                    last_update = price_values[0]['timestamp']
                    st.sidebar.caption(f"Prices: {last_update.strftime('%H:%M:%S')}")
            
            if st.sidebar.button("🔄 Refresh Now"):
                if st.session_state.mexc_api:
                    st.session_state.portfolio = st.session_state.mexc_api.get_account_balance()
                st.session_state.prices = self.get_current_prices()
                st.rerun()

    def run(self):
        """Główna pętla aplikacji"""
        st.title("🔄 Crypto Swap Matrix")
        st.markdown("---")
        
        # Inicjalizacja
        self.init_session_state()
        
        # Layout
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if not st.session_state.api_initialized:
                st.info("Configure MEXC API in sidebar")
            
            self.render_portfolio()
            st.markdown("---")
            self.render_matrix()
        
        with col2:
            self.setup_api_credentials()
            if st.session_state.api_initialized:
                self.render_control_panel()
        
        # Auto refresh
        if self.auto_refresh():
            st.rerun()

# Uruchomienie
if __name__ == "__main__":
    app = CryptoMatrix()
    app.run()