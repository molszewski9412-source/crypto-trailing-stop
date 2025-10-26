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
    page_title="Crypto Portfolio Tracker - MEXC",
    page_icon="📊",
    layout="wide"
)

class MexcPrivateAPI:
    def __init__(self, api_key: str, secret_key: str):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = "https://api.mexc.com"
        
    def _sign_request(self, params: dict) -> str:
        """Generuje podpis HMAC SHA256 dla requestów MEXC"""
        try:
            # MEXC wymaga specjalnego formatowania query string
            query_string = urllib.parse.urlencode(params, doseq=True)
            signature = hmac.new(
                self.secret_key.encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            return signature
        except Exception as e:
            st.error(f"❌ Signing error: {e}")
            return ""
    
    def _make_private_request(self, endpoint: str, params: dict = None) -> Optional[dict]:
        """Wysyła autoryzowany request do MEXC tylko do odczytu"""
        try:
            timestamp = int(time.time() * 1000)
            params = params or {}
            params.update({
                'timestamp': timestamp,
                'recvWindow': 5000
            })
            
            # Usuń None values
            params = {k: v for k, v in params.items() if v is not None}
            
            signature = self._sign_request(params)
            if not signature:
                return None
                
            params['signature'] = signature
            
            headers = {
                'X-MEXC-APIKEY': self.api_key,
                'Content-Type': 'application/json'
            }
            
            response = requests.get(
                f"{self.base_url}{endpoint}",
                params=params,
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"❌ API Error {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            st.error(f"❌ Request failed: {e}")
            return None
    
    def get_account_balance(self) -> Dict[str, float]:
        """Pobiera pełne portfolio z MEXC"""
        data = self._make_private_request('/api/v3/account')
        balances = {}
        
        if data and 'balances' in data:
            for balance in data['balances']:
                asset = balance['asset']
                free = float(balance['free'])
                locked = float(balance['locked'])
                total = free + locked
                
                if total > 0:  # Tylko tokeny z dodatnim balansem
                    balances[asset] = {
                        'free': free,
                        'locked': locked,
                        'total': total
                    }
        
        return balances
    
    def get_current_orders(self) -> List[dict]:
        """Pobiera aktualne zlecenia - tylko odczyt"""
        data = self._make_private_request('/api/v3/openOrders')
        return data or []
    
    def test_connection(self) -> bool:
        """Testuje połączenie z API"""
        data = self._make_private_request('/api/v3/account')
        return data is not None

class CryptoPortfolioTracker:
    def __init__(self):
        self.fee_rate = 0.00025
        self.tokens_to_track = [
            'BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'XRP', 'DOT', 'DOGE', 'AVAX', 'LTC',
            'LINK', 'ATOM', 'XLM', 'BCH', 'ALGO', 'FIL', 'ETC', 'XTZ', 'AAVE', 'COMP',
            'UNI', 'CRV', 'SUSHI', 'YFI', 'SNX', '1INCH', 'ZRX', 'TRX', 'VET', 'ONE',
            'CELO', 'RSR', 'NKN', 'STORJ', 'DODO', 'KAVA', 'RUNE', 'SAND', 'MANA', 'ENJ',
            'CHZ', 'ALICE', 'NEAR', 'ARB', 'OP', 'APT', 'SUI', 'SEI', 'INJ', 'RENDER'
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
            st.session_state.prices = self.get_current_prices()
        if 'tracking' not in st.session_state:
            st.session_state.tracking = False
        if 'baseline_equivalents' not in st.session_state:
            st.session_state.baseline_equivalents = {}
        if 'orders' not in st.session_state:
            st.session_state.orders = []
    
    def get_current_prices(self) -> Dict[str, Dict]:
        """Pobiera aktualne ceny bid/ask"""
        prices = {}
        try:
            response = requests.get("https://api.mexc.com/api/v3/ticker/bookTicker", timeout=10)
            if response.status_code == 200:
                data = response.json()
                usdt_pairs = {item['symbol']: item for item in data if item['symbol'].endswith('USDT')}
                
                for token in self.tokens_to_track:
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
        except Exception as e:
            st.error(f"❌ Błąd pobierania cen: {e}")
        return prices
    
    def calculate_equivalent(self, from_token: str, to_token: str, quantity: float) -> float:
        """Oblicza ekwiwalent między tokenami"""
        if from_token == to_token:
            return quantity * (1 - self.fee_rate)
        
        prices = st.session_state.prices
        if not prices:
            return 0.0
            
        if from_token == 'USDT':
            if to_token not in prices:
                return 0.0
            ask_price = prices[to_token]['ask']
            if ask_price <= 0:
                return 0.0
            equivalent = (quantity / ask_price) * (1 - self.fee_rate)
            return equivalent
        
        elif to_token == 'USDT':
            if from_token not in prices:
                return 0.0
            bid_price = prices[from_token]['bid']
            if bid_price <= 0:
                return 0.0
            equivalent = quantity * bid_price * (1 - self.fee_rate)
            return equivalent
        
        else:
            if from_token not in prices or to_token not in prices:
                return 0.0
                
            bid_price = prices[from_token]['bid']
            ask_price = prices[to_token]['ask']
            
            if bid_price <= 0 or ask_price <= 0:
                return 0.0
                
            usdt_value = quantity * bid_price * (1 - self.fee_rate)
            equivalent = (usdt_value / ask_price) * (1 - self.fee_rate)
            return equivalent

    def setup_api_credentials(self):
        """Setup bezpiecznego wprowadzania kluczy API"""
        st.sidebar.header("🔐 API Configuration")
        
        st.sidebar.info("""
        **Required API Permissions:**
        - ✅ Read Account Info  
        - ✅ Read Orders
        - ❌ Trading (NOT required)
        """)
        
        with st.sidebar.form("api_config"):
            api_key = st.text_input("MEXC API Key", type="password", help="Your MEXC API Key with READ permissions")
            secret_key = st.text_input("MEXC Secret Key", type="password", help="Your MEXC Secret Key")
            
            if st.form_submit_button("🔗 Connect to MEXC"):
                if api_key and secret_key:
                    with st.spinner("Testing API connection..."):
                        try:
                            # Test connection first
                            test_api = MexcPrivateAPI(api_key, secret_key)
                            if test_api.test_connection():
                                st.session_state.mexc_api = test_api
                                st.session_state.api_initialized = True
                                
                                # Load initial data
                                balance = st.session_state.mexc_api.get_account_balance()
                                orders = st.session_state.mexc_api.get_current_orders()
                                
                                if balance:
                                    st.session_state.portfolio = balance
                                    st.session_state.orders = orders
                                    st.session_state.tracking = True
                                    self.initialize_baseline_from_portfolio()
                                    st.success("✅ Connected to MEXC API - Read Only")
                                else:
                                    st.error("❌ Failed to load portfolio data")
                            else:
                                st.error("❌ API connection failed - check your keys and permissions")
                        except Exception as e:
                            st.error(f"❌ Connection error: {str(e)}")
                else:
                    st.error("❌ Please enter both API keys")

    def initialize_baseline_from_portfolio(self):
        """Inicjalizuje baseline z aktualnego portfolio"""
        baseline = {}
        prices = st.session_state.prices
        
        for asset, balance_info in st.session_state.portfolio.items():
            total_amount = balance_info['total']
            
            if asset == 'USDT':
                # Dla USDT: baseline to ile każdego tokena można kupić
                for target_token in self.tokens_to_track:
                    equivalent = self.calculate_equivalent('USDT', target_token, total_amount)
                    baseline[f"USDT_{target_token}"] = equivalent
            else:
                # Dla tokenów: baseline to ekwiwalenty w innych tokenach
                for target_token in self.tokens_to_track:
                    if target_token != asset:
                        equivalent = self.calculate_equivalent(asset, target_token, total_amount)
                        baseline[f"{asset}_{target_token}"] = equivalent
        
        st.session_state.baseline_equivalents = baseline
        st.session_state.baseline_time = datetime.now()
        st.session_state.initial_portfolio_value = self.calculate_portfolio_value()

    def calculate_portfolio_value(self) -> float:
        """Oblicza całkowitą wartość portfolio w USDT"""
        total_value = 0
        prices = st.session_state.prices
        
        for asset, balance_info in st.session_state.portfolio.items():
            if asset == 'USDT':
                total_value += balance_info['total']
            elif asset in prices:
                total_value += balance_info['total'] * prices[asset]['bid']
        
        return total_value

    def render_portfolio_overview(self):
        """Pokazuje aktualne portfolio z MEXC"""
        st.header("💰 Live Portfolio from MEXC")
        
        if not st.session_state.portfolio:
            st.info("📊 No portfolio data available")
            return
        
        prices = st.session_state.prices
        portfolio_data = []
        total_value = 0
        
        for asset, balance in st.session_state.portfolio.items():
            if asset == 'USDT':
                value = balance['total']
                total_value += value
                portfolio_data.append({
                    'Asset': asset,
                    'Free': f"{balance['free']:,.2f}",
                    'Locked': f"{balance['locked']:,.2f}",
                    'Total': f"{balance['total']:,.2f}",
                    'Value USDT': f"{value:,.2f}",
                    'Price': "1.0000"
                })
            elif asset in prices:
                value = balance['total'] * prices[asset]['bid']
                total_value += value
                portfolio_data.append({
                    'Asset': asset,
                    'Free': f"{balance['free']:.6f}",
                    'Locked': f"{balance['locked']:.6f}",
                    'Total': f"{balance['total']:.6f}",
                    'Value USDT': f"{value:,.2f}",
                    'Price': f"{prices[asset]['bid']:.4f}"
                })
        
        if portfolio_data:
            df = pd.DataFrame(portfolio_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Value", f"${total_value:,.2f}")
            with col2:
                if 'initial_portfolio_value' in st.session_state:
                    profit_loss = total_value - st.session_state.initial_portfolio_value
                    profit_pct = (profit_loss / st.session_state.initial_portfolio_value * 100) if st.session_state.initial_portfolio_value > 0 else 0
                    st.metric("P&L", f"${profit_loss:+,.2f}", f"{profit_pct:+.2f}%")
            with col3:
                st.metric("Assets", len(portfolio_data))

    def render_orders_overview(self):
        """Pokazuje aktualne zlecenia"""
        if not st.session_state.orders:
            return
            
        st.header("📋 Active Orders")
        
        orders_data = []
        for order in st.session_state.orders:
            orders_data.append({
                'Symbol': order.get('symbol', ''),
                'Side': order.get('side', ''),
                'Type': order.get('type', ''),
                'Quantity': float(order.get('origQty', 0)),
                'Price': float(order.get('price', 0)),
                'Status': order.get('status', '')
            })
        
        if orders_data:
            df = pd.DataFrame(orders_data)
            st.dataframe(df, use_container_width=True, hide_index=True)

    def render_swap_matrix(self):
        """Renderuje matrycę swapów dla każdego tokena w portfolio"""
        if not st.session_state.tracking:
            st.info("💡 Connect to MEXC API to see swap matrix")
            return
            
        st.header("🎯 Swap Opportunity Matrix")
        
        for asset, balance_info in st.session_state.portfolio.items():
            if asset == 'USDT' or balance_info['total'] <= 0:
                continue
                
            self.render_asset_matrix(asset, balance_info['total'])
    
    def render_asset_matrix(self, asset: str, amount: float):
        """Renderuje matrycę dla konkretnego assetu"""
        with st.expander(f"🔷 {asset} - {amount:.6f}"):
            matrix_data = []
            prices = st.session_state.prices
            
            for target_token in self.tokens_to_track:
                if target_token == asset:
                    continue
                    
                current_equivalent = self.calculate_equivalent(asset, target_token, amount)
                baseline_key = f"{asset}_{target_token}"
                baseline_equivalent = st.session_state.baseline_equivalents.get(baseline_key, current_equivalent)
                
                if baseline_equivalent > 0:
                    change_pct = ((current_equivalent - baseline_equivalent) / baseline_equivalent * 100)
                else:
                    change_pct = 0
                    
                usdt_value = self.calculate_equivalent(asset, 'USDT', amount)
                
                # Określ status na podstawie zmiany
                status = "🔴"
                if change_pct >= 2.0:
                    status = "🟢"
                elif change_pct >= 0.5:
                    status = "🟡"
                
                matrix_data.append({
                    'Target': target_token,
                    'Equivalent': current_equivalent,
                    'Baseline': baseline_equivalent,
                    'Change %': change_pct,
                    'USDT Value': usdt_value,
                    'Status': status
                })
            
            if matrix_data:
                df = pd.DataFrame(matrix_data)
                df = df.sort_values('Change %', ascending=False)
                
                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        'Equivalent': st.column_config.NumberColumn(format="%.6f"),
                        'Baseline': st.column_config.NumberColumn(format="%.6f"),
                        'Change %': st.column_config.NumberColumn(format="%+.2f%%"),
                        'USDT Value': st.column_config.NumberColumn(format="%.2f"),
                        'Status': st.column_config.TextColumn()
                    }
                )

    def refresh_data(self):
        """Odświeża dane z MEXC"""
        if st.session_state.api_initialized:
            new_portfolio = st.session_state.mexc_api.get_account_balance()
            new_orders = st.session_state.mexc_api.get_current_orders()
            
            if new_portfolio:
                st.session_state.portfolio = new_portfolio
            if new_orders:
                st.session_state.orders = new_orders
                
            return True
        return False

    def auto_refresh_data(self):
        """Automatyczne odświeżanie danych"""
        if st.session_state.tracking:
            current_time = datetime.now()
            
            # Odśwież ceny co 5 sekund
            if 'last_price_refresh' not in st.session_state:
                st.session_state.last_price_refresh = current_time
            
            if (current_time - st.session_state.last_price_refresh).seconds >= 5:
                new_prices = self.get_current_prices()
                if new_prices:
                    st.session_state.prices = new_prices
                    st.session_state.last_price_refresh = current_time
            
            # Odśwież portfolio co 30 sekund
            if 'last_portfolio_refresh' not in st.session_state:
                st.session_state.last_portfolio_refresh = current_time
            
            if (current_time - st.session_state.last_portfolio_refresh).seconds >= 30:
                if self.refresh_data():
                    return True
                st.session_state.last_portfolio_refresh = current_time
        
        return False

    def render_control_panel(self):
        """Panel kontrolny"""
        st.sidebar.header("🎮 Control Panel")
        
        if st.session_state.api_initialized:
            status = "🟢 LIVE" if st.session_state.tracking else "🔴 STOPPED"
            st.sidebar.metric("Status", status)
            
            if st.session_state.prices:
                price_values = list(st.session_state.prices.values())
                if price_values and 'last_update' in price_values[0]:
                    last_update = price_values[0]['last_update']
                    st.sidebar.caption(f"📊 Prices: {last_update.strftime('%H:%M:%S')}")
            
            if st.button("🔄 Refresh Data", use_container_width=True):
                self.refresh_data()
                st.session_state.prices = self.get_current_prices()
                st.rerun()
            
            if st.button("📊 Reset Baseline", use_container_width=True):
                self.initialize_baseline_from_portfolio()
                st.success("✅ Baseline reset")
            
            st.sidebar.markdown("---")
            
            if 'baseline_time' in st.session_state:
                st.sidebar.info(f"⏰ Baseline: {st.session_state.baseline_time.strftime('%H:%M:%S')}")

    def run(self):
        """Główna pętla aplikacji"""
        st.title("📊 Crypto Portfolio Tracker - MEXC")
        st.markdown("---")
        
        # Inicjalizacja
        self.init_session_state()
        
        # Layout
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # API Setup (jeśli nie połączone)
            if not st.session_state.api_initialized:
                st.info("🔐 Configure your MEXC API keys in the sidebar to start tracking")
            
            # Główne komponenty
            self.render_portfolio_overview()
            
            if st.session_state.api_initialized:
                self.render_orders_overview()
                st.markdown("---")
                self.render_swap_matrix()
        
        with col2:
            self.setup_api_credentials()
            if st.session_state.api_initialized:
                self.render_control_panel()
        
        # Auto refresh
        if self.auto_refresh_data():
            st.rerun()

# Uruchomienie aplikacji
if __name__ == "__main__":
    app = CryptoPortfolioTracker()
    app.run()