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
    page_title="Crypto Swap Matrix - MEXC",
    page_icon="🔄",
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
                
                if total > 0:
                    balances[asset] = {
                        'free': free,
                        'locked': locked,
                        'total': total
                    }
        
        return balances
    
    def test_connection(self) -> bool:
        """Testuje połączenie z API"""
        data = self._make_private_request('/api/v3/account')
        return data is not None

class SimpleSwapMatrix:
    def __init__(self):
        self.fee_rate = 0.00025
        self.swap_threshold = 0.5
        self.tokens_to_track = [
            'BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'XRP', 'DOT', 'DOGE', 'AVAX', 'LTC',
            'LINK', 'ATOM', 'XLM', 'BCH', 'ALGO', 'FIL', 'ETC', 'XTZ', 'AAVE', 'COMP',
            'UNI', 'CRV', 'SUSHI', 'YFI', 'SNX', '1INCH', 'ZRX', 'TRX', 'VET', 'ONE',
            'CELO', 'RSR', 'NKN', 'STORJ', 'DODO', 'KAVA', 'RUNE', 'SAND', 'MANA', 'ENJ',
            'CHZ', 'ALICE', 'NEAR', 'ARB', 'OP', 'APT', 'SUI', 'SEI', 'INJ', 'RENDER', 'MX'
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
        if 'current_asset' not in st.session_state:
            st.session_state.current_asset = None
        if 'baseline_equivalents' not in st.session_state:
            st.session_state.baseline_equivalents = {}
        if 'top_equivalents' not in st.session_state:
            st.session_state.top_equivalents = {}
        if 'last_swap_time' not in st.session_state:
            st.session_state.last_swap_time = datetime.now()
        
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

    def find_current_asset(self):
        """Znajduje aktualny asset (najwyższa wartość w USDT)"""
        max_value = 0
        current_asset = None
        prices = st.session_state.prices
        
        for asset, balance_info in st.session_state.portfolio.items():
            if asset in prices:
                value = balance_info['total'] * prices[asset]['bid']
            elif asset == 'USDT':
                value = balance_info['total']
            else:
                continue
                
            if value > max_value:
                max_value = value
                current_asset = asset
        
        return current_asset

    def detect_asset_change(self):
        """Wykrywa zmianę assetu i aktualizuje baseline/top"""
        current_asset = self.find_current_asset()
        
        # Sprawdź czy asset się zmienił
        if current_asset != st.session_state.current_asset:
            old_asset = st.session_state.current_asset
            st.session_state.current_asset = current_asset
            
            if current_asset:
                # Nowy asset - zainicjuj baseline i top
                self.initialize_asset_tracking(current_asset)
                st.success(f"🔄 Asset changed: {old_asset} → {current_asset}")
                return True
        
        return False

    def initialize_asset_tracking(self, asset: str):
        """Inicjalizuje śledzenie dla nowego assetu"""
        if asset not in st.session_state.portfolio:
            return
            
        amount = st.session_state.portfolio[asset]['total']
        
        # Oblicz baseline equivalents
        baseline = {}
        top = {}
        
        for token in self.tokens_to_track:
            if token != asset:
                equivalent = self.calculate_equivalent(asset, token, amount)
                baseline[token] = equivalent
                top[token] = equivalent  # Początkowo top = baseline
        
        st.session_state.baseline_equivalents = baseline
        st.session_state.top_equivalents = top
        st.session_state.last_swap_time = datetime.now()

    def update_top_equivalents(self):
        """Aktualizuje top equivalents dla aktualnego assetu"""
        if not st.session_state.current_asset:
            return
            
        current_asset = st.session_state.current_asset
        amount = st.session_state.portfolio[current_asset]['total']
        
        for token in self.tokens_to_track:
            if token != current_asset:
                current_equivalent = self.calculate_equivalent(current_asset, token, amount)
                current_top = st.session_state.top_equivalents.get(token, 0)
                
                if current_equivalent > current_top:
                    st.session_state.top_equivalents[token] = current_equivalent

    def setup_api_credentials(self):
        """Setup bezpiecznego wprowadzania kluczy API"""
        st.sidebar.header("🔐 API Configuration")
        
        with st.sidebar.form("api_config"):
            api_key = st.text_input("MEXC API Key", type="password")
            secret_key = st.text_input("MEXC Secret Key", type="password")
            
            if st.form_submit_button("🔗 Connect to MEXC"):
                if api_key and secret_key:
                    with st.spinner("Testing API connection..."):
                        try:
                            test_api = MexcPrivateAPI(api_key, secret_key)
                            if test_api.test_connection():
                                st.session_state.mexc_api = test_api
                                st.session_state.api_initialized = True
                                
                                # Load initial data
                                balance = st.session_state.mexc_api.get_account_balance()
                                if balance:
                                    st.session_state.portfolio = balance
                                    st.session_state.tracking = True
                                    
                                    # Znajdź początkowy asset
                                    initial_asset = self.find_current_asset()
                                    if initial_asset:
                                        st.session_state.current_asset = initial_asset
                                        self.initialize_asset_tracking(initial_asset)
                                    
                                    st.success("✅ Connected to MEXC API")
                                else:
                                    st.error("❌ Failed to load portfolio data")
                            else:
                                st.error("❌ API connection failed")
                        except Exception as e:
                            st.error(f"❌ Connection error: {str(e)}")
                else:
                    st.error("❌ Please enter both API keys")

    def render_portfolio_overview(self):
        """Pokazuje aktualne portfolio"""
        st.header("💰 Live Portfolio")
        
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
                    'Total': f"{balance['total']:,.2f}",
                    'Value USDT': f"{value:,.2f}",
                    'Price': "1.0000"
                })
            elif asset in prices:
                value = balance['total'] * prices[asset]['bid']
                total_value += value
                portfolio_data.append({
                    'Asset': asset,
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
                st.metric("Assets", len(portfolio_data))
            with col3:
                if st.session_state.current_asset:
                    current_value = 0
                    if st.session_state.current_asset == 'USDT':
                        current_value = st.session_state.portfolio['USDT']['total']
                    elif st.session_state.current_asset in prices:
                        current_value = st.session_state.portfolio[st.session_state.current_asset]['total'] * prices[st.session_state.current_asset]['bid']
                    st.metric("Current Asset", st.session_state.current_asset, f"${current_value:,.2f}")

    def render_swap_matrix(self):
        """Renderuje matrycę swap opportunities"""
        if not st.session_state.current_asset:
            st.info("💡 Connect to MEXC to see swap matrix")
            return
            
        current_asset = st.session_state.current_asset
        amount = st.session_state.portfolio[current_asset]['total']
        
        if current_asset == 'USDT':
            st.header("🛒 Purchase Matrix - USDT")
            st.info("Baseline: Last token → USDT swap time")
        else:
            st.header("🎯 Swap Matrix - " + current_asset)
            st.info("Baseline: Round start | Top: Historical max")
        
        matrix_data = []
        
        for target_token in self.tokens_to_track:
            if target_token == current_asset:
                continue
                
            current_equivalent = self.calculate_equivalent(current_asset, target_token, amount)
            
            if current_asset == 'USDT':
                # Dla USDT: porównanie z baseline
                baseline_equivalent = st.session_state.baseline_equivalents.get(target_token, current_equivalent)
                if baseline_equivalent > 0:
                    change_pct = ((current_equivalent - baseline_equivalent) / baseline_equivalent * 100)
                else:
                    change_pct = 0
                
                status = "🟢" if change_pct >= self.swap_threshold else "🔴"
                
                matrix_data.append({
                    'Target': target_token,
                    'Current': current_equivalent,
                    'Baseline': baseline_equivalent,
                    'Change %': change_pct,
                    'Status': status
                })
            else:
                # Dla tokenów: porównanie z top equivalent
                top_equivalent = st.session_state.top_equivalents.get(target_token, current_equivalent)
                if top_equivalent > 0:
                    gain_pct = ((current_equivalent - top_equivalent) / top_equivalent * 100)
                else:
                    gain_pct = 0
                
                status = "🟢 SWAP" if gain_pct >= self.swap_threshold else "🔴"
                
                matrix_data.append({
                    'Target': target_token,
                    'Current': current_equivalent,
                    'Top': top_equivalent,
                    'Gain %': gain_pct,
                    'Status': status
                })
        
        if matrix_data:
            if current_asset == 'USDT':
                df = pd.DataFrame(matrix_data)
                df = df.sort_values('Change %', ascending=False)
                
                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        'Current': st.column_config.NumberColumn(format="%.6f"),
                        'Baseline': st.column_config.NumberColumn(format="%.6f"),
                        'Change %': st.column_config.NumberColumn(format="%+.2f%%"),
                        'Status': st.column_config.TextColumn()
                    }
                )
            else:
                df = pd.DataFrame(matrix_data)
                df = df.sort_values('Gain %', ascending=False)
                
                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        'Current': st.column_config.NumberColumn(format="%.6f"),
                        'Top': st.column_config.NumberColumn(format="%.6f"),
                        'Gain %': st.column_config.NumberColumn(format="%+.2f%%"),
                        'Status': st.column_config.TextColumn()
                    }
                )
            
            # Show best opportunities
            if current_asset == 'USDT':
                best_ops = [op for op in matrix_data if op['Change %'] >= self.swap_threshold]
            else:
                best_ops = [op for op in matrix_data if op['Gain %'] >= self.swap_threshold]
                
            if best_ops:
                st.subheader("💎 Best Opportunities")
                for op in best_ops[:3]:  # Top 3
                    if current_asset == 'USDT':
                        st.success(f"**{op['Target']}**: +{op['Change %']:.2f}% from baseline")
                    else:
                        st.success(f"**{op['Target']}**: +{op['Gain %']:.2f}% from top")

    def auto_refresh_data(self):
        """Automatyczne odświeżanie danych"""
        if st.session_state.tracking and st.session_state.api_initialized:
            current_time = datetime.now()
            
            # Odśwież ceny co 3 sekundy
            if 'last_price_refresh' not in st.session_state:
                st.session_state.last_price_refresh = current_time
            
            if (current_time - st.session_state.last_price_refresh).seconds >= 3:
                st.session_state.prices = self.get_current_prices()
                st.session_state.last_price_refresh = current_time
            
            # Aktualizuj top equivalents co 5 sekund
            if 'last_top_update' not in st.session_state:
                st.session_state.last_top_update = current_time
            
            if (current_time - st.session_state.last_top_update).seconds >= 5:
                self.update_top_equivalents()
                st.session_state.last_top_update = current_time
            
            # Sprawdzaj zmianę assetu co 10 sekund
            if 'last_asset_check' not in st.session_state:
                st.session_state.last_asset_check = current_time
            
            if (current_time - st.session_state.last_asset_check).seconds >= 10:
                if self.detect_asset_change():
                    return True  # Wykryto zmianę assetu - potrzeba rerun
                st.session_state.last_asset_check = current_time
            
            # Odśwież portfolio co 15 sekund
            if 'last_portfolio_refresh' not in st.session_state:
                st.session_state.last_portfolio_refresh = current_time
            
            if (current_time - st.session_state.last_portfolio_refresh).seconds >= 15:
                new_portfolio = st.session_state.mexc_api.get_account_balance()
                if new_portfolio:
                    st.session_state.portfolio = new_portfolio
                    st.session_state.last_portfolio_refresh = current_time
        
        return False

    def render_control_panel(self):
        """Panel kontrolny"""
        st.sidebar.header("🎮 Control Panel")
        
        if st.session_state.api_initialized:
            status = "🟢 LIVE" if st.session_state.tracking else "🔴 STOPPED"
            st.sidebar.metric("Status", status)
            
            if st.session_state.current_asset:
                st.sidebar.metric("Current Asset", st.session_state.current_asset)
            
            if st.session_state.prices:
                price_values = list(st.session_state.prices.values())
                if price_values and 'last_update' in price_values[0]:
                    last_update = price_values[0]['last_update']
                    st.sidebar.caption(f"📊 Prices: {last_update.strftime('%H:%M:%S')}")
            
            if st.sidebar.button("🔄 Refresh Now", use_container_width=True):
                new_portfolio = st.session_state.mexc_api.get_account_balance()
                if new_portfolio:
                    st.session_state.portfolio = new_portfolio
                st.session_state.prices = self.get_current_prices()
                st.rerun()

    def run(self):
        """Główna pętla aplikacji"""
        st.title("🔄 Crypto Swap Matrix - MEXC")
        st.markdown("---")
        
        # Inicjalizacja
        self.init_session_state()
        
        # Layout
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if not st.session_state.api_initialized:
                st.info("🔐 Configure your MEXC API keys in the sidebar to start")
            
            self.render_portfolio_overview()
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
    app = SimpleSwapMatrix()
    app.run()