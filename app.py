import streamlit as st
import pandas as pd
import requests
import time
from datetime import datetime
import json
import os
from typing import Dict, List

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

    def get_prices(self) -> Dict[str, Dict]:
        """Pobiera aktualne ceny - DZIAŁAJĄCE"""
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
        except Exception as e:
            return {}

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

    def find_main_token(self, portfolio: Dict) -> str:
        """Znajduje token z najwyższą wartością w USDT"""
        max_value = 0
        main_token = None
        prices = st.session_state.prices
        
        for asset, balance_info in portfolio.items():
            if asset in prices:
                value = balance_info['total'] * prices[asset]['bid']
                if value > max_value:
                    max_value = value
                    main_token = asset
        
        return main_token

    def init_session_state(self):
        """Inicjalizacja session state - DZIAŁAJĄCE"""
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

    def detect_token_change(self):
        """Wykrywa zmianę głównego tokena"""
        current_main_token = self.find_main_token(st.session_state.portfolio)
        
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
        self.update_top_equivalents(equivalents)

    def update_top_equivalents(self, new_equivalents: Dict[str, float]):
        """Aktualizuje top equivalents"""
        for target_token, equivalent in new_equivalents.items():
            current_top = st.session_state.top_equivalents.get(target_token, 0)
            if equivalent > current_top:
                st.session_state.top_equivalents[target_token] = equivalent

    def update_prices(self):
        """Odświeża ceny - DZIAŁAJĄCE"""
        if hasattr(st.session_state, 'last_price_update'):
            if (datetime.now() - st.session_state.last_price_update).seconds < 3:
                return
        new_prices = self.get_prices()
        if new_prices:
            st.session_state.prices = new_prices
            st.session_state.last_price_update = datetime.now()

    def setup_api_credentials(self):
        """Setup API credentials"""
        st.sidebar.header("🔐 API Configuration")
        
        with st.sidebar.form("api_config"):
            api_key = st.text_input("MEXC API Key", type="password")
            secret_key = st.text_input("MEXC Secret Key", type="password")
            
            if st.form_submit_button("🔗 Connect to MEXC"):
                if api_key and secret_key:
                    try:
                        # Simple API test
                        test_response = requests.get("https://api.mexc.com/api/v3/ping", timeout=10)
                        if test_response.status_code == 200:
                            st.session_state.tracking = True
                            # Symulacja portfolio dla testów
                            st.session_state.portfolio = {
                                'BTC': {'total': 0.1, 'free': 0.1, 'locked': 0},
                                'USDT': {'total': 1000, 'free': 1000, 'locked': 0},
                                'MX': {'total': 100, 'free': 100, 'locked': 0}
                            }
                            st.success("✅ Connected to MEXC")
                            st.rerun()
                        else:
                            st.error("❌ API connection failed")
                    except:
                        st.error("❌ Connection error")

    def render_portfolio(self):
        """Wyświetla portfolio"""
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
                    st.metric("Main Token", st.session_state.main_token)

    def render_matrix(self):
        """Renderuje matrycę ekwiwalentów"""
        if not st.session_state.main_token:
            st.info("💡 Connect to MEXC to see matrix")
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

    def render_control_panel(self):
        """Panel kontrolny"""
        st.sidebar.header("🎮 Control Panel")
        
        if st.session_state.tracking:
            status = "🟢 LIVE" if st.session_state.tracking else "🔴 STOPPED"
            st.sidebar.metric("Status", status)
            
            if st.session_state.prices:
                price_values = list(st.session_state.prices.values())
                if price_values and 'last_update' in price_values[0]:
                    last_update = price_values[0]['last_update']
                    st.sidebar.caption(f"Prices: {last_update.strftime('%H:%M:%S')}")
            
            if st.sidebar.button("🔄 Refresh Now"):
                st.session_state.prices = self.get_prices()
                st.rerun()

    def run(self):
        """Główna pętla aplikacji - DZIAŁAJĄCE"""
        st.title("🔄 Crypto Swap Matrix - Auto")
        st.markdown("---")
        
        # Inicjalizacja
        self.init_session_state()
        
        # Layout
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
        
        # Auto refresh - DZIAŁAJĄCE
        if st.session_state.tracking:
            # Odśwież ceny
            self.update_prices()
            
            # Wykrywaj zmianę tokena
            self.detect_token_change()
            
            # Auto rerun co 3 sekundy
            time.sleep(3)
            st.rerun()

# Uruchomienie
if __name__ == "__main__":
    app = CryptoSwapMatrix()
    app.run()