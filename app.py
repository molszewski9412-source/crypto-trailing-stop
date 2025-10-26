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
from dataclasses import dataclass, asdict
import uuid

# ================== Konfiguracja strony ==================
st.set_page_config(
    page_title="Crypto Trader - MEXC",
    page_icon="🤖",
    layout="wide"
)

# ================== Data Classes ==================
@dataclass
class Trade:
    timestamp: datetime
    from_token: str
    to_token: str
    from_amount: float
    to_amount: float
    prices_at_trade: Dict[str, Dict]
    equivalent_at_trade: Dict[str, float]

@dataclass
class TradingRound:
    round_id: str
    start_time: datetime
    start_prices: Dict[str, Dict]
    baseline_equivalents: Dict[str, float]
    top_equivalents: Dict[str, float]
    current_token: str
    current_amount: float
    trade_history: List[Trade]
    status: str

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
    
    def get_current_orders(self) -> List[dict]:
        """Pobiera aktualne zlecenia - tylko odczyt"""
        data = self._make_private_request('/api/v3/openOrders')
        return data or []
    
    def test_connection(self) -> bool:
        """Testuje połączenie z API"""
        data = self._make_private_request('/api/v3/account')
        return data is not None

class CryptoAutoTrader:
    def __init__(self):
        self.fee_rate = 0.00025
        self.swap_threshold = 0.5
        self.target_profit = 0.02
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
            st.session_state.prices = self.get_current_prices_with_timestamp()
        if 'current_round' not in st.session_state:
            st.session_state.current_round = None
        if 'tracking' not in st.session_state:
            st.session_state.tracking = False
        
    def get_current_prices_with_timestamp(self) -> Dict[str, Dict]:
        """Pobiera aktualne ceny bid/ask z timestampem"""
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
                                    'timestamp': datetime.now()
                                }
                        except:
                            continue
        except Exception as e:
            st.error(f"❌ Błąd pobierania cen: {e}")
        return prices
    
    def calculate_equivalent_with_prices(self, from_token: str, to_token: str, quantity: float, prices: Dict) -> float:
        """Oblicza ekwiwalent między tokenami używając konkretnych cen"""
        if from_token == to_token:
            return quantity * (1 - self.fee_rate)
        
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

    def calculate_all_equivalents(self, from_token: str, quantity: float, prices: Dict) -> Dict[str, float]:
        """Oblicza ekwiwalenty dla wszystkich tokenów"""
        equivalents = {}
        for target_token in self.tokens_to_track:
            if target_token != from_token:
                equivalent = self.calculate_equivalent_with_prices(from_token, target_token, quantity, prices)
                equivalents[target_token] = equivalent
        return equivalents

    def find_main_trading_token(self, portfolio: Dict) -> str:
        """Znajduje token z najwyższą wartością w USDT (oprócz USDT)"""
        max_value = 0
        main_token = None
        prices = st.session_state.prices
        
        for asset, balance_info in portfolio.items():
            if asset == 'USDT':
                continue
                
            if asset in prices:
                value = balance_info['total'] * prices[asset]['bid']
                if value > max_value:
                    max_value = value
                    main_token = asset
        
        return main_token

    def start_new_round(self, selected_token: str):
        """Rozpoczyna nową rundę handlową"""
        current_prices = self.get_current_prices_with_timestamp()
        
        if 'USDT' not in st.session_state.portfolio:
            st.error("❌ No USDT in portfolio to start round")
            return
            
        usdt_amount = st.session_state.portfolio['USDT']['total']
        
        # Oblicz baseline equivalents
        baseline = {}
        for token in self.tokens_to_track:
            equivalent = self.calculate_equivalent_with_prices('USDT', token, usdt_amount, current_prices)
            baseline[token] = equivalent
        
        # Stwórz nową rundę
        new_round = TradingRound(
            round_id=f"round_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            start_time=datetime.now(),
            start_prices=current_prices,
            baseline_equivalents=baseline,
            top_equivalents=baseline.copy(),
            current_token=selected_token,
            current_amount=baseline[selected_token],
            trade_history=[],
            status='ACTIVE'
        )
        
        st.session_state.current_round = new_round
        st.session_state.tracking = True
        st.success(f"✅ New round started: USDT → {selected_token}")

    def detect_and_process_swap(self):
        """Wykrywa manualne swapy i przetwarza je"""
        if not st.session_state.current_round or st.session_state.current_round.status != 'ACTIVE':
            return False
            
        current_round = st.session_state.current_round
        
        # Pobierz aktualne portfolio
        current_portfolio = st.session_state.mexc_api.get_account_balance()
        if not current_portfolio:
            return False
            
        # Znajdź główny token
        new_main_token = self.find_main_trading_token(current_portfolio)
        if not new_main_token or new_main_token == current_round.current_token:
            return False
            
        # SWAP WYKRYTY
        swap_timestamp = datetime.now()
        prices_at_swap = self.get_current_prices_with_timestamp()
        new_amount = current_portfolio[new_main_token]['total']
        
        # Oblicz ekwiwalenty w momencie swapu
        equivalents_at_swap = self.calculate_all_equivalents(new_main_token, new_amount, prices_at_swap)
        
        # Stwórz rekord trade
        trade = Trade(
            timestamp=swap_timestamp,
            from_token=current_round.current_token,
            to_token=new_main_token,
            from_amount=current_round.current_amount,
            to_amount=new_amount,
            prices_at_trade=prices_at_swap,
            equivalent_at_trade=equivalents_at_swap
        )
        
        # Aktualizuj top equivalents
        self.update_top_equivalents(trade)
        
        # Zaktualizuj rundę
        current_round.current_token = new_main_token
        current_round.current_amount = new_amount
        current_round.trade_history.append(trade)
        
        st.success(f"🔄 Swap detected: {trade.from_token} → {trade.to_token}")
        return True

    def update_top_equivalents(self, trade: Trade):
        """Aktualizuje top equivalents na podstawie trade"""
        current_round = st.session_state.current_round
        
        # Dla nowego tokena: top = dokładna ilość z trade
        current_round.top_equivalents[trade.to_token] = trade.to_amount
        
        # Dla innych tokenów: użyj ekwiwalentów z momentu trade
        for token, equivalent in trade.equivalent_at_trade.items():
            current_top = current_round.top_equivalents.get(token, 0)
            if equivalent > current_top:
                current_round.top_equivalents[token] = equivalent

    def end_current_round(self):
        """Kończy aktualną rundę"""
        if st.session_state.current_round:
            st.session_state.current_round.status = 'COMPLETED'
            st.session_state.tracking = False
            st.success("✅ Round completed")

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
                                    st.success("✅ Connected to MEXC API - Read Only")
                                else:
                                    st.error("❌ Failed to load portfolio data")
                            else:
                                st.error("❌ API connection failed")
                        except Exception as e:
                            st.error(f"❌ Connection error: {str(e)}")
                else:
                    st.error("❌ Please enter both API keys")

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
                st.metric("Assets", len(portfolio_data))
            with col3:
                if st.session_state.current_round:
                    st.metric("Current Round", st.session_state.current_round.current_token)

    def render_round_control(self):
        """Panel kontrolny rundy"""
        st.sidebar.header("🎮 Round Control")
        
        if st.session_state.api_initialized and 'USDT' in st.session_state.portfolio:
            if not st.session_state.current_round or st.session_state.current_round.status == 'COMPLETED':
                # Start new round
                available_tokens = [t for t in self.tokens_to_track if t != 'USDT']
                selected_token = st.sidebar.selectbox("Select token to start round:", available_tokens)
                
                if st.sidebar.button("🚀 Start New Round", use_container_width=True):
                    self.start_new_round(selected_token)
                    st.rerun()
            
            elif st.session_state.current_round.status == 'ACTIVE':
                # Active round controls
                current_round = st.session_state.current_round
                
                st.sidebar.metric("Status", "🟢 ACTIVE")
                st.sidebar.metric("Current Token", current_round.current_token)
                st.sidebar.metric("Amount", f"{current_round.current_amount:.6f}")
                
                if st.sidebar.button("⏹️ End Round", use_container_width=True):
                    self.end_current_round()
                    st.rerun()

    def render_swap_matrix(self):
        """Renderuje matrycę swap opportunities"""
        if not st.session_state.current_round or st.session_state.current_round.status != 'ACTIVE':
            st.info("💡 Start a trading round to see swap opportunities")
            return
            
        current_round = st.session_state.current_round
        current_prices = st.session_state.prices
        
        st.header("🎯 Swap Opportunity Matrix")
        st.info(f"**Trading with:** {current_round.current_token} ({current_round.current_amount:.6f})")
        
        matrix_data = []
        for target_token in self.tokens_to_track:
            if target_token == current_round.current_token:
                continue
                
            # Oblicz aktualny ekwiwalent
            current_equivalent = self.calculate_equivalent_with_prices(
                current_round.current_token, 
                target_token, 
                current_round.current_amount,
                current_prices
            )
            
            top_equivalent = current_round.top_equivalents.get(target_token, 0)
            
            if top_equivalent > 0:
                change_pct = ((current_equivalent - top_equivalent) / top_equivalent * 100)
            else:
                change_pct = 0
            
            # Określ status
            status = "🔴"
            if change_pct >= self.swap_threshold:
                status = "🟢 SWAP"
            elif change_pct >= 0:
                status = "🟡"
                
            matrix_data.append({
                'Target': target_token,
                'Current': current_equivalent,
                'Top': top_equivalent,
                'Gain %': change_pct,
                'Status': status
            })
        
        if matrix_data:
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
            best_ops = df[df['Gain %'] >= self.swap_threshold]
            if not best_ops.empty:
                st.subheader("💎 Best Swap Opportunities")
                for _, row in best_ops.iterrows():
                    st.success(
                        f"**{row['Target']}**: +{row['Gain %']:.2f}% gain "
                        f"(Current: {row['Current']:.6f} vs Top: {row['Top']:.6f})"
                    )

    def render_trade_history(self):
        """Pokazuje historię trade w rundzie"""
        if not st.session_state.current_round or not st.session_state.current_round.trade_history:
            return
            
        st.header("📋 Trade History")
        
        history_data = []
        for trade in st.session_state.current_round.trade_history:
            history_data.append({
                'Time': trade.timestamp.strftime('%H:%M:%S'),
                'From': f"{trade.from_amount:.6f} {trade.from_token}",
                'To': f"{trade.to_amount:.6f} {trade.to_token}",
                'Type': f"{trade.from_token} → {trade.to_token}"
            })
        
        df = pd.DataFrame(history_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

    def auto_refresh_data(self):
        """Automatyczne odświeżanie danych"""
        if st.session_state.tracking and st.session_state.api_initialized:
            current_time = datetime.now()
            
            # Odśwież ceny co 3 sekundy
            if 'last_price_refresh' not in st.session_state:
                st.session_state.last_price_refresh = current_time
            
            if (current_time - st.session_state.last_price_refresh).seconds >= 3:
                st.session_state.prices = self.get_current_prices_with_timestamp()
                st.session_state.last_price_refresh = current_time
            
            # Sprawdzaj swapy co 5 sekund
            if 'last_swap_check' not in st.session_state:
                st.session_state.last_swap_check = current_time
            
            if (current_time - st.session_state.last_swap_check).seconds >= 5:
                if self.detect_and_process_swap():
                    return True  # Wykryto swap - potrzeba rerun
                st.session_state.last_swap_check = current_time
            
            # Odśwież portfolio co 10 sekund
            if 'last_portfolio_refresh' not in st.session_state:
                st.session_state.last_portfolio_refresh = current_time
            
            if (current_time - st.session_state.last_portfolio_refresh).seconds >= 10:
                new_portfolio = st.session_state.mexc_api.get_account_balance()
                if new_portfolio:
                    st.session_state.portfolio = new_portfolio
                    st.session_state.last_portfolio_refresh = current_time
        
        return False

    def run(self):
        """Główna pętla aplikacji"""
        st.title("🤖 Crypto Auto Trader - MEXC")
        st.markdown("---")
        
        # Inicjalizacja
        self.init_session_state()
        
        # Layout
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if not st.session_state.api_initialized:
                st.info("🔐 Configure your MEXC API keys in the sidebar to start")
            
            self.render_portfolio_overview()
            
            if st.session_state.api_initialized:
                self.render_swap_matrix()
                self.render_trade_history()
        
        with col2:
            self.setup_api_credentials()
            if st.session_state.api_initialized:
                self.render_round_control()
        
        # Auto refresh
        if self.auto_refresh_data():
            st.rerun()

# Uruchomienie aplikacji
if __name__ == "__main__":
    app = CryptoAutoTrader()
    app.run()