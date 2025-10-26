import streamlit as st
import pandas as pd
import requests
import time
from datetime import datetime
from typing import Dict, List

# ================== Konfiguracja strony ==================
st.set_page_config(
    page_title="Crypto Swap Matrix - Single Slot",
    page_icon="🔄",
    layout="wide"
)

class SingleSlotSwapMatrix:
    def __init__(self):
        self.fee_rate = 0.00025
        self.target_profit = 0.02
        self.tokens_to_track = [
            'BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'XRP', 'DOT', 'DOGE', 'AVAX', 'LTC',
            'LINK', 'ATOM', 'XLM', 'BCH', 'ALGO', 'FIL', 'ETC', 'XTZ', 'AAVE', 'COMP',
            'UNI', 'CRV', 'SUSHI', 'YFI', 'SNX', '1INCH', 'ZRX', 'TRX', 'VET', 'ONE',
            'CELO', 'RSR', 'NKN', 'STORJ', 'DODO', 'KAVA', 'RUNE', 'SAND', 'MANA', 'ENJ',
            'CHZ', 'ALICE', 'NEAR', 'ARB', 'OP', 'APT', 'SUI', 'SEI', 'INJ', 'RENDER'
        ]
    
    def get_prices(self) -> Dict:
        """Pobiera aktualne ceny z MEXC"""
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
            return prices
        except Exception as e:
            return {}

    def calculate_equivalent(self, from_token: str, to_token: str, quantity: float) -> float:
        """Oblicza ekwiwalent między tokenami"""
        if from_token == to_token:
            return quantity * (1 - self.fee_rate)
        
        prices = st.session_state.prices
        if not prices:
            return 0.0
            
        # Dla USDT -> Token
        if from_token == 'USDT':
            if to_token not in prices:
                return 0.0
            ask_price = prices[to_token]['ask']
            if ask_price <= 0:
                return 0.0
            equivalent = (quantity / ask_price) * (1 - self.fee_rate)
            return equivalent
        
        # Dla Token -> USDT
        elif to_token == 'USDT':
            if from_token not in prices:
                return 0.0
            bid_price = prices[from_token]['bid']
            if bid_price <= 0:
                return 0.0
            equivalent = quantity * bid_price * (1 - self.fee_rate)
            return equivalent
        
        # Dla Token -> Token
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

    def calculate_sell_price_for_profit(self, purchase_price: float) -> float:
        """Oblicza cenę sprzedaży dla zysku 2% po opłatach"""
        required_gross_return = (1 + self.target_profit) / (1 - self.fee_rate)
        return purchase_price * required_gross_return

    def initialize_baseline(self):
        """Inicjalizuje baseline dla aktualnego assetu"""
        asset = st.session_state.current_asset
        
        if asset['type'] == 'USDT':
            baseline = {}
            for token in self.tokens_to_track:
                equivalent = self.calculate_equivalent('USDT', token, asset['amount'])
                baseline[token] = equivalent
            st.session_state.baseline_equivalents = baseline
            
        else:
            baseline = {}
            for token in self.tokens_to_track:
                if token != asset['token']:
                    equivalent = self.calculate_equivalent(asset['token'], token, asset['amount'])
                    baseline[token] = equivalent
            st.session_state.baseline_equivalents = baseline
        
        st.session_state.baseline_time = datetime.now()

    def render_control_panel(self):
        """Panel kontrolny"""
        st.sidebar.header("🎮 Sterowanie")
        
        status = "🟢 AKTYWNE" if st.session_state.tracking else "🔴 WYŁĄCZONE"
        st.sidebar.metric("Status śledzenia", status)
        
        if st.session_state.prices:
            price_values = list(st.session_state.prices.values())
            if price_values and 'last_update' in price_values[0]:
                last_update = price_values[0]['last_update']
                st.sidebar.caption(f"🕒 Ostatnie dane: {last_update.strftime('%H:%M:%S')}")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("▶ Start", use_container_width=True) and not st.session_state.tracking:
                if st.session_state.prices:
                    st.session_state.tracking = True
                    self.initialize_baseline()
                    st.rerun()
                else:
                    st.error("❌ Brak danych cenowych")
        
        with col2:
            if st.button("⏹ Stop", use_container_width=True) and st.session_state.tracking:
                st.session_state.tracking = False
                st.rerun()
        
        st.sidebar.markdown("---")
        
        if 'baseline_time' in st.session_state:
            st.sidebar.info(f"📊 Baseline z: {st.session_state.baseline_time.strftime('%H:%M:%S')}")

    def render_asset_input(self):
        """Input dla assetu"""
        st.header("💰 Stan Portfolio - Jeden Slot")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            asset_type = st.radio(
                "Posiadam:",
                ["USDT", "Token"],
                horizontal=True,
                key="asset_type_input"
            )
        
        with col2:
            if asset_type == "USDT":
                usdt_amount = st.number_input(
                    "Ilość USDT:",
                    min_value=1.0,
                    value=1000.0,
                    step=100.0,
                    key="usdt_amount_input"
                )
                new_asset = {
                    'type': 'USDT',
                    'amount': usdt_amount,
                    'token': None,
                    'purchase_price': None
                }
            else:
                token = st.selectbox("Token:", self.tokens_to_track, key="token_select")
                token_amount = st.number_input(
                    f"Ilość {token}:",
                    min_value=0.000001,
                    value=1.0,
                    step=0.1,
                    format="%.6f",
                    key="token_amount_input"
                )
                purchase_price = st.number_input(
                    "Cena zakupu (USDT):",
                    min_value=0.000001,
                    value=1000.0,
                    step=1.0,
                    key="purchase_price_input"
                )
                new_asset = {
                    'type': 'TOKEN',
                    'amount': token_amount,
                    'token': token,
                    'purchase_price': purchase_price
                }
        
        with col3:
            current_asset = st.session_state.current_asset
            asset_changed = (new_asset['type'] != current_asset['type'] or 
                           new_asset['amount'] != current_asset['amount'] or
                           (new_asset['type'] == 'TOKEN' and 
                            new_asset['token'] != current_asset.get('token')))
            
            if asset_changed:
                st.session_state.current_asset = new_asset
                if st.session_state.tracking:
                    self.initialize_baseline()
                st.rerun()
            
            asset = st.session_state.current_asset
            if asset['type'] == 'USDT':
                st.metric("Stan", f"{asset['amount']:,.2f} USDT")
            else:
                current_price = st.session_state.prices.get(asset['token'], {}).get('bid', 0)
                current_value = asset['amount'] * current_price
                purchase_value = asset['amount'] * asset['purchase_price']
                profit_loss = current_value - purchase_value
                profit_pct = (profit_loss / purchase_value * 100) if purchase_value > 0 else 0
                
                st.metric(
                    "Stan", 
                    f"{asset['amount']:.6f} {asset['token']}",
                    delta=f"{profit_pct:+.2f}%"
                )

    def render_matrix(self, placeholder):
        """Renderuje matrycę ekwiwalentów z placeholder"""
        if 'baseline_equivalents' not in st.session_state or not st.session_state.tracking:
            placeholder.info("💡 Kliknij 'Start' aby rozpocząć śledzenie")
            return
        
        asset = st.session_state.current_asset
        prices = st.session_state.prices
        
        if not prices:
            placeholder.error("❌ Brak danych cenowych")
            return
        
        matrix_data = []
        
        if asset['type'] == 'USDT':
            placeholder.header("📊 Matryca zakupów - Śledzenie % zmiany od baseline")
            placeholder.info("🎯 Cel: Kupić więcej tokenów niż przy inicjacji")
            
            for token in self.tokens_to_track:
                if token in prices:
                    current_equivalent = self.calculate_equivalent('USDT', token, asset['amount'])
                    baseline_equivalent = st.session_state.baseline_equivalents.get(token, current_equivalent)
                    
                    if baseline_equivalent > 0:
                        change_pct = ((current_equivalent - baseline_equivalent) / baseline_equivalent * 100)
                    else:
                        change_pct = 0
                    
                    usdt_value = current_equivalent * prices[token]['bid'] * (1 - self.fee_rate)
                    
                    matrix_data.append({
                        'Token': token,
                        'Ekwiwalent': current_equivalent,
                        'Baseline': baseline_equivalent,
                        'Zmiana %': change_pct,
                        'Wartość USDT': usdt_value,
                        'Cena zakupu': prices[token]['ask']
                    })
            
            if matrix_data:
                df = pd.DataFrame(matrix_data)
                df = df.sort_values('Zmiana %', ascending=False)
                
                placeholder.dataframe(
                    df,
                    use_container_width=True,
                    column_config={
                        'Ekwiwalent': st.column_config.NumberColumn(format="%.6f"),
                        'Baseline': st.column_config.NumberColumn(format="%.6f"),
                        'Zmiana %': st.column_config.NumberColumn(format="%+.2f%%"),
                        'Wartość USDT': st.column_config.NumberColumn(format="%.2f"),
                        'Cena zakupu': st.column_config.NumberColumn(format="%.4f")
                    }
                )
            else:
                placeholder.error("❌ Brak danych do wyświetlenia")
            
        else:
            placeholder.header(f"📈 Correlation Matrix - {asset['token']}")
            placeholder.info("🎯 Cel: Akumulować więcej tokenów poprzez wymianę")
            
            current_token = asset['token']
            current_amount = asset['amount']
            
            for token in self.tokens_to_track:
                if token == current_token or token not in prices:
                    continue
                
                current_equivalent = self.calculate_equivalent(current_token, token, current_amount)
                baseline_equivalent = st.session_state.baseline_equivalents.get(token, current_equivalent)
                
                if baseline_equivalent > 0:
                    change_pct = ((current_equivalent - baseline_equivalent) / baseline_equivalent * 100)
                else:
                    change_pct = 0
                
                usdt_value = self.calculate_equivalent(current_token, 'USDT', current_amount)
                
                if asset['purchase_price']:
                    sell_target_price = self.calculate_sell_price_for_profit(asset['purchase_price'])
                    current_price = prices[current_token]['bid']
                    profit_pct = ((current_price - asset['purchase_price']) / asset['purchase_price'] * 100)
                else:
                    sell_target_price = 0
                    profit_pct = 0
                
                matrix_data.append({
                    'Token': token,
                    'Ekwiwalent': current_equivalent,
                    'Baseline': baseline_equivalent,
                    'Zmiana %': change_pct,
                    'Wartość USDT': usdt_value,
                    'Sell Price +2%': sell_target_price,
                    'Aktualny zysk USDT': profit_pct
                })
            
            if matrix_data:
                df = pd.DataFrame(matrix_data)
                df = df.sort_values('Zmiana %', ascending=False)
                
                placeholder.dataframe(
                    df,
                    use_container_width=True,
                    column_config={
                        'Ekwiwalent': st.column_config.NumberColumn(format="%.6f"),
                        'Baseline': st.column_config.NumberColumn(format="%.6f"),
                        'Zmiana %': st.column_config.NumberColumn(format="%+.2f%%"),
                        'Wartość USDT': st.column_config.NumberColumn(format="%.2f"),
                        'Sell Price +2%': st.column_config.NumberColumn(format="%.4f"),
                        'Aktualny zysk USDT': st.column_config.NumberColumn(format="%+.2f%%")
                    }
                )
            else:
                placeholder.error("❌ Brak danych do wyświetlenia")

    def render_swap_interface(self):
        """Interface do manualnych swapów"""
        if not st.session_state.tracking:
            return
            
        st.header("🔄 Manualne operacje")
        
        if st.session_state.current_asset['type'] == 'USDT':
            col1, col2 = st.columns([2, 1])
            with col1:
                target_token = st.selectbox("Wybierz token do zakupu:", self.tokens_to_track)
            
            with col2:
                if st.button("💰 Kup token", use_container_width=True, type="primary"):
                    asset = st.session_state.current_asset
                    prices = st.session_state.prices
                    
                    if target_token in prices:
                        equivalent = self.calculate_equivalent('USDT', target_token, asset['amount'])
                        st.session_state.current_asset = {
                            'type': 'TOKEN',
                            'amount': equivalent,
                            'token': target_token,
                            'purchase_price': prices[target_token]['ask']
                        }
                        self.initialize_baseline()
                        st.success(f"✅ Zakupiono {equivalent:.6f} {target_token}")
                        st.rerun()
        
        else:
            current_token = st.session_state.current_asset['token']
            available_tokens = [t for t in self.tokens_to_track if t != current_token]
            
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                target_token = st.selectbox("Wybierz token docelowy:", available_tokens)
            
            with col2:
                if st.button("🔄 Wymień token", use_container_width=True, type="primary"):
                    asset = st.session_state.current_asset
                    equivalent = self.calculate_equivalent(asset['token'], target_token, asset['amount'])
                    
                    st.session_state.current_asset = {
                        'type': 'TOKEN',
                        'amount': equivalent,
                        'token': target_token,
                        'purchase_price': st.session_state.prices[target_token]['ask']
                    }
                    self.initialize_baseline()
                    st.success(f"✅ Wymieniono {asset['token']} → {target_token}")
                    st.rerun()
            
            with col3:
                if st.button("💵 Sprzedaj do USDT", use_container_width=True):
                    asset = st.session_state.current_asset
                    equivalent = self.calculate_equivalent(asset['token'], 'USDT', asset['amount'])
                    
                    st.session_state.current_asset = {
                        'type': 'USDT',
                        'amount': equivalent,
                        'token': None,
                        'purchase_price': None
                    }
                    self.initialize_baseline()
                    st.success(f"✅ Sprzedano {asset['token']} za {equivalent:,.2f} USDT")
                    st.rerun()

    def auto_refresh(self):
        """Automatyczne odświeżanie danych co 2 sekundy"""
        if st.session_state.tracking:
            current_time = datetime.now()
            if 'last_refresh' not in st.session_state:
                st.session_state.last_refresh = current_time
            
            if (current_time - st.session_state.last_refresh).total_seconds() >= 2:
                new_prices = self.get_prices()
                if new_prices:
                    st.session_state.prices = new_prices
                    st.session_state.last_refresh = current_time
                    return True
        return False

    def init_session_state(self):
        """Inicjalizacja session state"""
        if 'prices' not in st.session_state:
            st.session_state.prices = self.get_prices()
        
        if 'current_asset' not in st.session_state:
            st.session_state.current_asset = {
                'type': 'USDT',
                'amount': 1000.0,
                'token': None,
                'purchase_price': None
            }
        
        if 'tracking' not in st.session_state:
            st.session_state.tracking = False
        
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = datetime.now()

    def run(self):
        """Główna pętla aplikacji"""
        st.title("🔄 Single Slot Crypto Matrix")
        st.markdown("---")
        
        # Inicjalizacja
        self.init_session_state()
        
        # Layout
        col1, col2 = st.columns([3, 1])
        
        with col1:
            self.render_asset_input()
            st.markdown("---")
            
            # Placeholder dla matrycy który będzie dynamicznie odświeżany
            matrix_placeholder = st.empty()
            
            # Renderowanie matrycy
            self.render_matrix(matrix_placeholder)
            
            st.markdown("---")
            self.render_swap_interface()
        
        with col2:
            self.render_control_panel()
        
        # Auto refresh z obsługą placeholder
        if self.auto_refresh():
        import streamlit as st
import pandas as pd
import requests
import hmac
import hashlib
import time
import json
from datetime import datetime
from typing import Dict, List, Optional

# ================== Konfiguracja strony ==================
st.set_page_config(
    page_title="Crypto Auto Trader - MEXC",
    page_icon="🤖",
    layout="wide"
)

class MexcPrivateAPI:
    def __init__(self, api_key: str, secret_key: str):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = "https://api.mexc.com"
        
    def _sign_request(self, params: dict) -> str:
        """Generuje podpis HMAC SHA256 dla requestów"""
        query_string = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
        return hmac.new(
            self.secret_key.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def _make_private_request(self, endpoint: str, params: dict = None, method: str = "GET") -> Optional[dict]:
        """Wysyła autoryzowany request do MEXC"""
        try:
            timestamp = int(time.time() * 1000)
            params = params or {}
            params.update({
                'timestamp': timestamp,
                'recvWindow': 5000
            })
            
            signature = self._sign_request(params)
            params['signature'] = signature
            
            headers = {
                'X-MEXC-APIKEY': self.api_key,
                'Content-Type': 'application/json'
            }
            
            if method == "GET":
                response = requests.get(
                    f"{self.base_url}{endpoint}",
                    params=params,
                    headers=headers,
                    timeout=10
                )
            else:  # POST
                response = requests.post(
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
    
    def get_current_orders(self, symbol: str = None) -> List[dict]:
        """Pobiera aktualne zlecenia"""
        params = {}
        if symbol:
            params['symbol'] = symbol
            
        data = self._make_private_request('/api/v3/openOrders', params)
        return data or []
    
    def create_order(self, symbol: str, side: str, order_type: str, 
                    quantity: float, price: float = None) -> Optional[dict]:
        """Wysyła zlecenie na giełdę"""
        params = {
            'symbol': symbol,
            'side': side.upper(),
            'type': order_type.upper(),
            'quantity': quantity
        }
        
        if price and order_type.upper() == 'LIMIT':
            params['price'] = price
            params['timeInForce'] = 'GTC'
        
        data = self._make_private_request('/api/v3/order', params, "POST")
        return data
    
    def cancel_order(self, symbol: str, order_id: str) -> Optional[dict]:
        """Anuluje zlecenie"""
        params = {
            'symbol': symbol,
            'orderId': order_id
        }
        
        data = self._make_private_request('/api/v3/order', params, "DELETE")
        return data

class CryptoAutoTrader:
    def __init__(self):
        self.fee_rate = 0.00025
        self.target_profit = 0.02
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
        if 'trade_history' not in st.session_state:
            st.session_state.trade_history = []
    
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
        
        with st.sidebar.form("api_config"):
            api_key = st.text_input("MEXC API Key", type="password")
            secret_key = st.text_input("MEXC Secret Key", type="password")
            
            if st.form_submit_button("🔗 Connect to MEXC"):
                if api_key and secret_key:
                    with st.spinner("Connecting to MEXC..."):
                        try:
                            st.session_state.mexc_api = MexcPrivateAPI(api_key, secret_key)
                            # Test connection
                            balance = st.session_state.mexc_api.get_account_balance()
                            if balance is not None:
                                st.session_state.api_initialized = True
                                st.session_state.portfolio = balance
                                st.session_state.tracking = True
                                self.initialize_baseline_from_portfolio()
                                st.success("✅ Connected to MEXC API")
                            else:
                                st.error("❌ Failed to connect - check your API keys")
                        except Exception as e:
                            st.error(f"❌ Connection error: {e}")
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
                    'Amount': f"{balance['total']:,.2f}",
                    'Value USDT': f"{value:,.2f}",
                    'Price': "1.0000"
                })
            elif asset in prices:
                value = balance['total'] * prices[asset]['bid']
                total_value += value
                portfolio_data.append({
                    'Asset': asset,
                    'Amount': f"{balance['total']:.6f}",
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

    def render_swap_matrix(self):
        """Renderuje matrycę swapów dla każdego tokena w portfolio"""
        if not st.session_state.tracking:
            st.info("💡 Start tracking to see swap matrix")
            return
            
        st.header("🎯 Swap Matrix")
        
        for asset, balance_info in st.session_state.portfolio.items():
            if asset == 'USDT' or balance_info['total'] <= 0:
                continue
                
            self.render_asset_matrix(asset, balance_info['total'])
    
    def render_asset_matrix(self, asset: str, amount: float):
        """Renderuje matrycę dla konkretnego assetu"""
        st.subheader(f"🔷 {asset} - {amount:.6f}")
        
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
            
            matrix_data.append({
                'Target Token': target_token,
                'Current Equivalent': current_equivalent,
                'Baseline': baseline_equivalent,
                'Change %': change_pct,
                'USDT Value': usdt_value
            })
        
        if matrix_data:
            df = pd.DataFrame(matrix_data)
            df = df.sort_values('Change %', ascending=False)
            
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    'Current Equivalent': st.column_config.NumberColumn(format="%.6f"),
                    'Baseline': st.column_config.NumberColumn(format="%.6f"),
                    'Change %': st.column_config.NumberColumn(format="%+.2f%%"),
                    'USDT Value': st.column_config.NumberColumn(format="%.2f")
                }
            )
            
            # Przyciski do szybkiego swapu
            best_swap = df.iloc[0]
            if best_swap['Change %'] > 1.0:  > 1% gain
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.info(f"🎯 Best swap: {asset} → {best_swap['Target Token']} (+{best_swap['Change %']:.2f}%)")
                with col2:
                    if st.button(f"Swap to {best_swap['Target Token']}", key=f"swap_{asset}"):
                        self.execute_swap(asset, best_swap['Target Token'], amount)

    def execute_swap(self, from_token: str, to_token: str, amount: float):
        """Wykonuje swap między tokenami"""
        if not st.session_state.api_initialized:
            st.error("❌ API not initialized")
            return
            
        # Oblicz ilość do sprzedaży (95% dostępnej)
        sell_amount = amount * 0.95
        
        # 1. Sprzedaj from_token za USDT
        symbol_sell = f"{from_token}USDT"
        current_price = st.session_state.prices[from_token]['bid']
        sell_price = current_price * 0.995  # 0.5% below current price
        
        sell_result = st.session_state.mexc_api.create_order(
            symbol=symbol_sell,
            side="SELL",
            order_type="LIMIT",
            quantity=sell_amount,
            price=sell_price
        )
        
        if sell_result:
            # 2. Kup to_token za USDT
            symbol_buy = f"{to_token}USDT"
            buy_price = st.session_state.prices[to_token]['ask'] * 1.005  # 0.5% above current price
            
            # Oblicz ile USDT otrzymaliśmy ze sprzedaży
            usdt_received = sell_amount * current_price * (1 - self.fee_rate)
            buy_amount = (usdt_received / buy_price) * (1 - self.fee_rate)
            
            buy_result = st.session_state.mexc_api.create_order(
                symbol=symbol_buy,
                side="BUY",
                order_type="LIMIT",
                quantity=buy_amount,
                price=buy_price
            )
            
            if buy_result:
                # Zapisz trade do historii
                trade = {
                    'timestamp': datetime.now(),
                    'from_token': from_token,
                    'to_token': to_token,
                    'from_amount': sell_amount,
                    'to_amount': buy_amount,
                    'usdt_value': usdt_received
                }
                st.session_state.trade_history.append(trade)
                st.success(f"✅ Swap executed: {from_token} → {to_token}")
                
                # Odśwież portfolio
                self.refresh_portfolio()
            else:
                st.error(f"❌ Buy order failed for {to_token}")

    def refresh_portfolio(self):
        """Odświeża portfolio z MEXC"""
        if st.session_state.api_initialized:
            new_portfolio = st.session_state.mexc_api.get_account_balance()
            if new_portfolio:
                st.session_state.portfolio = new_portfolio

    def render_trading_interface(self):
        """Interface do ręcznego handlu"""
        if not st.session_state.api_initialized:
            return
            
        st.header("🎮 Manual Trading")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💰 Buy Order")
            with st.form("buy_order"):
                buy_token = st.selectbox("Token to Buy", self.tokens_to_track)
                buy_amount = st.number_input("Amount", min_value=0.0001, value=0.001)
                buy_price = st.number_input("Price (USDT)", min_value=0.0001, value=1000.0)
                
                if st.form_submit_button("🟢 Place Buy Order"):
                    symbol = f"{buy_token}USDT"
                    result = st.session_state.mexc_api.create_order(
                        symbol=symbol,
                        side="BUY",
                        order_type="LIMIT",
                        quantity=buy_amount,
                        price=buy_price
                    )
                    if result:
                        st.success(f"✅ Buy order placed: {buy_amount} {buy_token}")
                        self.refresh_portfolio()

        with col2:
            st.subheader("💵 Sell Order") 
            with st.form("sell_order"):
                sell_token = st.selectbox("Token to Sell", [t for t in self.tokens_to_track if t in st.session_state.portfolio], key="sell_token")
                sell_amount = st.number_input("Amount", min_value=0.0001, value=0.001, key="sell_amount")
                sell_price = st.number_input("Price (USDT)", min_value=0.0001, value=1000.0, key="sell_price")
                
                if st.form_submit_button("🔴 Place Sell Order"):
                    symbol = f"{sell_token}USDT"
                    result = st.session_state.mexc_api.create_order(
                        symbol=symbol,
                        side="SELL", 
                        order_type="LIMIT",
                        quantity=sell_amount,
                        price=sell_price
                    )
                    if result:
                        st.success(f"✅ Sell order placed: {sell_amount} {sell_token}")
                        self.refresh_portfolio()

    def auto_refresh_data(self):
        """Automatyczne odświeżanie danych"""
        if st.session_state.tracking:
            current_time = datetime.now()
            
            # Odśwież ceny co 3 sekundy
            if 'last_price_refresh' not in st.session_state:
                st.session_state.last_price_refresh = current_time
            
            if (current_time - st.session_state.last_price_refresh).seconds >= 3:
                new_prices = self.get_current_prices()
                if new_prices:
                    st.session_state.prices = new_prices
                    st.session_state.last_price_refresh = current_time
            
            # Odśwież portfolio co 10 sekund
            if 'last_portfolio_refresh' not in st.session_state:
                st.session_state.last_portfolio_refresh = current_time
            
            if (current_time - st.session_state.last_portfolio_refresh).seconds >= 10:
                self.refresh_portfolio()
                st.session_state.last_portfolio_refresh = current_time
                return True
        
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
                    st.sidebar.caption(f"📊 Last update: {last_update.strftime('%H:%M:%S')}")
            
            col1, col2 = st.sidebar.columns(2)
            with col1:
                if st.button("🔄 Refresh Now", use_container_width=True):
                    self.refresh_portfolio()
                    st.session_state.prices = self.get_current_prices()
                    st.rerun()
            
            with col2:
                if st.button("📊 Reset Baseline", use_container_width=True):
                    self.initialize_baseline_from_portfolio()
                    st.success("✅ Baseline reset")
            
            st.sidebar.markdown("---")
            
            if 'baseline_time' in st.session_state:
                st.sidebar.info(f"⏰ Baseline: {st.session_state.baseline_time.strftime('%H:%M:%S')}")

    def run(self):
        """Główna pętla aplikacji"""
        st.title("🤖 Crypto Auto Trader - MEXC Integration")
        st.markdown("---")
        
        # Inicjalizacja
        self.init_session_state()
        
        # Layout
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # API Setup (jeśli nie połączone)
            if not st.session_state.api_initialized:
                st.info("🔐 Configure your MEXC API keys in the sidebar to start")
            
            # Główne komponenty
            self.render_portfolio_overview()
            st.markdown("---")
            self.render_swap_matrix()
            st.markdown("---")
            self.render_trading_interface()
        
        with col2:
            self.setup_api_credentials()
            if st.session_state.api_initialized:
                self.render_control_panel()
        
        # Auto refresh
        if self.auto_refresh_data():
            st.rerun()

# Uruchomienie aplikacji
if __name__ == "__main__":
    app = CryptoAutoTrader()
    app.run()    # Jeśli były nowe dane, przerysuj matrycę
            matrix_placeholder.empty()
            self.render_matrix(matrix_placeholder)

# Uruchomienie aplikacji
if __name__ == "__main__":
    app = SingleSlotSwapMatrix()
    app.run()