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
        self.target_profit = 0.02  # 2% target profit
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
            st.error(f"❌ Błąd pobierania cen: {e}")
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
            # Dla USDT: baseline to ile każdego tokena można kupić
            baseline = {}
            for token in self.tokens_to_track:
                equivalent = self.calculate_equivalent('USDT', token, asset['amount'])
                baseline[token] = equivalent
            st.session_state.baseline_equivalents = baseline
            
        else:  # Dla tokena
            # Dla tokena: baseline to ekwiwalenty w innych tokenach
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
        
        # Status śledzenia
        status = "🟢 AKTYWNE" if st.session_state.tracking else "🔴 WYŁĄCZONE"
        st.sidebar.metric("Status śledzenia", status)
        
        # Ostatnie dane
        if st.session_state.prices:
            price_values = list(st.session_state.prices.values())
            if price_values and 'last_update' in price_values[0]:
                last_update = price_values[0]['last_update']
                st.sidebar.caption(f"🕒 Ostatnie dane: {last_update.strftime('%H:%M:%S')}")
        
        # Przyciski kontrolne
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
        
        # Informacje o baseline
        if 'baseline_time' in st.session_state:
            st.sidebar.info(f"📊 Baseline z: {st.session_state.baseline_time.strftime('%H:%M:%S')}")

    def render_asset_input(self):
        """Input dla assetu - JEDEN SLOT"""
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
            # Sprawdź czy asset się zmienił
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
            
            # Wyświetl aktualny stan
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

    def render_matrix(self):
        """Renderuje matrycę ekwiwalentów - JEDEN SLOT"""
        if 'baseline_equivalents' not in st.session_state or not st.session_state.tracking:
            st.info("💡 Kliknij 'Start' aby rozpocząć śledzenie")
            return
        
        asset = st.session_state.current_asset
        prices = st.session_state.prices
        
        if not prices:
            st.error("❌ Brak danych cenowych")
            return
        
        matrix_data = []
        
        if asset['type'] == 'USDT':
            st.header("📊 Matryca zakupów - Śledzenie % zmiany od baseline")
            st.info("🎯 Cel: Kupić więcej tokenów niż przy inicjacji")
            
            for token in self.tokens_to_track:
                if token in prices:
                    # Obecny ekwiwalent
                    current_equivalent = self.calculate_equivalent('USDT', token, asset['amount'])
                    baseline_equivalent = st.session_state.baseline_equivalents.get(token, current_equivalent)
                    
                    # Zmiana % od baseline
                    if baseline_equivalent > 0:
                        change_pct = ((current_equivalent - baseline_equivalent) / baseline_equivalent * 100)
                    else:
                        change_pct = 0
                    
                    # Wartość w USDT
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
                
                st.dataframe(
                    df,
                    use_container_width=True,
                    column_config={
                        'Ekwiwalent': st.column_config.NumberColumn(format="%.6f"),
                        'Baseline': st.column_config.NumberColumn(format="%.6f"),
                        'Zmiana %': st.column_config.NumberColumn(
                            format="%+.2f%%",
                            help="Zmiana względem baseline - im wyższa tym lepiej"
                        ),
                        'Wartość USDT': st.column_config.NumberColumn(format="%.2f"),
                        'Cena zakupu': st.column_config.NumberColumn(format="%.4f")
                    }
                )
            else:
                st.error("❌ Brak danych do wyświetlenia")
            
        else:  # Token - Correlation Matrix
            st.header(f"📈 Correlation Matrix - {asset['token']}")
            st.info("🎯 Cel: Akumulować więcej tokenów poprzez wymianę")
            
            current_token = asset['token']
            current_amount = asset['amount']
            
            for token in self.tokens_to_track:
                if token == current_token or token not in prices:
                    continue
                
                # Obecny ekwiwalent
                current_equivalent = self.calculate_equivalent(current_token, token, current_amount)
                baseline_equivalent = st.session_state.baseline_equivalents.get(token, current_equivalent)
                
                # Zmiana % od baseline
                if baseline_equivalent > 0:
                    change_pct = ((current_equivalent - baseline_equivalent) / baseline_equivalent * 100)
                else:
                    change_pct = 0
                
                # Wartość w USDT
                usdt_value = self.calculate_equivalent(current_token, 'USDT', current_amount)
                
                # Cena sprzedaży dla 2% zysku
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
                
                st.dataframe(
                    df,
                    use_container_width=True,
                    column_config={
                        'Ekwiwalent': st.column_config.NumberColumn(format="%.6f"),
                        'Baseline': st.column_config.NumberColumn(format="%.6f"),
                        'Zmiana %': st.column_config.NumberColumn(
                            format="%+.2f%%",
                            help="Zmiana względem baseline - im wyższa tym lepsza wymiana"
                        ),
                        'Wartość USDT': st.column_config.NumberColumn(format="%.2f"),
                        'Sell Price +2%': st.column_config.NumberColumn(
                            format="%.4f",
                            help="Cena sprzedaży dla 2% zysku w USDT"
                        ),
                        'Aktualny zysk USDT': st.column_config.NumberColumn(
                            format="%+.2f%%",
                            help="Aktualny zysk/strata w USDT względem ceny zakupu"
                        )
                    }
                )
            else:
                st.error("❌ Brak danych do wyświetlenia")

    def render_swap_interface(self):
        """Interface do manualnych swapów - JEDEN SLOT"""
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
                        
                        # Reset baseline po zakupie
                        self.initialize_baseline()
                        
                        st.success(f"✅ Zakupiono {equivalent:.6f} {target_token}")
                        st.rerun()
        
        else:  # Swap między tokenami
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
                    
                    # Reset baseline po wymianie
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
                    
                    # Reset baseline po sprzedaży
                    self.initialize_baseline()
                    
                    st.success(f"✅ Sprzedano {asset['token']} za {equivalent:,.2f} USDT")
                    st.rerun()

    def auto_refresh(self):
        """Automatyczne odświeżanie danych co sekundę"""
        if st.session_state.tracking:
            current_time = datetime.now()
            if 'last_refresh' not in st.session_state:
                st.session_state.last_refresh = current_time
            
            # Odśwież co sekundę
            if (current_time - st.session_state.last_refresh).total_seconds() >= 1:
                new_prices = self.get_prices()
                if new_prices:  # Tylko jeśli udało się pobrać ceny
                    st.session_state.prices = new_prices
                    st.session_state.last_refresh = current_time
                    st.rerun()

    def init_session_state(self):
        """Inicjalizacja session state - JEDEN SLOT"""
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
            self.render_matrix()
            st.markdown("---")
            self.render_swap_interface()
        
        with col2:
            self.render_control_panel()
        
        # Auto refresh
        self.auto_refresh()

# Uruchomienie aplikacji
if __name__ == "__main__":
    app = SingleSlotSwapMatrix()
    app.run()