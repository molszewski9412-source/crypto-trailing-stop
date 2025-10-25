import streamlit as st
import pandas as pd
import requests
import time
from datetime import datetime
from typing import Dict, List

# ================== Konfiguracja strony ==================
st.set_page_config(
    page_title="Crypto Swap Matrix - Auto",
    page_icon="🔄",
    layout="wide"
)

class AutoSwapMatrix:
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
            st.error(f"❌ Błąd pobierania cen: {e}")
            return {}

    def calculate_equivalent(self, from_token: str, to_token: str, quantity: float) -> float:
        """Oblicza ekwiwalent między tokenami"""
        if from_token == to_token:
            return quantity * (1 - self.fee_rate)
        
        prices = st.session_state.prices
        if from_token not in prices or to_token not in prices:
            return 0.0
        
        try:
            # Dla USDT -> Token
            if from_token == 'USDT':
                equivalent = (quantity / prices[to_token]['ask']) * (1 - self.fee_rate)
            # Dla Token -> USDT
            elif to_token == 'USDT':
                equivalent = quantity * prices[from_token]['bid'] * (1 - self.fee_rate)
            # Dla Token -> Token
            else:
                usdt_value = quantity * prices[from_token]['bid'] * (1 - self.fee_rate)
                equivalent = (usdt_value / prices[to_token]['ask']) * (1 - self.fee_rate)
            
            return equivalent
        except:
            return 0.0

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
        
        if st.session_state.prices:
            last_update = list(st.session_state.prices.values())[0]['last_update']
            st.sidebar.caption(f"🕒 Ostatnie dane: {last_update.strftime('%H:%M:%S')}")
        
        # Przyciski kontrolne
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("▶ Start", use_container_width=True) and not st.session_state.tracking:
                st.session_state.tracking = True
                self.initialize_baseline()
                st.rerun()
        
        with col2:
            if st.button("⏹ Stop", use_container_width=True) and st.session_state.tracking:
                st.session_state.tracking = False
                st.rerun()
        
        st.sidebar.markdown("---")
        
        # Informacje o baseline
        if 'baseline_time' in st.session_state:
            st.sidebar.info(f"📊 Baseline z: {st.session_state.baseline_time.strftime('%H:%M:%S')}")

    def render_asset_input(self):
        """Input dla assetu"""
        st.header("💰 Stan Portfolio")
        
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
                st.info("🔄 Asset zaktualizowany!")
            
            # Wyświetl aktualny stan
            asset = st.session_state.current_asset
            if asset['type'] == 'USDT':
                st.metric("Stan", f"{asset['amount']:,.2f} USDT")
            else:
                current_price = st.session_state.prices.get(asset['token'], {}).get('bid', 0)
                current_value = asset['amount'] * current_price
                st.metric(
                    "Stan", 
                    f"{asset['amount']:.6f} {asset['token']}",
                    f"${current_value:,.2f}"
                )

    def render_matrix(self):
        """Renderuje matrycę ekwiwalentów"""
        if 'baseline_equivalents' not in st.session_state:
            st.info("💡 Kliknij 'Start' aby rozpocząć śledzenie")
            return
        
        asset = st.session_state.current_asset
        prices = st.session_state.prices
        
        if not prices:
            st.error("❌ Brak danych cenowych")
            return
        
        matrix_data = []
        
        if asset['type'] == 'USDT':
            st.header("📊 Matryca ekwiwalentów - USDT")
            
            for token in self.tokens_to_track:
                if token in prices:
                    # Obecny ekwiwalent
                    current_equivalent = self.calculate_equivalent('USDT', token, asset['amount'])
                    baseline_equivalent = st.session_state.baseline_equivalents.get(token, current_equivalent)
                    
                    # Zmiana % od baseline
                    change_pct = ((current_equivalent - baseline_equivalent) / baseline_equivalent * 100) if baseline_equivalent > 0 else 0
                    
                    # Wartość w USDT
                    usdt_value = current_equivalent * prices[token]['bid'] * (1 - self.fee_rate)
                    
                    matrix_data.append({
                        'Token': token,
                        'Ekwiwalent': current_equivalent,
                        'Baseline': baseline_equivalent,
                        'Zmiana %': change_pct,
                        'Wartość USDT': usdt_value,
                        'Cena': prices[token]['ask']
                    })
            
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
                        help="Zmiana względem baseline"
                    ),
                    'Wartość USDT': st.column_config.NumberColumn(format="%.2f"),
                    'Cena': st.column_config.NumberColumn(format="%.4f")
                }
            )
            
        else:  # Token
            st.header(f"📊 Matryca ekwiwalentów - {asset['token']}")
            
            current_token = asset['token']
            current_amount = asset['amount']
            
            for token in self.tokens_to_track:
                if token == current_token or token not in prices:
                    continue
                
                # Obecny ekwiwalent
                current_equivalent = self.calculate_equivalent(current_token, token, current_amount)
                baseline_equivalent = st.session_state.baseline_equivalents.get(token, current_equivalent)
                
                # Zmiana % od baseline
                change_pct = ((current_equivalent - baseline_equivalent) / baseline_equivalent * 100) if baseline_equivalent > 0 else 0
                
                # Wartość w USDT
                usdt_value = self.calculate_equivalent(current_token, 'USDT', current_amount)
                
                # Cena sprzedaży dla 2% zysku
                if asset['purchase_price']:
                    sell_target_price = self.calculate_sell_price_for_profit(asset['purchase_price'])
                else:
                    sell_target_price = 0
                
                matrix_data.append({
                    'Token': token,
                    'Ekwiwalent': current_equivalent,
                    'Baseline': baseline_equivalent,
                    'Zmiana %': change_pct,
                    'Wartość USDT': usdt_value,
                    'Sell Price +2%': sell_target_price
                })
            
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
                        help="Zmiana względem baseline"
                    ),
                    'Wartość USDT': st.column_config.NumberColumn(format="%.2f"),
                    'Sell Price +2%': st.column_config.NumberColumn(
                        format="%.4f",
                        help="Cena sprzedaży dla 2% zysku w USDT"
                    )
                }
            )

    def render_swap_interface(self):
        """Interface do manualnych swapów"""
        if st.session_state.current_asset['type'] == 'USDT':
            st.header("🔄 Manualny zakup tokena")
            
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                target_token = st.selectbox("Wybierz token do zakupu:", self.tokens_to_track)
            
            with col2:
                if st.button("Kup token", use_container_width=True):
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
                        
                        if st.session_state.tracking:
                            self.initialize_baseline()
                        
                        st.success(f"✅ Zakupiono {equivalent:.6f} {target_token}")
                        st.rerun()
            
            with col3:
                if st.button("Reset do USDT", use_container_width=True):
                    st.session_state.current_asset = {
                        'type': 'USDT',
                        'amount': 1000.0,
                        'token': None,
                        'purchase_price': None
                    }
                    if st.session_state.tracking:
                        self.initialize_baseline()
                    st.rerun()
        
        else:  # Swap między tokenami
            st.header("🔄 Manualna wymiana tokenów")
            
            current_token = st.session_state.current_asset['token']
            available_tokens = [t for t in self.tokens_to_track if t != current_token]
            
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                target_token = st.selectbox("Wybierz token docelowy:", available_tokens)
            
            with col2:
                if st.button("Wymień token", use_container_width=True):
                    asset = st.session_state.current_asset
                    equivalent = self.calculate_equivalent(asset['token'], target_token, asset['amount'])
                    
                    st.session_state.current_asset = {
                        'type': 'TOKEN',
                        'amount': equivalent,
                        'token': target_token,
                        'purchase_price': st.session_state.prices[target_token]['ask']
                    }
                    
                    if st.session_state.tracking:
                        self.initialize_baseline()
                    
                    st.success(f"✅ Wymieniono {asset['token']} → {target_token}")
                    st.rerun()
            
            with col3:
                if st.button("Sprzedaj do USDT", use_container_width=True):
                    asset = st.session_state.current_asset
                    equivalent = self.calculate_equivalent(asset['token'], 'USDT', asset['amount'])
                    
                    st.session_state.current_asset = {
                        'type': 'USDT',
                        'amount': equivalent,
                        'token': None,
                        'purchase_price': None
                    }
                    
                    if st.session_state.tracking:
                        self.initialize_baseline()
                    
                    st.success(f"✅ Sprzedano {asset['token']} za {equivalent:,.2f} USDT")
                    st.rerun()

    def auto_refresh(self):
        """Automatyczne odświeżanie danych"""
        if st.session_state.tracking:
            current_time = datetime.now()
            if 'last_refresh' not in st.session_state:
                st.session_state.last_refresh = current_time
            
            # Odśwież co sekundę
            if (current_time - st.session_state.last_refresh).total_seconds() >= 1:
                st.session_state.prices = self.get_prices()
                st.session_state.last_refresh = current_time
                st.rerun()

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
        st.title("🔄 Auto Crypto Swap Matrix")
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
    app = AutoSwapMatrix()
    app.run()