import streamlit as st
import pandas as pd
import requests
from datetime import datetime
from typing import Dict, List

# ================== Konfiguracja strony ==================
st.set_page_config(
    page_title="Crypto Swap Matrix - Manual",
    page_icon="🔄",
    layout="wide"
)

class ManualSwapMatrix:
    def __init__(self):
        self.fee_rate = 0.00025  # 0.025% fee
        self.target_profit = 0.02  # 2% target profit
        self.tokens_to_track = [
            'BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'XRP', 'DOT', 'DOGE', 'AVAX', 'LTC',
            'LINK', 'ATOM', 'XLM', 'BCH', 'ALGO', 'FIL', 'ETC', 'XTZ', 'AAVE', 'COMP'
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
                                    'spread': (ask_price - bid_price) / ask_price * 100
                                }
                        except:
                            continue
            return prices
        except Exception as e:
            st.error(f"❌ Błąd pobierania cen: {e}")
            return {}

    def calculate_sell_price_for_profit(self, purchase_price: float, target_profit: float = 0.02) -> float:
        """Oblicza cenę sprzedaży dla zysku 2% po opłatach"""
        # Cel: (sell_price * (1 - fee)) / purchase_price = 1 + target_profit
        required_gross_return = (1 + target_profit) / (1 - self.fee_rate)
        sell_price = purchase_price * required_gross_return
        return sell_price

    def render_input_section(self):
        """Sekcja wprowadzania danych"""
        st.header("💰 Stan Portfolio")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            asset_type = st.radio(
                "Posiadam:",
                ["USDT", "Token"],
                horizontal=True
            )
            
            if asset_type == "USDT":
                usdt_amount = st.number_input(
                    "Ilość USDT:",
                    min_value=1.0,
                    value=1000.0,
                    step=100.0
                )
                st.session_state.current_asset = {
                    'type': 'USDT',
                    'amount': usdt_amount,
                    'token': None,
                    'purchase_price': None
                }
                
            else:  # Token
                token = st.selectbox("Token:", self.tokens_to_track)
                token_amount = st.number_input(
                    f"Ilość {token}:",
                    min_value=0.000001,
                    value=1.0,
                    step=0.1,
                    format="%.6f"
                )
                purchase_price = st.number_input(
                    "Cena zakupu (USDT):",
                    min_value=0.000001,
                    value=1000.0,
                    step=1.0
                )
                st.session_state.current_asset = {
                    'type': 'TOKEN',
                    'amount': token_amount,
                    'token': token,
                    'purchase_price': purchase_price
                }
        
        with col2:
            if 'current_asset' in st.session_state:
                asset = st.session_state.current_asset
                if asset['type'] == 'USDT':
                    st.metric("Stan", f"{asset['amount']:,.2f} USDT")
                else:
                    current_value = asset['amount'] * st.session_state.prices.get(asset['token'], {}).get('bid', 0)
                    purchase_value = asset['amount'] * asset['purchase_price']
                    profit_loss = current_value - purchase_value
                    profit_pct = (profit_loss / purchase_value * 100) if purchase_value > 0 else 0
                    
                    st.metric(
                        "Stan", 
                        f"{asset['amount']:.6f} {asset['token']}",
                        delta=f"{profit_pct:+.2f}%"
                    )
                    st.metric(
                        "Wartość bieżąca", 
                        f"${current_value:,.2f}",
                        delta=f"${profit_loss:+.2f}"
                    )

    def render_matrix(self):
        """Renderuje matrycę ekwiwalentów"""
        if 'current_asset' not in st.session_state:
            st.info("💡 Wybierz co posiadasz powyżej")
            return
        
        asset = st.session_state.current_asset
        prices = st.session_state.prices
        
        if not prices:
            st.error("❌ Brak danych cenowych")
            return
        
        matrix_data = []
        
        if asset['type'] == 'USDT':
            # Matryca dla USDT - co możemy kupić
            st.header("🛒 Matryca zakupów za USDT")
            
            for token in self.tokens_to_track:
                if token in prices:
                    ask_price = prices[token]['ask']
                    bid_price = prices[token]['bid']
                    
                    # Ilość tokena po opłatach
                    quantity = (asset['amount'] / ask_price) * (1 - self.fee_rate)
                    # Wartość po natychmiastowej sprzedaży (z spreadem)
                    immediate_sell_value = quantity * bid_price * (1 - self.fee_rate)
                    # Efektywny spread
                    effective_spread = ((asset['amount'] - immediate_sell_value) / asset['amount']) * 100
                    
                    matrix_data.append({
                        'Token': token,
                        'Ilość do kupienia': quantity,
                        'Cena zakupu (ask)': ask_price,
                        'Wartość po sprzedaży': immediate_sell_value,
                        'Spread efektywny': effective_spread
                    })
            
            df = pd.DataFrame(matrix_data)
            st.dataframe(
                df,
                use_container_width=True,
                column_config={
                    'Ilość do kupienia': st.column_config.NumberColumn(format="%.6f"),
                    'Cena zakupu (ask)': st.column_config.NumberColumn(format="%.4f"),
                    'Wartość po sprzedaży': st.column_config.NumberColumn(format="%.2f"),
                    'Spread efektywny': st.column_config.NumberColumn(format="%.2f%%")
                }
            )
            
            # Przyciski do szybkiego wyboru
            st.subheader("🚀 Szybki wybór")
            cols = st.columns(4)
            for idx, token in enumerate(self.tokens_to_track[:8]):
                if token in prices:
                    with cols[idx % 4]:
                        if st.button(f"Kup {token}", key=f"buy_{token}"):
                            ask_price = prices[token]['ask']
                            quantity = (asset['amount'] / ask_price) * (1 - self.fee_rate)
                            st.session_state.current_asset = {
                                'type': 'TOKEN',
                                'amount': quantity,
                                'token': token,
                                'purchase_price': ask_price
                            }
                            st.rerun()
        
        else:  # Matryca dla tokena
            st.header("📊 Matryca ekwiwalentów i zysków")
            
            current_token = asset['token']
            current_amount = asset['amount']
            purchase_price = asset['purchase_price']
            
            if current_token not in prices:
                st.error(f"❌ Brak danych dla {current_token}")
                return
            
            current_bid_price = prices[current_token]['bid']
            current_usdt_value = current_amount * current_bid_price * (1 - self.fee_rate)
            
            for token in self.tokens_to_track:
                if token == current_token or token not in prices:
                    continue
                
                # Oblicz ekwiwalent w docelowym tokenie
                target_ask = prices[token]['ask']
                target_bid = prices[token]['bid']
                
                # Ekwiwalent po swapie (uwzględniając opłaty)
                equivalent_amount = (current_usdt_value / target_ask) * (1 - self.fee_rate)
                
                # Wartość w USDT po swapie
                equivalent_usdt_value = equivalent_amount * target_bid * (1 - self.fee_rate)
                
                # Cena sprzedaży dla 2% zysku
                sell_price_target = self.calculate_sell_price_for_profit(purchase_price, self.target_profit)
                
                # Aktualny zysk/strata w USDT
                current_profit_pct = ((current_bid_price - purchase_price) / purchase_price * 100) if purchase_price > 0 else 0
                
                matrix_data.append({
                    'Token': token,
                    'Ekwiwalent': equivalent_amount,
                    'Wartość USDT': equivalent_usdt_value,
                    'Sell Price dla +2%': sell_price_target,
                    'Aktualny zysk': current_profit_pct,
                    'Aktualna cena': current_bid_price
                })
            
            df = pd.DataFrame(matrix_data)
            
            # Sortowanie według zysku
            df = df.sort_values('Aktualny zysk', ascending=False)
            
            st.dataframe(
                df,
                use_container_width=True,
                column_config={
                    'Ekwiwalent': st.column_config.NumberColumn(format="%.6f"),
                    'Wartość USDT': st.column_config.NumberColumn(format="%.2f"),
                    'Sell Price dla +2%': st.column_config.NumberColumn(format="%.4f"),
                    'Aktualny zysk': st.column_config.NumberColumn(
                        format="%+.2f%%",
                        help="Zysk/strata względem ceny zakupu"
                    ),
                    'Aktualna cena': st.column_config.NumberColumn(format="%.4f")
                }
            )
            
            # Przyciski do swapów
            st.subheader("🔄 Manualne swapy")
            cols = st.columns(4)
            for idx, token in enumerate(self.tokens_to_track[:8]):
                if token in prices and token != current_token:
                    with cols[idx % 4]:
                        if st.button(f"Swap na {token}", key=f"swap_{token}"):
                            target_ask = prices[token]['ask']
                            equivalent_amount = (current_usdt_value / target_ask) * (1 - self.fee_rate)
                            st.session_state.current_asset = {
                                'type': 'TOKEN',
                                'amount': equivalent_amount,
                                'token': token,
                                'purchase_price': target_ask  # Nowa cena zakupu
                            }
                            st.success(f"✅ Wymieniono {current_token} → {token}")
                            st.rerun()

    def run(self):
        """Główna pętla aplikacji"""
        st.title("🔄 Manual Crypto Swap Matrix")
        st.markdown("---")
        
        # Inicjalizacja session state
        if 'prices' not in st.session_state:
            st.session_state.prices = self.get_prices()
        if 'current_asset' not in st.session_state:
            st.session_state.current_asset = {
                'type': 'USDT',
                'amount': 1000.0,
                'token': None,
                'purchase_price': None
            }
        
        # Auto-refresh cen
        if st.button("🔄 Odśwież ceny"):
            st.session_state.prices = self.get_prices()
            st.rerun()
        
        # Główne sekcje
        self.render_input_section()
        st.markdown("---")
        self.render_matrix()
        
        # Informacje o ostatnim odświeżeniu
        if st.session_state.prices:
            st.caption(f"📊 Ostatnie odświeżenie: {datetime.now().strftime('%H:%M:%S')}")
            st.caption(f"🎯 Target zysku: {self.target_profit*100}% | 📉 Opłata: {self.fee_rate*100}%")

# Uruchomienie aplikacji
if __name__ == "__main__":
    app = ManualSwapMatrix()
    app.run()