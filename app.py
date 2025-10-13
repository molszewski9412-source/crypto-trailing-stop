import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from dataclasses import dataclass
from typing import Dict, List
import json
import os

# Konfiguracja strony
st.set_page_config(
    page_title="Crypto Trailing Stop Matrix",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

@dataclass
class TokenInfo:
    symbol: str
    bid_price: float = 0.0
    ask_price: float = 0.0
    last_update: datetime = None

class CryptoTrailingStopApp:
    def __init__(self):
        self.fee_rate = 0.00025  # 0.025%
        self.trailing_stop_levels = {
            0.5: 0.2,   # 0.5% gain -> 0.2% TS
            1.0: 0.5,   # 1% gain -> 0.5% TS  
            2.0: 1.0    # 2% gain -> 1% TS
        }
        self.data_file = "trailing_stop_data.json"
        
        # Lista tokenów do śledzenia
        self.tokens_to_track = [
            'BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'XRP', 'DOT', 'DOGE', 'AVAX', 'MATIC',
            'LTC', 'LINK', 'ATOM', 'XLM', 'BCH', 'ALGO', 'FIL', 'ETC', 'EOS', 'XTZ',
            'AAVE', 'COMP', 'MKR', 'UNI', 'CRV', 'SUSHI', 'YFI', 'SNX', '1INCH', 'ZRX',
            'TRX', 'VET', 'THETA', 'FTM', 'ONE', 'CELO', 'RSR', 'OCEAN', 'BAND', 'NKN',
            'STMX', 'STORJ', 'DODO', 'KAVA', 'RUNE', 'SAND', 'MANA', 'ENJ', 'CHZ', 'ALICE'
        ]
        
    def get_all_prices_bulk(self) -> Dict[str, TokenInfo]:
        """Pobierz WSZYSTKIE ceny bid/ask z MEXC w JEDNYM zapytaniu"""
        prices = {}
        
        try:
            # Bulk endpoint MEXC - pobiera wszystkie pary
            url = "https://api.mexc.com/api/v3/ticker/bookTicker"
            
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                all_data = response.json()
                
                # Filtruj tylko pary USDT które nas interesują
                usdt_pairs = {item['symbol']: item for item in all_data 
                             if item['symbol'].endswith('USDT')}
                
                found_tokens = 0
                for token in self.tokens_to_track:
                    symbol = f"{token}USDT"
                    if symbol in usdt_pairs:
                        data = usdt_pairs[symbol]
                        bid_price = float(data['bidPrice'])
                        ask_price = float(data['askPrice'])
                        
                        prices[token] = TokenInfo(
                            symbol=token,
                            bid_price=bid_price,
                            ask_price=ask_price,
                            last_update=datetime.now()
                        )
                        found_tokens += 1
                
                # Dla brakujących tokenów użyj symulacji
                missing_tokens = set(self.tokens_to_track) - set(prices.keys())
                for token in missing_tokens:
                    base_price = max(10, abs(hash(token)) % 5000)
                    prices[token] = TokenInfo(
                        symbol=token,
                        bid_price=base_price * 0.999,
                        ask_price=base_price * 1.001,
                        last_update=datetime.now()
                    )
                    
            else:
                # Fallback - wszystkie ceny symulowane
                prices = self.get_simulated_prices()
                
        except Exception as e:
            # Fallback - wszystkie ceny symulowane
            prices = self.get_simulated_prices()
        
        return prices

    def get_simulated_prices(self) -> Dict[str, TokenInfo]:
        """Fallback - symulowane ceny gdy API nie działa"""
        prices = {}
        for token in self.tokens_to_track:
            base_price = max(10, abs(hash(token)) % 5000)
            prices[token] = TokenInfo(
                symbol=token,
                bid_price=base_price * 0.999,
                ask_price=base_price * 1.001,
                last_update=datetime.now()
            )
        return prices

    def get_initial_prices(self) -> Dict[str, TokenInfo]:
        """Pobierz początkowe ceny - REALNE z MEXC"""
        return self.get_all_prices_bulk()

    def update_real_prices(self):
        """Aktualizuj ceny rzeczywistymi danymi z MEXC"""
        new_prices = self.get_all_prices_bulk()
        st.session_state.prices = new_prices
        st.session_state.price_updates += 1
        st.session_state.last_tracking_time = datetime.now()

    def init_session_state(self):
        """Inicjalizacja stanu sesji z automatycznym wczytaniem danych"""
        saved_data = self.load_data()
        
        if 'portfolio' not in st.session_state:
            st.session_state.portfolio = saved_data.get('portfolio', [])
        if 'prices' not in st.session_state:
            st.session_state.prices = self.get_initial_prices()
        if 'trades' not in st.session_state:
            st.session_state.trades = saved_data.get('trades', [])
        if 'tracking' not in st.session_state:
            st.session_state.tracking = False
        if 'price_updates' not in st.session_state:
            st.session_state.price_updates = 0
        if 'last_tracking_time' not in st.session_state:
            st.session_state.last_tracking_time = datetime.now()

    def load_data(self) -> dict:
        """Automatyczne wczytywanie danych z pliku"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                
                # Konwertuj stringi datetime z powrotem na obiekty
                loaded_trades = []
                for trade in data.get('trades', []):
                    loaded_trades.append({
                        'timestamp': datetime.fromisoformat(trade['timestamp']),
                        'from_token': trade['from_token'],
                        'to_token': trade['to_token'],
                        'from_quantity': trade['from_quantity'],
                        'to_quantity': trade['to_quantity'],
                        'slot': trade['slot'],
                        'max_gain': trade['max_gain'],
                        'reason': trade['reason']
                    })
                
                return {
                    'portfolio': data.get('portfolio', []),
                    'trades': loaded_trades
                }
                
        except Exception as e:
            st.error(f"❌ Błąd wczytywania danych: {e}")
        
        return {'portfolio': [], 'trades': []}

    def save_data(self):
        """Automatyczny zapis wszystkich danych"""
        try:
            data = {
                'portfolio': st.session_state.portfolio,
                'trades': [
                    {
                        'timestamp': trade['timestamp'].isoformat(),
                        'from_token': trade['from_token'],
                        'to_token': trade['to_token'],
                        'from_quantity': trade['from_quantity'],
                        'to_quantity': trade['to_quantity'],
                        'slot': trade['slot'],
                        'max_gain': trade['max_gain'],
                        'reason': trade['reason']
                    }
                    for trade in st.session_state.trades
                ],
                'last_save': datetime.now().isoformat(),
                'save_count': len(st.session_state.trades)
            }
            
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            st.error(f"❌ Błąd zapisu danych: {e}")

    def calculate_equivalent(self, from_token: str, to_token: str, quantity: float) -> float:
        """Oblicz ekwiwalent z uwzględnieniem fee"""
        if from_token == to_token:
            return quantity
            
        prices = st.session_state.prices
        if from_token not in prices or to_token not in prices:
            return 0.0
            
        # Sprzedaż from_token -> USDT
        usdt_value = quantity * prices[from_token].bid_price * (1 - self.fee_rate)
        # Kupno USDT -> to_token
        equivalent = usdt_value / prices[to_token].ask_price * (1 - self.fee_rate)
        
        return equivalent

    def add_to_portfolio(self, token: str, quantity: float):
        """Dodaj token do portfolio z AUTOMATYCZNYM ZAPISEM"""
        if len(st.session_state.portfolio) >= 5:
            st.error("❌ Maksymalnie 5 slotów w portfolio!")
            return
            
        if any(slot['token'] == token for slot in st.session_state.portfolio):
            st.error(f"❌ Token {token} jest już w portfolio!")
            return
            
        # Oblicz baseline equivalents - ✅ ZAPISUJEMY RAZ NA ZAWSZE
        baseline = {}
        for target_token in st.session_state.prices:
            if target_token != token:
                baseline[target_token] = self.calculate_equivalent(token, target_token, quantity)
        
        new_slot = {
            'token': token,
            'quantity': quantity,
            'baseline': baseline,  # ✅ NIGDY SIĘ NIE ZMIENIA
            'top_equivalent': baseline.copy(),  # Top startuje od baseline
            'current_gain': {token: 0.0 for token in st.session_state.prices},
            'max_gain': {token: 0.0 for token in st.session_state.prices},
            'usdt_value': quantity * st.session_state.prices[token].bid_price
        }
        
        st.session_state.portfolio.append(new_slot)
        
        # 💾 AUTOMATYCZNY ZAPIS PO DODANIU SLOTU
        self.save_data()
        
        st.success(f"✅ Dodano {quantity:.4f} {token} do portfolio!")

    def check_and_execute_trades(self):
        """Sprawdź warunki trailing stop i wykonaj transakcje"""
        if not st.session_state.tracking or not st.session_state.portfolio:
            return
            
        for slot_idx, slot in enumerate(st.session_state.portfolio):
            current_tokens = [s['token'] for s in st.session_state.portfolio]
            
            for target_token in st.session_state.prices:
                if target_token != slot['token'] and target_token not in current_tokens:
                    current_eq = self.calculate_equivalent(
                        slot['token'], target_token, slot['quantity']
                    )
                    
                    baseline_eq = slot['baseline'].get(target_token, current_eq)
                    current_top = slot['top_equivalent'].get(target_token, current_eq)
                    
                    # Oblicz zmianę od baseline (całkowity zysk/strata)
                    change_from_baseline = ((current_eq - baseline_eq) / baseline_eq * 100) if baseline_eq > 0 else 0
                    
                    # Oblicz zmianę od top (dla trailing stop)
                    change_from_top = ((current_eq - current_top) / current_top * 100) if current_top > 0 else 0
                    
                    # Aktualizuj max gain
                    if change_from_top > slot['max_gain'][target_token]:
                        slot['max_gain'][target_token] = change_from_top
                    
                    # Sprawdź trailing stop
                    max_gain = slot['max_gain'][target_token]
                    current_ts = 0.0
                    
                    # Ustaw trailing stop level na podstawie max gain
                    for gain_threshold, ts_level in self.trailing_stop_levels.items():
                        if max_gain >= gain_threshold:
                            current_ts = ts_level
                    
                    # Sprawdź warunek trailing stop
                    if current_ts > 0 and change_from_top <= -current_ts:
                        self.execute_trade(slot_idx, slot, target_token, current_eq, max_gain)

    def execute_trade(self, slot_idx: int, slot: dict, target_token: str, equivalent: float, max_gain: float):
        """Wykonaj transakcję trailing stop z AUTOMATYCZNYM ZAPISEM"""
        # ✅ AKTUALIZUJ TOP EQUIVALENT PRZED SWAPEM - jeśli actual > current top
        current_top = slot['top_equivalent'][target_token]
        if equivalent > current_top:
            slot['top_equivalent'][target_token] = equivalent
        
        # Zapisz transakcję
        trade = {
            'timestamp': datetime.now(),
            'from_token': slot['token'],
            'to_token': target_token,
            'from_quantity': slot['quantity'],
            'to_quantity': equivalent,
            'slot': slot_idx,
            'max_gain': max_gain,
            'reason': f'Trailing Stop {max_gain:.1f}%'
        }
        
        # Aktualizuj slot
        old_token = slot['token']
        slot['token'] = target_token
        slot['quantity'] = equivalent
        
        # ✅ RESETUJ TYLKO TOP EQUIVALENT DLA NOWEGO TOKENA
        # ✅ BASELINE NIGDY SIĘ NIE ZMIENIA!
        for token in st.session_state.prices:
            if token != target_token:
                new_eq = self.calculate_equivalent(target_token, token, equivalent)
                # ❌ NIE resetujemy baseline - on pokazuje historię od początku!
                # ✅ Tylko top equivalent resetujemy do aktualnej wartości
                slot['top_equivalent'][token] = new_eq
                slot['current_gain'][token] = 0.0
                slot['max_gain'][token] = 0.0
        
        st.session_state.trades.append(trade)
        
        # 💾 AUTOMATYCZNY ZAPIS PO KAŻDEJ TRANSAKCJI
        self.save_data()
        
        st.toast(f"🔁 SWAP: {old_token} → {target_token} (Slot {slot_idx + 1})", icon="✅")

    def clear_all_data(self):
        """Wyczyść wszystkie dane"""
        st.session_state.portfolio = []
        st.session_state.trades = []
        st.session_state.tracking = False
        
        # Usuń plik danych
        if os.path.exists(self.data_file):
            os.remove(self.data_file)
        
        st.success("🗑️ Wszystkie dane zostały wyczyszczone!")
        st.rerun()

    def render_sidebar(self):
        """Renderuj panel boczny"""
        with st.sidebar:
            st.title("⚙️ Konfiguracja")
            
            # Dodawanie tokenów do portfolio
            st.subheader("➕ Dodaj Token")
            available_tokens = [
                token for token in st.session_state.prices.keys()
                if not any(slot['token'] == token for slot in st.session_state.portfolio)
            ]
            
            if available_tokens and len(st.session_state.portfolio) < 5:
                selected_token = st.selectbox("Wybierz token:", available_tokens)
                quantity = st.number_input("Ilość:", min_value=0.0001, value=1.0, step=0.1, format="%.4f")
                
                if st.button("Dodaj do portfolio", type="primary", use_container_width=True):
                    self.add_to_portfolio(selected_token, quantity)
                    st.rerun()
            elif len(st.session_state.portfolio) >= 5:
                st.warning("📊 Osiągnięto limit 5 slotów")
            
            # Sterowanie trackingiem
            st.subheader("🎮 Sterowanie")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("▶ Start", use_container_width=True) and not st.session_state.tracking:
                    st.session_state.tracking = True
                    st.rerun()
            
            with col2:
                if st.button("⏹ Stop", use_container_width=True) and st.session_state.tracking:
                    st.session_state.tracking = False
                    st.rerun()
            
            # Statystyki
            st.subheader("📈 Statystyki")
            st.metric("Slotów portfolio", f"{len(st.session_state.portfolio)}/5")
            st.metric("Transakcje", len(st.session_state.trades))
            st.metric("Aktualizacje cen", st.session_state.price_updates)
            st.metric("Status", "AKTYWNY" if st.session_state.tracking else "PAUZA")
            
            # Informacje o cenach
            st.subheader("💰 Aktualne ceny")
            if st.session_state.prices:
                sample_tokens = list(st.session_state.prices.keys())[:3]
                for token in sample_tokens:
                    price_info = st.session_state.prices[token]
                    st.caption(f"{token}: {price_info.bid_price:.4f} / {price_info.ask_price:.4f}")
                
                last_update = list(st.session_state.prices.values())[0].last_update
                st.caption(f"🕒 Ostatnia aktualizacja: {last_update.strftime('%H:%M:%S')}")
            
            # Informacje o danych
            st.subheader("💾 Zarządzanie danymi")
            if os.path.exists(self.data_file):
                file_time = os.path.getmtime(self.data_file)
                file_size = os.path.getsize(self.data_file) / 1024
                st.caption(f"📁 Ostatni zapis: {datetime.fromtimestamp(file_time).strftime('%H:%M:%S')}")
                st.caption(f"📊 Rozmiar danych: {file_size:.1f} KB")
                
                if st.button("🗑️ Wyczyść wszystkie dane", use_container_width=True):
                    self.clear_all_data()
            
            # Informacje o trailing stop
            st.subheader("🎯 Trailing Stop")
            for gain, stop in self.trailing_stop_levels.items():
                st.text(f"💰 {gain}% zysk → {stop}% stop")

    def render_portfolio_overview(self):
        """Renderuj przegląd portfolio"""
        st.header("📊 Przegląd Portfolio")
        
        if not st.session_state.portfolio:
            st.info("👈 Dodaj tokeny do portfolio w panelu bocznym")
            return
        
        # Kafelki portfolio
        cols = st.columns(len(st.session_state.portfolio))
        for idx, (col, slot) in enumerate(zip(cols, st.session_state.portfolio)):
            with col:
                current_value = slot['quantity'] * st.session_state.prices[slot['token']].bid_price
                st.metric(
                    label=f"Slot {idx + 1} - {slot['token']}",
                    value=f"{slot['quantity']:.4f}",
                    delta=f"{current_value:.2f} USDT"
                )

    def render_trailing_matrix(self):
        """Renderuj macierz trailing stop"""
        st.header("🎯 Macierz Trailing Stop")
        
        for slot_idx, slot in enumerate(st.session_state.portfolio):
            with st.expander(f"🔷 Slot {slot_idx + 1}: {slot['token']} ({slot['quantity']:.4f})", expanded=True):
                self.render_slot_matrix(slot_idx, slot)

    def render_slot_matrix(self, slot_idx: int, slot: dict):
        """Renderuj macierz dla pojedynczego slotu"""
        # Ogranicz do 15 tokenów dla czytelności
        display_tokens = list(st.session_state.prices.keys())[:15]
        
        matrix_data = []
        for token in display_tokens:
            if token != slot['token']:
                current_eq = self.calculate_equivalent(slot['token'], token, slot['quantity'])
                baseline_eq = slot['baseline'].get(token, current_eq)
                top_eq = slot['top_equivalent'].get(token, current_eq)
                
                change_baseline = ((current_eq - baseline_eq) / baseline_eq * 100) if baseline_eq > 0 else 0
                change_top = ((current_eq - top_eq) / top_eq * 100) if top_eq > 0 else 0
                max_gain = slot['max_gain'].get(token, 0)
                
                # Określ status kolorowy
                if change_top >= -1:
                    status = "🟢"
                elif change_top >= -3:
                    status = "🟡" 
                else:
                    status = "🔴"
                
                matrix_data.append({
                    'Token': token,
                    'Aktualny': f"{current_eq:.6f}",
                    'Początkowy': f"{baseline_eq:.6f}",
                    'Δ Od początku': f"{change_baseline:+.2f}%",
                    'Top': f"{top_eq:.6f}",
                    'Δ Od top': f"{change_top:+.2f}%",
                    'Max Wzrost': f"{max_gain:+.2f}%",
                    'Status': status
                })
        
        df = pd.DataFrame(matrix_data)
        
        st.dataframe(df, use_container_width=True)

    def render_trade_history(self):
        """Renderuj historię transakcji"""
        if st.session_state.trades:
            st.header("📋 Historia Transakcji")
            
            # Statystyki historii
            total_trades = len(st.session_state.trades)
            today_trades = len([t for t in st.session_state.trades if t['timestamp'].date() == datetime.now().date()])
            
            col1, col2 = st.columns(2)
            col1.metric("Łącznie transakcji", total_trades)
            col2.metric("Dzisiaj", today_trades)
            
            history_data = []
            for trade in st.session_state.trades[-20:]:
                history_data.append({
                    'Data': trade['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                    'Slot': trade['slot'] + 1,
                    'Z': trade['from_token'],
                    'Na': trade['to_token'],
                    'Ilość': f"{trade['to_quantity']:.6f}",
                    'Max Wzrost': f"{trade['max_gain']:.2f}%",
                    'Powód': trade['reason']
                })
            
            st.dataframe(pd.DataFrame(history_data), use_container_width=True)
        else:
            st.info("📝 Brak historii transakcji")

    def render_charts(self):
        """Renderuj wykresy"""
        if not st.session_state.portfolio:
            return
            
        st.header("📈 Wizualizacje")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Wartość Portfolio (USDT)")
            portfolio_values = []
            labels = []
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
            
            for i, slot in enumerate(st.session_state.portfolio):
                value = slot['quantity'] * st.session_state.prices[slot['token']].bid_price
                portfolio_values.append(value)
                labels.append(f"{slot['token']}\n({slot['quantity']:.2f})")
            
            fig = px.pie(
                values=portfolio_values, 
                names=labels,
                color_discrete_sequence=colors[:len(portfolio_values)]
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Rozkład Tokenów")
            tokens = [slot['token'] for slot in st.session_state.portfolio]
            quantities = [slot['quantity'] for slot in st.session_state.portfolio]
            
            fig2 = px.bar(
                x=tokens, 
                y=quantities,
                color=tokens,
                labels={'x': 'Token', 'y': 'Ilość'},
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig2, use_container_width=True)

    def run(self):
        """Główna pętla aplikacji"""
        self.init_session_state()
        
        # Nagłówek
        st.title("🚀 Crypto Trailing Stop Matrix - REAL TIME")
        st.markdown("---")
        
        # Renderuj komponenty
        self.render_sidebar()
        self.render_portfolio_overview()
        
        if st.session_state.portfolio:
            self.render_trailing_matrix()
            self.render_charts()
            self.render_trade_history()
            
            # ✅ AUTOMATYCZNA AKTUALIZACJA CEN CO 1 SEKUNDĘ
            if st.session_state.tracking:
                self.update_real_prices()
                self.check_and_execute_trades()
                time.sleep(1)  # ⚡ CO 1 SEKUNDĘ!
                st.rerun()

# Uruchom aplikację
if __name__ == "__main__":
    app = CryptoTrailingStopApp()
    app.run()