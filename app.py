import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from dataclasses import dataclass
from typing import Dict, List
import json

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
        
    def init_session_state(self):
        """Inicjalizacja stanu sesji"""
        if 'portfolio' not in st.session_state:
            st.session_state.portfolio = []
        if 'prices' not in st.session_state:
            st.session_state.prices = self.get_initial_prices()
        if 'trades' not in st.session_state:
            st.session_state.trades = []
        if 'tracking' not in st.session_state:
            st.session_state.tracking = False
        if 'price_updates' not in st.session_state:
            st.session_state.price_updates = 0

    def get_initial_prices(self) -> Dict[str, TokenInfo]:
        """Pobierz początkowe ceny - SYMULACJA"""
        popular_tokens = [
            'BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'XRP', 'DOT', 'DOGE', 'AVAX', 'MATIC',
            'LTC', 'LINK', 'ATOM', 'XLM', 'BCH', 'ALGO', 'FIL', 'ETC', 'EOS', 'XTZ',
            'AAVE', 'COMP', 'MKR', 'UNI', 'CRV', 'SUSHI', 'YFI', 'SNX', '1INCH', 'ZRX'
        ]
        
        prices = {}
        for token in popular_tokens:
            base_price = max(10, abs(hash(token)) % 5000)
            prices[token] = TokenInfo(
                symbol=token,
                bid_price=base_price * 0.999,
                ask_price=base_price * 1.001,
                last_update=datetime.now()
            )
        return prices

    def simulate_price_updates(self):
        """Symulacja aktualizacji cen - W PRODUKCJI ZASTĄP MEXC API"""
        for token, info in st.session_state.prices.items():
            change = (np.random.random() - 0.5) * 0.08  # ±4%
            current_mid = (info.bid_price + info.ask_price) / 2
            new_mid = max(0.01, current_mid * (1 + change))
            
            st.session_state.prices[token].bid_price = new_mid * 0.999
            st.session_state.prices[token].ask_price = new_mid * 1.001
            st.session_state.prices[token].last_update = datetime.now()
        
        st.session_state.price_updates += 1

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
        """Dodaj token do portfolio"""
        if len(st.session_state.portfolio) >= 5:
            st.error("❌ Maksymalnie 5 slotów w portfolio!")
            return
            
        if any(slot['token'] == token for slot in st.session_state.portfolio):
            st.error(f"❌ Token {token} jest już w portfolio!")
            return
            
        # Oblicz baseline equivalents
        baseline = {}
        for target_token in st.session_state.prices:
            if target_token != token:
                baseline[target_token] = self.calculate_equivalent(token, target_token, quantity)
        
        new_slot = {
            'token': token,
            'quantity': quantity,
            'baseline': baseline,
            'top_equivalent': baseline.copy(),
            'current_gain': {token: 0.0 for token in st.session_state.prices},
            'max_gain': {token: 0.0 for token in st.session_state.prices},
            'usdt_value': quantity * st.session_state.prices[token].bid_price
        }
        
        st.session_state.portfolio.append(new_slot)
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
        """Wykonaj transakcję trailing stop"""
        # Aktualizuj top equivalent jeśli current > top
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
        
        # Resetuj tracking dla nowego tokena (ALE ZACHOWAJ BASELINE!)
        for token in st.session_state.prices:
            if token != target_token:
                new_eq = self.calculate_equivalent(target_token, token, equivalent)
                slot['top_equivalent'][token] = new_eq
                slot['current_gain'][token] = 0.0
                slot['max_gain'][token] = 0.0
        
        st.session_state.trades.append(trade)
        
        # Komunikat o transakcji
        st.toast(f"🔁 SWAP: {old_token} → {target_token} (Slot {slot_idx + 1})", icon="✅")

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
                
                if st.button("Dodaj do portfolio", type="primary"):
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
                
                matrix_data.append({
                    'Token': token,
                    'Aktualny': current_eq,
                    'Początkowy': baseline_eq,
                    'Δ Od początku': change_baseline,
                    'Top': top_eq,
                    'Δ Od top': change_top,
                    'Max Wzrost': max_gain,
                    'Status': '🟢' if change_top >= -1 else '🟡' if change_top >= -3 else '🔴'
                })
        
        df = pd.DataFrame(matrix_data)
        
        # Stylowanie tabeli
        styled_df = df.style.format({
            'Aktualny': '{:.6f}',
            'Początkowy': '{:.6f}',
            'Δ Od początku': '{:+.2f}%',
            'Top': '{:.6f}',
            'Δ Od top': '{:+.2f}%',
            'Max Wzrost': '{:+.2f}%'
        }).background_gradient(
            subset=['Δ Od początku', 'Δ Od top'], 
            cmap='RdYlGn'
        )
        
        st.dataframe(styled_df, use_container_width=True)

    def render_trade_history(self):
        """Renderuj historię transakcji"""
        if st.session_state.trades:
            st.header("📋 Historia Transakcji")
            
            history_data = []
            for trade in st.session_state.trades[-20:]:  # Ostatnie 20 transakcji
                history_data.append({
                    'Czas': trade['timestamp'].strftime('%H:%M:%S'),
                    'Slot': trade['slot'] + 1,
                    'Z': trade['from_token'],
                    'Na': trade['to_token'],
                    'Ilość': f"{trade['to_quantity']:.6f}",
                    'Max Wzrost': f"{trade['max_gain']:.2f}%",
                    'Powód': trade['reason']
                })
            
            st.dataframe(pd.DataFrame(history_data), use_container_width=True)

    def render_charts(self):
        """Renderuj wykresy"""
        if not st.session_state.portfolio:
            return
            
        st.header("📈 Wizualizacje")
        
        # Wykres wartości portfolio w USDT
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Wartość Portfolio (USDT)")
            portfolio_values = []
            labels = []
            for slot in st.session_state.portfolio:
                value = slot['quantity'] * st.session_state.prices[slot['token']].bid_price
                portfolio_values.append(value)
                labels.append(f"{slot['token']}\n({slot['quantity']:.2f})")
            
            fig = px.pie(
                values=portfolio_values, 
                names=labels,
                color_discrete_sequence=px.colors.sequential.RdBu
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Rozkład Tokenów")
            tokens = [slot['token'] for slot in st.session_state.portfolio]
            quantities = [slot['quantity'] for slot in st.session_state.portfolio]
            
            fig2 = px.bar(
                x=tokens, 
                y=quantities,
                color=tokens,
                labels={'x': 'Token', 'y': 'Ilość'}
            )
            st.plotly_chart(fig2, use_container_width=True)

    def run(self):
        """Główna pętla aplikacji"""
        self.init_session_state()
        
        # Nagłówek
        st.title("🚀 Crypto Trailing Stop Matrix")
        st.markdown("---")
        
        # Renderuj komponenty
        self.render_sidebar()
        self.render_portfolio_overview()
        
        if st.session_state.portfolio:
            self.render_trailing_matrix()
            self.render_charts()
            self.render_trade_history()
            
            # Automatyczne aktualizacje gdy tracking aktywny
            if st.session_state.tracking:
                self.simulate_price_updates()
                self.check_and_execute_trades()
                time.sleep(2)  # Symulacja opóźnienia
                st.rerun()

# Uruchom aplikację
if __name__ == "__main__":
    app = CryptoTrailingStopApp()
    app.run()