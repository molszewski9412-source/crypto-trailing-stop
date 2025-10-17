import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List
import json
import os
import plotly.graph_objects as go

# ------------------------------------------------------
# Konfiguracja strony
# ------------------------------------------------------
st.set_page_config(
    page_title="Crypto Trailing Stop Matrix - 24/7",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------
# Klasy pomocnicze
# ------------------------------------------------------
@dataclass
class TokenInfo:
    symbol: str
    bid_price: float = 0.0
    ask_price: float = 0.0
    last_update: datetime = None

# ------------------------------------------------------
# Aplikacja Crypto Trailing Stop
# ------------------------------------------------------
class CryptoTrailingStopApp:
    def __init__(self):
        # opłata transakcyjna 0.025%
        self.fee_rate = 0.00025
        # poziomy trailing stop w procentach: max_gain -> trailing stop
        self.trailing_stop_levels = {0.5: 0.2, 1.0: 0.5, 2.0: 1.0, 5.0: 2.0}
        self.data_file = "trailing_stop_data.json"

        # Tokeny do śledzenia (można edytować)
        self.tokens_to_track = [
            'BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'XRP', 'DOT', 'DOGE', 'AVAX', 'LTC',
            'LINK', 'ATOM', 'XLM', 'BCH', 'ALGO', 'FIL', 'ETC', 'XTZ', 'AAVE', 'COMP',
            'UNI', 'CRV', 'SUSHI', 'YFI', 'SNX', '1INCH', 'ZRX', 'TRX', 'VET', 'ONE',
            'CELO', 'RSR', 'NKN', 'STORJ', 'DODO', 'KAVA', 'RUNE', 'SAND', 'MANA', 'ENJ',
            'CHZ', 'ALICE', 'NEAR', 'ARB', 'OP', 'APT', 'SUI', 'SEI', 'INJ', 'RENDER'
        ]

    # ------------------------------------------------------
    # Funkcja do pobrania poziomu trailing stop dla danego max_gain
    # ------------------------------------------------------
    def get_trailing_stop_level(self, max_gain: float) -> float:
        current_ts = 0.0
        for gain_threshold, ts_level in sorted(self.trailing_stop_levels.items()):
            if max_gain >= gain_threshold:
                current_ts = ts_level
        return current_ts

    # ------------------------------------------------------
    # Test połączenia z API MEXC
    # ------------------------------------------------------
    def test_connection(self):
        try:
            url = "https://api.mexc.com/api/v3/ping"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return True, "✅ Connection to MEXC API OK"
            else:
                return False, f"❌ MEXC API returned status: {response.status_code}"
        except Exception as e:
            return False, f"❌ Connection error: {e}"

    # ------------------------------------------------------
    # Pobranie wszystkich cen top-of-book dla par USDT
    # ------------------------------------------------------
    def get_all_prices_bulk(self) -> Dict[str, TokenInfo]:
        prices = {}
        try:
            url = "https://api.mexc.com/api/v3/ticker/bookTicker"
            response = requests.get(url, timeout=15)
            if response.status_code == 200:
                all_data = response.json()
                if not all_data:
                    st.error("❌ MEXC returned empty response")
                    return {}
                # Filtracja par USDT
                usdt_pairs = {item['symbol']: item for item in all_data if item['symbol'].endswith('USDT')}
                problematic_tokens = []
                for token in self.tokens_to_track:
                    symbol = f"{token}USDT"
                    if symbol in usdt_pairs:
                        data = usdt_pairs[symbol]
                        try:
                            bid_price = float(data['bidPrice'])
                            ask_price = float(data['askPrice'])
                            if bid_price > 0 and ask_price > 0 and bid_price <= ask_price:
                                prices[token] = TokenInfo(
                                    symbol=token,
                                    bid_price=bid_price,
                                    ask_price=ask_price,
                                    last_update=datetime.now()
                                )
                            else:
                                problematic_tokens.append(f"{token}(bad prices)")
                        except (ValueError, KeyError):
                            problematic_tokens.append(f"{token}(conversion error)")
                    else:
                        problematic_tokens.append(token)
                return prices
            else:
                st.error(f"❌ HTTP error {response.status_code} from MEXC")
                return {}
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")
            return {}

    # ------------------------------------------------------
    # Pobranie początkowych cen
    # ------------------------------------------------------
    def get_initial_prices(self):
        return self.get_all_prices_bulk()

    # ------------------------------------------------------
    # Aktualizacja cen z prostym rate-limitingiem
    # ------------------------------------------------------
    def update_real_prices(self):
        if hasattr(st.session_state, 'last_price_update'):
            time_diff = (datetime.now() - st.session_state.last_price_update).seconds
            if time_diff < 3:
                return
        new_prices = self.get_all_prices_bulk()
        if new_prices:
            st.session_state.prices = new_prices
            st.session_state.price_updates += 1
            st.session_state.last_price_update = datetime.now()

    # ------------------------------------------------------
    # Inicjalizacja portfela z USDT (5 slotów)
    # ------------------------------------------------------
    def initialize_portfolio_from_usdt(self, usdt_amount: float, selected_tokens: List[str]):
        if len(selected_tokens) != 5:
            st.error("❌ Select exactly 5 tokens")
            return False
        if usdt_amount <= 0:
            st.error("❌ USDT amount must be > 0")
            return False
        available_tokens = [t for t in selected_tokens if t in st.session_state.prices]
        if len(available_tokens) < 5:
            st.error(f"❌ Not enough tokens with prices: {len(available_tokens)}/5")
            return False
        st.session_state.portfolio = []
        st.session_state.trades = []
        usdt_per_slot = usdt_amount / 5
        for token in available_tokens:
            token_price = st.session_state.prices[token].ask_price
            quantity = (usdt_per_slot / token_price) * (1 - self.fee_rate)
            baseline, top_equivalent, current_gain, max_gain = {}, {}, {}, {}
            for target_token in self.tokens_to_track:
                if target_token in st.session_state.prices:
                    equiv = self.calculate_equivalent(token, target_token, quantity)
                    baseline[target_token] = equiv
                    top_equivalent[target_token] = equiv
                    current_gain[target_token] = 0.0
                    max_gain[target_token] = 0.0
            new_slot = {
                'token': token,
                'quantity': quantity,
                'baseline': baseline,
                'top_equivalent': top_equivalent,
                'current_gain': current_gain,
                'max_gain': max_gain
            }
            st.session_state.portfolio.append(new_slot)
        self.save_data()
        st.success(f"✅ Portfolio initialized: {usdt_amount} USDT → 5 slots")
        return True

    # ------------------------------------------------------
    # Inicjalizacja Streamlit session_state
    # ------------------------------------------------------
    def init_session_state(self):
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
        if 'last_price_update' not in st.session_state:
            st.session_state.last_price_update = datetime.now()
        if 'app_start_time' not in st.session_state:
            st.session_state.app_start_time = datetime.now()

    # ------------------------------------------------------
    # Ładowanie danych z pliku JSON
    # ------------------------------------------------------
    def load_data(self):
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                trades_loaded = []
                for trade in data.get('trades', []):
                    trades_loaded.append({
                        'timestamp': datetime.fromisoformat(trade['timestamp']),
                        'from_token': trade['from_token'],
                        'to_token': trade['to_token'],
                        'from_quantity': trade['from_quantity'],
                        'to_quantity': trade['to_quantity'],
                        'slot': trade['slot'],
                        'max_gain': trade.get('max_gain', 0.0),
                        'reason': trade.get('reason', '')
                    })
                return {'portfolio': data.get('portfolio', []), 'trades': trades_loaded}
        except Exception as e:
            st.error(f"❌ Error loading data: {e}")
        return {'portfolio': [], 'trades': []}

    # ------------------------------------------------------
    # Zapis danych do pliku JSON
    # ------------------------------------------------------
    def save_data(self):
        try:
            data = {
                'portfolio': st.session_state.portfolio,
                'trades': [
                    {
                        'timestamp': t['timestamp'].isoformat(),
                        'from_token': t['from_token'],
                        'to_token': t['to_token'],
                        'from_quantity': t['from_quantity'],
                        'to_quantity': t['to_quantity'],
                        'slot': t['slot'],
                        'max_gain': t.get('max_gain', 0.0),
                        'reason': t.get('reason', '')
                    } for t in st.session_state.trades
                ]
            }
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            st.error(f"❌ Error saving data: {e}")

    # ------------------------------------------------------
    # Obliczanie ekwiwalentu tokena
    # ------------------------------------------------------
    def calculate_equivalent(self, from_token: str, to_token: str, quantity: float) -> float:
        if from_token == to_token:
            return quantity * (1 - self.fee_rate)
        prices = st.session_state.prices
        if from_token not in prices or to_token not in prices:
            return 0.0
        try:
            usdt_value = quantity * prices[from_token].bid_price * (1 - self.fee_rate)
            equivalent = (usdt_value / prices[to_token].ask_price) * (1 - self.fee_rate)
            return equivalent
        except:
            return 0.0

    # ------------------------------------------------------
    # Render nowoczesnej tabeli Plotly w dark mode
    # ------------------------------------------------------
    def render_slot_matrix_plotly(self, slot_idx: int, slot: dict):
        data = []
        tokens = []
        current_eqs = []
        baseline_eqs = []
        top_eqs = []
        change_from_baseline = []
        change_from_top = []

        for token in self.tokens_to_track:
            tokens.append(token)
            current_eq = self.calculate_equivalent(slot['token'], token, slot['quantity'])
            baseline_eq = slot['baseline'].get(token, current_eq)
            top_eq = slot['top_equivalent'].get(token, current_eq)
            change_from_baseline.append(round((current_eq - baseline_eq) / baseline_eq * 100, 2))
            change_from_top.append(round((current_eq - top_eq) / top_eq * 100, 2))
            current_eqs.append(current_eq)
            baseline_eqs.append(baseline_eq)
            top_eqs.append(top_eq)

        fig = go.Figure(data=[go.Table(
            header=dict(
                values=["Token", "Current", "Baseline", "Δ from Baseline %", "Top", "Δ from Top %"],
                fill_color='rgb(50,50,50)',
                font=dict(color='white', size=14)
            ),
            cells=dict(
                values=[tokens, current_eqs, baseline_eqs, change_from_baseline, top_eqs, change_from_top],
                fill_color='rgb(40,40,40)',
                font=dict(color='white', size=12)
            )
        )])
        fig.update_layout(
            height=800,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor='rgb(30,30,30)',
            plot_bgcolor='rgb(30,30,30)'
        )
        st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------
    # Renderowanie slotów w dark minimalistycznych kartach
    # ------------------------------------------------------
    def render_portfolio_overview(self):
        st.header("📊 Portfolio Overview")
        if not st.session_state.portfolio:
            st.info("👈 Initialize portfolio from sidebar")
            return
        cols = st.columns(len(st.session_state.portfolio))
        for idx, (col, slot) in enumerate(zip(cols, st.session_state.portfolio)):
            with col:
                if slot['token'] in st.session_state.prices:
                    current_value = slot['quantity'] * st.session_state.prices[slot['token']].bid_price
                    st.metric(
                        label=f"Slot {idx + 1} - {slot['token']}",
                        value=f"{slot['quantity']:.6f}",
                        delta=f"{current_value:.2f} USDT"
                    )

    # ------------------------------------------------------
    # Renderowanie macierzy dla wszystkich slotów
    # ------------------------------------------------------
    def render_trailing_matrix(self):
        st.header("🎯 Trailing Stop Matrix")
        for slot_idx, slot in enumerate(st.session_state.portfolio):
            st.subheader(f"🔷 Slot {slot_idx + 1}: {slot['token']} ({slot['quantity']:.6f})")
            self.render_slot_matrix_plotly(slot_idx, slot)

    # ------------------------------------------------------
    # Główna pętla aplikacji
    # ------------------------------------------------------
    def run(self):
        try:
            self.init_session_state()
            st.title("🚀 Crypto Trailing Stop Matrix - Modern Dark UX")
            st.markdown("---")

            # Sidebar
            with st.sidebar:
                st.title("⚙️ Config 24/7")
                st.metric("Slots", f"{len(st.session_state.portfolio)}/5")
                st.metric("Trades", len(st.session_state.trades))
                st.metric("Price updates", st.session_state.price_updates)
                st.subheader("💰 Init Portfolio")
                if not st.session_state.portfolio:
                    usdt_amount = st.number_input("USDT amount:", min_value=10.0, value=1000.0, step=100.0)
                    available_tokens = list(st.session_state.prices.keys())
                    available_tokens.sort()
                    selected_tokens = st.multiselect("Select 5 tokens:", available_tokens, default=available_tokens[:5])
                    if st.button("🏁 Initialize Portfolio"):
                        if len(selected_tokens) == 5:
                            self.initialize_portfolio_from_usdt(usdt_amount, selected_tokens)
                            st.rerun()

            # Render
            self.render_portfolio_overview()
            if st.session_state.portfolio:
                self.render_trailing_matrix()

        except Exception as e:
            st.error(f"🔴 Critical error: {e}")

# ------------------------------------------------------
# Run app
# ------------------------------------------------------
if __name__ == "__main__":
    app = CryptoTrailingStopApp()
    app.run()
