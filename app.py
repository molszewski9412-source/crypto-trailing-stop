import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import json
import os
from dataclasses import dataclass
from typing import Dict, List

# ================== Konfiguracja strony ==================
st.set_page_config(
    page_title="Crypto Trailing Stop Matrix - 24/7",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================== Data Classes ==================
@dataclass
class TokenInfo:
    symbol: str
    bid_price: float = 0.0
    ask_price: float = 0.0
    last_update: datetime = None

# ================== Main App ==================
class CryptoTrailingStopApp:
    def __init__(self):
        self.fee_rate = 0.00025
        self.trailing_stop_levels = {0.5: 0.2, 1.0: 0.5, 2.0: 1.0, 5.0: 2.0}
        self.data_file = "trailing_stop_data.json"
        self.tokens_to_track = [
            'BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'XRP', 'DOT', 'DOGE', 'AVAX', 'LTC',
            'LINK', 'ATOM', 'XLM', 'BCH', 'ALGO', 'FIL', 'ETC', 'XTZ', 'AAVE', 'COMP',
            'UNI', 'CRV', 'SUSHI', 'YFI', 'SNX', '1INCH', 'ZRX', 'TRX', 'VET', 'ONE',
            'CELO', 'RSR', 'NKN', 'STORJ', 'DODO', 'KAVA', 'RUNE', 'SAND', 'MANA', 'ENJ',
            'CHZ', 'ALICE', 'NEAR', 'ARB', 'OP', 'APT', 'SUI', 'SEI', 'INJ', 'RENDER'
        ]

    # ================== API Helpers ==================
    def test_connection(self):
        try:
            url = "https://api.mexc.com/api/v3/ping"
            r = requests.get(url, timeout=10)
            return r.status_code == 200, f"{'✅' if r.status_code==200 else '❌'} MEXC status {r.status_code}"
        except Exception as e:
            return False, f"❌ Connection error: {e}"

    def get_all_prices_bulk(self) -> Dict[str, TokenInfo]:
        prices = {}
        try:
            r = requests.get("https://api.mexc.com/api/v3/ticker/bookTicker", timeout=15)
            if r.status_code != 200:
                st.error(f"❌ HTTP {r.status_code}")
                return {}
            data = r.json()
            usdt_pairs = {item['symbol']: item for item in data if item['symbol'].endswith('USDT')}
            for token in self.tokens_to_track:
                sym = f"{token}USDT"
                if sym in usdt_pairs:
                    try:
                        bid = float(usdt_pairs[sym]['bidPrice'])
                        ask = float(usdt_pairs[sym]['askPrice'])
                        if bid <= 0 or ask <= 0 or bid > ask:
                            continue
                        prices[token] = TokenInfo(symbol=token, bid_price=bid, ask_price=ask, last_update=datetime.now())
                    except:
                        continue
            return prices
        except Exception as e:
            st.error(f"❌ Błąd pobierania danych: {e}")
            return {}

    def get_initial_prices(self):
        return self.get_all_prices_bulk()

    def update_real_prices(self):
        if hasattr(st.session_state, 'last_price_update'):
            if (datetime.now() - st.session_state.last_price_update).seconds < 3:
                return
        new_prices = self.get_all_prices_bulk()
        if new_prices:
            st.session_state.prices = new_prices
            st.session_state.price_updates += 1
            st.session_state.last_tracking_time = datetime.now()
            st.session_state.last_price_update = datetime.now()

    # ================== Portfolio ==================
    def initialize_portfolio_from_usdt(self, usdt_amount: float, selected_tokens: List[str]):
        if len(selected_tokens) != 5:
            st.error("❌ Select exactly 5 tokens")
            return False
        if usdt_amount <= 0:
            st.error("❌ USDT must be > 0")
            return False
        available_tokens = [t for t in selected_tokens if t in st.session_state.prices]
        if len(available_tokens) < 5:
            st.error(f"❌ Not enough price data: {len(available_tokens)}/5")
            return False
        st.session_state.portfolio = []
        st.session_state.trades = []
        usdt_per_slot = usdt_amount / 5
        for token in available_tokens:
            token_price = st.session_state.prices[token].ask_price
            quantity = (usdt_per_slot / token_price) * (1 - self.fee_rate)
            baseline = {}
            top_equivalent = {}
            current_gain = {}
            max_gain = {}
            for t in self.tokens_to_track:
                if t in st.session_state.prices:
                    eq = self.calculate_equivalent(token, t, quantity)
                    baseline[t] = eq
                    top_equivalent[t] = eq
                    current_gain[t] = 0.0
                    max_gain[t] = 0.0
            slot = {
                'token': token,
                'quantity': quantity,
                'baseline': baseline,
                'top_equivalent': top_equivalent,
                'current_gain': current_gain,
                'max_gain': max_gain,
                'usdt_value': quantity * st.session_state.prices[token].bid_price,
                'baseline_usdt': usdt_per_slot,
                'quantity_history': [quantity],
                'timestamp_history': [datetime.now()]
            }
            st.session_state.portfolio.append(slot)
        self.save_data()
        st.success(f"✅ Portfolio initialized: {usdt_amount} USDT → 5 slots")
        return True

    # ================== Session State ==================
    def init_session_state(self):
        saved = self.load_data()
        if 'portfolio' not in st.session_state:
            st.session_state.portfolio = saved.get('portfolio', [])
        if 'prices' not in st.session_state:
            st.session_state.prices = self.get_initial_prices()
        if 'trades' not in st.session_state:
            st.session_state.trades = saved.get('trades', [])
        if 'tracking' not in st.session_state:
            st.session_state.tracking = False
        if 'price_updates' not in st.session_state:
            st.session_state.price_updates = 0
        if 'last_tracking_time' not in st.session_state:
            st.session_state.last_tracking_time = datetime.now()
        if 'last_price_update' not in st.session_state:
            st.session_state.last_price_update = datetime.now()
        if 'app_start_time' not in st.session_state:
            st.session_state.app_start_time = datetime.now()
        if 'trailing_stop_levels' not in st.session_state:
            st.session_state.trailing_stop_levels = self.trailing_stop_levels
        
        # Napraw brakujące pola w istniejącym portfolio
        self.fix_portfolio_data()

    def fix_portfolio_data(self):
        """Naprawia brakujące pola w istniejącym portfolio"""
        if hasattr(st.session_state, 'portfolio'):
            for slot in st.session_state.portfolio:
                # Dodaj brakujące pola jeśli nie istnieją
                if 'quantity_history' not in slot:
                    slot['quantity_history'] = [slot['quantity']]
                if 'timestamp_history' not in slot:
                    slot['timestamp_history'] = [datetime.now()]
                if 'baseline_usdt' not in slot:
                    # Oblicz baseline_usdt na podstawie aktualnej ceny
                    if slot['token'] in st.session_state.prices:
                        current_price = st.session_state.prices[slot['token']].bid_price
                        slot['baseline_usdt'] = slot['quantity'] * current_price
                    else:
                        slot['baseline_usdt'] = 0.0

    # ================== Load/Save ==================
    def load_data(self):
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                trades = []
                for t in data.get('trades', []):
                    trades.append({
                        'timestamp': datetime.fromisoformat(t['timestamp']),
                        'from_token': t['from_token'],
                        'to_token': t['to_token'],
                        'from_quantity': t['from_quantity'],
                        'to_quantity': t['to_quantity'],
                        'slot': t['slot'],
                        'max_gain': t.get('max_gain', 0.0),
                        'reason': t.get('reason', '')
                    })
                
                # Dodaj brakujące pola do załadowanego portfolio
                portfolio = data.get('portfolio', [])
                for slot in portfolio:
                    if 'quantity_history' not in slot:
                        slot['quantity_history'] = [slot['quantity']]
                    if 'timestamp_history' not in slot:
                        slot['timestamp_history'] = [datetime.now()]
                    if 'baseline_usdt' not in slot:
                        slot['baseline_usdt'] = 0.0
                
                return {'portfolio': portfolio, 'trades': trades}
        except Exception as e:
            st.error(f"❌ {e}")
        return {'portfolio': [], 'trades': []}

    def save_data(self):
        try:
            data = {
                'portfolio': st.session_state.portfolio,
                'trades': [{
                    'timestamp': t['timestamp'].isoformat(),
                    'from_token': t['from_token'],
                    'to_token': t['to_token'],
                    'from_quantity': t['from_quantity'],
                    'to_quantity': t['to_quantity'],
                    'slot': t['slot'],
                    'max_gain': t.get('max_gain', 0.0),
                    'reason': t.get('reason', '')
                } for t in st.session_state.trades]
            }
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            st.error(f"❌ {e}")

    # ================== Equivalents ==================
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

    # ================== Trailing Stop Logic ==================
    def get_trailing_stop_level(self, max_gain: float) -> float:
        current_ts = 0.0
        for gain_threshold, ts in sorted(st.session_state.trailing_stop_levels.items()):
            if max_gain >= gain_threshold:
                current_ts = ts
        return current_ts

    def check_and_execute_trades(self):
        if not st.session_state.tracking or not st.session_state.portfolio:
            return
        slot_candidates = {}
        for idx, slot in enumerate(st.session_state.portfolio):
            swap_candidates = []
            current_tokens = [s['token'] for s in st.session_state.portfolio]
            from_token = slot['token']
            qty = slot['quantity']
            if qty <= 0 or from_token not in st.session_state.prices:
                continue
            for target_token in self.tokens_to_track:
                if target_token == from_token or target_token in current_tokens:
                    continue
                current_eq = self.calculate_equivalent(from_token, target_token, qty)
                if current_eq <= 0:
                    continue
                baseline_eq = slot['baseline'].get(target_token, current_eq)
                top_eq = slot['top_equivalent'].get(target_token, current_eq)
                if top_eq <= 0:
                    top_eq = current_eq
                    slot['top_equivalent'][target_token] = current_eq
                gain_from_baseline = ((current_eq - baseline_eq) / baseline_eq * 100) if baseline_eq > 0 else 0
                slot['current_gain'][target_token] = gain_from_baseline
                gain_from_top = ((current_eq - top_eq) / top_eq * 100) if top_eq > 0 else 0
                prev_max = slot['max_gain'].get(target_token, 0.0)
                if gain_from_top > prev_max:
                    slot['max_gain'][target_token] = gain_from_top
                current_max_gain = slot['max_gain'].get(target_token, 0.0)
                if current_max_gain >= 0.5 or gain_from_top >= 0.5:
                    ts = self.get_trailing_stop_level(current_max_gain)
                    swap_threshold = current_max_gain - ts
                    if gain_from_top <= swap_threshold and current_eq >= top_eq:
                        swap_candidates.append({
                            'target_token': target_token,
                            'current_eq': current_eq,
                            'max_gain': current_max_gain
                        })
            if swap_candidates:
                swap_candidates.sort(key=lambda x: x['max_gain'], reverse=True)
                slot_candidates[idx] = swap_candidates[0]
        executed_targets = set()
        for idx in sorted(slot_candidates.keys()):
            candidate = slot_candidates[idx]
            target = candidate['target_token']
            if target in executed_targets:
                continue
            current_tokens = [s['token'] for s in st.session_state.portfolio]
            if target in current_tokens:
                continue
            slot = st.session_state.portfolio[idx]
            self.execute_trade(idx, slot, target, candidate['current_eq'], candidate['max_gain'])
            executed_targets.add(target)

    def execute_trade(self, slot_idx: int, slot: dict, target_token: str, equivalent: float, max_gain: float):
        from_token = slot['token']
        from_qty = slot['quantity']

        to_qty = self.calculate_equivalent(from_token, target_token, from_qty)
        if to_qty <= 0:
            st.warning("⚠️ Swap aborted: computed to_qty <= 0")
            return

        # Aktualizacja historii ilości
        if 'quantity_history' not in slot:
            slot['quantity_history'] = [from_qty]
        if 'timestamp_history' not in slot:
            slot['timestamp_history'] = [datetime.now()]
            
        slot['quantity_history'].append(to_qty)
        slot['timestamp_history'].append(datetime.now())
        # Zachowaj tylko ostatnie 50 punktów danych
        if len(slot['quantity_history']) > 50:
            slot['quantity_history'] = slot['quantity_history'][-50:]
            slot['timestamp_history'] = slot['timestamp_history'][-50:]

        trade = {
            'timestamp': datetime.now(),
            'from_token': from_token,
            'to_token': target_token,
            'from_quantity': from_qty,
            'to_quantity': to_qty,
            'slot': slot_idx,
            'max_gain': max_gain,
            'reason': f'Trailing Stop triggered (max_gain={max_gain:.2f}%)'
        }

        slot['token'] = target_token
        slot['quantity'] = to_qty

        for token in self.tokens_to_track:
            if token == target_token:
                slot['top_equivalent'][token] = to_qty
            else:
                new_equiv = self.calculate_equivalent(target_token, token, to_qty)
                slot['top_equivalent'][token] = new_equiv if new_equiv is not None else 0.0
                slot['current_gain'][token] = 0.0
                slot['max_gain'][token] = 0.0

        st.session_state.trades.append(trade)
        self.save_data()

        st.toast(f"🔁 SWAP: {from_token} → {target_token} (Slot {slot_idx + 1})", icon="✅")
        st.success(f"💰 Executed SWAP: {from_token} → {target_token} | max_gain observed: {max_gain:.2f}%")

    # ================== UI ==================
    def render_sidebar(self):
        with st.sidebar:
            st.title("⚙️ Config 24/7")
            if hasattr(st.session_state, 'app_start_time'):
                uptime = datetime.now() - st.session_state.app_start_time
                st.metric("Uptime", f"{uptime.seconds//3600}h {(uptime.seconds%3600)//60}m")
            st.metric("Slots", f"{len(st.session_state.portfolio)}/5")
            st.metric("Trades", len(st.session_state.trades))
            st.metric("Price updates", st.session_state.price_updates)
            st.metric("Status", "🟢 RUNNING" if st.session_state.tracking else "🟡 PAUSED")
            
            st.subheader("🎯 Trailing Stop Levels")
            # Edytowalne poziomy trailing stop
            ts_levels = st.session_state.trailing_stop_levels
            new_levels = {}
            
            col1, col2 = st.columns(2)
            with col1:
                new_levels[0.5] = st.number_input("0.5% gain → TS:", value=float(ts_levels.get(0.5, 0.2)), min_value=0.1, max_value=5.0, step=0.1, key="ts_0.5")
                new_levels[1.0] = st.number_input("1.0% gain → TS:", value=float(ts_levels.get(1.0, 0.5)), min_value=0.1, max_value=5.0, step=0.1, key="ts_1.0")
            with col2:
                new_levels[2.0] = st.number_input("2.0% gain → TS:", value=float(ts_levels.get(2.0, 1.0)), min_value=0.1, max_value=5.0, step=0.1, key="ts_2.0")
                new_levels[5.0] = st.number_input("5.0% gain → TS:", value=float(ts_levels.get(5.0, 2.0)), min_value=0.1, max_value=5.0, step=0.1, key="ts_5.0")
            
            st.session_state.trailing_stop_levels = new_levels

            if not st.session_state.portfolio:
                st.subheader("💰 Init Portfolio")
                usdt_amount = st.number_input("USDT amount:", min_value=10.0, value=1000.0, step=100.0)
                available_tokens = list(st.session_state.prices.keys()) if hasattr(st.session_state, 'prices') else []
                available_tokens.sort()
                selected_tokens = st.multiselect("Select 5 tokens:", available_tokens,
                                                 default=available_tokens[:5], max_selections=5)
                if st.button("🏁 Initialize Portfolio", use_container_width=True):
                    if len(selected_tokens)==5:
                        self.initialize_portfolio_from_usdt(usdt_amount, selected_tokens)
                        st.rerun()
                    else:
                        st.error("❌ Select exactly 5 tokens")
            
            st.subheader("🎮 Controls")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("▶ Start", use_container_width=True) and not st.session_state.tracking:
                    if st.session_state.prices:
                        st.session_state.tracking = True
                        st.session_state.app_start_time = datetime.now()
                        st.rerun()
                    else:
                        st.error("❌ No prices")
            with col2:
                if st.button("⏹ Stop", use_container_width=True) and st.session_state.tracking:
                    st.session_state.tracking = False
                    st.rerun()
            
            st.subheader("💾 Data")
            if os.path.exists(self.data_file):
                file_time = os.path.getmtime(self.data_file)
                file_size = os.path.getsize(self.data_file)/1024
                st.caption(f"📁 Last save: {datetime.fromtimestamp(file_time).strftime('%H:%M:%S')}")
                st.caption(f"📊 Data size: {file_size:.1f} KB")
                if st.button("🗑️ Clear all data", use_container_width=True):
                    self.clear_all_data()

    def render_portfolio_overview(self):
        st.header("📊 Portfolio Overview")
        if not st.session_state.portfolio:
            st.info("👈 Initialize portfolio from sidebar")
            return
        
        cols = st.columns(len(st.session_state.portfolio))
        for idx, (col, slot) in enumerate(zip(cols, st.session_state.portfolio)):
            with col:
                if slot['token'] in st.session_state.prices:
                    current_usdt = slot['quantity'] * st.session_state.prices[slot['token']].bid_price
                    baseline_usdt = slot.get('baseline_usdt', current_usdt)
                    change_percent = ((current_usdt - baseline_usdt) / baseline_usdt * 100) if baseline_usdt > 0 else 0
                    
                    # Okno z metrykami
                    with st.container():
                        st.subheader(f"Slot {idx+1} - {slot['token']}")
                        
                        # Wyświetlanie z odpowiednim kolorem i strzałką
                        delta_color = "normal"
                        if change_percent > 0:
                            delta_color = "normal"
                            delta_value = f"↑ {change_percent:.2f}%"
                        elif change_percent < 0:
                            delta_color = "inverse" 
                            delta_value = f"↓ {abs(change_percent):.2f}%"
                        else:
                            delta_value = f"{change_percent:.2f}%"
                        
                        st.metric(
                            label="Current Value",
                            value=f"{current_usdt:.2f} USDT",
                            delta=delta_value,
                            delta_color=delta_color
                        )
                        
                        st.metric(
                            label="Quantity",
                            value=f"{slot['quantity']:.6f} {slot['token']}"
                        )
                        
                        st.metric(
                            label="Baseline",
                            value=f"{baseline_usdt:.2f} USDT"
                        )
                        
                        # Wykres ilościowy z zaokrągleniem
                        if 'quantity_history' in slot and len(slot['quantity_history']) > 1:
                            chart_data = pd.DataFrame({
                                'Quantity': slot['quantity_history'],
                                'Time': range(len(slot['quantity_history']))
                            })
                            st.line_chart(
                                chart_data, 
                                x='Time', 
                                y='Quantity',
                                use_container_width=True
                            )
                        else:
                            st.info("No history data yet")
                else:
                    st.error(f"❌ No price data for {slot['token']}")

    def render_trailing_matrix(self):
        st.header("🎯 Trailing Stop Matrix")
        for idx, slot in enumerate(st.session_state.portfolio):
            self.render_slot_with_history(idx, slot)

    def render_slot_with_history(self, slot_idx: int, slot: dict):
        st.subheader(f"🔷 Slot {slot_idx + 1}: {slot['token']} ({slot['quantity']:.6f})")
        self.render_slot_matrix(slot_idx, slot)
        self.render_slot_trade_history(slot_idx)

    def render_slot_matrix(self, slot_idx: int, slot: dict):
        matrix_data = []
        best_pair_gain = -999.0
        best_pair_token = None

        for token in self.tokens_to_track:
            current_max_gain = slot.get('max_gain', {}).get(token, 0.0)
            if current_max_gain > best_pair_gain:
                best_pair_gain = current_max_gain
                best_pair_token = token

        for token in self.tokens_to_track:
            current_eq = self.calculate_equivalent(slot['token'], token, slot['quantity'])
            try:
                current_eq = float(current_eq)
            except:
                current_eq = 0.0
            baseline_eq = slot.get('baseline', {}).get(token, current_eq)
            try:
                baseline_eq = float(baseline_eq)
            except:
                baseline_eq = 0.0
            top_eq = slot.get('top_equivalent', {}).get(token, current_eq)
            try:
                top_eq = float(top_eq)
            except:
                top_eq = 0.0

            current_gain = slot.get('current_gain', {}).get(token, 0.0)
            max_gain = slot.get('max_gain', {}).get(token, 0.0)

            change_from_baseline = ((current_eq - baseline_eq) / baseline_eq * 100) if baseline_eq > 0 else 0.0
            change_from_top = ((current_eq - top_eq) / top_eq * 100) if top_eq > 0 else 0.0

            # Ulepszony status z adnotacjami
            status = ""
            if token == slot['token']:
                status = "🔵 Current Token"
            elif token == best_pair_token and best_pair_gain >= 0.5:
                status = "⭐ Best Candidate"
            elif change_from_top >= -1:
                status = "🟢 Good Position"
            elif change_from_top >= -3:
                status = "🟡 Watch"
            else:
                status = "🔴 Poor Position"

            matrix_data.append({
                'Token': token,
                'Aktualny': current_eq,
                'Początkowy': baseline_eq,
                'Δ Od początku': change_from_baseline,
                'Top': top_eq,
                'Δ Od top': change_from_top,
                'Current Gain': current_gain,
                'Max Wzrost': max_gain,
                'Status': status
            })

        df = pd.DataFrame(matrix_data)
        
        # Sortowalna tabela
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                'Aktualny': st.column_config.NumberColumn(format="%.6f"),
                'Początkowy': st.column_config.NumberColumn(format="%.6f"),
                'Δ Od początku': st.column_config.NumberColumn(format="%+.2f%%"),
                'Top': st.column_config.NumberColumn(format="%.6f"),
                'Δ Od top': st.column_config.NumberColumn(format="%+.2f%%"),
                'Current Gain': st.column_config.NumberColumn(format="%+.2f%%"),
                'Max Wzrost': st.column_config.NumberColumn(format="%+.2f%%"),
            }
        )

    def render_slot_trade_history(self, idx):
        trades = [t for t in st.session_state.trades if t['slot']==idx]
        if trades:
            st.subheader(f"📋 History Slot {idx+1}")
            data = []
            for t in trades[-10:]:
                data.append({
                    'Data': t['timestamp'].strftime('%H:%M:%S'),
                    'Z': t['from_token'],
                    'Na': t['to_token'],
                    'Ilość': f"{t['to_quantity']:.6f}",
                    'Max Wzrost': f"{t['max_gain']:.2f}%",
                    'Powód': t['reason']
                })
            st.dataframe(pd.DataFrame(data), use_container_width=True)
        else:
            st.caption("📝 No trades for this slot yet")

    def keep_app_alive(self):
        if not hasattr(st.session_state, 'last_active_ping'):
            st.session_state.last_active_ping = datetime.now()
        if (datetime.now() - st.session_state.last_active_ping).seconds>120:
            st.session_state.last_active_ping = datetime.now()

    def clear_all_data(self):
        st.session_state.portfolio = []
        st.session_state.trades = []
        st.session_state.tracking = False
        if os.path.exists(self.data_file):
            os.remove(self.data_file)
        st.success("🗑️ All data cleared")
        st.rerun()

    def run(self):
        try:
            self.init_session_state()
            # Usunięty tytuł strony
            st.markdown("---")
            self.keep_app_alive()
            self.render_sidebar()
            if st.session_state.prices:
                self.render_portfolio_overview()
                if st.session_state.portfolio:
                    self.render_trailing_matrix()
                    if st.session_state.tracking:
                        st.success(f"🟢 TRACKING ACTIVE | Last update: {datetime.now().strftime('%H:%M:%S')}")
                        self.update_real_prices()
                        self.check_and_execute_trades()
                        time.sleep(3)
                        st.rerun()
            else:
                st.error("🚫 No price data - checking connection...")
                if st.button("🔄 Refresh prices") or st.session_state.tracking:
                    st.session_state.prices = self.get_initial_prices()
                    time.sleep(2)
                    st.rerun()
        except Exception as e:
            st.error(f"🔴 Critical error: {e}")
            st.info("🔄 Auto-restart in 10 seconds...")
            time.sleep(10)
            st.rerun()

# ================== Run App ==================
if __name__=="__main__":
    app = CryptoTrailingStopApp()
    app.run()
