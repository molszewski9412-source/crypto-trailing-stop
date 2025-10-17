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
            st.error(f"❌ {e}")
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
                'usdt_value': quantity * st.session_state.prices[token].bid_price
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
                return {'portfolio': data.get('portfolio', []), 'trades': trades}
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
        for gain_threshold, ts in sorted(self.trailing_stop_levels.items()):
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
        """
        Wykonuje swap pełnego slotu.
        1. Oblicza dokładną ilość target_token uzyskaną w swapie.
        2. Aktualizuje top_equivalent dla swapowanego tokenu dokładnie jako uzyskana ilość.
        3. Dla pozostałych tokenów przelicza top_equivalent względem nowego tokenu i jego ilości.
        4. Resetuje current_gain i max_gain tylko dla tokenów innych niż swapowany.
        5. Zapisuje trade w historii.
        """

        from_token = slot['token']
        from_qty = slot['quantity']

        # Dokładna ilość uzyskana w swapie (uwzględnia fee)
        to_qty = self.calculate_equivalent(from_token, target_token, from_qty)
        if to_qty <= 0:
            st.warning("⚠️ Swap aborted: computed to_qty <= 0")
            return

        # Zapis trade przed zmianą stanu slotu
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

        # Wykonanie swapu: zamiana tokenu i ilości w slocie
        slot['token'] = target_token
        slot['quantity'] = to_qty

        # -------------------------
        # TOP EQUIVALENT LOGIKA
        # -------------------------

        for token in self.tokens_to_track:
            if token == target_token:
                # Swapowany token → top_equivalent = dokładna ilość uzyskana w swapie
                slot['top_equivalent'][token] = to_qty
            else:
                # Pozostałe tokeny → przeliczamy ekwiwalent względem nowego tokenu i jego ilości
                new_equiv = self.calculate_equivalent(target_token, token, to_qty)
                slot['top_equivalent'][token] = new_equiv if new_equiv is not None else 0.0
                # Resetujemy gains tylko dla innych tokenów
                slot['current_gain'][token] = 0.0
                slot['max_gain'][token] = 0.0

        # Dodanie trade do historii
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
            st.subheader("🔍 Diagnostics")
            if st.button("🧪 Test MEXC connection"):
                ok, msg = self.test_connection()
                st.success(msg) if ok else st.error(msg)
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
            st.subheader("🎯 Trailing Stop Levels")
            for gain, stop in sorted(self.trailing_stop_levels.items()):
                st.text(f"💰 {gain}% gain → trailing stop {stop}% below max")

    def render_portfolio_overview(self):
        st.header("📊 Portfolio Overview")
        if not st.session_state.portfolio:
            st.info("👈 Initialize portfolio from sidebar")
            return
        cols = st.columns(len(st.session_state.portfolio))
        for idx, (col, slot) in enumerate(zip(cols, st.session_state.portfolio)):
            with col:
                if slot['token'] in st.session_state.prices:
                    val = slot['quantity']*st.session_state.prices[slot['token']].bid_price
                    st.metric(f"Slot {idx+1} - {slot['token']}", f"{slot['quantity']:.6f}", f"{val:.2f} USDT")
                else:
                    st.metric(f"Slot {idx+1} - {slot['token']}", f"{slot['quantity']:.6f}", "N/A")

    def render_trailing_matrix(self):
        st.header("🎯 Trailing Stop Matrix")
        for idx, slot in enumerate(st.session_state.portfolio):
            self.render_slot_with_history(idx, slot)

    def render_slot_with_history(self, slot_idx: int, slot: dict):
        # Wyświetlamy nagłówek dla slotu
        st.subheader(f"🔷 Slot {slot_idx + 1}: {slot['token']} ({slot['quantity']:.6f})")
        
        # Wyświetlamy macierz z aktualnymi danymi
        self.render_slot_matrix(slot_idx, slot)
        
        # Wyświetlamy historię swapów dla slotu
        self.render_slot_trade_history(slot_idx)

    def render_slot_matrix(self, idx, slot):
        data = []
        # safe best token selection
        if slot['max_gain']:
            best_token = max(slot['max_gain'], key=lambda k: slot['max_gain'][k])
            best_gain = slot['max_gain'][best_token]
        else:
            best_token = slot['token']
            best_gain = 0.0
        for t in self.tokens_to_track:
            cur = self.calculate_equivalent(slot['token'], t, slot['quantity'])
            base = slot['baseline'].get(t, cur)
            top = slot['top_equivalent'].get(t, cur)
            cur_gain = slot['current_gain'].get(t, 0.0)
            max_gain = slot['max_gain'].get(t, 0.0)
            delta_base = ((cur - base) / base * 100) if base > 0 else 0
            delta_top = ((cur - top) / top * 100) if top > 0 else 0
            status = (
                "🔵" if t == slot['token'] else
                "⭐" if t == best_token and best_gain >= 0.5 else
                "🟢" if delta_top >= -1 else
                "🟡" if delta_top >= -3 else
                "🔴"
            )
            data.append({
                'Token': t,
                'Aktualny': f"{cur:.6f}",
                'Początkowy': f"{base:.6f}",
                'Δ Od początku': f"{delta_base:+.2f}%",
                'Top': f"{top:.6f}",
                'Δ Od top': f"{delta_top:+.2f}%",
                'Current Gain': f"{cur_gain:+.2f}%",
                'Max Wzrost': f"{max_gain:+.2f}%",
                'Status': status
            })
        df = pd.DataFrame(data)
        # width='stretch' oraz height=None powoduje, że tabela rozciąga się na całą zawartość kontenera
        st.dataframe(df, width='stretch', height=None)

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
            st.title("🚀 Crypto Trailing Stop Matrix - 24/7")
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
                st.error("🚫 No price data")
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
