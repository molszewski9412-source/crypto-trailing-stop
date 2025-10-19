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
        
        # Prawidłowe tickery z MEXC (tylko te które mają pary USDT)
        self.tokens_to_track = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT', 'XRPUSDT', 'DOTUSDT', 
            'DOGEUSDT', 'AVAXUSDT', 'LTCUSDT', 'LINKUSDT', 'ATOMUSDT', 'XLMUSDT', 
            'BCHUSDT', 'ALGOUSDT', 'FILUSDT', 'ETCUSDT', 'XTZUSDT', 'AAVEUSDT', 
            'COMPUSDT', 'UNIUSDT', 'CRVUSDT', 'SUSHIUSDT', 'YFIUSDT', 'SNXUSDT', 
            '1INCHUSDT', 'ZRXUSDT', 'TRXUSDT', 'VETUSDT', 'MATICUSDT', 'NEARUSDT', 
            'ALICEUSDT', 'FTMUSDT', 'SANDUSDT', 'MANAUSDT', 'ENJUSDT', 'CHZUSDT', 
            'AXSUSDT', 'GALAUSDT', 'APEUSDT', 'LDOUSDT', 'ARBUSDT', 'OPUSDT', 
            'APTUSDT', 'SUIUSDT', 'SEIUSDT', 'INJUSDT', 'RNDRUSDT', 'IMXUSDT'
        ]

    def get_token_symbol(self, full_symbol: str) -> str:
        """Konwertuje pełny symbol (np. BTCUSDT) na ticker tokena (BTC)"""
        if full_symbol.endswith('USDT'):
            return full_symbol[:-4]
        return full_symbol

    def get_full_symbol(self, token: str) -> str:
        """Konwertuje ticker tokena (BTC) na pełny symbol (BTCUSDT)"""
        if not token.endswith('USDT'):
            return f"{token}USDT"
        return token

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
            
            # Filtruj tylko pary które nas interesują
            available_pairs = {item['symbol']: item for item in data if item['symbol'] in self.tokens_to_track}
            
            for full_symbol in self.tokens_to_track:
                if full_symbol in available_pairs:
                    try:
                        item = available_pairs[full_symbol]
                        bid = float(item['bidPrice'])
                        ask = float(item['askPrice'])
                        if bid <= 0 or ask <= 0 or bid > ask:
                            continue
                        token_symbol = self.get_token_symbol(full_symbol)
                        prices[token_symbol] = TokenInfo(
                            symbol=token_symbol, 
                            bid_price=bid, 
                            ask_price=ask, 
                            last_update=datetime.now()
                        )
                    except (KeyError, ValueError, TypeError) as e:
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
        
        # Konwertuj wybrane tokeny na pełne symbole dla sprawdzenia dostępności
        selected_full_symbols = [self.get_full_symbol(token) for token in selected_tokens]
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
            
            for full_symbol in self.tokens_to_track:
                target_token = self.get_token_symbol(full_symbol)
                if target_token in st.session_state.prices:
                    eq = self.calculate_equivalent(token, target_token, quantity)
                    baseline[target_token] = eq  # Zapisujemy tylko raz przy inicjacji
                    top_equivalent[target_token] = eq  # Top początkowo równy baseline
                    current_gain[target_token] = 0.0
                    max_gain[target_token] = 0.0
            
            slot = {
                'token': token,
                'quantity': quantity,
                'baseline': baseline,  # NIGDY nie aktualizowane po inicjacji
                'top_equivalent': top_equivalent,  # Aktualizowane tylko przy swapie
                'current_gain': current_gain,
                'max_gain': max_gain,
                'usdt_value': quantity * st.session_state.prices[token].bid_price,
                'baseline_quantity': quantity,  # Baseline jako ilość tokena
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
                if 'baseline_quantity' not in slot:
                    slot['baseline_quantity'] = slot['quantity']

    # ================== Load/Save ==================
    def load_data(self):
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Sprawdź czy dane mają poprawny format
                if not isinstance(data, dict):
                    st.error("❌ Invalid data format in file")
                    return {'portfolio': [], 'trades': []}
                
                trades = []
                for t in data.get('trades', []):
                    try:
                        trades.append({
                            'timestamp': datetime.fromisoformat(t['timestamp']),
                            'from_token': t['from_token'],
                            'to_token': t['to_token'],
                            'from_quantity': float(t['from_quantity']),
                            'to_quantity': float(t['to_quantity']),
                            'slot': int(t['slot']),
                            'max_gain': float(t.get('max_gain', 0.0)),
                            'reason': t.get('reason', '')
                        })
                    except (KeyError, ValueError, TypeError) as e:
                        st.warning(f"⚠️ Skipping invalid trade record: {e}")
                        continue
                
                # Dodaj brakujące pola do załadowanego portfolio
                portfolio = data.get('portfolio', [])
                for slot in portfolio:
                    if not isinstance(slot, dict):
                        continue
                    if 'quantity_history' not in slot:
                        slot['quantity_history'] = [slot.get('quantity', 0)]
                    if 'timestamp_history' not in slot:
                        slot['timestamp_history'] = [datetime.now()]
                    if 'baseline_quantity' not in slot:
                        slot['baseline_quantity'] = slot.get('quantity', 0)
                
                return {'portfolio': portfolio, 'trades': trades}
                
        except json.JSONDecodeError as e:
            st.error(f"❌ JSON decode error: {e}")
            # Jeśli plik jest uszkodzony, utwórz backup i zacznij od nowa
            self.create_backup_and_reset()
        except Exception as e:
            st.error(f"❌ Error loading data: {e}")
        
        return {'portfolio': [], 'trades': []}

    def create_backup_and_reset(self):
        """Tworzy backup uszkodzonego pliku i resetuje dane"""
        try:
            if os.path.exists(self.data_file):
                backup_file = f"{self.data_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                os.rename(self.data_file, backup_file)
                st.warning(f"⚠️ Created backup of corrupted file: {backup_file}")
        except Exception as e:
            st.error(f"❌ Backup creation failed: {e}")

    def save_data(self):
        try:
            # Przygotuj dane do zapisu - konwersja na typy podstawowe
            data = {
                'portfolio': [],
                'trades': []
            }
            
            # Zapisz portfolio
            for slot in st.session_state.portfolio:
                portfolio_slot = {
                    'token': slot['token'],
                    'quantity': float(slot['quantity']),
                    'baseline': {k: float(v) for k, v in slot['baseline'].items()},
                    'top_equivalent': {k: float(v) for k, v in slot['top_equivalent'].items()},
                    'current_gain': {k: float(v) for k, v in slot['current_gain'].items()},
                    'max_gain': {k: float(v) for k, v in slot['max_gain'].items()},
                    'usdt_value': float(slot.get('usdt_value', 0)),
                    'baseline_quantity': float(slot.get('baseline_quantity', 0)),
                    'quantity_history': [float(q) for q in slot.get('quantity_history', [])],
                    'timestamp_history': [t.isoformat() for t in slot.get('timestamp_history', [])]
                }
                data['portfolio'].append(portfolio_slot)
            
            # Zapisz trades
            for t in st.session_state.trades:
                trade = {
                    'timestamp': t['timestamp'].isoformat(),
                    'from_token': t['from_token'],
                    'to_token': t['to_token'],
                    'from_quantity': float(t['from_quantity']),
                    'to_quantity': float(t['to_quantity']),
                    'slot': int(t['slot']),
                    'max_gain': float(t.get('max_gain', 0.0)),
                    'reason': t.get('reason', '')
                }
                data['trades'].append(trade)
            
            # Zapisz do pliku
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            st.error(f"❌ Save error: {e}")

    # ================== Equivalents ==================
    def calculate_equivalent(self, from_token: str, to_token: str, quantity: float) -> float:
        if from_token == to_token:
            return quantity * (1 - self.fee_rate)
        
        prices = st.session_state.prices
        if from_token not in prices or to_token not in prices:
            return 0.0
        
        try:
            # Używamy prawidłowych cen bid/ask
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
            
            for full_symbol in self.tokens_to_track:
                target_token = self.get_token_symbol(full_symbol)
                
                if target_token == from_token or target_token in current_tokens:
                    continue
                
                # Oblicz aktualny ekwiwalent
                current_eq = self.calculate_equivalent(from_token, target_token, qty)
                if current_eq <= 0:
                    continue
                
                # Pobierz top equivalent (aktualizowany tylko przy swapie)
                top_eq = slot['top_equivalent'].get(target_token, current_eq)
                
                # Oblicz gain % od top
                gain_from_top = ((current_eq - top_eq) / top_eq * 100) if top_eq > 0 else 0
                
                # Aktualizuj current_gain
                slot['current_gain'][target_token] = gain_from_top
                
                # Aktualizuj max_gain tylko jeśli gain_from_top jest wyższy niż 0.5%
                # i jest wyższy niż poprzedni max_gain
                prev_max = slot['max_gain'].get(target_token, 0.0)
                if gain_from_top >= 0.5 and gain_from_top > prev_max:
                    slot['max_gain'][target_token] = gain_from_top
                
                current_max_gain = slot['max_gain'].get(target_token, 0.0)
                
                # Sprawdź warunki trailing stop tylko jeśli max_gain >= 0.5%
                if current_max_gain >= 0.5:
                    ts = self.get_trailing_stop_level(current_max_gain)
                    swap_threshold = current_max_gain - ts
                    
                    # Sprawdź czy gain spadł poniżej progu trailing stop
                    if gain_from_top <= swap_threshold:
                        swap_candidates.append({
                            'target_token': target_token,
                            'current_eq': current_eq,
                            'max_gain': current_max_gain,
                            'gain_from_top': gain_from_top
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
        if len(slot['quantity_history']) > 50:
            slot['quantity_history'] = slot['quantity_history'][-50:]
            slot['timestamp_history'] = slot['timestamp_history'][-50:]

        # ZAPIS TRADE PRZED AKTUALIZACJĄ TOP
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

        # WYMIANA TOKENA
        slot['token'] = target_token
        slot['quantity'] = to_qty

        # AKTUALIZACJA TOP EQUIVALENT - TYLKO PRZY SWAPIE!
        for full_symbol in self.tokens_to_track:
            token = self.get_token_symbol(full_symbol)
            if token == target_token:
                # Dla tokena docelowego: top = dokładna ilość uzyskana w swapie
                slot['top_equivalent'][token] = to_qty
            else:
                # Dla innych tokenów: oblicz nowy ekwiwalent i sprawdź czy jest wyższy niż dotychczasowy top
                new_equiv = self.calculate_equivalent(target_token, token, to_qty)
                current_top = slot['top_equivalent'].get(token, 0.0)
                if new_equiv > current_top:
                    slot['top_equivalent'][token] = new_equiv
            
            # Reset gains dla wszystkich tokenów
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
                                                 default=available_tokens[:5] if len(available_tokens) >= 5 else available_tokens, 
                                                 max_selections=5)
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
        
        # Jeden slot pod drugim zamiast w linii
        for idx, slot in enumerate(st.session_state.portfolio):
            with st.container():
                st.subheader(f"Slot {idx+1} - {slot['token']}")
                
                if slot['token'] in st.session_state.prices:
                    current_usdt = slot['quantity'] * st.session_state.prices[slot['token']].bid_price
                    baseline_quantity = slot.get('baseline_quantity', slot['quantity'])
                    current_quantity = slot['quantity']
                    
                    # Oblicz zmianę ilości od baseline
                    quantity_change = ((current_quantity - baseline_quantity) / baseline_quantity * 100) if baseline_quantity > 0 else 0
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        delta_color = "normal" if quantity_change >= 0 else "inverse"
                        delta_value = f"{quantity_change:+.2f}%"
                        st.metric(
                            label="Current Value",
                            value=f"{current_usdt:.2f} USDT",
                            delta=delta_value,
                            delta_color=delta_color
                        )
                    
                    with col2:
                        st.metric(
                            label="Current Quantity",
                            value=f"{current_quantity:.6f} {slot['token']}"
                        )
                    
                    with col3:
                        st.metric(
                            label="Baseline Quantity",
                            value=f"{baseline_quantity:.6f} {slot['token']}"
                        )
                    
                    with col4:
                        st.metric(
                            label="Quantity Change",
                            value=f"{quantity_change:+.2f}%"
                        )
                        
                else:
                    st.error(f"❌ No price data for {slot['token']}")
                
                st.markdown("---")

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

        # Znajdź najlepszą parę
        for full_symbol in self.tokens_to_track:
            token = self.get_token_symbol(full_symbol)
            current_max_gain = slot.get('max_gain', {}).get(token, 0.0)
            if current_max_gain > best_pair_gain:
                best_pair_gain = current_max_gain
                best_pair_token = token

        for full_symbol in self.tokens_to_track:
            token = self.get_token_symbol(full_symbol)
            
            # Oblicz aktualny ekwiwalent
            current_eq = self.calculate_equivalent(slot['token'], token, slot['quantity'])
            current_eq = float(current_eq) if current_eq else 0.0
            
            # Pobierz baseline (NIGDY nieaktualizowany)
            baseline_eq = slot.get('baseline', {}).get(token, current_eq)
            baseline_eq = float(baseline_eq) if baseline_eq else 0.0
            
            # Pobierz top (aktualizowany tylko przy swapie)
            top_eq = slot.get('top_equivalent', {}).get(token, current_eq)
            top_eq = float(top_eq) if top_eq else 0.0

            current_gain = slot.get('current_gain', {}).get(token, 0.0)  # Gain % od top
            max_gain = slot.get('max_gain', {}).get(token, 0.0)         # Max gain % od top

            # Oblicz zmianę od baseline (tylko do informacji)
            change_from_baseline = ((current_eq - baseline_eq) / baseline_eq * 100) if baseline_eq > 0 else 0.0

            # Status
            status = ""
            if token == slot['token']:
                status = "🔵 Current Token"
            elif token == best_pair_token and best_pair_gain >= 0.5:
                status = f"⭐ Best Candidate ({best_pair_gain:.2f}%)"
            elif current_gain >= 0:
                status = "🟢 Above Top"
            elif current_gain >= -1:
                status = "🟢 Good Position"
            elif current_gain >= -3:
                status = "🟡 Watch"
            else:
                status = "🔴 Poor Position"

            matrix_data.append({
                'Token': token,
                'Aktualny': current_eq,
                'Początkowy': baseline_eq,
                'Δ Od początku': change_from_baseline,
                'Top': top_eq,
                'Gain %': current_gain,  # Gain % od top
                'Max Wzrost': max_gain,  # Max gain % od top
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
                'Gain %': st.column_config.NumberColumn(format="%+.2f%%"),
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