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
    page_title="Crypto Trailing Stop Matrix - 24/7",
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
        self.fee_rate = 0.00025
        # trailing_stop_levels keys and values are in percent (e.g. 0.5 -> 0.2 means 0.5% -> 0.2%)
        self.trailing_stop_levels = {0.5: 0.2, 1.0: 0.5, 2.0: 1.0, 5.0: 2.0}
        self.data_file = "trailing_stop_data.json"
        
        # Lista tokenów - sprawdzone na MEXC (możesz edytować)
        self.tokens_to_track = [
            'BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'XRP', 'DOT', 'DOGE', 'AVAX', 'LTC',
            'LINK', 'ATOM', 'XLM', 'BCH', 'ALGO', 'FIL', 'ETC', 'XTZ', 'AAVE', 'COMP',
            'UNI', 'CRV', 'SUSHI', 'YFI', 'SNX', '1INCH', 'ZRX', 'TRX', 'VET', 'ONE',
            'CELO', 'RSR', 'NKN', 'STORJ', 'DODO', 'KAVA', 'RUNE', 'SAND', 'MANA', 'ENJ',
            'CHZ', 'ALICE', 'NEAR', 'ARB', 'OP', 'APT', 'SUI', 'SEI', 'INJ', 'RENDER'
        ]
        
    def get_trailing_stop_level(self, max_gain: float) -> float:
        """Return trailing stop level (in percent) for a given max_gain (in percent)."""
        current_ts = 0.0
        # iterate thresholds in order
        for gain_threshold, ts_level in sorted(self.trailing_stop_levels.items()):
            if max_gain >= gain_threshold:
                current_ts = ts_level
        return current_ts
        
    def test_connection(self):
        """Test connection to MEXC API"""
        try:
            url = "https://api.mexc.com/api/v3/ping"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return True, "✅ Connection to MEXC API OK"
            else:
                return False, f"❌ MEXC API returned status: {response.status_code}"
        except Exception as e:
            return False, f"❌ Connection error: {e}"

    def get_all_prices_bulk(self) -> Dict[str, TokenInfo]:
        """Fetch top-of-book for all USDT pairs from MEXC"""
        prices = {}
        
        try:
            url = "https://api.mexc.com/api/v3/ticker/bookTicker"
            response = requests.get(url, timeout=15)
            
            if response.status_code == 200:
                all_data = response.json()
                if not all_data:
                    st.error("❌ MEXC returned empty response")
                    return {}
                
                # Filter USDT pairs
                usdt_pairs = {}
                for item in all_data:
                    symbol = item.get('symbol', '')
                    if symbol.endswith('USDT'):
                        usdt_pairs[symbol] = item
                
                found_tokens = 0
                problematic_tokens = []
                
                for token in self.tokens_to_track:
                    symbol = f"{token}USDT"
                    if symbol in usdt_pairs:
                        data = usdt_pairs[symbol]
                        try:
                            bid_price = float(data['bidPrice'])
                            ask_price = float(data['askPrice'])
                            
                            # validate
                            if bid_price > 0 and ask_price > 0 and bid_price <= ask_price:
                                prices[token] = TokenInfo(
                                    symbol=token,
                                    bid_price=bid_price,
                                    ask_price=ask_price,
                                    last_update=datetime.now()
                                )
                                found_tokens += 1
                            else:
                                problematic_tokens.append(f"{token}(bad prices)")
                        except (ValueError, KeyError):
                            problematic_tokens.append(f"{token}(conversion error)")
                    else:
                        problematic_tokens.append(token)
                
                if len(problematic_tokens) > 40:
                    st.warning(f"⚠️ Missing prices for {len(problematic_tokens)} tokens")
                
                if prices:
                    return prices
                else:
                    st.error("🚫 Could not fetch any prices")
                    return {}
                    
            else:
                st.error(f"❌ HTTP error {response.status_code} from MEXC")
                return {}
                
        except requests.exceptions.Timeout:
            st.error("⏰ Timeout - MEXC API not responding")
            return {}
        except requests.exceptions.ConnectionError:
            st.error("🌐 Connection error - check your internet")
            return {}
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")
            return {}

    def get_initial_prices(self) -> Dict[str, TokenInfo]:
        """Get initial prices"""
        return self.get_all_prices_bulk()

    def update_real_prices(self):
        """Update prices with simple rate-limiting for 24/7 usage"""
        if hasattr(st.session_state, 'last_price_update'):
            time_diff = (datetime.now() - st.session_state.last_price_update).seconds
            if time_diff < 3:
                return
        
        new_prices = self.get_all_prices_bulk()
        if new_prices:
            st.session_state.prices = new_prices
            st.session_state.price_updates += 1
            st.session_state.last_tracking_time = datetime.now()
            st.session_state.last_price_update = datetime.now()

    def initialize_portfolio_from_usdt(self, usdt_amount: float, selected_tokens: List[str]):
        """Initialize portfolio dividing USDT across 5 tokens"""
        if len(selected_tokens) != 5:
            st.error("❌ Select exactly 5 tokens")
            return False
            
        if usdt_amount <= 0:
            st.error("❌ USDT amount must be > 0")
            return False
            
        available_tokens = []
        for token in selected_tokens:
            if token in st.session_state.prices:
                available_tokens.append(token)
        
        if len(available_tokens) < 5:
            st.error(f"❌ Not enough tokens with prices: {len(available_tokens)}/5")
            return False
        
        st.session_state.portfolio = []
        st.session_state.trades = []
        
        usdt_per_slot = usdt_amount / 5
        
        for token in available_tokens:
            token_price = st.session_state.prices[token].ask_price
            # buy full slot quantity (apply sell fee on later sells via calculate_equivalent)
            quantity = (usdt_per_slot / token_price) * (1 - self.fee_rate)
            
            baseline = {}
            top_equivalent = {}
            current_gain = {}
            max_gain = {}
            
            # For baseline/top we store absolute equivalents for the FULL slot quantity.
            for target_token in self.tokens_to_track:
                if target_token in st.session_state.prices:
                    equivalent = self.calculate_equivalent(token, target_token, quantity)
                    baseline[target_token] = equivalent
                    top_equivalent[target_token] = equivalent
                    current_gain[target_token] = 0.0
                    max_gain[target_token] = 0.0
            
            new_slot = {
                'token': token,
                'quantity': quantity,
                'baseline': baseline,
                'top_equivalent': top_equivalent,
                'current_gain': current_gain,
                'max_gain': max_gain,
                'usdt_value': quantity * st.session_state.prices[token].bid_price
            }
            
            st.session_state.portfolio.append(new_slot)
        
        self.save_data()
        st.success(f"✅ Portfolio initialized: {usdt_amount} USDT → 5 slots")
        return True

    def init_session_state(self):
        """Initialize streamlit session state and load stored data if present"""
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
        if 'last_price_update' not in st.session_state:
            st.session_state.last_price_update = datetime.now()
        if 'app_start_time' not in st.session_state:
            st.session_state.app_start_time = datetime.now()

    def load_data(self) -> dict:
        """Load saved data from file"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                
                loaded_trades = []
                for trade in data.get('trades', []):
                    loaded_trades.append({
                        'timestamp': datetime.fromisoformat(trade['timestamp']),
                        'from_token': trade['from_token'],
                        'to_token': trade['to_token'],
                        'from_quantity': trade['from_quantity'],
                        'to_quantity': trade['to_quantity'],
                        'slot': trade['slot'],
                        'max_gain': trade.get('max_gain', 0.0),
                        'reason': trade.get('reason', '')
                    })
                
                return {
                    'portfolio': data.get('portfolio', []),
                    'trades': loaded_trades
                }
                
        except Exception as e:
            st.error(f"❌ Error loading data: {e}")
        
        return {'portfolio': [], 'trades': []}

    def save_data(self):
        """Save portfolio and trades to disk"""
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
                        'max_gain': trade.get('max_gain', 0.0),
                        'reason': trade.get('reason', '')
                    }
                    for trade in st.session_state.trades
                ],
                'last_save': datetime.now().isoformat(),
                'save_count': len(st.session_state.trades)
            }
            
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            st.error(f"❌ Error saving data: {e}")

    def calculate_equivalent(self, from_token: str, to_token: str, quantity: float) -> float:
        """Calculate equivalent amount of `to_token` when selling `quantity` of `from_token`.
           Fees are applied on both sides (sell and buy) as f = 1 - fee_rate.
           Returns absolute qty of to_token that would be obtained by swapping the full `quantity`.
        """
        if from_token == to_token:
            # swapping to itself, apply a single-side fee (or no-op depending on convention). Keep consistent with initialization.
            return quantity * (1 - self.fee_rate)
            
        prices = st.session_state.prices
        
        if from_token not in prices or to_token not in prices:
            return 0.0
            
        try:
            # Sell from_token at bid -> receive USDT net after sell fee
            usdt_value = quantity * prices[from_token].bid_price * (1 - self.fee_rate)
            # Buy to_token at ask -> qty net after buy fee
            equivalent = (usdt_value / prices[to_token].ask_price) * (1 - self.fee_rate)
            return equivalent
        except (ZeroDivisionError, KeyError):
            return 0.0

    def check_and_execute_trades(self):
        """Check trailing stop conditions and execute trades.
           Correct logic: top_equivalent is NOT updated before swap.
           We only update max_gain as the maximum observed gain_from_top (in percent).
           When trailing stop triggers and zero-loss guard holds, we execute swap and then update top_equivalent for that slot.
        """
        if not st.session_state.tracking or not st.session_state.portfolio:
            return
        
        slot_candidates = {}
        
        # Build list of candidate swap for each slot (best candidate per slot)
        for slot_idx, slot in enumerate(st.session_state.portfolio):
            swap_candidates = []
            current_tokens = [s['token'] for s in st.session_state.portfolio]
            from_token = slot['token']
            qty = slot['quantity']
            
            # if qty is zero or from_token missing in prices, skip
            if qty <= 0 or from_token not in st.session_state.prices:
                continue
            
            for target_token in self.tokens_to_track:
                if target_token == from_token:
                    continue
                # ensure target token not currently owned by other slot (unique tokens)
                if target_token in current_tokens:
                    continue
                
                # current equivalent (how many target_token we'd get selling full slot now)
                current_eq = self.calculate_equivalent(from_token, target_token, qty)
                if current_eq <= 0:
                    continue
                
                # baseline and current top stored as absolute amounts for the slot's quantity
                baseline_eq = slot['baseline'].get(target_token, None)
                current_top = slot['top_equivalent'].get(target_token, None)
                if current_top is None or current_top <= 0:
                    # if top not present (rare), set to current_eq as safe fallback (but do not persist here)
                    current_top = current_eq
                    slot['top_equivalent'][target_token] = current_top
                
                # Gain from baseline used for reporting
                gain_from_baseline = ((current_eq - baseline_eq) / baseline_eq * 100) if baseline_eq and baseline_eq > 0 else 0.0
                slot['current_gain'][target_token] = gain_from_baseline
                
                # Gain relative to top (in percent)
                gain_from_top = ((current_eq - current_top) / current_top * 100) if current_top > 0 else 0.0
                
                # Update max_gain observed (we keep maximum observed gain_from_top)
                prev_max = slot['max_gain'].get(target_token, 0.0)
                if gain_from_top > prev_max:
                    slot['max_gain'][target_token] = gain_from_top
                    prev_max = gain_from_top
                
                current_max_gain = slot['max_gain'].get(target_token, 0.0)
                
                # If we've observed a sufficient rise (>=0.5%), compute TS and check drop from max
                if current_max_gain >= 0.5 or gain_from_top >= 0.5:
                    # TS depends on current_max_gain (in percent)
                    current_ts = self.get_trailing_stop_level(current_max_gain)
                    # swap threshold is: execute when gain_from_top <= current_max_gain - TS
                    swap_threshold = current_max_gain - current_ts
                    # Candidate if current gain_from_top is at or below threshold (i.e., has retraced enough)
                    if gain_from_top <= swap_threshold:
                        # zero-loss guard: ensure current_eq >= top_equivalent (we'll not perform token-decreasing swaps)
                        if current_eq >= current_top:
                            swap_candidates.append({
                                'target_token': target_token,
                                'current_eq': current_eq,
                                'gain_from_baseline': gain_from_baseline,
                                'gain_from_top': gain_from_top,
                                'max_gain': current_max_gain,
                                'trailing_stop': current_ts,
                                'swap_threshold': swap_threshold,
                                'priority_score': current_max_gain  # choose highest max_gain first
                            })
            # choose top candidate per slot if any
            if swap_candidates:
                swap_candidates.sort(key=lambda x: x['priority_score'], reverse=True)
                slot_candidates[slot_idx] = swap_candidates[0]
        
        # Execute candidates: to avoid conflicts (two slots target same token), run deterministic order
        executed_targets = set()
        executed_slots = []
        for slot_idx in sorted(slot_candidates.keys()):
            candidate = slot_candidates[slot_idx]
            target = candidate['target_token']
            # skip if another executed slot already took that target
            if target in executed_targets:
                continue
            # double-check target still free
            current_tokens = [s['token'] for s in st.session_state.portfolio]
            if target in current_tokens:
                continue
            # execute trade
            slot = st.session_state.portfolio[slot_idx]
            self.execute_trade(slot_idx, slot, target, candidate['current_eq'], candidate['max_gain'])
            executed_targets.add(target)
            executed_slots.append(slot_idx)

    def execute_trade(self, slot_idx: int, slot: dict, target_token: str, equivalent: float, max_gain: float):
        """Perform full-slot swap. IMPORTANT: top_equivalent must be updated ONLY AFTER swap for that slot.
           baseline remains unchanged.
        """
        from_token = slot['token']
        from_qty = slot['quantity']
        
        # Recalculate actual to_quantity at exact moment (consistency)
        to_qty = self.calculate_equivalent(from_token, target_token, from_qty)
        if to_qty <= 0:
            st.warning("⚠️ Swap aborted: computed to_qty <= 0")
            return
        
        # Record trade (timestamp before changing slot state)
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
        
        # Execute swap: replace token and qty in the slot
        slot['token'] = target_token
        slot['quantity'] = to_qty
        
        # AFTER SWAP: update top_equivalent for this slot based on new base (the new token and its full quantity)
        # We compute absolute equivalents for the full slot quantity (consistent with baseline representation)
        for token in self.tokens_to_track:
            if token == target_token:
                # equivalent of token->itself is the current quantity
                slot['top_equivalent'][token] = to_qty
            else:
                new_equiv = self.calculate_equivalent(target_token, token, to_qty)
                slot['top_equivalent'][token] = new_equiv if new_equiv is not None else 0.0
            
            # reset gains for all pairs in this slot
            slot['current_gain'][token] = 0.0
            slot['max_gain'][token] = 0.0
        
        # Save trade in history
        st.session_state.trades.append(trade)
        self.save_data()
        
        st.toast(f"🔁 SWAP: {from_token} → {target_token} (Slot {slot_idx + 1})", icon="✅")
        st.success(f"💰 Executed SWAP: {from_token} → {target_token} | max_gain observed: {max_gain:.2f}%")

    def clear_all_data(self):
        """Clear stored data"""
        st.session_state.portfolio = []
        st.session_state.trades = []
        st.session_state.tracking = False
        
        if os.path.exists(self.data_file):
            os.remove(self.data_file)
        
        st.success("🗑️ All data cleared")
        st.rerun()

    def render_sidebar(self):
        """Render sidebar controls"""
        with st.sidebar:
            st.title("⚙️ Config 24/7")
            
            if hasattr(st.session_state, 'app_start_time'):
                uptime = datetime.now() - st.session_state.app_start_time
                st.metric("Uptime", f"{uptime.seconds // 3600}h {(uptime.seconds % 3600) // 60}m")
            
            st.metric("Slots", f"{len(st.session_state.portfolio)}/5")
            st.metric("Trades", len(st.session_state.trades))
            st.metric("Price updates", st.session_state.price_updates)
            st.metric("Status", "🟢 RUNNING" if st.session_state.tracking else "🟡 PAUSED")
            
            st.subheader("🔍 Diagnostics")
            if st.button("🧪 Test MEXC connection"):
                connection_ok, message = self.test_connection()
                if connection_ok:
                    st.success(message)
                else:
                    st.error(message)
            
            if not st.session_state.portfolio:
                st.subheader("💰 Init Portfolio")
                usdt_amount = st.number_input("USDT amount:", min_value=10.0, value=1000.0, step=100.0)
                
                available_tokens = []
                if hasattr(st.session_state, 'prices'):
                    available_tokens = list(st.session_state.prices.keys())
                    available_tokens.sort()
                
                if not available_tokens:
                    st.error("🚫 No price data yet. Wait for update.")
                else:
                    selected_tokens = st.multiselect(
                        "Select 5 tokens:", 
                        available_tokens,
                        default=available_tokens[:5] if len(available_tokens) >= 5 else available_tokens,
                        max_selections=5
                    )
                    
                    st.caption(f"✅ Available tokens: {len(available_tokens)}")
                    
                    if st.button("🏁 Initialize Portfolio", type="primary", use_container_width=True):
                        if len(selected_tokens) == 5:
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
                        st.error("❌ No prices to track")
            
            with col2:
                if st.button("⏹ Stop", use_container_width=True) and st.session_state.tracking:
                    st.session_state.tracking = False
                    st.rerun()
            
            st.subheader("💾 Data")
            if os.path.exists(self.data_file):
                file_time = os.path.getmtime(self.data_file)
                file_size = os.path.getsize(self.data_file) / 1024
                st.caption(f"📁 Last save: {datetime.fromtimestamp(file_time).strftime('%H:%M:%S')}")
                st.caption(f"📊 Data size: {file_size:.1f} KB")
                
                if st.button("🗑️ Clear all data", use_container_width=True):
                    self.clear_all_data()
            
            st.subheader("🎯 Trailing Stop Levels")
            for gain, stop in sorted(self.trailing_stop_levels.items()):
                st.text(f"💰 {gain}% gain → trailing stop {stop}% below max")

    def render_portfolio_overview(self):
        """Render portfolio overview"""
        st.header("📊 Portfolio Overview")
        
        if not st.session_state.portfolio:
            st.info("👈 Initialize portfolio from sidebar")
            return
        
        cols = st.columns(len(st.session_state.portfolio))
        for idx, (col, slot) in enumerate(zip(cols, st.session_state.portfolio)):
            with col:
                # Check availability of price for the slot token
                if slot['token'] in st.session_state.prices:
                    current_value = slot['quantity'] * st.session_state.prices[slot['token']].bid_price
                    st.metric(
                        label=f"Slot {idx + 1} - {slot['token']}",
                        value=f"{slot['quantity']:.6f}",
                        delta=f"{current_value:.2f} USDT"
                    )
                else:
                    st.metric(label=f"Slot {idx + 1} - {slot['token']}", value=f"{slot['quantity']:.6f}", delta="N/A")

    def render_trailing_matrix(self):
        """Render trailing matrix and history for each slot"""
        st.header("🎯 Trailing Stop Matrix")
        
        for slot_idx, slot in enumerate(st.session_state.portfolio):
            self.render_slot_with_history(slot_idx, slot)

    def render_slot_with_history(self, slot_idx: int, slot: dict):
        """Render a slot with matrix and trade history"""
        with st.expander(f"🔷 Slot {slot_idx + 1}: {slot['token']} ({slot['quantity']:.6f})", expanded=True):
            self.render_slot_matrix(slot_idx, slot)
            self.render_slot_trade_history(slot_idx)

    def render_slot_matrix(self, slot_idx: int, slot: dict):
        """Render matrix for a single slot"""
        matrix_data = []
        best_pair_gain = -999.0
        best_pair_token = None
        
        for token in self.tokens_to_track:
            current_max_gain = slot['max_gain'].get(token, 0.0)
            if current_max_gain > best_pair_gain:
                best_pair_gain = current_max_gain
                best_pair_token = token
        
        for token in self.tokens_to_track:
            current_eq = self.calculate_equivalent(slot['token'], token, slot['quantity'])
            baseline_eq = slot['baseline'].get(token, current_eq)
            top_eq = slot['top_equivalent'].get(token, current_eq)
            current_gain = slot['current_gain'].get(token, 0.0)
            max_gain = slot['max_gain'].get(token, 0.0)
            
            change_from_baseline = ((current_eq - baseline_eq) / baseline_eq * 100) if baseline_eq > 0 else 0
            change_from_top = ((current_eq - top_eq) / top_eq * 100) if top_eq > 0 else 0
            
            status = "🟢" if change_from_top >= -1 else "🟡" if change_from_top >= -3 else "🔴"
            if token == slot['token']:
                status = "🔵"
            elif token == best_pair_token and best_pair_gain >= 0.5:
                status = "⭐"
            
            matrix_data.append({
                'Token': token,
                'Aktualny': f"{current_eq:.6f}",
                'Początkowy': f"{baseline_eq:.6f}",
                'Δ Od początku': f"{change_from_baseline:+.2f}%",
                'Top': f"{top_eq:.6f}",
                'Δ Od top': f"{change_from_top:+.2f}%",
                'Current Gain': f"{current_gain:+.2f}%",
                'Max Wzrost': f"{max_gain:+.2f}%",
                'Status': status
            })
        
        df = pd.DataFrame(matrix_data)
        st.dataframe(df, use_container_width=True, height=800)

    def render_slot_trade_history(self, slot_idx: int):
        """Render trade history for a slot"""
        slot_trades = [t for t in st.session_state.trades if t['slot'] == slot_idx]
        
        if slot_trades:
            st.subheader(f"📋 History Slot {slot_idx + 1}")
            
            history_data = []
            for trade in slot_trades[-10:]:
                history_data.append({
                    'Data': trade['timestamp'].strftime('%H:%M:%S'),
                    'Z': trade['from_token'],
                    'Na': trade['to_token'],
                    'Ilość': f"{trade['to_quantity']:.6f}",
                    'Max Wzrost': f"{trade['max_gain']:.2f}%",
                    'Powód': trade['reason']
                })
            
            st.dataframe(pd.DataFrame(history_data), use_container_width=True)
        else:
            st.caption("📝 No trades for this slot yet")

    def keep_app_alive(self):
        """Keep app alive heartbeat"""
        if not hasattr(st.session_state, 'last_active_ping'):
            st.session_state.last_active_ping = datetime.now()
        
        time_diff = (datetime.now() - st.session_state.last_active_ping).seconds
        if time_diff > 120:
            st.session_state.last_active_ping = datetime.now()

    def run(self):
        """Main app loop"""
        try:
            self.init_session_state()
            
            st.title("🚀 Crypto Trailing Stop Matrix - 24/7")
            st.markdown("---")
            
            self.keep_app_alive()
            
            self.render_sidebar()
            
            if st.session_state.portfolio and not st.session_state.tracking:
                with st.sidebar:
                    if st.button("▶ Auto-start tracking", type="primary", use_container_width=True):
                        st.session_state.tracking = True
                        st.rerun()
            
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

# Run app
if __name__ == "__main__":
    app = CryptoTrailingStopApp()
    app.run()
