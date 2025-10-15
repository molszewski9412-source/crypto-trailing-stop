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
        self.fee_rate = 0.00025
        self.trailing_stop_levels = {0.5: 0.2, 1.0: 0.5, 2.0: 1.0, 5.0: 2.0}
        self.data_file = "trailing_stop_data.json"
        
        # Lista tokenów - sprawdzone na MEXC
        self.tokens_to_track = [
            'BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'XRP', 'DOT', 'DOGE', 'AVAX', 'LTC',
            'LINK', 'ATOM', 'XLM', 'BCH', 'ALGO', 'FIL', 'ETC', 'XTZ', 'AAVE', 'COMP',
            'UNI', 'CRV', 'SUSHI', 'YFI', 'SNX', '1INCH', 'ZRX', 'TRX', 'VET', 'ONE',
            'CELO', 'RSR', 'NKN', 'STORJ', 'DODO', 'KAVA', 'RUNE', 'SAND', 'MANA', 'ENJ',
            'CHZ', 'ALICE', 'NEAR', 'ARB', 'OP', 'APT', 'SUI', 'SEI', 'INJ', 'RENDER'
        ]
        
    def get_trailing_stop_level(self, max_gain: float) -> float:
        """Zwraca trailing stop level dla danego max_gain"""
        current_ts = 0.0
        for gain_threshold, ts_level in sorted(self.trailing_stop_levels.items()):
            if max_gain >= gain_threshold:
                current_ts = ts_level
        return current_ts
        
    def test_connection(self):
        """Testuj połączenie z MEXC API"""
        try:
            url = "https://api.mexc.com/api/v3/ping"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return True, "✅ Połączenie z MEXC API działa"
            else:
                return False, f"❌ MEXC API zwraca status: {response.status_code}"
        except Exception as e:
            return False, f"❌ Błąd połączenia: {e}"

    def get_all_prices_bulk(self) -> Dict[str, TokenInfo]:
        """Pobierz ceny z MEXC - bezpieczna wersja z diagnostyką"""
        prices = {}
        
        try:
            url = "https://api.mexc.com/api/v3/ticker/bookTicker"
            response = requests.get(url, timeout=15)
            
            if response.status_code == 200:
                all_data = response.json()
                
                if not all_data:
                    st.error("❌ MEXC zwrócił pustą odpowiedź")
                    return {}
                
                # Filtruj pary USDT
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
                            
                            # Walidacja cen
                            if bid_price > 0 and ask_price > 0 and bid_price <= ask_price:
                                prices[token] = TokenInfo(
                                    symbol=token,
                                    bid_price=bid_price,
                                    ask_price=ask_price,
                                    last_update=datetime.now()
                                )
                                found_tokens += 1
                            else:
                                problematic_tokens.append(f"{token}(nieprawidłowe ceny)")
                        except (ValueError, KeyError):
                            problematic_tokens.append(f"{token}(błąd konwersji)")
                    else:
                        problematic_tokens.append(token)
                
                if problematic_tokens:
                    st.warning(f"⚠️ Brak cen dla: {', '.join(problematic_tokens[:10])}")
                
                if prices:
                    st.success(f"✅ Pobrano ceny dla {found_tokens}/50 tokenów")
                    return prices
                else:
                    st.error("🚫 Nie udało się pobrać żadnych cen")
                    return {}
                    
            else:
                st.error(f"❌ Błąd HTTP {response.status_code} od MEXC")
                return {}
                
        except requests.exceptions.Timeout:
            st.error("⏰ Timeout - MEXC API nie odpowiada")
            return {}
        except requests.exceptions.ConnectionError:
            st.error("🌐 Błąd połączenia - sprawdź internet")
            return {}
        except Exception as e:
            st.error(f"❌ Nieoczekiwany błąd: {e}")
            return {}

    def get_initial_prices(self) -> Dict[str, TokenInfo]:
        """Pobierz początkowe ceny"""
        return self.get_all_prices_bulk()

    def update_real_prices(self):
        """Aktualizuj ceny rzeczywistymi danymi z MEXC"""
        new_prices = self.get_all_prices_bulk()
        if new_prices:
            st.session_state.prices = new_prices
            st.session_state.price_updates += 1
            st.session_state.last_tracking_time = datetime.now()

    def initialize_portfolio_from_usdt(self, usdt_amount: float, selected_tokens: List[str]):
        """Inicjuj portfolio z USDT - podział na 5 tokenów"""
        if len(selected_tokens) != 5:
            st.error("❌ Wybierz dokładnie 5 tokenów")
            return False
            
        if usdt_amount <= 0:
            st.error("❌ Kwota USDT musi być większa od 0")
            return False
            
        # Sprawdź które tokeny mają ceny
        available_tokens = []
        for token in selected_tokens:
            if token in st.session_state.prices:
                available_tokens.append(token)
        
        if len(available_tokens) < 5:
            st.error(f"❌ Za mało tokenów z dostępnymi cenami: {len(available_tokens)}/5")
            return False
        
        # Wyczyść istniejące portfolio
        st.session_state.portfolio = []
        st.session_state.trades = []
        
        usdt_per_slot = usdt_amount / 5
        
        for token in available_tokens:
            # Oblicz ilość tokena na podstawie ceny ask
            token_price = st.session_state.prices[token].ask_price
            quantity = (usdt_per_slot / token_price) * (1 - self.fee_rate)
            
            # Oblicz baseline equivalents dla WSZYSTKICH 50 tokenów
            baseline = {}
            top_equivalent = {}
            current_gain = {}
            max_gain = {}
            
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
        
        # Zapisz dane
        self.save_data()
        st.success(f"✅ Utworzono portfolio: {usdt_amount} USDT → 5 slotów")
        return True

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
        """Oblicz ekwiwalent z uwzględnieniem fee - bezpieczna wersja"""
        if from_token == to_token:
            return quantity * (1 - self.fee_rate)
            
        prices = st.session_state.prices
        
        # Sprawdź czy oba tokeny mają ceny
        if from_token not in prices or to_token not in prices:
            return 0.0
            
        try:
            # Sprzedaż from_token -> USDT
            usdt_value = quantity * prices[from_token].bid_price * (1 - self.fee_rate)
            # Kupno USDT -> to_token
            equivalent = usdt_value / prices[to_token].ask_price * (1 - self.fee_rate)
            
            return equivalent
        except (ZeroDivisionError, KeyError):
            return 0.0

    def check_and_execute_trades(self):
        """Sprawdź warunki trailing stop - PRIORYTET W OBRĘBIE SLOTU"""
        if not st.session_state.tracking or not st.session_state.portfolio:
            return
        
        # ✅ ZBIERZ KANDYDATÓW DLA KAŻDEGO SLOTU OSOBNO
        slot_candidates = {}
        
        for slot_idx, slot in enumerate(st.session_state.portfolio):
            swap_candidates = []
            current_tokens = [s['token'] for s in st.session_state.portfolio]
            
            # ✅ ŚLEDŹ WSZYSTKIE PARY (ale z bezpieczeństwem)
            for target_token in self.tokens_to_track:
                if target_token != slot['token'] and target_token not in current_tokens:  # ✅ BEZPIECZEŃSTWO
                    
                    current_eq = self.calculate_equivalent(slot['token'], target_token, slot['quantity'])
                    current_top = slot['top_equivalent'].get(target_token, current_eq)
                    current_max_gain = slot['max_gain'].get(target_token, 0.0)
                    
                    # Oblicz aktualny gain
                    current_gain = ((current_eq - current_top) / current_top * 100) if current_top > 0 else 0
                    
                    # ✅ AKTUALIZUJ max_gain (najwyższa wartość w serii dla tej pary)
                    if current_gain > current_max_gain:
                        slot['max_gain'][target_token] = current_gain
                        current_max_gain = current_gain
                    
                    slot['current_gain'][target_token] = current_gain
                    
                    # ✅ SPRAWDŹ CZY PARA OSIĄGNĘŁA 0.5% (aktywacja trailing stop)
                    if current_max_gain >= 0.5:
                        current_ts = self.get_trailing_stop_level(current_max_gain)
                        
                        # ✅ SPRAWDŹ WARUNEK SWAPU
                        if current_gain <= (current_max_gain - current_ts) and current_eq > current_top:
                            swap_candidates.append({
                                'target_token': target_token,
                                'current_eq': current_eq,
                                'current_gain': current_gain,
                                'max_gain': current_max_gain,
                                'trailing_stop': current_ts,
                                'priority_score': current_max_gain  # ✅ PRIORYTET: najwyższy max_gain
                            })
            
            # ✅ POSORTOWANIE KANDYDATÓW W OBRĘBIE SLOTU
            if swap_candidates:
                swap_candidates.sort(key=lambda x: x['priority_score'], reverse=True)
                slot_candidates[slot_idx] = swap_candidates[0]  # ✅ NAJLEPSZY KANDYDAT DLA SLOTU
        
        # ✅ WYKONAJ SWAPY DLA WSZYSTKICH SLOTÓW (każdy slot może wykonać 1 swap)
        executed_slots = []
        for slot_idx, candidate in slot_candidates.items():
            if slot_idx not in executed_slots:
                slot = st.session_state.portfolio[slot_idx]
                
                # ✅ DODATKOWA WERYFIKACJA BEZPIECZEŃSTWA
                current_tokens = [s['token'] for s in st.session_state.portfolio]
                if candidate['target_token'] not in current_tokens:
                    self.execute_trade(
                        slot_idx, 
                        slot, 
                        candidate['target_token'], 
                        candidate['current_eq'], 
                        candidate['max_gain']
                    )
                    executed_slots.append(slot_idx)
                    st.success(f"🎯 Slot {slot_idx+1}: WYBRANO {slot['token']}→{candidate['target_token']} (max_gain={candidate['max_gain']:.2f}%)")

    def execute_trade(self, slot_idx: int, slot: dict, target_token: str, equivalent: float, max_gain: float):
        """Wykonaj transakcję trailing stop - AKTUALIZUJ TOP TYLKO PRZY SWAPIE"""
        
        # ✅ 1. PRZED SWAPEM: Aktualizuj top equivalent dla WSZYSTKICH par
        updated_tokens = []
        for token in self.tokens_to_track:
            if token != slot['token'] and token not in [s['token'] for s in st.session_state.portfolio]:
                current_actual = self.calculate_equivalent(slot['token'], token, slot['quantity'])
                current_top = slot['top_equivalent'].get(token, current_actual)
                
                # ✅ TYLKO TUTAJ aktualizujemy top equivalent!
                if current_actual > current_top:
                    slot['top_equivalent'][token] = current_actual
                    updated_tokens.append(token)
        
        if updated_tokens:
            st.success(f"📈 Zaktualizowano top dla {len(updated_tokens)} tokenów: {', '.join(updated_tokens[:3])}")
        
        # ✅ 2. FINALNA WERYFIKACJA: Czy nadal actual > top dla target_token
        final_actual = self.calculate_equivalent(slot['token'], target_token, slot['quantity'])
        final_top = slot['top_equivalent'].get(target_token, final_actual)
        
        # ❌ BEZPIECZNIK: Jeśli actual ≤ top - ANULUJ
        if final_actual <= final_top:
            st.error(f"🚫 SWAP ANULOWANY: {slot['token']}→{target_token} - actual ≤ top ({final_actual:.6f} ≤ {final_top:.6f})")
            return
        
        # ✅ 3. WYKONAJ SWAP (gwarantowany wzrost)
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
        
        # ✅ 4. PO SWAPIE: Resetuj top equivalent dla nowego tokena
        for token in self.tokens_to_track:
            if token != target_token:
                new_actual = self.calculate_equivalent(target_token, token, equivalent)
                slot['top_equivalent'][token] = new_actual  # Top = actual po swapie
                slot['current_gain'][token] = 0.0
                slot['max_gain'][token] = 0.0
        
        st.session_state.trades.append(trade)
        self.save_data()
        
        # ✅ 5. POTWIERDZENIE WZROSTU
        growth_percentage = ((equivalent - final_top) / final_top * 100) if final_top > 0 else 0
        st.toast(f"🔁 SWAP: {old_token} → {target_token} (Slot {slot_idx + 1})", icon="✅")
        st.success(f"💰 GWARANTOWANY WZROST: +{growth_percentage:+.3f}% od poprzedniego top")

    def clear_all_data(self):
        """Wyczyść wszystkie dane"""
        st.session_state.portfolio = []
        st.session_state.trades = []
        st.session_state.tracking = False
        
        if os.path.exists(self.data_file):
            os.remove(self.data_file)
        
        st.success("🗑️ Wszystkie dane zostały wyczyszczone!")
        st.rerun()

    def render_sidebar(self):
        """Renderuj panel boczny"""
        with st.sidebar:
            st.title("⚙️ Konfiguracja")
            
            # Diagnostyka
            st.subheader("🔍 Diagnostyka")
            if st.button("🧪 Testuj połączenie z MEXC"):
                connection_ok, message = self.test_connection()
                if connection_ok:
                    st.success(message)
                else:
                    st.error(message)
            
            # Inicjacja z USDT
            if not st.session_state.portfolio:
                st.subheader("💰 Inicjacja Portfolio z USDT")
                usdt_amount = st.number_input("Kwota USDT:", min_value=10.0, value=1000.0, step=100.0)
                
                # Pokazuj tylko tokeny, które mają ceny
                available_tokens = []
                if hasattr(st.session_state, 'prices'):
                    available_tokens = list(st.session_state.prices.keys())
                    available_tokens.sort()
                
                if not available_tokens:
                    st.error("🚫 Brak dostępnych tokenów. Poczekaj na aktualizację cen.")
                else:
                    selected_tokens = st.multiselect(
                        "Wybierz 5 tokenów:", 
                        available_tokens,
                        default=available_tokens[:5] if len(available_tokens) >= 5 else available_tokens,
                        max_selections=5
                    )
                    
                    st.caption(f"✅ Dostępne tokeny: {len(available_tokens)}")
                    
                    if st.button("🏁 Inicjuj Portfolio", type="primary", use_container_width=True):
                        if len(selected_tokens) == 5:
                            self.initialize_portfolio_from_usdt(usdt_amount, selected_tokens)
                            st.rerun()
                        else:
                            st.error("❌ Wybierz dokładnie 5 tokenów")
            
            # Sterowanie trackingiem
            st.subheader("🎮 Sterowanie")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("▶ Start", use_container_width=True) and not st.session_state.tracking:
                    if st.session_state.prices:
                        st.session_state.tracking = True
                        st.rerun()
                    else:
                        st.error("❌ Brak cen do śledzenia")
            
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
            else:
                st.caption("🚫 Brak danych cenowych")
            
            # Zarządzanie danymi
            st.subheader("💾 Zarządzanie danymi")
            if os.path.exists(self.data_file):
                file_time = os.path.getmtime(self.data_file)
                file_size = os.path.getsize(self.data_file) / 1024
                st.caption(f"📁 Ostatni zapis: {datetime.fromtimestamp(file_time).strftime('%H:%M:%S')}")
                st.caption(f"📊 Rozmiar danych: {file_size:.1f} KB")
                
                if st.button("🗑️ Wyczyść wszystkie dane", use_container_width=True):
                    self.clear_all_data()
            
            # Trailing stop info
            st.subheader("🎯 Trailing Stop")
            for gain, stop in sorted(self.trailing_stop_levels.items()):
                st.text(f"💰 {gain}% zysk → {stop}% stop")

    def render_portfolio_overview(self):
        """Renderuj przegląd portfolio"""
        st.header("📊 Przegląd Portfolio")
        
        if not st.session_state.portfolio:
            st.info("👈 Zainicjuj portfolio z USDT w panelu bocznym")
            return
        
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
        """Renderuj macierz trailing stop z historią per slot"""
        st.header("🎯 Macierz Trailing Stop")
        
        for slot_idx, slot in enumerate(st.session_state.portfolio):
            self.render_slot_with_history(slot_idx, slot)

    def render_slot_with_history(self, slot_idx: int, slot: dict):
        """Renderuj slot z macierzą i historią"""
        with st.expander(f"🔷 Slot {slot_idx + 1}: {slot['token']} ({slot['quantity']:.4f})", expanded=True):
            self.render_slot_matrix(slot_idx, slot)
            self.render_slot_trade_history(slot_idx)

    def render_slot_matrix(self, slot_idx: int, slot: dict):
        """Renderuj macierz dla pojedynczego slotu z oznaczeniem najlepszej pary"""
        matrix_data = []
        best_pair_gain = -999
        best_pair_token = None
        
        # Znajdź najlepszą parę dla tego slotu
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
            
            change_baseline = ((current_eq - baseline_eq) / baseline_eq * 100) if baseline_eq > 0 else 0
            change_top = ((current_eq - top_eq) / top_eq * 100) if top_eq > 0 else 0
            
            # ✅ OZNACZENIE STATUSU Z NAJLEPSZĄ PARĄ
            status = "🟢" if change_top >= -1 else "🟡" if change_top >= -3 else "🔴"
            if token == slot['token']:
                status = "🔵"
            elif token == best_pair_token and best_pair_gain >= 0.5:
                status = "⭐"  # ✅ NAJLEPSZA PARA!
            
            matrix_data.append({
                'Token': token,
                'Aktualny': f"{current_eq:.6f}",
                'Początkowy': f"{baseline_eq:.6f}",
                'Δ Od początku': f"{change_baseline:+.2f}%",
                'Top': f"{top_eq:.6f}",
                'Δ Od top': f"{change_top:+.2f}%",
                'Current Gain': f"{current_gain:+.2f}%",
                'Max Wzrost': f"{max_gain:+.2f}%",
                'Status': status
            })
        
        df = pd.DataFrame(matrix_data)
        st.dataframe(df, use_container_width=True, height=800)

    def render_slot_trade_history(self, slot_idx: int):
        """Renderuj historię transakcji dla slotu"""
        slot_trades = [t for t in st.session_state.trades if t['slot'] == slot_idx]
        
        if slot_trades:
            st.subheader(f"📋 Historia Slot {slot_idx + 1}")
            
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
            st.caption("📝 Brak historii transakcji dla tego slotu")

    def run(self):
        """Główna pętla aplikacji"""
        self.init_session_state()
        
        st.title("🚀 Crypto Trailing Stop Matrix - REAL TIME")
        st.markdown("---")
        
        self.render_sidebar()
        
        if st.session_state.prices:
            self.render_portfolio_overview()
            
            if st.session_state.portfolio:
                self.render_trailing_matrix()
                
                if st.session_state.tracking:
                    self.update_real_prices()
                    self.check_and_execute_trades()
                    time.sleep(1)
                    st.rerun()
        else:
            st.error("🚫 Nie można pobrać cen z MEXC. Sprawdź połączenie.")

# Uruchom aplikację
if __name__ == "__main__":
    app = CryptoTrailingStopApp()
    app.run()