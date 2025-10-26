# app.py
import streamlit as st
import pandas as pd
import requests
import time
from datetime import datetime
import json
import hmac
import hashlib
from typing import Dict, Optional, Tuple

# ============= Page config ============
st.set_page_config(page_title="Crypto Swap Matrix - Auto", page_icon="🔄", layout="wide")

# ============= App class ==============
class CryptoSwapMatrix:
    def __init__(self):
        self.fee_rate = 0.00025
        self.swap_threshold = 0.5  # percent threshold for swap signal (example)
        self.tokens_to_track = [
            'BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'XRP', 'DOT', 'DOGE', 'AVAX', 'LTC',
            'LINK', 'ATOM', 'XLM', 'BCH', 'ALGO', 'FIL', 'ETC', 'XTZ', 'AAVE', 'COMP',
            'UNI', 'CRV', 'SUSHI', 'YFI', 'SNX', '1INCH', 'ZRX', 'TRX', 'VET', 'ONE',
            'CELO', 'RSR', 'NKN', 'STORJ', 'DODO', 'KAVA', 'RUNE', 'SAND', 'MANA', 'ENJ',
            'CHZ', 'ALICE', 'NEAR', 'ARB', 'OP', 'APT', 'SUI', 'SEI', 'INJ', 'RENDER', 'MX', 'USDT'
        ]

    # ---------------- prices ----------------
    def get_prices(self) -> Dict[str, Dict]:
        prices = {}
        try:
            resp = requests.get("https://api.mexc.com/api/v3/ticker/bookTicker", timeout=10)
            resp.raise_for_status()
            data = resp.json()
            usdt_pairs = {item['symbol']: item for item in data if item['symbol'].endswith('USDT')}
            for token in self.tokens_to_track:
                if token == 'USDT':
                    prices['USDT'] = {'bid': 1.0, 'ask': 1.0, 'last_update': datetime.now()}
                    continue
                symbol = f"{token}USDT"
                if symbol in usdt_pairs:
                    try:
                        bid = float(usdt_pairs[symbol]['bidPrice'])
                        ask = float(usdt_pairs[symbol]['askPrice'])
                        if bid > 0 and ask > 0:
                            prices[token] = {'bid': bid, 'ask': ask, 'last_update': datetime.now()}
                    except Exception:
                        continue
            return prices
        except Exception as e:
            st.warning(f"Could not fetch prices: {e}")
            return {}

    # ---------------- portfolio (real) ----------------
    def _sign(self, secret_key: str, query_string: str) -> str:
        return hmac.new(secret_key.encode(), query_string.encode(), hashlib.sha256).hexdigest()

    def get_real_portfolio(self, api_key: str, secret_key: str) -> Dict[str, Dict]:
        """Get account balances from MEXC. Returns dict asset -> {total, free, locked}"""
        try:
            timestamp = int(time.time() * 1000)
            qs = f"timestamp={timestamp}"
            signature = self._sign(secret_key, qs)
            headers = {"X-MEXC-APIKEY": api_key}
            url = f"https://api.mexc.com/api/v3/account?{qs}&signature={signature}"
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code != 200:
                st.error(f"API error fetching account: {resp.status_code} {resp.text}")
                return {}
            data = resp.json()
            balances = data.get("balances", []) if isinstance(data, dict) else []
            portfolio = {}
            for b in balances:
                asset = b.get("asset") or b.get("coin") or b.get("symbol")
                free = float(b.get("free", 0))
                locked = float(b.get("locked", 0))
                total = free + locked
                if total > 0:
                    portfolio[asset] = {"total": total, "free": free, "locked": locked}
            return portfolio
        except Exception as e:
            st.error(f"Error fetching portfolio: {e}")
            return {}

    # ---------------- trade history & historical prices ----------------
    def get_last_buy_timestamp(self, api_key: str, secret_key: str, asset: str) -> Optional[int]:
        """
        Tries to fetch last BUY trade timestamp (ms) for assetUSDT by scanning myTrades.
        If not available, returns None.
        """
        try:
            symbol = f"{asset}USDT"
            timestamp = int(time.time() * 1000)
            qs = f"timestamp={timestamp}"
            signature = self._sign(secret_key, qs)
            headers = {"X-MEXC-APIKEY": api_key}
            url = f"https://api.mexc.com/api/v3/myTrades?symbol={symbol}&{qs}&signature={signature}"
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code != 200:
                return None
            trades = resp.json()
            # trades usually sorted ascending; we'll find latest trade where we were buyer
            last_buy_ts = None
            for t in trades[::-1]:
                # Many exchanges use field 'isBuyer' or 'isBuyerMaker' or 'buyer'
                is_buyer = t.get('isBuyer', None)
                side = t.get('isBuyer', None)  # fallback
                if is_buyer is True:
                    last_buy_ts = int(t.get('time') or t.get('timestamp') or 0)
                    break
                # If there's 'side' field
                if t.get('side') == 'BUY':
                    last_buy_ts = int(t.get('time') or t.get('timestamp') or 0)
                    break
            return last_buy_ts
        except Exception:
            return None

    def get_historical_price(self, symbol: str, timestamp_ms: int) -> Optional[Tuple[float, float]]:
        """
        Try to get historical bid/ask approximations for symbol (e.g. BTCUSDT) at timestamp using klines (1m).
        Returns (bid, ask) approximate by using open/high/low/close or close as price.
        If not available, returns None.
        """
        try:
            # Use a small window around timestamp
            start = max(0, timestamp_ms - 60_000)  # 1 minute earlier
            end = timestamp_ms + 60_000
            url = f"https://api.mexc.com/api/v3/klines?symbol={symbol}&interval=1m&startTime={start}&endTime={end}&limit=1"
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                return None
            klines = resp.json()
            if not klines:
                return None
            # kline format: [openTime, open, high, low, close, volume, closeTime, ...]
            k = klines[0]
            open_p, high, low, close_p = float(k[1]), float(k[2]), float(k[3]), float(k[4])
            # Approximate bid as low, ask as high (conservative)
            bid = low if low > 0 else close_p
            ask = high if high > 0 else close_p
            return (bid, ask)
        except Exception:
            return None

    # ---------------- core calculations ----------------
    def calculate_equivalent_with_prices(self, from_token: str, to_token: str, quantity: float,
                                         prices_snapshot: Dict[str, Dict]) -> float:
        """
        Calculate equivalent using provided prices_snapshot (mapping token -> {'bid','ask'}).
        This variant used for baseline if we have historical snapshot prices.
        """
        if from_token == to_token:
            return quantity * (1 - self.fee_rate)
        if from_token == 'USDT':
            to_ask = prices_snapshot.get(to_token, {}).get('ask')
            if not to_ask or to_ask <= 0:
                return 0.0
            return (quantity / to_ask) * (1 - self.fee_rate)
        elif to_token == 'USDT':
            from_bid = prices_snapshot.get(from_token, {}).get('bid')
            if not from_bid or from_bid <= 0:
                return 0.0
            return quantity * from_bid * (1 - self.fee_rate)
        else:
            from_bid = prices_snapshot.get(from_token, {}).get('bid')
            to_ask = prices_snapshot.get(to_token, {}).get('ask')
            if not from_bid or not to_ask or from_bid <= 0 or to_ask <= 0:
                return 0.0
            usdt_value = quantity * from_bid * (1 - self.fee_rate)
            return (usdt_value / to_ask) * (1 - self.fee_rate)

    def calculate_equivalent(self, from_token: str, to_token: str, quantity: float) -> float:
        """Calculate equivalent using CURRENT st.session_state.prices"""
        prices = st.session_state.prices
        if not prices or from_token not in prices or to_token not in prices:
            return 0.0
        if from_token == to_token:
            return quantity * (1 - self.fee_rate)
        if from_token == 'USDT':
            ask_price = prices[to_token]['ask']
            return (quantity / ask_price) * (1 - self.fee_rate)
        elif to_token == 'USDT':
            bid_price = prices[from_token]['bid']
            return quantity * bid_price * (1 - self.fee_rate)
        else:
            bid_price = prices[from_token]['bid']
            ask_price = prices[to_token]['ask']
            usdt_value = quantity * bid_price * (1 - self.fee_rate)
            return (usdt_value / ask_price) * (1 - self.fee_rate)

    def find_main_token(self, portfolio: Dict) -> Optional[str]:
        """Find asset with highest USDT value using CURRENT prices"""
        prices = st.session_state.prices
        max_val = 0
        main = None
        for asset, bal in portfolio.items():
            if asset in prices:
                total = bal.get('total', 0)
                val = total * prices[asset]['bid']
                if val > max_val:
                    max_val = val
                    main = asset
        return main

    # ---------------- baseline & top logic ----------------
    def set_baseline_for_token(self, token: str, amount: float, api_key: Optional[str], secret_key: Optional[str]):
        """
        Set baseline_data[token] = {
            timestamp: ms,
            equivalents: {target: amountEquivalentAtBaseline},
            usdt_value: <usdt value at baseline>
        }
        If token == USDT -> we use current prices and now timestamp.
        If token != USDT -> try to find last buy timestamp and historical prices; fallback to current prices.
        """
        now_ms = int(time.time() * 1000)
        prices_snapshot = {}  # mapping token-> {'bid','ask'}
        baseline_ts = now_ms
        # If USDT -> baseline is direct
        if token == 'USDT':
            prices_snapshot = {t: {'bid': st.session_state.prices.get(t, {}).get('bid', 0),
                                   'ask': st.session_state.prices.get(t, {}).get('ask', 0)} for t in self.tokens_to_track}
            usdt_value = amount  # USDT amount
        else:
            # try to get last buy timestamp
            baseline_ts = None
            if api_key and secret_key:
                baseline_ts = self.get_last_buy_timestamp(api_key, secret_key, token)
            if baseline_ts:
                # try to fetch historical prices for main token and all targets at baseline_ts
                for t in self.tokens_to_track:
                    pair = f"{t}USDT"
                    hist = self.get_historical_price(pair, baseline_ts)
                    if hist:
                        bid, ask = hist
                        prices_snapshot[t] = {'bid': bid, 'ask': ask}
                    else:
                        # if missing, fallback to current
                        cur = st.session_state.prices.get(t)
                        if cur:
                            prices_snapshot[t] = {'bid': cur['bid'], 'ask': cur['ask']}
                # compute usdt_value using historical bid of main token
                main_bid = prices_snapshot.get(token, {}).get('bid')
                if main_bid and main_bid > 0:
                    usdt_value = amount * main_bid
                else:
                    # fallback to current bid
                    cur = st.session_state.prices.get(token)
                    if cur:
                        usdt_value = amount * cur['bid']
                    else:
                        usdt_value = 0
            else:
                # fallback: use current prices
                baseline_ts = now_ms
                for t in self.tokens_to_track:
                    cur = st.session_state.prices.get(t)
                    if cur:
                        prices_snapshot[t] = {'bid': cur['bid'], 'ask': cur['ask']}
                cur_main = st.session_state.prices.get(token)
                usdt_value = amount * cur_main['bid'] if cur_main else 0.0

        # compute equivalents for each target token using prices_snapshot
        equivalents = {}
        for target in self.tokens_to_track:
            if target == token:
                continue
            eq = self.calculate_equivalent_with_prices(token, target, amount, prices_snapshot)
            equivalents[target] = eq

        # store baseline
        st.session_state.baseline_data[token] = {
            'timestamp': baseline_ts,
            'equivalents': equivalents,
            'usdt_value': usdt_value
        }

        # also set top_equivalents for token as exact amount (for main token store amount)
        st.session_state.top_equivalents.setdefault(token, amount)
        st.session_state.top_usdt_values.setdefault(token, usdt_value)

    def update_top_after_swap(self, old_token: str, new_token: str):
        """
        Called when swap detected (main token changed).
        Update top_equivalents for all tokens using current equivalents of new_token.
        Also update top_usdt_values mapping for tokens.
        """
        if new_token not in st.session_state.portfolio:
            return
        amount = st.session_state.portfolio[new_token]['total']
        # current equivalents w.r.t. new_token using CURRENT prices
        current_equivalents = {}
        for t in self.tokens_to_track:
            if t == new_token:
                continue
            current_equivalents[t] = self.calculate_equivalent(new_token, t, amount)
        # new_token top is exact amount
        st.session_state.top_equivalents[new_token] = amount
        st.session_state.top_usdt_values[new_token] = amount * st.session_state.prices.get(new_token, {}).get('bid', 0)
        # update others if higher
        for target, eq in current_equivalents.items():
            current_top = st.session_state.top_equivalents.get(target, 0)
            if eq > current_top:
                st.session_state.top_equivalents[target] = eq
                # also update USDT top approx
                st.session_state.top_usdt_values[target] = eq * st.session_state.prices.get(target, {}).get('bid', 0)

    def detect_swap_and_handle(self, api_key: Optional[str], secret_key: Optional[str]) -> bool:
        """Detect swap: if main token changed compared to session state - handle baseline & top"""
        current_main = self.find_main_token(st.session_state.portfolio)
        if current_main is None:
            return False
        # init
        if st.session_state.main_token is None:
            st.session_state.main_token = current_main
            amt = st.session_state.portfolio[current_main]['total']
            # set baseline at init
            self.set_baseline_for_token(current_main, amt, api_key, secret_key)
            return False
        # swap detected
        if current_main != st.session_state.main_token:
            old = st.session_state.main_token
            new = current_main
            st.session_state.main_token = new
            self.update_top_after_swap(old, new)
            # set new baseline for new token
            amt = st.session_state.portfolio[new]['total']
            self.set_baseline_for_token(new, amt, api_key, secret_key)
            st.success(f"🔄 Swap detected: {old} → {new}")
            return True
        return False

    # ---------------- session init ----------------
    def init_session_state(self):
        if 'prices' not in st.session_state:
            st.session_state.prices = self.get_prices()
        if 'portfolio' not in st.session_state:
            st.session_state.portfolio = {}
        if 'main_token' not in st.session_state:
            st.session_state.main_token = None
        if 'baseline_data' not in st.session_state:
            st.session_state.baseline_data = {}
        if 'top_equivalents' not in st.session_state:
            st.session_state.top_equivalents = {}
        if 'top_usdt_values' not in st.session_state:
            st.session_state.top_usdt_values = {}
        if 'tracking' not in st.session_state:
            st.session_state.tracking = False
        if 'last_price_update' not in st.session_state:
            st.session_state.last_price_update = datetime.now()

    # ---------------- UI rendering ----------------
    def render_portfolio(self):
        st.header("💰 Portfolio")
        if not st.session_state.portfolio:
            st.info("No portfolio data")
            return
        prices = st.session_state.prices
        rows = []
        total_value = 0.0
        for asset, bal in st.session_state.portfolio.items():
            total = bal.get('total', 0)
            if asset in prices:
                val = total * prices[asset]['bid']
                total_value += val
                price = prices[asset]['bid']
            else:
                val = 0.0
                price = 0.0
            rows.append({'Asset': asset, 'Amount': f"{total:.6f}", 'Price (bid)': f"{price:.6f}", 'Value USDT': f"${val:,.2f}"})
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Value", f"${total_value:,.2f}")
        with col2:
            st.metric("Assets", len(rows))
        with col3:
            mt = st.session_state.main_token or "-"
            main_val = 0.0
            if mt and mt in st.session_state.portfolio and mt in prices:
                main_val = st.session_state.portfolio[mt]['total'] * prices[mt]['bid']
            st.metric("Main Token", mt, f"${main_val:,.2f}")

    def render_matrix(self):
        st.header("🎯 Swap Matrix & Baseline / Top tracking")
        if not st.session_state.main_token:
            st.info("Connect to MEXC and allow fetching portfolio to compute matrix.")
            return
        main = st.session_state.main_token
        amt = st.session_state.portfolio[main]['total']
        baseline = st.session_state.baseline_data.get(main, {})
        st.subheader(f"Main token: {main} ({amt:.6f})")
        if baseline:
            ts = baseline.get('timestamp')
            ts_display = datetime.fromtimestamp(ts/1000).strftime("%Y-%m-%d %H:%M:%S") if ts else "N/A"
            st.caption(f"Baseline timestamp: {ts_display} | Baseline USDT value: ${baseline.get('usdt_value',0):,.2f}")
        # build table
        rows = []
        for target in self.tokens_to_track:
            if target == main:
                continue
            current_eq = self.calculate_equivalent(main, target, amt)
            baseline_eq = baseline.get('equivalents', {}).get(target, current_eq)
            top_eq = st.session_state.top_equivalents.get(target, baseline_eq)
            # percent changes
            pct_from_baseline = ((current_eq - baseline_eq) / baseline_eq * 100) if baseline_eq > 0 else 0.0
            top_usdt = st.session_state.top_usdt_values.get(target, top_eq * (st.session_state.prices.get(target,{}).get('bid',0)))
            pct_from_top = ((current_eq - top_eq) / top_eq * 100) if top_eq > 0 else 0.0
            status = "🔴"
            if pct_from_top >= self.swap_threshold:
                status = "🟢 SWAP"
            elif pct_from_baseline >= 0:
                status = "🟡"
            rows.append({
                "Target": target,
                "Current eq": f"{current_eq:.6f}",
                "Baseline eq": f"{baseline_eq:.6f}",
                "Δ from baseline": f"{pct_from_baseline:+.2f}%",
                "Top eq": f"{top_eq:.6f}",
                "Δ from top": f"{pct_from_top:+.2f}%",
                "Top USDT (approx)": f"${top_usdt:,.2f}",
                "Status": status
            })
        df = pd.DataFrame(rows).sort_values("Δ from top", ascending=False, key=lambda col: col.map(lambda x: float(x.replace('%',''))))
        st.dataframe(df, use_container_width=True, hide_index=True)

    # ---------------- sidebar / controls ----------------
    def setup_api_credentials(self):
        st.sidebar.header("🔐 MEXC API")
        with st.sidebar.form("api_form"):
            api_key = st.text_input("API Key", type="password")
            secret_key = st.text_input("Secret Key", type="password")
            submit = st.form_submit_button("Connect to MEXC")
            if submit:
                if api_key and secret_key:
                    try:
                        ping = requests.get("https://api.mexc.com/api/v3/ping", timeout=10)
                        if ping.status_code == 200:
                            st.session_state.tracking = True
                            st.session_state.api_key = api_key
                            st.session_state.secret_key = secret_key
                            # fetch portfolio immediately
                            st.session_state.portfolio = self.get_real_portfolio(api_key, secret_key)
                            # refresh prices
                            st.session_state.prices = self.get_prices()
                            # set baseline & main detection on connect
                            self.detect_swap_and_handle(api_key, secret_key)
                            st.experimental_rerun()
                        else:
                            st.error("API ping failed")
                    except Exception as e:
                        st.error(f"Connection error: {e}")
                else:
                    st.error("Provide both API key and secret")

        if st.session_state.tracking:
            if st.sidebar.button("Refresh now"):
                st.session_state.prices = self.get_prices()
                st.session_state.portfolio = self.get_real_portfolio(st.session_state.api_key, st.session_state.secret_key)
                # detect & handle any swaps
                self.detect_swap_and_handle(st.session_state.api_key, st.session_state.secret_key)
                st.experimental_rerun()

    # ---------------- run loop ----------------
    def run(self):
        st.title("🔄 Crypto Swap Matrix - Real Portfolio")
        st.markdown("---")
        self.init_session_state()
        col1, col2 = st.columns([3, 1])
        with col1:
            if not st.session_state.tracking:
                st.info("Configure MEXC API in the sidebar to fetch your portfolio.")
            self.render_portfolio()
            st.markdown("---")
            self.render_matrix()
        with col2:
            self.setup_api_credentials()
            if st.session_state.tracking:
                st.sidebar.markdown("---")
                st.sidebar.write("Auto-refresh every 3s (prices + portfolio). Close tab to stop.")
        # auto refresh & detect swaps
        if st.session_state.tracking:
            # update prices no more often than 3s
            if (datetime.now() - st.session_state.last_price_update).total_seconds() >= 3:
                st.session_state.prices = self.get_prices()
                st.session_state.last_price_update = datetime.now()
            # refresh portfolio
            st.session_state.portfolio = self.get_real_portfolio(st.session_state.api_key, st.session_state.secret_key)
            # detect swap & handle baseline/top
            self.detect_swap_and_handle(st.session_state.api_key, st.session_state.secret_key)
            time.sleep(3)
            st.experimental_rerun()


# ============= entry point ==============
if __name__ == "__main__":
    app = CryptoSwapMatrix()
    app.run()
