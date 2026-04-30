"""
PER-PRODUCT MAP:
  MORNING_BREATH:  morning_regime
  EVENING_BREATH:  evening_regime  
  MINT:            ema_divergence  
  CHOCOLATE:       mm passive      
  GARLIC:          garlic_regime  
"""
import json
from typing import Optional

from datamodel import Order, OrderDepth, Symbol, TradingState

DAY_LENGTH = 100_000


class Trader:
    POSITION_LIMIT = 10
    MIN_SPREAD = 2

    RULES = {
        "OXYGEN_SHAKE_EVENING_BREATH": {
            "kind": "evening_regime",
            "span": 2.0,
            "entry": 20.0,
            "exit": 0.0,
            "low_open_ceiling": 9000.0,
            "mid_open_ceiling": 9800.0,
            "mid_open_revert_entry": 25.0,
        },
        "OXYGEN_SHAKE_CHOCOLATE": {
            "kind": "mm",
            "quote_size": 3,
            "make_offset": 1,
            "skew_div": 10,
            "max_make_vol": 5,
        },
        "OXYGEN_SHAKE_MINT": {
            "kind": "ema_divergence",
            "fast_alpha": 0.15,
            "slow_alpha": 0.01,
            "vol_alpha": 0.003,
            "base_threshold": 120.0,
            "vol_mult": 0.5,
            "max_spread": 20,
        },
        "OXYGEN_SHAKE_MORNING_BREATH": {
            "kind": "morning_regime",
            "low_open_ceiling": 9800.0,
            "low_open_revert_entry": 200.0,
            "high_open_floor": 10500.0,
            "exit": 0.0,
        },
        "OXYGEN_SHAKE_GARLIC": {
            "kind": "garlic_regime",
            "low_open_ceiling": 11000.0,
            "first_break": 300.0,
        },
    }

    def _best_bid_ask(self, depth: OrderDepth):
        best_bid = max(depth.buy_orders) if depth.buy_orders else None
        best_ask = min(depth.sell_orders) if depth.sell_orders else None
        return best_bid, best_ask

    def _mid_price(self, depth: OrderDepth):
        bb, ba = self._best_bid_ask(depth)
        if bb is None and ba is None:
            return None
        if bb is None:
            return float(ba)
        if ba is None:
            return float(bb)
        return (bb + ba) / 2.0

    def _target_from_signal(self, position, signal, entry, exit_edge):
        if position == 0:
            if signal > entry: return self.POSITION_LIMIT
            if signal < -entry: return -self.POSITION_LIMIT
            return 0
        if position > 0:
            if signal < -entry: return -self.POSITION_LIMIT
            if signal < exit_edge: return 0
            return self.POSITION_LIMIT
        if signal > entry: return self.POSITION_LIMIT
        if signal > -exit_edge: return 0
        return -self.POSITION_LIMIT

    def _target_position(self, rule, position, mid, ts):
        kind = rule["kind"]
        if kind == "static_long":
            return self.POSITION_LIMIT
        if kind == "static_short":
            return -self.POSITION_LIMIT
        if kind == "time_window_short":
            ts_in_day = ts % DAY_LENGTH
            start = int(rule.get("window_start", 0))
            end = int(rule.get("window_end", DAY_LENGTH))
            max_pos = min(int(rule.get("max_position", self.POSITION_LIMIT)), self.POSITION_LIMIT)
            entry_pace = float(rule.get("entry_pace", 0.0))
            if start <= ts_in_day < end:
                if entry_pace > 0:
                    window_size = end - start
                    entry_dur = window_size * entry_pace
                    ts_into = ts_in_day - start
                    if ts_into >= entry_dur:
                        return -max_pos
                    return -int(round(max_pos * ts_into / entry_dur))
                return -max_pos
            return 0
        if kind == "time_window_long":
            ts_in_day = ts % DAY_LENGTH
            start = int(rule.get("window_start", 0))
            end = int(rule.get("window_end", DAY_LENGTH))
            max_pos = min(int(rule.get("max_position", self.POSITION_LIMIT)), self.POSITION_LIMIT)
            entry_pace = float(rule.get("entry_pace", 0.0))
            if start <= ts_in_day < end:
                if entry_pace > 0:
                    window_size = end - start
                    entry_dur = window_size * entry_pace
                    ts_into = ts_in_day - start
                    if ts_into >= entry_dur:
                        return max_pos
                    return int(round(max_pos * ts_into / entry_dur))
                return max_pos
            return 0
        if kind == "evening_regime":
            open_mid = float(rule["open"])
            if open_mid < float(rule["low_open_ceiling"]):
                return self.POSITION_LIMIT
            if open_mid < float(rule["mid_open_ceiling"]):
                signal = open_mid - mid
                return self._target_from_signal(position, signal, float(rule["mid_open_revert_entry"]), 0.0)
            signal = float(rule["fair"]) - mid
            return self._target_from_signal(position, signal, float(rule["entry"]), float(rule["exit"]))
        if kind == "mint_regime":
            open_mid = float(rule["open"])
            if open_mid < float(rule["low_open_ceiling"]):
                entry = float(rule["low_open_entry"])
            elif open_mid > float(rule["high_open_floor"]):
                entry = float(rule["high_open_entry"])
            else:
                entry = float(rule["entry"])
            signal = open_mid - mid
            return self._target_from_signal(position, signal, entry, float(rule["exit"]))
        if kind == "morning_regime":
            open_mid = float(rule["open"])
            if open_mid >= float(rule["high_open_floor"]):
                return -self.POSITION_LIMIT
            if open_mid < float(rule["low_open_ceiling"]):
                signal = open_mid - mid
                return self._target_from_signal(position, signal, float(rule["low_open_revert_entry"]), 0.0)
            return self.POSITION_LIMIT
        if kind == "garlic_regime":
            open_mid = float(rule["open"])
            if open_mid < float(rule["low_open_ceiling"]):
                return self.POSITION_LIMIT
            mode = int(float(rule.get("mode", 0.0)))
            if mode == 1:
                signal = open_mid - mid
                return self._target_from_signal(position, signal, float(rule["first_break"]), 0.0)
            if mode == 2:
                return self.POSITION_LIMIT
            return 0
        if kind == "chocolate_combo":
            open_mid = float(rule["open"])
            if float(rule["revert_open_floor"]) <= open_mid < float(rule["revert_open_ceiling"]):
                signal = open_mid - mid
                return self._target_from_signal(position, signal, float(rule["revert_entry"]), 0.0)
            signal = float(rule["fair"]) - mid
            return self._target_from_signal(position, signal, float(rule["entry"]), float(rule["exit"]))
        return 0

    def _mm_orders(self, product, depth, position, rule):
        bb, ba = self._best_bid_ask(depth)
        if bb is None or ba is None or ba - bb < self.MIN_SPREAD:
            return []
        wall_mid = (bb + ba) / 2.0
        skew_div = float(rule.get("skew_div", 10))
        fv = wall_mid - position / skew_div
        bid_px = bb + int(rule.get("make_offset", 1))
        ask_px = ba - int(rule.get("make_offset", 1))
        if bid_px >= fv:
            bid_px = int(fv) - 1
        if ask_px <= fv:
            ask_px = int(fv) + 1
        if bid_px >= ask_px:
            return []
        long_room = self.POSITION_LIMIT - position
        short_room = self.POSITION_LIMIT + position
        qsize = int(rule.get("quote_size", 3))
        cap = int(rule.get("max_make_vol", 5))
        out = []
        if long_room > 0:
            out.append(Order(product, bid_px, min(qsize, cap, long_room)))
        if short_room > 0:
            out.append(Order(product, ask_px, -min(qsize, cap, short_room)))
        return out

    def _ema_divergence_orders(self, product, depth, position, rule, persisted, next_state):
        """Fast/slow EMA divergence trader. Goes ±LIMIT on extreme stretches."""
        bb, ba = self._best_bid_ask(depth)
        if bb is None or ba is None:
            return []
        mid = (bb + ba) / 2.0
        spread = ba - bb
        if spread > float(rule.get("max_spread", 20)):
            return []

        fast_key = f"mint_fast:{product}"
        slow_key = f"mint_slow:{product}"
        vol_key = f"mint_vol:{product}"
        fast = persisted.get(fast_key, mid)
        slow = persisted.get(slow_key, mid)
        signal_vol = persisted.get(vol_key, 0.0)

        fa = float(rule.get("fast_alpha", 0.15))
        sa = float(rule.get("slow_alpha", 0.01))
        va = float(rule.get("vol_alpha", 0.003))

        fast = (1.0 - fa) * fast + fa * mid
        slow = (1.0 - sa) * slow + sa * mid
        signal = slow - fast
        signal_vol = (1.0 - va) * signal_vol + va * abs(signal)

        next_state[fast_key] = fast
        next_state[slow_key] = slow
        next_state[vol_key] = signal_vol

        threshold = float(rule.get("base_threshold", 120.0)) + float(rule.get("vol_mult", 0.5)) * signal_vol

        target = position
        if signal > threshold:
            target = self.POSITION_LIMIT
        elif signal < -threshold:
            target = -self.POSITION_LIMIT

        out = []
        if target > position:
            best_ask_volume = -depth.sell_orders[ba]
            qty = min(target - position, self.POSITION_LIMIT - position, best_ask_volume)
            if qty > 0:
                out.append(Order(product, ba, qty))
        elif target < position:
            best_bid_volume = depth.buy_orders[bb]
            qty = min(position - target, self.POSITION_LIMIT + position, best_bid_volume)
            if qty > 0:
                out.append(Order(product, bb, -qty))
        return out

    def _sweep_buy_price(self, depth, quantity):
        remaining = quantity; last = None
        for px in sorted(depth.sell_orders):
            last = px
            remaining -= abs(depth.sell_orders[px])
            if remaining <= 0: return px
        return last

    def _sweep_sell_price(self, depth, quantity):
        remaining = quantity; last = None
        for px in sorted(depth.buy_orders, reverse=True):
            last = px
            remaining -= depth.buy_orders[px]
            if remaining <= 0: return px
        return last

    def run(self, state: TradingState):
        persisted: dict[str, float] = {}
        if state.traderData:
            try:
                loaded = json.loads(state.traderData)
                if isinstance(loaded, dict):
                    persisted = {str(k): float(v) for k, v in loaded.items()}
            except Exception:
                pass

        result: dict[Symbol, list[Order]] = {}
        next_state: dict[str, float] = {}

        for product, base_rule in self.RULES.items():
            depth = state.order_depths.get(product)
            if depth is None: continue
            mid = self._mid_price(depth)
            bb, ba = self._best_bid_ask(depth)
            if mid is None or bb is None or ba is None: continue

            rule = dict(base_rule)
            kind = rule["kind"]

            # MM kind: passive quoting, separate code path
            if kind == "mm":
                position = state.position.get(product, 0)
                orders = self._mm_orders(product, depth, position, rule)
                if orders:
                    result[product] = orders
                continue

            # EMA-divergence kind: fast/slow EMA + adaptive threshold, separate path
            if kind == "ema_divergence":
                position = state.position.get(product, 0)
                orders = self._ema_divergence_orders(
                    product, depth, position, rule, persisted, next_state
                )
                if orders:
                    result[product] = orders
                continue

            # Track EWMA fair for relevant kinds
            if kind in {"ewma_revert", "chocolate_combo", "evening_regime"}:
                fair_key = f"fair:{product}"
                prev_fair = persisted.get(fair_key, mid)
                alpha = 2.0 / (float(rule["span"]) + 1.0)
                fair = alpha * mid + (1.0 - alpha) * prev_fair
                rule["fair"] = fair
                next_state[fair_key] = fair

            # Track open price for regime modes
            if kind in {"open_trend", "open_revert", "chocolate_combo",
                        "evening_regime", "mint_regime", "morning_regime",
                        "garlic_regime"}:
                open_key = f"open:{product}"
                open_mid = persisted.get(open_key, mid)
                rule["open"] = open_mid
                next_state[open_key] = open_mid

            # Garlic mode tracking (first-break detection)
            if kind == "garlic_regime":
                mode_key = f"mode:{product}"
                mode = persisted.get(mode_key, 0.0)
                open_mid = float(rule["open"])
                if mode == 0.0 and open_mid >= float(rule["low_open_ceiling"]):
                    fb = float(rule["first_break"])
                    if mid - open_mid >= fb: mode = 1.0
                    elif open_mid - mid >= fb: mode = 2.0
                rule["mode"] = mode
                next_state[mode_key] = mode

            position = state.position.get(product, 0)
            target = self._target_position(rule, position, mid, state.timestamp)
            delta = target - position

            if delta > 0:
                price = self._sweep_buy_price(depth, delta)
                if price is not None:
                    result[product] = [Order(product, price, delta)]
            elif delta < 0:
                price = self._sweep_sell_price(depth, -delta)
                if price is not None:
                    result[product] = [Order(product, price, delta)]

        trader_data = json.dumps(next_state, separators=(",", ":"))
        return result, 0, trader_data
