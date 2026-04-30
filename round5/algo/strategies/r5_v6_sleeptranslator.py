"""
LAMB_WOOL: lock-and-hold fade (no tick wait, threshold-only)
Other 4 sleep: EMA target, INV_CAP=5
All 5 translators: EMA target, INV_CAP=5
"""
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
import json
from typing import Any, List

####################################################################################
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *o, sep=" ", end="\n"):
        self.logs += sep.join(map(str, o)) + end

    def flush(self, state, orders, conversions, trader_data):
        base_length = len(self.to_json([self.compress_state(state, ""), self.compress_orders(orders), conversions, "", ""]))
        max_item_length = (self.max_log_length - base_length) // 3
        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData or "", max_item_length)),
            self.compress_orders(orders), conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))
        self.logs = ""

    def compress_state(self, s, td):
        return [s.timestamp, td, self.compress_listings(s.listings),
                self.compress_order_depths(s.order_depths),
                self.compress_trades(s.own_trades), self.compress_trades(s.market_trades),
                s.position, self.compress_observations(s.observations)]

    def compress_listings(self, ls):
        return [[l.symbol, l.product, l.denomination] for l in ls.values()]

    def compress_order_depths(self, ods):
        return {s: [od.buy_orders, od.sell_orders] for s, od in ods.items()}

    def compress_trades(self, ts):
        return [[t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp]
                for tl in ts.values() for t in tl]

    def compress_observations(self, obs):
        co = {p: [o.bidPrice, o.askPrice, o.transportFees, o.exportTariff,
                  o.importTariff, o.sugarPrice, o.sunlightIndex]
              for p, o in obs.conversionObservations.items()}
        return [obs.plainValueObservations, co]

    def compress_orders(self, orders):
        return [[o.symbol, o.price, o.quantity] for ol in orders.values() for o in ol]

    def to_json(self, v):
        return json.dumps(v, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, v, mx):
        lo, hi = 0, min(len(v), mx)
        result = ""
        while lo <= hi:
            mid = (lo + hi) // 2
            cand = v[:mid] + ("..." if mid < len(v) else "")
            if len(json.dumps(cand)) <= mx:
                result = cand; lo = mid + 1
            else:
                hi = mid - 1
        return result

####################################################################################
LAMB = "SLEEP_POD_LAMB_WOOL"
OTHER_SLEEP = ["SLEEP_POD_SUEDE", "SLEEP_POD_POLYESTER",
               "SLEEP_POD_NYLON", "SLEEP_POD_COTTON"]
TRANSLATORS = ["TRANSLATOR_SPACE_GRAY", "TRANSLATOR_ASTRO_BLACK",
               "TRANSLATOR_ECLIPSE_CHARCOAL", "TRANSLATOR_GRAPHITE_MIST",
               "TRANSLATOR_VOID_BLUE"]

LIMIT = 10
QUOTE_SIZE = 3
TAKE_EDGE = 4

LAMB_THRESHOLD = 0.01

GEN_ALPHA = 0.005
GEN_SCALE = 100
SLEEP_CAP = 5
TRANS_CAP = 5


def best_bid_ask(d: OrderDepth):
    return (max(d.buy_orders) if d.buy_orders else None,
            min(d.sell_orders) if d.sell_orders else None)


def lamb_wool_fade(state: TradingState, result: dict, mem: dict) -> None:
    if LAMB not in state.order_depths:
        return
    od = state.order_depths[LAMB]
    if not od.buy_orders or not od.sell_orders:
        return

    bid = max(od.buy_orders)
    ask = min(od.sell_orders)
    mid = (bid + ask) / 2

    day_ts = state.timestamp % 1_000_000
    last_day_ts = mem.get("lamb_last_day_ts", day_ts)
    if day_ts < last_day_ts:
        mem.pop("lamb_open", None)
        mem.pop("lamb_signal", None)

    if "lamb_open" not in mem:
        mem["lamb_open"] = mid

    if "lamb_signal" not in mem:
        early_ret = (mid - mem["lamb_open"]) / mem["lamb_open"]
        if abs(early_ret) >= LAMB_THRESHOLD:
            mem["lamb_signal"] = -1 if early_ret > 0 else 1

    mem["lamb_last_day_ts"] = day_ts

    if "lamb_signal" in mem:
        target = mem["lamb_signal"] * LIMIT
        pos = state.position.get(LAMB, 0)
        needed = target - pos
        if needed > 0:
            result[LAMB] = [Order(LAMB, ask, needed)]
        elif needed < 0:
            result[LAMB] = [Order(LAMB, bid, needed)]


def revert_target(mid: float, slow: float, scale: float, cap: int) -> int:
    raw = -(mid - slow) / scale
    return max(-cap, min(cap, int(round(raw))))


def mm_one(product: str, state: TradingState, result: dict, slow_ema: dict,
           cap: int) -> None:
    if product not in state.order_depths:
        return
    od = state.order_depths[product]
    bid, ask = best_bid_ask(od)
    if bid is None or ask is None:
        return

    pos = state.position.get(product, 0)
    mid = (bid + ask) / 2
    spread = ask - bid

    if product not in slow_ema:
        slow_ema[product] = mid
    else:
        slow_ema[product] = GEN_ALPHA * mid + (1 - GEN_ALPHA) * slow_ema[product]

    target = revert_target(mid, slow_ema[product], GEN_SCALE, cap)

    long_room = LIMIT - pos
    short_room = LIMIT + pos

    orders: List[Order] = []

    if long_room > 0:
        for ap in sorted(od.sell_orders.keys()):
            vol = -od.sell_orders[ap]
            if ap <= mid - TAKE_EDGE:
                qty = min(vol, long_room)
                if qty > 0:
                    orders.append(Order(product, ap, qty))
                    long_room -= qty; pos += qty
            if long_room <= 0:
                break

    if short_room > 0:
        for bp in sorted(od.buy_orders.keys(), reverse=True):
            vol = od.buy_orders[bp]
            if bp >= mid + TAKE_EDGE:
                qty = min(vol, short_room)
                if qty > 0:
                    orders.append(Order(product, bp, -qty))
                    short_room -= qty; pos -= qty
            if short_room <= 0:
                break

    if spread > 1:
        cap_long = max(0, cap - pos)
        cap_short = max(0, cap + pos)
        bid_size = max(0, min(QUOTE_SIZE + max(0, target - pos), long_room, cap_long))
        ask_size = max(0, min(QUOTE_SIZE + max(0, pos - target), short_room, cap_short))
        if bid_size > 0:
            orders.append(Order(product, bid + 1, bid_size))
        if ask_size > 0:
            orders.append(Order(product, ask - 1, -ask_size))

    if orders:
        result[product] = orders


logger = Logger()


class Trader:
    def run(self, state: TradingState):
        data = json.loads(state.traderData) if state.traderData else {}
        slow_ema = data.get("slow_ema", {})
        mem = data.get("mem", {})

        result: dict[Symbol, list[Order]] = {}

        lamb_wool_fade(state, result, mem)
        for p in OTHER_SLEEP:
            mm_one(p, state, result, slow_ema, SLEEP_CAP)
        for p in TRANSLATORS:
            mm_one(p, state, result, slow_ema, TRANS_CAP)

        data["slow_ema"] = slow_ema
        data["mem"] = mem
        new_td = json.dumps(data)
        logger.flush(state, result, 0, new_td)
        return result, 0, new_td
