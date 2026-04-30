"""
R5 Combined — All 8 strategy files in one Trader
=================================================
1. Snackpack (5)      — MM + pairs trading
2. Sleep+Translator (10) — LAMB_WOOL fade + EMA MM
3. Robot (5)          — IRONING mean-revert, LAUNDRY/VAC imb, DISHES/MOP passive
4. Pebbles (5)        — allsum basket arb
5. UV Visor (5)       — wall-mid MM per-product config
6. Microchip (5)      — MM with per-product cap/uncap
7. Oxygen (5)         — per-product regime strategies
8. Panel (5)          — mixed (skip 1X2, imb, passive, EMA div)
Total: 45 products
"""

import json
import math
from typing import Dict, List, Optional, Tuple
from datamodel import Order, OrderDepth, TradingState

LIMIT = 10
DAY_LENGTH = 100_000


# ═══════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ═══════════════════════════════════════════════════════════════════════

def _bb_ba(d):
    return (max(d.buy_orders) if d.buy_orders else None,
            min(d.sell_orders) if d.sell_orders else None)

def _mid(d):
    if not d.buy_orders or not d.sell_orders: return None
    return (max(d.buy_orders) + min(d.sell_orders)) / 2.0

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))

def _imb_l1(d):
    b, a = _bb_ba(d)
    if b is None or a is None: return 0.0
    bv = d.buy_orders[b]; av = abs(d.sell_orders[a])
    if bv + av == 0: return 0.0
    return (bv - av) / (bv + av)

def _wmid(od):
    if not od.buy_orders or not od.sell_orders: return None
    bb = max(od.buy_orders); ba = min(od.sell_orders)
    bv = od.buy_orders[bb]; av = abs(od.sell_orders[ba])
    tv = bv + av
    if tv == 0: return (bb + ba) / 2.0
    return (bb * av + ba * bv) / tv


# ═══════════════════════════════════════════════════════════════════════
# 1. SNACKPACK — MM + pairs
# ═══════════════════════════════════════════════════════════════════════

SNACK_PRODUCTS = ["SNACKPACK_CHOCOLATE", "SNACKPACK_PISTACHIO",
                  "SNACKPACK_RASPBERRY", "SNACKPACK_STRAWBERRY", "SNACKPACK_VANILLA"]
SNACK_TAKE_EDGE = 0.5; SNACK_MAKE_OFFSET = 1; SNACK_MIN_SPREAD = 2
SNACK_BASKET_DEV = 8; SNACK_BASKET_MAX_BIAS = 3
SNACK_SOFT = 6; SNACK_INV_SKEW = 0.5; SNACK_EMA_SPAN = 12

SNACK_PAIRS = [
    ("SNACKPACK_CHOCOLATE", "SNACKPACK_VANILLA",    "sum",  "choc_van"),
    ("SNACKPACK_RASPBERRY", "SNACKPACK_STRAWBERRY",  "sum",  "rasp_straw"),
    ("SNACKPACK_PISTACHIO", "SNACKPACK_RASPBERRY",   "sum",  "pist_rasp"),
    ("SNACKPACK_PISTACHIO", "SNACKPACK_STRAWBERRY",  "diff", "pist_straw"),
]
SNACK_PAIR_EWMA = 0.995; SNACK_PAIR_MAD_A = 0.995
SNACK_PAIR_ENTRY_Z = 1.5; SNACK_PAIR_MAX_LEG = 5; SNACK_PAIR_MIN_MAD = 5.0

def _snack_pair_targets(fvs, ps):
    targets = {}
    for pa, pb, pt, label in SNACK_PAIRS:
        if pa not in fvs or pb not in fvs: continue
        sp = fvs[pa] + fvs[pb] if pt == "sum" else fvs[pa] - fvs[pb]
        s = ps.get(label, {})
        sm = s.get("mean"); sm = sp if sm is None else SNACK_PAIR_EWMA * sm + (1 - SNACK_PAIR_EWMA) * sp
        s["mean"] = sm
        ad = abs(sp - sm)
        mad = s.get("mad"); mad = ad if mad is None else SNACK_PAIR_MAD_A * mad + (1 - SNACK_PAIR_MAD_A) * ad
        mad = max(mad, SNACK_PAIR_MIN_MAD); s["mad"] = mad; ps[label] = s
        z = (sp - sm) / mad
        if abs(z) < SNACK_PAIR_ENTRY_Z: continue
        lots = min(int(round(1 + (abs(z) - SNACK_PAIR_ENTRY_Z) / SNACK_PAIR_ENTRY_Z * (SNACK_PAIR_MAX_LEG - 1))), SNACK_PAIR_MAX_LEG)
        if pt == "sum":
            d = -1 if z > 0 else 1
            targets[pa] = targets.get(pa, 0) + d * lots
            targets[pb] = targets.get(pb, 0) + d * lots
        else:
            if z > 0: targets[pa] = targets.get(pa, 0) - lots; targets[pb] = targets.get(pb, 0) + lots
            else: targets[pa] = targets.get(pa, 0) + lots; targets[pb] = targets.get(pb, 0) - lots
    return {p: _clamp(int(round(v)), -LIMIT, LIMIT) for p, v in targets.items()}

def trade_snackpack(state, result, data):
    emas = data.get("snk_ema", {}); ps = data.get("snk_pairs", {})
    fvs = {}
    for p in SNACK_PRODUCTS:
        od = state.order_depths.get(p)
        if od is None: continue
        fv = _wmid(od) or _mid(od)
        if fv is not None:
            fvs[p] = fv
            prev = emas.get(p)
            alpha = 2.0 / (SNACK_EMA_SPAN + 1)
            emas[p] = fv if prev is None else alpha * fv + (1 - alpha) * prev
    avail_emas = {p: emas[p] for p in SNACK_PRODUCTS if p in emas}
    basket_mid = sum(avail_emas.values()) / len(avail_emas) if len(avail_emas) >= 3 else None
    pt = _snack_pair_targets(fvs, ps)

    for p in SNACK_PRODUCTS:
        if p not in fvs or p not in state.order_depths: continue
        od = state.order_depths[p]; fv = fvs[p]
        pos = state.position.get(p, 0); orders = []
        mb = LIMIT - pos; ms = LIMIT + pos
        tp = pt.get(p, 0); ep = pos - tp
        bb_bias = 0.0
        if basket_mid is not None and p in emas:
            dev = emas[p] - basket_mid
            if abs(dev) > SNACK_BASKET_DEV:
                raw = (abs(dev) - SNACK_BASKET_DEV) / 10.0
                bb_bias = -min(raw, SNACK_BASKET_MAX_BIAS) if dev < 0 else min(raw, SNACK_BASKET_MAX_BIAS)
        afv = fv - bb_bias
        be = SNACK_TAKE_EDGE; se = SNACK_TAKE_EDGE
        if ep > SNACK_SOFT: be += (ep - SNACK_SOFT) * 0.5; se = max(se - 0.5, 0)
        elif ep < -SNACK_SOFT: se += (-ep - SNACK_SOFT) * 0.5; be = max(be - 0.5, 0)
        rb = mb
        for ap in sorted(od.sell_orders.keys()):
            if ap <= afv - be and rb > 0:
                q = min(-od.sell_orders[ap], rb); orders.append(Order(p, ap, q)); rb -= q
            else: break
        rs = ms
        for bp in sorted(od.buy_orders.keys(), reverse=True):
            if bp >= afv + se and rs > 0:
                q = min(od.buy_orders[bp], rs); orders.append(Order(p, bp, -q)); rs -= q
            else: break
        if tp != 0:
            gap = tp - pos
            if gap > 0 and rb > 0 and od.sell_orders:
                cq = min(gap, rb); bask = min(od.sell_orders.keys())
                if bask <= afv + 2:
                    for ap in sorted(od.sell_orders.keys()):
                        if ap <= afv + 2 and cq > 0:
                            q = min(-od.sell_orders[ap], cq); orders.append(Order(p, ap, q)); rb -= q; cq -= q
                        else: break
            elif gap < 0 and rs > 0 and od.buy_orders:
                cq = min(-gap, rs); bbid = max(od.buy_orders.keys())
                if bbid >= afv - 2:
                    for bp in sorted(od.buy_orders.keys(), reverse=True):
                        if bp >= afv - 2 and cq > 0:
                            q = min(od.buy_orders[bp], cq); orders.append(Order(p, bp, -q)); rs -= q; cq -= q
                        else: break
        bb, ba = _bb_ba(od)
        if bb is not None and ba is not None:
            sp = ba - bb
            if sp >= SNACK_MIN_SPREAD:
                bpx = bb + SNACK_MAKE_OFFSET; spx = ba - SNACK_MAKE_OFFSET
                if abs(ep) > SNACK_SOFT:
                    exc = abs(ep) - SNACK_SOFT; sk = int(round(exc * SNACK_INV_SKEW))
                    if ep > 0: spx -= sk; bpx -= sk
                    else: bpx += sk; spx += sk
                elif tp != 0 and pos != tp:
                    if pos < tp: bpx = min(bpx + 1, int(math.floor(afv)))
                    elif pos > tp: spx = max(spx - 1, int(math.ceil(afv)))
                if bb_bias < -0.5: bpx = min(bpx + 1, int(afv))
                elif bb_bias > 0.5: spx = max(spx - 1, int(afv))
                bpx = min(int(bpx), int(math.floor(afv))); spx = max(int(spx), int(math.ceil(afv)))
                if bpx < spx:
                    if rb > 0: orders.append(Order(p, bpx, rb))
                    if rs > 0: orders.append(Order(p, spx, -rs))
            elif abs(ep) > SNACK_SOFT:
                if ep > SNACK_SOFT:
                    fq = min(rs, int(ep - SNACK_SOFT))
                    if fq > 0: orders.append(Order(p, bb, -fq))
                elif ep < -SNACK_SOFT:
                    fq = min(rb, int(-ep - SNACK_SOFT))
                    if fq > 0: orders.append(Order(p, ba, fq))
        result[p] = orders
    data["snk_ema"] = emas; data["snk_pairs"] = ps


# ═══════════════════════════════════════════════════════════════════════
# 2. SLEEP + TRANSLATOR — LAMB_WOOL fade + EMA MM
# ═══════════════════════════════════════════════════════════════════════

LAMB = "SLEEP_POD_LAMB_WOOL"
OTHER_SLEEP = ["SLEEP_POD_SUEDE", "SLEEP_POD_POLYESTER", "SLEEP_POD_NYLON", "SLEEP_POD_COTTON"]
TRANSLATORS = ["TRANSLATOR_SPACE_GRAY", "TRANSLATOR_ASTRO_BLACK",
               "TRANSLATOR_ECLIPSE_CHARCOAL", "TRANSLATOR_GRAPHITE_MIST", "TRANSLATOR_VOID_BLUE"]
ST_QS = 3; ST_TAKE = 4; ST_LAMB_THR = 0.01
ST_GEN_ALPHA = 0.005; ST_GEN_SCALE = 100; ST_SLEEP_CAP = 5; ST_TRANS_CAP = 5

def _st_revert_target(mid, slow, scale, cap):
    raw = -(mid - slow) / scale
    return max(-cap, min(cap, int(round(raw))))

def _st_mm_one(product, state, result, slow_ema, cap):
    od = state.order_depths.get(product)
    if od is None: return
    b, a = _bb_ba(od)
    if b is None or a is None: return
    pos = state.position.get(product, 0)
    mid = (b + a) / 2; sp = a - b
    if product not in slow_ema: slow_ema[product] = mid
    else: slow_ema[product] = ST_GEN_ALPHA * mid + (1 - ST_GEN_ALPHA) * slow_ema[product]
    target = _st_revert_target(mid, slow_ema[product], ST_GEN_SCALE, cap)
    lr = LIMIT - pos; sr = LIMIT + pos; orders = []
    if lr > 0:
        for ap in sorted(od.sell_orders.keys()):
            vol = -od.sell_orders[ap]
            if ap <= mid - ST_TAKE:
                q = min(vol, lr); orders.append(Order(product, ap, q)); lr -= q; pos += q
            if lr <= 0: break
    if sr > 0:
        for bp in sorted(od.buy_orders.keys(), reverse=True):
            vol = od.buy_orders[bp]
            if bp >= mid + ST_TAKE:
                q = min(vol, sr); orders.append(Order(product, bp, -q)); sr -= q; pos -= q
            if sr <= 0: break
    if sp > 1:
        cl = max(0, cap - pos); cs = max(0, cap + pos)
        bs = max(0, min(ST_QS + max(0, target - pos), lr, cl))
        asz = max(0, min(ST_QS + max(0, pos - target), sr, cs))
        if bs > 0: orders.append(Order(product, b + 1, bs))
        if asz > 0: orders.append(Order(product, a - 1, -asz))
    if orders: result[product] = orders

def trade_sleep_translator(state, result, data):
    sema = data.get("st_ema", {}); mem = data.get("st_mem", {})
    # LAMB_WOOL fade
    if LAMB in state.order_depths:
        od = state.order_depths[LAMB]
        if od.buy_orders and od.sell_orders:
            b = max(od.buy_orders); a = min(od.sell_orders); mid = (b + a) / 2
            day_ts = state.timestamp % 1_000_000
            last = mem.get("lamb_last_day_ts", day_ts)
            if day_ts < last: mem.pop("lamb_open", None); mem.pop("lamb_signal", None)
            if "lamb_open" not in mem: mem["lamb_open"] = mid
            if "lamb_signal" not in mem:
                er = (mid - mem["lamb_open"]) / mem["lamb_open"]
                if abs(er) >= ST_LAMB_THR: mem["lamb_signal"] = -1 if er > 0 else 1
            mem["lamb_last_day_ts"] = day_ts
            if "lamb_signal" in mem:
                target = mem["lamb_signal"] * LIMIT
                pos = state.position.get(LAMB, 0); needed = target - pos
                if needed > 0: result[LAMB] = [Order(LAMB, a, needed)]
                elif needed < 0: result[LAMB] = [Order(LAMB, b, needed)]
    for p in OTHER_SLEEP: _st_mm_one(p, state, result, sema, ST_SLEEP_CAP)
    for p in TRANSLATORS: _st_mm_one(p, state, result, sema, ST_TRANS_CAP)
    data["st_ema"] = sema; data["st_mem"] = mem


# ═══════════════════════════════════════════════════════════════════════
# 3. ROBOT — v28
# ═══════════════════════════════════════════════════════════════════════

ROBOTS = ["ROBOT_VACUUMING", "ROBOT_MOPPING", "ROBOT_DISHES", "ROBOT_LAUNDRY", "ROBOT_IRONING"]
ROB_QS = 5; ROB_IMB_THR = 0.4
ROB_IRO_ALPHA = 0.05; ROB_IRO_SCALE = 3.0; ROB_IRO_DRIFT = -3

def _rob_passive(sym, depth, state):
    b, a = _bb_ba(depth)
    if b is None or a is None or a <= b + 1: return []
    pos = state.position.get(sym, 0)
    bs = max(0, min(ROB_QS, LIMIT - pos)); asz = max(0, min(ROB_QS, LIMIT + pos))
    out = []
    if bs > 0: out.append(Order(sym, b + 1, bs))
    if asz > 0: out.append(Order(sym, a - 1, -asz))
    return out

def _rob_imb(sym, depth, state):
    b, a = _bb_ba(depth)
    if b is None or a is None or a <= b + 1: return []
    pos = state.position.get(sym, 0)
    bs = max(0, min(ROB_QS, LIMIT - pos)); asz = max(0, min(ROB_QS, LIMIT + pos))
    imb = _imb_l1(depth)
    if imb > ROB_IMB_THR: asz = 0
    elif imb < -ROB_IMB_THR: bs = 0
    out = []
    if bs > 0: out.append(Order(sym, b + 1, bs))
    if asz > 0: out.append(Order(sym, a - 1, -asz))
    return out

def _rob_ironing(sym, depth, state, slow):
    b, a = _bb_ba(depth)
    if b is None or a is None or a <= b + 1: return []
    mid = (b + a) / 2.0; pos = state.position.get(sym, 0)
    dev = mid - slow
    tgt = max(-LIMIT, min(LIMIT, int(round(ROB_IRO_DRIFT + (-dev / ROB_IRO_SCALE)))))
    lr = LIMIT - pos; sr = LIMIT + pos
    be = max(0, tgt - pos); ae = max(0, pos - tgt)
    bs = max(0, min(ROB_QS + be, lr)); asz = max(0, min(ROB_QS + ae, sr))
    out = []
    if bs > 0: out.append(Order(sym, b + 1, bs))
    if asz > 0: out.append(Order(sym, a - 1, -asz))
    return out

def trade_robot(state, result, data):
    iro_ema = data.get("rob_iro_ema")
    for sym in ROBOTS:
        if sym not in state.order_depths: continue
        depth = state.order_depths[sym]
        if sym == "ROBOT_IRONING":
            mid = _mid(depth)
            if mid is not None:
                iro_ema = mid if iro_ema is None else ROB_IRO_ALPHA * mid + (1 - ROB_IRO_ALPHA) * iro_ema
                data["rob_iro_ema"] = iro_ema
                so = _rob_ironing(sym, depth, state, iro_ema)
            else: so = []
        elif sym in ("ROBOT_LAUNDRY", "ROBOT_VACUUMING"):
            so = _rob_imb(sym, depth, state)
        else:
            so = _rob_passive(sym, depth, state)
        if so: result[sym] = so


# ═══════════════════════════════════════════════════════════════════════
# 4. PEBBLES — allsum basket arb
# ═══════════════════════════════════════════════════════════════════════

PEBBLES = ["PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L", "PEBBLES_XL"]
PEB_TAKE = 30; PEB_ME = 1; PEB_SOFT = 7; PEB_ALPHA = 2.0 / (200 + 1)

def trade_pebbles(state, result, data):
    bs = data.get("peb_bsum")
    mids = {}
    for p in PEBBLES:
        od = state.order_depths.get(p)
        m = _mid(od) if od else None
        if m is not None: mids[p] = m
    if len(mids) < len(PEBBLES): data["peb_bsum"] = bs; return
    obs = sum(mids.values())
    bs = obs if bs is None else PEB_ALPHA * obs + (1 - PEB_ALPHA) * bs
    fair = {}; prem = {}
    for p in PEBBLES:
        oth = sum(mids[q] for q in PEBBLES if q != p)
        fair[p] = bs - oth; prem[p] = mids[p] - fair[p]
    for p in PEBBLES:
        od = state.order_depths.get(p)
        if od is None: continue
        bb, ba = _bb_ba(od); pos = state.position.get(p, 0)
        mb = LIMIT - pos; ms = LIMIT + pos; pr = prem[p]; orders = []
        if pr > PEB_TAKE and bb is not None and ms > 0:
            rem = ms
            for px in sorted(od.buy_orders.keys(), reverse=True):
                if px > fair[p] and rem > 0: q = min(od.buy_orders[px], rem); orders.append(Order(p, px, -q)); rem -= q
        elif pr < -PEB_TAKE and ba is not None and mb > 0:
            rem = mb
            for px in sorted(od.sell_orders.keys()):
                if px < fair[p] and rem > 0: q = min(-od.sell_orders[px], rem); orders.append(Order(p, px, q)); rem -= q
        if bb is not None and ba is not None and ba - bb > 2 * PEB_ME:
            mkb = bb + PEB_ME; mka = ba - PEB_ME; inv = pos / LIMIT
            bq = _clamp(int(mb * (1 - inv) / 2 + 1), 0, mb)
            sq = _clamp(int(ms * (1 + inv) / 2 + 1), 0, ms)
            if pr > 0: bq = _clamp(bq - 1, 0, mb); sq = _clamp(sq + 1, 0, ms)
            elif pr < 0: bq = _clamp(bq + 1, 0, mb); sq = _clamp(sq - 1, 0, ms)
            if bq > 0: orders.append(Order(p, mkb, bq))
            if sq > 0: orders.append(Order(p, mka, -sq))
        if abs(pos) >= PEB_SOFT and abs(pr) < PEB_TAKE:
            fp = round(mids[p])
            if pos > 0 and ms > 0: orders.append(Order(p, fp, -1))
            elif pos < 0 and mb > 0: orders.append(Order(p, fp, 1))
        result[p] = orders
    data["peb_bsum"] = bs


# ═══════════════════════════════════════════════════════════════════════
# 5. UV VISOR — wall-mid MM
# ═══════════════════════════════════════════════════════════════════════

VISOR_CFG = {
    "UV_VISOR_AMBER":   {"skew_div": 5,  "max_make_vol": 5, "fv_bias": -1.0},
    "UV_VISOR_YELLOW":  {"skew_div": 10, "max_make_vol": 5, "fv_bias":  0.0},
    "UV_VISOR_MAGENTA": {"skew_div": 10, "max_make_vol": 5, "fv_bias":  0.0},
    "UV_VISOR_RED":     {"skew_div": 10, "max_make_vol": 5, "fv_bias":  0.0},
    "UV_VISOR_ORANGE":  {"skew_div": 10, "max_make_vol": 5, "fv_bias":  0.0},
}

def trade_visor(state, result, data):
    for product, cfg in VISOR_CFG.items():
        od = state.order_depths.get(product)
        if od is None or not od.buy_orders or not od.sell_orders: continue
        pos = state.position.get(product, 0)
        bls = sorted(od.buy_orders.items(), key=lambda x: -x[0])
        sls = sorted(od.sell_orders.items(), key=lambda x: x[0])
        wb = bls[-1][0]; wa = sls[-1][0]; wm = (wb + wa) / 2.0
        bb = bls[0][0]; ba = sls[0][0]
        fv = wm - pos / cfg["skew_div"] + cfg["fv_bias"]
        mb = LIMIT - pos; ms = LIMIT + pos; orders = []
        for ap, av in sls:
            av = abs(av)
            if ap <= fv and mb > 0: v = min(av, mb); orders.append(Order(product, ap, v)); mb -= v
            else: break
        for bp, bv in bls:
            bv = abs(bv)
            if bp >= fv and ms > 0: v = min(bv, ms); orders.append(Order(product, bp, -v)); ms -= v
            else: break
        bpx = bb + 1
        for bp, bv in bls:
            if bv > 1 and bp + 1 < fv: bpx = max(bpx, bp + 1); break
            elif bp < fv: bpx = max(bpx, bp); break
        apx = ba - 1
        for ap, av in sls:
            av = abs(av)
            if av > 1 and ap - 1 > fv: apx = min(apx, ap - 1); break
            elif ap > fv: apx = min(apx, ap); break
        bpx = int(bpx); apx = int(apx)
        if bpx >= apx: bpx = math.floor(fv) - 1; apx = math.ceil(fv) + 1
        mmv = cfg["max_make_vol"]
        if mb > 0: orders.append(Order(product, bpx, min(mmv, mb)))
        if ms > 0: orders.append(Order(product, apx, -min(mmv, ms)))
        result[product] = orders


# ═══════════════════════════════════════════════════════════════════════
# 6. MICROCHIP — MM with per-product cap/uncap
# ═══════════════════════════════════════════════════════════════════════

MC_CAPPED = {"MICROCHIP_RECTANGLE", "MICROCHIP_SQUARE"}
MC_UNCAPPED = {"MICROCHIP_CIRCLE", "MICROCHIP_OVAL", "MICROCHIP_TRIANGLE"}
MC_ALL = MC_CAPPED | MC_UNCAPPED
MC_QS = 3; MC_TAKE = 4; MC_INV_CAP = 5

def trade_microchip(state, result, data):
    for product in MC_ALL:
        od = state.order_depths.get(product)
        if od is None: continue
        b, a = _bb_ba(od)
        if b is None or a is None: continue
        pos = state.position.get(product, 0); mid = (b + a) / 2; sp = a - b
        lr = LIMIT - pos; sr = LIMIT + pos; orders = []
        if lr > 0:
            for ap in sorted(od.sell_orders.keys()):
                vol = -od.sell_orders[ap]
                if ap <= mid - MC_TAKE: q = min(vol, lr); orders.append(Order(product, ap, q)); lr -= q; pos += q
                if lr <= 0: break
        if sr > 0:
            for bp in sorted(od.buy_orders.keys(), reverse=True):
                vol = od.buy_orders[bp]
                if bp >= mid + MC_TAKE: q = min(vol, sr); orders.append(Order(product, bp, -q)); sr -= q; pos -= q
                if sr <= 0: break
        if sp > 1:
            if product in MC_CAPPED:
                cl = max(0, MC_INV_CAP - pos); cs = max(0, MC_INV_CAP + pos)
                bs = max(0, min(MC_QS + max(0, -pos), lr, cl))
                asz = max(0, min(MC_QS + max(0, pos), sr, cs))
            else:
                bs = max(0, min(MC_QS + max(0, -pos), lr))
                asz = max(0, min(MC_QS + max(0, pos), sr))
            if bs > 0: orders.append(Order(product, b + 1, bs))
            if asz > 0: orders.append(Order(product, a - 1, -asz))
        if orders: result[product] = orders


# ═══════════════════════════════════════════════════════════════════════
# 7. OXYGEN — per-product regime
# ═══════════════════════════════════════════════════════════════════════

OXY_RULES = {
    "OXYGEN_SHAKE_EVENING_BREATH": {"kind": "evening_regime", "span": 2.0, "entry": 20.0, "exit": 0.0, "low_open_ceiling": 9000.0, "mid_open_ceiling": 9800.0, "mid_open_revert_entry": 25.0},
    "OXYGEN_SHAKE_CHOCOLATE": {"kind": "mm", "quote_size": 3, "make_offset": 1, "skew_div": 10, "max_make_vol": 5},
    "OXYGEN_SHAKE_MINT": {"kind": "ema_divergence", "fast_alpha": 0.15, "slow_alpha": 0.01, "vol_alpha": 0.003, "base_threshold": 120.0, "vol_mult": 0.5, "max_spread": 20},
    "OXYGEN_SHAKE_MORNING_BREATH": {"kind": "morning_regime", "low_open_ceiling": 9800.0, "low_open_revert_entry": 200.0, "high_open_floor": 10500.0, "exit": 0.0},
    "OXYGEN_SHAKE_GARLIC": {"kind": "garlic_regime", "low_open_ceiling": 11000.0, "first_break": 300.0},
}

def _oxy_target_from_signal(pos, signal, entry, exit_e):
    if pos == 0:
        if signal > entry: return LIMIT
        if signal < -entry: return -LIMIT
        return 0
    if pos > 0:
        if signal < -entry: return -LIMIT
        if signal < exit_e: return 0
        return LIMIT
    if signal > entry: return LIMIT
    if signal > -exit_e: return 0
    return -LIMIT

def _oxy_target(rule, pos, mid, ts):
    kind = rule["kind"]
    if kind == "evening_regime":
        om = float(rule["open"])
        if om < float(rule["low_open_ceiling"]): return LIMIT
        if om < float(rule["mid_open_ceiling"]):
            return _oxy_target_from_signal(pos, om - mid, float(rule["mid_open_revert_entry"]), 0.0)
        return _oxy_target_from_signal(pos, float(rule["fair"]) - mid, float(rule["entry"]), float(rule["exit"]))
    if kind == "morning_regime":
        om = float(rule["open"])
        if om >= float(rule["high_open_floor"]): return -LIMIT
        if om < float(rule["low_open_ceiling"]):
            return _oxy_target_from_signal(pos, om - mid, float(rule["low_open_revert_entry"]), 0.0)
        return LIMIT
    if kind == "garlic_regime":
        om = float(rule["open"])
        if om < float(rule["low_open_ceiling"]): return LIMIT
        mode = int(float(rule.get("mode", 0.0)))
        if mode == 1: return _oxy_target_from_signal(pos, om - mid, float(rule["first_break"]), 0.0)
        if mode == 2: return LIMIT
        return 0
    return 0

def trade_oxygen(state, result, data):
    pers = data.get("oxy", {})
    ns = {}
    for product, base_rule in OXY_RULES.items():
        od = state.order_depths.get(product)
        if od is None: continue
        b, a = _bb_ba(od)
        if b is None or a is None: continue
        mid = (b + a) / 2.0; rule = dict(base_rule); kind = rule["kind"]
        pos = state.position.get(product, 0)
        if kind == "mm":
            sp = a - b
            if sp < 2: continue
            wm = mid; sd = float(rule.get("skew_div", 10))
            fv = wm - pos / sd
            bpx = b + int(rule.get("make_offset", 1)); apx = a - int(rule.get("make_offset", 1))
            if bpx >= fv: bpx = int(fv) - 1
            if apx <= fv: apx = int(fv) + 1
            if bpx >= apx: continue
            lr = LIMIT - pos; sr = LIMIT + pos
            qs = int(rule.get("quote_size", 3)); cap = int(rule.get("max_make_vol", 5))
            orders = []
            if lr > 0: orders.append(Order(product, bpx, min(qs, cap, lr)))
            if sr > 0: orders.append(Order(product, apx, -min(qs, cap, sr)))
            if orders: result[product] = orders
            continue
        if kind == "ema_divergence":
            sp = a - b
            if sp > float(rule.get("max_spread", 20)): continue
            fk = f"mf:{product}"; sk = f"ms:{product}"; vk = f"mv:{product}"
            fast = pers.get(fk, mid); slow = pers.get(sk, mid); sv = pers.get(vk, 0.0)
            fa = float(rule["fast_alpha"]); sa = float(rule["slow_alpha"]); va = float(rule["vol_alpha"])
            fast = (1 - fa) * fast + fa * mid; slow = (1 - sa) * slow + sa * mid
            sig = slow - fast; sv = (1 - va) * sv + va * abs(sig)
            ns[fk] = fast; ns[sk] = slow; ns[vk] = sv
            thr = float(rule["base_threshold"]) + float(rule["vol_mult"]) * sv
            target = pos
            if sig > thr: target = LIMIT
            elif sig < -thr: target = -LIMIT
            orders = []
            if target > pos:
                q = min(target - pos, LIMIT - pos, -od.sell_orders[a])
                if q > 0: orders.append(Order(product, a, q))
            elif target < pos:
                q = min(pos - target, LIMIT + pos, od.buy_orders[b])
                if q > 0: orders.append(Order(product, b, -q))
            if orders: result[product] = orders
            continue
        if kind in ("evening_regime",):
            fk = f"of:{product}"; pf = pers.get(fk, mid)
            alpha = 2.0 / (float(rule["span"]) + 1.0)
            fair = alpha * mid + (1 - alpha) * pf; rule["fair"] = fair; ns[fk] = fair
        if kind in ("evening_regime", "morning_regime", "garlic_regime"):
            ok = f"oo:{product}"; om = pers.get(ok, mid); rule["open"] = om; ns[ok] = om
        if kind == "garlic_regime":
            mk = f"om:{product}"; mode = pers.get(mk, 0.0)
            om = float(rule["open"])
            if mode == 0.0 and om >= float(rule["low_open_ceiling"]):
                fb = float(rule["first_break"])
                if mid - om >= fb: mode = 1.0
                elif om - mid >= fb: mode = 2.0
            rule["mode"] = mode; ns[mk] = mode
        target = _oxy_target(rule, pos, mid, state.timestamp)
        delta = target - pos; orders = []
        if delta > 0:
            rem = delta
            for ap in sorted(od.sell_orders.keys()):
                if rem <= 0: break
                vol = -od.sell_orders[ap]; q = min(rem, vol)
                if q > 0: orders.append(Order(product, ap, q)); rem -= q
        elif delta < 0:
            rem = -delta
            for bp in sorted(od.buy_orders.keys(), reverse=True):
                if rem <= 0: break
                vol = od.buy_orders[bp]; q = min(rem, vol)
                if q > 0: orders.append(Order(product, bp, -q)); rem -= q
        if orders: result[product] = orders
    data["oxy"] = ns


# ═══════════════════════════════════════════════════════════════════════
# 8. PANEL — mixed
# ═══════════════════════════════════════════════════════════════════════

PNL_QS = 3

def _pnl_passive(sym, depth, state, target=0):
    b, a = _bb_ba(depth)
    if b is None or a is None or a <= b + 1: return []
    pos = state.position.get(sym, 0)
    lr = LIMIT - pos; sr = LIMIT + pos
    bs = max(0, min(PNL_QS + max(0, target - pos), lr))
    asz = max(0, min(PNL_QS + max(0, pos - target), sr))
    out = []
    if bs > 0: out.append(Order(sym, b + 1, bs))
    if asz > 0: out.append(Order(sym, a - 1, -asz))
    return out

def _pnl_imb_momentum(product, state, result):
    od = state.order_depths.get(product)
    if od is None or not od.buy_orders or not od.sell_orders: return
    bps = sorted(od.buy_orders.keys(), reverse=True); aps = sorted(od.sell_orders.keys())
    bb = bps[0]; ba = aps[0]; sp = ba - bb
    if sp > 10: return
    bv = sum(od.buy_orders[p] for p in bps[:2])
    av = sum(-od.sell_orders[p] for p in aps[:2])
    if bv + av == 0: return
    imb = (bv - av) / (bv + av)
    pos = state.position.get(product, 0); target = pos
    if imb > 0.30: target = LIMIT
    elif imb < -0.30: target = -LIMIT
    orders = []
    if target > pos:
        q = min(target - pos, -od.sell_orders[ba])
        if q > 0: orders.append(Order(product, ba, q))
    elif target < pos:
        q = min(pos - target, od.buy_orders[bb])
        if q > 0: orders.append(Order(product, bb, -q))
    if orders: result[product] = orders

def _pnl_ema_div(product, state, result, data, fa, sa, entry, exit_):
    od = state.order_depths.get(product)
    if od is None or not od.buy_orders or not od.sell_orders: return
    bb = max(od.buy_orders); ba = min(od.sell_orders); mid = (bb + ba) / 2
    pd = data.setdefault(product, {})
    if "fast" not in pd: pd["fast"] = mid; pd["slow"] = mid; pd["target"] = 0; return
    pd["fast"] += fa * (mid - pd["fast"]); pd["slow"] += sa * (mid - pd["slow"])
    sig = pd["fast"] - pd["slow"]; tgt = int(pd.get("target", 0))
    if sig > entry: tgt = LIMIT
    elif sig < -entry: tgt = -LIMIT
    elif abs(sig) < exit_: tgt = 0
    pd["target"] = tgt; pos = state.position.get(product, 0); dc = tgt - pos; orders = []
    if dc > 0:
        rem = dc
        for ap in sorted(od.sell_orders.keys()):
            if rem <= 0: break
            q = min(rem, -od.sell_orders[ap])
            if q > 0: orders.append(Order(product, ap, q)); rem -= q
    elif dc < 0:
        rem = -dc
        for bp in sorted(od.buy_orders.keys(), reverse=True):
            if rem <= 0: break
            q = min(rem, od.buy_orders[bp])
            if q > 0: orders.append(Order(product, bp, -q)); rem -= q
    if orders: result[product] = orders

def trade_panel(state, result, data):
    pd = data.get("pnl", {})
    # PANEL_1X2 — skipped
    _pnl_imb_momentum("PANEL_1X4", state, result)
    _pnl_imb_momentum("PANEL_2X2", state, result)
    if "PANEL_2X4" in state.order_depths:
        so = _pnl_passive("PANEL_2X4", state.order_depths["PANEL_2X4"], state, target=0)
        if so: result["PANEL_2X4"] = so
    _pnl_ema_div("PANEL_4X4", state, result, pd, fa=2/(100+1), sa=2/(800+1), entry=8.0, exit_=3.0)
    data["pnl"] = pd


# ═══════════════════════════════════════════════════════════════════════
# MAIN TRADER
# ═══════════════════════════════════════════════════════════════════════

class Trader:
    def run(self, state: TradingState):
        data = json.loads(state.traderData) if state.traderData else {}
        result = {}

        trade_snackpack(state, result, data)
        trade_sleep_translator(state, result, data)
        trade_robot(state, result, data)
        trade_pebbles(state, result, data)
        trade_visor(state, result, data)
        trade_microchip(state, result, data)
        trade_oxygen(state, result, data)
        trade_panel(state, result, data)

        return result, 0, json.dumps(data)
