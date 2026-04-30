"""Insider-traders sentiment-driven prediction model for the Ashflow Alpha
manual challenge.

Pipeline:
  1. Encode each news item as a structured feature vector (sentiment,
     conviction, time horizon, signal type, ambiguity).
  2. Run several "insider personas" — each a different trading philosophy
     that maps features → (direction, percentage allocation).
  3. Aggregate persona votes into a consensus allocation.
  4. Monte-Carlo simulate PnL: for each news item, sample a price move from
     a distribution conditioned on its features. Compute net PnL (after the
     quadratic fee = (p/100)² * BUDGET) per persona and per consensus.

Usage:
    python3 insider_traders.py
"""
import math, random, statistics
from collections import defaultdict

BUDGET = 1_000_000

# ==============================================================================
# 1. NEWS FEATURE ENCODING
# ==============================================================================
#
# sentiment:    -1.0 (very bearish) to +1.0 (very bullish), price-direction polarity
# conviction:    0.0 (speculation/rumor) to 1.0 (concrete event happening now)
# horizon:      "immediate" | "tomorrow" | "short" | "medium" | "long"
# signal_type:  "demand_event"   - new product launch / hot drop
#               "supply_shock"   - factory halt, recall
#               "regulatory"     - tax change, rule change
#               "scandal"        - PR crisis
#               "forecast"       - analyst projection
#               "pump"           - influencer/social momentum
#               "index_flow"     - forced index buying
# ambiguity:    0.0 (clear direction) to 1.0 (could go either way)

NEWS = [
    {
        "name":        "Magma Ink",
        "headline":    "Limited-edition Lava Fountain Pen with Magma Ink — fans waited 6+ hours",
        "sentiment":   +0.65,
        "conviction":   0.80,
        "horizon":     "immediate",
        "signal_type": "demand_event",
        "ambiguity":    0.15,
    },
    {
        "name":        "Obsidian Cutlery",
        "headline":    "Manufacturing halted: blades cut through assembly line, contamination protocols",
        "sentiment":   +0.45,
        "conviction":   0.65,
        "horizon":     "short",
        "signal_type": "supply_shock",
        "ambiguity":    0.30,  # supply-shock direction is usually up but contamination panic could hurt demand
    },
    {
        "name":        "Pyroflex Cells",
        "headline":    "Tax cut abolished tomorrow — effectively doubles the levy",
        "sentiment":   +0.10,  # weak buy: tax-inclusive prices may rise but demand may fall
        "conviction":   0.35,
        "horizon":     "tomorrow",
        "signal_type": "regulatory",
        "ambiguity":    0.85,  # very ambiguous — depends on whether traded price is pre-tax or post-tax
    },
    {
        "name":        "Thermalite Core",
        "headline":    "Smart-home users projected 1.42M -> 3.89M next quarter (analyst speculation)",
        "sentiment":   +0.55,
        "conviction":   0.40,  # forecast/projection, not concrete
        "horizon":     "medium",  # next quarter, not tomorrow
        "signal_type": "forecast",
        "ambiguity":    0.40,
    },
    {
        "name":        "Lava Cake",
        "headline":    "Actual lava found in cakes — sales halted, lawsuits piling up",
        "sentiment":   -0.85,
        "conviction":   0.95,
        "horizon":     "immediate",
        "signal_type": "scandal",
        "ambiguity":    0.05,
    },
    {
        "name":        "Magma Ink",  # already above; using duplicate name only as sanity check
        "headline":    "(duplicate placeholder, ignored)",
        "sentiment":   0.0, "conviction": 0.0, "horizon": "long", "signal_type": "forecast", "ambiguity": 1.0,
    },
    {
        "name":        "Scoria Paste",
        "headline":    'Lava D. Ray ("self-proclaimed market medium") urges stockpiling',
        "sentiment":   +0.40,
        "conviction":   0.30,
        "horizon":     "short",
        "signal_type": "pump",
        "ambiguity":    0.55,
    },
    {
        "name":        "Ashes of the Phoenix",
        "headline":    "Resurfaced video — phoenix burned for ash, public outcry",
        "sentiment":   -0.70,
        "conviction":   0.85,
        "horizon":     "immediate",
        "signal_type": "scandal",
        "ambiguity":    0.10,
    },
    {
        "name":        "Volcanic Incense",
        "headline":    "Already rallying — Whiff Nostralico calls for people to follow",
        "sentiment":   +0.50,
        "conviction":   0.65,  # rally is real (data confirmed); hype-driven
        "horizon":     "short",
        "signal_type": "pump",
        "ambiguity":    0.30,
    },
    {
        "name":        "Sulfur Reactor",
        "headline":    "Added to Elemental Index 118 — index funds will rebalance",
        "sentiment":   +0.55,
        "conviction":   0.85,  # confirmed inclusion, forced flow
        "horizon":     "short",
        "signal_type": "index_flow",
        "ambiguity":    0.20,
    },
]
# Drop the placeholder dup
NEWS = [n for n in NEWS if n["headline"] != "(duplicate placeholder, ignored)"]
PRODUCTS = [n["name"] for n in NEWS]

# ==============================================================================
# 2. INSIDER PERSONAS
# ==============================================================================
# Each persona returns dict[product] -> signed percentage allocation.
# Positive = BUY, negative = SELL. Magnitude = % of budget.
# Sum of |allocations| should not exceed 100%.

def normalize(allocs, max_total=85.0):
    """Cap total absolute allocation, preserving relative weights."""
    total = sum(abs(v) for v in allocs.values())
    if total == 0: return allocs
    if total <= max_total:
        return allocs
    scale = max_total / total
    return {k: v * scale for k, v in allocs.items()}


def the_reactor(news):
    """Trades raw sentiment polarity. Ignores conviction/ambiguity — just
    polarity-weighted. Allocates proportional to |sentiment|."""
    raw = {n["name"]: n["sentiment"] * 25.0 for n in news}
    return normalize(raw, 80.0)


def the_insider(news):
    """Only trusts CONCRETE events (high conviction, immediate/tomorrow).
    Skips forecasts and pumps. Allocation = sentiment * conviction * 30."""
    out = {}
    for n in news:
        if n["signal_type"] in ("forecast", "pump"):
            continue
        if n["horizon"] not in ("immediate", "tomorrow", "short"):
            continue
        out[n["name"]] = n["sentiment"] * n["conviction"] * 30.0
    return normalize(out, 80.0)


def the_contrarian(news):
    """Fades pump signals (when crowd buys, contrarian sells/avoids). Trusts
    genuine scandals & regulatory shifts. Slightly fades 'forecast' too
    (assumes priced in)."""
    out = {}
    for n in news:
        if n["signal_type"] == "pump":
            # fade the pump direction
            out[n["name"]] = -n["sentiment"] * 15.0
        elif n["signal_type"] in ("scandal", "regulatory", "supply_shock"):
            out[n["name"]] = n["sentiment"] * n["conviction"] * 30.0
        elif n["signal_type"] == "forecast":
            out[n["name"]] = n["sentiment"] * 5.0  # priced-in discount
        else:
            out[n["name"]] = n["sentiment"] * 10.0
    return normalize(out, 80.0)


def the_quant(news):
    """Pure mathematical: sentiment * conviction * (1 - ambiguity) * horizon_weight."""
    horizon_w = {"immediate": 1.00, "tomorrow": 0.95, "short": 0.75,
                 "medium": 0.40, "long": 0.20}
    out = {}
    for n in news:
        score = (n["sentiment"] * n["conviction"]
                 * (1 - n["ambiguity"]) * horizon_w[n["horizon"]])
        out[n["name"]] = score * 35.0
    return normalize(out, 80.0)


def the_flow_chaser(news):
    """Loves index flow & supply shocks (mechanical, forced trades).
    Light on sentiment-only signals."""
    out = {}
    for n in news:
        st = n["signal_type"]
        if st == "index_flow":
            out[n["name"]] = n["sentiment"] * 20.0  # heavy on index
        elif st == "supply_shock":
            out[n["name"]] = n["sentiment"] * n["conviction"] * 20.0
        elif st == "scandal":
            out[n["name"]] = n["sentiment"] * n["conviction"] * 15.0
        elif st == "regulatory":
            out[n["name"]] = n["sentiment"] * 10.0
        else:
            out[n["name"]] = n["sentiment"] * 5.0
    return normalize(out, 80.0)


def the_momentum_trader(news):
    """Loves pumps and rallies. Buys what's already rising. Skeptical of
    pure forecasts (no momentum yet)."""
    out = {}
    for n in news:
        st = n["signal_type"]
        if st == "pump":
            out[n["name"]] = n["sentiment"] * n["conviction"] * 25.0
        elif st == "demand_event":
            out[n["name"]] = n["sentiment"] * 20.0
        elif st == "index_flow":
            out[n["name"]] = n["sentiment"] * 15.0
        elif st == "scandal":
            out[n["name"]] = n["sentiment"] * n["conviction"] * 15.0
        elif st == "supply_shock":
            out[n["name"]] = n["sentiment"] * 10.0
        else:
            out[n["name"]] = n["sentiment"] * 5.0
    return normalize(out, 80.0)


def the_risk_averse(news):
    """Only the highest-conviction, lowest-ambiguity trades. Most slots
    will be 0."""
    out = {}
    for n in news:
        score = n["sentiment"] * n["conviction"] * (1 - n["ambiguity"])
        if abs(score) >= 0.55:
            out[n["name"]] = score * 25.0
        else:
            out[n["name"]] = 0.0
    return normalize(out, 70.0)


PERSONAS = {
    "Reactor":         the_reactor,
    "Insider":         the_insider,
    "Contrarian":      the_contrarian,
    "Quant":           the_quant,
    "FlowChaser":      the_flow_chaser,
    "Momentum":        the_momentum_trader,
    "RiskAverse":      the_risk_averse,
}


# ==============================================================================
# 3. CONSENSUS AGGREGATION
# ==============================================================================

def consensus(persona_allocs, weights=None):
    """Weighted average across personas."""
    if weights is None:
        weights = {p: 1.0 for p in persona_allocs}
    total_w = sum(weights.values())
    out = defaultdict(float)
    for persona, allocs in persona_allocs.items():
        w = weights.get(persona, 0) / total_w
        for prod, alloc in allocs.items():
            out[prod] += w * alloc
    return normalize(dict(out), 85.0)


# ==============================================================================
# 4. MONTE-CARLO SIMULATION
# ==============================================================================

def true_move_distribution(n):
    """Sample a 'true' overnight % price move conditional on news features.

    The mean is sentiment*scale (large for high-conviction, immediate news,
    smaller for forecasts/ambiguous). The std reflects ambiguity + low
    conviction.
    """
    horizon_scale = {"immediate": 0.25, "tomorrow": 0.20,
                     "short": 0.15, "medium": 0.07, "long": 0.03}
    type_scale = {"scandal": 1.20, "supply_shock": 1.00, "regulatory": 0.80,
                  "demand_event": 0.90, "index_flow": 0.70, "pump": 0.65,
                  "forecast": 0.40}
    mean = (n["sentiment"]
            * n["conviction"]
            * horizon_scale[n["horizon"]]
            * type_scale.get(n["signal_type"], 0.5))
    std = (0.03                                 # base noise
           + 0.15 * n["ambiguity"]               # ambiguity widens spread
           + 0.05 * (1 - n["conviction"]))       # uncertainty widens spread
    return mean, std


def simulate_pnl(allocations, news_list, n_runs=10_000, seed=0):
    """For each MC run, draw a price move per product from its conditional
    distribution; compute net PnL after quadratic fee."""
    random.seed(seed)
    pnls = []
    by_product = defaultdict(list)
    by_news = {n["name"]: n for n in news_list}
    for _ in range(n_runs):
        total = 0.0
        for prod, p in allocations.items():
            if abs(p) < 1e-6: continue
            n = by_news[prod]
            mu, sigma = true_move_distribution(n)
            move = random.gauss(mu, sigma)            # fractional, e.g. 0.10 = +10%
            volume = (p / 100.0) * BUDGET             # signed; negative = short
            gross = volume * move                     # PnL from move (long: + when move>0)
            fee = ((abs(p) / 100.0) ** 2) * BUDGET
            net = gross - fee
            total += net
            by_product[prod].append(net)
        pnls.append(total)
    return pnls, by_product


def fmt_pct(x): return f"{x:+6.2f}%"
def fmt_money(x): return f"{x:>+10,.0f}"


# ==============================================================================
# 5. MAIN
# ==============================================================================

def main():
    print("=" * 90)
    print("ASHFLOW ALPHA — INSIDER TRADER ENSEMBLE  (budget = ${:,})".format(BUDGET))
    print("=" * 90)
    print()

    print("News inputs:")
    print(f"  {'PRODUCT':<22s} {'sent':>7s} {'conv':>6s} {'horizon':>10s} {'type':>14s} {'ambig':>7s}")
    for n in NEWS:
        print(f"  {n['name']:<22s} {n['sentiment']:>+7.2f} {n['conviction']:>6.2f} "
              f"{n['horizon']:>10s} {n['signal_type']:>14s} {n['ambiguity']:>7.2f}")
    print()

    # Run each persona
    persona_allocs = {name: fn(NEWS) for name, fn in PERSONAS.items()}

    print("Persona allocations (% of budget; + = BUY, - = SELL):")
    header = f"  {'PRODUCT':<22s}"
    for p in PERSONAS: header += f" {p:>11s}"
    header += f" {'CONSENSUS':>11s}"
    print(header)

    consensus_alloc = consensus(persona_allocs)

    for prod in PRODUCTS:
        row = f"  {prod:<22s}"
        for p in PERSONAS:
            v = persona_allocs[p].get(prod, 0.0)
            row += f" {v:>+11.2f}" if abs(v) >= 0.01 else f" {'·':>11s}"
        v = consensus_alloc.get(prod, 0.0)
        row += f" {v:>+11.2f}" if abs(v) >= 0.01 else f" {'·':>11s}"
        print(row)

    # Totals row
    row = f"  {'TOTAL |alloc|':<22s}"
    for p in PERSONAS:
        tot = sum(abs(v) for v in persona_allocs[p].values())
        row += f" {tot:>11.1f}"
    tot = sum(abs(v) for v in consensus_alloc.values())
    row += f" {tot:>11.1f}"
    print(row)
    print()

    # Monte Carlo per persona + consensus
    print("Monte-Carlo simulation (10,000 runs, gaussian price moves conditional on news features):")
    print(f"  {'STRATEGY':<14s} {'mean PnL':>12s} {'median':>12s} {'P10':>12s} {'P90':>12s} {'sharpe':>8s} {'P(>0)':>7s}")
    rows = []
    for name, allocs in persona_allocs.items():
        pnls, _ = simulate_pnl(allocs, NEWS, n_runs=10_000, seed=42)
        mean = statistics.mean(pnls)
        median = statistics.median(pnls)
        std = statistics.pstdev(pnls)
        p10 = sorted(pnls)[1000]; p90 = sorted(pnls)[9000]
        sharpe = mean / std if std > 0 else 0
        pwin = sum(1 for p in pnls if p > 0) / len(pnls)
        rows.append((name, mean, median, p10, p90, sharpe, pwin))
        print(f"  {name:<14s} {fmt_money(mean)} {fmt_money(median)} {fmt_money(p10)} {fmt_money(p90)} "
              f"{sharpe:>+8.3f} {pwin:>6.1%}")
    pnls, _ = simulate_pnl(consensus_alloc, NEWS, n_runs=10_000, seed=42)
    mean = statistics.mean(pnls); median = statistics.median(pnls)
    std = statistics.pstdev(pnls); p10 = sorted(pnls)[1000]; p90 = sorted(pnls)[9000]
    sharpe = mean / std if std > 0 else 0
    pwin = sum(1 for p in pnls if p > 0) / len(pnls)
    print(f"  {'CONSENSUS':<14s} {fmt_money(mean)} {fmt_money(median)} {fmt_money(p10)} {fmt_money(p90)} "
          f"{sharpe:>+8.3f} {pwin:>6.1%}")
    print()

    # Final recommended allocation: print consensus rounded to 1%
    print("Final recommended allocation (consensus, rounded):")
    print(f"  {'PRODUCT':<22s} {'DIR':>5s} {'%':>6s}")
    for prod in PRODUCTS:
        v = consensus_alloc.get(prod, 0)
        if abs(v) < 0.5:
            print(f"  {prod:<22s} {'-':>5s} {0:>6.0f}")
        else:
            direction = "BUY" if v > 0 else "SELL"
            print(f"  {prod:<22s} {direction:>5s} {abs(v):>6.0f}")
    total = sum(abs(round(v)) for v in consensus_alloc.values() if abs(v) >= 0.5)
    print(f"  {'TOTAL':<22s} {' ':>5s} {total:>6.0f}")


if __name__ == "__main__":
    main()
