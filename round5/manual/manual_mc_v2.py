"""Monte Carlo v2 — compares 3 allocations under 2 return models.

Models:
  A. Sentiment model (notebook): returns follow polarity of sentiment tags.
     Lava Cake = -60%, Volcanic Incense = +5%, etc.
  B. Crowding/contrarian model (mirofish report): over-crowded shorts squeeze,
     over-crowded longs capitulate. Lava Cake = +10% (squeeze), Volcanic = -10%
     (capitulation), Phoenix Ash = +6% (deep readers buy), Pyroflex = +6%
     (over-sold rebound).

Each allocation evaluated under both models with structural noise.
"""
import random, statistics
from collections import defaultdict

BUDGET = 1_000_000

# ==============================================================================
# Two return models
# ==============================================================================
# A. Sentiment model — straight from the notebook tags
PRED_R_SENTIMENT = {
    'ObsidianCutlery':   -0.05,
    'PyroflexCells':     -0.30,
    'ThermaliteCore':    +0.35,
    'LavaCake':          -0.60,
    'MagmaInk':          +0.05,
    'ScoriaPaste':       +0.05,
    'AshesOfThePhoenix': +0.00,
    'VolcanicIncense':   +0.05,
    'SulfurReactor':     +0.25,
}

# B. Crowding model — from the mirofish report's predicted crowd behavior
PRED_R_CROWDING = {
    'ObsidianCutlery':   +0.00,    # ambiguous, they say SKIP
    'PyroflexCells':     +0.06,    # over-sold rebound (cognitive confusion)
    'ThermaliteCore':    +0.20,    # under-reacted, smaller than sentiment
    'LavaCake':          +0.10,    # SHORT-crowded squeeze
    'MagmaInk':          +0.00,    # priced in, SKIP
    'ScoriaPaste':       -0.05,    # KOL split, fade
    'AshesOfThePhoenix': +0.06,    # brand-crack realization
    'VolcanicIncense':   -0.10,    # liquidity illusion, capitulation
    'SulfurReactor':     +0.05,    # front-running exhausted
}

PRODUCTS = list(PRED_R_SENTIMENT.keys())

# Sigma scales with magnitude of expected move
def sigma_for(r):
    return 0.05 + 0.30 * abs(r)


# ==============================================================================
# Three candidate allocations
# ==============================================================================
def _opt_p(r):
    """Optimal integer p given quadratic fee: p* = 50r, then round."""
    return int(round(50 * r))


def bayes_avg_alloc():
    """p = 50 * (r_sent + r_crowd) / 2 — assumes 50/50 model uncertainty."""
    out = {}
    for p in PRODUCTS:
        avg_r = (PRED_R_SENTIMENT[p] + PRED_R_CROWDING[p]) / 2
        out[p] = _opt_p(avg_r)
    return out


def robust_min_r_alloc():
    """Only trade where both models agree on sign; size to MIN |r| with that sign."""
    out = {}
    for p in PRODUCTS:
        rs, rc = PRED_R_SENTIMENT[p], PRED_R_CROWDING[p]
        # Both nonzero same sign → take min magnitude
        if rs > 0 and rc > 0:
            out[p] = _opt_p(min(rs, rc))
        elif rs < 0 and rc < 0:
            out[p] = _opt_p(max(rs, rc))  # closest-to-zero negative
        else:
            out[p] = 0
    return out


ALLOCS = {
    'Notebook (sent)': {
        'ObsidianCutlery': -3,  'PyroflexCells': -15, 'ThermaliteCore': +18,
        'LavaCake': -30, 'MagmaInk': +3, 'ScoriaPaste': +3,
        'AshesOfThePhoenix': 0, 'VolcanicIncense': +3, 'SulfurReactor': +13,
    },
    'Mirofish': {
        'ObsidianCutlery': 0,   'PyroflexCells': +10, 'ThermaliteCore': +18,
        'LavaCake': +15, 'MagmaInk': 0, 'ScoriaPaste': -12,
        'AshesOfThePhoenix': +12, 'VolcanicIncense': -20, 'SulfurReactor': +8,
    },
    'Hybrid':           {  # keep notebook's high-conviction Lava SELL but adopt mirofish's flips on Phoenix/Pyroflex/Volcanic
        'ObsidianCutlery': -2,  'PyroflexCells': +6,  'ThermaliteCore': +18,
        'LavaCake': -25, 'MagmaInk': +2, 'ScoriaPaste': -3,
        'AshesOfThePhoenix': +6, 'VolcanicIncense': -8, 'SulfurReactor': +13,
    },
    'Bayes avg (50/50)': bayes_avg_alloc(),
    'Robust min-r':      robust_min_r_alloc(),
}


def pnl_for_run(alloc, r_dict):
    total = 0.0
    per_prod = {}
    for prod, p in alloc.items():
        if p == 0:
            per_prod[prod] = 0.0; continue
        r = r_dict[prod]
        gross = 10_000 * r * p
        fee = 100 * p * p
        per_prod[prod] = gross - fee
        total += gross - fee
    return total, per_prod


def simulate(alloc, pred_r, n_runs=20_000, seed=42):
    random.seed(seed)
    pnls = []; per_prod_all = defaultdict(list)
    for _ in range(n_runs):
        r_dict = {p: random.gauss(pred_r[p], sigma_for(pred_r[p])) for p in pred_r}
        total, per_prod = pnl_for_run(alloc, r_dict)
        pnls.append(total)
        for k, v in per_prod.items():
            per_prod_all[k].append(v)
    return pnls, per_prod_all


def stats(pnls):
    s = sorted(pnls); n = len(s)
    mu = statistics.mean(pnls); sd = statistics.pstdev(pnls)
    return {
        'mean': mu, 'median': s[n // 2], 'std': sd,
        'p10': s[n // 10], 'p90': s[9 * n // 10],
        'p_win': sum(1 for x in pnls if x > 0) / n,
        'sharpe': mu / (sd or 1),
    }


def main():
    print("=" * 100)
    print("Round 5 manual — Monte Carlo v2 (allocation × return-model matrix, 20k runs each)")
    print("=" * 100)
    print()

    # Show the allocations
    print("Allocations (% of budget):")
    print(f"  {'PRODUCT':<22s} " + ' '.join(f'{n:>16s}' for n in ALLOCS))
    for prod in PRODUCTS:
        row = f"  {prod:<22s} " + ' '.join(f'{ALLOCS[n][prod]:>+16d}' for n in ALLOCS)
        print(row)
    print(f"  {'L1 used':<22s} "
          + ' '.join(f"{sum(abs(v) for v in ALLOCS[n].values()):>16d}" for n in ALLOCS))
    print()

    # Show the return assumptions
    print(f"Return models:")
    print(f"  {'PRODUCT':<22s} {'Sentiment':>12s} {'Crowding':>12s}")
    for p in PRODUCTS:
        print(f"  {p:<22s} {PRED_R_SENTIMENT[p]:>+12.2%} {PRED_R_CROWDING[p]:>+12.2%}")
    print()

    # Run each (allocation × model) combo
    results = {}
    for alloc_name, alloc in ALLOCS.items():
        for model_name, pred_r in [('Sentiment', PRED_R_SENTIMENT), ('Crowding', PRED_R_CROWDING)]:
            pnls, _ = simulate(alloc, pred_r)
            results[(alloc_name, model_name)] = stats(pnls)

    # Print the matrix
    print("PnL matrix (mean of MC):")
    print(f"  {'ALLOC':<18s}  {'sentiment-mean':>18s} {'crowding-mean':>18s} "
          f"{'sent P(>0)':>12s} {'crowd P(>0)':>12s} {'avg':>12s}")
    print("  " + "-" * 90)
    for alloc_name in ALLOCS:
        s = results[(alloc_name, 'Sentiment')]
        c = results[(alloc_name, 'Crowding')]
        avg = (s['mean'] + c['mean']) / 2
        print(f"  {alloc_name:<18s}  {s['mean']:>+18,.0f} {c['mean']:>+18,.0f} "
              f"{s['p_win']:>12.1%} {c['p_win']:>12.1%} {avg:>+12,.0f}")
    print()

    print("Tail risk — P10 (worst 10%) per scenario:")
    print(f"  {'ALLOC':<18s}  {'sent P10':>14s} {'crowd P10':>14s} {'sent P90':>14s} {'crowd P90':>14s}")
    for alloc_name in ALLOCS:
        s = results[(alloc_name, 'Sentiment')]
        c = results[(alloc_name, 'Crowding')]
        print(f"  {alloc_name:<18s}  {s['p10']:>+14,.0f} {c['p10']:>+14,.0f} "
              f"{s['p90']:>+14,.0f} {c['p90']:>+14,.0f}")
    print()

    print("Sharpe per (allocation × model):")
    print(f"  {'ALLOC':<18s}  {'sentiment':>12s} {'crowding':>12s}")
    for alloc_name in ALLOCS:
        s = results[(alloc_name, 'Sentiment')]
        c = results[(alloc_name, 'Crowding')]
        print(f"  {alloc_name:<18s}  {s['sharpe']:>+12.3f} {c['sharpe']:>+12.3f}")
    print()

    # Recommend by robustness — which allocation has the smallest gap between best/worst model?
    print("Robustness — min(sentiment_mean, crowding_mean) (i.e. worst-case assumption):")
    for alloc_name in ALLOCS:
        s = results[(alloc_name, 'Sentiment')]['mean']
        c = results[(alloc_name, 'Crowding')]['mean']
        worst = min(s, c)
        print(f"  {alloc_name:<18s}  worst-case mean = {worst:>+10,.0f}  (sent={s:>+10,.0f}, crowd={c:>+10,.0f})")


if __name__ == "__main__":
    main()
