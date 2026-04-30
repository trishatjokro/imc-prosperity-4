"""Monte Carlo for the Round 5 manual challenge.

Tests the notebook's framework + the revised allocation under a distribution
of plausible actual returns. Uncertainty is structural (we don't know the
true r) — not pulled from N=1 last year.

For each MC run:
  - Draw an actual return r_i for each product from N(predicted_r_i, sigma_i)
  - sigma_i scales with the predicted magnitude: bigger predicted move = more
    uncertainty. This is a structural prior on volatility, not last-year specific.
  - Compute PnL for each candidate allocation:
      pnl_i = 10000 · r_i · p_i − 100 · p_i²
  - Sum across products → portfolio PnL.

Compare four candidate allocations:
  A. Notebook's allocation (88%)
  B. Revised (84%, weak signals trimmed to 2%)
  C. Conservative (skip all weak signals entirely, ~76%)
  D. Aggressive (boost Lava Cake to 38% based on N=1 — included for contrast)

Output: PnL distribution per allocation, P(profit), P10, median, P90, sharpe.
"""
import random, statistics
from collections import defaultdict

BUDGET = 1_000_000

# Sentiment → predicted return. Same scale as notebook.
returns_map = {
    '=':    0.00,
    '+':   +0.05, '++':  +0.15, '+++':  +0.25, '++++': +0.35,
    '-':   -0.05, '--':  -0.10, '---':  -0.30, '----': -0.60,
}

# Per-product predicted return (from notebook's tagging)
PRED_R = {
    'ObsidianCutlery':   returns_map['-'],     # -0.05
    'PyroflexCells':     returns_map['---'],   # -0.30
    'ThermaliteCore':    returns_map['++++'],  # +0.35
    'LavaCake':          returns_map['----'],  # -0.60
    'MagmaInk':          returns_map['+'],     # +0.05
    'ScoriaPaste':       returns_map['+'],     # +0.05
    'AshesOfThePhoenix': returns_map['='],     # 0.00
    'VolcanicIncense':   returns_map['+'],     # +0.05
    'SulfurReactor':     returns_map['+++'],   # +0.25
}

# Structural uncertainty: bigger expected move → bigger absolute uncertainty.
# sigma = base + slope * |predicted_r|. Calibrated to be wide enough that
# weak signals can land near 0 and strong signals can be ±15-20% off.
def sigma_for(r):
    return 0.04 + 0.30 * abs(r)

# Candidate allocations (% of budget; sign = direction)
ALLOCS = {
    'A. Notebook':       {'ObsidianCutlery': -3,  'PyroflexCells': -15, 'ThermaliteCore': +18,
                          'LavaCake': -30, 'MagmaInk': +3, 'ScoriaPaste': +3,
                          'AshesOfThePhoenix': 0, 'VolcanicIncense': +3, 'SulfurReactor': +13},
    'B. Revised':        {'ObsidianCutlery': -2,  'PyroflexCells': -15, 'ThermaliteCore': +18,
                          'LavaCake': -30, 'MagmaInk': +2, 'ScoriaPaste': +2,
                          'AshesOfThePhoenix': 0, 'VolcanicIncense': +2, 'SulfurReactor': +13},
    'C. Conservative':   {'ObsidianCutlery':  0,  'PyroflexCells': -15, 'ThermaliteCore': +18,
                          'LavaCake': -30, 'MagmaInk':  0, 'ScoriaPaste':  0,
                          'AshesOfThePhoenix': 0, 'VolcanicIncense':  0, 'SulfurReactor': +13},
    'D. Aggressive':     {'ObsidianCutlery': -3,  'PyroflexCells': -18, 'ThermaliteCore': +18,
                          'LavaCake': -38, 'MagmaInk': +3, 'ScoriaPaste': +3,
                          'AshesOfThePhoenix': 0, 'VolcanicIncense': +3, 'SulfurReactor': +13},
}


def pnl_for_run(alloc, r_dict):
    total = 0.0
    per_prod = {}
    for prod, p in alloc.items():
        if p == 0:
            per_prod[prod] = 0.0
            continue
        r = r_dict[prod]
        gross = 10_000 * r * p
        fee = 100 * p * p
        net = gross - fee
        per_prod[prod] = net
        total += net
    return total, per_prod


def simulate(alloc, n_runs=20_000, seed=42):
    random.seed(seed)
    pnls = []
    per_prod_pnls = defaultdict(list)
    for _ in range(n_runs):
        # Draw actual returns this run
        r_dict = {prod: random.gauss(PRED_R[prod], sigma_for(PRED_R[prod]))
                  for prod in PRED_R}
        total, per_prod = pnl_for_run(alloc, r_dict)
        pnls.append(total)
        for k, v in per_prod.items():
            per_prod_pnls[k].append(v)
    return pnls, per_prod_pnls


def stats(pnls):
    pnls_sorted = sorted(pnls)
    n = len(pnls_sorted)
    return {
        'mean':   statistics.mean(pnls),
        'median': pnls_sorted[n // 2],
        'std':    statistics.pstdev(pnls),
        'p10':    pnls_sorted[n // 10],
        'p90':    pnls_sorted[9 * n // 10],
        'p_win':  sum(1 for p in pnls if p > 0) / n,
        'sharpe': statistics.mean(pnls) / (statistics.pstdev(pnls) or 1),
    }


def main():
    print("=" * 90)
    print("Round 5 manual — Monte Carlo (20,000 runs, structural uncertainty around predictions)")
    print("=" * 90)
    print()
    print("Predicted returns + sampling sigma:")
    print(f"  {'PRODUCT':<20s} {'pred r':>8s} {'sigma':>8s}")
    for prod, r in PRED_R.items():
        s = sigma_for(r)
        print(f"  {prod:<20s} {r:>+8.2%} {s:>8.3f}")
    print()
    # Show alloc summary
    print("Allocations (% of budget):")
    print(f"  {'PRODUCT':<20s} " + ' '.join(f'{name[:14]:>14s}' for name in ALLOCS))
    for prod in PRED_R:
        row = f"  {prod:<20s} "
        for name in ALLOCS:
            row += f"{ALLOCS[name][prod]:>+14d}"
        print(row)
    print(f"  {'TOTAL |%|':<20s} " + ' '.join(
        f'{sum(abs(v) for v in ALLOCS[name].values()):>14d}' for name in ALLOCS))
    print()

    # Simulate each
    print("MC results:")
    print(f"  {'STRATEGY':<18s} {'mean':>10s} {'median':>10s} {'std':>10s} {'P10':>10s} "
          f"{'P90':>10s} {'P(>0)':>7s} {'sharpe':>8s}")
    print("  " + "-" * 86)
    all_pnls = {}
    all_per_prod = {}
    for name, alloc in ALLOCS.items():
        pnls, per_prod = simulate(alloc, n_runs=20_000, seed=42)
        s = stats(pnls)
        all_pnls[name] = pnls
        all_per_prod[name] = per_prod
        print(f"  {name:<18s} {s['mean']:>+10,.0f} {s['median']:>+10,.0f} {s['std']:>10,.0f} "
              f"{s['p10']:>+10,.0f} {s['p90']:>+10,.0f} {s['p_win']:>6.1%} {s['sharpe']:>+8.3f}")
    print()

    # Per-product mean PnL contribution under best allocation (B)
    print("Per-product mean PnL under each allocation:")
    print(f"  {'PRODUCT':<20s} " + ' '.join(f'{name[:14]:>14s}' for name in ALLOCS))
    for prod in PRED_R:
        row = f"  {prod:<20s} "
        for name in ALLOCS:
            mean_p = statistics.mean(all_per_prod[name][prod])
            row += f"{mean_p:>+14,.0f}"
        print(row)
    print()

    # Risk: tail outcomes
    print("Tail-risk view (worst 5% of MC runs):")
    for name in ALLOCS:
        pnls = sorted(all_pnls[name])
        worst5 = pnls[: len(pnls) // 20]
        avg_worst = statistics.mean(worst5)
        print(f"  {name:<18s}  avg of worst 5%: {avg_worst:>+10,.0f}")
    print()

    # Recommend
    best = max(ALLOCS, key=lambda k: stats(all_pnls[k])['mean'])
    print(f"Highest expected: {best}")
    best_sharpe = max(ALLOCS, key=lambda k: stats(all_pnls[k])['sharpe'])
    print(f"Highest Sharpe:   {best_sharpe}")


if __name__ == "__main__":
    main()
