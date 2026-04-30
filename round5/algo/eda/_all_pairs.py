"""All-pairs scan: every (X, Y) pair across the 50 products. Flag pairs that
look pairs-tradeable: high return correlation AND stationary spread.

A pairs trade needs:
  1. High return correlation (>0.3, ideally >0.5) — they actually move together
  2. Stationary spread — std(X-Y)/range(X-Y) low, OR AR(1) half-life short
  3. Spread is wide enough to capture a few ticks per round-trip

Filter:
  - return correlation > 0.3
  - AR(1) coefficient on spread (close to 0 = mean-reverting, close to 1 = random walk)
"""
import csv
from collections import defaultdict
from pathlib import Path


def load_mids():
    mids = defaultdict(dict)
    products = set()
    for d in [2, 3, 4]:
        with open(DATA / f"prices_round_5_day_{d}.csv") as f:
            for r in csv.DictReader(f, delimiter=";"):
                try:
                    mp = float(r['mid_price'])
                    mids[(d, int(r['timestamp']))][r['product']] = mp
                    products.add(r['product'])
                except: pass
    return mids, sorted(products)


def pearson(x, y):
    n = len(x)
    if n < 2: return 0
    mx = sum(x) / n; my = sum(y) / n
    sx = sum((xi - mx) ** 2 for xi in x) ** 0.5
    sy = sum((yi - my) ** 2 for yi in y) ** 0.5
    if sx == 0 or sy == 0: return 0
    return sum((x[i] - mx) * (y[i] - my) for i in range(n)) / (sx * sy)


def ar1_coef(s):
    """AR(1) coefficient. 1 = random walk; 0 = white noise; close to 0 = mean-revert fast."""
    if len(s) < 2: return 1.0
    return pearson(s[:-1], s[1:])


def main():
    mids, products = load_mids()
    keys = sorted(mids.keys())
    print(f"products: {len(products)}, ticks: {len(keys)}")

    # Build per-product series
    series = {p: {} for p in products}
    for k in keys:
        for p, m in mids[k].items():
            series[p][k] = m

    # Pairwise scan
    print("\nScanning all pairs (this takes ~3 min)...")
    candidates = []  # (X, Y, n, lvl_r, ret_r, ar1, spread_mean, spread_std)
    for i, X in enumerate(products):
        for Y in products[i+1:]:
            xkeys = set(series[X].keys()); ykeys = set(series[Y].keys())
            common = sorted(xkeys & ykeys)
            if len(common) < 1000:
                continue

            xv = [series[X][k] for k in common]
            yv = [series[Y][k] for k in common]

            # Tick returns
            xr, yr = [], []
            for j in range(len(common)-1):
                k1, k2 = common[j], common[j+1]
                if k1[0] == k2[0] and k2[1] - k1[1] == 100:
                    xr.append(xv[j+1] - xv[j])
                    yr.append(yv[j+1] - yv[j])

            ret_r = pearson(xr, yr) if xr else 0

            # Only investigate pairs above the noise floor
            if abs(ret_r) < 0.20:
                continue

            lvl_r = pearson(xv, yv)
            spread = [xv[j] - yv[j] for j in range(len(common))]
            sm = sum(spread) / len(spread)
            ss = (sum((x - sm) ** 2 for x in spread) / len(spread)) ** 0.5
            ar1 = ar1_coef(spread)
            candidates.append((X, Y, len(common), lvl_r, ret_r, ar1, sm, ss))
        if (i+1) % 10 == 0:
            print(f"  {i+1}/{len(products)} done, candidates so far: {len(candidates)}")

    # Sort by return correlation (descending magnitude)
    candidates.sort(key=lambda x: -abs(x[4]))

    print(f"\n=== PAIRS WITH |ret_r| > 0.20 (n={len(candidates)}) ===")
    print(f"{'X':<35s} {'Y':<35s} {'n':>6s} {'lvl_r':>7s} {'ret_r':>7s} {'AR1':>6s} {'sp_std':>8s}")
    for X, Y, n, lvl, ret, ar1, sm, ss in candidates[:50]:
        print(f"{X:<35s} {Y:<35s} {n:>6d} {lvl:>7.3f} {ret:>7.3f} {ar1:>6.3f} {ss:>8.0f}")

    # Of those, the actually pairs-tradeable: ret_r > 0.3 AND AR1 < 0.95 (spread reverts)
    print(f"\n=== PAIRS-TRADEABLE: |ret_r|>0.30 AND |AR1|<0.95 ===")
    print(f"{'X':<35s} {'Y':<35s} {'n':>6s} {'ret_r':>7s} {'AR1':>6s} {'sp_mean':>9s} {'sp_std':>8s}")
    tradeable = [c for c in candidates if abs(c[4]) > 0.30 and abs(c[5]) < 0.95]
    for X, Y, n, lvl, ret, ar1, sm, ss in tradeable:
        print(f"{X:<35s} {Y:<35s} {n:>6d} {ret:>7.3f} {ar1:>6.3f} {sm:>9.1f} {ss:>8.0f}")
    if not tradeable:
        print("  none.")


if __name__ == "__main__":
    main()
