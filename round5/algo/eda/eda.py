"""Round 5 EDA — survey 50 products across 10 families, 3 days."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────
def load_prices():
    frames = []
    for day in (2, 3, 4):
        df = pd.read_csv(ROOT / f'prices_round_5_day_{day}.csv', sep=';')
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

def load_trades():
    frames = []
    for day in (2, 3, 4):
        df = pd.read_csv(ROOT / f'trades_round_5_day_{day}.csv', sep=';')
        df['day'] = day
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

prices = load_prices()
trades = load_trades()
print(f'prices: {len(prices):,} rows, {prices["product"].nunique()} products')
print(f'trades: {len(trades):,} rows')
print(f'days:   {sorted(prices["day"].unique())}')

# ─────────────────────────────────────────────────────────────
# Helper: assign each product to a family
# ─────────────────────────────────────────────────────────────
def family_of(product):
    # Family = everything before the last underscore-separated suffix.
    # Special-case PEBBLES_X (1 token suffix) and PANEL_X (1 token suffix)
    parts = product.split('_')
    # Family name is the prefix before the variant; conservatively, return
    # the first 1-2 tokens depending on the family.
    families = ['GALAXY_SOUNDS', 'MICROCHIP', 'OXYGEN_SHAKE', 'PANEL',
                'PEBBLES', 'ROBOT', 'SLEEP_POD', 'SNACKPACK', 'TRANSLATOR',
                'UV_VISOR']
    for f in families:
        if product.startswith(f + '_') or product == f:
            return f
    return parts[0]

prices['family']  = prices['product'].apply(family_of)
trades['family']  = trades['symbol'].apply(family_of)
prices['variant'] = prices.apply(
    lambda r: r['product'][len(r['family']) + 1:], axis=1)

# ─────────────────────────────────────────────────────────────
# Per-product summary stats (across all days)
# ─────────────────────────────────────────────────────────────
agg = prices.groupby('product').agg(
    family=('family', 'first'),
    variant=('variant', 'first'),
    mid_mean=('mid_price', 'mean'),
    mid_std=('mid_price', 'std'),
    mid_min=('mid_price', 'min'),
    mid_max=('mid_price', 'max'),
    spread_mean=('mid_price', lambda x: 0),  # placeholder
    n_obs=('mid_price', 'size'),
).reset_index()

# Compute mean spread directly
prices['spread'] = prices['ask_price_1'] - prices['bid_price_1']
spread_agg = prices.groupby('product')['spread'].mean().rename('spread_mean')
agg = agg.drop(columns='spread_mean').merge(spread_agg, on='product')

agg['mid_range'] = agg['mid_max'] - agg['mid_min']
agg = agg.sort_values(['family', 'variant']).reset_index(drop=True)
print('\n===== Per-product summary =====')
print(agg.to_string(index=False))
agg.to_csv(OUT / 'product_summary.csv', index=False)

# ─────────────────────────────────────────────────────────────
# Per-family summary: does one variant stand out?
# ─────────────────────────────────────────────────────────────
print('\n===== Per-family stats =====')
fam_summary = []
for fam, g in agg.groupby('family'):
    line = f'\n--- {fam} ({len(g)} variants) ---'
    print(line)
    print(g[['variant', 'mid_mean', 'mid_std', 'mid_min',
             'mid_max', 'mid_range', 'spread_mean']].to_string(index=False))
    # Identify the high/low/most-volatile variants
    fam_summary.append({
        'family': fam,
        'highest_mean': g.loc[g['mid_mean'].idxmax(), 'variant'],
        'lowest_mean':  g.loc[g['mid_mean'].idxmin(), 'variant'],
        'most_vol':     g.loc[g['mid_std'].idxmax(),  'variant'],
        'least_vol':    g.loc[g['mid_std'].idxmin(),  'variant'],
        'mean_range':   g['mid_mean'].max() - g['mid_mean'].min(),
        'avg_std':      g['mid_std'].mean(),
    })
fam_df = pd.DataFrame(fam_summary)
print('\n===== Family-level overview =====')
print(fam_df.to_string(index=False))
fam_df.to_csv(OUT / 'family_summary.csv', index=False)

# ─────────────────────────────────────────────────────────────
# Mid-price time series by family — one figure per family
# ─────────────────────────────────────────────────────────────
prices = prices.sort_values(['day', 'timestamp']).reset_index(drop=True)
prices['t_global'] = prices['day'] * 1_000_000 + prices['timestamp']

for fam in agg['family'].unique():
    fam_prices = prices[prices['family'] == fam]
    variants = sorted(fam_prices['variant'].unique())
    fig, ax = plt.subplots(figsize=(13, 5))
    for v in variants:
        sub = fam_prices[fam_prices['variant'] == v]
        ax.plot(sub['t_global'], sub['mid_price'], label=v, lw=0.9)
    ax.set_title(f'{fam} — mid price (days 2,3,4)')
    ax.set_xlabel('global timestamp (day*1e6 + ts)')
    ax.set_ylabel('mid price')
    ax.legend(fontsize=8, loc='best')
    fig.tight_layout()
    fig.savefig(OUT / f'mid_{fam}.png', dpi=110)
    plt.close(fig)
print(f'\nSaved per-family time-series plots to {OUT}')

# ─────────────────────────────────────────────────────────────
# Cross-correlation matrix — within each family
# ─────────────────────────────────────────────────────────────
for fam in agg['family'].unique():
    fam_prices = prices[prices['family'] == fam]
    pivot = fam_prices.pivot_table(index='t_global', columns='variant',
                                    values='mid_price', aggfunc='first')
    corr = pivot.corr()
    print(f'\n--- {fam} corr matrix ---')
    print(corr.round(3).to_string())
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(corr.index, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f'{corr.iat[i,j]:.2f}',
                    ha='center', va='center', fontsize=7,
                    color='white' if abs(corr.iat[i,j]) > 0.5 else 'black')
    ax.set_title(f'{fam} — variant correlation')
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(OUT / f'corr_{fam}.png', dpi=110)
    plt.close(fig)

# ─────────────────────────────────────────────────────────────
# Volume / activity by product
# ─────────────────────────────────────────────────────────────
trade_vol = (trades.groupby(['symbol'])
             .agg(n_trades=('quantity', 'size'),
                  total_qty=('quantity', 'sum'),
                  avg_qty=('quantity', 'mean'),
                  family=('family', 'first'))
             .reset_index()
             .sort_values('total_qty', ascending=False))
print('\n===== Trade activity (top 30 by total qty) =====')
print(trade_vol.head(30).to_string(index=False))
trade_vol.to_csv(OUT / 'trade_activity.csv', index=False)

# Check if any trades have non-empty buyer/seller (counterparty reveal)
has_party = trades[(trades['buyer'].notna() & (trades['buyer'] != '')) |
                   (trades['seller'].notna() & (trades['seller'] != ''))]
print(f'\nTrades with buyer or seller filled: {len(has_party):,} / {len(trades):,}')
if len(has_party):
    print(has_party.head(10).to_string(index=False))
    parties = pd.concat([has_party['buyer'], has_party['seller']]).dropna()
    parties = parties[parties != '']
    print('\nUnique counterparties:', parties.unique())

print(f'\nDone. Outputs in: {OUT}')
