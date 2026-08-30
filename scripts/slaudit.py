import sqlite3
from collections import defaultdict

c = sqlite3.connect("file:/freqtrade/user_data/tradesv3.sqlite?mode=ro", uri=True)
cur = c.cursor()

# Ambil semua trade SL
cur.execute("""SELECT t.id, t.pair, t.is_short, substr(t.open_date,12,2), t.close_profit,
  (SELECT cd_value FROM trade_custom_data d WHERE d.ft_trade_id=t.id AND d.cd_key='entry_imbalance')
  FROM trades t WHERE t.is_open=0 AND t.exit_reason='stoploss_on_exchange' ORDER BY t.id""")
rows = cur.fetchall()
print(f"Total SL hit: {len(rows)}")

# Per jam
by_hour = defaultdict(lambda: [0, 0.0])
for r in rows:
    by_hour[int(r[3])][0] += 1
    by_hour[int(r[3])][1] += r[4] or 0

print("\n=== 1. BREAKDOWN PER JAM UTC ===")
print(f"{'Jam':>4} {'N':>4} {'AvgP%':>8}")
for h in sorted(by_hour):
    n, tp = by_hour[h]
    print(f"{h:02d}    {n:>4} {tp/n*100:>+7.2f}%")

# Per pair
by_pair = defaultdict(lambda: [0, 0.0])
for r in rows:
    by_pair[r[1]][0] += 1
    by_pair[r[1]][1] += r[4] or 0

print("\n=== 2. BREAKDOWN PER PAIR (top 15) ===")
print(f"{'Pair':<16} {'N':>4} {'AvgP%':>8}")
for p, (n, tp) in sorted(by_pair.items(), key=lambda x: -x[1][0])[:15]:
    print(f"{p:<16} {n:>4} {tp/n*100:>+7.2f}%")

# Imbalance
imb_bins = {"neg": [0, 0.0], "near": [0, 0.0], "pos": [0, 0.0]}
for r in rows:
    imb = r[5]
    if imb is None: continue
    imb = float(imb)
    if imb < -0.05: b = "neg"
    elif imb > 0.05: b = "pos"
    else: b = "near"
    imb_bins[b][0] += 1
    imb_bins[b][1] += r[4] or 0

print("\n=== 3. I(t) DISTRIBUTION (SL TRADES) ===")
print(f"{'Bucket':<8} {'N':>4} {'AvgP%':>8}")
for b in ["neg", "near", "pos"]:
    n, tp = imb_bins[b]
    if n: print(f"{b:<8} {n:>4} {tp/n*100:>+7.2f}%")
