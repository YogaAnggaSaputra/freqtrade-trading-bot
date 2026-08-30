"""Backtest manual: apa yang terjadi kalau SL 0.8% / 1.0% vs aktual 1.5%?"""
import sqlite3
from collections import defaultdict

c = sqlite3.connect("file:/freqtrade/user_data/tradesv3.sqlite?mode=ro", uri=True)
cur = c.cursor()

# Ambil semua trade yang SL_pct tersimpan (trade dengan custom_data)
cur.execute("""SELECT t.id, t.pair, t.is_short, t.open_rate, t.max_rate, t.min_rate,
  t.close_profit, t.close_profit_abs, t.exit_reason,
  CAST((SELECT cd_value FROM trade_custom_data d WHERE d.ft_trade_id=t.id AND d.cd_key='sl_pct') AS REAL) sl_pct
  FROM trades t WHERE t.is_open=0 AND t.id >= 95 ORDER BY t.id""")
rows = cur.fetchall()

# Untuk setiap trade, hitung:
# - SL aktual (open_rate * (1 - sl_pct)) untuk LONG
# - apakah max_rate pernah melampaui SL 0.8%, 1.0%, 1.2%?
# - kalau max_rate >= open_rate + threshold%, trade AKAN WIN (sebelum kena SL lama)
# - kalau max_rate < open_rate + threshold%, trade tetap LOSS (SL hit duluan)

results = {"sl_0.8%": {"wins_caught": 0, "losses_kept": 0, "losses_saved": 0.0, "wins_lost": 0.0},
           "sl_1.0%": {"wins_caught": 0, "losses_kept": 0, "losses_saved": 0.0, "wins_lost": 0.0},
           "sl_1.2%": {"wins_caught": 0, "losses_kept": 0, "losses_saved": 0.0, "wins_lost": 0.0}}

pair_outcome = defaultdict(lambda: {"n": 0, "loss": 0, "win": 0, "loss_sum": 0.0})

for tid, pair, short, o, maxr, minr, pf, pa, ex, sl_pct in rows:
    if not o or not maxr or not minr or not sl_pct: continue
    if short: continue  # fokus LONG dulu (mayoritas loss kita)
    pair_outcome[pair]["n"] += 1
    if pf < 0:
        pair_outcome[pair]["loss"] += 1
        pair_outcome[pair]["loss_sum"] += pa
    elif pf > 0:
        pair_outcome[pair]["win"] += 1

    # Untuk threshold SL, cek apakah max > threshold profit
    for th_name, th in [("sl_0.8%", 0.008), ("sl_1.0%", 0.010), ("sl_1.2%", 0.012)]:
        # kalau max_rate >= open * (1+th), trade BERHASIL escape SL → menang th% minimal
        if maxr >= o * (1 + th):
            results[th_name]["wins_caught"] += 1
            results[th_name]["wins_lost"] += 0  # hypothetical gain
        else:
            # kalau max gak pernah naik th%, trade akan kena SL th% = loss -th
            results[th_name]["losses_kept"] += 1
            results[th_name]["losses_saved"] += (sl_pct - th) * pa  # (saved loss = old loss - new loss)

print("=" * 80)
print("SIMULASI: ganti SL aktual ke 0.8% / 1.0% / 1.2%")
print("Logic: kalau max_rate pernah >= threshold, trade WIN; kalau tidak, LOSS")
print("=" * 80)

for th_name, r in results.items():
    print(f"\n{th_name}:")
    print(f"  Trade yang BERHASIL escape (max pernah >= threshold): {r['wins_caught']}")
    print(f"  Trade yang tetap KENA SL baru (max tidak pernah sampai threshold): {r['losses_kept']}")

print("\n" + "=" * 80)
print("PAIR-PAIR PALING RUGI (kandidat filter)")
print("=" * 80)
print(f"{'Pair':<16} {'N':>3} {'Loss':>4} {'Win':>3} {'LossSum':>9} {'Avg loss':>9}")
loss_pairs = []
for pair, d in sorted(pair_outcome.items(), key=lambda x: x[1]['loss_sum']):
    if d['loss'] > d['win']:
        loss_pairs.append(pair)
        avg = d['loss_sum'] / d['loss'] if d['loss'] else 0
        print(f"{pair:<16} {d['n']:>3} {d['loss']:>4} {d['win']:>3} {d['loss_sum']:>+8.4f} {avg*100:>+7.2f}%")
print(f"\nTotal pair dengan loss > win: {len(loss_pairs)}")
print("Pair ini kandidat filter confluence lebih tinggi:")
print(", ".join(loss_pairs[:30]))

print("\n" + "=" * 80)
print("BANDINGKAN: aktual loss SL hit (avg)")
print("=" * 80)
cur.execute("""SELECT AVG(close_profit), COUNT(*) FROM trades
  WHERE is_open=0 AND exit_reason='stoploss_on_exchange'""")
avg_loss, n_sl = cur.fetchone()
print(f"SL hit aktual: N={n_sl}, avg loss = {avg_loss*100:+.3f}%")
print(f"\nKalau SL 0.8%: avg loss potensial ~ -0.8% per trade (vs aktual -1.11%)")
print(f"  Saving per trade: {(0.0111 - 0.008)*100:.2f}%")
print(f"  Total saving kalau 57 trade SL: {57 * (0.0111 - 0.008):.4f} P&L unit")
