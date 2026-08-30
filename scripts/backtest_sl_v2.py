"""Backtest manual v2: SL 0.8/1.0/1.2% terhadap SEMUA trade (bukan cuma id>=95)"""
import sqlite3
from collections import defaultdict

c = sqlite3.connect("file:/freqtrade/user_data/tradesv3.sqlite?mode=ro", uri=True)
cur = c.cursor()

# Ambil SEMUA closed trade dengan data max/min
cur.execute("""SELECT t.id, t.pair, t.is_short, t.open_rate, t.max_rate, t.min_rate,
  t.close_profit, t.close_profit_abs, t.exit_reason
  FROM trades t WHERE t.is_open=0 AND t.open_rate IS NOT NULL
    AND t.max_rate IS NOT NULL AND t.min_rate IS NOT NULL ORDER BY t.id""")
rows = cur.fetchall()
print(f"Total closed trade: {len(rows)}")

results = {}
for label, th in [("SL 0.8%", 0.008), ("SL 1.0%", 0.010), ("SL 1.2%", 0.012), ("SL 1.5% (aktual)", 0.015)]:
    win = loss = scratch = 0
    pnl_sum = 0.0
    for tid, pair, short, o, maxr, minr, pf, pa, ex in rows:
        if short:
            # SHORT: profit kalau harga turun; SL di atas entry
            # escape kalau min_rate <= o * (1 - th)
            escaped = minr <= o * (1 - th)
            hit_sl = maxr >= o * (1 + th)
        else:
            # LONG: escape kalau max_rate >= o * (1 + th)
            escaped = maxr >= o * (1 + th)
            hit_sl = minr <= o * (1 - th)

        if escaped:
            # Trade berhasil capai threshold → minimal +th% (harga) → x5 leverage margin
            win += 1
            pnl_sum += th * 5  # asumsi 5x leverage
        elif hit_sl:
            loss += 1
            pnl_sum -= th * 5
        else:
            scratch += 1
            # tidak kena dua-duanya → exit via emergency/reversal (pakai aktual)
            pnl_sum += (pf or 0)

    results[label] = {"win": win, "loss": loss, "scratch": scratch, "pnl": pnl_sum}

print("\n" + "=" * 80)
print("SIMULASI SL TERHADAP SEMUA TRADE")
print("Escape = max/min pernah sentuh threshold | Hit SL = lawannya sentuh")
print("=" * 80)
print(f"{'Setting':<18} {'Win':>4} {'Loss':>4} {'Scratch':>8} {'P&L unit':>10}")
for label, r in results.items():
    print(f"{label:<18} {r['win']:>4} {r['loss']:>4} {r['scratch']:>8} {r['pnl']:>+9.2f}")

# Bandingkan dengan P&L aktual
cur.execute("SELECT SUM(close_profit_abs) FROM trades WHERE is_open=0")
actual_pnl = cur.fetchone()[0]
print(f"\nP&L aktual (semua exit): {actual_pnl:+.4f} USDT")

# Breakdown per pair: pair yang paling banyak SL hit
print("\n" + "=" * 80)
print("TOP PAIR DENGAN LOSS TERBESAR (semua trade, bukan cuma id>=95)")
print("=" * 80)
pair_stats = defaultdict(lambda: {"n": 0, "wins": 0, "pnl": 0.0})
for tid, pair, short, o, maxr, minr, pf, pa, ex in rows:
    pair_stats[pair]["n"] += 1
    if pf and pf > 0: pair_stats[pair]["wins"] += 1
    if pf: pair_stats[pair]["pnl"] += pf

worst = sorted(pair_stats.items(), key=lambda x: x[1]['pnl'])[:15]
print(f"{'Pair':<16} {'N':>3} {'Wins':>4} {'P&L':>9}")
for pair, d in worst:
    print(f"{pair:<16} {d['n']:>3} {d['wins']:>4} {d['pnl']*100:>+8.2f}%")
