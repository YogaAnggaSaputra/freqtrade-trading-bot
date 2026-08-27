import sys; sys.path.insert(0, '/app')
import asyncio, sqlite3
from shared.feedback import emit_trade_closed

c = sqlite3.connect('file:/freqtrade/user_data/tradesv3.sqlite?mode=ro', uri=True).cursor()
c.execute("""
    SELECT id, pair, is_short, open_rate, close_rate, close_profit, close_profit_abs,
           exit_reason, open_date, close_date
    FROM trades WHERE is_open=0 ORDER BY id
""")
rows = c.fetchall()
print(f"Found {len(rows)} closed historical trades", flush=True)

async def run_all():
    n_ok = n_skip = 0
    for r in rows:
        tid, pair, is_short, open_rate, close_rate, close_profit, close_profit_abs, exit_reason, open_date, close_date = r
        if not open_rate or not close_rate or not open_date or not close_date:
            n_skip += 1
            continue
        side = "short" if is_short else "long"
        pnl_pct = float(close_profit or 0.0)
        actual_rr = (abs(pnl_pct) / 0.01) * (1 if pnl_pct >= 0 else -1)
        outcome = {
            "trade_id": int(tid), "pair": pair, "timeframe": "5m",
            "entry_conditions": {
                "side": side, "entry_rate": float(open_rate), "exit_rate": float(close_rate),
                "regime": None, "predicted_rr": None, "ml_signal": None, "ml_prob": None,
                "conf_score": None, "atr_ratio": None, "_source": "historical_backfill",
            },
            "exit_reason": str(exit_reason or "unknown"),
            "pnl_pct": pnl_pct, "pnl_abs": float(close_profit_abs or 0.0),
            "predicted_rr": None, "actual_rr": actual_rr, "regime_at_entry": None,
            "timestamp_entry": str(open_date), "timestamp_exit": str(close_date),
        }
        try:
            ok = await emit_trade_closed(outcome)
            if ok: n_ok += 1
            else: n_skip += 1
            print(f"  trade {tid} {pair} -> {ok}", flush=True)
        except Exception as e:
            n_skip += 1
            print(f"  trade {tid} ERROR: {e}", flush=True)
    print(f"Backfill result: emitted={n_ok}, skipped={n_skip}", flush=True)

asyncio.run(run_all())
