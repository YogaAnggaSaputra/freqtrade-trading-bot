# Status Implementasi Engine Roadmap

Dokumen ini memisahkan implementasi kode dari verifikasi operasional. Semua
komponen di bawah sudah memiliki kontrak service, konfigurasi Compose, dan
jalur integrasi yang relevan. Default yang dapat mengubah ukuran order,
memblokir entry, menutup posisi, atau mempromosikan model tetap opt-in.

## Matriks implementasi

| Layer | Komponen | Status kode | Integrasi utama |
|---|---|---:|---|
| Intelligence | Quant engine | Selesai (Supreme Math v2) | Monte Carlo path/ruin, VaR/CVaR, Jacobi PCA, HMM, Platt/PAVA, risk parity |
| Intelligence | Sentiment engine | Selesai | Fear & Greed, CryptoPanic, optional Telegram channel bot |
| Intelligence | On-chain engine | Selesai | Binance/Bybit/OKX funding, OI delta, optional netflow/whale provider |
| Intelligence | News alpha | Selesai | keyword fallback + optional Ollama-compatible Qwen/Llama |
| Execution | Execution quality | Selesai | slippage, fill profile, maker/market decision |
| Execution | Orderbook intelligence | Selesai | DOM, VPIN-like, spoofing, iceberg evidence, optional gate |
| Automation | Auto retraining | Selesai | event/schedule trigger, drift/feature feedback, candidate artifact |
| Automation | Champion–challenger | Selesai | 48-hour durable shadow state, holdout scorer, promotion threshold |
| Automation | Adaptive whitelist | Selesai | volume/ATR/edge ranking, one-hour cache, strategy fallback |
| Automation | Capital allocation | Selesai (Risk Parity Edition) | CCD equal-risk contribution, min-variance, volatility parity |
| Safety | Anomaly detection | Selesai | OHLC/data quality, return z-score, market anomaly channel |
| Safety | Kill switch | Selesai | PIN Telegram/API, auto pause, graceful levels, dead-man alert |
| Safety | Execution guard | Selesai (Computed Corr) | real-time Pearson correlation matrix, fallback btc-cluster guard |
| Analytics | Performance attribution | Selesai | factor/timing/selection and exit-regret report |
| Analytics | Regime calendar | Selesai | UTC hour/day/funding-reset reports |
| Open position | Position health | Selesai | thesis, momentum decay, regime, funding, stress, persisted snapshots |
| Open position | Dynamic exit consensus | Selesai | weighted regime/ML/momentum/reversal/volume/funding score |
| Open position | Stress/adversarial | Selesai | wick/gap/cascade stress and stop-hunt/fakeout inputs |
| Close position | Exit quality | Selesai | order-type decision and execution profiling |
| Close position | Regret minimizer | Selesai | next 5/10/20 candle counterfactual worker + DB persistence |
| Close position | Partial-close intelligence | Selesai | volatility-adjusted TP1/TP2 and runner protection |
| Predictive | Target/liquidity/MTF | Selesai | measured move, liquidity levels, multi-timeframe alignment |

“Selesai” berarti implementasi dan wiring tersedia; bukan berarti data
provider eksternal, database, exchange testnet, atau 48 jam shadow sudah
memiliki data aktual.

## Urutan aktivasi aman

1. Salin `.env.example` menjadi `.env`, isi secret, dan mulai dengan
   `TRADE_MODE=dry` atau Binance Futures testnet.
2. Jalankan migrasi satu kali:
   `docker compose --profile ops run --rm migrations`.
3. Nyalakan recorder, sentiment/on-chain, whitelist, dan analytics; pastikan
   endpoint `/health` seluruh service sehat.
4. Jalankan backtest dan walk-forward dengan data lokal. Runner menolak
   membuat metrik palsu bila executable Freqtrade tidak tersedia.
5. Isi metrik shadow challenger secara berkala melalui endpoint orchestrator;
   promosi baru valid setelah 48 jam, sample minimum, akurasi minimum, dan
   improvement minimum terpenuhi.
6. Setelah testnet lulus, aktifkan guard/gate satu per satu. Pertahankan
   `AUTO_PROMOTE_CHALLENGER=false` kecuali proses approval operator sudah
   diuji.

## Kontrak provider opsional

- `NEWS_LLM_URL` menerima endpoint Ollama `/api/generate`; model default
  `qwen2.5:1.5b`. `NEWS_LLM_ENABLED=false` tetap fallback ke keyword.
- `TELEGRAM_SENTIMENT_BOT_TOKEN` harus memakai bot terpisah dari bot
  emergency-control karena Telegram hanya mengizinkan satu long-poll consumer
  per token. Bot perlu akses ke channel yang dipantau.
- `ONCHAIN_DATA_PROVIDER_URL` harus mengembalikan JSON ternormalisasi dengan
  `netflow`, optional `netflow_signal` (-1..1), dan/atau
  `large_transactions`/`whale_alerts`. Tanpa provider,
  engine mengembalikan `null`/list kosong dan tidak mengarang sinyal.

## Verifikasi lokal terakhir

- `python -m unittest discover -s tests -v` — lulus.
- `python -m compileall -q shared services scripts configs` — lulus.
- `git diff --check` — tidak ada whitespace error.
- `docker compose config --quiet` — lulus.

Build image, migrasi database, health check antar-container, exchange
testnet, dan observasi 48 jam perlu dijalankan pada host yang Docker Desktop,
Postgres, Redis, serta credential/provider-nya aktif.
