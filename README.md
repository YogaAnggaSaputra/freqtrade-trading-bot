# Hermes Adaptive Futures Trading Platform

> **Modular Monolith**: Binance USDT-M Futures + Freqtrade + Risk Gateway + Hermes Agent

## Overview

Autonomous crypto futures trading platform with AI-driven strategy improvement, built on safety-first architecture:

- **Exchange**: Binance USDT-M Futures (mainnet; testnet via `BINANCE_FUTURES_TESTNET=true`)
- **Trading Engine**: Freqtrade (custom Binance Futures exchange plugin + AITradingStrategy)
- **Risk Control**: Fail-closed Risk Gateway with locked policies
- **AI Agent**: Hermes (read-only analysis → proposals → experiments)
- **Data**: PostgreSQL + TimescaleDB + Parquet (market data lake)
- **Observability**: Prometheus + Grafana + Alertmanager + Discord

## Quick Start

```bash
# 1. Setup
cp .env.example .env
# Edit .env — tentukan mode:
#   TRADE_MODE=demo  (default, dry-run, AMAN — tidak kirim order real)
#   TRADE_MODE=live  (order real ke Binance Futures!)

# 2. Create secrets
mkdir -p secrets
echo "YOUR_API_KEY" > secrets/binance_api_key.txt
echo "YOUR_API_SECRET" > secrets/binance_api_secret.txt
echo "YOUR_DISCORD_BOT_TOKEN" > secrets/discord_bot_token.txt

# 3. Deploy
docker compose up -d

# 4. Migrate DB
docker compose exec postgres alembic upgrade head

# 5. Access
# Grafana: http://localhost:3000 (admin / $GRAFANA_ADMIN_PASSWORD)
# Prometheus: http://localhost:9090
```

### Mode Trading

| Mode | `TRADE_MODE` | `BINANCE_FUTURES_TESTNET` | Perilaku |
|------|-------------|--------------------------|----------|
| **Demo (default)** | `demo` | apa saja | Dry-run, order virtual, aman untuk testing |
| **Live** | `live` | `false` | Order real ke Binance Futures mainnet |

> **PENTING untuk modal kecil ($1-50):** bot dikonfigurasi untuk micro-account —
> pair murah (DOGE, SOL, 1000PEPE, dkk), min-notional floor $5, exposure 100%.
> Satu posisi ≈ sebagian besar balance. Ini mode high-risk yang disengaja.

### Alur Live (sesudah perubahan)

```
Strategi (5m scalping + ML confirmation)
  → Risk Gateway (/validate: kill switch, policy, margin-exposure, SL valid,
     strategy approved via deployments table, min-notional)
  → Binance Adapter (order real ke exchange)
  → Webhook → TradeDossier → Loss Analyzer → Hermes → Experiment → Re-deploy
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    BINANCE USDT-M FUTURES (Mainnet)                  │
│                   REST / WebSocket / Futures API                     │
└────────────────────────────┬────────────────────────────────────────┘
                             │
        ┌────────────────────┴────────────────────┐
        ▼                                         ▼
┌────────────────────────┐            ┌────────────────────────┐
│ Market Data Gateway    │            │ Exchange Reconciler    │
│ candles, ticker, OI    │            │ orders, fills, balance │
└───────────┬────────────┘            └───────────┬────────────┘
            │                                      │
            └────────────────┬────────────────────┘
                             ▼
            ┌─────────────────────────────────────┐
            │ Operational DB + Raw Data Lake       │
            │ PostgreSQL + TimescaleDB + Parquet   │
            └───────────────┬─────────────────────┘
                            │
         ┌──────────────────┼──────────────────────────────┐
         ▼                  ▼                              ▼
┌───────────────┐   ┌──────────────────┐          ┌──────────────────┐
│ Feature Engine│   │ Loss Analyzer    │          │ Monitoring/Alert │
│ indicators    │   │ incident labels  │          │  Grafana/Discord │
└──────┬────────┘   └────────┬─────────┘          └──────────────────┘
       │                     │
       ▼                     ▼
┌────────────────────┐ ┌─────────────────────────────────┐
│ ML Inference       │ │ Hermes Agent                     │
│ probability/regime │ │ evidence → hypothesis → proposal │
└─────────┬──────────┘ └───────────────┬─────────────────┘
          │                             │
          └───────────────┬─────────────┘
                          ▼
             ┌─────────────────────────┐
             │ Policy & Approval Gate  │
             │ locked limits, validator│
             └───────┬────────┬────────┘
                     │        │
               rejected       ▼
                     │  ┌──────────────────────────┐
                     │  │ Experiment Orchestrator  │
                     │  │ test/backtest/walkforward│
                     │  └─────────────┬────────────┘
                     │                ▼
                     │  ┌──────────────────────────┐
                     │  │ Model/Strategy Registry  │
                     │  │ approved versions only   │
                     │  └─────────────┬────────────┘
                     │                ▼
                     │  ┌──────────────────────────┐
                     └─►│ Freqtrade Runtime        │
                        │ strategy + protections   │
                        └─────────────┬────────────┘
                                      ▼
                        ┌──────────────────────────┐
                        │ External Risk Gateway    │
                        │ fail-closed + kill switch│
                        └─────────────┬────────────┘
                                      ▼
                        ┌──────────────────────────┐
                        │ Binance Execution Adapter│
                        │ order / SL / TP / verify │
                        └──────────────────────────┘
```

## Safety Principles

| Component | Can Do | Cannot Do |
|-----------|--------|-----------|
| Hermes Agent | Read data, diagnose losses, create proposals | Access API keys, send orders, modify live config |
| Experiment Runner | Create candidates, run tests | Auto-promote to live |
| Policy Gateway | Approve experiments per rules | Modify locked policies |
| Freqtrade | Run approved strategies | Bypass risk gateway (order WAJIB lewat /validate) |
| Risk Gateway | Reject/approve trade intent, pause, kill switch | Place orders without exchange validation |
| Binance Adapter | Send approved orders | Accept commands directly from Hermes |

> **Enforcement nyata:** semua order entry/exit dari Freqtrade sekarang WAJIB
> melewati Risk Gateway `/validate` (fail-closed). Order yang ditolak tidak pernah
> sampai ke exchange. SL/TP proteksi tetap langsung ke exchange agar posisi
> terbuka selalu terlindungi.

## Locked Policies (Immutable by Hermes)

| Policy | Value |
|--------|-------|
| Mode | Demo trading |
| Margin | Isolated |
| Pair | DOGE, SOL, 1000PEPE, 1000SHIB, XRP, ADA (USDT-M) |
| Max Leverage | 5x |
| Risk per Trade | 50% equity (micro-account: 1 posisi ≈ seluruh balance) |
| Total Exposure | 100% equity (margin-based) |
| Daily Loss Limit | 50% equity |
| Max Drawdown | 80% equity |
| Max Positions | 1 |
| Position Adjustment | Disabled |
| Stop-loss | Mandatory |
| API Withdrawal | Disabled |
| Live Promotion | Owner approval required |

## Strategi & SL/TP (AITradingStrategy v2.4 — scalping 5m)

- **Timeframe**: 5m (indikator MTF 15m/1h/4h/1d untuk konfirmasi arah)
- **Entry**: kill zone + confluence score (≥70) + ADX ≥ 20 + volume ≥ 1.2x + ML confirmation
- **Position sizing**: risk-based dengan **floor min-notional $5** (agar order tidak ditolak exchange di micro-account)

### SL/TP Pintar (berlapis)

| Mekanisme | Cara kerja |
|-----------|-----------|
| **SL dinamis** | ATR 1.5× (floor 0.5% agar tidak kena noise 5m) |
| **SL/TP ML** | `/mae-mfe-predict` — rekomendasi SL & TP dari data historis (live saja) |
| **Trailing adaptif** | breakeven di 1R, lock 0.5R di 2R (berbasis ATR) |
| **Partial TP** | TP1 (1R) close 40%, TP2 (1.8R) close 35%, runner ke 3R |
| **Dynamic TP** | exit penuh saat harga sentuh TP rekomendasi ML |
| **Reversal exit** | engulfing/shooting star berlawanan saat profit |
| **Regime exit** | keluar jika trend berbalik |
| **Circuit breaker** | daily loss > 50% → berhenti |

> ML layer aktif di **live/dry-run**; otomatis nonaktif di backtest
> (model-inference tidak berjalan di container backtest).

## Project Structure

```
botbinance/
├── configs/
│   ├── freqtrade/           # Freqtrade config + strategies (AITradingStrategy)
│   ├── risk-gateway/        # Locked policy.yaml
│   ├── prometheus/          # Prometheus rules
│   ├── grafana/             # Dashboard provisioning
│   └── alertmanager/        # Alert routing
├── services/
│   ├── binance-adapter/     # REST + WS client, order mgmt
│   ├── freqtrade-runtime/   # Freqtrade wrapper + custom exchange
│   ├── risk-gateway/        # Fail-closed pre-trade validator
│   ├── market-data-gateway/ # WS → PG + Parquet
│   ├── reconciler/          # State reconciliation cron
│   ├── discord-bot/         # Alerts + manual controls
│   ├── hermes-agent/        # Read-only analyzer (Phase 3)
│   ├── loss-analyzer/       # Trade dossiers + loss classification
│   ├── model-inference/     # ML probability, regime, MAE/MFE (Phase 4)
│   └── experiment-orchestrator/ # Backtest → walkforward → demo
├── shared/
│   ├── schemas/             # Pydantic v2 models
│   ├── db/                  # SQLAlchemy 2.0 + Alembic
│   ├── messaging/           # Redis pub/sub
│   └── security/            # Docker Secrets loader
├── tests/
│   ├── unit/                # Schema, logic tests
│   ├── integration/         # Service integration tests
│   └── e2e/                 # Full cycle tests
├── scripts/                 # Deploy, backup, maintenance
├── alembic/                 # DB migrations
├── docker-compose.yml       # Demo stack
├── docker-compose.prod.yml  # Production overrides
└── Makefile                 # Dev commands
```

## Deployment Phases

| Phase | Description | Duration |
|-------|-------------|----------|
| **1. Foundation** | Binance adapter, Freqtrade demo, Risk gateway, PG, Discord, Kill switch | 1 month |
| **2. Loss Analytics** | Trade dossiers, loss classifier, dashboards | 2 weeks |
| **3. Hermes Read-Only** | Proposals from loss data, policy gateway | 2 weeks |
| **4. Auto Experiments** | Backtest → walkforward → stress → registry | 3 weeks |
| **5. Shadow/Canary** | Shadow signals, 14-day demo, canary, rollback | 4 weeks |
| **6. Live Limited** | 6 pair murah, 1 posisi, manual scale-up | Ongoing |

## Monitoring

### Dashboards (Grafana)
- **Trading Overview**: Equity, PnL, DD, exposure, leverage, positions
- **Execution**: Orders, fills, SL/TP status, latency, reject rate
- **Risk**: Daily loss, max DD, streak, kill switch state
- **Data Health**: Candle freshness, WS status, API latency, rate limits
- **System**: Container health, DB size, Redis memory

### Critical Alerts (Discord)
- Position mismatch (local vs exchange)
- SL/TP not placed after fill
- Daily loss / max DD limits hit
- WebSocket disconnected > 60s
- API key invalid / rate limited
- Kill switch activated
- Hermes proposal created
- Container crash loop

### Discord Commands
```
/status      - Full status (equity, positions, deployments, kill switch)
/positions   - Open positions with PnL
/equity      - PnL summary + recent trades
/kill yellow|orange|red|black [reason] - Activate kill switch
/resume      - Resume trading (clear kill switch)
/logs [n]    - Last n incidents
```

## Development

```bash
# Start dev stack
make up

# View logs
make logs

# Run tests
make test

# Lint
make lint

# DB migrations
make db-migrate
make db-revision message="add new column"

# Shell access
make shell-db
make shell-redis
```

## Security

- **Secrets**: Docker Secrets only (never in Git, .env, or logs)
- **API Keys**: Trade-only, withdrawal disabled
- **Network**: Internal Docker network, no external DB/Redis access
- **Audit**: All actions logged to `audit_events` table
- **Fail-Closed**: Risk Gateway rejects on any uncertainty

## License

Proprietary - Internal use only.