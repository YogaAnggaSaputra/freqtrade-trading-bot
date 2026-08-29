# Freqtrade Trading Bot System

AI-powered cryptocurrency trading bot built on Freqtrade framework with advanced risk management and machine learning integration.

> ⚠️ **DISCLAIMER**: Repo ini untuk **educational purposes only**. Trading cryptocurrency melibatkan risiko tinggi. Gunakan dengan bijak dan hanya dengan dana yang siap hilang.

## 📊 Stats

| Metric | Value |
|---|---|
| **Framework** | Freqtrade 2026.3 |
| **Exchange** | Binance Futures |
| **Strategy** | AITradingStrategy (custom) |
| **Leverage** | 3x - 5x |
| **SL Floor** | 1.5% |
| **TP R-Multiple** | 5R / 10R / 20R |
| **Deployment target** | Dry-run/testnet first |

## 🚀 Fitur Utama

### Risk Management
- ✅ **SL Floor 1.5%** — Proteksi minimum dari wick
- ✅ **Trailing Stop** — Lock profit otomatis (Layer 1-4)
- ✅ **Circuit Breaker** — Stop trading setelah 3 loss beruntun
- ✅ **Kelly Sizing** — Position sizing dinamis
- ✅ **ADX Filter** — Exit reversal hanya saat tren lemah
- ✅ **Volume Filter** — Konfirmasi volume untuk reversal exit

### AI/ML Integration
- 🤖 **Ensemble Model** — Multi-model prediction
- 🤖 **Regime Classifier** — Deteksi market regime (bull/bear/ranging)
- 🤖 **MAE/MFE Predictor** — Dynamic stoploss optimization
- 🤖 **Quant/Sentiment/On-chain Engines** — Parameter adaptif, Fear & Greed, funding, OI delta
- 🤖 **Champion–Challenger** — Shadow 48 jam dan evaluasi holdout time-based
- 🤖 **Orderbook Intelligence** — DOM, spoofing, iceberg, VPIN-like

### Monitoring
- 📡 **Health Check** — Monitoring tiap 5 menit
- 📡 **Discord Notif** — Trade open/close real-time
- 📡 **Pipeline Auto-Recovery** — Self-healing training pipeline

## 📦 Struktur

```
.
├── configs/freqtrade/strategies/
│   └── AITradingStrategy.py      # Main strategy
├── services/
│   ├── freqtrade-runtime/        # Main trading bot
│   ├── model-inference/          # AI inference & training
│   ├── risk-gateway/             # Risk management
│   ├── market-data-gateway/      # Market data feed
│   ├── binance-adapter/          # Exchange integration
│   ├── discord-bot/              # Notification service
│   ├── quant-engine/             # Adaptive quant parameters/statistics
│   ├── sentiment-engine/         # Fear & Greed/CryptoPanic sentiment
│   ├── on-chain-engine/          # Funding and open-interest aggregation
│   ├── tick-recorder/            # Binance bookTicker recorder
│   ├── anomaly-detection/        # OHLC/return anomaly checks
│   ├── adaptive-whitelist/       # Dynamic volume/volatility whitelist
│   ├── orderbook-intelligence/   # DOM/spoofing/VPIN-like features
│   ├── execution-guard/          # Last-mile order validation
│   ├── execution-quality/        # Slippage/fill quality analysis
│   ├── capital-allocation/       # Drawdown/regime sizing
│   ├── position-monitor/         # Funding/stress assessment
│   ├── predictive-layer/         # Targets/liquidity/MTF alignment
│   ├── post-exit-regret/         # Counterfactual exit analysis
│   ├── telegram-control/         # Optional emergency control
│   ├── hermes-agent/             # AI monitoring agent
│   ├── loss-analyzer/            # Loss attribution analysis
│   ├── experiment-orchestrator/  # Hyperopt & optimization
│   └── reconciler/               # Data reconciliation
├── shared/
│   ├── db/                       # Database models
│   ├── feedback/                 # Feedback loop
│   ├── schemas/                  # Data schemas
│   └── utils/                    # Utilities
├── scripts/                      # Maintenance scripts
├── alembic/versions/             # DB migrations
├── docker-compose.yml            # Container orchestration
└── .env.example                  # Environment template
```

## 🛠️ Setup

### Prerequisites
- Docker & Docker Compose
- Binance Futures API key
- Python 3.11+

### Quick Start

```bash
# 1. Clone repo
git clone https://github.com/YogaAnggaSaputra/freqtrade-trading-bot.git
cd freqtrade-trading-bot

# 2. Setup environment
cp .env.example .env
# Edit .env dengan credential Anda

# 3. Run with Docker
docker compose up -d

# 4. Check status
curl http://localhost:8002/status
docker logs -f deploy-freqtrade-runtime-1
```

### Roadmap Engine Activation

Semua engine baru memiliki default aman. Jalankan migration sebelum service
analytics memakai database:

```bash
# Jika Alembic tersedia di host:
alembic upgrade head
```

Dengan Compose, gunakan migration runner satu kali:

```bash
docker compose --profile ops run --rm migrations
```

Aktifkan bertahap di `.env`, mulai dari dry-run/testnet:

```env
TRADE_MODE=dry
TICK_RECORDER_ENABLED=true
TICK_SYMBOLS=btcusdt,ethusdt
ADAPTIVE_WHITELIST_ENABLED=true
ON_CHAIN_ENGINE_ENABLED=true
NEWS_ALPHA_ENABLED=true
NEWS_LLM_ENABLED=false
CAPITAL_ALLOCATION_ENABLED=true
POSITION_RISK_ENABLED=true
```

Fitur yang dapat mengubah atau menghentikan order tetap opt-in:

```env
ORDERBOOK_GATE_ENABLED=false
ORDERBOOK_INTELLIGENCE_ENABLED=false
EXIT_CONSENSUS_ENABLED=false
EXECUTION_GUARD_ENABLED=false
KILL_SWITCH_CLOSE_POSITIONS=false
TELEGRAM_CONTROL_ENABLED=false
AUTO_PROMOTE_CHALLENGER=false
DEAD_MAN_ENABLED=false
ANOMALY_AUTO_PAUSE=true
EXECUTION_AUTO_PAUSE=false
```

Aktifkan `EXECUTION_GUARD_ENABLED` dan `KILL_SWITCH_CLOSE_POSITIONS` hanya
setelah testnet memverifikasi order validation dan reduce-only close-all.

## 🔧 Konfigurasi Strategi

### Parameter Utama

| Parameter | Value | Deskripsi |
|---|---|---|
| `SL_FLOOR` | 1.5% | Stop loss minimum |
| `MIN_RRR` | 1.3 | Risk-reward ratio minimum |
| `TP_R_MULTIPLE` | [5.0, 10.0, 20.0] | Take profit levels |
| `ADX_FILTER` | 30 | Threshold ADX untuk reversal exit |
| `VOLUME_FILTER` | 1.3 | Volume ratio minimum |
| `DUAL_CANDLE` | True | Dual-candle confirmation |

### Services Ports

| Service | Port | Purpose |
|---|---|---|
| freqtrade-runtime | 8002 | Bot API |
| model-inference | 8005 | ML inference |
| market-data-gateway | 8003 | Data feed |
| risk-gateway | 8004 | Risk validation |
| binance-adapter | 8001 | Exchange adapter |
| hermes-agent / loss-analyzer / experiment-orchestrator | 8006 / 8007 / 8008 | Monitoring & experiments |
| on-chain-engine | 8020 | Funding/OI/netflow contract |
| quant / sentiment / tick-recorder / anomaly | 8021 / 8022 / 8023 / 8024 | Intelligence & data quality |
| whitelist / execution-guard / execution-quality | 8025 / 8026 / 8027 | Entry and execution controls |
| capital / attribution / position-monitor | 8028 / 8029 / 8030 | Allocation and analytics |
| news / regret / orderbook / predictive | 8031 / 8032 / 8033 / 8034 | Alpha and open/close engines |

## 🐛 Bug Fixes

### [FIX-TRIGGER] Cegah SL Update Invalid (Aug 2026)
- **Problem**: SL price trailing bisa lebih tinggi dari current price (LONG) atau lebih rendah (SHORT)
- **Error**: Binance reject dengan `-2021: Order would immediately trigger`
- **Fix**: Validasi SL price sebelum update — skip kalau tidak valid

### [FIX-SL-FLOOR] SL Floor 1.5% (Aug 2026)
- **Problem**: SL terlalu rapat (0.5%) → kena noise wick
- **Fix**: Floor minimum 1.5% untuk semua SL

### [FIX-ADX] ADX Filter (Aug 2026)
- **Problem**: Reversal exit saat tren kuat → loss besar
- **Fix**: Exit reversal hanya saat ADX < 30

## 📝 Quick Commands

```bash
# Status containers
docker compose ps

# View logs
docker logs -f deploy-freqtrade-runtime-1

# Restart bot
docker compose restart freqtrade-runtime

# Stop all
docker compose down

# Rebuild
docker compose up -d --build
```

## 📊 Performance

> ⚠️ **Past performance does not guarantee future results.**

Bot ini ditujukan untuk Binance Futures dengan konfigurasi awal:
- **Mode**: DRY/TESTNET sampai verifikasi operasional selesai
- **Stake**: 1 USDT per trade
- **Max Open Trades**: 1
- **Pairs**: 100+ altcoins

## 🔐 Security

- ✅ **No hardcoded credentials** — semua via environment variables
- ✅ **API keys stored separately** — di VPS, tidak di repo
- ✅ **Read-only DB access** — untuk monitoring
- ✅ **Rate limiting** — anti-ban dari exchange

## 🤝 Contributing

1. Fork repo
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

MIT License — see [LICENSE](LICENSE) file.

## 🔗 Links

- [Freqtrade Documentation](https://www.freqtrade.io/)
- [Binance Futures API](https://binance-docs.github.io/apidocs/futures/en/)
- [Original VPS](https://github.com/YogaAnggaSaputra) — Private backup

---

**Last Updated**: 29 August 2026
**Maintainer**: YogaAnggaSaputra
**Status**: 🟡 Implementasi roadmap selesai; runtime, migrasi, dan testnet perlu dijalankan operator

Lihat [ROADMAP_ENGINE_STATUS.md](ROADMAP_ENGINE_STATUS.md) untuk matriks
komponen, kontrak provider, dan checklist aktivasi.
