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
| **Live Since** | August 2026 |

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
- 🤖 **Orderbook Imbalance** — I(t) probe untuk entry signal

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

Bot ini running di Binance Futures dengan konfigurasi:
- **Mode**: LIVE (real money)
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

**Last Updated**: 28 August 2026  
**Maintainer**: YogaAnggaSaputra  
**Status**: 🟢 Production Live
