# Freqtrade Trading Bot System

Backup sistem trading bot Freqtrade dari VPS YogaAnggaSaputra.

## ⚠️ PENTING

Repo ini adalah **backup source code saja**. Tidak termasuk:
- Credential API (Binance, Telegram, Discord)
- Database trades
- Data historis candle
- Model ML yang sudah di-train

## 📦 Struktur

```
.
├── configs/freqtrade/strategies/
│   └── AITradingStrategy.py      # Strategi utama (ADX filter, SL floor, TP R-multiple)
├── services/
│   ├── freqtrade-runtime/        # Container bot utama
│   ├── model-inference/          # AI model (ensemble, feature engine, retrainer)
│   ├── risk-gateway/             # Risk management (Kelly sizing, macro filter)
│   ├── market-data-gateway/      # Data fetcher (candle, orderbook, liquidation)
│   ├── binance-adapter/          # Binance client & execution
│   ├── discord-bot/              # Notifikasi Discord
│   ├── hermes-agent/             # AI monitoring agent
│   ├── loss-analyzer/            # Analisis loss attribution
│   ├── experiment-orchestrator/  # Optuna optimizer
│   └── reconciler/               # Data reconciliation
├── shared/
│   ├── db/                       # Database models & session
│   ├── feedback/                 # Feedback loop
│   ├── schemas/                  # Data schemas
│   └── utils/                    # Shared utilities
├── scripts/                      # Maintenance scripts
├── alembic/versions/             # Database migrations
├── docker-compose.yml            # Orkestrasi container
└── .env.example                  # Template environment variables
```

## 🚀 Setup dari Nol

### 1. Clone & Persiapan

```bash
git clone https://github.com/YogaAnggaSaputra/freqtrade-trading-bot.git
cd freqtrade-trading-bot
```

### 2. Environment Variables

```bash
cp .env.example .env
# Edit .env dengan credential asli:
# - BINANCE_API_KEY
# - BINANCE_API_SECRET
# - TELEGRAM_BOT_TOKEN
# - DISCORD_WEBHOOK_URL
# - DB_PASSWORD
```

### 3. Jalankan dengan Docker

```bash
docker compose up -d
```

### 4. Verifikasi

- Bot status: `curl http://localhost:8002/status`
- Logs: `docker logs -f deploy-freqtrade-runtime-1`

## 🔧 Konfigurasi Penting

### Strategi (AITradingStrategy.py)

```python
# Parameter utama (per 27 Aug 2026):
SL_FLOOR = 1.5%           # Stop loss minimum
MIN_RRR = 1.3             # Risk-reward ratio minimum
TP_R_MULTIPLE = [5.0, 10.0, 20.0]  # TP1/TP2/TP3
ADX_FILTER_THRESHOLD = 30 # Exit reversal hanya saat ADX < 30
VOLUME_FILTER = 1.3       # Volume ratio minimum untuk reversal exit
DUAL_CANDLE_CONFIRM = True # Butuh 2 candle berturut-turut
```

### Services

| Service | Port | Fungsi |
|---|---|---|
| freqtrade-runtime | 8002 | Bot trading utama |
| model-inference | 8005 | AI inference & training |
| market-data-gateway | 8003 | Data feed |
| risk-gateway | 8004 | Risk validation |

## 📊 Monitoring

- Health check tiap 5 menit via cronjob
- Notifikasi trade open/close via Discord
- Pipeline auto-recovery tiap 6 jam

## ⚡ Quick Commands

```bash
# Cek status bot
docker compose ps

# Lihat log
docker logs -f deploy-freqtrade-runtime-1

# Restart bot
docker compose restart freqtrade-runtime

# Stop semua
docker compose down
```

## 📝 Catatan

- **VPS asli**: Jangan diubah langsung dari repo ini
- **Backup ini**: Hanya source code, bukan live system
- **Update**: Kalau ada perubahan di VPS, commit & push ke repo ini

---

**Last Updated**: 27 August 2026
**Source**: VPS YogaAnggaSaputra
