#!/usr/bin/env python3
"""Download OHLCV data for all pairs from whitelist"""
import requests
import pandas as pd
import pyarrow.feather as feather
import time
import os

# Create directory
base_dir = "/freqtrade/user_data/data/binanceusdm/futures/5m"
os.makedirs(base_dir, exist_ok=True)

# All pairs from whitelist
pairs = [
    "0GUSDT", "1000000BOBUSDT", "1000000MOGSUSDT", "1000BONKSUSDT", "1000CATSUSDT",
    "1000CHEEMSSUSDT", "1000FLOKISUSDT", "1000LUNCSUSDT", "1000PEPESUSDT", "1000RATSSUSDT",
    "1000SATSSUSDT", "1000SHIBSUSDT", "1000XECUSDT", "1INCHUSDT", "1MBABYDOGESUSDT",
    "2ZUSDT", "4USDT", "AUSDT", "AAVEUSDT", "ACEUSDT", "ACHUSDT", "ACTUSDT",
    "ACUUSDT", "ADAUSDT", "AEROUSDT", "AEVOUSDT", "AGLDUSDT", "AGTUSDT", "AIAUSDT",
    "AIGENSYNUSDT", "AINUSDT", "AIOUSDT", "AIOTUSDT", "AIXBTUSDT", "AKEUSDT", "AKTUSDT",
    "ALCHUSDT", "ALGOUSDT", "ALICEUSDT", "ALLUSDT", "ALLOUSDT", "ALPINEUSDT", "ALTUSDT",
    "ANIMEUSDT", "ANKRUSDT", "APEUSDT", "API3USDT", "APRUSDT", "APTUSDT", "ARUSDT",
    "ARBUSDT", "ARCUSDT", "ARIAUSDT", "ARKUSDT", "ARKMUSDT", "ARPAUSDT", "ARXUSDT",
    "ASRUSDT", "ASTERUSDT", "ASTRUSDT", "ATUSDT", "ATHUSDT", "ATOMUSDT", "AUCTIONUSDT",
    "AVAUSDT", "AVAAIUSDT", "AVAXUSDT", "AVNTUSDT", "AWEUSDT", "AXLUSDT", "AXSUSDT",
    "AZTECUSDT", "BUSDT", "B2USDT", "BABYUSDT", "BANUSDT", "BANANAUSDT", "BANANAS31USDT",
    "BANDUSDT", "BANKUSDT", "BARDUSDT", "BASUSDT", "BASEDUSDT", "BATUSDT", "BBUSDT",
    "BCHUSDT", "BEAMXUSDT", "BEATUSDT", "BELUSDT", "BERAUSDT", "BICOUSDT", "BIGTIMEUSDT",
    "BILLUSDT", "BIOUSDT", "BIRBUSDT", "BLESSUSDT", "BLUAIUSDT", "BLURUSDT", "BMTUSDT",
    "BNBUSDT", "BNTUSDT", "BOMEUSDT", "BRUSDT", "BRETTUSDT", "BREVUSDT", "BROCCOLI714USDT",
    "BROCCOLIF3BUSDT", "BSBUSDT", "BSVUSDT", "BTCUSDT", "BTCDOMUSDT", "BTRUSDT", "BTWUSDT",
    "BULLAUSDT", "CUSDT", "C98USDT", "CAKEUSDT", "CAPUSDT", "CARVUSDT", "CATIUSDT",
    "CCUSDT", "CELOUSDT", "CELRUSDT", "CETUSUSDT", "CFGUSDT", "CFXUSDT", "CGPTUSDT",
    "CHILLGUYUSDT", "CHIPUSDT", "CHRUSDT", "CHZUSDT", "CKBUSDT", "CLANKERUSDT", "CLOUSDT",
    "COAIUSDT", "COLLECTUSDT", "COMPUSDT", "COOKIEUSDT", "COTIUSDT", "COWUSDT", "CROSSUSDT",
    "CRVUSDT", "CTKUSDT", "CTRUSDT", "CTSIUSDT", "CVCUSDT", "CVXUSDT", "CYBERUSDT",
    "CYSUSDT", "DASHUSDT", "DATAIPUSDT", "DEEPUSDT", "DEXEUSDT", "DIAUSDT", "DODOXUSDT",
    "DOGEUSDT", "DOGSUSDT", "DOLOUSDT", "DOODUSDT", "DOSUSDT", "DOTUSDT", "DRIFTUSDT",
    "DUSKUSDT", "DYDXUSDT", "DYMUSDT", "EDENUSDT", "EDGEUSDT", "EDUUSDT", "EGLDUSDT",
    "EIGENUSDT", "ELSAUSDT", "ENOUSDT", "ENJUSDT", "ENSUSDT", "ENSOUSDT", "EPICUSDT",
    "ERAUSDT", "ESPUSDT", "ESPORTSUSDT", "ETCUSDT", "ETHUSDT", "ETHFIUSDT", "ETHWUSDT",
    "EULUSDT", "EVAAUSDT", "FUSDT", "FARTCOINUSDT", "FETUSDT", "FFUSDT", "FHEUSDT",
    "FIDAUSDT", "FIGHTUSDT", "FILUSDT", "FLOCKUSDT", "FLOWUSDT", "FLUIDUSDT", "FLUXUSDT",
    "FOGOUSDT", "FOLKSUSDT", "FORMUSDT", "FRAXUSDT", "GUSDT", "GALAUSDT", "GASUSDT",
    "GENIUSUSDT", "GIGGLEUSDT", "GLMUSDT", "GMTUSDT", "GMXUSDT", "GOATUSDT", "GPSUSDT",
    "GRAMUSDT", "GRASSUSDT", "GRIFFAINUSDT", "GRTUSDT", "GRVTUSDT", "GTCUSDT", "GUAUSDT",
    "GUNUSDT", "GWEIUSDT", "HUSDT", "HAEDALUSDT", "HANAUSDT", "HBARUSDT", "HEIUSDT",
    "HEMIUSDT", "HIVEUSDT", "HMSTRUSDT", "HOLOUSDT", "HOMEUSDT", "HOTUSDT", "HUMAUSDT",
    "HYPEUSDT", "HYPERUSDT", "ICNTUSDT", "ICPUSDT", "ICXUSDT", "IDUSDT", "IDOLUSDT",
    "ILVUSDT", "IMXUSDT", "INUSDT", "INITUSDT", "INJUSDT", "INXUSDT", "IOUSDT", "IOSTUSDT",
    "IOTAUSDT", "IOTXUSDT", "IRYSUSDT", "JASMYUSDT", "JCTUSDT", "JELLYJELLYUSDT", "JOEUSDT",
    "JSTUSDT", "JTOUSDT", "JUPUSDT", "KAIAUSDT", "KAITOUSDT", "KASUSDT", "KATUSDT",
    "KAVAUSDT", "KERNELUSDT", "KGENUSDT", "KITEUSDT", "KMNOUSDT", "KNCUSDT", "KOMAUSDT",
    "KSMUSDT", "LAUSDT", "LABUSDT", "LAYERUSDT", "LDOUSDT", "LIGHTUSDT", "LINEAUSDT",
    "LINKUSDT", "LISTAUSDT", "LITUSDT", "LPTUSDT", "LQTYUSDT", "LSKUSDT", "LTCUSDT",
    "LUMIAUSDT", "LUNA2USDT", "LYNUSDT", "MUSDT", "MAGICUSDT", "MAGMAUSDT", "MANAUSDT",
    "MANTAUSDT", "MANTRAUSDT", "MASKUSDT", "MAVUSDT", "MAVIAUSDT", "MEUSDT", "MEGAUSDT",
    "MELANIAUSDT", "MEMEUSDT", "MERLUSDT", "METUSDT", "METISUSDT", "MEWUSDT", "MINAUSDT",
    "MIRAUSDT", "MITOUSDT", "MMTUSDT", "MOCAUSDT", "MONUSDT", "MOODENGUSDT", "MORPHOUSDT",
    "MOVEUSDT", "MOVRUSDT", "MTLUSDT", "MUBARAKUSDT", "MYXUSDT", "NAORISUSDT", "NEARUSDT",
    "NEIROUSDT", "NEOUSDT", "NEWTUSDT", "NIGHTUSDT", "NILUSDT", "NMRUSDT", "NOMUSDT",
    "NOTUSDT", "NXPCUSDT", "OUSDT", "OGUSDT", "OGNUSDT", "ONUSDT", "ONDOUSDT", "ONEUSDT",
    "ONGUSDT", "ONTUSDT", "OPUSDT", "OPENUSDT", "OPGUSDT", "OPNUSDT", "ORCAUSDT", "ORDERUSDT",
    "ORDIUSDT", "PARTIUSDT", "PAXGUSDT", "PENDLEUSDT", "PENGUUSDT", "PEOPLEUSDT", "PHAUSDT",
    "PHAROSUSDT", "PIEVERSEUSDT", "PIPPINUSDT", "PIXELUSDT", "PLAYUSDT", "PLUMEUSDT",
    "PNUTUSDT", "POLUSDT", "POLYXUSDT", "POPCATUSDT", "PORTALUSDT", "POWERUSDT", "POWRUSDT",
    "PRLUSDT", "PROMUSDT", "PROMPTUSDT", "PROVEUSDT", "PTBUSDT", "PUMPUSDT", "PUMPBTCUSDT",
    "PUNDIXUSDT", "PYTHUSDT", "QUSDT", "QNTUSDT", "QTUMUSDT", "RAREUSDT", "RAVEUSDT",
    "RAYSOLUSDT", "REUSDT", "RECALLUSDT", "REDUSDT", "RENDERUSDT", "RESOLVUSDT", "REZUSDT",
    "RIFUSDT", "RIVERUSDT", "RLCUSDT", "ROBOUSDT", "RONINUSDT", "ROSEUSDT", "RPLUSDT",
    "RSRUSDT", "RUNEUSDT", "RVNUSDT", "SUSDT", "SAFEUSDT", "SAGAUSDT", "SAHARAUSDT",
    "SANDUSDT", "SANTOSUSDT", "SAPIENUSDT", "SCRUSDT", "SCRTUSDT", "SEIUSDT", "SENTUSDT",
    "SFPUSDT", "SHELLUSDT", "SIGNUSDT", "SIRENUSDT", "SKLUSDT", "SKRUSDT", "SKYUSDT",
    "SKYAIUSDT", "SLPUSDT", "SLXUSDT", "SNXUSDT", "SOLUSDT", "SOLVUSDT", "SOMIUSDT",
    "SONICUSDT", "SOONUSDT", "SOPHUSDT", "SPACEUSDT", "SPELLUSDT", "SPKUSDT", "SPORTFUNUSDT",
    "SPXUSDT", "SQDUSDT", "SSVUSDT", "STABLEUSDT", "STARUSDT", "STBLUSDT", "STEEMUSDT",
    "STGUSDT", "STOUSDT", "STORJUSDT", "STRKUSDT", "STXUSDT", "SUIUSDT", "SUNUSDT",
    "SUPERUSDT", "SUSHIUSDT", "SWARMSUSDT", "SXTUSDT", "SYNUSDT", "SYRUPUSDT", "TUSDT",
    "TAUSDT", "TACUSDT", "TAGUSDT", "TAIKOUSDT", "TAKEUSDT", "TAOUSDT", "THEUSDT",
    "THETAUSDT", "TIAUSDT", "TLMUSDT", "TNSRUSDT", "TOSHIUSDT", "TOWNSUSDT", "TRADOORUSDT",
    "TRBUSDT", "TREEUSDT", "TRIAUSDT", "TRUMPUSDT", "TRUSTUSDT", "TRUTHUSDT", "TRXUSDT",
    "TSTUSDT", "TURBOUSDT", "TURTLEUSDT", "TUTUSDT", "TWTUSDT", "UAIUSDT", "UBUSDT",
    "UMAUSDT", "UNIUSDT", "USUSDT", "USDCUSDT", "USELESSUSDT", "USTCUSDT", "USUALUSDT",
    "VANAUSDT", "VELODROMEUSDT", "VELVETUSDT", "VETUSDT", "VIRTUALUSDT", "VTHOUSDT",
    "VVVUSDT", "WUSDT", "WALUSDT", "WAXPUSDT", "WCTUSDT", "WETUSDT", "WIFUSDT",
    "WLDUSDT", "WLFIUSDT", "WOOUSDT", "XAIUSDT", "XANUSDT", "XAUTUSDT", "XLMUSDT",
    "XMRUSDT", "XNYUSDT", "XPINUSDT", "XPLUSDT", "XRPUSDT", "XTZUSDT", "XVGUSDT",
    "XVSUSDT", "YBUSDT", "YFIUSDT", "YGGUSDT", "ZAMAUSDT", "ZBTUSDT", "ZECUSDT",
    "ZENUSDT", "ZEREBROUSDT", "ZESTUSDT", "ZETAUSDT", "ZILUSDT", "ZKUSDT", "ZKCUSDT",
    "ZKPUSDT", "ZORAUSDT", "ZROUSDT", "ZRXUSDT"
]

print(f"Total pairs to download: {len(pairs)}")
print("Starting download...")

success_count = 0
error_count = 0
skipped_count = 0

for i, symbol in enumerate(pairs):
    try:
        # Format for Binance futures
        pair = symbol.replace("USDT", "USDT:USDT")
        
        # Get all candles
        all_candles = []
        start_time = int(time.time() - 14*24*60*60*1000)  # 14 days ago
        end_time = int(time.time())
        
        while start_time < end_time:
            url = f"https://fapi.binance.com/fapi/v1/klines?symbol={pair}&interval=5m&startTime={start_time}&endTime={end_time}&limit=1500"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if not data:
                    break
                
                for candle in data:
                    all_candles.append({
                        "timestamp": pd.Timestamp(candle[0], unit="ms"),
                        "open": float(candle[1]),
                        "high": float(candle[2]),
                        "low": float(candle[3]),
                        "close": float(candle[4]),
                        "volume": float(candle[5])
                    })
                
                start_time = data[-1][0] + 1
            else:
                break
        
        if all_candles:
            df = pd.DataFrame(all_candles)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")
            
            # Create directory
            coin_name = symbol.replace("USDT", "")
            dir_path = f"{base_dir}/{coin_name}"
            os.makedirs(dir_path, exist_ok=True)
            
            # Save
            filename = f"{dir_path}/{coin_name}_5m.feather"
            feather.write_feather(df, filename)
            
            print(f"[{i+1}/{len(pairs)}] ✅ {coin_name}: {len(df)} candles")
            success_count += 1
        else:
            print(f"[{i+1}/{len(pairs)}] ⏭️ {coin_name}: No data")
            skipped_count += 1
            
        # Rate limiting
        time.sleep(0.3)
        
    except Exception as e:
        print(f"[{i+1}/{len(pairs)}] ❌ {symbol}: {str(e)[:50]}")
        error_count += 1
        time.sleep(1)

print(f"\n{'='*50}")
print(f"Download complete!")
print(f"✅ Success: {success_count}")
print(f"⏭️ Skipped: {skipped_count}")
print(f"❌ Errors: {error_count}")
print(f"{'='*50}")
