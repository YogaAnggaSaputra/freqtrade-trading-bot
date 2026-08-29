FROM freqtradeorg/freqtrade:stable
USER root
WORKDIR /freqtrade
COPY configs/freqtrade/strategies/ /freqtrade/configs/strategies/
COPY shared/ /freqtrade/shared/
COPY services/freqtrade-runtime/ /freqtrade/runtime/
CMD ["python", "/freqtrade/runtime/main.py"]
