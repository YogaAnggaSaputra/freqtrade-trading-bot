FROM freqtradeorg/freqtrade:stable
USER root
ENV PYTHONPATH=/freqtrade:/home/ftuser/.local/lib/python3.14/site-packages
WORKDIR /freqtrade
ENTRYPOINT []
RUN pip install --no-cache-dir prometheus-client==0.21.1
COPY configs/freqtrade/strategies/ /freqtrade/configs/strategies/
COPY shared/ /freqtrade/shared/
COPY services/freqtrade-runtime/ /freqtrade/runtime/
CMD ["python", "/freqtrade/runtime/main.py"]
