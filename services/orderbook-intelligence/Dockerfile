FROM python:3.11-slim
WORKDIR /app
COPY services/orderbook-intelligence/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY shared/ shared/
COPY services/orderbook-intelligence/ .
EXPOSE 8000
CMD ["python", "main.py"]
