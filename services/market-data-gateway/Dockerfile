FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Build context = root project (lihat docker-compose.yml)
COPY services/market-data-gateway/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Shared library + kode service
COPY shared/ shared/
COPY services/market-data-gateway/ .

EXPOSE 8000

CMD ["python", "main.py"]
