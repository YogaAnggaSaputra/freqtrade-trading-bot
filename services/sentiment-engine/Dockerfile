FROM python:3.11-slim
WORKDIR /app
COPY services/sentiment-engine/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY shared/ shared/
COPY services/sentiment-engine/ .
EXPOSE 8000
CMD ["python", "main.py"]
