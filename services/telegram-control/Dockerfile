FROM python:3.11-slim
WORKDIR /app
COPY services/telegram-control/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY services/telegram-control/ .
CMD ["python", "main.py"]
