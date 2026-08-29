FROM python:3.11-slim
WORKDIR /app
COPY services/execution-guard/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY shared/ shared/
COPY services/execution-guard/ .
EXPOSE 8000
CMD ["python", "main.py"]
