FROM python:3.11-slim
WORKDIR /app
COPY services/execution-quality/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY shared/ shared/
COPY services/execution-quality/ .
EXPOSE 8000
CMD ["python", "main.py"]
