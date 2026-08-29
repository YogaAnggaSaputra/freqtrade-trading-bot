FROM python:3.11-slim
WORKDIR /app
COPY services/capital-allocation/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY shared/ shared/
COPY services/capital-allocation/ .
EXPOSE 8000
CMD ["python", "main.py"]
