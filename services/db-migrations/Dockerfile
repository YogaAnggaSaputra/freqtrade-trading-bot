FROM python:3.11-slim
WORKDIR /app
COPY services/db-migrations/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY alembic.ini .
COPY alembic/ alembic/
COPY shared/ shared/
CMD ["alembic", "upgrade", "head"]
