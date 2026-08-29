FROM python:3.11-slim
WORKDIR /app
COPY services/post-exit-regret/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY shared/ shared/
COPY services/post-exit-regret/ .
COPY scripts/post_exit_regret_worker.py /app/post_exit_regret_worker.py
EXPOSE 8000
CMD ["python", "main.py"]
