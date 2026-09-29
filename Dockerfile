FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8005

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

RUN useradd --system --uid 10001 --home-dir /app appuser \
    && mkdir -p /app/cache /app/results/demo_runs \
    && chown -R appuser:appuser /app/cache /app/results

COPY app/ ./app/
COPY scripts/health_check.py ./scripts/health_check.py

USER appuser

EXPOSE 8005

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python scripts/health_check.py || exit 1

CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port \"$PORT\""]
