FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY grok_pool.py telegram_bot.py push_scheduler.py server.py ./

RUN mkdir -p /app/data && chmod -R 777 /app/data

ENV DATA_DIR=/app/data

EXPOSE 10000

CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-10000}"]
