FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    RANK_V11_HOST=0.0.0.0 \
    RANK_V11_PORT=8765 \
    RANK_V11_DEVICE=auto \
    RANK_V11_CACHE_DIR=/tmp/rank_v11_cache

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . ./

EXPOSE 8765
HEALTHCHECK --interval=5s --timeout=3s --start-period=120s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/health', timeout=2)" || exit 1

CMD ["python", "server.py", "--host", "0.0.0.0", "--port", "8765", "--device", "auto", "--mode", "accurate"]
