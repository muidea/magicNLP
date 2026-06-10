FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    HF_ENDPOINT=https://hf-mirror.com \
    EMBED_MODEL=intfloat/multilingual-e5-small \
    LOCAL_FILES_ONLY=false \
    EMBED_DEFAULT_INPUT_TYPE=passage \
    EMBED_QUERY_PREFIX="query: " \
    EMBED_PASSAGE_PREFIX="passage: "

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY nlpAPI/ /app/

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=3).read()" || exit 1

CMD ["gunicorn", "-b", "0.0.0.0:8080", "--workers", "1", "--threads", "4", "--timeout", "300", "embed_server:app"]
