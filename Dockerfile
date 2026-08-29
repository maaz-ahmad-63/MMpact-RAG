FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DATA_DIR=/app/storage

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# RAG storage: uploaded files and local database cache live here.
RUN useradd --create-home --uid 10001 agent \
    && mkdir -p /app/storage \
    && chown -R agent:agent /app/storage
USER agent

# A fresh named volume mounted at /app/storage inherits this ownership, so the
# non-root user can write to it.

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,os,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8000')+'/', timeout=4).status==200 else 1)"

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
