FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    AUTOANNOTATION_HOST=0.0.0.0 \
    AUTOANNOTATION_PORT=8010 \
    AUTOANNOTATION_CONFIG=/app/config/runtime.json

WORKDIR /app

COPY requirements-docker.txt /app/requirements-docker.txt
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --retries 20 --index-url https://download.pytorch.org/whl/cpu torch==2.11.0 torchvision==0.26.0 && \
    pip install --retries 20 -r /app/requirements-docker.txt && \
    pip install --retries 20 --no-deps ultralytics==8.4.31

COPY app /app/app
COPY front /app/front
COPY config /app/config
COPY README.md /app/README.md

EXPOSE 8010

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8010"]
