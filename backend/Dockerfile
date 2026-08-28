FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# git: the ingester clones the transcript corpus at runtime rather than
# vendoring 26MB of third-party content into this repository.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

COPY requirements.txt ./
# pip's default 15s read timeout is too short for large wheels (onnxruntime,
# numpy) on a slow or congested connection -- an evaluator on imperfect
# network shouldn't see a spurious build failure over it.
RUN pip install --default-timeout=120 --retries 5 -r requirements.txt

COPY alembic.ini ./
COPY alembic ./alembic
COPY app ./app
COPY entrypoint.sh ./
RUN chmod +x entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["./entrypoint.sh"]
