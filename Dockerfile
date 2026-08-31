FROM ghcr.io/astral-sh/uv:0.12.3 AS uv

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/app/.venv/bin:$PATH"
WORKDIR /app

COPY --from=uv /uv /uvx /bin/
RUN addgroup --system paperroute && adduser --system --ingroup paperroute paperroute
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY data ./data
COPY docs ./docs
COPY scripts ./scripts
RUN uv sync --locked --no-dev && \
    mkdir -p /app/data/runtime /app/data/papers && \
    chown -R paperroute:paperroute /app

USER paperroute
EXPOSE 8000
VOLUME ["/app/data/runtime", "/app/data/papers"]
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3)"
CMD ["uvicorn", "paperroute.main:app", "--host", "0.0.0.0", "--port", "8000"]
