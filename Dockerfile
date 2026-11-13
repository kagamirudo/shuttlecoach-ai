# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm AS dependencies
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/venv
COPY requirements-lock.txt /tmp/requirements-lock.txt
# CPU wheels avoid downloading CUDA libraries into the Mac/Linux CPU image.
RUN /opt/venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.15.0 torchvision==0.29.2
RUN /opt/venv/bin/pip install --no-deps -r /tmp/requirements-lock.txt && /opt/venv/bin/pip check

FROM python:3.12-slim-bookworm
ARG APP_DIR=/app
ENV PATH="/opt/venv/bin:$PATH" HOME=/tmp/shuttlecoach PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 SHUTTLECOACH_CONTAINER=1
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates libglib2.0-0 libgl1 make && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 1000 shuttlecoach && useradd --uid 1000 --gid 1000 --no-create-home shuttlecoach
COPY --from=dependencies /opt/venv /opt/venv
WORKDIR ${APP_DIR}
COPY --chown=1000:1000 . .
RUN mkdir -p data models results .cache && chown -R 1000:1000 data models results .cache \
    && ln -s /opt/venv .venv && install -m 755 docker/entrypoint.sh /usr/local/bin/shuttlecoach-entrypoint
USER 1000:1000
EXPOSE 8501
HEALTHCHECK --interval=10s --timeout=3s --start-period=120s --retries=6 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health',timeout=2)"
ENTRYPOINT ["shuttlecoach-entrypoint"]
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--server.maxUploadSize=4096", "--browser.gatherUsageStats=false"]
