FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

WORKDIR /app
COPY pyproject.toml ./
RUN uv lock && uv sync --no-install-project

COPY . .
RUN chmod +x entrypoint.sh

CMD ["bash", "entrypoint.sh"]
