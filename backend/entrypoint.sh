#!/bin/bash
set -e

echo "Waiting for database..."
while ! uv run python -c "import psycopg; psycopg.connect('$DATABASE_URL')" 2>/dev/null; do
    sleep 1
done
echo "Database ready."

echo "Running migrations..."
uv run python manage.py migrate --noinput

echo "Starting server..."
exec uv run uvicorn config.asgi:application --host 0.0.0.0 --port 8000 --reload
