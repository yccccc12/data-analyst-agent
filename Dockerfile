# One image for both services; docker-compose.yml picks the command.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend ./backend
COPY frontend ./frontend
COPY data ./data

# Build backend/sample.db from the CSV (the .db is gitignored, so it is not in the repo).
RUN python backend/app/scripts/load_csv.py

# The API must be started from backend/ so `app` imports.
WORKDIR /app/backend
EXPOSE 8000 8501
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
