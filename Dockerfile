FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY feature_extraction.py .
COPY app.py .
COPY model/ ./model/
COPY templates/ ./templates/
ENV PORT=8080
EXPOSE 8080

# gunicorn: production WSGI server, 2 workers is a reasonable default for a
# CPU-bound sklearn model; tune to your instance size / expected QPS.
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--threads", "4", "app:app"]
