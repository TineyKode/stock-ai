FROM python:3.11-slim

WORKDIR /app

# Torch, TensorFlow, and the sentiment model are not in this image.
# Weekly retrain stays on GitHub Actions. This container only serves signals.
COPY requirements-dashboard.txt .
RUN pip install --no-cache-dir -r requirements-dashboard.txt

COPY . .

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--browser.gatherUsageStats=false"]
