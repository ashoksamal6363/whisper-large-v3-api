FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git ffmpeg gcc g++ libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py /app/app.py

ENV HF_HOME=/tmp/hf_cache
RUN mkdir -p /tmp/hf_cache && chmod -R 777 /tmp/hf_cache

EXPOSE 7860

CMD ["python", "app.py"]
