FROM python:3.11-slim

WORKDIR /app
ENV PIP_NO_CACHE_DIR=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git gcc g++ libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Writable cache dirs
ENV HF_HOME=/tmp/hf_cache
ENV TRANSFORMERS_CACHE=/tmp/hf_cache
ENV MPLCONFIGDIR=/tmp/matplotlib
RUN mkdir -p /tmp/hf_cache /tmp/matplotlib && chmod -R 777 /tmp/hf_cache /tmp/matplotlib

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 7860
ENV PORT=7860

CMD ["python", "app.py"]
