FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
 && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /models && chmod -R 777 /models

ENV XDG_CACHE_HOME=/models
ENV TORCH_HOME=/models
ENV TRANSFORMERS_CACHE=/models

RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    soundfile \
    openai-whisper \
    python-multipart \
    requests

COPY app.py .

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
