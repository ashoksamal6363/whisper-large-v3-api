FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# System dependencies: ffmpeg + sound libs + git
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
 && rm -rf /var/lib/apt/lists/*

# Python dependencies
# (torch already comes with the base image, so we don't reinstall it)
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    soundfile \
    openai-whisper

# If you prefer using requirements.txt instead, comment the above RUN and do:
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

ENV HF_HOME=/models
ENV TRANSFORMERS_CACHE=/models

# Optional NVIDIA runtime hints (OpenShift GPU Operator usually handles this)
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
