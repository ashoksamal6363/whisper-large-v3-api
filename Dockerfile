FROM python:3.11-slim

WORKDIR /app

ENV PIP_NO_CACHE_DIR=1

# ffmpeg + git if you still need it
RUN apt-get update && apt-get install -y ffmpeg git && rm -rf /var/lib/apt/lists/*

# >>> ADD THESE LINES <<<
ENV HF_HOME=/tmp/hf
ENV TRANSFORMERS_CACHE=/tmp/hf
RUN mkdir -p /tmp/hf && chmod -R 777 /tmp/hf
# <<< END ADD >>>

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 7860
ENV PORT=7860

CMD ["python", "app.py"]
