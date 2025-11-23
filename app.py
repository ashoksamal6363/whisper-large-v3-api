import io
import tempfile
import os
import requests

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import whisper
import soundfile as sf
import torch

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"

# You may have changed model name to "medium" for speed; keep what you use now:
model = whisper.load_model("medium", device=device)  # or "large-v3", "small", etc.

# Llama endpoint + token (set LLAMA_URL / LLAMA_TOKEN as env vars in deployment if needed)
LLAMA_URL = os.getenv(
    "LLAMA_URL",
    "https://redhataillama-31-8b-instruct3-llama-stack.apps.cluster-gltrd.gltrd.sandbox2574.opentlc.com/v1/chat/completions",
)
LLAMA_TOKEN = os.getenv("LLAMA_TOKEN", "")  # optional, if your endpoint needs auth


def translate_to_english(arabic_text: str) -> str:
    if not arabic_text.strip():
        return ""

    payload = {
        "model": "llama-3.1-8b-instruct",  # adjust if your stack expects a specific name
        "messages": [
            {
                "role": "system",
                "content": "You are a translation assistant that translates Arabic into clear English.",
            },
            {
                "role": "user",
                "content": f"Translate this Arabic text to English:\n\n{arabic_text}",
            },
        ],
        "temperature": 0.0,
        "max_tokens": 512,
    }

    headers = {
        "Content-Type": "application/json",
    }
    if LLAMA_TOKEN:
        headers["Authorization"] = f"Bearer {LLAMA_TOKEN}"

    resp = requests.post(LLAMA_URL, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    # OpenAI-style response
    english = data["choices"][0]["message"]["content"].strip()
    return english


@app.get("/health")
def health():
    return {"status": "ok", "device": device}


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()
        data, samplerate = sf.read(io.BytesIO(audio_bytes))

        # Optional: limit to 60 seconds for speed
        max_seconds = 60
        max_samples = int(max_seconds * samplerate)
        if data.shape[0] > max_samples:
            data = data[:max_samples]

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            sf.write(tmp.name, data, samplerate)

            decode_options = dict(
                language="ar",
                task="transcribe",
                temperature=0.0,
                beam_size=1,
                best_of=1,
                condition_on_previous_text=False,
            )

            result = model.transcribe(tmp.name, **decode_options)

        arabic_text = result.get("text", "").strip()

        # Call Llama to translate
        english_text = ""
        try:
            english_text = translate_to_english(arabic_text)
        except Exception as e:
            # If translation fails, still return Arabic
            english_text = ""
            return JSONResponse(
                {"arabic_text": arabic_text, "english_text": english_text, "translation_error": str(e)},
                status_code=200,
            )

        return JSONResponse(
            {
                "arabic_text": arabic_text,
                "english_text": english_text,
            }
        )

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
