import io
import os
import tempfile

import requests
import soundfile as sf
import torch
import whisper
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI()

# --------- Whisper model (speech -> Arabic text) ----------

device = "cuda" if torch.cuda.is_available() else "cpu"

WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "medium")  # change to large-v3 if you want
model = whisper.load_model(WHISPER_MODEL_NAME, device=device)

# --------- Llama endpoint config (Arabic -> English) ----------

LLAMA_URL = os.getenv(
    "LLAMA_URL",
    "https://redhataillama-31-8b-instruct3-llama-stack.apps.cluster-gltrd.gltrd.sandbox2574.opentlc.com/v1/chat/completions",
)

LLAMA_TOKEN = os.getenv("LLAMA_TOKEN", "")
LLAMA_MODEL = os.getenv("LLAMA_MODEL", "redhataillama-31-8b-instruct3")



def translate_to_english(arabic_text: str) -> str:
    """Call Llama endpoint to translate Arabic text to English."""
    if not arabic_text.strip():
        return ""

    payload = {
        "model": LLAMA_MODEL,
        "input": f"Translate this Arabic text to English:\n\n{arabic_text}",
        "parameters": {
            "temperature": 0.0,
            "max_new_tokens": 512
        }
    }

    # Adjust headers to match how your Llama endpoint is secured.
    headers = {
        "Content-Type": "application/json",
    }

    # If your endpoint uses Bearer token auth
    if LLAMA_TOKEN:
        headers["Authorization"] = f"Bearer {LLAMA_TOKEN}"

    resp = requests.post(LLAMA_URL, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    # OpenAI-style response structure
    english = data["choices"][0]["message"]["content"].strip()
    return english


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": device,
        "whisper_model": WHISPER_MODEL_NAME,
        "llama_url": LLAMA_URL,
        "llama_model": LLAMA_MODEL,
    }


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    try:
        # Read audio bytes
        audio_bytes = await file.read()
        data, samplerate = sf.read(io.BytesIO(audio_bytes))

        # Optional: limit to 60 seconds for latency control
        max_seconds = 60
        max_samples = int(max_seconds * samplerate)
        if data.shape[0] > max_samples:
            data = data[:max_samples]

        # Write to temp wav (Whisper expects a file path)
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

        # Now call Llama to get English translation
        translation_error = None
        english_text = ""
        try:
            english_text = translate_to_english(arabic_text)
        except Exception as e:
            # Do not fail the whole request if translation fails; just return error info
            translation_error = str(e)

        response = {
            "arabic_text": arabic_text,
            "english_text": english_text,
        }

        if translation_error is not None:
            response["translation_error"] = translation_error

        return JSONResponse(response)

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
