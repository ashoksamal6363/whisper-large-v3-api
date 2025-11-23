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

BASE_URL = "https://redhataillama-31-8b-instruct3-llama-stack.apps.cluster-gltrd.gltrd.sandbox2574.opentlc.com"
MODEL_ENDPOINT = f"{BASE_URL}/v1/chat/completions"
TOKEN = "sha256~5DlvevZJury0P0CMJddlK2yNPgt9Qq9lSTmrLr7EJ0w"   # replace with new token

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {TOKEN}"
}

def translate_to_english(arabic_text: str) -> str:
    """Call Llama endpoint (OpenAI chat-compatible) to translate Arabic text to English."""
    if not arabic_text.strip():
        return ""

    payload = {
        "model": "redhataillama-31-8b-instruct3",

        "messages": [
            {
                "role": "system",
                "content": "You are a translation engine. You translate ONLY from Arabic to English. You never reply in Arabic."
            },
            {
                "role": "user",
                "content": f"Translate to English:\n\n{arabic_text}"
            }
        ],

        "temperature": 0
    }

    resp = requests.post(MODEL_ENDPOINT, headers=headers, data=json.dumps(payload))

    # For debugging 4xx errors, don't just raise blindly
    if resp.status_code >= 400:
        # return the raw Llama error text so you can see the true reason
        raise RuntimeError(f"Llama error {resp.status_code}: {resp.text}")

    data = resp.json()
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
