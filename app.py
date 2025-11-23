import io
import tempfile
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import whisper
import soundfile as sf
import torch

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("large-v3", device=device)

@app.get("/health")
def health():
    return {"status": "ok", "device": device}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()
        data, samplerate = sf.read(io.BytesIO(audio_bytes))

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            sf.write(tmp.name, data, samplerate)
            result = model.transcribe(
                tmp.name,
                language="ar",
                task="transcribe"
            )

        return JSONResponse({"text": result.get("text", "")})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
