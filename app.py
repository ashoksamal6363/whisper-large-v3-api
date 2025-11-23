import gradio as gr
from transformers import pipeline
import torch

# Force HF cache into writable directory for OpenShift
import os
os.environ["HF_HOME"] = "/tmp/hf_cache"

device = 0 if torch.cuda.is_available() else "cpu"
pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3",
    torch_dtype=torch.float16 if device != "cpu" else torch.float32,
    device=device
)

def transcribe(audio):
    text = pipe(audio)["text"]
    return text

with gr.Blocks() as demo:
    gr.Markdown("# Whisper Large V3 – Transcribe Audio")
    audio_input = gr.Audio(type="filepath", label="Upload Audio")
    output_text = gr.Textbox(label="Transcription")
    btn = gr.Button("Transcribe")
    btn.click(fn=transcribe, inputs=audio_input, outputs=output_text)

# Expose UI & API
app = demo
fastapi_app = demo.server_app

@fastapi_app.post("/transcribe")
async def api_transcribe(file: bytes):
    import tempfile
    import uuid

    tmp = f"/tmp/{uuid.uuid4()}.wav"
    with open(tmp, "wb") as f:
        f.write(file)

    result = pipe(tmp)["text"]
    return {"text": result}
