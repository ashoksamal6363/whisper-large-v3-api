import os
import torch
import gradio as gr
from transformers import pipeline

# Hugging Face cache to writable dir
os.environ["HF_HOME"] = "/tmp/hf_cache"

MODEL_NAME = "openai/whisper-large-v3"

device = 0 if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device != "cpu" else torch.float32

pipe = pipeline(
    "automatic-speech-recognition",
    model=MODEL_NAME,
    torch_dtype=torch_dtype,
    device=device,
    chunk_length_s=30,
)

def transcribe(audio):
    if audio is None:
        return ""
    out = pipe(audio)
    return out["text"]

with gr.Blocks() as demo:
    gr.Markdown("# Whisper Large V3 – Transcribe Audio")
    audio_in = gr.Audio(type="filepath", label="Upload audio")
    task_out = gr.Textbox(label="Transcription / Translation", lines=8)
    btn = gr.Button("Transcribe")
    btn.click(fn=transcribe, inputs=audio_in, outputs=task_out)

if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port)
