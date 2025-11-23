import os
import torch
import gradio as gr
from transformers import pipeline

MODEL_NAME = "openai/whisper-large-v3"
BATCH_SIZE = 8

device = 0 if torch.cuda.is_available() else "cpu"

pipe = pipeline(
    task="automatic-speech-recognition",
    model=MODEL_NAME,
    chunk_length_s=30,
    device=device,
)

def transcribe(inputs, task):
    if inputs is None:
        raise gr.Error(
            "No audio file submitted! Please upload or record an audio file before submitting your request."
        )
    out = pipe(
        inputs,
        batch_size=BATCH_SIZE,
        generate_kwargs={"task": task},    # "transcribe" or "translate"
        return_timestamps=True,
    )
    return out["text"]

demo = gr.Interface(
    fn=transcribe,
    inputs=[
        gr.Audio(type="filepath", label="Audio"),
        gr.Radio(["transcribe", "translate"], label="Task", value="transcribe"),
    ],
    outputs=gr.Textbox(lines=10, label="output"),
    title="Whisper Large V3: Transcribe Audio",
    description=(
        "Transcribe long-form microphone or audio inputs with the click of a button! "
        f"Demo uses the OpenAI Whisper checkpoint [{MODEL_NAME}] and Transformers."
    ),
    allow_flagging="never",
)

if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port)
