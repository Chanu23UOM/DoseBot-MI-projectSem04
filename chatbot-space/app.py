"""
DoseBot AI Medical Chatbot — Hugging Face Space
=================================================
A lightweight Gradio endpoint that proxies chat requests to an open-source
medical LLM via the Hugging Face Inference Providers API (free tier).

Deploy this as a NEW Hugging Face Space (SDK: Gradio, Hardware: CPU basic).
Set your HF token as a Space Secret named HF_TOKEN.

The frontend (webapp/app.js) calls this Space's /predict endpoint using
the @gradio/client library — identical pattern to the OCR Space.
"""

import os
import json
import gradio as gr
from huggingface_hub import InferenceClient

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Models that are actively supported by HF Inference Providers (free tier).
# These are routed through partner providers (Together, Fireworks, etc.)
MODEL_CHOICES = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-2-2b-it",
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
]

# Medical system prompt
SYSTEM_PROMPT = """You are DoseBot AI, a helpful medical assistant for a smart medication dispensing kiosk in rural Sri Lanka. Your role is to:

1. Answer questions about common medications (dosage, side effects, interactions, storage).
2. Provide general health guidance for common conditions (fever, headache, cold, diabetes management, etc.).
3. Help patients understand their prescriptions.
4. Advise on proper medication storage conditions (temperature, humidity — relevant to the DoseBot kiosk).

Important rules:
- Always recommend consulting a doctor for serious symptoms or before changing medication.
- Never diagnose conditions — only provide general educational information.
- Be concise but thorough. Use simple, clear language.
- If asked about something outside medical/health topics, politely redirect.
- When discussing dosage, always mention that the patient should follow their doctor's prescribed dosage.
- Always respond in English by default. Only respond in Sinhala if the user writes in Sinhala or explicitly asks for Sinhala.
"""


def chat(message: str, history: str = "") -> str:
    """
    Main chat endpoint.

    Parameters
    ----------
    message : str
        The latest user message.
    history : str
        JSON-encoded list of prior messages [{role, content}, ...].

    Returns
    -------
    str
        The assistant's reply.
    """
    # Build messages list
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Parse conversation history
    if history:
        try:
            past = json.loads(history) if isinstance(history, str) else history
            if isinstance(past, list):
                for msg in past[-10:]:
                    if isinstance(msg, dict) and "role" in msg and "content" in msg:
                        messages.append({
                            "role": msg["role"],
                            "content": msg["content"]
                        })
        except (json.JSONDecodeError, TypeError):
            pass

    messages.append({"role": "user", "content": message})

    # Create client with token
    client = InferenceClient(api_key=HF_TOKEN if HF_TOKEN else None)

    # Try each model in the fallback chain
    last_error = None
    for model_id in MODEL_CHOICES:
        try:
            completion = client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=512,
                temperature=0.7,
                top_p=0.9,
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            last_error = e
            continue

    return (
        f"I'm sorry, I couldn't connect to any AI model right now. "
        f"This usually means the free Hugging Face servers are busy. "
        f"Please try again in a moment.\n\n"
        f"(Technical detail: {str(last_error)[:200]})"
    )


# ---------------------------------------------------------------------------
# Gradio Interface
# ---------------------------------------------------------------------------
demo = gr.Interface(
    fn=chat,
    inputs=[
        gr.Textbox(label="Message", placeholder="Ask about medications, dosage, side effects..."),
        gr.Textbox(label="History (JSON)", visible=False),
    ],
    outputs=gr.Textbox(label="Reply"),
    title="DoseBot AI Medical Chatbot",
    description=(
        "Ask about medications, dosage, side effects, drug interactions, "
        "or general health questions. Powered by open-source LLMs via "
        "the Hugging Face Inference API."
    ),
    examples=[
        ["What are the common side effects of Paracetamol?", ""],
        ["What is the recommended dosage of Amoxicillin for adults?", ""],
        ["How should insulin be stored?", ""],
        ["Can I take ibuprofen with blood pressure medication?", ""],
    ],
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
