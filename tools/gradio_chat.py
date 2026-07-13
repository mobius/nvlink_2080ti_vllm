#!/usr/bin/env python3
"""
极简 Gradio 网页聊天界面，连接本地 vLLM OpenAI API。
"""
import os
from openai import OpenAI
import gradio as gr

# vLLM 服务地址
BASE_URL = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
API_KEY = os.getenv("VLLM_API_KEY", "sk-vllm")
MODEL = os.getenv("VLLM_MODEL", "qwen27b-int4-tqk8v4-256K-mtp3-text-only-cu128")
PORT = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
HOST = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


def chat(message: str, history: list, max_tokens: int, temperature: float):
    messages = []
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": message})

    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
    )

    partial = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        partial += delta
        yield partial


if __name__ == "__main__":
    demo = gr.ChatInterface(
        fn=chat,
        additional_inputs=[
            gr.Slider(64, 4096, value=512, step=64, label="Max tokens"),
            gr.Slider(0.0, 2.0, value=0.0, step=0.1, label="Temperature"),
        ],
        title="Qwen3.6-27B on 2×RTX 2080Ti (vLLM)",
        description=f"Model: `{MODEL}`<br>API: `{BASE_URL}`",
        chatbot=gr.Chatbot(height=600),
    )
    print(f"Starting Gradio chat UI on http://{HOST}:{PORT}")
    demo.queue().launch(
        server_name=HOST,
        server_port=PORT,
        share=False,
        inbrowser=False,
        show_error=True,
    )
