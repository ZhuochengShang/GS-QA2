#!/usr/bin/env python3
import argparse
import time
import uuid
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer


class ChatMessage(BaseModel):
    role: str
    content: Any


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    temperature: float = 0.0
    max_tokens: int = 512


def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text", item)))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return "" if content is None else str(content)


def build_app(model_id: str, torch_dtype: str, device: str) -> FastAPI:
    app = FastAPI()
    dtype = torch.float16 if torch_dtype == "float16" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    @app.get("/v1/models")
    def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [{"id": model_id, "object": "model", "created": 0, "owned_by": "local"}],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest) -> dict[str, Any]:
        messages = [
            {"role": msg.role, "content": content_to_text(msg.content)}
            for msg in req.messages
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        do_sample = req.temperature > 0
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=req.max_tokens,
                do_sample=do_sample,
                temperature=req.temperature if do_sample else None,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = output_ids[0][inputs["input_ids"].shape[-1]:]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model or model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": int(inputs["input_ids"].shape[-1]),
                "completion_tokens": int(generated.shape[-1]),
                "total_tokens": int(output_ids.shape[-1]),
            },
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a HuggingFace chat model with a small OpenAI-compatible API.")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--torch-dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    app = build_app(args.model, args.torch_dtype, args.device)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
