from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class Completion:
    content: str
    reasoning: str
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None


class CompletionClient(Protocol):
    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        enable_thinking: bool,
    ) -> Completion: ...


class TransformersClient:
    def __init__(self, base_model: Path, adapter: Path, *, device: str = "cuda") -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required but no GPU is visible")
        self.torch = torch
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            adapter,
            local_files_only=True,
            trust_remote_code=False,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            local_files_only=True,
            trust_remote_code=False,
            attn_implementation="sdpa",
        )
        self.model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
        self.model.to(self.device)
        self.model.eval()
        self.model.generation_config.do_sample = False
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.generation_config.top_k = None

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        enable_thinking: bool,
    ) -> Completion:
        rendered = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        inputs = self.tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        with self.torch.inference_mode():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
        generated = output_ids[0, input_ids.shape[1] :]
        content = self.tokenizer.decode(generated, skip_special_tokens=True)
        eos = self.tokenizer.eos_token_id
        stopped = bool(generated.numel()) and eos is not None and int(generated[-1]) == eos
        return Completion(
            content=content,
            reasoning="",
            finish_reason="stop" if stopped else "length",
            prompt_tokens=int(input_ids.shape[1]),
            completion_tokens=int(generated.shape[0]),
        )


class VLLMClient:
    def __init__(
        self,
        endpoint: str,
        model: str,
        *,
        timeout: int = 300,
        retries: int = 2,
    ) -> None:
        self.endpoint = endpoint
        self.model = model
        self.timeout = timeout
        self.retries = retries

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        enable_thinking: bool,
    ) -> Completion:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    obj: dict[str, Any] = json.loads(response.read().decode("utf-8"))
                choice = obj["choices"][0]
                message = choice.get("message") or {}
                usage = obj.get("usage") or {}
                return Completion(
                    content=str(message.get("content") or ""),
                    reasoning=str(message.get("reasoning_content") or ""),
                    finish_reason=choice.get("finish_reason"),
                    prompt_tokens=usage.get("prompt_tokens"),
                    completion_tokens=usage.get("completion_tokens"),
                )
            except (OSError, ValueError, KeyError, urllib.error.HTTPError) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(1.0 + attempt)
        raise RuntimeError(f"local model request failed: {last_error}")
