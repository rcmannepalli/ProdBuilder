"""Ollama Cloud LLM client (OpenAI-compatible) and provider abstraction.

Ollama Cloud (https://ollama.com) exposes an OpenAI-compatible endpoint at
``/v1/chat/completions``. We talk to it directly with httpx so the build loop
works without any heavyweight dependency; ``crew.py`` layers CrewAI on top of
the same configuration when it is installed.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class LLMConfig:
    base_url: str
    api_key: str
    model_map: dict[str, str]

    def model_for(self, role: str) -> str:
        return self.model_map.get(role) or next(iter(self.model_map.values()), "")


def _v1_url(base_url: str) -> str:
    base = (base_url or "").rstrip("/")
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


class OllamaClient:
    """Thin OpenAI-compatible chat client for Ollama Cloud."""

    def __init__(self, cfg: LLMConfig, timeout: float = 180.0):
        self.cfg = cfg
        self.timeout = timeout

    def chat(self, role: str, system: str, user: str,
             temperature: float = 0.2, json_mode: bool = False) -> str:
        model = self.cfg.model_for(role)
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "stream": False,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Content-Type": "application/json"}
        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"
        url = _v1_url(self.cfg.base_url)
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        return data["choices"][0]["message"]["content"]

    def test_connection(self) -> tuple[bool, str]:
        """Return (ok, message). Tries a minimal completion."""
        try:
            out = self.chat(
                "planner",
                "You are a health check. Reply with the single word: ok.",
                "ping",
                temperature=0.0,
            )
            return True, f"Connected. Model responded: {out.strip()[:80]}"
        except httpx.HTTPStatusError as e:
            return False, f"HTTP {e.response.status_code}: {e.response.text[:200]}"
        except Exception as e:  # noqa: BLE001
            return False, f"{type(e).__name__}: {e}"


def extract_json(text: str) -> Any:
    """Best-effort extraction of a JSON object/array from an LLM response."""
    if text is None:
        return None
    text = text.strip()
    # Strip markdown fences.
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    # Find the first balanced { } or [ ].
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == opener:
                depth += 1
            elif text[i] == closer:
                depth -= 1
                if depth == 0:
                    chunk = text[start:i + 1]
                    try:
                        return json.loads(chunk)
                    except Exception:
                        break
    return None
