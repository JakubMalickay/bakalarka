"""Chat client supporting Azure AI Foundry (GPT-OSS) and local Ollama backends.

Backend is selected via chatbot/config.yaml:
  backend: azure   →  uses AZURE_AI_ENDPOINT / AZURE_AI_API_KEY from .env
  backend: ollama  →  uses local Ollama OpenAI-compatible API (no key needed)
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import yaml
from dotenv import load_dotenv
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import (
    AzureChatPromptExecutionSettings,
    OpenAIChatCompletion,
)
from semantic_kernel.contents import ChatHistory
from openai import AsyncOpenAI

try:  # Optional reuse of existing settings if available in this repo
    from backend.config.config import settings  # type: ignore
except Exception:  # noqa: BLE001
    settings = None

# Load .env early so environment variables are available when building config.
load_dotenv()

_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

DEFAULT_MODEL_KEY = "gpt-oss-120b"
DEFAULT_API_VERSION = "2024-05-01-preview"


def _load_yaml_config() -> dict:
    if _CONFIG_PATH.exists():
        with _CONFIG_PATH.open(encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    return {}


def load_chatbot_config() -> dict:
    """Return the parsed chatbot config.yaml content."""
    return _load_yaml_config()


# ── Azure config ──────────────────────────────────────────────────────────────

@dataclass
class AzureAIChatConfig:
    endpoint: str
    api_key: str
    deployment: str
    api_version: str

    @classmethod
    def from_env(cls, model_key: str = DEFAULT_MODEL_KEY) -> "AzureAIChatConfig":
        cfg = _load_yaml_config().get("azure", {})

        endpoint = cfg.get("endpoint") or os.environ.get("AZURE_AI_ENDPOINT")
        api_key = cfg.get("api_key") or os.environ.get("AZURE_AI_API_KEY")
        deployment = cfg.get("deployment") or os.environ.get("AZURE_AI_DEPLOYMENT", model_key)
        api_version = cfg.get("api_version") or os.environ.get("AZURE_AI_API_VERSION", DEFAULT_API_VERSION)

        if settings and model_key in settings.azure_models:
            model_meta = settings.azure_models[model_key]
            deployment = model_meta.get("deployment_name", deployment)
            api_version = model_meta.get("api_version", api_version)

        if not endpoint or not api_key:
            raise ValueError("Set AZURE_AI_ENDPOINT and AZURE_AI_API_KEY in .env or chatbot/config.yaml")

        return cls(endpoint=endpoint, api_key=api_key, deployment=deployment, api_version=api_version)


# ── Ollama config ─────────────────────────────────────────────────────────────

@dataclass
class OllamaChatConfig:
    base_url: str = "http://localhost:11434/v1"
    model: str = "llama3"
    temperature: float = 0.1
    max_tokens: int = 4096

    @classmethod
    def from_yaml(cls) -> "OllamaChatConfig":
        cfg = _load_yaml_config().get("ollama", {})
        return cls(
            base_url=cfg.get("base_url", "http://localhost:11434/v1"),
            model=cfg.get("model", "llama3"),
            temperature=float(cfg.get("temperature", 0.1)),
            max_tokens=int(cfg.get("max_tokens", 4096)),
        )


# ── Unified client factory ────────────────────────────────────────────────────

def build_client(temperature: float | None = None, max_tokens: int | None = None) -> "SemanticKernelChatClient":
    """Read config.yaml and return the appropriate chat client."""
    backend = _load_yaml_config().get("backend", "azure").strip().lower()
    if backend == "ollama":
        cfg = OllamaChatConfig.from_yaml()
        return SemanticKernelChatClient(
            cfg,
            temperature=temperature if temperature is not None else cfg.temperature,
            max_tokens=max_tokens if max_tokens is not None else cfg.max_tokens,
        )
    else:
        cfg = AzureAIChatConfig.from_env()
        return SemanticKernelChatClient(
            cfg,
            temperature=temperature if temperature is not None else 0.7,
            max_tokens=max_tokens if max_tokens is not None else 128000,
        )


# ── Chat client ───────────────────────────────────────────────────────────────

class SemanticKernelChatClient:
    """Chat client that works with both Azure AI Foundry and Ollama backends."""

    def __init__(
        self,
        cfg: AzureAIChatConfig | OllamaChatConfig,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> None:
        self.cfg = cfg
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.service_id = "chat"
        self.kernel = Kernel()

        if isinstance(cfg, OllamaChatConfig):
            # Ollama exposes an OpenAI-compatible API — no real key required
            async_client = AsyncOpenAI(api_key="ollama", base_url=cfg.base_url.rstrip("/"))
            model_id = cfg.model
            print(f"[backend] ollama  model={model_id}  url={cfg.base_url}")
        else:
            async_client = AsyncOpenAI(api_key=cfg.api_key, base_url=cfg.endpoint.rstrip("/"))
            model_id = cfg.deployment
            print(f"[backend] azure  deployment={model_id}")

        self.kernel.add_service(
            OpenAIChatCompletion(
                service_id=self.service_id,
                ai_model_id=model_id,
                async_client=async_client,
            )
        )

    async def _achat(self, messages: List[str], system: Optional[str]) -> str:
        chat = self.kernel.get_service(type=OpenAIChatCompletion, service_id=self.service_id)
        history = ChatHistory(system_message=system or "You are a helpful assistant.")
        for msg in messages:
            history.add_user_message(msg)

        exec_settings = AzureChatPromptExecutionSettings(
            temperature=self.temperature, max_tokens=self.max_tokens
        )
        result = await self._call_chat_service(chat, history, exec_settings)
        text = self._extract_text(result)
        return text if text is not None else str(result)

    async def _call_chat_service(self, chat, history, exec_settings):
        fn = getattr(chat, "get_chat_message_contents", None)
        if not callable(fn):
            raise AttributeError("Semantic Kernel chat service lacks get_chat_message_contents")
        result = fn(history, settings=exec_settings)
        if asyncio.iscoroutine(result):
            result = await result
        return result

    @staticmethod
    def _extract_text(messages):
        if not messages:
            return None
        first = messages[0]
        if getattr(first, "text", None):
            return first.text
        for item in getattr(first, "items", None) or []:
            if getattr(item, "text", None):
                return item.text
        content = getattr(first, "content", None)
        if isinstance(content, str) and content:
            return content
        return None

    def chat(self, messages: List[str], system: Optional[str] = None) -> str:
        return asyncio.run(self._achat(messages, system))


if __name__ == "__main__":
    client = build_client(temperature=0.3)
    reply = client.chat(["What is the capital of France?"], system="You are concise.")
    print(reply)




