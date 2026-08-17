"""Reusable HTTP client for communicating with an external llama.cpp server.

Provides both text and vision (multimodal) completions via the OpenAI-compatible
API that llama-server exposes at ``/v1/chat/completions`` and ``/v1/models``.
"""

from __future__ import annotations

import base64
import json
import logging
from typing import Any, Optional
from urllib.parse import urljoin

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom exception hierarchy
# ---------------------------------------------------------------------------


class LlamaServerError(Exception):
    """Base exception for llama-server communication errors."""


class LlamaServerUnreachableError(LlamaServerError):
    """The server could not be reached (connection refused / timeout)."""


class LlamaServerHttpError(LlamaServerError):
    """The server returned a non-2xx HTTP status code."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {detail}")


class LlamaServerInvalidResponseError(LlamaServerError):
    """The server returned an invalid or empty response."""


class LlamaServerVisionError(LlamaServerError):
    """The vision request was rejected or the model lacks vision support."""


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class LlamaServerClient:
    """HTTP client for an OpenAI-compatible llama.cpp server.

    Parameters
    ----------
    base_url:
        Root URL of the llama-server, e.g. ``http://127.0.0.1:8080``.
    api_key:
        Optional bearer token.  Omitted from log output.
    timeout:
        Default request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: int = 120,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout

    # -- helpers -------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    @staticmethod
    def _build_url(base: str, path: str) -> str:
        return urljoin(base if base.endswith("/") else base + "/", path.lstrip("/"))

    # -- public API ----------------------------------------------------------

    async def list_models(self) -> list[str]:
        """GET /v1/models – returns a list of model ID strings."""
        url = self._build_url(self._base_url, "/v1/models")
        logger.debug("Listing models from llama-server: %s", self._base_url)
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(url, headers=self._headers())
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise LlamaServerUnreachableError(
                f"Cannot connect to llama-server at {self._base_url}: {exc}"
            ) from exc

        if resp.status_code != 200:
            raise LlamaServerHttpError(resp.status_code, resp.text)

        try:
            data = resp.json()
        except json.JSONDecodeError as exc:
            raise LlamaServerInvalidResponseError(
                f"Invalid JSON from /v1/models: {exc}"
            ) from exc

        models: list[str] = []
        for obj in data.get("data", []):
            model_id = obj.get("id")
            if model_id:
                models.append(str(model_id))

        if not models:
            raise LlamaServerInvalidResponseError(
                "The llama-server returned an empty model list."
            )

        logger.debug("Available models from llama-server: %s", models)
        return models

    async def test_connection(self) -> dict[str, Any]:
        """Attempt to connect and list models, returning status info.

        Returns a dict like::

            {"ok": True, "models": ["model-a", "model-b"]}

        or::

            {"ok": False, "error": "..."}
        """
        try:
            models = await self.list_models()
            return {"ok": True, "models": models}
        except LlamaServerError as exc:
            return {"ok": False, "error": str(exc)}

    async def text_completion(
        self,
        prompt: str,
        model: str,
        *,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """Send a text-only chat completion request.

        Returns the generated text from ``choices[0].message.content``.
        """
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }

        logger.debug("Sending text completion to llama-server (model=%s)", model)
        return await self._post_chat_completion(body)

    async def vision_completion(
        self,
        image_bytes: bytes,
        instruction: str,
        model: str,
        *,
        image_format: str = "jpeg",
        max_tokens: int = 1024,
    ) -> str:
        """Send a multimodal (vision) chat completion request.

        Parameters
        ----------
        image_bytes:
            Raw image bytes (will be base64-encoded).
        image_format:
            MIME sub-type for the base64 data URI (``"jpeg"`` or ``"png"``).
        instruction:
            Text instruction / prompt to accompany the image.
        model:
            Model identifier reported by ``/v1/models``.
        max_tokens:
            Max tokens to generate.
        """
        b64 = base64.b64encode(image_bytes).decode("ascii")
        data_uri = f"data:image/{image_format};base64,{b64}"

        user_content: list[dict[str, Any]] = [
            {"type": "text", "text": instruction},
            {
                "type": "image_url",
                "image_url": {"url": data_uri},
            },
        ]

        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": user_content}],
            "max_tokens": max_tokens,
            "stream": False,
        }

        logger.debug("Sending vision completion to llama-server (model=%s)", model)
        try:
            return await self._post_chat_completion(body)
        except LlamaServerHttpError as exc:
            # 400, 422, or 500 may indicate the model does not support vision
            if exc.status_code in (400, 422, 500):
                raise LlamaServerVisionError(
                    "The selected llama-server model does not appear to support "
                    "vision/multimodal input. Please ensure a vision-capable "
                    f"model is loaded. Server detail: {exc.detail}"
                ) from exc
            raise

    # -- internal ------------------------------------------------------------

    async def _post_chat_completion(self, body: dict[str, Any]) -> str:
        url = self._build_url(self._base_url, "/v1/chat/completions")
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(url, json=body, headers=self._headers())
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise LlamaServerUnreachableError(
                f"Cannot connect to llama-server at {self._base_url}: {exc}"
            ) from exc
        except httpx.ReadTimeout as exc:
            raise LlamaServerUnreachableError(
                f"Read timeout from llama-server at {self._base_url}: {exc}"
            ) from exc

        if resp.status_code != 200:
            raise LlamaServerHttpError(resp.status_code, resp.text)

        try:
            data = resp.json()
        except json.JSONDecodeError as exc:
            raise LlamaServerInvalidResponseError(
                f"Invalid JSON from /v1/chat/completions: {exc}"
            ) from exc

        # Extract generated text
        choices = data.get("choices")
        if not choices:
            raise LlamaServerInvalidResponseError(
                "The llama-server response contained no 'choices' field."
            )

        content = choices[0].get("message", {}).get("content")
        if not content:
            raise LlamaServerInvalidResponseError(
                "The llama-server returned an empty response."
            )

        logger.debug("Received response from llama-server (content length=%d)", len(content))
        return content

    # -- convenience factory -------------------------------------------------

    @classmethod
    def from_config(cls, config: Any) -> Optional[LlamaServerClient]:
        """Build a client from an ``InvokeAIAppConfig`` instance.

        Returns ``None`` when the external llama-server is disabled.
        """
        if not getattr(config, "external_llama_enabled", False):
            return None
        url = getattr(config, "external_llama_url", None)
        if not url:
            return None
        return cls(
            base_url=url,
            api_key=getattr(config, "external_llama_api_key", None),
            timeout=getattr(config, "external_llama_timeout", 120),
        )
