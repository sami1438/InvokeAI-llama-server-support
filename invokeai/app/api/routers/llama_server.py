"""API router for external llama.cpp server integration."""

from __future__ import annotations

import io
import logging
from typing import Optional

from fastapi import HTTPException
from fastapi.routing import APIRouter
from pydantic import BaseModel, Field

from invokeai.app.api.auth_dependencies import AdminUserOrDefault, CurrentUserOrDefault
from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.services.config.config_default import get_config
from invokeai.backend.llama_server.client import (
    LlamaServerClient,
    LlamaServerError,
    LlamaServerUnreachableError,
    LlamaServerVisionError,
)

logger = logging.getLogger(__name__)

llama_server_router = APIRouter(prefix="/v1/llm/external", tags=["llm-external"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class LlamaServerTestConnectionResponse(BaseModel):
    ok: bool = Field(description="Whether the connection was successful")
    models: Optional[list[str]] = Field(default=None, description="Available model names")
    error: Optional[str] = Field(default=None, description="Error message if connection failed")


class LlamaServerExpandPromptRequest(BaseModel):
    prompt: str = Field(description="The prompt to expand")
    system_prompt: Optional[str] = Field(default=None, description="Optional system prompt")
    model: Optional[str] = Field(default=None, description="Model name (uses configured default if omitted)")
    max_tokens: Optional[int] = Field(default=None, ge=1, le=8192, description="Max tokens to generate")
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0, description="Sampling temperature")
    task_id: Optional[str] = Field(default=None, description="Client-supplied task ID for progress events")


class LlamaServerExpandPromptResponse(BaseModel):
    expanded_prompt: str
    error: Optional[str] = None


class LlamaServerImageToPromptRequest(BaseModel):
    image_name: str = Field(description="Name of the image in InvokeAI's image store")
    instruction: str = Field(
        default="Describe this image in detail for use as an AI image generation prompt.",
        description="Instruction to send with the image to the vision model.",
    )
    model: Optional[str] = Field(default=None, description="Vision model name (uses configured default if omitted)")
    max_tokens: Optional[int] = Field(default=None, ge=1, le=8192, description="Max tokens to generate")
    task_id: Optional[str] = Field(default=None, description="Client-supplied task ID for progress events")


class LlamaServerImageToPromptResponse(BaseModel):
    prompt: str
    error: Optional[str] = None


class LlamaServerConfigResponse(BaseModel):
    enabled: bool
    url: Optional[str] = None
    text_model: Optional[str] = None
    vision_model: Optional[str] = None
    timeout: int = 120
    max_tokens: int = 1024
    temperature: float = 0.7
    api_key_configured: bool = False


class LlamaServerConfigUpdateRequest(BaseModel):
    enabled: Optional[bool] = None
    url: Optional[str] = None
    api_key: Optional[str] = None
    text_model: Optional[str] = None
    vision_model: Optional[str] = None
    timeout: Optional[int] = Field(default=None, ge=5, le=600)
    max_tokens: Optional[int] = Field(default=None, ge=1, le=8192)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_client() -> Optional[LlamaServerClient]:
    """Create a LlamaServerClient from the current app config, or None if disabled."""
    config = get_config()
    return LlamaServerClient.from_config(config)


def _get_config_value(field: str):
    return getattr(get_config(), field)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@llama_server_router.get(
    "/config",
    operation_id="get_llama_server_config",
    status_code=200,
    response_model=LlamaServerConfigResponse,
)
async def get_llama_server_config(current_user: CurrentUserOrDefault) -> LlamaServerConfigResponse:
    """Get the current external llama-server configuration."""
    config = get_config()
    return LlamaServerConfigResponse(
        enabled=config.external_llama_enabled,
        url=config.external_llama_url,
        text_model=config.external_llama_text_model,
        vision_model=config.external_llama_vision_model,
        timeout=config.external_llama_timeout,
        max_tokens=config.external_llama_max_tokens,
        temperature=config.external_llama_temperature,
        api_key_configured=bool(config.external_llama_api_key),
    )


@llama_server_router.patch(
    "/config",
    operation_id="update_llama_server_config",
    status_code=200,
    response_model=LlamaServerConfigResponse,
)
async def update_llama_server_config(
    _: AdminUserOrDefault,
    update: LlamaServerConfigUpdateRequest,
) -> LlamaServerConfigResponse:
    """Update the external llama-server configuration."""
    config = get_config()
    update_dict = update.model_dump(exclude_unset=True)

    if not update_dict:
        raise HTTPException(status_code=400, detail="No fields to update")

    # Map frontend field names to config field names
    field_mapping = {
        "url": "external_llama_url",
        "api_key": "external_llama_api_key",
        "text_model": "external_llama_text_model",
        "vision_model": "external_llama_vision_model",
        "timeout": "external_llama_timeout",
        "max_tokens": "external_llama_max_tokens",
        "temperature": "external_llama_temperature",
    }

    config_updates: dict = {}
    for key, value in update_dict.items():
        if key in field_mapping:
            config_updates[field_mapping[key]] = value
        elif key == "enabled":
            config_updates["external_llama_enabled"] = value

    config.update_config(config_updates)

    # Persist to file
    from invokeai.app.services.config.config_default import (
        DefaultInvokeAIAppConfig,
        load_and_migrate_config,
    )

    config_path = config.config_file_path
    if config_path.exists():
        persisted = load_and_migrate_config(config_path)
    else:
        persisted = DefaultInvokeAIAppConfig()
    persisted.update_config(config_updates)
    persisted.write_file(config_path)

    # Log without the API key
    log_safe = {k: v for k, v in config_updates.items() if k != "external_llama_api_key"}
    logger.info("Updated llama-server config: %s", log_safe)

    return LlamaServerConfigResponse(
        enabled=config.external_llama_enabled,
        url=config.external_llama_url,
        text_model=config.external_llama_text_model,
        vision_model=config.external_llama_vision_model,
        timeout=config.external_llama_timeout,
        max_tokens=config.external_llama_max_tokens,
        temperature=config.external_llama_temperature,
        api_key_configured=bool(config.external_llama_api_key),
    )


@llama_server_router.post(
    "/test",
    operation_id="test_llama_server_connection",
    status_code=200,
    response_model=LlamaServerTestConnectionResponse,
)
async def test_llama_server_connection(
    current_user: CurrentUserOrDefault,
) -> LlamaServerTestConnectionResponse:
    """Test the connection to the external llama-server and list available models."""
    client = _get_client()
    if client is None:
        return LlamaServerTestConnectionResponse(
            ok=False,
            error="External llama-server is not enabled or URL is not configured.",
        )

    result = await client.test_connection()
    return LlamaServerTestConnectionResponse(**result)


@llama_server_router.get(
    "/models",
    operation_id="list_llama_server_models",
    status_code=200,
)
async def list_llama_server_models(current_user: CurrentUserOrDefault) -> list[str]:
    """List models available on the external llama-server."""
    client = _get_client()
    if client is None:
        raise HTTPException(
            status_code=400,
            detail="External llama-server is not enabled or URL is not configured.",
        )

    try:
        return await client.list_models()
    except LlamaServerUnreachableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LlamaServerError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@llama_server_router.post(
    "/expand-prompt",
    operation_id="expand_prompt_llama_server",
    status_code=200,
    response_model=LlamaServerExpandPromptResponse,
)
async def expand_prompt_llama_server(
    current_user: CurrentUserOrDefault,
    body: LlamaServerExpandPromptRequest,
) -> LlamaServerExpandPromptResponse:
    """Expand a prompt using the external llama-server text model."""
    config = get_config()
    client = _get_client()
    if client is None:
        raise HTTPException(
            status_code=400,
            detail="External llama-server is not enabled or URL is not configured.",
        )

    model = body.model or config.external_llama_text_model
    if not model:
        raise HTTPException(
            status_code=400,
            detail="No text model configured for external llama-server.",
        )

    events = ApiDependencies.invoker.services.events
    user_id = current_user.user_id
    task_id = body.task_id

    if task_id:
        events.emit_llm_task_progress(
            task_id=task_id, user_id=user_id, phase="generating", message="Generating"
        )

    try:
        max_tokens = body.max_tokens if body.max_tokens is not None else config.external_llama_max_tokens
        temperature = body.temperature if body.temperature is not None else config.external_llama_temperature

        expanded = await client.text_completion(
            prompt=body.prompt,
            model=model,
            system_prompt=body.system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        if task_id:
            events.emit_llm_task_complete(task_id=task_id, user_id=user_id)
        return LlamaServerExpandPromptResponse(expanded_prompt=expanded)

    except LlamaServerUnreachableError as exc:
        if task_id:
            events.emit_llm_task_error(task_id=task_id, user_id=user_id, error=str(exc))
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LlamaServerError as exc:
        if task_id:
            events.emit_llm_task_error(task_id=task_id, user_id=user_id, error=str(exc))
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        if task_id:
            events.emit_llm_task_error(task_id=task_id, user_id=user_id, error=str(exc))
        logger.error("Error expanding prompt via llama-server: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@llama_server_router.post(
    "/image-to-prompt",
    operation_id="image_to_prompt_llama_server",
    status_code=200,
    response_model=LlamaServerImageToPromptResponse,
)
async def image_to_prompt_llama_server(
    current_user: CurrentUserOrDefault,
    body: LlamaServerImageToPromptRequest,
) -> LlamaServerImageToPromptResponse:
    """Generate a prompt from an image using the external llama-server vision model."""
    from invokeai.app.api.routers._access import assert_image_read_access
    from invokeai.app.services.image_files.image_files_common import ImageFileNotFoundException

    config = get_config()
    client = _get_client()
    if client is None:
        raise HTTPException(
            status_code=400,
            detail="External llama-server is not enabled or URL is not configured.",
        )

    model = body.model or config.external_llama_vision_model
    if not model:
        raise HTTPException(
            status_code=400,
            detail="No vision model configured for external llama-server.",
        )

    # Check image access
    assert_image_read_access(body.image_name, current_user)

    events = ApiDependencies.invoker.services.events
    user_id = current_user.user_id
    task_id = body.task_id

    if task_id:
        events.emit_llm_task_progress(
            task_id=task_id, user_id=user_id, phase="generating", message="Generating"
        )

    try:
        # Load image from InvokeAI's image store and convert to bytes
        pil_image = ApiDependencies.invoker.services.images.get_pil_image(body.image_name)
        pil_image = pil_image.convert("RGB")

        buf = io.BytesIO()
        pil_image.save(buf, format="JPEG", quality=85)
        image_bytes = buf.getvalue()

        max_tokens = body.max_tokens if body.max_tokens is not None else config.external_llama_max_tokens

        prompt = await client.vision_completion(
            image_bytes=image_bytes,
            instruction=body.instruction,
            model=model,
            image_format="jpeg",
            max_tokens=max_tokens,
        )

        if task_id:
            events.emit_llm_task_complete(task_id=task_id, user_id=user_id)
        return LlamaServerImageToPromptResponse(prompt=prompt)

    except ImageFileNotFoundException as exc:
        if task_id:
            events.emit_llm_task_error(task_id=task_id, user_id=user_id, error="Image not found")
        raise HTTPException(status_code=404, detail=f"Image '{body.image_name}' not found") from exc
    except LlamaServerVisionError as exc:
        if task_id:
            events.emit_llm_task_error(task_id=task_id, user_id=user_id, error=str(exc))
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except LlamaServerUnreachableError as exc:
        if task_id:
            events.emit_llm_task_error(task_id=task_id, user_id=user_id, error=str(exc))
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LlamaServerError as exc:
        if task_id:
            events.emit_llm_task_error(task_id=task_id, user_id=user_id, error=str(exc))
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        if task_id:
            events.emit_llm_task_error(task_id=task_id, user_id=user_id, error=str(exc))
        logger.error("Error generating prompt from image via llama-server: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
