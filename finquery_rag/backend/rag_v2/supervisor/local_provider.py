from __future__ import annotations

from .api_provider import APIProvider


class LocalProvider(APIProvider):
    """Local OpenAI-compatible endpoint adapter; no model is hard-coded."""

    provider_name = "local"
