from __future__ import annotations

from .api_provider import APIProvider


class StrongGeneralAPIProvider(APIProvider):
    """Explicit R1 provider; it cannot be configured as the financial SFT role."""

    provider_name = "strong_general_api"

    def __init__(self, **kwargs):
        kwargs["provider_role"] = "supervisor"
        kwargs["model_role"] = "strong_general_llm"
        super().__init__(**kwargs)
