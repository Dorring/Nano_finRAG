"""Configuration-driven route-to-generator policy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class RuntimeRouteV1(str, Enum):
    DIRECT_FACT = "DIRECT_FACT"
    CALCULATION = "CALCULATION"
    MULTI_EVIDENCE = "MULTI_EVIDENCE"


@dataclass(frozen=True)
class GeneratorRouteConfigV1:
    primary: str | None
    fallback: str | None = None
    fallback_on_soft_fail: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {"primary": self.primary, "fallback": self.fallback,
                "fallback_on_soft_fail": self.fallback_on_soft_fail}


class GeneratorRoutingPolicyV1:
    def __init__(self, routes: Mapping[str | RuntimeRouteV1, GeneratorRouteConfigV1], *,
                 enabled: bool = True) -> None:
        self._routes = {str(key.value if isinstance(key, RuntimeRouteV1) else key): value
                        for key, value in routes.items()}
        self.enabled = enabled

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "GeneratorRoutingPolicyV1":
        section = config.get("generation", config)
        routing = section.get("routing", {}) if isinstance(section, Mapping) else {}
        routes: dict[str, GeneratorRouteConfigV1] = {}
        for route in RuntimeRouteV1:
            raw = routing.get(route.value, {})
            routes[route.value] = GeneratorRouteConfigV1(
                primary=raw.get("primary") if isinstance(raw, Mapping) else None,
                fallback=raw.get("fallback") if isinstance(raw, Mapping) else None,
                fallback_on_soft_fail=bool(raw.get("fallback_on_soft_fail", True)) if isinstance(raw, Mapping) else True,
            )
        return cls(routes, enabled=True)

    def for_route(self, route: str | RuntimeRouteV1) -> GeneratorRouteConfigV1:
        key = route.value if isinstance(route, RuntimeRouteV1) else str(route)
        return self._routes.get(key, GeneratorRouteConfigV1(None, None))

    def to_dict(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "routing": {key: value.to_dict() for key, value in self._routes.items()}}
