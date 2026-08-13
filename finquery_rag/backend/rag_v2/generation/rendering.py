"""Narrow renderer seam for future grounded generation views."""

from __future__ import annotations

from typing import Mapping, Protocol

from .contracts import GenerationInputV1


class GenerationInputRendererV1(Protocol):
    renderer_id: str

    def render(self, packet: Mapping[str, object]) -> GenerationInputV1: ...


class GenericVerifiedPacketRendererV1:
    renderer_id = "generic_packet_renderer_v1"

    def render(self, packet: Mapping[str, object]) -> GenerationInputV1:
        query_id = str(packet.get("query_id", ""))
        return GenerationInputV1(query_id=query_id, route=str(packet.get("route", "")),
                                 question=str(packet.get("question", "")), packet=packet,
                                 renderer_id=self.renderer_id)
