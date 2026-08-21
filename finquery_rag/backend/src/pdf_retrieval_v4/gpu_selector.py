"""Read-only NVIDIA GPU discovery and deterministic selection."""

from __future__ import annotations

import csv
import datetime as dt
import io
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any

MIB_PER_GIB = 1024
TIER1_MIN_FREE_MIB = 24 * MIB_PER_GIB
TIER2_MIN_FREE_MIB = 16 * MIB_PER_GIB
SHARED_MAX_UTILIZATION_PCT = 20


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _run_smi(args: list[str]) -> str:
    proc = subprocess.run(
        ["nvidia-smi", *args], check=False, capture_output=True, text=True
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"nvidia_smi_failed:{(proc.stderr or proc.stdout).strip()[:300]}"
        )
    return proc.stdout.strip()


def _int_or_none(value: str) -> int | None:
    value = value.strip()
    if not value or value.upper() in {"N/A", "NA", "NOT SUPPORTED"}:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


@dataclass
class GPURecord:
    physical_index: int
    uuid: str
    name: str
    total_vram_mib: int | None
    used_vram_mib: int | None
    free_vram_mib: int | None
    utilization_pct: int | None
    memory_utilization_pct: int | None
    temperature_c: int | None
    active_processes: list[dict[str, Any]] = field(default_factory=list)

    @property
    def active_compute_process_count(self) -> int:
        return len(self.active_processes)

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["active_compute_process_count"] = self.active_compute_process_count
        return data


def discover_gpus() -> dict[str, Any]:
    gpu_text = _run_smi(
        [
            "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,"
            "utilization.gpu,utilization.memory,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    process_text = _run_smi(
        [
            "--query-compute-apps=gpu_uuid,pid,used_memory,process_name",
            "--format=csv,noheader,nounits",
        ]
    )
    processes_by_uuid: dict[str, list[dict[str, Any]]] = {}
    for row in csv.reader(io.StringIO(process_text)) if process_text else []:
        if len(row) < 4:
            continue
        uuid, pid, used_memory, process_name = (part.strip() for part in row[:4])
        processes_by_uuid.setdefault(uuid, []).append(
            {
                "pid": _int_or_none(pid),
                "used_vram_mib": _int_or_none(used_memory),
                "process_name": process_name,
            }
        )
    records: list[GPURecord] = []
    for row in csv.reader(io.StringIO(gpu_text)):
        if len(row) < 9:
            continue
        values = [part.strip() for part in row[:9]]
        records.append(
            GPURecord(
                physical_index=_int_or_none(values[0]) or 0,
                uuid=values[1],
                name=values[2],
                total_vram_mib=_int_or_none(values[3]),
                used_vram_mib=_int_or_none(values[4]),
                free_vram_mib=_int_or_none(values[5]),
                utilization_pct=_int_or_none(values[6]),
                memory_utilization_pct=_int_or_none(values[7]),
                temperature_c=_int_or_none(values[8]),
                active_processes=processes_by_uuid.get(values[1], []),
            )
        )
    records.sort(key=lambda item: item.physical_index)
    return {
        "captured_at_utc": _utc_now(),
        "query_tool": "nvidia-smi",
        "gpus": [record.as_dict() for record in records],
    }


def _record_from_dict(item: dict[str, Any]) -> GPURecord:
    fields = {
        "physical_index",
        "uuid",
        "name",
        "total_vram_mib",
        "used_vram_mib",
        "free_vram_mib",
        "utilization_pct",
        "memory_utilization_pct",
        "temperature_c",
        "active_processes",
    }
    return GPURecord(**{key: item.get(key) for key in fields})


def _candidate(record: GPURecord, tier: int) -> bool:
    free = record.free_vram_mib or 0
    if tier == 1:
        return record.active_compute_process_count == 0 and free >= TIER1_MIN_FREE_MIB
    if tier == 2:
        return record.active_compute_process_count == 0 and free >= TIER2_MIN_FREE_MIB
    return (
        record.active_compute_process_count > 0
        and free >= TIER1_MIN_FREE_MIB
        and (record.utilization_pct or 100) <= SHARED_MAX_UTILIZATION_PCT
    )


def select_gpu(snapshot: dict[str, Any]) -> dict[str, Any]:
    records = [_record_from_dict(item) for item in snapshot.get("gpus", [])]
    for tier in (1, 2, 3):
        candidates = [record for record in records if _candidate(record, tier)]
        if not candidates:
            continue
        selected = sorted(
            candidates,
            key=lambda item: (
                -(item.free_vram_mib or 0),
                item.utilization_pct if item.utilization_pct is not None else 100,
                item.physical_index,
            ),
        )[0]
        reason = {
            1: "process_free_and_free_vram_at_least_24_gib",
            2: "process_free_and_free_vram_at_least_16_gib",
            3: "shared_gpu_low_utilization_and_free_vram_at_least_24_gib",
        }[tier]
        return {
            "selection_timestamp_utc": snapshot.get("captured_at_utc"),
            "selected_physical_gpu": selected.physical_index,
            "gpu_uuid": selected.uuid,
            "gpu_name": selected.name,
            "free_vram_mib_before": selected.free_vram_mib,
            "utilization_pct_before": selected.utilization_pct,
            "active_processes_before": selected.active_processes,
            "selection_tier": tier,
            "selection_reason": reason,
            "cuda_visible_devices": str(selected.physical_index),
            "visible_logical_gpu": "cuda:0",
        }
    return {
        "selection_timestamp_utc": snapshot.get("captured_at_utc"),
        "selected_physical_gpu": None,
        "selection_tier": None,
        "selection_reason": "no_gpu_meets_safety_policy",
        "cuda_visible_devices": None,
        "visible_logical_gpu": "cuda:0",
    }


def selected_gpu_is_still_eligible(
    snapshot: dict[str, Any], selection: dict[str, Any]
) -> bool:
    physical = selection.get("selected_physical_gpu")
    tier = selection.get("selection_tier")
    if physical is None or tier is None:
        return False
    for item in snapshot.get("gpus", []):
        if int(item.get("physical_index", -1)) == int(physical):
            return _candidate(_record_from_dict(item), int(tier))
    return False
