"""Runtime-only operational metric aggregation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .contracts import TrustedRAGResponseV2


@dataclass
class RuntimeMetricAggregatorV1:
    queries_total: int = 0
    route_distribution: Counter[str] = field(default_factory=Counter)
    trusted_evidence_queries: int = 0
    generator_invocations: int = 0
    primary_pass: int = 0
    fallback_queries: int = 0
    fallback_success: int = 0
    final_releases: int = 0
    final_abstentions: int = 0
    provider_errors: int = 0
    hard_validation_failures: int = 0
    no_answer_queries: int = 0
    no_answer_generator_false_positive: int = 0
    total_attempts: int = 0

    def observe(self, response: TrustedRAGResponseV2) -> None:
        self.queries_total += 1
        if response.route:
            self.route_distribution[response.route] += 1
        if response.trace.trusted_evidence_available:
            self.trusted_evidence_queries += 1
        self.total_attempts += response.attempt_count
        self.generator_invocations += int(response.attempt_count > 0)
        if response.attempt_count and response.trace.validator_codes and not response.trace.validator_codes[0]:
            self.primary_pass += 1
        if response.used_fallback:
            self.fallback_queries += 1
            self.fallback_success += int(response.released)
        self.final_releases += int(response.released)
        self.final_abstentions += int(not response.released)
        self.provider_errors += int(response.terminal_reason.value == "TR5_PROVIDER_ERROR")
        self.hard_validation_failures += int(any(code.startswith("GV") for group in response.trace.validator_codes for code in group))
        if response.terminal_reason.value == "TR7_NO_ANSWER":
            self.no_answer_queries += 1
            self.no_answer_generator_false_positive += int(response.attempt_count > 0)

    def snapshot(self) -> dict[str, Any]:
        total = self.queries_total or 1
        generated = self.generator_invocations or 1
        fallbacks = self.fallback_queries or 1
        return {
            "queries_total": self.queries_total,
            "route_distribution": dict(self.route_distribution),
            "trusted_evidence_rate": self.trusted_evidence_queries / total,
            "generator_invocation_rate": self.generator_invocations / total,
            "primary_pass_rate": self.primary_pass / generated,
            "fallback_rate": self.fallback_queries / total,
            "fallback_success_rate": self.fallback_success / fallbacks,
            "final_release_rate": self.final_releases / total,
            "final_abstention_rate": self.final_abstentions / total,
            "provider_error_rate": self.provider_errors / total,
            "hard_validation_failure_rate": self.hard_validation_failures / total,
            "no_answer_generator_false_positive": self.no_answer_generator_false_positive,
            "average_attempts_per_query": self.total_attempts / total,
            "financial_primary_attempts": None, "financial_primary_pass": None,
            "financial_primary_pass_rate": None, "financial_to_general_fallback": None,
            "financial_fallback_rate": None, "general_fallback_success": None,
            "financial_final_release": None,
        }
