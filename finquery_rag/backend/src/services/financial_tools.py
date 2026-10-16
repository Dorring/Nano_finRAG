"""Backward-compat shim: deterministic financial primitives have moved.

Phase 3 Commit 3 migrated all primitives to ``src.finance.primitive_tools``
so the finance layer is the canonical home for pure calculation logic. This
module re-exports the public API plus the underscore-prefixed internals that
existing consumers (``evaluation.evaluation``, architecture tests) rely on.

All new code should import from ``src.finance.primitive_tools`` directly.
"""

from src.finance.primitive_tools import (  # noqa: F401
    ToolResult,
    average_values,
    convert_scale,
    debt_ratio,
    difference,
    format_ratio_percent,
    gross_margin,
    growth_rate,
    net_margin,
    parse_financial_number,
    percentage_share,
    sum_values,
    verify_sum,
    compute_cagr,
    valuation_multiple,
    spread,
    median_values,
)
from src.finance.primitive_tools import (  # noqa: F401
    _SCALE_FACTORS,
    _ensure_decimal,
    _infer_scale,
    _quantize,
    _scale_factor,
)
