"""Safe NF-V2-04 R1 entry point.

Stage A is deliberately the only executable stage in this handoff.  It seals
Gold-blind repaired packets and stops before any provider call.
"""

from scripts.evaluation.run_nf_v2_04_r1_stage_a import main


if __name__ == "__main__":
    raise SystemExit(main())
