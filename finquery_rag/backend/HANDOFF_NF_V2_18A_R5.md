# NF-V2-18A-R5 Candidate Ranking Recovery — Execution Handoff

Status: handoff prepared only. No R5 execution, model inference, benchmark replay, embedding build, download, or production change was performed while preparing this document.

## Immutable starting point

- Project: `nano_finance / finquery_rag`
- Branch: `exp/nf-v2-18-retrieval-recovery`
- Expected and inspected HEAD: `2a4b870ff6c64d37417879d9ed4434da28653372`
- Production: `V1`; production switch: `false`
- Development set: `CONSUMED_DEVELOPMENT_REGRESSION` (120 questions: 105 answerable, 15 unanswerable; GOOGL/AMZN). It may be tuned in R5, but must never again be called fresh-blind.
- NF-V2-17 B3 fresh-blind outputs, questions, Gold, and references remain immutable.

R5 is the next task: freeze the A4 Top-200 candidate universe, then recover Top-5/Top-10 ordering with generic structured features and route-specific ranking. Do not change the generator, validators, calculator arithmetic, authorization policy, temporal policy, or production defaults.

## Remote Access / SSH

The remote command was verified from the current client:

```text
ssh mxf@10.157.195.124
```

`ssh -G` reports user `mxf`, host `10.157.195.124`, port `22`; no usable alias, ProxyJump, or ProxyCommand was discoverable. Do not invent an alias. The remote host reports:

```text
hostname: amax-Rack-Server
whoami:   mxf
mount:    /mnt/disk/mxf
```

The existing SSH/user environment supplies credentials; no key, password, token, cookie, or API credential is recorded here. A harmless SSH warning about local port-forward listen port `5566` appeared during inspection; remote commands still completed. No port forwarding is required for R5.

Recommended entry:

```bash
ssh mxf@10.157.195.124
cd /mnt/disk/mxf/projects/Qhhhhhhaaa/nanochat/.worktrees/nf-v2-18-retrieval-recovery
```

## Project paths and ownership

| Path | Exists | Purpose | Git / mutation policy |
|---|---:|---|---|
| `/mnt/disk/mxf/projects/Qhhhhhhaaa/nanochat/.worktrees/nf-v2-18-retrieval-recovery` | yes | actual R5 worktree/repository root | tracked; modify only R5 handoff/implementation files |
| `/mnt/disk/mxf/projects/Qhhhhhhaaa/nanochat/.worktrees/nf-v2-18-retrieval-recovery/finquery_rag/backend` | yes | backend root | tracked; R5 code and artifacts live here |
| `/mnt/disk/mxf/projects/Qhhhhhhaaa/nanochat/.worktrees/nf-v2-18-retrieval-recovery/finquery_rag/backend/artifacts/evaluation` | yes | evaluation artifacts | tracked small manifests/reports; do not commit large caches |
| `/mnt/disk/mxf/projects/Qhhhhhhaaa/financial_corpus_v2` | yes | frozen Corpus V2 source/parsed data | external; read-only for R5 |
| `/mnt/disk/mxf/projects/Qhhhhhhaaa/financial_corpus_v2/indexes/nf-v2-18-retrieval-recovery/r4-qwen3-embedding` | yes | R4 external Qwen dense index | external; do not rebuild or overwrite |
| `/mnt/disk/mxf/.cache/huggingface/hub` | yes | Hugging Face model cache | external; no downloads in handoff; do not commit |
| `/mnt/disk/mxf/.cache/huggingface/hub/models--Qwen--Qwen3-Reranker-4B/snapshots/22e683669bc0f0bd69640a1354a6d0aebcfeede5` | yes | exact R3P0 reranker snapshot | external immutable snapshot |
| `/mnt/disk/mxf/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-4B/snapshots/5cf2132abc99cad020ac570b19d031efec650f2b` | yes | R4 embedding snapshot | external; R5 does not select it |
| `/mnt/disk/mxf/projects/Qhhhhhhaaa/nanochat/finquery_rag/.worktrees/pdf-sr-v2/finquery_rag/backend` | yes | historical location checked during audit | do not use as the current worktree |

## Git state

Remote: `origin https://github.com/Dorring/Nano_finRAG.git` (fetch/push). The inspected worktree was clean, had no submodules, and had no untracked files. Do not reset, rebase, cherry-pick, or overwrite another worktree.

Relevant history, newest first:

```text
2a4b870ff6c64d37417879d9ed4434da28653372 feat(finquery): add R4 strong first-stage embedding diagnostics
f34934b85b70ced100457f0b6c455bf8fed67572 feat(finquery): add A4-preserving hierarchical Qwen retrieval
29103493ff6dfd07c470444424a11dd0d639dd32 feat(nf-v2-18): recover qwen reranker runtime
f68fcc9e6894ee2f6e9a3b474c639b3870b9e680 feat(nf-v2-18): adapt HTML tables to shared semantics
2c281727154b8402df226d83623de7e798e48339 audit(nf-v2-18): reuse existing table semantics
d1dc63fdb666959b0f57a4ce8786a579ecc8be8e exp(nf-v2-18): fine evidence recovery audit
122a96b302c8c53c71eb1185b9df86a7103567e7 fix(nf-v2-18): finalize selected config and latency audit
149b0887a429ae78d4c9c2c97beb8954955e5b8e exp(nf-v2-18): recover financial retrieval candidates
820e754dcb57aef00bca5c2324bf5533815b6753 forensics(nf-v2-17): audit fresh-blind retrieval failures
```

## Environment inventory

Use the normal backend environment for CPU/SQLite evaluation and tests:

```text
/mnt/disk/mxf/anaconda3/bin/python
Python 3.12.2
```

Use QhChat only for Qwen inference:

```text
/mnt/disk/mxf/anaconda3/envs/QhChat/bin/python
Python 3.12.12
PyTorch 2.9.1+cu128; torch CUDA 12.8
Transformers 4.57.1
Hugging Face Hub 0.36.0
Tokenizers 0.22.2
safetensors 0.7.0
Accelerate 1.11.0
```

Do not mutate either environment merely to run R5. The R3P0 recovery explicitly used the current QhChat versions and did not require upgrading Transformers.

## GPU execution policy

The read-only utility is `finquery_rag/backend/src/pdf_retrieval_v4/gpu_selector.py`:

- `discover_gpus()` queries GPU and compute-process state with `nvidia-smi`.
- `select_gpu()` chooses Tier 1: no compute processes and at least 24 GiB free; Tier 2: no processes and at least 16 GiB; Tier 3 only as a shared fallback with at least 24 GiB free and low utilization.
- `selected_gpu_is_still_eligible()` performs the race check.

The generic launch pattern is mandatory: inspect/select a physical GPU immediately before model load, launch the child with `CUDA_VISIBLE_DEVICES=<selected physical index>`, and address it inside the child only as `cuda:0`. Never hardcode GPU 3 or any other index and never kill/evict a process. A cheap diagnostic is:

```bash
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
```

The inspection snapshot showed eight RTX A6000 devices; physical devices 0, 3, 5, and 6 had approximately 48,536 MiB free and 0% GPU utilization at that instant. This is not a reservation or a R5 configuration; re-discover and re-check at runtime.

## Qwen3-Reranker-4B

The exact restored model is:

```text
repo:       Qwen/Qwen3-Reranker-4B
revision:   22e683669bc0f0bd69640a1354a6d0aebcfeede5
snapshot:   /mnt/disk/mxf/.cache/huggingface/hub/models--Qwen--Qwen3-Reranker-4B/snapshots/22e683669bc0f0bd69640a1354a6d0aebcfeede5
snapshot SHA: 32b52d29edef232618c27a728cde56d369ba862c81d49c6527528dfd82db39c1
runtime SHA:  d7c1841432198b60f2266e1d86c107354b11af8c03be5562fdc9f859b173f01a
dtype:      bfloat16
default batch: 4
```

The tracked R3P0 recovery artifacts are under `artifacts/evaluation/nf-v2-18-qwen-reranker-recovery/`; `snapshot-manifest.sha256` was verified as the expected snapshot SHA. The historical scorer is `src/pdf_retrieval_v4/qwen3_reranker.py` (`build_input_ids`, `score_batch`). The reusable wrapper is `src/pdf_retrieval_v4/qwen3_reranker_runtime.py` (`Qwen3RerankerRuntime.load`, `score_pairs`). The wrapper preserves the historical yes/no-logit protocol, left padding, max length 8192, and descending `reranker_score`. Use the wrapper rather than the stale 0.6B defaults in the low-level config dataclass.

R3P0's existing health command (not run during this handoff) is:

```bash
cd /mnt/disk/mxf/projects/Qhhhhhhaaa/nanochat/.worktrees/nf-v2-18-retrieval-recovery/finquery_rag/backend
/mnt/disk/mxf/anaconda3/envs/QhChat/bin/python scripts/evaluation/run_nf_v2_18_r3p0_recovery.py
```

It writes `artifacts/evaluation/nf-v2-18-qwen-reranker-recovery/reranker-health-check.json`. For a future R5 score cache, set `CUDA_VISIBLE_DEVICES` from the dynamic selector first, then call `Qwen3RerankerRuntime.load(snapshot)` and `score_pairs(pairs, batch_size=4, instruction=...)`. Do not call this in the handoff task.

## Qwen3-Embedding-4B R4 status

R4 used (but did not select for R5):

```text
repo/revision: Qwen/Qwen3-Embedding-4B@5cf2132abc99cad020ac570b19d031efec650f2b
snapshot manifest SHA: 54c2a3b9ef650ea026acab844c56917c23126f0426f1d4909154adbf485887dc
dimension: 2560
external index: /mnt/disk/mxf/projects/Qhhhhhhaaa/financial_corpus_v2/indexes/nf-v2-18-retrieval-recovery/r4-qwen3-embedding
```

R4 candidate headroom was A4@200 `95/105`, Qwen dense@200 `48/105`, and union@200 `95/105`; it contributed no meaningful new headroom. Do not rebuild or reopen this first-stage experiment in R5 unless a separate task explicitly changes scope.

## Corpus and benchmark scope

Corpus V2 contains 10 companies, 60 primary SEC filings (30 annual and 30 quarterly). R4's external index was intentionally limited to the GOOGL/AMZN entity scope used by the consumed development questions; it is not a production-corpus replacement. Production indices and configs remain unchanged.

NF-V2-17 B3 is immutable. Relevant frozen hashes (for audit only) are runtime output `9e02df6701268e83cd9dafdcc36d95736167c19cca8d244a134a69149538dd83`, trace `3b3b9ee227a49631dcf4f1c820172fd63cf465cbb85514d67ce9d06d2d5f6ebd`, and evaluation freeze `c3648925f07e878123e78e0fed21b12e0499a461d2c83e28616cfe80789c920a`. R5 may use the already consumed 120-question regression set for development, but must not edit questions, Gold, B3 outputs, or call the result fresh-blind.

## Authoritative current metrics

R4's A4 candidate headroom on 105 answerable questions is:

| A4 candidate depth | exact canonical inclusion |
|---:|---:|
| 20 | 77/105 |
| 50 | 88/105 |
| 100 | 92/105 |
| 200 | 95/105 |

The R4 authoritative selected ranked starting point (A4-preserving) is `R@1 33/105`, `R@3 55/105`, `R@5 62/105`, `R@10 68/105`, `R@20 77/105`. These values are distinct from the earlier R3 route-specific diagnostic, which reported exact `R@5 63/105`, `R@10 71/105`, multi `Any@5 19/20`, `All@5 10/20`, `All@10 12/20`, `All@20 12/20`, and operand-complete `@5 5/15`, `@10 5/15`, `@20 6/15`. Do not mix R3 diagnostic values into the R4 baseline.

R4's route diagnostics were not an improvement: multi route `Any@5 17/20`, `All@5 3/20`, `All@10 6/20`, `All@20 8/20`; calculation route operand complete `@5 3/15`, `@10 5/15`, `@20 6/15`. R4 decision was `RETRIEVAL_RECOVERY_FAILED`, ceiling `RANKING_CEILING`, recommendation `CONTINUE_TARGETED_RETRIEVAL`.

## Do Not Repeat These Experiments

1. The old SQLite FTS5 all-token AND query overconstraint has already been repaired. Keep hard metadata scope separate from content-bearing lexical terms; never require question function words, issuer words, and period words in the FTS query.
2. Do not make the all-MiniLM-L6-v2 row-dense path the primary selector; it did not recover sufficient recall.
3. Do not replace A4 with Qwen3-Embedding-4B; R4 proves its @200 headroom is below A4 and the union adds no headroom.
4. Do not let shared table semantics replace A4. R2/S5 damaged global exact recall; use semantic data as enrichment/features while preserving the A4 candidate set.
5. Do not make Qwen3-Reranker the global authority. R3 rescued 8 questions and damaged 9 (net -1); treat Qwen as an additional ranking feature and preserve A4 candidates on mapping/reranker failure.
6. Do not reintroduce global iXBRL fusion. Keep structured lookup separate and only consider it for an explicitly justified calculation/operand route.

## Existing R5 implementation entry points

| Absolute/relative file | Existing entry point | Role | R5 action |
|---|---|---|---|
| `scripts/evaluation/run_nf_v2_18a_recovery.py` | `qbuild`, `fts`, `dense_load`, `merge`, `expand`, `metrics` | repaired A4 BM25/dense/coarse candidate generation and metrics | read/freeze; do not silently alter baseline |
| `scripts/evaluation/run_nf_v2_18a_r1_fine.py` | `derive_slots`, `coarse_replay`, `local_fine`, `slot_retrieve`, `stage_metricset`, `stage_slots`, `family_fine_audit` | A4 replay, runtime-derived slots, local evidence diagnostics | reuse for freeze and route diagnostics |
| `scripts/evaluation/run_nf_v2_18a_r2_shared.py` | `make_semantic_corpus`, `enrich_records`, `semantic_rank`, `slot_retrieve_semantic` | shared HTML table semantics and semantic ranking diagnostics | feature source only; not a replacement retriever |
| `scripts/evaluation/run_nf_v2_18a_r3_hierarchical_qwen.py` | `child_rows`, `gate`, `semantic_expand`, `doc_view`, `slot_results`, `compare` | local child expansion, period gate, historical Qwen process contract | reuse protocol; do not reuse global replacement behavior |
| `scripts/evaluation/run_nf_v2_18a_r4_strong_first_stage.py` | `load_inputs`, `build_entries`, `dense_hits`, `a4_phrase_hits`, `atomic_lexical_hits`, `metrics`, `compare`, `run_reranker` | R4 candidate/index diagnostics and no-loss route selection | read authoritative R4 artifacts; do not rerun as the freeze command |
| `src/pdf_retrieval_v4/semantic_graph_models.py` | `LogicalTable`, `SemanticRow`, `MetricPath`, `SemanticAxisBinding`, `AtomicFact`, deterministic ID builders | canonical shared table/semantic identities | reuse |
| `src/pdf_retrieval_v4/html_semantic_adapter.py` | `build_semantic_corpus`, `attach_semantics`, `_header_paths`, `_match_ixbrl` | HTML/iXBRL metadata and table enrichment | reuse |
| `src/pdf_retrieval_v4/metric_binding_contract_v2.py` | `bind_metric`, `infer_measurement_kind` | deterministic metric/unit compatibility | reuse for gates |
| `src/pdf_retrieval_v4/operand_planner.py` | `build_operand_slots` | generic calculation operand-slot decomposition | reuse; no calculator arithmetic change |
| `src/pdf_retrieval_v4/qwen3_reranker_runtime.py` | `Qwen3RerankerRuntime.load`, `score_pairs` | exact Qwen wrapper, ordered batching, VRAM diagnostics | optional score cache only |
| `src/pdf_retrieval_v4/gpu_selector.py` | `discover_gpus`, `select_gpu`, `selected_gpu_is_still_eligible` | dynamic physical-to-logical GPU mapping | reuse |
| `src/pdf_retrieval_v4/r5_rank_contract.py` | `cutoff_label`, `classify_rank_migration`, `recovered_to_cutoff` | small deterministic rank-migration helpers | extend only if needed |
| `tests/pdf_retrieval_v4/test_gate_08_hierarchical_retrieval.py`, `test_gate_08_r5.py`, `test_gate_08_r5_1_r6.py`, `test_nf_opt_20_r1_period_boundary_guard.py` | existing hierarchy/ranking/period safety tests | regression coverage | run/reuse |

R5 does not yet have a freeze/ablation CLI. Add it only as a development evaluation script (for example `scripts/evaluation/run_nf_v2_18a_r5_ranking_recovery.py`) and, if useful, a small feature helper (`src/pdf_retrieval_v4/r5_rank_features.py`). This handoff intentionally does not add either implementation.

Do not touch generator weights, Supervisor policy, `SemanticClaimVerifier`, `RuntimeGenerationValidator`, calculator arithmetic, production retrieval configuration, or the immutable B3 artifacts.

## R5 implementation contract

1. First freeze A4 Top-200 candidate lists for all 105 answerable questions. All rank ablations must read the same sealed JSONL. Expected exact inclusion is 77/105, 88/105, 92/105, 95/105 at 20/50/100/200. If regenerated values differ, stop and investigate candidate-generation drift.
2. Record Gold rank distribution and classify Top-10 failures without changing Gold. Preserve the canonical evidence IDs and provenance.
3. Build only generic features available from candidate/runtime data: A4 raw score/rank (primary), MetricPath/metric phrase similarity, row-label and table-title similarity, section, explicit period compatibility, header path, unit/currency/scale, evidence type, semantic completeness, and optionally a cached Qwen score. Never use question IDs, Gold IDs, answer values, reference answers, or company-specific manual maps.
4. Use deterministic blended ranking first. If a learning-to-rank model is tried, use one lightweight method with grouped splits by source filing/question and report cross-validated estimates; do not leak candidates from one question across train/validation.
5. Preserve no-loss behavior: semantic mapping failure or Qwen failure retains the original A4 candidate. A ranker may reorder valid candidates only.
6. Preserve route-specific slot retrieval. Multi evidence must rank per runtime-derived slot and allocate a balanced final budget; one slot must not consume the entire Top-K. Calculation must measure operand presence at 20/50/100/200 before optimizing ranking and must not execute the calculator unless binding is already complete.
7. Apply hard gates before ranking. `UNKNOWN` period metadata remains unresolved; it cannot satisfy an explicit period slot. Zero target for false binding, wrong-period binding, authorization/entity/fiscal/document/version violations, silent relaxation, and `created_at` misuse.

Feature separability and Qwen-score caches must be produced without Gold/answer fields in feature computation. Use the Qwen score as a feature, not a replacement for A4 ordering.

## R5 Suggested Runbook

Commands below are repository-specific. Commands marked `TO_IMPLEMENT` are required future work, not commands run by this handoff.

```bash
# 1–3. enter and verify the exact worktree
ssh mxf@10.157.195.124
cd /mnt/disk/mxf/projects/Qhhhhhhaaa/nanochat/.worktrees/nf-v2-18-retrieval-recovery
git status --short
git rev-parse HEAD
git branch --show-current

# 4–5. ordinary environment and cheap regressions
cd finquery_rag/backend
/mnt/disk/mxf/anaconda3/bin/python -m pytest -q tests/test_phase6_reranker.py tests/validation/test_metric_period_grounding.py

# syntax/lint for changed or newly added scripts
/mnt/disk/mxf/anaconda3/bin/python -m py_compile scripts/evaluation/run_nf_v2_18a_r5_ranking_recovery.py
ruff check scripts/evaluation/run_nf_v2_18a_r5_ranking_recovery.py src/pdf_retrieval_v4/r5_rank_features.py

# 6. TO_IMPLEMENT: freeze A4 Top-200, using R1/A4 paths and the external corpus/index.
#    Do not run the full R4 script: it rebuilds/loads the R4 embedding experiment.
/mnt/disk/mxf/anaconda3/bin/python scripts/evaluation/run_nf_v2_18a_r5_ranking_recovery.py --stage freeze-a4-top200

# 7–10. TO_IMPLEMENT stages, all reading the same frozen JSONL
/mnt/disk/mxf/anaconda3/bin/python scripts/evaluation/run_nf_v2_18a_r5_ranking_recovery.py --stage rank-audit
/mnt/disk/mxf/anaconda3/bin/python scripts/evaluation/run_nf_v2_18a_r5_ranking_recovery.py --stage features
/mnt/disk/mxf/anaconda3/bin/python scripts/evaluation/run_nf_v2_18a_r5_ranking_recovery.py --stage ablations

# 11. only if the ablation proves Qwen is useful: dynamically select a GPU,
# then run the existing R3P0 health check / Qwen wrapper in QhChat.
/mnt/disk/mxf/anaconda3/envs/QhChat/bin/python scripts/evaluation/run_nf_v2_18_r3p0_recovery.py

# 12–14. TO_IMPLEMENT route, multi-budget, calculation, safety and CV stages
/mnt/disk/mxf/anaconda3/bin/python scripts/evaluation/run_nf_v2_18a_r5_ranking_recovery.py --stage route-specific
/mnt/disk/mxf/anaconda3/bin/python scripts/evaluation/run_nf_v2_18a_r5_ranking_recovery.py --stage cross-validation
/mnt/disk/mxf/anaconda3/bin/python scripts/evaluation/run_nf_v2_18a_r5_ranking_recovery.py --stage safety

# 15–18. validate, review diff, commit only R5 implementation/artifacts
/mnt/disk/mxf/anaconda3/bin/python -m py_compile scripts/evaluation/run_nf_v2_18a_r5_ranking_recovery.py
ruff check scripts/evaluation/run_nf_v2_18a_r5_ranking_recovery.py src/pdf_retrieval_v4/r5_rank_features.py
git diff --check
git status --short
git add <R5-only-files>
git commit -m "feat(finquery): recover candidate ranking"
git status --short
```

The current R4 artifact integrity checker is an existing cheap command (use it for reference validation, not to claim R5 completion):

```bash
/mnt/disk/mxf/anaconda3/bin/python scripts/evaluation/validate_nf_v2_18a_r4_artifacts.py artifacts/evaluation/nf-v2-18-r4-strong-first-stage
```

## Required R5 artifacts

Create under `finquery_rag/backend/artifacts/evaluation/nf-v2-18-r5-ranking-recovery/`:

```text
frozen-a4-top200.jsonl                 frozen-a4-top200.sha256
gold-rank-distribution.json            top10-failure-audit.json
ranking-feature-spec.json              feature-separability.json
qwen-score-cache-manifest.json         general-ranking-ablation.json
table-ranking-ablation.json            text-ranking-ablation.json
multi-slot-ranking.json                multi-budget-allocation.json
calculation-headroom.json              calculation-ranking.json
calculation-budget-allocation.json     cross-validation.json
route-specific-selection.json          safety-regression.json
latency.json                           selected-config.json
selected-config.sha256                 final-report.md
decision.json
```

The selected decision must report exact R@1/3/5/10/20 on all 120 and 105 answerable, family-to-exact conversion, multi Any/All at 5/10/20, calculation operand presence/completeness, route breakdown, latency, Qwen pairs/VRAM if used, and every safety count. Targets are R@5 ≥ 78/105, R@10 ≥ 85/105, Multi All@10 ≥ 14/20, and maximum calculation completeness up to the measured candidate oracle ceiling; targets are not permission to alter Gold.

## Stop / acceptance rules

- `RANKING_RECOVERED`: existing candidates can be promoted into target cutoffs with all safety counts zero.
- `RANKER_SIGNAL_CEILING`: correct evidence is in Top-200 but generic features cannot separate it reliably; do not immediately try another embedding model.
- `EVIDENCE_BUDGET_ALLOCATION_CEILING`: multi-slot evidence is retrieved but final allocation drops valid slots.
- `OPERAND_CANDIDATE_CEILING`: required operands are absent from A4 Top-200.
- `OPERAND_BINDING_CEILING`: operands are present but the existing Binder rejects them.

R5 is development-only. Do not open full runtime or production cutover automatically; production remains V1 with switch false.

## Common Failure Modes / Recovery

- Wrong Conda environment: print `sys.executable` and versions; use base Python for CPU evaluation and QhChat only for Qwen.
- CUDA mapping confusion/hardcoded index: select a physical GPU with `gpu_selector.py`, set `CUDA_VISIBLE_DEVICES`, and use only `cuda:0` in the child.
- Snapshot missing or health failure: inspect the exact revision/path and `snapshot-manifest.sha256`; do not download a different revision or silently use CPU.
- Candidate universe differs from A4: stop before ranking and compare the frozen A4 Top-200 hash and 20/50/100/200 counts.
- Wrong denominator: report both all/120 and answerable/105; multi is /20 and calculations /15.
- Fresh-blind confusion or B3 mutation: label the set `CONSUMED_DEVELOPMENT_REGRESSION`; never edit B3/questions/Gold.
- Global iXBRL or shared-semantics fusion reintroduced: keep them route-specific/enrichment-only and preserve A4 no-loss candidates.
- FTS5 all-token AND regression: inspect lexical query construction and keep metadata terms out of the lexical query.
- Qwen globally replaces A4: compare rescued/damaged counts and retain A4 order/candidate on failure.
- Production index/config accidentally modified: inspect `git diff`, external index paths, and production configuration before commit; R5 must not write there.

## Handoff completion checklist

The next agent can answer from this file alone: where to SSH, actual worktree/backend paths, branch/HEAD, environments, dynamic GPU procedure, exact Qwen snapshot/runtime, R4 embedding status, corpus/benchmark scope, immutable artifacts, A4 metrics, ruled-out experiments, R5 entry points, runbook, artifacts, targets, and stop criteria. No secret or credential is present.
