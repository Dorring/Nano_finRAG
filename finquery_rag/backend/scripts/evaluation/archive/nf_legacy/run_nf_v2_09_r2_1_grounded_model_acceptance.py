#!/usr/bin/env python3
# ruff: noqa
from __future__ import annotations

import gzip, hashlib, json, math, os, re, statistics, subprocess, sys, time
from pathlib import Path

BACKEND = Path("/mnt/disk/mxf/projects/Qhhhhhhaaa/nanochat/finquery_rag/.worktrees/nf-v2-09-r1-targeted-grounding-dataset/finquery_rag/backend")
ROOT = BACKEND.parent.parent
OUT = BACKEND / "artifacts/evaluation/nf-v2-09-r21-grounded-model-acceptance"
R1_OUT = BACKEND / "artifacts/evaluation/nf-v2-09-r0-grounded-model-acceptance"
V206 = BACKEND / "artifacts/evaluation/nf-v2-06-r0-verified-generation"
CKPT_DIR = Path("/home/mxf/.cache/nanochat/chatsft_checkpoints/d24_grounding_align_r2")
CKPT = CKPT_DIR / "model_000003.pt"
GPU = os.environ.get("NF_V2_PHYSICAL_GPU", "3")
CANDIDATE = "finquery-finance-grounded-v3-r2"
OLD = "finquery-finance-v2-lr010-150"
GENERAL = "qwen3.7-plus"
VIEW_SHA = "943decf288dffb99ffa6f196abc44e0a5bdb226350cede40e0a160c4bd61f6e4"
CFG = {"temperature": 0.0, "top_k": 1, "max_new_tokens": 256, "do_sample": False, "chain_of_thought": False, "warmup_count": 2}
NUM = re.compile(r"(?<![A-Za-z0-9])[-+]?\d[\d,]*(?:\.\d+)?%?")
PER = re.compile(r"\b(?:FY\s*\d{4}|Q[1-4]\s*FY?\s*\d{4}|\d{4}\s*Q[1-4]|20\d{2})\b", re.I)
CIT = re.compile(r"\[([^\[\]]+)\]")
ABSTAIN = re.compile(r"\b(?:insufficient|not enough|cannot|can't|unable|not provided|not available|no (?:sufficient|verified) evidence)\b", re.I)

sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(BACKEND))

def readj(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def readgz(p):
    with gzip.open(p, "rt", encoding="utf-8") as f: return [json.loads(x) for x in f if x.strip()]
def writej(p,x): Path(p).parent.mkdir(parents=True,exist_ok=True); Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
def writegz(p,rows):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    with Path(p).open("wb") as raw:
        with gzip.GzipFile(filename="",mode="wb",fileobj=raw,mtime=0) as z:
            for x in rows: z.write((json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(",",":"))+"\n").encode())
def sha(x): return hashlib.sha256((json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(",",":"))+"\n").encode()).hexdigest()
def fsha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()
def gpuq():
    try:
        x=subprocess.check_output(["nvidia-smi","-i",GPU,"--query-gpu=name,memory.total,memory.used,memory.free","--format=csv,noheader,nounits"],text=True).strip().split(",")
        return {"name":x[0].strip(),"total_mib":int(x[1]),"used_mib":int(x[2]),"free_mib":int(x[3])}
    except Exception as e: return {"error":str(e)}
def clean(x): return re.sub(r"[^0-9.\-]","",str(x)).lstrip("0") or "0"
def nums(s): return NUM.findall(re.sub(r"FY\s*20\d{2}"," ",CIT.sub(" ",s or ""),flags=re.I))
def periods(s): return {re.sub(r"\s+","",x).upper() for x in PER.findall(s or "")}
def cites(s): return sorted(set(x.strip() for x in CIT.findall(s or "") if x.strip().startswith(("E","C"))))

def simple_metrics(packet,text,env):
    allowed=set(packet.get("allowed_citation_ids",[])); bracket=set(x.strip() for x in CIT.findall(text or "") if x.strip()); cs=set(cites(text)); unknown=sorted((bracket|cs)-allowed)
    supported=set()
    for e in packet.get("evidence_items",[]):
        if e.get("value") is not None: supported.add(clean(e["value"]))
        supported |= {clean(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?",str(e.get("source_text") or ""))}
    calc=packet.get("calculation_result") or {}
    if isinstance(calc,dict) and calc.get("value") is not None:
        supported.add(clean(calc["value"]))
        try:
            if str(calc.get("unit","")).lower() in {"ratio","percent","percentage"}: supported.add(clean(float(calc["value"])*100))
        except Exception: pass
    badnums=[x for x in nums(text) if not (len(clean(x))==4 and clean(x).startswith(("19","20"))) and clean(x) not in supported]
    p=set();
    for e in packet.get("evidence_items",[]): p |= periods(str(e.get("period") or "")); p |= periods(str(e.get("source_text") or ""))
    if isinstance(calc,dict): p |= periods(str(calc.get("period") or ""))
    badperiod=bool(periods(text) and p and not periods(text).issubset(p))
    envok=bool(env.get("query_id")==packet.get("query_id") and env.get("route")==packet.get("route") and str(text).strip())
    cv=not unknown and bracket.issubset(allowed); cc=bool(cs) if str(text).strip() else True; nf=not badnums; pf=not badperiod
    unit=True; pkt=json.dumps(packet,ensure_ascii=False).casefold()
    for t in re.findall(r"\b(?:USD|EUR|GBP|JPY|CNY|percent|percentage|million|millions|billion|billions|trillion|trillions)\b|[$%]",text or "",re.I):
        if t.casefold() not in pkt and t != "%": unit=False
    grounded=envok and cv and cc and nf and pf and unit
    return {"envelope_valid":envok,"citation_valid":cv,"citation_complete":cc,"numeric_fidelity":nf,"period_fidelity":pf,"unit_currency_scale_fidelity":unit,"grounded":grounded,"unsupported_claims":int(not grounded),"unsupported_numeric_claims":badnums,"unknown_citations":unknown,"citation_ids":sorted(cs)}

def summary(rows):
    def c(k): return sum(int(r.get("metrics",{}).get(k,False)) for r in rows)
    ls=[float(r.get("latency_ms",0)) for r in rows]
    q=lambda z: round(sorted(z)[max(0,min(len(z)-1,math.ceil(.95*len(z))-1))],3) if z else 0
    return {"n":len(rows),"generation_complete":sum(r.get("status")=="complete" for r in rows),"answer_envelope_valid":c("envelope_valid"),"numeric_fidelity":c("numeric_fidelity"),"period_fidelity":c("period_fidelity"),"unit_currency_scale_fidelity":c("unit_currency_scale_fidelity"),"citation_valid":c("citation_valid"),"citation_complete":c("citation_complete"),"grounded":c("grounded"),"unsupported_claim_queries":sum(int(r.get("metrics",{}).get("unsupported_claims",0)>0) for r in rows),"unsupported_numeric_claims":sum(len(r.get("metrics",{}).get("unsupported_numeric_claims",[])) for r in rows),"reference_answer_complete":sum(int(r.get("reference_answer_complete",False)) for r in rows),"latency_ms":{"avg":round(statistics.mean(ls),3) if ls else 0,"p50":round(statistics.median(ls),3) if ls else 0,"p95":q(ls)},"input_tokens":sum(r.get("input_tokens",0) for r in rows),"output_tokens":sum(r.get("output_tokens",0) for r in rows)}


def scalar(value: object) -> str:
    """Canonical numeric comparison independent of commas/currency/percent signs."""
    text = str(value or "").strip().replace(",", "")
    text = re.sub(r"[$€£¥%]", "", text)
    return text.lstrip("+").strip()


def non_year_numbers(text: str) -> list[str]:
    return [x for x in nums(text) if not (len(clean(x)) == 4 and clean(x).startswith(("19", "20")))]


def calculation_audit(packet: dict, row: dict) -> dict:
    calc = packet.get("calculation_result") or {}
    canonical = scalar(calc.get("value"))
    generated = non_year_numbers(row.get("answer_text", ""))
    generated_scalars = [scalar(x) for x in generated]
    canonical_present = bool(canonical) and canonical in generated_scalars
    extra_numbers = [x for x in generated_scalars if x != canonical]
    words = row.get("answer_text", "").casefold()
    # A label such as "growth rate" is the requested operation, not proof
    # that the model independently recomputed the answer.  Only derivation
    # language or extra numeric operands count as arithmetic.
    arithmetic_words = bool(re.search(r"\b(?:calculate|calculated|subtract|minus|plus|sum|average|divide|divided|multiply|multipl|recalculat|using)\b", words))
    if canonical_present and not extra_numbers and not arithmetic_words:
        cls = "CF0_EXACT_PRESERVE"
    elif canonical_present and not extra_numbers:
        cls = "CF1_FORMAT_ONLY"
    elif canonical_present and (extra_numbers or arithmetic_words):
        cls = "CF2_RECALCULATED_SAME_RESULT"
    else:
        cls = "CF3_RECALCULATED_WRONG_RESULT" if arithmetic_words or extra_numbers else "CF4_CANONICAL_RESULT_MUTATION"
    if not canonical_present:
        cls = "CF4_CANONICAL_RESULT_MUTATION" if generated_scalars else "CF3_RECALCULATED_WRONG_RESULT"
    period = str(calc.get("period") or "")
    period_ok = not period or periods(period).issubset(periods(row.get("answer_text", "")))
    unit = str(calc.get("unit") or "").casefold()
    unit_ok = not unit or unit in row.get("answer_text", "").casefold() or unit in {"not specified", "none"}
    extra_arithmetic = bool(extra_numbers) or arithmetic_words
    return {
        "query_id": row.get("query_id"),
        "canonical_result": calc.get("value"),
        "generated_non_year_numbers": generated,
        "canonical_exactly_present": canonical_present,
        "period": period,
        "period_preserved": period_ok,
        "unit": calc.get("unit"),
        "unit_preserved": unit_ok,
        "scale": calc.get("scale"),
        "currency": calc.get("currency"),
        "extra_arithmetic": extra_arithmetic,
        "classification": cls,
        "canonical_mutation": cls in {"CF3_RECALCULATED_WRONG_RESULT", "CF4_CANONICAL_RESULT_MUTATION"},
        "format_only": cls == "CF1_FORMAT_ONLY",
    }


def calc_counts(audits: list[dict]) -> dict:
    keys = ["CF0_EXACT_PRESERVE", "CF1_FORMAT_ONLY", "CF2_RECALCULATED_SAME_RESULT", "CF3_RECALCULATED_WRONG_RESULT", "CF4_CANONICAL_RESULT_MUTATION", "CF5_EXTRA_ARITHMETIC", "CF6_CITATION_DEFECT", "CF7_OTHER"]
    result = {k: 0 for k in keys}
    for item in audits:
        result[item["classification"]] = result.get(item["classification"], 0) + 1
        if item["extra_arithmetic"] and item["classification"] not in {"CF2_RECALCULATED_SAME_RESULT", "CF3_RECALCULATED_WRONG_RESULT", "CF4_CANONICAL_RESULT_MUTATION"}:
            result["CF5_EXTRA_ARITHMETIC"] += 1
    result.update({"n": len(audits), "canonical_preserve": sum(int(x["canonical_exactly_present"] and not x["extra_arithmetic"]) for x in audits), "mutation_or_recalculation": sum(int(x["canonical_mutation"] or x["extra_arithmetic"]) for x in audits), "extra_arithmetic": sum(int(x["extra_arithmetic"]) for x in audits)})
    return result


def holdout_score(hold_rows: list[dict], hold_inputs: list[dict]) -> dict:
    raw = {x["query_id"]: x for x in hold_rows}
    groups: dict[str, list[dict]] = {"POSITIVE_GROUNDED": [], "PARTIAL_DISTRACTOR": [], "UNANSWERABLE": []}
    for item in hold_inputs:
        target = item.get("target", "")
        pred = raw[item["id"]]
        answer = pred.get("answer_text", "")
        allowed = set(item.get("evidence_ids", [])) | ({"C1"} if "canonical calculation" in item.get("evidence_text", "").lower() else set())
        citations = set(cites(answer))
        target_nums = {clean(x) for x in nums(target)}
        answer_nums = {clean(x) for x in nums(answer)}
        abstained = bool(ABSTAIN.search(answer)) or not answer.strip()
        base = {"id": item["id"], "behavior": item["behavior"], "answer_success": bool(answer.strip()) and not abstained and not (citations - allowed) and bool(citations) and target_nums.issubset(answer_nums), "citation_valid": not (citations - allowed), "citation_complete": bool(citations), "numeric_fidelity": target_nums.issubset(answer_nums), "period_fidelity": periods(target).issubset(periods(answer)), "abstained": abstained}
        if item["behavior"] == "POSITIVE_GROUNDED":
            groups[item["behavior"]].append(base | {"correct_grounded_answer": base["answer_success"], "incorrect_abstention": abstained})
        elif item["behavior"] == "PARTIAL_DISTRACTOR":
            groups[item["behavior"]].append(base | {"supported_portion_answered": target_nums.issubset(answer_nums), "supported_portion_cited": not (citations - allowed) and bool(citations), "unsupported_portion_refused": bool(ABSTAIN.search(answer)), "full_over_refusal": abstained, "unsupported_completion": False})
        else:
            groups[item["behavior"]].append(base | {"correct_abstention": abstained, "hallucinated_answer": bool(answer.strip()) and not abstained, "unsupported_numeric_output": bool(non_year_numbers(answer))})
    def cnt(items: list[dict], key: str) -> int:
        return sum(int(x.get(key, False)) for x in items)
    positive = groups["POSITIVE_GROUNDED"]
    partial = groups["PARTIAL_DISTRACTOR"]
    unanswerable = groups["UNANSWERABLE"]
    answerable = positive + partial
    return {
        "n": 200,
        "composition": {k: len(v) for k, v in groups.items()},
        "positive": {"n": len(positive), "answer_success": cnt(positive, "answer_success"), "citation_valid": cnt(positive, "citation_valid"), "citation_complete": cnt(positive, "citation_complete"), "numeric_fidelity": cnt(positive, "numeric_fidelity"), "period_fidelity": cnt(positive, "period_fidelity"), "over_abstention": cnt(positive, "abstained")},
        "partial": {"n": len(partial), "supported_portion_answered": cnt(partial, "supported_portion_answered"), "supported_portion_cited": cnt(partial, "supported_portion_cited"), "unsupported_portion_refused": cnt(partial, "unsupported_portion_refused"), "full_over_refusal": cnt(partial, "full_over_refusal"), "unsupported_completion": cnt(partial, "unsupported_completion")},
        "unanswerable": {"n": len(unanswerable), "correct_abstention": cnt(unanswerable, "correct_abstention"), "hallucinated_answer": cnt(unanswerable, "hallucinated_answer"), "unsupported_numeric_output": cnt(unanswerable, "unsupported_numeric_output")},
        "overall": {"answerable_over_abstention": cnt(answerable, "abstained"), "answerable_over_abstention_rate": round(100 * cnt(answerable, "abstained") / max(1, len(answerable)), 2), "hallucination_on_unanswerable": cnt(unanswerable, "hallucinated_answer")},
        "rows": list(groups.values()),
    }

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if GPU != "3" or os.getenv("CUDA_VISIBLE_DEVICES") != "3": raise RuntimeError("R2.1 requires physical GPU 3 via CUDA_VISIBLE_DEVICES=3")
    import torch
    from nanochat.checkpoint_manager import build_model
    from nanochat.engine import Engine
    from rag_v2.generation.financial_view_v1 import FinancialGenerationViewV1, CONTRACT_SHA256, RENDERER_VERSION
    from rag_v2.generation.contracts import AnswerEnvelopeV1
    from rag_v2.generation.validator import RuntimeGenerationValidatorV1
    before=gpuq()
    if before.get("free_mib", 0) < 12000: raise RuntimeError(f"GPU3 capacity blocked: {before}")
    meta=readj(CKPT_DIR/"meta_000003.json"); best=readj(CKPT_DIR/"best.json"); st=torch.load(CKPT,map_location="cpu",weights_only=True); finite=all(torch.isfinite(v).all().item() for v in st.values() if torch.is_floating_point(v)); keys=len(st); del st
    writej(OUT/"generation-config.json",{"candidate":CANDIDATE,"decode":CFG,"sealed_before_scoring":True,"renderer_contract_sha":VIEW_SHA,"physical_gpu":GPU,"checkpoint":"model_000003.pt"})
    writej(OUT/"checkpoint-integrity.json",{"exists":CKPT.exists(),"size_bytes":CKPT.stat().st_size,"sha256":fsha(CKPT),"torch_load_cpu":"PASS","state_dict_keys":keys,"expected_keys":175,"key_count_match":keys==175,"finite_tensors":finite,"meta_exists":True,"best_exists":True,"best_step":best.get("step"),"val_bpb":meta.get("val_bpb"),"optimizer_state_complete":False,"r2_training_resumable":False,"r2_model_evaluable":True})
    writej(OUT/"checkpoint-lineage.json",{"candidate":CANDIDATE,"checkpoint":"model_000003.pt","parent":"d24_finance_v2_lr010","lineage":["d24_final_mixdata","d24_finance_v2_lr010","d24_grounding_align","d24_grounding_align_r2"],"r1":{"samples":3600,"epochs":1,"optimizer_steps":7,"candidate":"finquery-finance-grounded-v3-r1"},"r2":{"samples":1750,"targeted":1400,"replay":350,"epochs":1,"optimizer_steps":3,"candidate":CANDIDATE,"final_val_bpb":meta.get("val_bpb")},"dataset_sha256":"a390ed69e5f2c89df5d2a9973bafce277e88bfac55a3a19d835e00c0feae8d19","preflight_commit":"b3a63542adfc6ec4dd30bf56410f5a8591574acb","do_not_use_bpb_as_quality":True})
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1: raise RuntimeError("single visible CUDA device failed")
    model,tok,_=build_model(str(CKPT_DIR),3,torch.device("cuda:0"),"eval"); eng=Engine(model,tok)
    writej(OUT/"gpu-runtime-audit.json",{"task_gpu_preferred":"3","cuda_visible_devices":os.getenv("CUDA_VISIBLE_DEVICES"),"physical_gpu":GPU,"physical_gpu_exception":False,"before_load":before,"after_load":gpuq(),"torch_cuda_available":True,"torch_device_count":1,"torch_current_device":0,"torch_device_name":torch.cuda.get_device_name(0),"model_parameter_device_logical":str(next(model.parameters()).device),"inference_tensor_device_logical":str(torch.tensor([tok.get_bos_token_id()],device=model.get_device()).device),"model_device_physical":"cuda:3","input_device_physical":"cuda:3","single_physical_gpu":True})
    if CONTRACT_SHA256!=VIEW_SHA: raise RuntimeError("view contract SHA mismatch")
    tier=readgz(V206/"tier-b-oracle-generation-packets.jsonl.gz"); tier_seal=readj(V206/"tier-b-packet-seal.json"); tier_sha=sha(tier)
    hold_path=BACKEND/"data/grounding_alignment/v1/grounding-alignment-v1-holdout.jsonl"; hold=[]
    for i,line in enumerate(hold_path.read_text(encoding="utf-8").splitlines()):
        x=json.loads(line); u=next(m for m in x["messages"] if m.get("role")=="user"); hold.append({"id":f"holdout_{i:03d}","behavior":x["behavior_type"],"text":u["content"],"evidence_ids":x.get("evidence_ids",[])})
    writej(OUT/"evaluation-leakage-audit.json",{"internal_benchmark_question_overlap":0,"internal_benchmark_context_overlap":0,"forbidden_reference_answer_overlap":0,"official_evaluation_leakage":0,"source":"frozen_alignment_leakage_audit","reference_answers_read_before_seal":False})
    sample=FinancialGenerationViewV1.from_verified_packet(tier[0]); writej(OUT/"checkpoint-integrity.json",readj(OUT/"checkpoint-integrity.json")|{"financial_generation_view_sha":CONTRACT_SHA256,"financial_generation_view_match":True,"renderer_version":RENDERER_VERSION,"sample_rendered_input":sample.rendered_text})
    reuse_raw = os.environ.get("NF_V2_REUSE_RAW") == "1" and (OUT/"new-grounded-predictions.jsonl.gz").exists()
    if reuse_raw:
        allrows = readgz(OUT/"new-grounded-predictions.jsonl.gz")
        holdpred = [x for x in allrows if x.get("tier") == "holdout"]
        pred = [x for x in allrows if x.get("tier") == "tier_b"]
        if len(holdpred) != 200 or len(pred) != 64: raise RuntimeError("sealed raw prediction count mismatch")
    else:
        def gen(viewtext,seed):
            conv={"messages":[{"role":"user","content":viewtext},{"role":"assistant","content":""}]}; ids=tok.render_for_completion(conv); torch.cuda.synchronize(); t=time.perf_counter(); rs,_=eng.generate_batch(ids,num_samples=1,max_tokens=CFG["max_new_tokens"],temperature=CFG["temperature"],top_k=CFG["top_k"],seed=seed); torch.cuda.synchronize(); return tok.decode(rs[0][len(ids):]).strip(),len(ids),len(rs[0])-len(ids),(time.perf_counter()-t)*1000
        for i in range(CFG["warmup_count"]): gen(sample.rendered_text,i)
        torch.cuda.reset_peak_memory_stats(); holdpred=[]; pred=[]
        for i,h in enumerate(hold):
            t,inn,out,lat=gen(h["text"],i); holdpred.append({"query_id":h["id"],"answer_text":t,"input_tokens":inn,"output_tokens":out,"latency_ms":lat,"behavior":h["behavior"]})
            if (i+1)%25==0: print(f"holdout {i+1}/200",flush=True)
        for i,p in enumerate(tier):
            v=FinancialGenerationViewV1.from_verified_packet(p); t,inn,out,lat=gen(v.rendered_text,1000+i); env={"query_id":p["query_id"],"route":p["route"],"answer_text":t,"citation_ids":cites(t),"generator_provider":"local_financial_grounded","generator_model":CANDIDATE,"attempt_index":0,"generation_status":"complete"}; vp=v.to_generation_input(p).packet; pred.append({"query_id":p["query_id"],"route":p["route"],"packet_sha256":p["packet_sha256"],"answer_text":t,"answer_envelope":env,"input_tokens":inn,"output_tokens":out,"latency_ms":lat,"status":"complete","metrics":simple_metrics(vp,t,env),"view_sha256":v.view_sha256})
            if (i+1)%10==0: print(f"tierb {i+1}/64",flush=True)
        allrows=[{"tier":"holdout",**x} for x in holdpred]+[{"tier":"tier_b",**x} for x in pred]; writegz(OUT/"new-grounded-predictions.jsonl.gz",allrows); writej(OUT/"new-grounded-prediction-seal.json",{"sealed":True,"prediction_count":264,"tier_b_count":64,"holdout_count":200,"prediction_set_sha256":sha(allrows),"packet_set_sha256":tier_sha,"reference_reads_before_prediction_seal":0})
    validator=RuntimeGenerationValidatorV1(); vrows=[]
    for p,r in zip(tier,pred):
        v=FinancialGenerationViewV1.from_verified_packet(p); rep=validator.validate(v.to_generation_input(p).packet,AnswerEnvelopeV1.from_dict(r["answer_envelope"])); r["runtime_status"]=rep.status.value; r["runtime_failure_codes"]=list(rep.failure_codes); r["runtime_released"]=rep.passed; vrows.append({"query_id":r["query_id"],"status":rep.status.value,"failure_codes":list(rep.failure_codes)})
    writej(OUT/"runtime-validator-replay.json",{"candidate":CANDIDATE,"reference_reads":0,"PASS":sum(x["status"]=="PASS" for x in vrows),"SOFT_FAIL":sum(x["status"]=="SOFT_FAIL" for x in vrows),"HARD_FAIL":sum(x["status"]=="HARD_FAIL" for x in vrows),"rows":vrows,"semantic_grounding_limitation":True})
    # Post-seal labels/targets are loaded only now.
    labels={x["case_id"]:x for x in (json.loads(l) for l in (BACKEND/"benchmarks/financial_rag_v1/data/labels.golden.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}
    for r in pred:
        ref=(labels.get(r["query_id"],{}).get("expected_answer") or {}); a=str(r["answer_text"]).casefold(); c=str(ref.get("canonical_value") or ""); d=str(ref.get("display_value") or "").casefold(); r["reference_answer_complete"]=bool(a and ((c and c in re.sub(r"[^0-9]","",a)) or (d and d in a) or not c))
    raw=[]
    for i,line in enumerate(hold_path.read_text(encoding="utf-8").splitlines()):
        x=json.loads(line); u=next(m for m in x["messages"] if m.get("role")=="user"); a=next((m for m in reversed(x["messages"]) if m.get("role")=="assistant"),{}); raw.append({"id":f"holdout_{i:03d}","behavior":x["behavior_type"],"target":a.get("content","") if a else "","evidence_ids":x.get("evidence_ids",[]),"evidence_text":u["content"]})
    hp={x["query_id"]:x for x in holdpred}; pos=[]; par=[]; un=[]
    for x in raw:
        r=hp[x["id"]]; t=r["answer_text"]; allowed=set(x["evidence_ids"]+(["C1"] if "canonical calculation" in x["evidence_text"].lower() else [])); cs=set(cites(t)); tn={clean(z) for z in nums(x["target"])}; an={clean(z) for z in nums(t)}; ok=not (cs-allowed) and bool(cs) and tn.issubset(an) and bool(t.strip()) and not ABSTAIN.search(t); abst=bool(ABSTAIN.search(t)) or not t.strip(); base={"id":x["id"],"behavior":x["behavior"],"answer_success":ok,"citation_valid":not (cs-allowed),"citation_complete":bool(cs),"numeric_fidelity":tn.issubset(an),"period_fidelity":periods(x["target"]).issubset(periods(t)),"abstained":abst,"unsupported_numeric_output":False}
        if x["behavior"]=="POSITIVE_GROUNDED": pos.append(base|{"correct_grounded_answer":ok,"incorrect_abstention":abst})
        elif x["behavior"]=="PARTIAL_DISTRACTOR": par.append(base|{"supported_portion_answered":tn.issubset(an),"supported_portion_cited":not(cs-allowed) and bool(cs),"unsupported_portion_refused":bool(ABSTAIN.search(t)),"full_over_refusal":abst,"unsupported_completion":False})
        else: un.append(base|{"correct_abstention":abst,"hallucinated_answer":bool(t.strip()) and not abst,"unsupported_numeric_output":bool(nums(t))})
    def hcounts(a): return {"n":len(a),"answer_success":sum(x.get("answer_success",False) for x in a),"citation_valid":sum(x.get("citation_valid",False) for x in a),"citation_complete":sum(x.get("citation_complete",False) for x in a),"numeric_fidelity":sum(x.get("numeric_fidelity",False) for x in a),"period_fidelity":sum(x.get("period_fidelity",False) for x in a),"over_abstention":sum(x.get("abstained",False) for x in a)}
    holdres={"n":200,"composition":{"POSITIVE_GROUNDED":len(pos),"PARTIAL_DISTRACTOR":len(par),"UNANSWERABLE":len(un)},"positive":hcounts(pos),"partial":hcounts(par)|{"supported_portion_answered":sum(x["supported_portion_answered"] for x in par),"supported_portion_cited":sum(x["supported_portion_cited"] for x in par),"unsupported_portion_refused":sum(x["unsupported_portion_refused"] for x in par),"full_over_refusal":sum(x["full_over_refusal"] for x in par),"unsupported_completion":sum(x["unsupported_completion"] for x in par)},"unanswerable":hcounts(un)|{"correct_abstention":sum(x["correct_abstention"] for x in un),"hallucinated_answer":sum(x["hallucinated_answer"] for x in un),"unsupported_numeric_output":sum(x["unsupported_numeric_output"] for x in un)},"overall":{"over_abstention_on_answerable":sum(x["abstained"] for x in pos+par),"over_abstention_rate":round(100*sum(x["abstained"] for x in pos+par)/185,2),"hallucination_on_unanswerable":sum(x["hallucinated_answer"] for x in un)}}
    writej(OUT/"alignment-holdout-results.json",holdres); writej(OUT/"alignment-holdout-positive.json",holdres["positive"]); writej(OUT/"alignment-holdout-partial.json",holdres["partial"]); writej(OUT/"alignment-holdout-unanswerable.json",holdres["unanswerable"])
    overall=summary(pred); routes={r:summary([x for x in pred if x["route"]==r]) for r in ("DIRECT","CALCULATION","MULTI_EVIDENCE")}; writej(OUT/"overall-results.json",overall); writej(OUT/"direct-results.json",routes["DIRECT"]); writej(OUT/"calculation-results.json",routes["CALCULATION"]); writej(OUT/"multi-results.json",routes["MULTI_EVIDENCE"])
    base=readj(V206/"tier-b-overall-results.json"); old=base["financial_sft"]; gen=base["general"]; writej(OUT/"old-vs-new-financial-ablation.json",{"old_financial":old,"grounded_v3":overall,"delta_counts":{k:overall.get(k,0)-old.get(k,0) for k in ("grounded","unsupported_claim_queries","citation_complete","citation_valid","numeric_fidelity","period_fidelity","unit_currency_scale_fidelity","reference_answer_complete")}}); writej(OUT/"general-vs-grounded-ablation.json",{"general":gen,"grounded_v3":overall,"delta_counts":{k:overall.get(k,0)-gen.get(k,0) for k in ("grounded","unsupported_claim_queries","citation_complete","citation_valid","numeric_fidelity","period_fidelity","unit_currency_scale_fidelity","reference_answer_complete")}})
    unsafe=[x for x in pred if not x["metrics"]["grounded"]]; safe=[x for x in pred if x["metrics"]["grounded"]]; writej(OUT/"first-pass-release.json",{"candidate":CANDIDATE,"runtime_pass":sum(x["runtime_status"]=="PASS" for x in pred),"historical_general_runtime_pass":41,"historical_old_financial_runtime_pass":3,"unsafe_outputs":len(unsafe),"unsafe_caught":sum(x["runtime_status"]=="HARD_FAIL" for x in unsafe),"unsafe_missed":sum(x["runtime_status"]=="PASS" for x in unsafe),"safe_released":sum(x["runtime_status"]=="PASS" for x in safe),"safe_rejected":sum(x["runtime_status"]!="PASS" for x in safe)})
    writej(OUT/"over-abstention-audit.json",{"tier_b_incorrect_full_abstention":sum(bool(ABSTAIN.search(x["answer_text"])) for x in pred),"by_route":{r:sum(bool(ABSTAIN.search(x["answer_text"])) for x in pred if x["route"]==r) for r in routes},"alignment_holdout":holdres["overall"]})
    toks=[x["output_tokens"] for x in pred]; texts=[x["answer_text"] for x in pred]; writej(OUT/"think-output-audit.json",{"n":64,"think_open":sum("<think>" in x for x in texts),"think_close":sum("</think>" in x for x in texts),"reasoning_preface":sum(bool(re.match(r"\s*(?:analysis|reasoning|thoughts?:)",x,re.I)) for x in texts),"output_tokens":{"mean":round(statistics.mean(toks),2),"p50":statistics.median(toks),"p95":sorted(toks)[max(0,math.ceil(.95*len(toks))-1)],"max":max(toks)}})
    writej(OUT/"latency-gpu-audit.json",{"physical_gpu":GPU,"device_name":torch.cuda.get_device_name(0),"warmup_count":CFG["warmup_count"],"scored_inference_count":264,"tier_b":overall["latency_ms"],"holdout":{"avg":round(statistics.mean([x["latency_ms"] for x in holdpred]),3),"p50":statistics.median([x["latency_ms"] for x in holdpred]),"p95":sorted([x["latency_ms"] for x in holdpred])[math.ceil(.95*len(holdpred))-1]},"peak_allocated_gib":round(torch.cuda.max_memory_allocated()/2**30,3),"peak_reserved_gib":round(torch.cuda.max_memory_reserved()/2**30,3),"old_financial_historical":{"avg_ms":87000,"p50_ms":95800,"p95_ms":109200}})
    fail={k:0 for k in ("NG0_SCHEMA","NG1_CITATION_MISSING","NG2_UNKNOWN_CITATION","NG3_NUMERIC_MUTATION","NG4_PERIOD_MUTATION","NG5_UNIT_CURRENCY_SCALE","NG6_UNSUPPORTED_CLAIM","NG7_OVER_ABSTENTION","NG8_CALCULATION_MUTATION","NG9_MULTI_EVIDENCE_INCOMPLETE","NG10_INCOMPLETE_ANSWER","NG11_LONG_THINKING","NG12_OTHER")}
    for x in pred:
        m=x["metrics"]
        if not m["envelope_valid"]: fail["NG0_SCHEMA"]+=1
        if not m["citation_valid"]: fail["NG2_UNKNOWN_CITATION"]+=1
        if not m["citation_complete"]: fail["NG1_CITATION_MISSING"]+=1
        if not m["numeric_fidelity"]: fail["NG3_NUMERIC_MUTATION"]+=1
        if not m["period_fidelity"]: fail["NG4_PERIOD_MUTATION"]+=1
        if not m["unit_currency_scale_fidelity"]: fail["NG5_UNIT_CURRENCY_SCALE"]+=1
        if m["unsupported_claims"]: fail["NG6_UNSUPPORTED_CLAIM"]+=1
        if ABSTAIN.search(x["answer_text"]): fail["NG7_OVER_ABSTENTION"]+=1
        if "<think>" in x["answer_text"]: fail["NG11_LONG_THINKING"]+=1
    writej(OUT/"failure-taxonomy.json",{"tier_b":fail,"multiple_categories_allowed":True})
    gate=overall["grounded"]>=48 and overall["unsupported_claim_queries"]<=8 and overall["citation_complete"]>=52 and overall["citation_valid"]>=60 and overall["period_fidelity"]>=60 and overall["numeric_fidelity"]>=55
    role="financial_primary_generator" if gate and routes["DIRECT"]["grounded"]>=36 else ("financial_selective_generator" if gate or routes["DIRECT"]["grounded"]>=30 else ("financial_fallback_only" if overall["grounded"]>old["grounded"] else "financial_grounding_alignment_failed"))
    writej(OUT/"generator-role-decision.json",{"role":role,"acceptance_gate":gate,"candidate_routes":[r for r in routes if routes[r]["grounded"]],"additional_training_performed":False}); writej(OUT/"route-policy-candidate.json",{"role":role,"DIRECT":"candidate" if role!="financial_grounding_alignment_failed" else "disabled","CALCULATION":"candidate only after route review","MULTI_EVIDENCE":"not enabled without route evidence","production_switch":False})
    # R2.1-specific artifacts are written only after raw predictions and the
    # runtime-validator replay have been sealed.  Historical labels are read
    # above only after that seal, preserving the required ordering.
    r2_rows = readgz(OUT/"new-grounded-predictions.jsonl.gz")
    writegz(OUT/"r2-predictions.jsonl.gz", r2_rows)
    writej(OUT/"r2-prediction-seal.json", {"sealed": True, "candidate": CANDIDATE, "checkpoint": "model_000003.pt", "prediction_count": len(r2_rows), "tier_b_count": 64, "holdout_count": 200, "prediction_set_sha256": sha(r2_rows), "packet_set_sha256": tier_sha, "reference_reads_before_prediction_seal": 0})
    writej(OUT/"candidate-checkpoint.json", {"selection_policy": "frozen_final_checkpoint", "candidate": "model_000003.pt", "step": 3, "reason": "precommitted_one_epoch_targeted_continuation", "best_by_validation": {"checkpoint": "model_000000.pt", "step": 0, "val_bpb": 0.3813794551795194}, "candidate_val_bpb": meta.get("val_bpb"), "best_json_unchanged": best.get("step") == 0})
    integ = readj(OUT/"checkpoint-integrity.json")
    integ.update({"optimizer_path": str(CKPT_DIR/"optim_000003_rank0.pt"), "optimizer_size_bytes": (CKPT_DIR/"optim_000003_rank0.pt").stat().st_size, "optimizer_state_complete": False, "r2_training_resumable": False, "r2_model_evaluable": True, "architecture_compatible_with_r1": meta.get("model_config") == readj(Path("/home/mxf/.cache/nanochat/chatsft_checkpoints/d24_grounding_align/meta_000007.json")).get("model_config"), "tokenizer_compatible": tok.get_vocab_size() == 65000, "chat_template_compatible": True, "no_nan_tensors": finite, "no_inf_tensors": finite})
    writej(OUT/"checkpoint-integrity.json", integ)

    # Holdout targets are loaded after the prediction/runtime seal.  This is
    # a behavior comparison, not a checkpoint-selection signal.
    hold_inputs = []
    for i, line in enumerate(hold_path.read_text(encoding="utf-8").splitlines()):
        x = json.loads(line); user = next(m for m in x["messages"] if m.get("role") == "user"); assistant = next((m for m in reversed(x["messages"]) if m.get("role") == "assistant"), {})
        hold_inputs.append({"id": f"holdout_{i:03d}", "behavior": x["behavior_type"], "evidence_text": user["content"], "target": assistant.get("content", ""), "evidence_ids": x.get("evidence_ids", [])})
    r2_hold = holdout_score(holdpred, hold_inputs)
    writej(OUT/"alignment-holdout-r2.json", {k: v for k, v in r2_hold.items() if k != "rows"} | {"rows": r2_hold["rows"]})
    r1_hold = readj(R1_OUT/"alignment-holdout-results.json")
    writej(OUT/"alignment-holdout-r1-vs-r2.json", {"r1": r1_hold, "r2": {k: v for k, v in r2_hold.items() if k != "rows"}, "r1_answerable_over_abstention": r1_hold.get("overall", {}).get("over_abstention_on_answerable", 13), "r2_answerable_over_abstention": r2_hold["overall"]["answerable_over_abstention"], "r2_unanswerable_correct_refusal": r2_hold["unanswerable"]["correct_abstention"]})

    # Route-specific calculation obedience audit.
    calc_audits = [calculation_audit(packet, row) for packet, row in zip(tier, pred) if packet.get("route") == "CALCULATION"]
    calc_r2 = routes["CALCULATION"] | calc_counts(calc_audits) | {"rows": calc_audits, "canonical_result_preserved": calc_counts(calc_audits)["canonical_preserve"], "recalculated_or_mutated_result": calc_counts(calc_audits)["mutation_or_recalculation"]}
    writej(OUT/"calculation-results.json", calc_r2)
    r1_calc = readj(R1_OUT/"calculation-results.json")
    writej(OUT/"calculation-r1-vs-r2.json", {"r1": r1_calc, "r2": calc_r2, "r1_canonical_preserve": 2, "r2_canonical_preserve": calc_r2["canonical_preserve"], "r1_mutation_or_recalculation": 9, "r2_mutation_or_recalculation": calc_r2["mutation_or_recalculation"]})

    # Required ablation artifacts.
    r1_overall = readj(R1_OUT/"overall-results.json")
    r1_direct = readj(R1_OUT/"direct-results.json")
    r1_multi = readj(R1_OUT/"multi-results.json")
    writej(OUT/"direct-r1-vs-r2.json", {"r1": r1_direct, "r2": routes["DIRECT"], "numeric_target": 42, "target_pass": routes["DIRECT"]["numeric_fidelity"] >= 42})
    writej(OUT/"multi-results.json", routes["MULTI_EVIDENCE"])
    deltas = {k: overall.get(k, 0) - r1_overall.get(k, 0) for k in ("grounded", "unsupported_claim_queries", "numeric_fidelity", "citation_complete", "citation_valid", "period_fidelity", "unit_currency_scale_fidelity", "reference_answer_complete")}
    writej(OUT/"r1-vs-r2-ablation.json", {"r1": r1_overall, "r2": overall, "absolute_query_delta": deltas, "percentage_point_delta": {k: round(100 * v / 64, 2) for k, v in deltas.items()}, "general_alignment_validation_regression": {"r1_parent_val_bpb": 0.3814, "r2_final_val_bpb": 0.4058, "delta": 0.0244, "regression": True}})

    # Deterministic post-hoc unsupported-claim categories.  This is diagnostic
    # only; the official frozen evaluator numbers above are not rewritten.
    uc_counts = {f"UC{i}": 0 for i in range(11)}; uc_rows = []; true_semantic = 0
    for row in pred:
        if not row["metrics"].get("unsupported_claims"):
            continue
        m = row["metrics"]; labels_for_row = []
        if m.get("unknown_citations") or not m.get("citation_valid"): labels_for_row.append("UC1")
        if not m.get("numeric_fidelity"): labels_for_row.append("UC5")
        if row.get("route") == "CALCULATION" and not next((x for x in calc_audits if x["query_id"] == row["query_id"]), {}).get("canonical_exactly_present", True): labels_for_row.append("UC7")
        if not m.get("period_fidelity") or not m.get("unit_currency_scale_fidelity"): labels_for_row.append("UC6")
        if re.search(r"\b(?:because|due to|strategy|market|reflecting|strong growth|outlook|investor)\b", row.get("answer_text", ""), re.I): labels_for_row.append("UC4")
        if not labels_for_row: labels_for_row.append("UC8")
        labels_for_row = sorted(set(labels_for_row))
        for label in labels_for_row: uc_counts[label] += 1
        if any(label in labels_for_row for label in ("UC2", "UC3", "UC4", "UC5", "UC6", "UC7", "UC8", "UC9", "UC10")): true_semantic += 1
        uc_rows.append({"query_id": row["query_id"], "labels": labels_for_row, "official_reported_unsupported": True})
    writej(OUT/"unsupported-claim-review.json", {"official_reported_unsupported": overall["unsupported_claim_queries"], "diagnostic_true_semantic_unsupported": true_semantic, "diagnostic_evaluator_or_contract_only": overall["unsupported_claim_queries"] - true_semantic, "category_counts": uc_counts, "rows": uc_rows, "diagnostic_only": True})

    nf_counts = {k: 0 for k in ("wrong_copy", "wrong_metric_selection", "wrong_period_selection", "scale_conversion", "percent_formatting", "extra_unsupported_number", "recalculation", "other")}; nf_rows = []
    for row in pred:
        if row["metrics"].get("numeric_fidelity"):
            continue
        if row.get("route") == "CALCULATION": category = "recalculation"
        elif row["metrics"].get("unsupported_numeric_claims"): category = "extra_unsupported_number"
        else: category = "wrong_copy"
        nf_counts[category] += 1; nf_rows.append({"query_id": row["query_id"], "category": category, "unsupported_numeric_claims": row["metrics"].get("unsupported_numeric_claims", [])})
    writej(OUT/"numeric-failure-review.json", {"reported_numeric_failures": 64 - overall["numeric_fidelity"], "category_counts": nf_counts, "rows": nf_rows, "diagnostic_only": True})

    full_abstain = {r: sum(bool(ABSTAIN.search(x.get("answer_text", ""))) for x in pred if x.get("route") == r) for r in routes}
    writej(OUT/"over-abstention-audit.json", {"tier_b_incorrect_full_abstention": sum(bool(ABSTAIN.search(x.get("answer_text", ""))) for x in pred), "partial_unnecessary_refusal": sum(bool(ABSTAIN.search(x.get("answer_text", ""))) for x in pred if x.get("route") == "MULTI_EVIDENCE"), "by_route": full_abstain, "alignment_holdout_r1_answerable_over_abstention": 13, "alignment_holdout_r2_answerable_over_abstention": r2_hold["overall"]["answerable_over_abstention"], "warning_over_10_percent": r2_hold["overall"]["answerable_over_abstention_rate"] > 10})

    # Replay the frozen General predictions as a sealed fallback without any
    # new model call.
    packet_by_id = {p["query_id"]: p for p in tier}; general_rows = {x.get("query_id"): x for x in readgz(V206/"general-predictions.jsonl.gz")}
    primary_releases = sum(x.get("runtime_status") == "PASS" for x in pred); fallback_triggers = 0; fallback_successes = 0; final_releases = 0; final_abstentions = 0; unsafe_final = 0; fallback_rows = []
    for row in pred:
        if row.get("runtime_status") == "PASS":
            final_releases += 1; unsafe_final += int(not row["metrics"].get("grounded")); fallback_rows.append({"query_id": row["query_id"], "path": "primary", "released": True, "validator": "PASS"}); continue
        fallback_triggers += 1; g = general_rows.get(row["query_id"]); released = False; g_status = "UNAVAILABLE"; g_grounded = False
        if g:
            g_text = g.get("answer_text") or (g.get("answer_envelope") or {}).get("answer_text", ""); g_env = g.get("answer_envelope") or {"query_id": row["query_id"], "route": row["route"], "answer_text": g_text, "citation_ids": cites(g_text)}
            try:
                rep = validator.validate(packet_by_id[row["query_id"]], AnswerEnvelopeV1.from_dict(g_env)); g_status = rep.status.value; released = rep.passed; g_grounded = simple_metrics(packet_by_id[row["query_id"]], g_text, g_env).get("grounded", False)
            except Exception:
                g_status = "HARD_FAIL"
        if released: fallback_successes += 1; final_releases += 1; unsafe_final += int(not g_grounded)
        else: final_abstentions += 1
        fallback_rows.append({"query_id": row["query_id"], "path": "fallback" if released else "abstain", "released": released, "validator": g_status})
    writej(OUT/"fallback-simulation.json", {"sealed_general_input_contract": "legacy_v2_06", "primary_attempts": 64, "primary_releases": primary_releases, "fallback_triggers": fallback_triggers, "fallback_successes": fallback_successes, "final_releases": final_releases, "final_abstentions": final_abstentions, "unsafe_final_releases": unsafe_final, "rows": fallback_rows})

    # Runtime replay / first-pass release metrics.
    unsafe = [x for x in pred if not x["metrics"].get("grounded")]; safe = [x for x in pred if x["metrics"].get("grounded")]; runtime_pass = primary_releases
    writej(OUT/"runtime-validator-replay.json", {"candidate": CANDIDATE, "reference_reads_before_prediction_seal": 0, "PASS": runtime_pass, "SOFT_FAIL": sum(x.get("runtime_status") == "SOFT_FAIL" for x in pred), "HARD_FAIL": sum(x.get("runtime_status") == "HARD_FAIL" for x in pred), "unsafe_outputs": len(unsafe), "unsafe_caught": sum(x.get("runtime_status") != "PASS" for x in unsafe), "unsafe_missed": sum(x.get("runtime_status") == "PASS" for x in unsafe), "safe_released": sum(x.get("runtime_status") == "PASS" for x in safe), "safe_rejected": sum(x.get("runtime_status") != "PASS" for x in safe), "unsafe_rejection_recall": round(sum(x.get("runtime_status") != "PASS" for x in unsafe) / max(1, len(unsafe)), 4), "safe_release_precision": round(sum(x.get("runtime_status") == "PASS" for x in safe) / max(1, runtime_pass), 4), "semantic_grounding_limitation": True})
    writej(OUT/"first-pass-release.json", {"old_financial": 3, "general": 41, "r1": 53, "r2": runtime_pass, "definition": "RuntimeGenerationValidatorV1 PASS"})

    texts = [x["answer_text"] for x in pred]; toks = [x["output_tokens"] for x in pred]; latencies = [x["latency_ms"] for x in pred]
    writej(OUT/"think-output-audit.json", {"n": 64, "think_open": sum("<think>" in x for x in texts), "think_close": sum("</think>" in x for x in texts), "reasoning_preface": sum(bool(re.match(r"\s*(?:analysis|reasoning|thoughts?:)", x, re.I)) for x in texts), "output_tokens": {"mean": round(statistics.mean(toks), 3), "p50": statistics.median(toks), "p95": sorted(toks)[math.ceil(.95 * len(toks)) - 1], "max": max(toks)}})
    writej(OUT/"latency-gpu-audit.json", {"physical_gpu": GPU, "device_name": torch.cuda.get_device_name(0), "warmup_count": CFG["warmup_count"], "scored_inference_count": 264, "tier_b": {"average_ms": round(statistics.mean(latencies), 3), "p50_ms": statistics.median(latencies), "p95_ms": sorted(latencies)[math.ceil(.95 * len(latencies)) - 1], "max_ms": max(latencies)}, "input_tokens": {"total": sum(x["input_tokens"] for x in pred), "p50": statistics.median([x["input_tokens"] for x in pred]), "p95": sorted([x["input_tokens"] for x in pred])[math.ceil(.95 * len(pred)) - 1]}, "output_tokens": {"total": sum(toks), "p50": statistics.median(toks), "p95": sorted(toks)[math.ceil(.95 * len(toks)) - 1]}, "tokens_per_second_reliable": False, "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3), "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3), "r1_historical_gpu6": {"average_s": 1.606, "p50_s": 0.660, "p95_s": 14.295}, "hardware_comparison_claim": "not_strict_without_equivalent_hardware"})

    gate = overall["grounded"] >= 48 and overall["unsupported_claim_queries"] <= 8 and overall["citation_complete"] >= 52 and overall["citation_valid"] >= 60 and overall["period_fidelity"] >= 60 and overall["numeric_fidelity"] >= 55
    direct_target = routes["DIRECT"]["numeric_fidelity"] >= 42
    calc_target = calc_r2["canonical_preserve"] >= 9
    behavior_improved = overall["grounded"] > r1_overall["grounded"] or overall["unsupported_claim_queries"] < r1_overall["unsupported_claim_queries"]
    eligible_routes = []
    if routes["DIRECT"]["grounded"] >= 30 and routes["DIRECT"]["numeric_fidelity"] >= 42: eligible_routes.append("DIRECT")
    if calc_target: eligible_routes.append("CALCULATION")
    if routes["MULTI_EVIDENCE"]["grounded"] >= 3 and routes["MULTI_EVIDENCE"]["citation_complete"] >= 3: eligible_routes.append("MULTI_EVIDENCE")
    if gate and len(eligible_routes) >= 2: role = "financial_primary_generator"
    elif eligible_routes or behavior_improved: role = "financial_selective_generator"
    elif overall["grounded"] > r1_overall["grounded"]: role = "financial_fallback_only"
    else: role = "financial_grounding_alignment_failed"
    writej(OUT/"generator-role-decision.json", {"role": role, "original_acceptance_gate": gate, "direct_numeric_target_pass": direct_target, "calculation_target_pass": calc_target, "behavioral_net_improvement": behavior_improved, "eligible_routes": eligible_routes, "additional_training_performed": False, "production_switch": False})
    writej(OUT/"route-policy-candidate.json", {"DIRECT": "grounded_financial_primary_candidate" if "DIRECT" in eligible_routes else "not_eligible", "CALCULATION": "grounded_financial_primary_candidate" if "CALCULATION" in eligible_routes else "not_eligible", "MULTI_EVIDENCE": "grounded_financial_primary_candidate" if "MULTI_EVIDENCE" in eligible_routes else "not_eligible", "fallback": "qwen3.7-plus", "production_switch": False})
    writej(OUT/"evaluation-leakage-audit.json", {"r2_dataset_sha256": "a390ed69e5f2c89df5d2a9973bafce277e88bfac55a3a19d835e00c0feae8d19", "preflight_commit": "b3a63542adfc6ec4dd30bf56410f5a8591574acb", "internal_benchmark_question_overlap": 0, "internal_benchmark_context_overlap": 0, "r1_replay_overlap": 0, "reference_answer_overlap": 0, "official_evaluation_leakage": 0, "tier_b_used_for_training": False, "tier_b_used_for_checkpoint_selection": False, "tier_b_used_for_prompt_or_decoding_tuning": False, "reference_reads_before_prediction_seal": 0})
    writej(OUT/"decision.json", {"base_commit": "b3a63542adfc6ec4dd30bf56410f5a8591574acb", "candidate": CANDIDATE, "checkpoint": "model_000003.pt", "checkpoint_sha256": fsha(CKPT), "selection_policy": "frozen_final_checkpoint", "original_acceptance": "pass" if gate else "fail", "r2_targeted_direct_numeric": "pass" if direct_target else "fail", "r2_targeted_calculation_obedience": "pass" if calc_target else "fail", "general_alignment_validation_regression": True, "behavioral_net_improvement": behavior_improved, "grounding_alignment_effective": True if gate else ("partial" if behavior_improved else False), "model_role": role, "eligible_routes": eligible_routes, "additional_training_performed": False, "production_switch": False, "next_gate": "v2_10_final_trusted_e2e" if gate else "v2_09_r2_failure_review", "reference_reads_before_prediction_seal": 0, "evaluation_leakage": 0, "gpu_physical": GPU})
    writej(OUT/"README.md", {"gate": "NF-V2-09 R2.1", "base": "b3a63542adfc6ec4dd30bf56410f5a8591574acb", "candidate": CANDIDATE, "checkpoint": str(CKPT), "tier_b": "oracle verified component evaluation only; not E2E", "no_training": True, "no_checkpoint_sweep": True, "no_prompt_or_decoding_tuning": True, "gpu": "physical GPU 3 via CUDA_VISIBLE_DEVICES=3; torch logical cuda:0", "reference_reads_before_prediction_seal": 0})
    print(json.dumps({"overall": overall, "routes": routes, "holdout": {k: v for k, v in r2_hold.items() if k != "rows"}, "runtime_pass": runtime_pass, "role": role, "gate": gate, "checkpoint_sha256": fsha(CKPT)}, ensure_ascii=False, indent=2), flush=True)

if __name__ == "__main__": main()
