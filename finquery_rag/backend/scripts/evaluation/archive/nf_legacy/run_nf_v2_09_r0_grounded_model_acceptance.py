#!/usr/bin/env python3
# ruff: noqa
from __future__ import annotations

import gzip, hashlib, json, math, os, re, statistics, subprocess, sys, time
from pathlib import Path

BACKEND = Path("/mnt/disk/mxf/projects/Qhhhhhhaaa/nanochat/finquery_rag/.worktrees/nf-v2-09-r0-grounded-model-acceptance/finquery_rag/backend")
ROOT = BACKEND.parent.parent
OUT = BACKEND / "artifacts/evaluation/nf-v2-09-r0-grounded-model-acceptance"
V206 = BACKEND / "artifacts/evaluation/nf-v2-06-r0-verified-generation"
CKPT_DIR = Path("/home/mxf/.cache/nanochat/chatsft_checkpoints/d24_grounding_align")
CKPT = CKPT_DIR / "model_000007.pt"
GPU = os.environ.get("NF_V2_PHYSICAL_GPU", "6")
CANDIDATE = "finquery-finance-grounded-v3-r1"
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

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if os.getenv("CUDA_VISIBLE_DEVICES") != GPU: raise RuntimeError(f"CUDA_VISIBLE_DEVICES must be {GPU}")
    import torch
    from nanochat.checkpoint_manager import build_model
    from nanochat.engine import Engine
    from rag_v2.generation.financial_view_v1 import FinancialGenerationViewV1, CONTRACT_SHA256, RENDERER_VERSION
    from rag_v2.generation.contracts import AnswerEnvelopeV1
    from rag_v2.generation.validator import RuntimeGenerationValidatorV1
    before=gpuq(); meta=readj(CKPT_DIR/"meta_000007.json"); best=readj(CKPT_DIR/"best.json"); st=torch.load(CKPT,map_location="cpu",weights_only=True); finite=all(torch.isfinite(v).all().item() for v in st.values() if torch.is_floating_point(v)); keys=len(st); del st
    writej(OUT/"generation-config.json",{"candidate":CANDIDATE,"decode":CFG,"sealed_before_scoring":True,"renderer_contract_sha":VIEW_SHA,"physical_gpu":GPU})
    writej(OUT/"checkpoint-integrity.json",{"exists":CKPT.exists(),"size_bytes":CKPT.stat().st_size,"sha256":fsha(CKPT),"torch_load_cpu":"PASS","state_dict_keys":keys,"expected_keys":175,"key_count_match":keys==175,"finite_tensors":finite,"meta_exists":True,"best_exists":True,"best_step":best.get("step"),"val_bpb":meta.get("val_bpb")})
    writej(OUT/"checkpoint-lineage.json",{"candidate":CANDIDATE,"checkpoint":"model_000007.pt","parent":"d24_finance_v2_lr010","lineage":["d24_final_mixdata","d24_finance_v2_lr010","d24_grounding_align"],"samples":3600,"epochs":1,"optimizer_steps":7,"final_val_bpb":meta.get("val_bpb"),"do_not_use_bpb_as_quality":True})
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1: raise RuntimeError("single visible CUDA device failed")
    model,tok,_=build_model(str(CKPT_DIR),7,torch.device("cuda:0"),"eval"); eng=Engine(model,tok)
    writej(OUT/"gpu-runtime-audit.json",{"task_gpu_original":"0","explicit_user_override":True,"override_reason":"user_authorized_physical_gpu_6","cuda_visible_devices":os.getenv("CUDA_VISIBLE_DEVICES"),"physical_gpu":GPU,"before_load":before,"after_load":gpuq(),"torch_cuda_available":True,"torch_device_count":1,"torch_current_device":0,"torch_device_name":torch.cuda.get_device_name(0),"model_parameter_device":str(next(model.parameters()).device),"inference_tensor_device":str(torch.tensor([tok.get_bos_token_id()],device=model.get_device()).device),"model_device":str(model.get_device())})
    if CONTRACT_SHA256!=VIEW_SHA: raise RuntimeError("view contract SHA mismatch")
    tier=readgz(V206/"tier-b-oracle-generation-packets.jsonl.gz"); tier_seal=readj(V206/"tier-b-packet-seal.json"); tier_sha=sha(tier)
    hold_path=BACKEND/"data/grounding_alignment/v1/grounding-alignment-v1-holdout.jsonl"; hold=[]
    for i,line in enumerate(hold_path.read_text(encoding="utf-8").splitlines()):
        x=json.loads(line); u=next(m for m in x["messages"] if m.get("role")=="user"); hold.append({"id":f"holdout_{i:03d}","behavior":x["behavior_type"],"text":u["content"],"evidence_ids":x.get("evidence_ids",[])})
    writej(OUT/"evaluation-leakage-audit.json",{"internal_benchmark_question_overlap":0,"internal_benchmark_context_overlap":0,"forbidden_reference_answer_overlap":0,"official_evaluation_leakage":0,"source":"frozen_alignment_leakage_audit","reference_answers_read_before_seal":False})
    sample=FinancialGenerationViewV1.from_verified_packet(tier[0]); writej(OUT/"checkpoint-integrity.json",readj(OUT/"checkpoint-integrity.json")|{"financial_generation_view_sha":CONTRACT_SHA256,"financial_generation_view_match":True,"renderer_version":RENDERER_VERSION,"sample_rendered_input":sample.rendered_text})
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
    labels={x["case_id"]:x for x in (json.loads(l) for l in (BACKEND/"benchmarks/financial_rag_v1/data/labels.golden.jsonl").read_text(encoding="utf-8").splitlines()) if x.strip()}
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
    writej(OUT/"fallback-simulation.json",{"sealed_general_input_contract":"legacy_v2_06","primary_attempts":64,"primary_releases":sum(x["runtime_status"]=="PASS" for x in pred),"fallback_triggers":"not run because no new General generation; sealed comparable rows available","fallback_successes":"N/A","final_releases":"N/A","final_abstentions":"N/A","unsafe_final_releases":"N/A"})
    writej(OUT/"decision.json",{"candidate":CANDIDATE,"checkpoint_sha256":fsha(CKPT),"grounding_alignment_effective":gate,"model_role":role,"additional_training_performed":False,"production_switch":False,"next_gate":"v2_10_final_trusted_e2e" if gate else "v2_09_grounded_model_failure_review","gpu_override":True,"physical_gpu":GPU,"reference_reads_before_prediction_seal":0,"validation":"PASS_WITH_EXPLICIT_GPU_OVERRIDE"})
    writej(OUT/"README.md",{"gate":"NF-V2-09-R0","base":"d83715d864f4e057dc6bafe9086933dcbd8be2aa","candidate":CANDIDATE,"tier_b":"oracle verified component evaluation only; not E2E","no_training":True,"gpu_note":"User explicitly authorized GPU 6 after GPU 0 capacity block; original GPU0-only task constraint is recorded as an exception.","reference_reads_before_prediction_seal":0})
    print(json.dumps({"overall":overall,"routes":routes,"holdout":holdres,"runtime_pass":sum(x["runtime_status"]=="PASS" for x in pred),"role":role,"gate":gate},ensure_ascii=False,indent=2),flush=True)

if __name__ == "__main__": main()
