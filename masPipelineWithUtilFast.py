

import os
import re
import json
import argparse
import time
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


# ---------------------------
# Prompts
# ---------------------------

ADVOCATE_SYSTEM = """You are Agent 1: The Advocate.
Argue FOR classifying the claim as Y02 (green technology).

Rules:
- Use only evidence from the claim text. Do not invent facts.
- Quote short snippets from the claim as evidence.
- Be concise and technical.
- Output MUST follow this exact format:

key_green_signals:
- <bullet>
- <bullet>
y02_reasoning: <3-6 sentences>
confidence: Low|Medium|High
questions:
- <bullet or 'None'>
"""

SKEPTIC_SYSTEM = """You are Agent 2: The Skeptic.
Argue AGAINST classifying the claim as Y02 (green technology).
Look for greenwashing, generic optimization, and missing environmental mechanisms.

Rules:
- Use only evidence from the claim text. Do not invent facts.
- Quote short snippets from the claim as evidence.
- Be concise and technical.
- Output MUST follow this exact format:

reasons_not_y02:
- <bullet>
- <bullet>
greenwashing_flags:
- <bullet or 'None'>
counter_reasoning: <3-6 sentences>
confidence: Low|Medium|High
what_would_change_my_mind:
- <bullet>
"""

JUDGE_SYSTEM = """You are Agent 3: The Judge.
Decide whether the claim belongs to Y02 (green technology) by weighing:
(1) the claim text,
(2) the Advocate argument,
(3) the Skeptic argument.

Decision rules:
- Prefer explicit technical environmental mechanisms over vague benefits.
- Use only information present in the claim and the two arguments.
- If uncertainty is high or arguments conflict strongly, set human_review_required=true.

Output MUST be ONLY a single JSON object (no markdown, no code fences, no extra text).
The JSON MUST match this schema:
{
  "claim_id": "string",
  "is_green": boolean,
  "y02_relevance": "direct"|"indirect"|"none",
  "confidence": number (0.0 to 1.0),
  "decision_rationale": "string (10-600 chars)",
  "key_evidence": [{"quote":"string","why_it_matters":"string"}] (1-5 items),
  "advocate_summary": "string",
  "skeptic_summary": "string",
  "disagreement": boolean,
  "human_review_required": boolean
}
"""

USER_TEMPLATE = """Claim (doc_id={doc_id}):
{claim_text}
"""

JUDGE_USER_TEMPLATE = """Claim (doc_id={doc_id}):
{claim_text}

Advocate argument:
{advocate_output}

Skeptic argument:
{skeptic_output}
"""


# ---------------------------
# Utilities
# ---------------------------

def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

CONF_RE = re.compile(r"confidence:\s*(Low|Medium|High)\b", re.IGNORECASE)

def extract_confidence(text: str) -> str:
    m = CONF_RE.search(text or "")
    return m.group(1).capitalize() if m else "Unknown"

def conf_score(conf: str) -> int:
    return {"Unknown": 0, "Low": 1, "Medium": 2, "High": 3}.get(conf, 0)

def safe_float(x) -> Optional[float]:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        return float(x)
    except Exception:
        return None

def normalize_input_path(path: str) -> str:
    if os.path.isfile(path):
        return path
    if os.path.isfile(path + ".csv"):
        return path + ".csv"
    raise FileNotFoundError(f"Could not find input file: {path} (or {path}.csv)")

def add_slurm_job_id_suffix(output_path: str) -> str:
    job_id = os.getenv("SLURM_JOB_ID")
    if not job_id:
        return output_path
    root, ext = os.path.splitext(output_path)
    if ext.lower() != ".csv":
        return f"{output_path}_{job_id}.csv"
    return f"{root}_{job_id}{ext}"

def build_messages(system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
    return [{"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}]

def precompute_llama3_tokens_supported(tokenizer) -> bool:
    bos = "<|begin_of_text|>"
    sh = "<|start_header_id|>"
    eh = "<|end_header_id|>"
    eot = "<|eot_id|>"
    vocab = tokenizer.get_vocab()
    return all(t in vocab for t in [bos, sh, eh, eot])

def render_chat_prompt(tokenizer, messages, has_llama3_tokens: bool) -> str:
    # If chat template exists, use it
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Otherwise Llama-3-ish tokens if present
    if has_llama3_tokens:
        bos = "<|begin_of_text|>"
        sh = "<|start_header_id|>"
        eh = "<|end_header_id|>"
        eot = "<|eot_id|>"
        sys_msg = messages[0]["content"]
        usr_msg = messages[1]["content"]
        return (
            f"{bos}"
            f"{sh}system{eh}\n{sys_msg}{eot}"
            f"{sh}user{eh}\n{usr_msg}{eot}"
            f"{sh}assistant{eh}\n"
        )

    # Final fallback
    sys_msg = messages[0]["content"]
    usr_msg = messages[1]["content"]
    return f"SYSTEM:\n{sys_msg}\n\nUSER:\n{usr_msg}\n\nASSISTANT:\n"

@torch.inference_mode()
def generate_batch(
    model,
    tokenizer,
    has_llama3_tokens: bool,
    system_prompt: str,
    user_prompts: List[str],
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> List[str]:
    prompts = [
        render_chat_prompt(tokenizer, build_messages(system_prompt, up), has_llama3_tokens)
        for up in user_prompts
    ]

    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,  # safety cap
    )
    enc = {k: v.to(model.device) for k, v in enc.items()}
    input_lens = enc["attention_mask"].sum(dim=1).tolist()

    if do_sample:
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    else:
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    results: List[str] = []
    for i in range(out.size(0)):
        gen_ids = out[i, input_lens[i]:]
        results.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())
    return results

def extract_first_json_object(text: str) -> Optional[str]:
    if not text:
        return None
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    return None

def validate_judge_obj(obj: Dict[str, Any]) -> Tuple[bool, str]:
    required = [
        "claim_id", "is_green", "y02_relevance", "confidence", "decision_rationale",
        "key_evidence", "advocate_summary", "skeptic_summary",
        "disagreement", "human_review_required"
    ]
    for k in required:
        if k not in obj:
            return False, f"Missing field: {k}"

    if not isinstance(obj["claim_id"], str) or not obj["claim_id"]:
        return False, "claim_id must be non-empty string"
    if not isinstance(obj["is_green"], bool):
        return False, "is_green must be boolean"
    if obj["y02_relevance"] not in ("direct", "indirect", "none"):
        return False, "y02_relevance must be one of direct|indirect|none"
    if not (isinstance(obj["confidence"], (int, float)) and 0.0 <= float(obj["confidence"]) <= 1.0):
        return False, "confidence must be number in [0,1]"
    if not isinstance(obj["decision_rationale"], str) or not (10 <= len(obj["decision_rationale"]) <= 600):
        return False, "decision_rationale length must be 10..600"
    if not isinstance(obj["key_evidence"], list) or not (1 <= len(obj["key_evidence"]) <= 5):
        return False, "key_evidence must be list with 1..5 items"
    for it in obj["key_evidence"]:
        if not isinstance(it, dict):
            return False, "key_evidence items must be objects"
        if set(it.keys()) != {"quote", "why_it_matters"}:
            return False, "key_evidence items must have only quote and why_it_matters"
    if not isinstance(obj["advocate_summary"], str) or not (5 <= len(obj["advocate_summary"]) <= 350):
        return False, "advocate_summary length must be 5..350"
    if not isinstance(obj["skeptic_summary"], str) or not (5 <= len(obj["skeptic_summary"]) <= 350):
        return False, "skeptic_summary length must be 5..350"
    if not isinstance(obj["disagreement"], bool):
        return False, "disagreement must be boolean"
    if not isinstance(obj["human_review_required"], bool):
        return False, "human_review_required must be boolean"

    return True, "ok"

def disagreement_heuristic(p_green: Optional[float], u: Optional[float], adv_conf: str, skp_conf: str) -> Tuple[bool, str]:
    if conf_score(adv_conf) >= 2 and conf_score(skp_conf) >= 2:
        return True, "Both agents confident (>=Medium) in opposing roles."
    reasons = []
    if p_green is not None and 0.40 <= p_green <= 0.60:
        reasons.append(f"p_green near boundary ({p_green:.3f}).")
    if u is not None and u >= 0.50:
        reasons.append(f"High uncertainty u ({u:.3f}).")
    if reasons:
        return True, " ".join(reasons)
    return False, "No flag."

def chunks(lst: List[Any], n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/hitl_green_100", help="Input CSV path (with or without .csv)")
    parser.add_argument("--output", default="data/hitl_green_100_mas_output.csv", help="Output CSV path (job id suffix auto-added)")

    parser.add_argument("--base_model", default=os.getenv("BASE_MODEL_ID", "unsloth/llama-3-8b-bnb-4bit"))
    parser.add_argument("--adv_adapter", default=os.getenv("ADVOCATE_ADAPTER_DIR", "./adapter_advocate"))
    parser.add_argument("--skp_adapter", default=os.getenv("SKEPTIC_ADAPTER_DIR", "./adapter_skeptic"))

    parser.add_argument("--batch_size", type=int, default=4, help="Increase to 6-8 if VRAM allows")
    parser.add_argument("--status_every_batches", type=int, default=2, help="Print status every N batches")

    # Token limits: lower = much faster
    parser.add_argument("--adv_max_new_tokens", type=int, default=140)
    parser.add_argument("--skp_max_new_tokens", type=int, default=140)
    parser.add_argument("--judge_max_new_tokens", type=int, default=180)

    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)

    args = parser.parse_args()

    input_path = normalize_input_path(args.input)
    output_path = add_slurm_job_id_suffix(args.output)

    df = pd.read_csv(input_path)
    required_cols = ["doc_id", "text", "p_green", "u", "human_label_is_green", "human_notes"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Speed knobs
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # Prefer BF16 on L4/Hopper/Ampere+
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    has_llama3_tokens = precompute_llama3_tokens_supported(tokenizer)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=compute_dtype,
        attn_implementation="sdpa",
    )
    base_model.config.use_cache = True
    base_model.eval()

    # Single model that can do advocate/skeptic, and judge via disable_adapter()
    model_with_adapters = PeftModel.from_pretrained(base_model, args.adv_adapter, adapter_name="advocate")
    model_with_adapters.load_adapter(args.skp_adapter, adapter_name="skeptic")
    model_with_adapters.eval()

    log(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'} | dtype={compute_dtype}")
    log(f"Loaded {len(df)} rows from {input_path}")
    log(f"Output: {output_path}")
    log(f"Batch size: {args.batch_size}")
    log(f"Tokens: adv={args.adv_max_new_tokens} skp={args.skp_max_new_tokens} judge={args.judge_max_new_tokens}")
    log("Starting batched MAS inference...")

    rows = df.to_dict(orient="records")
    rows_out: List[Dict[str, Any]] = []
    start_all = time.time()

    batch_iter = list(chunks(rows, args.batch_size))
    iterator = tqdm(batch_iter, total=len(batch_iter), desc="MAS batches", mininterval=10) if tqdm else batch_iter

    for b_idx, batch in enumerate(iterator, start=1):
        t_batch = time.time()

        doc_ids = [str(r["doc_id"]) for r in batch]
        texts = [str(r["text"]) for r in batch]
        p_greens = [safe_float(r.get("p_green")) for r in batch]
        us = [safe_float(r.get("u")) for r in batch]

        user_prompts = [USER_TEMPLATE.format(doc_id=d, claim_text=t) for d, t in zip(doc_ids, texts)]

        # --- Advocate (batched) ---
        model_with_adapters.set_adapter("advocate")
        adv_outs = generate_batch(
            model_with_adapters, tokenizer, has_llama3_tokens,
            ADVOCATE_SYSTEM, user_prompts,
            max_new_tokens=args.adv_max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        adv_confs = [extract_confidence(x) for x in adv_outs]

        # --- Skeptic (batched) ---
        model_with_adapters.set_adapter("skeptic")
        skp_outs = generate_batch(
            model_with_adapters, tokenizer, has_llama3_tokens,
            SKEPTIC_SYSTEM, user_prompts,
            max_new_tokens=args.skp_max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        skp_confs = [extract_confidence(x) for x in skp_outs]

        # --- Judge inputs ---
        judge_user_prompts = [
            JUDGE_USER_TEMPLATE.format(
                doc_id=d,
                claim_text=t,
                advocate_output=ao,
                skeptic_output=so,
            )
            for d, t, ao, so in zip(doc_ids, texts, adv_outs, skp_outs)
        ]

        # --- Judge (batched, greedy, adapters DISABLED) ---
        with model_with_adapters.disable_adapter():
            judge_raws = generate_batch(
                model_with_adapters, tokenizer, has_llama3_tokens,
                JUDGE_SYSTEM, judge_user_prompts,
                max_new_tokens=args.judge_max_new_tokens,
                do_sample=False,   # greedy
                temperature=0.0,
                top_p=1.0,
            )

        judge_objs: List[Optional[Dict[str, Any]]] = [None] * len(batch)
        judge_jsons: List[str] = [""] * len(batch)
        invalid_indices: List[int] = []

        for i, raw in enumerate(judge_raws):
            j = extract_first_json_object(raw) or raw
            judge_jsons[i] = j
            try:
                obj = json.loads(j)
                ok, _ = validate_judge_obj(obj)
                if ok:
                    judge_objs[i] = obj
                else:
                    invalid_indices.append(i)
            except Exception:
                invalid_indices.append(i)

        # --- Repair invalid judge outputs (batched, adapters DISABLED) ---
        if invalid_indices:
            repair_system = JUDGE_SYSTEM + "\n\nYou previously returned invalid JSON. Re-output ONLY a single valid JSON object matching the schema. No extra text."
            repair_prompts = [judge_user_prompts[i] for i in invalid_indices]

            with model_with_adapters.disable_adapter():
                repair_raws = generate_batch(
                    model_with_adapters, tokenizer, has_llama3_tokens,
                    repair_system, repair_prompts,
                    max_new_tokens=args.judge_max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                    top_p=1.0,
                )

            for idx_local, raw in enumerate(repair_raws):
                i = invalid_indices[idx_local]
                j = extract_first_json_object(raw) or raw
                judge_jsons[i] = j
                try:
                    obj = json.loads(j)
                    ok, _ = validate_judge_obj(obj)
                    if ok:
                        judge_objs[i] = obj
                except Exception:
                    pass

        # --- Build output rows ---
        for i, r in enumerate(batch):
            disagree_flag, reason = disagreement_heuristic(p_greens[i], us[i], adv_confs[i], skp_confs[i])
            jo = judge_objs[i] or {}

            rows_out.append({
                "doc_id": doc_ids[i],
                "text": texts[i],
                "p_green": p_greens[i],
                "u": us[i],

                "advocate_confidence": adv_confs[i],
                "advocate_output": adv_outs[i],

                "skeptic_confidence": skp_confs[i],
                "skeptic_output": skp_outs[i],

                "judge_is_green": jo.get("is_green"),
                "judge_confidence": jo.get("confidence"),
                "judge_y02_relevance": jo.get("y02_relevance"),
                "judge_disagreement": jo.get("disagreement"),
                "judge_human_review_required": jo.get("human_review_required"),
                "judge_rationale": jo.get("decision_rationale"),
                "judge_json_raw": judge_jsons[i],

                "disagreement_flag": disagree_flag,
                "flag_reason": reason,

                "human_label_is_green": r.get("human_label_is_green"),
                "human_notes": r.get("human_notes"),
            })

        # --- Logging ---
        if b_idx == 1 or (b_idx % args.status_every_batches == 0):
            done = min(b_idx * args.batch_size, len(rows))
            elapsed_batch = time.time() - t_batch
            total_elapsed = time.time() - start_all
            avg_per_claim = total_elapsed / max(done, 1)
            eta = avg_per_claim * (len(rows) - done)
            flags = sum(1 for x in rows_out if x["disagreement_flag"])
            log(
                f"Processed {done}/{len(rows)} claims | "
                f"batch={elapsed_batch:.1f}s avg/claim={avg_per_claim:.1f}s ETA~{eta/60:.1f}min | "
                f"flags={flags} | judge_repairs={len(invalid_indices)}"
            )

    out_df = pd.DataFrame(rows_out)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    out_df.to_csv(output_path, index=False)

    total_time = time.time() - start_all
    flagged = int(out_df["disagreement_flag"].sum())
    log(f"Done. Wrote: {output_path}")
    log(f"Flagged for review: {flagged}/{len(out_df)}")
    log(f"Total runtime: {total_time/60:.1f} minutes")


if __name__ == "__main__":
    main()