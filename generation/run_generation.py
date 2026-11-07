#!/usr/bin/env python
"""
HotpotQA empirical RAG experiment — XPU v5

Clean component-replacement 2 x 2 design.

Retrieval component:
  F = BM25 top-2 from the ten HotpotQA distractor paragraphs
  I = oracle two gold supporting paragraphs

Generator component:
  F = Qwen/Qwen3-1.7B
  I = Qwen/Qwen3-4B

Both generator models use the SAME direct-answer prompt, decoding settings,
answer parser, and non-thinking mode. Thus the generator intervention is a
capacity/model-component replacement rather than a prompt or token-budget change.

Regimes:
  FF = BM25 + Qwen3-1.7B
  IF = oracle retrieval + Qwen3-1.7B
  FI = BM25 + Qwen3-4B
  II = oracle retrieval + Qwen3-4B

Outputs:
  hotpot_empirical_v5/hotpot_empirical_long.csv
  hotpot_empirical_v5/hotpot_empirical_counts.csv
  hotpot_empirical_v5/hotpot_retrieval_diagnostics.csv
"""

import argparse, csv, json, re, string, gc, time
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForCausalLM


def normalize_answer(s):
    s = str(s).lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred, gold):
    return int(normalize_answer(pred) == normalize_answer(gold))


def token_f1(pred, gold):
    p = normalize_answer(pred).split()
    g = normalize_answer(gold).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    common = Counter(p) & Counter(g)
    n = sum(common.values())
    if n == 0:
        return 0.0
    precision = n / len(p)
    recall = n / len(g)
    return 2 * precision * recall / (precision + recall)


def bm25_tokens(text):
    return re.findall(r"[A-Za-z0-9]+", str(text).lower())


def paragraph_text(title, sentences):
    return f"{title}. " + " ".join(sentences)


def bm25_top2(example):
    titles = list(example["context"]["title"])
    sents = list(example["context"]["sentences"])
    docs = [paragraph_text(t, ss) for t, ss in zip(titles, sents)]
    bm25 = BM25Okapi([bm25_tokens(d) for d in docs])
    scores = bm25.get_scores(bm25_tokens(example["question"]))
    idx = np.argsort(-np.asarray(scores))[:2].tolist()
    return [titles[i] for i in idx], [docs[i] for i in idx], [float(scores[i]) for i in idx]


def oracle_top2(example):
    cmap = {
        str(t): paragraph_text(t, ss)
        for t, ss in zip(example["context"]["title"], example["context"]["sentences"])
    }
    gold_titles = []
    for t in example["supporting_facts"]["title"]:
        t = str(t)
        if t not in gold_titles and t in cmap:
            gold_titles.append(t)

    if len(gold_titles) < 2:
        b_titles, _, _ = bm25_top2(example)
        for t in b_titles:
            if t not in gold_titles:
                gold_titles.append(t)
            if len(gold_titles) == 2:
                break

    gold_titles = gold_titles[:2]
    return gold_titles, [cmap[t] for t in gold_titles]


def build_messages(question, contexts):
    block = "\n\n".join(
        f"PASSAGE {i+1}:\n{x}"
        for i, x in enumerate(contexts)
    )

    return [
        {
            "role": "system",
            "content": (
                "You are a careful question-answering system. "
                "Use only the supplied evidence. "
                "Follow the requested answer format exactly."
            ),
        },
        {
            "role": "user",
            "content": (
                "Answer the question using only the supplied passages. "
                "Do not explain your answer. "
                "Return exactly one line:\n"
                "FINAL: <answer>\n\n"
                f"{block}\n\nQUESTION:\n{question}"
            ),
        },
    ]


def render_prompt(tokenizer, messages):
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def parse_final_answer(raw):
    visible = str(raw)

    visible = re.sub(
        r"<\|(?:im_start|im_end|endoftext)\|>",
        " ",
        visible,
        flags=re.I,
    )

    visible = re.sub(
        r"<think>.*?</think>",
        " ",
        visible,
        flags=re.S | re.I,
    ).strip()

    matches = re.findall(
        r"(?:^|\n)\s*FINAL\s*:\s*(.+?)\s*(?=\n|$)",
        visible,
        flags=re.I,
    )
    if matches:
        return matches[-1].strip().strip('"').strip("'"), "FINAL_marker"

    matches = re.findall(
        r"(?:^|\n)\s*(?:final answer|answer)\s*:\s*(.+?)\s*(?=\n|$)",
        visible,
        flags=re.I,
    )
    if matches:
        return matches[-1].strip().strip('"').strip("'"), "answer_marker"

    lines = [x.strip() for x in visible.splitlines() if x.strip()]
    if not lines:
        return "", "empty"

    return lines[-1].strip().strip('"').strip("'"), "last_line"


@torch.inference_mode()
def generate_one(
    model,
    tokenizer,
    messages,
    seed,
    temperature,
    top_p,
    max_new_tokens,
):
    torch.manual_seed(seed)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.manual_seed_all(seed)

    prompt = render_prompt(tokenizer, messages)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    out = model.generate(
        **inputs,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
    )

    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=False)
    pred, parse_method = parse_final_answer(raw)

    return raw, pred, parse_method, int(new_tokens.numel())


def load_model(model_name, device, dtype):
    print("\nLoading:", model_name)
    tok = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    model.to(device)
    model.eval()

    print("Model device:", next(model.parameters()).device)

    if device.type == "xpu":
        try:
            free_b, total_b = torch.xpu.mem_get_info()
            print(
                f"XPU memory free/total after load: "
                f"{free_b/2**30:.2f}/{total_b/2**30:.2f} GiB"
            )
        except Exception:
            pass

    return tok, model


def unload_model(tokenizer, model):
    del model
    del tokenizer
    gc.collect()

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        try:
            torch.xpu.empty_cache()
            torch.xpu.synchronize()
        except Exception:
            pass

    time.sleep(1)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--baseline-model",
        default="Qwen/Qwen3-1.7B",
    )
    ap.add_argument(
        "--improved-model",
        default="Qwen/Qwen3-4B",
    )
    ap.add_argument("--device", default="xpu")
    ap.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    ap.add_argument("--sample-size", type=int, default=20)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20261008)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument(
        "--output-dir",
        default="hotpot_empirical_v5",
    )

    args = ap.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.device == "xpu":
        if not (
            hasattr(torch, "xpu")
            and torch.xpu.is_available()
        ):
            raise RuntimeError(
                "Requested XPU but torch.xpu.is_available() is False."
            )
        device = torch.device("xpu")
    else:
        device = torch.device(args.device)

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    print("Loading HotpotQA...")
    ds = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
        split="validation",
    )

    rng = np.random.default_rng(args.seed)
    idx = np.sort(
        rng.choice(
            len(ds),
            size=args.sample_size,
            replace=False,
        )
    )
    sample = ds.select(idx.tolist())

    # Precompute retrieval once so both generator models see exactly
    # the same query-specific contexts.
    plans = []

    for qi, ex in enumerate(sample):
        qid = str(ex["id"])
        qtype = str(ex["type"])
        level = str(ex["level"])
        question = str(ex["question"])
        gold = str(ex["answer"])

        b_titles, b_contexts, b_scores = bm25_top2(ex)
        o_titles, o_contexts = oracle_top2(ex)

        gold_titles = []
        for t in ex["supporting_facts"]["title"]:
            t = str(t)
            if t not in gold_titles:
                gold_titles.append(t)

        hit = sum(
            t in set(b_titles)
            for t in gold_titles
        )

        plans.append({
            "qi": qi,
            "query_id": qid,
            "question_type": qtype,
            "level": level,
            "question": question,
            "gold_answer": gold,
            "bm25_titles": b_titles,
            "bm25_contexts": b_contexts,
            "bm25_scores": b_scores,
            "oracle_titles": o_titles,
            "oracle_contexts": o_contexts,
            "gold_titles": gold_titles,
            "gold_recall_at2":
                hit / max(1, len(gold_titles)),
            "both_gold_at2":
                int(hit == len(gold_titles)),
        })

    ret_path = (
        outdir
        / "hotpot_retrieval_diagnostics.csv"
    )

    with open(
        ret_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        fields = [
            "query_id",
            "question_type",
            "level",
            "gold_titles",
            "bm25_titles",
            "bm25_scores",
            "gold_recall_at2",
            "both_gold_at2",
        ]

        w = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        w.writeheader()

        for p in plans:
            w.writerow({
                "query_id": p["query_id"],
                "question_type":
                    p["question_type"],
                "level": p["level"],
                "gold_titles":
                    json.dumps(
                        p["gold_titles"],
                        ensure_ascii=False,
                    ),
                "bm25_titles":
                    json.dumps(
                        p["bm25_titles"],
                        ensure_ascii=False,
                    ),
                "bm25_scores":
                    json.dumps(
                        p["bm25_scores"]
                    ),
                "gold_recall_at2":
                    p["gold_recall_at2"],
                "both_gold_at2":
                    p["both_gold_at2"],
            })

    long_path = (
        outdir
        / "hotpot_empirical_long.csv"
    )

    long_fields = [
        "query_id",
        "question_type",
        "level",
        "question",
        "gold_answer",
        "regime",
        "retrieval_mode",
        "generator_model",
        "generator_level",
        "replicate",
        "seed",
        "retrieved_titles",
        "raw_output",
        "prediction",
        "parse_method",
        "generated_tokens",
        "em",
        "f1",
    ]

    with open(
        long_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        w = csv.DictWriter(
            f,
            fieldnames=long_fields,
        )
        w.writeheader()

    # Generator F:
    # FF = baseline retrieval
    # IF = improved retrieval
    model_specs = [
        (
            "F",
            args.baseline_model,
            [
                ("FF", "baseline"),
                ("IF", "oracle"),
            ],
        ),
        (
            "I",
            args.improved_model,
            [
                ("FI", "baseline"),
                ("II", "oracle"),
            ],
        ),
    ]

    for (
        generator_level,
        model_name,
        regimes,
    ) in model_specs:

        tokenizer, model = load_model(
            model_name,
            device,
            dtype,
        )

        with open(
            long_path,
            "a",
            newline="",
            encoding="utf-8",
        ) as f:

            w = csv.DictWriter(
                f,
                fieldnames=long_fields,
            )

            for p in plans:

                print(
                    f"[{p['qi']+1}/{len(plans)}] "
                    f"{p['query_id']} "
                    f"{p['question_type']} "
                    f"generator={generator_level}"
                )

                for (
                    regime,
                    retrieval_mode,
                ) in regimes:

                    if retrieval_mode == "baseline":
                        titles = (
                            p["bm25_titles"]
                        )
                        contexts = (
                            p["bm25_contexts"]
                        )
                    else:
                        titles = (
                            p["oracle_titles"]
                        )
                        contexts = (
                            p["oracle_contexts"]
                        )

                    messages = build_messages(
                        p["question"],
                        contexts,
                    )

                    regime_index = [
                        "FF",
                        "IF",
                        "FI",
                        "II",
                    ].index(regime)

                    for rep in range(
                        1,
                        args.k + 1,
                    ):

                        seed = (
                            args.seed
                            + p["qi"] * 10000
                            + regime_index * 100
                            + rep
                        )

                        (
                            raw,
                            pred,
                            parse_method,
                            generated_tokens,
                        ) = generate_one(
                            model=model,
                            tokenizer=tokenizer,
                            messages=messages,
                            seed=seed,
                            temperature=
                                args.temperature,
                            top_p=args.top_p,
                            max_new_tokens=
                                args.max_new_tokens,
                        )

                        w.writerow({
                            "query_id":
                                p["query_id"],
                            "question_type":
                                p["question_type"],
                            "level":
                                p["level"],
                            "question":
                                p["question"],
                            "gold_answer":
                                p["gold_answer"],
                            "regime":
                                regime,
                            "retrieval_mode":
                                retrieval_mode,
                            "generator_model":
                                model_name,
                            "generator_level":
                                generator_level,
                            "replicate":
                                rep,
                            "seed":
                                seed,
                            "retrieved_titles":
                                json.dumps(
                                    titles,
                                    ensure_ascii=False,
                                ),
                            "raw_output":
                                raw,
                            "prediction":
                                pred,
                            "parse_method":
                                parse_method,
                            "generated_tokens":
                                generated_tokens,
                            "em":
                                exact_match(
                                    pred,
                                    p["gold_answer"],
                                ),
                            "f1":
                                token_f1(
                                    pred,
                                    p["gold_answer"],
                                ),
                        })

                        f.flush()

        unload_model(
            tokenizer,
            model,
        )

    d = pd.read_csv(long_path)

    meta = (
        d[
            [
                "query_id",
                "question_type",
                "level",
                "question",
                "gold_answer",
            ]
        ]
        .drop_duplicates("query_id")
        .set_index("query_id")
    )

    counts = (
        d.groupby(
            ["query_id", "regime"]
        )["em"]
        .sum()
        .unstack("regime")
        .rename(
            columns=lambda x: f"C_{x}"
        )
    )

    counts["K"] = args.k

    counts_path = (
        outdir
        / "hotpot_empirical_counts.csv"
    )

    meta.join(
        counts
    ).reset_index().to_csv(
        counts_path,
        index=False,
    )

    print("\nDone:", counts_path)

    print("\nParse-method counts:")
    print(
        d.groupby(
            ["regime", "parse_method"]
        ).size()
    )

    rates = (
        d.groupby("regime")
        .agg(
            EM=("em", "mean"),
            F1=("f1", "mean"),
        )
    )

    print("\nCell performance:")
    print(rates)

    ff = rates.loc["FF", "EM"]
    iff = rates.loc["IF", "EM"]
    fi = rates.loc["FI", "EM"]
    ii = rates.loc["II", "EM"]

    print("\nComponent contrasts (EM):")
    print(
        "Retrieval gain under baseline generator:",
        iff - ff,
    )
    print(
        "Retrieval gain under improved generator:",
        ii - fi,
    )
    print(
        "Generator gain under baseline retrieval:",
        fi - ff,
    )
    print(
        "Generator gain under improved retrieval:",
        ii - iff,
    )
    print(
        "2x2 interaction:",
        ii - iff - fi + ff,
    )


if __name__ == "__main__":
    main()
