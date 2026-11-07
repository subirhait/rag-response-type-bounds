# Certified bounds on component-level response types: HotpotQA replication files

Replication materials for:

Hait, S. (2026). *Certified Bounds on Component-Level Response Types in Stochastic AI Systems, with an Application to Retrieval-Augmented Generation.* Manuscript submitted to *RSS: Data Science and Artificial Intelligence*.

## Design

1,000 questions from the HotpotQA distractor validation split, sampled with seed 20261008. Four regimes, five generations each (20,000 generations):

| Regime | Retrieval | Generator |
|---|---|---|
| FF | BM25 top 2 of the 10 candidate paragraphs | Qwen3-1.7B |
| IF | the two gold (annotated supporting) paragraphs | Qwen3-1.7B |
| FI | BM25 top 2 | Qwen3-4B |
| II | the two gold paragraphs | Qwen3-4B |

Decoding: non-thinking mode, sampling with temperature 0.7, top-p 0.9, at most 64 new tokens, bfloat16. Each generation has its own recorded seed.

## Contents

```
data/
  sampled_question_ids.csv    query_id, question_type, level
  generations.csv             one row per generation (20,000): regime, retrieval_mode, generator_model,
                              replicate, seed, retrieved_titles, raw_output, prediction, parse_method,
                              generated_tokens, em, f1
  counts.csv                  one row per question: C_FF, C_FI, C_IF, C_II (correct runs out of K = 5)
  retrieval_diagnostics.csv   gold titles, BM25 titles and scores, gold recall at 2
generation/
  run_generation.py           retrieval cells, prompt, decoding, exact-match and F1 scoring
analysis/
  fullcoord.py                full-coordinate Bernstein-Handelman certificates (216 coordinates at K = 5)
  run_hotpotqa.py             response-type bounds (Table 2, Figure 5)
  countlaw_box_coverage.py    coverage of the count distribution: Clopper-Pearson vs max-t
  fig4.py                     Figure 5
simulation/
  notebook and saved outputs for the synthetic studies (Tables 3-4, Figures 2-4)
results/
  saved summaries from the generation run (cell means, contrasts, strata, diagnostics)
```

Question text and gold answers are not redistributed; they can be recovered from HotpotQA with `query_id`.

## Reproducing the response-type bounds

```
pip install -r requirements.txt
python analysis/run_hotpotqa.py data/generations.csv results/hotpotqa_full_coordinate.csv
```

The script reports 58 nonempty count cells out of 216 and writes the naive shares, plug-in brackets, and simultaneous 95% Clopper-Pearson outer intervals for the retrieval-binding, generation-binding, and retrieval-harm shares at thresholds .10 to .50.

## Regenerating the outputs

```
python generation/run_generation.py --sample-size 1000 --k 5
```

All other options were left at their defaults. The run used an Intel GPU through PyTorch's XPU backend (`--device xpu`). Outputs can differ across hardware and library versions; all analyses use the saved outputs in `data/generations.csv`.

## Licences

Code: MIT (see `LICENSE`). HotpotQA (Yang et al., 2018) is distributed under CC BY-SA 4.0. Qwen3 models are released by the Qwen team under their own licence.

## Contact

Subir Hait, Michigan State University, haitsubi@msu.edu
