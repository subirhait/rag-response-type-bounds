# Certified bounds on component-level response types: HotpotQA replication files

This repository contains the data and code for:

> Hait, S. (2026). *Certified Bounds on Component-Level Response Types in Stochastic AI Systems, with an Application to Retrieval-Augmented Generation.* Manuscript submitted to *RSS: Data Science and Artificial Intelligence*.

The paper asks for what share of inputs a component of a stochastic AI system, such as the retriever or the generator in a retrieval-augmented generation (RAG) pipeline, is the binding constraint. With a finite number of repeated runs, this share is only partially identified. The code computes certified bounds on it and simultaneous confidence intervals.

## Experimental design

We drew 1,000 questions from the HotpotQA distractor validation split by simple random sampling (seed 20261008). Each question was run five times in each of four regimes, for 20,000 generations in total.

| Regime | Retrieval | Generator |
|---|---|---|
| FF | BM25, top 2 of the 10 candidate paragraphs | Qwen3-1.7B |
| IF | the two gold (annotated supporting) paragraphs | Qwen3-1.7B |
| FI | BM25, top 2 of the 10 candidate paragraphs | Qwen3-4B |
| II | the two gold (annotated supporting) paragraphs | Qwen3-4B |

- **Paragraph order.** BM25 paragraphs appear in score order. Gold paragraphs appear in the order in which their titles first occur in the HotpotQA supporting facts.
- **Decoding.** Non-thinking mode; sampling with temperature 0.7 and top-p 0.9; at most 64 new tokens; bfloat16 precision. Each generation has its own recorded seed.
- **Outcome.** Normalised exact match (`em`). Token-level F1 (`f1`) is also recorded.

## Repository contents

```
data/
  sampled_question_ids.csv    query_id, question_type, level
  generations.csv             one row per generation (20,000 rows): query_id, regime, retrieval_mode,
                              generator_model, replicate, seed, retrieved_titles, raw_output,
                              prediction, parse_method, generated_tokens, em, f1
  counts.csv                  one row per question: correct runs out of K = 5 in each regime
                              (C_FF, C_IF, C_FI, C_II)
  retrieval_diagnostics.csv   gold titles, BM25 titles and scores, gold recall at 2
generation/
  run_generation.py           retrieval, prompt, decoding, and exact-match and F1 scoring
analysis/
  fullcoord.py                full-coordinate Bernstein-Handelman certificates (216 coordinates at K = 5)
  run_hotpotqa.py             certified response-type bounds (Table 4)
  fig4.py                     HotpotQA response-type figure (Figure 5)
  countlaw_box_coverage.py    coverage of the count distribution: Clopper-Pearson versus max-t regions
simulation/
  notebook and saved outputs for the synthetic studies (Tables 1-2, Figures 2-4)
results/
  saved summaries from the generation run: cell means and contrasts (Table 3), retrieval strata,
  question-type summaries, and monotonicity diagnostics
```

HotpotQA question text and gold answers are not redistributed here. They can be recovered from HotpotQA with `query_id`.

## Reproducing the response-type bounds

```
pip install -r requirements.txt
python analysis/run_hotpotqa.py data/generations.csv results/hotpotqa_full_coordinate.csv
```

The script uses the FF, IF, and FI regimes. It reports that 58 of the 216 count cells are nonempty. It then writes, for each threshold from .10 to .50, the naive whole-count shares, the plug-in brackets, and the simultaneous 95% Clopper-Pearson outer intervals for the retrieval-binding, generation-binding, and retrieval-harm shares.

## Regenerating the model outputs

```
python generation/run_generation.py --sample-size 1000 --k 5
```

All other options were left at their defaults. The original run used an Intel GPU through the PyTorch XPU backend (`--device xpu`). Regenerated outputs can differ across hardware and library versions, so all analyses in the paper use the saved outputs in `data/generations.csv`.

## Licences

The code is released under the MIT licence (see `LICENSE`). HotpotQA (Yang et al., 2018) is distributed under CC BY-SA 4.0. The Qwen3 models are released by the Qwen team under their own licence.

## Citation

If you use these materials, please cite the paper above and this archive.

## Contact

Subir Hait, Michigan State University, haitsubi@msu.edu
