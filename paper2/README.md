# Paper 2: Adaptive Replication for Certified Diagnosis of Stochastic LLM Pipelines

**Working title:** *How Many Generations Are Enough? Adaptive Replication for Certified Diagnosis of Stochastic LLM Pipelines*

This folder contains the adaptive-replication follow-up to the companion paper in the repository root. Paper 1 studies what finite stochastic replication identifies. Paper 2 asks how to allocate a limited generation budget across queries while preserving that identifying information.

## Main method: disagreement-triggered replication (DTR)

For maximum replication `K >= 3`, DTR generates twice in each query-regime cell, stops if the two outcomes agree, and continues to `K` only after disagreement. The `K+1` stopped-history likelihoods form a basis for the same univariate degree-`K` polynomial space as ordinary fixed-`K` binomial counts. Tensor products across regimes therefore identify the same coordinate-degree space `V_K` as uniform `K` replication.

For a Bernoulli success probability `p`, expected cost per query-regime cell is

`2 + 2*(K-2)*p*(1-p)`

and is at most `(K+2)/2`. Thus expected fractional saving relative to uniform `K` is at least `(K-2)/(2K)`; for `K=5`, at least 30% for every latent distribution and close to 60% when `p` is near 0 or 1.

## Real HotpotQA retrospective experiment

Paper 1 stores all 20,000 generations with `query_id`, regime, replicate number, random seed, and exact-match outcome in `../data/generations.csv`. Applying DTR to the recorded order in regimes FF, IF, and FI gives:

- uniform `K=5` cost: **15,000** generations;
- DTR `2->5` cost: **6,189** generations;
- saved: **8,811 generations (58.74%)**;
- stopped after two draws: **2,937 / 3,000 query-regime cells (97.9%)**;
- query total costs: **942** queries used 6 generations, **53** used 9, **5** used 12, and **0** required all 15.

The exact confidence-interval comparison at `delta=.10` is archived in `results/real_hotpot_dtr_results.csv`. At almost equal cost, uniform `K=2` can be narrower under conservative Bonferroni--Clopper--Pearson projection because it has only 27 observable cells, whereas DTR preserves the 216-coordinate `K=5` identified space. This is a central result: preserving population identification is not the same as preserving finite-sample precision.

Reproduce the cost calculation with:

```bash
python paper2/code/real_hotpot_dtr.py
```

## Simulation evidence

The corner-concentrated simulation uses the DGP from Paper 1. The adaptive `2->5` population certificate matches the fixed-`K=5` population certificate to numerical precision, while using far fewer generations. At equal budget, adaptive `2->5` improves over fixed `K=5`; at low budgets, coarse fixed `K=2` can still be more precise because it covers more queries. As budget grows, DTR eventually benefits from its finer `K=5` population identification limit.

Simulation result tables are under `results/`. `budget_curve.png` is a planning curve based on population pseudo-counts; it is not a Monte Carlo average.

## Files

- `paper2.tex` / `paper2.pdf` — current full working paper.
- `references.bib`, `paper2.bbl` — bibliography files.
- `budget_curve.png` — budget-planning figure.
- `code/adaptive_design.py` — DTR transformation, cost functions, and adaptive certificate system.
- `code/run_equal_budget_simulation.py` — equal-budget Monte Carlo driver.
- `code/real_hotpot_dtr.py` — reconstructs the real HotpotQA DTR compute result from Paper 1 outputs.
- `code/ragpid/` — certificate routines used by the simulations.
- `results/` — simulation tables and real HotpotQA adaptive results.
- `sdss2027/` — blinded two-page SDSS 2027 extended abstract, source, overview, and submission notes.

## Status

The current paper contains a complete identification theorem, cost theorem, exact stopped-history inference construction, simulation study, and a retrospective real-output HotpotQA adaptive experiment. A future journal-scale extension may add a high-replication (`K=50-100`) reference panel, sharper likelihood/KL confidence regions, and additional model/task families. Those extensions are not required for the present SDSS/arXiv version.

## Relationship to Paper 1

Paper 2 uses Paper 1's publicly released HotpotQA outputs but asks a distinct design question. Paper 1: *What is identified under finite stochastic replication?* Paper 2: *How should stochastic evaluations be allocated under a generation budget?*