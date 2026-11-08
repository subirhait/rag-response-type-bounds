#!/usr/bin/env python
"""Retrospective disagreement-triggered replication (DTR) on Paper 1 HotpotQA outputs.

Run from repository root:
    python paper2/code/real_hotpot_dtr.py
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INFILE = ROOT / "data" / "generations.csv"
OUTFILE = ROOT / "paper2" / "results" / "real_hotpot_dtr_recomputed.csv"
REGIMES = ["FF", "IF", "FI"]

def main():
    df = pd.read_csv(INFILE)
    x = df[df["regime"].isin(REGIMES)].copy()
    x = x.sort_values(["query_id", "regime", "replicate"])

    rows = []
    for (qid, regime), g in x.groupby(["query_id", "regime"], sort=False):
        if len(g) != 5:
            raise ValueError(f"Expected five replicates for {qid}/{regime}; got {len(g)}")
        y = g["em"].astype(int).tolist()
        agree = y[0] == y[1]
        cost = 2 if agree else 5
        history = (0 if y[0] == 0 else 5) if agree else sum(y)
        rows.append({
            "query_id": qid,
            "regime": regime,
            "stop_after_2": bool(agree),
            "dtr_cost": cost,
            "history": history,
        })

    h = pd.DataFrame(rows)
    total_cost = int(h["dtr_cost"].sum())
    uniform_cost = int(len(h) * 5)
    saving = uniform_cost - total_cost
    by_query = h.groupby("query_id")["dtr_cost"].sum()

    summary = pd.DataFrame([
        {"quantity": "query_regime_cells", "value": len(h)},
        {"quantity": "queries", "value": h["query_id"].nunique()},
        {"quantity": "uniform_K5_generations", "value": uniform_cost},
        {"quantity": "DTR_generations", "value": total_cost},
        {"quantity": "generations_saved", "value": saving},
        {"quantity": "saving_fraction", "value": saving / uniform_cost},
        {"quantity": "stop_fraction", "value": h["stop_after_2"].mean()},
        {"quantity": "queries_cost_6", "value": int((by_query == 6).sum())},
        {"quantity": "queries_cost_9", "value": int((by_query == 9).sum())},
        {"quantity": "queries_cost_12", "value": int((by_query == 12).sum())},
        {"quantity": "queries_cost_15", "value": int((by_query == 15).sum())},
    ])
    OUTFILE.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTFILE, index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved {OUTFILE}")

if __name__ == "__main__":
    main()
