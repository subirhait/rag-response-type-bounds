"""Full-coordinate HotpotQA rerun.
Usage:  python run_hotpotqa.py generations.csv [out.csv]
Input (long format, one row per generation): columns query_id, regime, rep, em
  regime in {FF, IF, FI, II}; em in {0,1}. II rows are ignored (target uses FF, IF, FI).
Also accepts a wide count file with columns query_id, FF, IF, FI (counts out of K)."""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fullcoord import System, regions, cp_box, counts_from_queries

src = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else "hotpotqa_full_coordinate.csv"
d = pd.read_csv(src)
if {"regime", "em"}.issubset(d.columns):
    d = d[d.regime.isin(["FF", "IF", "FI"])]
    reps = d.groupby(["query_id", "regime"]).size()
    assert reps.nunique() == 1, "unequal replications per query-regime"
    K = int(reps.iloc[0])
    C = d.pivot_table(index="query_id", columns="regime", values="em", aggfunc="sum")[["FF", "IF", "FI"]]
else:
    C = d.set_index("query_id")[["FF", "IF", "FI"]]; K = int(C.values.max())
assert not C.isna().any().any(), "missing query-regime cells"
n = len(C); x = counts_from_queries(C.values, K)
print(f"n={n} queries, K={K}, nonempty cells {np.sum(x>0)}/{len(x)}")
ext = np.isin(C.values, [0, K]); print(f"extreme cells {ext.mean():.4f}; all-3 extreme {ext.all(1).mean():.4f}")
S = System(K); box = cp_box(x); rows = []
for delta in [.10, .20, .30, .40, .50]:
    for name, g in regions(delta).items():
        pt = S.bracket(g, pi=x / n); ci = S.bracket(g, pi=x / n, box=box)
        naive = None
        if name == "retrieval_binding":
            naive = np.mean(((C.IF - C.FF) / K > delta) & ((C.FI - C.FF) / K <= delta))
        elif name == "generation_binding":
            naive = np.mean(((C.FI - C.FF) / K > delta) & ((C.IF - C.FF) / K <= delta))
        else:
            naive = np.mean((C.FF - C.IF) / K >= delta)
        rows.append(dict(delta=delta, region=name, naive=naive,
                         point_lower=pt["lower"], point_upper=pt["upper"], point_coherent=bool(pt["coherent"]),
                         cp95_lower=ci["lower"], cp95_upper=ci["upper"]))
        print(rows[-1])
pd.DataFrame(rows).to_csv(out, index=False); print("saved", out)
