"""Full-coordinate Bernstein-Handelman certificates for any polyhedral response region
in (x,y,z) = (p_FF, p_IF, p_FI). Generalises the notebook's build_certificate_system:
target region = {g_i >= 0 for all i}; complement covered by pieces {-g_i >= 0}."""
import itertools, math
from collections import defaultdict
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csc_matrix, eye, hstack, vstack
from scipy.stats import beta

def solve_lp(c, **kw):
    for method, pre in [("highs-ds", False), ("highs-ipm", False), ("highs", True)]:
        r = linprog(c, method=method, options={"presolve": pre}, **kw)
        if r.success:
            return r
    raise RuntimeError(r.message)

def affine(const, coeffs):
    out = {(0, 0, 0): float(const)} if const else {}
    for j, a in enumerate(coeffs):
        if a:
            e = [0, 0, 0]; e[j] = 1; out[tuple(e)] = float(a)
    return out

def neg(p):
    return {e: -c for e, c in p.items()}

class System:
    def __init__(self, K):
        self.K = K
        self.cells = list(itertools.product(range(K + 1), repeat=3))
        self.idx = {e: i for i, e in enumerate(self.cells)}
        self.N = len(self.cells)
        T = np.zeros((self.N, self.N))
        for i, c in enumerate(self.cells):
            for j, e in enumerate(self.cells):
                if all(e[r] <= c[r] for r in range(3)):
                    T[i, j] = np.prod([math.comb(c[r], e[r]) / math.comb(K, e[r]) for r in range(3)])
        self.T = T  # monomial coeffs -> tensor Bernstein coeffs

    def mul(self, p, q):
        out = defaultdict(float)
        for e1, c1 in p.items():
            for e2, c2 in q.items():
                e = tuple(a + b for a, b in zip(e1, e2))
                if max(e) <= self.K:
                    out[e] += c1 * c2
        return {e: c for e, c in out.items() if abs(c) > 1e-14}

    def bern(self, degs, ids):
        out = {(0, 0, 0): 1.0}
        for v, (d, i) in enumerate(zip(degs, ids)):
            b = {}
            for j in range(d - i + 1):
                e = [0, 0, 0]; e[v] = i + j
                b[tuple(e)] = math.comb(d, i) * math.comb(d - i, j) * (-1.0) ** j
            out = self.mul(out, b)
        return out

    def cone(self, gs):
        """Columns: prod g^a * residual tensor Bernstein; all nonneg on {g>=0} ∩ cube."""
        masks = [sorted({j for e in g for j in range(3) if e[j]}) for g in gs]
        rows, cols, data, col = [], [], [], 0
        def rec(i, used, prod):
            nonlocal col
            if i == len(gs):
                res = tuple(self.K - u for u in used)
                for bi in itertools.product(*[range(d + 1) for d in res]):
                    for e, c in self.mul(self.bern(res, bi), prod).items():
                        rows.append(self.idx[e]); cols.append(col); data.append(c)
                    col += 1
                return
            maxp = min([self.K - used[j] for j in masks[i]], default=self.K)
            gp = {(0, 0, 0): 1.0}
            for pw in range(maxp + 1):
                nu = list(used)
                for j in masks[i]:
                    nu[j] += pw
                rec(i + 1, nu, self.mul(prod, gp))
                gp = self.mul(gp, gs[i])
        rec(0, [0, 0, 0], {(0, 0, 0): 1.0})
        return csc_matrix((data, (rows, cols)), shape=(self.N, col))

    @staticmethod
    def support(beta_c, lo, up):
        """sup beta.pi over {lo<=pi<=up, sum pi = 1} (exact greedy)."""
        val = float(beta_c @ lo); rem = max(0.0, 1.0 - lo.sum())
        for i in np.argsort(-beta_c):
            if rem <= 0: break
            a = min(up[i] - lo[i], rem); val += beta_c[i] * a; rem -= a
        return val

    def majorant(self, pieces, pi=None, box=None):
        """min E q over q >= 0 on cube, q-1 in cone(piece) for every piece. Returns (value, safety)."""
        Hs = [self.cone(p) for p in pieces]
        ns = [H.shape[1] for H in Hs]; nl = sum(ns)
        B1 = csc_matrix(self.T @ Hs[0])
        # equality: H1 l1 = Hk lk for k>=2
        eq = []
        off = np.cumsum([0] + ns)
        for k in range(1, len(Hs)):
            blocks = [csc_matrix((self.N, n)) for n in ns]
            blocks[0] = Hs[0]; blocks[k] = -Hs[k]
            eq.append(hstack(blocks, format="csc"))
        pad = lambda M, extra: hstack([M, csc_matrix((M.shape[0], extra))], format="csc")
        Bfull = hstack([B1] + [csc_matrix((self.N, n)) for n in ns[1:]], format="csc")
        ones = np.ones(self.N)
        if box is None:
            c = np.r_[np.asarray(B1.T @ pi).ravel(), np.zeros(nl - ns[0])]
            kw = dict(A_ub=-Bfull, b_ub=ones, bounds=[(0, None)] * nl)
            if eq: kw.update(A_eq=vstack(eq, format="csc"), b_eq=np.zeros(self.N * len(eq)))
        else:
            lo, up = box
            c = np.r_[np.asarray(B1.T @ lo).ravel(), np.zeros(nl - ns[0]), 1 - lo.sum(), up - lo]
            r1 = pad(-Bfull, 1 + self.N)
            r2 = hstack([Bfull, -np.ones((self.N, 1)), -eye(self.N, format="csc")], format="csc")
            kw = dict(A_ub=vstack([r1, r2], format="csc"), b_ub=np.r_[ones, np.zeros(self.N)],
                      bounds=[(0, None)] * nl + [(None, None)] + [(0, None)] * self.N)
            if eq: kw.update(A_eq=pad(vstack(eq, format="csc"), 1 + self.N), b_eq=np.zeros(self.N * len(eq)))
        r = solve_lp(c, **kw)
        lam = np.maximum(r.x[:nl], 0)
        base = Hs[0] @ lam[:ns[0]]
        resid = sum(np.abs(base - Hs[k] @ lam[off[k]:off[k + 1]]).sum() for k in range(1, len(Hs)))
        th = base.copy(); th[self.idx[(0, 0, 0)]] += 1.0
        b = self.T @ th
        s = max(0.0, -b.min(), resid) + 5e-12
        b = b + s
        val = float(b @ pi) if box is None else self.support(b, *box)
        return val, s

    def bracket(self, gs, pi=None, box=None):
        """Outer bracket for P(P in {g>=0 all g}). pi: count-cell law; box: (lo, up) confidence box."""
        up, su = self.majorant([gs], pi, box)
        cu, sl = self.majorant([[neg(g)] for g in gs], pi, box)
        lo_v, up_v = max(0.0, 1 - cu), min(1.0, up)
        return dict(lower=lo_v, upper=up_v, coherent=lo_v <= up_v + 1e-7, safety=(sl, su))

def regions(delta):
    # x=p_FF, y=p_IF, z=p_FI
    return {
        "retrieval_binding": [affine(-delta, [-1, 1, 0]), affine(delta, [1, 0, -1])],  # y-x>=d, z-x<=d
        "generation_binding": [affine(-delta, [-1, 0, 1]), affine(delta, [1, -1, 0])],  # z-x>=d, y-x<=d
        "retrieval_harm": [affine(-delta, [1, -1, 0])],                                 # x-y>=d
    }

def cp_box(counts, alpha=0.05):
    x = np.asarray(counts); n = x.sum(); a = alpha / len(x)
    lo = np.where(x == 0, 0.0, beta.ppf(a / 2, np.maximum(x, 1), n - x + 1))
    up = np.where(x == n, 1.0, beta.ppf(1 - a / 2, x + 1, np.maximum(n - x, 1)))
    return lo, up

def counts_from_queries(C, K):
    """C: (n,3) integer array of counts (FF, IF, FI) per query -> cell count vector."""
    S = System.__new__(System); cells = list(itertools.product(range(K + 1), repeat=3))
    idx = {c: i for i, c in enumerate(cells)}; v = np.zeros(len(cells), int)
    for row in np.asarray(C, int):
        v[idx[tuple(row)]] += 1
    return v
