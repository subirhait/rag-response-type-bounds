"""Adaptive replication utilities for Paper 2.

Implements disagreement-triggered replication (DTR), the exact linear
transformation between fixed-K Bernstein count coordinates and DTR stopped
histories, and continuous-support certificate inference in adaptive-history
coordinates.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.optimize import linprog as scipy_linprog
from scipy.sparse import csc_matrix, eye, hstack, vstack
from ragpid import core

def solve_lp(*args, **kwargs):
    kwargs = dict(kwargs); kwargs.pop("method", None)
    base = dict(kwargs.pop("options", {}) or {})
    last = None
    for method, presolve in [("highs-ds",False),("highs-ipm",False),("highs",False),("highs-ds",True),("highs-ipm",True)]:
        options = dict(base); options["presolve"] = presolve
        result = scipy_linprog(*args, method=method, options=options, **kwargs)
        if result.success: return result
        last = result
    return last

def adaptive_transform(K:int)->np.ndarray:
    if K < 3: raise ValueError("DTR requires K >= 3")
    R=K-2; A=np.zeros((K+1,K+1),float)
    for k in range(K-1): A[0,k]=math.comb(R,k)/math.comb(K,k)
    for d in range(R+1):
        k=d+1; A[k,k]=2*math.comb(R,d)/math.comb(K,k)
    for k in range(2,K+1): A[K,k]=math.comb(R,k-2)/math.comb(K,k)
    return A

def tensor_transform(K:int,J:int=3)->np.ndarray:
    A=adaptive_transform(K); out=A
    for _ in range(J-1): out=np.kron(out,A)
    return out

def expected_cost_per_regime(p,K):
    p=np.asarray(p,float); return 2.0+2.0*(K-2)*p*(1-p)

def guaranteed_saving_fraction(K:int)->float:
    if K<3: raise ValueError("K must be >=3")
    return (K-2)/(2.0*K)

def support_box_simplex(beta,lower,upper):
    beta=np.asarray(beta,float); lo=np.asarray(lower,float); up=np.asarray(upper,float)
    cap=up-lo; remaining=1.0-lo.sum()
    remaining=min(max(remaining,0.0),cap.sum())
    value=float(beta@lo)
    for i in np.argsort(-beta):
        if remaining<=1e-15: break
        add=min(cap[i],remaining); value+=beta[i]*add; remaining-=add
    return value

class AdaptiveCertificateSystem:
    def __init__(self,K=5,delta=.10,target="binding",adaptive=True):
        self.K=K; self.N=(K+1)**3; self.delta=delta; self.target=target
        basis=tuple(core.tensor_exponents(K)); self.basis=basis
        T=core.monomial_to_bernstein_matrix(K,basis); self.T=T
        target_constraints,complement=core._target_spec(target,delta)
        HA=core.bernstein_handelman_region_matrix(K,target_constraints)
        Hc=[core.bernstein_handelman_region_matrix(K,p) for p in complement]
        BA=csc_matrix(T@HA); Bc=[csc_matrix(T@H) for H in Hc]
        if adaptive:
            A3=tensor_transform(K,3); M=np.linalg.inv(A3.T)
        else:
            A3=np.eye(self.N); M=np.eye(self.N)
        self.adaptive=adaptive; self.A3=A3; self.M=M
        self.HA=HA; self.Hc=Hc; self.BA=BA; self.Bc=Bc
        self.GA=csc_matrix(M@BA.toarray()); self.Gc=[csc_matrix(M@B.toarray()) for B in Bc]
        self.one=np.ones(self.N)

    def certificate(self,eta,cell_box=None):
        eta=np.asarray(eta,float); N=self.N; one=self.one
        HA,BA,GA=self.HA,self.BA,self.GA; Hc,Bc,Gc=self.Hc,self.Bc,self.Gc
        if cell_box is None:
            objective=np.asarray(GA.T@eta).ravel()
            upper_lp=solve_lp(objective,A_ub=-BA,b_ub=one,bounds=[(0,None)]*HA.shape[1])
        else:
            lo,up=map(np.asarray,cell_box); cap=up-lo; remaining=1.0-lo.sum()
            objective=np.r_[np.asarray(GA.T@lo).ravel(),remaining,cap]
            row1=hstack([-BA,csc_matrix((N,1+N))],format="csc")
            row2=hstack([GA,-np.ones((N,1)),-eye(N,format="csc")],format="csc")
            upper_lp=solve_lp(objective,A_ub=vstack([row1,row2],format="csc"),
                b_ub=np.r_[one,np.zeros(N)],
                bounds=[(0,None)]*HA.shape[1]+[(None,None)]+[(0,None)]*N)
        if not upper_lp.success: raise RuntimeError("upper certificate LP failed: "+str(upper_lp.message))
        lam=np.maximum(upper_lp.x[:HA.shape[1]],0.0)
        theta=np.zeros(len(self.basis)); theta[0]=1.0; theta+=HA@lam
        beta=self.T@theta; safety_upper=max(0.0,-float(beta.min()))+5e-12; beta+=safety_upper
        gamma=self.M@beta
        upper=float(gamma@eta) if cell_box is None else support_box_simplex(gamma,*cell_box)

        nvars=[H.shape[1] for H in Hc]; base_H,base_B,base_G=Hc[0],Bc[0],Gc[0]
        eq_rows=[]
        for j in range(1,len(Hc)):
            blocks=[]
            for r,H in enumerate(Hc):
                if r==0: blocks.append(base_H)
                elif r==j: blocks.append(-H)
                else: blocks.append(csc_matrix((base_H.shape[0],H.shape[1])))
            if cell_box is not None: blocks.append(csc_matrix((base_H.shape[0],1+N)))
            eq_rows.append(hstack(blocks,format="csc"))
        A_eq=vstack(eq_rows,format="csc") if eq_rows else None
        b_eq=np.zeros(A_eq.shape[0]) if A_eq is not None else None

        if cell_box is None:
            parts=[np.asarray(base_G.T@eta).ravel()]+[np.zeros(n) for n in nvars[1:]]
            objective=np.concatenate(parts)
            A_ub=hstack([-base_B]+[csc_matrix((N,n)) for n in nvars[1:]],format="csc")
            bounds=sum(([(0,None)]*n for n in nvars),[])
            lower_lp=solve_lp(objective,A_ub=A_ub,b_ub=one,A_eq=A_eq,b_eq=b_eq,bounds=bounds)
        else:
            lo,up=map(np.asarray,cell_box); cap=up-lo; remaining=1.0-lo.sum()
            parts=[np.asarray(base_G.T@lo).ravel()]+[np.zeros(n) for n in nvars[1:]]
            objective=np.r_[np.concatenate(parts),remaining,cap]
            row1=hstack([-base_B]+[csc_matrix((N,n)) for n in nvars[1:]]+[csc_matrix((N,1+N))],format="csc")
            row2=hstack([base_G]+[csc_matrix((N,n)) for n in nvars[1:]]+[-np.ones((N,1)),-eye(N,format="csc")],format="csc")
            bounds=sum(([(0,None)]*n for n in nvars),[])+[(None,None)]+[(0,None)]*N
            lower_lp=solve_lp(objective,A_ub=vstack([row1,row2],format="csc"),
                b_ub=np.r_[one,np.zeros(N)],A_eq=A_eq,b_eq=b_eq,bounds=bounds)
        if not lower_lp.success: raise RuntimeError("lower certificate LP failed: "+str(lower_lp.message))

        lambdas=[]; pos=0
        for nv in nvars:
            lambdas.append(np.maximum(lower_lp.x[pos:pos+nv],0.0)); pos+=nv
        base=Hc[0]@lambdas[0]
        theta=np.zeros(len(self.basis)); theta[0]=1.0; theta+=base
        beta=self.T@theta; safety_lower=max(0.0,-float(beta.min()))
        for H,lmb in zip(Hc[1:],lambdas[1:]):
            safety_lower=max(safety_lower,float(np.abs(base-H@lmb).sum()))
        safety_lower+=5e-12; beta+=safety_lower
        gamma=self.M@beta
        complement_upper=float(gamma@eta) if cell_box is None else support_box_simplex(gamma,*cell_box)
        lower=1.0-complement_upper
        return max(0.0,min(1.0,lower)),max(0.0,min(1.0,upper))

def sample_dtr_states(p,K,rng):
    p=np.asarray(p,float); first=rng.binomial(2,p)
    state=np.empty_like(first); cost=np.full_like(first,2)
    state[first==0]=0; state[first==2]=K
    mixed=first==1
    if np.any(mixed):
        extra=rng.binomial(K-2,p); state[mixed]=extra[mixed]+1; cost[mixed]=K
    return state,cost

def categorical_counts(states,K):
    cells=core.tensor_exponents(K); index={c:i for i,c in enumerate(cells)}
    counts=np.zeros(len(cells),dtype=int)
    for row in map(tuple,np.asarray(states,dtype=int).tolist()): counts[index[row]]+=1
    return counts
