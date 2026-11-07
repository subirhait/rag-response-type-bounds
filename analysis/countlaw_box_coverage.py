import numpy as np, math, itertools
from scipy.stats import beta
K=5; lo_,hi_=.0075,.9925
types=[((lo_,lo_,lo_),.30),((hi_,hi_,hi_),.25),((lo_,hi_,lo_),.17),((hi_,lo_,hi_),.04),((lo_,lo_,hi_),.12),((lo_,hi_,hi_),.12)]
cells=list(itertools.product(range(K+1),repeat=3))
b=lambda c,p: math.comb(K,c)*p**c*(1-p)**(K-c)
pi=np.zeros(len(cells))
for (p,w) in types: pi+=w*np.array([b(c[0],p[0])*b(c[1],p[1])*b(c[2],p[2]) for c in cells])
print("cells with pi<1e-3:",(pi<1e-3).sum()," 1e-3..1e-2:",((pi>=1e-3)&(pi<1e-2)).sum())
def cp(x,n,a):
    M=len(x); ac=a/M
    lo=np.where(x==0,0,beta.ppf(ac/2,np.maximum(x,1),n-x+1)); up=np.where(x==n,1,beta.ppf(1-ac/2,x+1,np.maximum(n-x,1)));return lo,up
def maxt(x,n,a,B=5000,seed=1,s=.5):
    M=len(x); pe=x/n; pv=(x+s)/(n+s*M); sd=np.sqrt(pv*(1-pv)); r=np.random.default_rng(seed); T=[]
    for _ in range(B//1000):
        G=r.standard_normal((1000,M)); Y=G*np.sqrt(pv)-(G@np.sqrt(pv))[:,None]*pv[None,:]; T.append(np.max(np.abs(Y)/sd,1))
    c=np.quantile(np.concatenate(T),1-a); h=c*sd/np.sqrt(n); return pe-h,pe+h
rng=np.random.default_rng(7); R=400
for n in [500,1000,2000]:
    covcp=covmt=0; viol=[]
    for r in range(R):
        x=rng.multinomial(n,pi)
        l,u=cp(x,n,.05); covcp+= np.all((pi>=l)&(pi<=u))
        l,u=maxt(x,n,.05,seed=r); ok=(pi>=l)&(pi<=u); covmt+=ok.all()
        if not ok.all(): viol.append(int((~ok).sum()))
    print(f"n={n}: simultaneous box coverage CP={covcp/R:.3f}  max-t={covmt/R:.3f}  mean #violating cells when max-t fails={np.mean(viol) if viol else 0:.2f}")
