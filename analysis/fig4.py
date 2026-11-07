import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, sys
d=np.array([.1,.2,.3,.4,.5])
R={"Retrieval-binding":([.143,.136,.136,.132,.132],[.0637,.0724,.0745,.0751,.0732],[.4695,.4349,.4185,.4087,.4061]),
   "Generation-binding":([.134,.129,.129,.128,.128],[.0528,.0661,.0699,.0711,.0712],[.4782,.4357,.4164,.4050,.4015]),
   "Retrieval-harm":([.066,.066,.052,.052,.044],[.0125,.0118,.0104,.0087,.0066],[.4280,.3775,.3535,.3386,.3317])}
fig,ax=plt.subplots(1,3,figsize=(10,3.4),sharey=True)
for a,(t,(nv,lo,up)) in zip(ax,R.items()):
    a.vlines(d,lo,up,color="black",lw=1.6)
    a.plot(d,lo,"_",color="black",ms=10); a.plot(d,up,"_",color="black",ms=10)
    a.plot(d,nv,"o",mfc="white",mec="black",ms=5,label="Naive K=5 classifier")
    a.axhline(0,color="gray",lw=.6,ls=":")
    a.set_title(t,fontsize=11); a.set_xlabel(r"Threshold $\delta$"); a.set_xticks(d)
    for s in ["top","right"]: a.spines[s].set_visible(False)
ax[0].set_ylabel("Prevalence"); ax[0].set_ylim(-.02,.52); ax[0].legend(frameon=False,fontsize=8,loc="upper right")
plt.tight_layout(); plt.savefig(sys.argv[1],dpi=300)
