# The network predicts one harmonic (or noise band) at a time from its frequency, so what it learns is the
# instrument's spectral envelope as a function of frequency, pitch and loudness, not a table per note.
import json, numpy as np, sys
rng=np.random.default_rng(1)
d=json.load(open('sax_frames.json'))
SR=48000; HOP=240; K=48; NB=16; FLOOR=-11.5
nb_c=np.sqrt(np.geomspace(150,16000,NB+1)[:-1]*np.geomspace(150,16000,NB+1)[1:])
def feat(fq, kind, midi, layer, L, t):
    # fq Hz; kind 0 harmonic, 1 noise. Sines of log-frequency at four scales let the net draw sharp formants.
    lf=np.log2(fq/1000)/3
    ff=[g(np.pi*2**m*lf) for m in range(4) for g in (np.sin,np.cos)]
    return np.stack([lf, kind*2-1+0*fq, (midi-66)/22+0*fq, (layer*2-1)+0*fq, np.clip(L,-60,0)/30+1, np.minimum(t,0.4)/0.4*2-1, *ff],-1)
rows=[]
for s in d:
    H=np.array(s['H']); N=np.array(s['N']); f0=np.array(s['f0'])
    norm=np.sqrt((H**2).sum(1))+1e-12; L=20*np.log10(norm); L-=L.max()
    t=np.arange(len(H))*HOP/SR
    for i in range(len(H)):
        if L[i]<=-50: continue
        k=np.arange(1,K+1); fq=k*f0[i]; ok=fq<16000
        xh=feat(fq[ok],0,s['midi'],s['layer'],L[i]+0*fq[ok],t[i]+0*fq[ok]); yh=np.clip(np.log(np.maximum(H[i][ok]/norm[i],1e-5)),FLOOR,0)
        xn=feat(nb_c,1,s['midi'],s['layer'],L[i]+0*nb_c,t[i]+0*nb_c); yn=np.clip(np.log(np.maximum(N[i]/norm[i],1e-5)),FLOOR,0)
        g=s['midi']*10+s['layer']
        rows.append((np.concatenate([xh,xn]),np.concatenate([yh,yn]),np.full(len(yh)+len(yn),g),np.concatenate([np.zeros(len(yh)),np.ones(len(yn))])))
X=np.concatenate([r[0] for r in rows]).astype(np.float32); Y=np.concatenate([r[1] for r in rows]).astype(np.float32)
G=np.concatenate([r[2] for r in rows]); KIND=np.concatenate([r[3] for r in rows])
print('samples',len(X))
hold=(G==601)|(G==740)
D=X.shape[1]; H1=int(sys.argv[2]) if len(sys.argv)>2 else 64
def init(a,b): return (rng.standard_normal((a,b))*np.sqrt(1/a)).astype(np.float32), np.zeros(b,np.float32)
def fwd(P,x):
    W1,b1,W2,b2,W3,b3=P; a1=np.tanh(x@W1+b1); a2=np.tanh(a1@W2+b2); return a1,a2,(a2@W3+b3)[:,0]
def run(mask, steps, lr=3e-3):
    Q=[*init(D,H1),*init(H1,H1),*init(H1,1)]; Q[5]+=Y[mask].mean()
    m=[np.zeros_like(p) for p in Q]; v=[np.zeros_like(p) for p in Q]
    x=X[mask];y=Y[mask]
    for it in range(1,steps+1):
        idx=rng.integers(0,len(x),4096); xb=x[idx];yb=y[idx]
        W1,b1,W2,b2,W3,b3=Q; a1,a2,o=fwd(Q,xb)
        e=o-yb; go=(2*e/len(e))[:,None]
        gW3=a2.T@go; gb3=go.sum(0); ga2=go@W3.T*(1-a2**2); gW2=a1.T@ga2; gb2=ga2.sum(0); ga1=ga2@W2.T*(1-a1**2); gW1=xb.T@ga1; gb1=ga1.sum(0)
        lr_t=lr*(0.5*(1+np.cos(np.pi*it/steps)))+1e-5
        WD=float(__import__("os").environ.get("WD","0"))
        gW1=gW1+WD*W1; gW2=gW2+WD*W2; gW3=gW3+WD*W3
        for i,g in enumerate([gW1,gb1,gW2,gb2,gW3,gb3]):
            m[i]=0.9*m[i]+0.1*g; v[i]=0.999*v[i]+0.001*g*g
            Q[i]-=lr_t*(m[i]/(1-0.9**it))/(np.sqrt(v[i]/(1-0.999**it))+1e-8)
    return Q
def err(Q,mask):
    _,_,o=fwd(Q,X[mask]); e=np.abs(o-Y[mask])*20/np.log(10); k=KIND[mask]
    return e[k==0].mean(), e[k==1].mean()
steps=int(sys.argv[1]) if len(sys.argv)>1 else 6000
Q=run(~hold,steps)
print('held-out harm %.1f dB noise %.1f dB | train %.1f / %.1f'%(*err(Q,hold),*err(Q,~hold)))
# baseline for the held-out notes: copy the nearest training note's harmonic amplitudes at the same frame
if len(sys.argv)>3:
    Qf=run(np.ones(len(X),bool),steps); print('final all %.1f / %.1f'%err(Qf,np.ones(len(X),bool))); np.savez('sax_env.npz',*Qf)
