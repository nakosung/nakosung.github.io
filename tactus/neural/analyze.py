import numpy as np, subprocess, glob, re, json
SR=48000; HOP=240; K=48; NB=16; WIN=2048
KIT='../kit/'
def load(f):
    raw=subprocess.run(['ffmpeg','-v','error','-i',f,'-ac','1','-ar',str(SR),'-f','f32le','-'],capture_output=True,check=True).stdout
    return np.frombuffer(raw,np.float32).copy()
def mtof(m): return 440*2**((m-69)/12)
win=np.hanning(WIN)
nb_edges=np.geomspace(150,16000,NB+1)
data=[]
for f in sorted(glob.glob(KIT+'sax_*.mp3')):
    m=re.match(r'.*sax_(\d+)_(\d+)\.mp3',f); midi=int(m.group(1)); idx=int(m.group(2)); layer=0 if idx<8 else 1
    x=load(f)
    # onset
    pk=np.abs(x).max(); on=np.argmax(np.abs(x)>pk*0.02); x=x[max(0,on-96):]
    f0n=mtof(midi)
    frames=[]
    for c in range(0, len(x)-WIN, HOP):
        seg=x[c:c+WIN]*win
        S=np.fft.rfft(seg); mag=np.abs(S)*2/win.sum(); fr=np.fft.rfftfreq(WIN,1/SR)
        # refine f0 from harmonic peaks 1..4
        est=[]
        for h in range(1,5):
            lo=int((h*f0n*0.94)/SR*WIN); hi=int((h*f0n*1.06)/SR*WIN)+2
            j=lo+np.argmax(mag[lo:hi]); 
            if 1<=j<len(mag)-1:
                a,b,cc=np.log(mag[j-1]+1e-12),np.log(mag[j]+1e-12),np.log(mag[j+1]+1e-12); d=0.5*(a-cc)/(a-2*b+cc+1e-12)
                est.append((j+d)*SR/WIN/h)
        f0=np.median(est) if est else f0n
        if abs(f0/f0n-1)>0.05: f0=f0n
        H=np.zeros(K); harm_mask=np.zeros_like(mag,bool)
        for h in range(1,K+1):
            fh=h*f0
            if fh>min(16000,SR/2-500): break
            j=int(round(fh/SR*WIN)); lo=max(0,j-3); hi=min(len(mag),j+4)
            H[h-1]=mag[lo:hi].max(); harm_mask[lo:hi]=True
        resid=np.where(harm_mask,0,mag)
        Nz=np.zeros(NB)
        for b in range(NB):
            sel=(fr>=nb_edges[b])&(fr<nb_edges[b+1])&(~harm_mask)
            Nz[b]=np.sqrt((resid[sel]**2).mean()) if sel.any() else 0
        frames.append((f0,H,Nz))
    f0s=np.array([a for a,_,_ in frames]); Hs=np.array([b for _,b,_ in frames]); Ns=np.array([c for _,_,c in frames])
    data.append(dict(midi=midi,layer=layer,f0=f0s.tolist(),H=Hs.tolist(),N=Ns.tolist()))
    loud=20*np.log10(np.sqrt((Hs**2).sum(1))+1e-9)
    print(midi,layer,len(frames),'f0 dev cents %.1f'%(1200*np.log2(np.median(f0s)/f0n)),'loud peak %.1f end %.1f'%(loud.max(),loud[-1]), 'noise/harm %.3f'%(Ns.mean()/ (Hs.max(1).mean()+1e-9)))
json.dump(data,open('sax_frames.json','w'))
