# Measure kkwaenggwari partials (frequency, level, decay) from isolated hits of a reference performance,
# for kkw_synth.py. Usage: python3 kkw_fit.py <recording>. Only measured parameters are kept, no audio.
import numpy as np, subprocess, json, wave, sys
SR=44100; N=4096; H=256
x=np.frombuffer(subprocess.run(['ffmpeg','-v','error','-i',(sys.argv[1] if len(sys.argv) > 1 else 'kkwaenggwari.mp3'),'-ac','1','-ar',str(SR),'-f','f32le','-'],capture_output=True).stdout,np.float32)
def onset_near(t):
    i=int(t*SR); seg=np.abs(x[i-2000:i+2000]); return i-2000+int(np.argmax(seg>seg.max()*0.3))
def hit_stft(t, dur):
    o=onset_near(t)-200; frames=int(dur*SR/H)
    w=np.hanning(N); return np.array([np.abs(np.fft.rfft(x[o+k*H:o+k*H+N]*w)) for k in range(frames)])
fr=np.fft.rfftfreq(N,1/SR)
def fit(times, dur):
    S=np.mean([hit_stft(t,dur)**2 for t in times],axis=0)  # power, frames x bins
    P=10*np.log10(S+1e-12)
    early=P[1:4].max(0)
    # peaks: local maxima above a smoothed floor
    k=31; floor=np.convolve(early,np.ones(k)/k,'same')
    cand=[j for j in range(3,len(early)-3) if 250<fr[j]<16000 and early[j]==early[j-3:j+4].max() and early[j]>floor[j]+6]
    cand=sorted(cand,key=lambda j:-early[j])[:160]
    parts=[]
    tt=np.arange(P.shape[0])*H/SR
    for j in sorted(cand):
        tr=P[:,j-1:j+2].max(1); pk=tr.max(); i0=int(np.argmax(tr))
        use=(np.arange(len(tr))>=i0)&(tr>pk-45)
        if use.sum()<3: continue
        slope=np.polyfit(tt[use],tr[use],1)[0]   # dB/s
        t60=float(np.clip(-60/min(slope,-1),0.02,4))
        # refine frequency by parabolic interpolation
        a,b,c=early[j-1],early[j],early[j+1]; d=0.5*(a-c)/(a-2*b+c+1e-9)
        parts.append(((j+d)*SR/N, float(pk), t60))
    # residual noise: early spectrum envelope minus peaks, in 1/3-octave bands, with its decay
    bands=np.geomspace(200,18000,22); nz=[]
    for b0,b1 in zip(bands[:-1],bands[1:]):
        sel=(fr>=b0)&(fr<b1)
        if not sel.any(): continue
        tr=10*np.log10(S[:,sel].mean(1)+1e-12); pk=tr.max(); i0=int(np.argmax(tr)); use=(np.arange(len(tr))>=i0)&(tr>pk-40)
        slope=np.polyfit(tt[use],tr[use],1)[0] if use.sum()>=3 else -200
        nz.append((float(np.sqrt(b0*b1)), float(np.percentile(10*np.log10(S[1:4,sel]+1e-12),30)), float(np.clip(-60/min(slope,-1),0.02,3))))
    top=max(p[1] for p in parts)
    return [(f,l-top,t) for f,l,t in parts], [(f,l-top,t) for f,l,t in nz]
open_hits=[0.49,0.89,2.12,2.56,2.98,3.78,13.18,16.53]
muted_hits=[1.72,3.39,13.55,26.93]
po,no=fit(open_hits,0.34); pm,nm=fit(muted_hits,0.3)
json.dump({'open':{'partials':po,'noise':no},'muted':{'partials':pm,'noise':nm}},open('kkw_fit.json','w'))
print('open partials',len(po),'muted',len(pm))
for f,l,t in po[:40]: print('%7.0f Hz %6.1f dB  T60 %.2fs'%(f,l,t))
print('noise bands', [(round(f),round(l),round(t,2)) for f,l,t in no])
