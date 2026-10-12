# Resynthesize kkwaenggwari hits from measured partials (kkw_fit.json): each partial a slightly beating pair
# with its own decay, plus third-octave bands of decaying noise calibrated to the measured band levels.
import numpy as np, json, wave
SR=44100; N=4096; H=256
rng=np.random.default_rng(11)
fit=json.load(open('kkw_fit.json'))
def stft_early(y):
    w=np.hanning(N); S=np.array([np.abs(np.fft.rfft(y[k*H:k*H+N]*w))**2 for k in range(1,4)]); return S.mean(0)
fr=np.fft.rfftfreq(N,1/SR)
def band_noise(fc, t60, n):
    lo,hi=fc/2**(1/6),fc*2**(1/6); z=rng.standard_normal(n); Z=np.fft.rfft(z); f=np.fft.rfftfreq(n,1/SR); Z[(f<lo)|(f>hi)]=0
    return np.fft.irfft(Z,n)*np.exp(-6.9078*np.arange(n)/SR/t60)
def render(spec, dur, pitch=1.0, gains=None, seed=0):
    n=int(dur*SR); t=np.arange(n)/SR; y=np.zeros(n)
    rs=np.random.default_rng(seed)
    for f,l,t60 in spec['partials']:
        a=10**(l/20); beat=rs.uniform(1.5,7)
        for df,wt in ((-beat/2,0.55),(beat/2,0.45)):
            y+=a*wt*np.exp(-6.9078*t/t60)*np.sin(2*np.pi*(f*pitch+df)*t+rs.uniform(0,2*np.pi))
    nz=np.zeros(n); bands=spec['noise']
    for i,(fc,l,t60) in enumerate(bands):
        if l<-100: continue
        g=(gains[i] if gains is not None else 1.0)
        nz+=g*10**(l/20)*band_noise(fc*pitch,t60,n)
    y=y+nz
    y*=np.minimum(1,t/0.0006)
    return y
def calibrate(spec, dur):
    # match each noise band's early level (per STFT bin) to the measurement, relative to the strongest partial
    gains=np.ones(len(spec['noise']))
    for it in range(3):
        y=render(spec,dur,gains=gains,seed=1); S=stft_early(np.concatenate([np.zeros(200),y])); P=10*np.log10(S+1e-12)
        top=P[(fr>800)&(fr<900)].max()
        for i,(fc,l,t60) in enumerate(spec['noise']):
            if l<-100: continue
            sel=(fr>=fc/2**(1/6))&(fr<fc*2**(1/6))
            have=np.percentile(P[sel],30)-top
            gains[i]*=10**((l-have)/20)
    return gains
def fade(y,sec):
    k=int(sec*SR); y=y.copy(); y[-k:]*=np.linspace(1,0,k)**2; return y
def save(name,y):
    y=y/np.abs(y).max()*0.89; pcm=(y*32767).astype('<i2')
    with wave.open(name+'.wav','wb') as w: w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
go=calibrate(fit['open'],1.6); gm=calibrate(fit['muted'],0.4)
for k,(p,s) in enumerate([(1.0,0),(0.996,1),(1.004,2)]): save(f'kkwaeng_{k}',fade(render(fit['open'],1.6,p,go,s),0.4))
for k,(p,s) in enumerate([(1.0,3),(1.003,4)]):
    y=render(fit['muted'],0.4,p,gm,s); y*=np.exp(-np.arange(len(y))/SR/0.07)   # the hand stops it
    save(f'kkwaengm_{k}',fade(y,0.08))
print('ok', np.round(go,2))
