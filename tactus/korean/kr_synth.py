"""Korean percussion one-shots, synthesized from physical models (CC0, made for Tactus).
Drum heads (janggu, buk) are solved as the 2-D wave equation on a circular membrane, the same scheme as
Tactus's GPU 'membrane' voice, with a stick or mallet contact shape, a tension glide on hard hits, and a
radiated sound that mixes the head's volume velocity (the thump) with a point pickup (the ring).
The gongs (jing, kkwaenggwari) are modal: partial frequencies and levels measured from reference recordings,
each partial split into a slowly beating pair, decays per partial, and a slow 'bloom' of the upper partials."""
import numpy as np, wave, sys
SR = 44100
rng = np.random.default_rng(7)

def membrane(f11, dur, strike_r, strike_w, amp, T60, bright, glide=0.1, gtau=0.05, N=96, pickup_mix=0.6):
    # radius and Courant number so the (1,1) mode lands on f11; stable while lam^2 + 4 kap <= 1/2
    kap = 0.004 + 0.03 * (1 - bright)
    lam_max = np.sqrt(0.48 - 4 * kap); J = 3.832
    R = 45.0; lam = 2 * np.pi * R * f11 / (J * SR)
    if lam > lam_max: R = max(8, np.floor(lam_max * J * SR / (2 * np.pi * f11))); lam = min(lam_max, 2 * np.pi * R * f11 / (J * SR))
    c0 = (N - 1) / 2; yy, xx = np.mgrid[0:N, 0:N]
    mask = (np.hypot(xx - c0, yy - c0) < R).astype(np.float32); mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = 0
    sig = 6.9078 / (T60 * SR) * 0.5
    sx, sy = c0 - strike_r * R, c0 + 0.08 * R
    r = np.hypot(xx - sx, yy - sy)
    shape = np.where(r < strike_w, 0.5 * (1 + np.cos(np.pi * r / strike_w)), 0) * mask
    prev = (-amp * shape * lam).astype(np.float32); cur = np.zeros_like(prev)
    n = int(dur * SR); out = np.zeros(n, np.float32)
    pa = (int(round(c0 + 0.12 * R)), int(round(c0 + 0.4 * R)))
    def lap(u): return np.roll(u, 1, 0) + np.roll(u, -1, 0) + np.roll(u, 1, 1) + np.roll(u, -1, 1) - 4 * u
    lp = lap(prev)
    for s in range(n):
        l2 = lam * lam * (1 + glide * amp * np.exp(-s / (gtau * SR)))
        lc = lap(cur)
        nxt = (2 * cur - (1 - sig) * prev + l2 * lc + kap * (lc - lp)) / (1 + sig) * mask
        v = nxt - cur
        out[s] = pickup_mix * v.sum() / (R * R) * 0.05 + (1 - pickup_mix) * v[pa]
        prev, cur, lp = cur, nxt, lc
    return out

def noise_burst(dur, f_lo, f_hi, decay):
    n = int(dur * SR); x = rng.standard_normal(n)
    X = np.fft.rfft(x); f = np.fft.rfftfreq(n, 1 / SR); X[(f < f_lo) | (f > f_hi)] = 0
    return (np.fft.irfft(X, n) * np.exp(-np.arange(n) / (decay * SR))).astype(np.float32)

def resonator(x, f, q, g):
    # two-pole band-pass: the janggu's hourglass body and the buk's barrel
    w = 2 * np.pi * f / SR; r = np.exp(-w / (2 * q)); a1, a2 = -2 * r * np.cos(w), r * r
    y = np.zeros_like(x); y1 = y2 = 0.0
    for i in range(len(x)): y0 = (1 - r) * x[i] - a1 * y1 - a2 * y2; y[i] = y0; y2, y1 = y1, y0
    return x + g * y

def modal(partials, dur, attack, bloom=None):
    n = int(dur * SR); t = np.arange(n) / SR; out = np.zeros(n)
    for f, db, t60, beat in partials:
        a = 10 ** (db / 20)
        for k, (df, w) in enumerate([(-beat / 2, 0.6), (beat / 2, 0.4)] if beat else [(0, 1)]):
            env = np.exp(-6.9078 * t / t60)
            if bloom and f > bloom[0]: env *= 1 - np.exp(-t / bloom[1]) * 0.85   # upper partials swell in after the hit
            out += a * w * env * np.sin(2 * np.pi * (f + df) * t + rng.uniform(0, 2 * np.pi))
    out *= np.minimum(1, t / attack)
    return out.astype(np.float32)

def norm(x, peak=0.89):
    x = x - x.mean() * 0; return (x / (np.abs(x).max() + 1e-12) * peak).astype(np.float32)
def fade(x, sec):
    n = min(len(x), int(sec * SR)); x = x.copy(); x[-n:] *= np.linspace(1, 0, n) ** 2; return x
def save(name, x):
    pcm = (np.clip(x, -1, 1) * 32767).astype('<i2')
    with wave.open(name + '.wav', 'wb') as w: w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())

def kung(f, amp, seed):   # gungpyeon: thick low head, round-headed gunggulchae mallet near the centre
    rng.bit_generator.advance(seed)
    x = membrane(f, 0.75, 0.3, 5.0, amp, 0.55, 0.5, glide=0.14, gtau=0.06, pickup_mix=0.45)
    x = norm(x) + amp * 0.05 * norm(noise_burst(0.75, 250, 2500, 0.006))     # the mallet's felt on the skin
    return fade(norm(x), 0.25)
def deok(f, amp, seed):   # chaepyeon: tight high head, thin bamboo yeolchae striking near the rim, a dry crack
    rng.bit_generator.advance(seed)
    x = membrane(f, 0.32, 0.72, 1.6, amp, 0.22, 0.85, glide=0.06, gtau=0.02, pickup_mix=0.35)
    x = norm(x) + amp * 0.55 * norm(noise_burst(0.32, 1800, 12000, 0.008)) + amp * 0.25 * norm(noise_burst(0.32, 600, 2500, 0.02))
    return fade(norm(x), 0.12)
def buk(f, amp, seed):    # buk: barrel drum, wooden stick on the head
    rng.bit_generator.advance(seed)
    x = membrane(f, 0.9, 0.45, 3.5, amp, 0.6, 0.45, glide=0.12, gtau=0.05, pickup_mix=0.7)
    x = norm(x) + amp * 0.3 * norm(noise_burst(0.9, 700, 5000, 0.008))
    return fade(norm(x), 0.3)
def buktak():             # a hit on the buk's wooden shell
    x = modal([(1180, 0, 0.09, 0), (1730, -4, 0.07, 0), (2650, -8, 0.05, 0), (3900, -12, 0.04, 0)], 0.25, 0.0005)
    x = norm(x) + 0.6 * norm(noise_burst(0.25, 1500, 8000, 0.004))
    return fade(norm(x), 0.08)
# measured: fundamental ~118 Hz beating with 113/124, then 199, 234, 299, 318, 342, 353, 363, 398, 468, 484 Hz
JING = [(1.000, 0, 9.0, 5.5), (1.686, -12, 7.0, 3.0), (1.983, -10, 6.5, 2.5), (2.534, -18, 5.0, 2.0), (2.695, -15, 5.0, 2.2), (2.898, -19, 4.5, 1.6),
        (2.992, -6, 5.5, 2.4), (3.076, -7, 5.0, 2.0), (3.373, -16, 4.0, 1.5), (3.966, -11, 3.6, 1.8), (4.102, -13, 3.4, 1.2), (5.03, -20, 2.6, 1.0),
        (5.9, -24, 2.0, 0.8), (7.4, -28, 1.5, 0.6), (9.1, -32, 1.1, 0.5)]
def jing(f0, amp):
    x = modal([(f0 * r, db + (9 if r > 1.5 else 0) + (8 if r > 3.5 else 0) - (1 - amp) * 6 * (r - 1) / 8, t60, b) for r, db, t60, b in JING], 7.0, 0.012, bloom=(f0 * 1.6, 0.25))
    x = norm(x) + amp * 0.08 * norm(noise_burst(7.0, 400, 3000, 0.05))          # the padded mallet's thud
    return fade(norm(x), 2.0)
# measured: 840, 1190, 1585, 2054, 2196, 2684, 2864, 3262, 3367, 3755, 4630, 4883, 5564, 5930 Hz
KKW = [(1.000, -2, 1.4, 6), (1.417, 0, 1.2, 9), (1.887, -13, 0.9, 5), (2.445, -13, 0.8, 7), (2.614, -6, 0.9, 8), (3.195, -5, 0.8, 11), (3.410, -8, 0.7, 6),
       (3.883, -11, 0.6, 9), (4.008, -12, 0.6, 7), (4.470, -5, 0.6, 12), (5.512, -13, 0.45, 8), (5.813, -11, 0.45, 10), (6.624, -10, 0.35, 9), (7.06, -14, 0.3, 8),
       (8.3, -12, 0.25, 7), (9.6, -14, 0.2, 6), (11.2, -16, 0.15, 6), (12.7, -19, 0.12, 5), (14.3, -22, 0.1, 5)]
def kkwaeng(f0, amp, muted=False):
    damp = 0.09 if muted else 1.0
    x = modal([(f0 * r, db + (8 if r > 3 else 0) + (6 if r > 5 else 0), t60 * damp + (0.04 if muted else 0), b) for r, db, t60, b in KKW], 0.5 if muted else 1.8, 0.0008)
    x = norm(x) + amp * 0.7 * norm(noise_burst(len(x) / SR, 3500, 16000, 0.02))  # hard mallet on brass
    return fade(norm(x), 0.06 if muted else 0.5)

if __name__ == '__main__':
    out = {}
    for k, (f, a) in enumerate([(172, 0.55), (166, 0.8), (174, 0.8), (162, 1.1)]): save(f'kung_{k}', kung(f, a, k))
    for k, (f, a) in enumerate([(360, 0.55), (352, 0.8), (366, 0.8), (348, 1.1)]): save(f'deok_{k}', deok(f, a, k))
    for k, (f, a) in enumerate([(80, 0.6), (78, 0.9), (82, 0.9), (76, 1.2)]): save(f'buk_{k}', buk(f, a, k))
    save('buktak_0', buktak())
    save('jing_0', jing(118, 0.6)); save('jing_1', jing(118, 1.0)); save('jing_2', jing(119.5, 1.0))
    save('kkwaeng_0', kkwaeng(840, 0.6)); save('kkwaeng_1', kkwaeng(840, 1.0)); save('kkwaeng_2', kkwaeng(846, 1.0))
    save('kkwaengm_0', kkwaeng(840, 0.9, True)); save('kkwaengm_1', kkwaeng(838, 1.0, True))
    print('done')
