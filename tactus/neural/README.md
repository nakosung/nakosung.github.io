# Tactus neural sax (`nsax`)

A small network that learned the saxophone's spectral envelope from the 20 sax recordings in Tactus's built-in kit (VCSL, CC0). The weights are embedded in [`../play/`](../play/). There, WebGPU runs the network for every 5 ms frame and then sums the resulting harmonics and breath noise into sound.

## What the network does

The network is 14 → 64 → 64 → 1 with tanh activations, about 5,200 weights. It is asked about one spectral component at a time. Each query passes in:

- the component's frequency, plus sines of the log-frequency at four scales so the network can draw sharp formants;
- whether the component is a harmonic or a band of breath noise;
- the note;
- the dynamic layer (soft or normal);
- the loudness at that moment;
- the time since the attack.

It returns the component's level relative to the frame's total. Because it is queried by frequency instead of by harmonic number, it learns a smooth envelope across pitches rather than a table for each recorded note.

At play time, each note gets a loudness curve: breath attack, a slight swell and sag, and the release. The network is run on 48 harmonics and 16 noise bands for every frame. The synthesis then:

- adds vibrato to the harmonics;
- renders each noise band as 64 random sines.

## Accuracy

Measured on two notes left out of training (C4 normal, D5 soft), as mean absolute error in dB:

| Method | Harmonics |
|---|---|
| This network | 9.5 dB |
| Re-pitching the nearest recorded note | 13–20 dB |

Re-pitching the nearest recorded note is what sample playback does. On the training notes the network is within about 4.4 dB per harmonic.

## Files

| File | Purpose |
|---|---|
| `analyze.py` | Decodes each recording and tracks f0. Measures 48 harmonic amplitudes and 16 bands of residual noise every 5 ms. |
| `train.py` | Trains the network in plain NumPy (Adam, weight decay 1e-3, 6,000 steps), reports the held-out error, and saves the weights. |

Running them needs `ffmpeg` and NumPy. `analyze.py` expects the kit's sax MP3s in `../kit/` as `sax_<midi>_<index>.mp3`. Indices 0–7 are the soft layer and 8–19 the normal layer. To train, run `python3 train.py 6000 64 save` with `WD=0.001` set in the environment.
