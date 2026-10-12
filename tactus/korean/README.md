# Tactus Korean percussion

The janggu, buk, jing and kkwaenggwari in [Tactus](../) are synthesized one-shots made from physical models. No recordings are included: free (CC0) recordings of these instruments were not available. The one-shots are released as CC0.

| Instrument | Sound | Model |
|---|---|---|
| `kung` | Janggu low head (궁편), struck with the round-headed mallet | 2-D wave equation on a circular membrane: thick head, wide contact near the centre, tension glide on hard hits |
| `deok` | Janggu high head (채편), struck with the thin bamboo stick | Tight head struck near the rim with a narrow contact, plus the stick's dry crack |
| `buk`, `buktak` | Buk head, and a hit on the wooden shell | Membrane with a stick contact; modal wood shell |
| `jing` | Large bronze gong | Modal: 15 partials, each split into a slowly beating pair. The upper partials swell in after the hit. |
| `kkwaeng`, `kkwaengm` | Kkwaenggwari, open and hand-muted | Modal: 19 bright partials and the hard mallet's attack |

The membrane solver uses the same scheme as Tactus's GPU `membrane` voice:

- three rotating buffers;
- a damping term plus a term that damps high modes faster;
- a radiated sound that mixes the head's volume velocity (the thump) with a point pickup (the ring).

The gong partials come from reference recordings, used only for measurement:

- kkwaenggwari peaks at 840, 1190, 2196, 2684, 3755 Hz and up;
- jing's fundamental is about 118 Hz, beating between 113 and 124 Hz, with partials at 199, 234, 353, 363 and 468 Hz.

Octave-band levels were then matched to the same references.

Each instrument has round-robin hits and up to three strengths. They are encoded as MP3 (LAME VBR quality 2) inside the page.

`kr_synth.py` renders every one-shot as a WAV file. It needs only NumPy.
