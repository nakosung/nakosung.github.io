# Tactus DSP (Rust → WebAssembly)

The audio-thread voices of [Tactus](../), written in Rust and compiled to a small WebAssembly module that runs inside an `AudioWorklet`.

- **Karplus-Strong plucked strings.** Up to 16 voices at once, with vibrato and a release damping. They play the gayageum, koto, guitar and harp sounds.
- **A TB-303 style acid voice.** A polyBLEP saw or square feeds a saturating 4-pole ladder filter, with slide and accent.

The crate is `no_std` and never allocates, so the audio thread never pauses for garbage collection. The worklet passes each note event in at its exact sample frame and calls `render(n)` for the samples between events, so timing stays sample-accurate. If WebAssembly is unavailable, Tactus falls back to the original JavaScript voices; add `?dsp=js` to the URL to force that path. The build stamp in the app shows `· Rust DSP` when this module is running.

## Files

| File | Purpose |
|---|---|
| `src/lib.rs` | The voices: `init`, `ks_on`, `acid_on`, `live`, `render`, `out_l`, `out_r` |
| `tactus_dsp.wasm` | The compiled module (11.6 KB) that is embedded in `../play/index.html` |
| `bench/bench.js` | Renders the same events through the JavaScript voices (`bench/synth.js`) and through the module, then compares speed and output |

## Build

```sh
rustup target add wasm32-unknown-unknown
cargo build --release --target wasm32-unknown-unknown
# → target/wasm32-unknown-unknown/release/tactus_dsp.wasm
```

The committed binary was built in an environment that could not download the `wasm32` standard library. It was compiled with `rustc` against a `core` built from the Rust 1.97.0 sources, using `-C opt-level=3 -C lto -C panic=abort -C link-arg=-zstack-size=65536`. A normal toolchain with the target installed should give the same code.

## Benchmark

```sh
node bench/bench.js 30
```

Both paths render the same workload in 128-sample blocks: 16 ringing strings plus a 16th-note acid line, 30 seconds of audio at 48 kHz. Results are best of three on Node 22:

| | Time | Speed vs realtime |
|---|---|---|
| JavaScript voices | ≈1,000 ms | ≈30× |
| Rust → WASM | ≈640 ms | ≈47× |

That makes Rust about 1.6× faster on the same load.

The acid voice has no randomness, so its two outputs can be compared sample by sample: they match within 0.2% of peak. The strings start from random noise, so for them only the overall level is compared, and it matches.

Two details carry most of the speed:

1. **Flushing tiny values in the string delay lines.** A decaying string otherwise drifts into denormal floats, which are very slow on x86. Before this fix the Rust version was about 3× slower than JavaScript.
2. **Processing one string at a time across the block.** Each voice's state stays in registers instead of being reloaded for every sample.
