# Beyond Token Time — Revision History

The web note and position paper share the same version number. Git history is the source of truth; this file records the conceptual changes that define each published revision.

## Published paper snapshots

- **v1.1 (2026-09-02):** [paper-v1.1.pdf](./paper-v1.1.pdf) · [paper-v1.1.tex](./paper-v1.1.tex)
- **v1.0 (2026-08-21):** [paper-v1.0.pdf](./paper-v1.0.pdf) · [paper-v1.0.tex](./paper-v1.0.tex)
- `paper.pdf` and `paper.tex` remain aliases for the latest version (currently v1.1).
- A compact version index is available at [paper-versions.html](./paper-versions.html).

## v1.1 — 2026-09-02

- Added the proposed **null-token + gated continuous-vector recurrence** mechanism.
- Reframed explicit chain-of-thought as a **discrete recurrent loop** whose intermediate state is forced through a semantic token bottleneck.
- Distinguished **fixed recurrent depth** from **adaptive variable recurrence** learned through a null/silent action.
- Added a continuous-reasoning training stage, with RL as a natural way to learn variable loop count while noting that supervised/distilled variants are possible.
- Added inference/serving implications: per-sequence latent side state, dynamic input embeddings, KV-cache updates for silent steps, batching/scheduling, and suppression of null actions from visible output.
- Added the extension from one latent carry vector to multiple persistent latent channels.
- Added a monitorability section: once reasoning can proceed without serialization, human-readable traces are no longer structurally guaranteed and monitoring must be deliberately preserved or reconstructed.
- Added a clearly labeled comparison with public reporting about OpenAI Astra. The paper **does not claim** that Astra uses this null-token mechanism; it treats it as one minimal candidate implementation that may be functionally related to reported recurrent depth.
- Added recurrent-depth / looped-transformer prior work and new compute-matched ablations.
- Regenerated `paper.pdf` from the v1.1 LaTeX source.

## v1.0 — 2026-08-21

Initial position paper and web note covering:

- Persistent Multi-Stream Latent World State.
- Modalities as lenses rather than the native format of intelligence.
- Latent Event Bus and selective event-driven state updates.
- Separation of token time, compute time, and world time.
- Observer-relative event ordering and causal precedence.
- Frame-conditioned latent readout.
- Compute-matched falsification experiments.

Last pre-v1.1 manuscript commit: [`948318f`](https://github.com/nakosung/nakosung.github.io/commit/948318f47134a9f3f2a890e798443c63c27c7d57).
