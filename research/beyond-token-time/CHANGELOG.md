# Beyond Token Time — Revision History

The web note and position paper share the same version number. Git history is the source of truth; this file records the conceptual changes that define each published revision.

## v1.2 — 2026-09-02

- Reframed the central thesis around the gap between the model's **broad internal representation space** and the narrower interfaces through which that state is forced to evolve.
- Unified four constraints as related **interface bottlenecks**: discrete tokens, fixed/shared timing, modality boundaries, and a single serial stream.
- Clarified the common failure mode as **premature commitment**: forcing state too early into a symbol, order, modality partition, or one update clock.
- Restored **modality as a lens** and **token / compute / world time** to first-class parts of the argument rather than background context.
- Repositioned null-token + gated latent recurrence from the center of the paper to a **concrete relaxation mechanism** for the token and timing bottlenecks.
- Repositioned persistent multi-stream latent state as a concrete relaxation of seriality and modality coupling.
- Reframed monitorability as the general cost of freeing internal representation from interface serialization, not merely a side effect of silent reasoning.
- Reorganized falsifiable tests by bottleneck: discrete, temporal, modal, serial, and monitorability cost.
- Published immutable `paper-v1.2.tex` / `paper-v1.2.pdf` snapshots while keeping `paper.tex` / `paper.pdf` as latest aliases.

## v1.1 — 2026-09-02

- Added the proposed **null-token + gated continuous-vector recurrence** mechanism.
- Reframed explicit chain-of-thought as a **discrete recurrent loop** whose intermediate state is forced through a semantic token bottleneck.
- Distinguished **fixed recurrent depth** from **adaptive variable recurrence** learned through a null/silent action.
- Added continuous-reasoning training, serving implications, multiple latent channels, monitorability discussion, and an explicitly speculative Astra comparison.

Immutable snapshots: [`paper-v1.1.pdf`](./paper-v1.1.pdf) · [`paper-v1.1.tex`](./paper-v1.1.tex)

## v1.0 — 2026-08-21

Initial position paper and web note covering persistent multi-stream latent world state, modalities as lenses, Latent Event Bus, separation of token / compute / world time, observer-relative event ordering, frame-conditioned latent readout, and compute-matched falsification experiments.

Immutable snapshots: [`paper-v1.0.pdf`](./paper-v1.0.pdf) · [`paper-v1.0.tex`](./paper-v1.0.tex)

Last pre-v1.1 manuscript commit: [`948318f`](https://github.com/nakosung/nakosung.github.io/commit/948318f47134a9f3f2a890e798443c63c27c7d57).
