// ===================== Tactus GPU (WebGPU) =====================
// The audio thread cannot see the GPU (WebGPU is not exposed inside an AudioWorklet), so the GPU works
// ahead of time on the main thread: it renders whole notes and textures into buffers that the scheduler
// then plays at the exact time. That suits sounds that are too heavy for the CPU but can be prepared a
// moment early:
//   swarm  a supersaw pad of 40 detuned voices x 48 harmonics (about 1,900 sine partials per note)
//   choir  24 detuned voices with vowel formants (about 1,000 partials per note)
//   plate  a struck metal plate: every inharmonic mode split into six beating neighbours, each with its own decay
//          (a few hundred partials in the middle register, more for low notes)
//   membrane  a drum head simulated as the 2-D wave equation on a 96 x 96 grid, one time step per sample,
//          four velocities at once (one GPU workgroup each)
//   nsax   a small neural network trained on the sax recordings drives an additive voice (see 'neural voice')
//   fx ... | freeze   spectral freeze: the live mix's spectrum resynthesized as ~2,000 sines
//   space hall        modal reverb: up to 65,536 room modes rendered into the reverb's impulse response
// and it draws four backdrops behind the code (terrain, contour, dots, demo) as full-screen shaders.
// Without WebGPU every part falls back: the GPU instruments to CPU voices, freeze to a looped grain.

const GPU = { state: 'idle', dev: null, name: '', err: '', synth: null, cache: new Map(), bytes: 0, jobs: new Map(), chain: Promise.resolve(), renders: 0, lastMs: 0 };
const GPU_INSTS = { swarm: 'supersaw', choir: 'pad', plate: 'bell', membrane: 'pluck', nsax: 'sax' };   // and their CPU fallbacks
const GPU_CACHE_BYTES = 160 << 20;

function gpuInit() {
  if (GPU.ready) return GPU.ready;
  GPU.state = 'loading';
  GPU.ready = (async () => {
    try {
      if (!navigator.gpu) throw new Error('WebGPU is not available in this browser');
      if (/[?&]gpu=off\b/.test(location.search)) throw new Error('turned off with ?gpu=off');
      const ad = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (!ad) throw new Error('no GPU adapter');
      GPU.adapter = ad;   // keep the adapter alive: if it is collected, pending GPU reads fail
      const info = ad.info || {};
      GPU.name = [info.vendor, info.architecture || info.description].filter(Boolean).join(' ') || 'GPU';
      GPU.dev = await ad.requestDevice({ requiredLimits: { maxStorageBufferBindingSize: Math.min(ad.limits.maxStorageBufferBindingSize, 1 << 27), maxBufferSize: Math.min(ad.limits.maxBufferSize, 1 << 27) } });
      GPU.dev.lost.then(i => { GPU.state = 'off'; GPU.err = 'GPU device lost: ' + (i.message || i.reason); S.dirty = true; });
      gpuBuildSynth();
      GPU.state = 'ok';
    } catch (e) { GPU.state = 'off'; GPU.err = String(e.message || e); }
    S.dirty = true;
    return GPU.state === 'ok';
  })();
  return GPU.ready;
}

// ---------- the partial synthesizer ----------
// Each GPU thread computes one output sample as a sum over every partial. Partials are staged through
// workgroup memory 256 at a time. Phase is kept precise over long notes by splitting the sample index
// into 4096-sample blocks: cycles = fract(cyclesPerBlock * block) + freq/sr * (index within block).
const SYNTH_WGSL = `
struct P { n: u32, np: u32, atk: f32, rel0: f32, relk: f32, gain: f32, win: u32, off: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> parts: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> outb: array<f32>;
var<workgroup> tile: array<vec4f, 512>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) li: u32) {
  let i = gid.x + p.off;
  let hi = f32(i >> 12u);
  let lo = f32(i & 4095u);
  let t = f32(i);
  var l = 0.0;
  var r = 0.0;
  for (var base = 0u; base < p.np; base += 256u) {
    let k = base + li;
    if (k < p.np) { tile[li * 2u] = parts[k * 2u]; tile[li * 2u + 1u] = parts[k * 2u + 1u]; }
    else { tile[li * 2u] = vec4f(0.0); tile[li * 2u + 1u] = vec4f(0.0); }
    workgroupBarrier();
    let m = min(256u, p.np - base);
    for (var j = 0u; j < m; j++) {
      let a = tile[j * 2u];
      let b = tile[j * 2u + 1u];
      let cyc = fract(fract(a.y * hi) + a.x * lo + a.z);
      let s = sin(6.2831853 * cyc) * a.w * exp(-b.x * t);
      l += s * b.y;
      r += s * b.z;
    }
    workgroupBarrier();
  }
  if (i >= p.n) { return; }
  var env = min(1.0, t / max(1.0, p.atk));
  if (t > p.rel0) { env *= exp(-(t - p.rel0) * p.relk); }
  if (p.win == 1u) { env *= sin(3.14159265 * t / f32(p.n)); }
  outb[i] = l * env * p.gain;
  outb[i + p.n] = r * env * p.gain;
}`;
function gpuBuildSynth() {
  const d = GPU.dev;
  GPU.synth = d.createComputePipeline({ layout: 'auto', compute: { module: d.createShaderModule({ code: SYNTH_WGSL }), entryPoint: 'main' } });
}
// parts: Float32Array, 8 floats per partial: freq/sr, fract(freq*4096/sr), phase(cycles), amp, decay/sample, gainL, gainR, 0
// The work is cut into slices of 16,384 samples, one submit each, so no single dispatch runs long
// enough to trip the GPU watchdog on slower machines.
const GPU_SLICE = 16384;
async function gpuRender(parts, np, n, o = {}) {
  const d = GPU.dev, t0 = performance.now();
  const pb = d.createBuffer({ size: Math.max(32, np * 32), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  d.queue.writeBuffer(pb, 0, parts.buffer, parts.byteOffset, np * 32);
  const ob = d.createBuffer({ size: n * 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const rb = d.createBuffer({ size: n * 8, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const unis = [];
  for (let off = 0; off < n; off += GPU_SLICE) {
    const uni = d.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const u = new ArrayBuffer(32), dv = new DataView(u);
    dv.setUint32(0, n, true); dv.setUint32(4, np, true); dv.setFloat32(8, o.atk || 1, true); dv.setFloat32(12, o.rel0 ?? 1e30, true);
    dv.setFloat32(16, o.relk || 0, true); dv.setFloat32(20, o.gain ?? 1, true); dv.setUint32(24, o.win ? 1 : 0, true); dv.setUint32(28, off, true);
    d.queue.writeBuffer(uni, 0, u); unis.push(uni);
    const bg = d.createBindGroup({ layout: GPU.synth.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: uni } }, { binding: 1, resource: { buffer: pb } }, { binding: 2, resource: { buffer: ob } }] });
    const enc = d.createCommandEncoder(), pass = enc.beginComputePass();
    pass.setPipeline(GPU.synth); pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(Math.min(GPU_SLICE, n - off) / 256));
    pass.end();
    d.queue.submit([enc.finish()]);
    await d.queue.onSubmittedWorkDone();
  }
  const enc = d.createCommandEncoder();
  enc.copyBufferToBuffer(ob, 0, rb, 0, n * 8);
  d.queue.submit([enc.finish()]);
  await rb.mapAsync(GPUMapMode.READ);
  const out = new Float32Array(rb.getMappedRange().slice(0));
  rb.unmap(); for (const b of [pb, ob, rb, ...unis]) b.destroy();
  GPU.renders++; GPU.lastMs = performance.now() - t0;
  return out;
}
// the same sum on the CPU, for the benchmark and as a reference (renders samples [a, b) only)
function cpuRender(parts, np, n, o = {}, a = 0, b = n) {
  const out = new Float32Array((b - a) * 2), TAU = Math.PI * 2;
  for (let k = 0; k < np; k++) {
    const q = k * 8, fps = parts[q], cpb = parts[q + 1], ph = parts[q + 2], amp = parts[q + 3], dk = parts[q + 4], gl = parts[q + 5], gr = parts[q + 6];
    for (let i = a; i < b; i++) {
      const c = (cpb * (i >> 12)) % 1 + fps * (i & 4095) + ph;
      const s = Math.sin(TAU * c) * amp * Math.exp(-dk * i);
      out[i - a] += s * gl; out[i - a + (b - a)] += s * gr;
    }
  }
  return out;
}

// ---------- GPU instruments ----------
function rngOf(seed) { let s = seed | 0 || 1; return () => { s ^= s << 13; s ^= s >>> 17; s ^= s << 5; return (s >>> 0) / 4294967296; }; }
function partial(arr, k, f, sr, amp, phase, decayT60, pan) {
  const q = k * 8, fps = f / sr, p = (Math.max(-1, Math.min(1, pan)) + 1) * Math.PI / 4;
  arr[q] = fps; arr[q + 1] = (f * 4096 / sr) % 1; arr[q + 2] = phase; arr[q + 3] = amp;
  arr[q + 4] = decayT60 > 0 ? 6.9078 / (decayT60 * sr) : 0; arr[q + 5] = Math.cos(p); arr[q + 6] = Math.sin(p); arr[q + 7] = 0;
}
const FORMANTS = { a: [[800, 80, 1], [1150, 90, 0.5], [2900, 120, 0.25], [3900, 130, 0.2]], o: [[450, 70, 1], [800, 80, 0.35], [2830, 100, 0.1], [3800, 130, 0.08]] };
// returns partials plus render options for one note
function gpuVoice(inst, midi, dur, P, sr) {
  const f0 = mtof(midi), ny = sr * 0.45, rnd = rngOf(midi * 7919 + 17);
  const parts = new Float32Array(4096 * 8); let np = 0, sum2 = 0;
  const add = (f, amp, ph, t60, pan) => { if (f <= 20 || f >= ny || np >= 4096 || amp <= 0) return; partial(parts, np++, f, sr, amp, ph, t60, pan); sum2 += amp * amp; };
  const atk = Math.max(0.002, P.atk), rel = Math.max(0.05, P.rel), width = Math.max(0, Math.min(1, P.width ?? 0.7));
  let n, o;
  if (inst === 'plate') {
    // simply supported rectangular plate: f_mn grows with m^2 + n^2 (scaled for the aspect ratio)
    const ay = 1.37, x0 = 0.31, y0 = 0.43, base = 1 + 1 / (ay * ay), T = 1.2 + 6 * Math.max(0.05, P.dec);
    const bright = Math.max(0, Math.min(1, P.bright ?? 0.5));
    for (let m = 1; m <= 32; m++) for (let k = 1; k <= 32; k++) {
      const f = f0 * (m * m + (k * k) / (ay * ay)) / base;
      if (f >= ny) continue;
      const shape = Math.abs(Math.sin(m * Math.PI * x0) * Math.sin(k * Math.PI * y0));
      const amp = shape / Math.pow(f / f0, 1.25 - 0.7 * bright), t60 = Math.max(0.15, T * Math.pow(f0 / f, 0.5));
      // a real plate is never perfectly symmetric, so each mode splits into close neighbours that beat slowly
      for (let c = 0; c < 6; c++) add(f * (1 + (rnd() - 0.5) * 0.004 * (c ? 1 : 0)), amp * (c ? 0.35 + rnd() * 0.4 : 1), c ? rnd() : 0, t60 * (0.8 + rnd() * 0.4), (rnd() * 2 - 1) * width * 0.6);
    }
    n = Math.round(Math.min(10, T * 1.05) * sr);
    o = { atk: 0.0015 * sr };
  } else {
    // swarm and choir: many detuned voices, each a full harmonic series; their beating is the chorus
    const choir = inst === 'choir', V = choir ? 24 : 40, H = choir ? 44 : 48, det = Math.max(2, P.detune ?? 18) * (choir ? 0.6 : 1);
    const bright = Math.max(0, Math.min(1, P.bright ?? 0.6)), fc = f0 * (3 + 45 * bright);
    const vow = FORMANTS.a.map((a, i) => [a[0] + (FORMANTS.o[i][0] - a[0]) * (1 - bright), a[1], a[2] + (FORMANTS.o[i][2] - a[2]) * (1 - bright)]);
    for (let v = 0; v < V; v++) {
      const x = V > 1 ? v / (V - 1) * 2 - 1 : 0, cents = x * det + (rnd() - 0.5) * det * 0.3, fv = f0 * Math.pow(2, cents / 1200), pan = x * width;
      for (let h = 1; h <= H; h++) {
        const f = fv * h;
        let amp;
        if (choir) { amp = 0.08 / h; for (const [F, B, G] of vow) amp += G / (1 + ((f - F) / B) ** 2) / Math.sqrt(h); }
        else amp = (1 / h) / Math.sqrt(1 + Math.pow(f / fc, 4));
        add(f, amp, rnd(), 0, pan);
      }
    }
    const hold = Math.max(0.05, dur);
    n = Math.round(Math.min(10, hold + rel * 3) * sr);
    o = { atk: atk * sr, rel0: hold * sr, relk: 1 / (rel / 3 * sr) * 1.0 };
  }
  // random phases make the sum's level grow as the root of the summed powers
  o.gain = 0.22 / Math.sqrt(Math.max(1e-9, sum2 / 2));
  return { parts, np, n, o };
}
// ---------- membrane: the wave equation on a drum head ----------
// Finite differences on a 96 x 96 grid; cells outside a circle stay at zero, which is the rim. Each GPU
// workgroup owns one head and steps it once per output sample: every thread updates its share of the
// cells, then all wait at a barrier before the next step. Three buffers rotate (previous, current, next).
// Loss is a plain damping term plus a stiffness-like term that damps high modes faster. A hard hit
// tightens the skin for a moment, so the pitch starts high and settles: the tension term follows the hit.
const MEM_N = 96, MEM_LAYERS = 4, MEM_STEPS = 2048;
const MEMBRANE_WGSL = `
struct MP { n: u32, steps: u32, start: u32, N: u32, lam2: f32, sig: f32, kap: f32, glide: f32, gtau: f32, p0: f32, p1: f32, p2: f32 };
@group(0) @binding(0) var<uniform> mp: MP;
@group(0) @binding(1) var<storage, read_write> st: array<f32>;
@group(0) @binding(2) var<storage, read> mask: array<f32>;
@group(0) @binding(3) var<storage, read_write> outb: array<f32>;
@group(0) @binding(4) var<storage, read> vp: array<vec4f>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let v = wg.x;
  let NN = mp.N * mp.N;
  let base = v * 3u * NN;
  let pk = vp[v];
  for (var s = 0u; s < mp.steps; s++) {
    let gs = mp.start + s;
    let prv = base + (gs % 3u) * NN;
    let cur = base + ((gs + 1u) % 3u) * NN;
    let nxt = base + ((gs + 2u) % 3u) * NN;
    let l2 = mp.lam2 * (1.0 + pk.w * mp.glide * exp(-f32(gs) / mp.gtau));
    for (var c = li; c < NN; c += 256u) {
      if (mask[c] == 0.0) { continue; }
      let uc = st[cur + c];
      let lap = st[cur + c - 1u] + st[cur + c + 1u] + st[cur + c - mp.N] + st[cur + c + mp.N] - 4.0 * uc;
      let up = st[prv + c];
      let lapP = st[prv + c - 1u] + st[prv + c + 1u] + st[prv + c - mp.N] + st[prv + c + mp.N] - 4.0 * up;
      st[nxt + c] = (2.0 * uc - (1.0 - mp.sig) * up + l2 * lap + mp.kap * (lap - lapP)) / (1.0 + mp.sig);
    }
    storageBarrier();
    if (li == 0u && gs < mp.n) {
      // the pickups hear the skin's velocity
      let a = u32(pk.x); let b = u32(pk.y);
      outb[v * mp.n * 2u + gs] = (st[nxt + a] - st[cur + a]) * pk.z;
      outb[v * mp.n * 2u + mp.n + gs] = (st[nxt + b] - st[cur + b]) * pk.z;
    }
    storageBarrier();
  }
}`;
function gpuBuildMembrane() {
  const d = GPU.dev;
  GPU.mem = d.createComputePipeline({ layout: 'auto', compute: { module: d.createShaderModule({ code: MEMBRANE_WGSL }), entryPoint: 'main' } });
}
// set-up shared by the GPU and the JavaScript reference: grid, strike shapes, pickups, coefficients
function membraneSetup(midi, P, sr) {
  const N = MEM_N, f0 = mtof(midi);
  const dec = Math.max(0.05, P.dec ?? 1), bright = Math.max(0, Math.min(1, P.bright ?? 0.5));
  const T60 = 0.18 + 0.9 * dec, sig = 6.9078 / (T60 * sr) * 0.5;
  const kap = 0.004 + 0.03 * (1 - bright);
  // the note is the (1,1) mode, as on a timpani, where it is the tone you hear: f11 = 3.832 c / (2 pi R).
  // Pick the radius and Courant number to hit it; this scheme stays stable while lam^2 + 4 kap <= 1/2.
  const lamMax = Math.sqrt(0.48 - 4 * kap), J = 3.832;
  let R = 45, lam = 2 * Math.PI * R * f0 / (J * sr);
  if (lam > lamMax) { R = Math.max(8, Math.floor(lamMax * J * sr / (2 * Math.PI * f0))); lam = Math.min(lamMax, 2 * Math.PI * R * f0 / (J * sr)); }
  const c0 = (N - 1) / 2, mask = new Float32Array(N * N);
  for (let y = 1; y < N - 1; y++) for (let x = 1; x < N - 1; x++) if (Math.hypot(x - c0, y - c0) < R) mask[y * N + x] = 1;
  const n = Math.round(Math.min(4, T60 * 1.1) * sr);
  const idx = (dx, dy) => Math.round(c0 + dy * R) * N + Math.round(c0 + dx * R);
  const st = new Float32Array(MEM_LAYERS * 3 * N * N), vp = new Float32Array(MEM_LAYERS * 4);
  for (let L = 0; L < MEM_LAYERS; L++) {
    // harder hits: more force, a smaller contact patch (brighter), more tension glide
    const amp = [0.35, 0.6, 0.85, 1.1][L], w = (4.6 - L * 0.8) * (0.6 + 0.4 * R / 45) * (1.3 - 0.6 * bright);
    const sx = c0 - 0.62 * R, sy = c0 + 0.1 * R, base = L * 3 * N * N;   // struck about 40% in from the rim, as on a timpani
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      const r = Math.hypot(x - sx, y - sy);
      if (r < w && mask[y * N + x]) st[base + y * N + x] = -amp * 0.5 * (1 + Math.cos(Math.PI * r / w)) * lam;   // u(-1): initial velocity
    }
    vp.set([idx(-0.38, 0.12), idx(0.36, -0.15), 1, [0.25, 0.5, 0.75, 1][L]], L * 4);
  }
  return { N, R, lam2: lam * lam, sig, kap, n, mask, st, vp, glide: 0.12, gtau: 0.06 * sr };
}
async function gpuMembrane(midi, P, sr) {
  const d = GPU.dev, m = membraneSetup(midi, P, sr), t0 = performance.now();
  if (!GPU.mem) gpuBuildMembrane();
  const mk = (data, usage) => { const b = d.createBuffer({ size: Math.max(16, data.byteLength), usage: usage | GPUBufferUsage.COPY_DST }); d.queue.writeBuffer(b, 0, data); return b; };
  const sb = mk(m.st, GPUBufferUsage.STORAGE), mb = mk(m.mask, GPUBufferUsage.STORAGE), vb = mk(m.vp, GPUBufferUsage.STORAGE);
  const ob = d.createBuffer({ size: MEM_LAYERS * m.n * 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const rb = d.createBuffer({ size: MEM_LAYERS * m.n * 8, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const unis = [];
  for (let s0 = 0; s0 < m.n; s0 += MEM_STEPS) {
    const u = new ArrayBuffer(48), dv = new DataView(u);
    dv.setUint32(0, m.n, true); dv.setUint32(4, Math.min(MEM_STEPS, m.n - s0), true); dv.setUint32(8, s0, true); dv.setUint32(12, m.N, true);
    dv.setFloat32(16, m.lam2, true); dv.setFloat32(20, m.sig, true); dv.setFloat32(24, m.kap, true); dv.setFloat32(28, m.glide, true); dv.setFloat32(32, m.gtau, true);
    const ub = mk(new Uint8Array(u), GPUBufferUsage.UNIFORM); unis.push(ub);
    const bg = d.createBindGroup({ layout: GPU.mem.getBindGroupLayout(0), entries: [ub, sb, mb, ob, vb].map((b, i) => ({ binding: i, resource: { buffer: b } })) });
    const enc = d.createCommandEncoder(), pass = enc.beginComputePass();
    pass.setPipeline(GPU.mem); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(MEM_LAYERS); pass.end();
    d.queue.submit([enc.finish()]);
    await d.queue.onSubmittedWorkDone();
  }
  const enc = d.createCommandEncoder(); enc.copyBufferToBuffer(ob, 0, rb, 0, MEM_LAYERS * m.n * 8); d.queue.submit([enc.finish()]);
  await rb.mapAsync(GPUMapMode.READ);
  const out = new Float32Array(rb.getMappedRange().slice(0)); rb.unmap();
  for (const b of [sb, mb, vb, ob, rb, ...unis]) b.destroy();
  GPU.renders++; GPU.lastMs = performance.now() - t0;
  return { out, n: m.n };
}
// the same simulation in JavaScript, for the benchmark and as a reference (one layer, first `steps` samples)
function cpuMembrane(m, L, steps) {
  const N = m.N, NN = N * N, B = [m.st.slice(L * 3 * NN, L * 3 * NN + NN), m.st.slice(L * 3 * NN + NN, L * 3 * NN + 2 * NN), new Float32Array(NN)];
  const out = new Float32Array(steps * 2), a = m.vp[L * 4], b = m.vp[L * 4 + 1], gw = m.vp[L * 4 + 3];
  for (let s = 0; s < steps; s++) {
    const prv = B[s % 3], cur = B[(s + 1) % 3], nxt = B[(s + 2) % 3], l2 = m.lam2 * (1 + gw * m.glide * Math.exp(-s / m.gtau));
    for (let c = 0; c < NN; c++) {
      if (!m.mask[c]) continue;
      const uc = cur[c], up = prv[c];
      const lap = cur[c - 1] + cur[c + 1] + cur[c - N] + cur[c + N] - 4 * uc, lapP = prv[c - 1] + prv[c + 1] + prv[c - N] + prv[c + N] - 4 * up;
      nxt[c] = (2 * uc - (1 - m.sig) * up + l2 * lap + m.kap * (lap - lapP)) / (1 + m.sig);
    }
    out[s] = nxt[a] - cur[a]; out[steps + s] = nxt[b] - cur[b];
  }
  return out;
}
// ---------- neural voice: nsax ----------
// A small network (14 -> 64 -> 64 -> 1, tanh) learned the sax's spectral envelope from the kit's recordings:
// given a frequency, whether it is a harmonic or breath noise, the note, the dynamic layer, the loudness at
// that moment and the time since the attack, it predicts that component's level. Because it is asked one
// frequency at a time it learns a smooth envelope instead of a table per recorded note; on notes left out
// of training it is off by about 9.5 dB per harmonic, where re-pitching the nearest recording is off by 13-20.
// The GPU runs the network for every 5 ms frame x 64 components, then an additive pass sums 48 harmonics
// (with vibrato) and 16 bands of breath noise (64 random sines each) for every sample.
const NS = { K: 48, NB: 16, NOISE: 64, FPS: 200, H: 64, IN: 14, ST: 16, w: null };
const NS_BANDS = (() => { const e = []; for (let b = 0; b <= 16; b++) e.push(150 * Math.pow(16000 / 150, b / 16)); return e; })();
function nsWeights() {
  if (NS.w) return NS.w;
  const bin = atob(NSAX_NET_B64), u = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
  return NS.w = new Float32Array(u.buffer);
}
// loudness (dB) over the note: breath attack, a slight swell and sag while held, then the release
function nsLoudness(nF, hold, rel, atk, rnd) {
  const L = new Float32Array(nF), dt = 1 / NS.FPS;
  let drift = 0;
  for (let j = 0; j < nF; j++) {
    const t = j * dt;
    let db = t < atk ? -45 + 45 * Math.pow(t / atk, 0.6) : -1.2 * Math.min(1, (t - atk) / 0.4) + 1.2 * Math.sin(Math.min(1, (t - atk) / 1.5) * Math.PI) * 0.5 - 0.6 * (t - atk);
    drift += (rnd() - 0.5) * 0.15; drift *= 0.97; db += drift;
    if (t > hold) db -= 60 * (t - hold) / rel;
    L[j] = Math.max(-80, Math.min(0, db));
  }
  return L;
}
// the inputs for every (frame, component); the GPU and the JavaScript reference share them
function nsPlan(midi, dur, P, sr) {
  const f0 = mtof(midi), hold = Math.max(0.08, dur), rel = Math.max(0.08, Math.min(1.5, P.rel)), atk = Math.max(0.02, P.atk);
  const nF = Math.ceil((hold + rel) * NS.FPS) + 2, n = Math.ceil((hold + rel) * sr), rnd = rngOf(midi * 131 + 7);
  const L = nsLoudness(nF, hold, rel, atk, rnd), layer = (P.vel ?? 1) < 0.75 ? 0 : 1;
  const C = NS.K + NS.NB, X = new Float32Array(nF * C * NS.ST);
  const bandC = k => Math.sqrt(NS_BANDS[k] * NS_BANDS[k + 1]);
  for (let j = 0; j < nF; j++) for (let c = 0; c < C; c++) {
    const noise = c >= NS.K, fq = noise ? bandC(c - NS.K) : (c + 1) * f0, q = (j * C + c) * NS.ST, lf = Math.log2(fq / 1000) / 3;
    X.set([lf, noise ? 1 : -1, (midi - 66) / 22, layer * 2 - 1, Math.max(-60, L[j]) / 30 + 1, Math.min(j / NS.FPS, 0.4) / 0.4 * 2 - 1], q);
    // sines of the log-frequency at four scales, so the network can draw sharp formants
    for (let m = 0; m < 4; m++) { X[q + 6 + m * 2] = Math.sin(Math.PI * 2 ** m * lf); X[q + 7 + m * 2] = Math.cos(Math.PI * 2 ** m * lf); }
    X[q + 14] = Math.pow(10, L[j] / 20) * (noise || fq < Math.min(16000, sr * 0.45) ? 1 : 0);
  }
  // breath noise as clusters of random sines inside each band; level matched to the analysis (per-bin magnitude)
  const ns = new Float32Array(NS.NB * NS.NOISE * 4);
  for (let b = 0; b < NS.NB; b++) {
    const lo = NS_BANDS[b], hi = Math.min(NS_BANDS[b + 1], sr * 0.45), bins = Math.max(1, (hi - lo) / (48000 / 2048));
    for (let s = 0; s < NS.NOISE; s++) { const f = lo + (hi - lo) * rnd(); ns.set([f / sr, (f * 4096 / sr) % 1, rnd(), Math.sqrt(bins / NS.NOISE)], (b * NS.NOISE + s) * 4); }
  }
  const vib = { rate: 5.1 + rnd() * 0.4, depth: (10 + 30 * (P.vib || 0)) / 1200, start: 0.32 };
  return { f0, n, nF, C, X, ns, vib, sr };
}
function nsEvalJS(X, nF, C) {
  const w = nsWeights(), H = NS.H, IN = NS.IN, W1 = 0, b1 = IN * H, W2 = b1 + H, b2 = W2 + H * H, W3 = b2 + H, b3 = W3 + H;
  const A = new Float32Array(nF * C), h1 = new Float32Array(H), h2 = new Float32Array(H);
  for (let r = 0; r < nF * C; r++) {
    const q = r * NS.ST;
    if (!X[q + 14]) continue;
    for (let o = 0; o < H; o++) { let s = w[b1 + o]; for (let i = 0; i < IN; i++) s += X[q + i] * w[W1 + i * H + o]; h1[o] = Math.tanh(s); }
    for (let o = 0; o < H; o++) { let s = w[b2 + o]; for (let i = 0; i < H; i++) s += h1[i] * w[W2 + i * H + o]; h2[o] = Math.tanh(s); }
    let y = w[b3]; for (let i = 0; i < H; i++) y += h2[i] * w[W3 + i];
    A[r] = Math.exp(Math.max(-11.5, Math.min(0, y))) * X[q + 14];
  }
  return A;
}
const NSAX_WGSL = `
struct NP { nF: u32, C: u32, n: u32, off: u32, f0: f32, sr: f32, fps: f32, vrate: f32, vdepth: f32, vstart: f32, gain: f32, cpb: f32 };
@group(0) @binding(0) var<uniform> np: NP;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<f32>;
@group(0) @binding(3) var<storage, read_write> A: array<f32>;
@group(0) @binding(4) var<storage, read> NZ: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> outb: array<f32>;
const HN = 64u;
// one thread per (frame, component): the network
@compute @workgroup_size(64)
fn net(@builtin(global_invocation_id) gid: vec3u) {
  let r = gid.x;
  if (r >= np.nF * np.C) { return; }
  let q = r * 16u;
  if (X[q + 14u] == 0.0) { A[r] = 0.0; return; }
  var h1: array<f32, 64>;
  let b1 = 14u * HN; let W2 = b1 + HN; let b2 = W2 + HN * HN; let W3 = b2 + HN; let b3 = W3 + HN;
  for (var o = 0u; o < HN; o++) {
    var s = W[b1 + o];
    for (var i = 0u; i < 14u; i++) { s += X[q + i] * W[i * HN + o]; }
    h1[o] = tanh(s);
  }
  var y = W[b3];
  for (var o = 0u; o < HN; o++) {
    var s = W[b2 + o];
    for (var i = 0u; i < HN; i++) { s += h1[i] * W[W2 + i * HN + o]; }
    y += tanh(s) * W[W3 + o];
  }
  A[r] = exp(clamp(y, -11.5, 0.0)) * X[q + 14u];
}
// one thread per sample: 48 harmonics with vibrato plus the breath noise, levels interpolated between frames
@compute @workgroup_size(256)
fn synth(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x + np.off;
  if (i >= np.n) { return; }
  let t = f32(i) / np.sr;
  let fp = t * np.fps; let j = min(u32(fp), np.nF - 2u); let w = fp - f32(j);
  let ramp = clamp((t - np.vstart) / 0.4, 0.0, 1.0);
  let wv = 6.2831853 * np.vrate;
  // phase in cycles, kept precise over long notes by splitting the sample index into 4096-sample blocks
  let hi = f32(i >> 12u); let lo = f32(i & 4095u);
  let fps = np.f0 / np.sr;
  let vph = np.f0 * np.vdepth * ramp * (1.0 - cos(wv * max(0.0, t - np.vstart))) / wv;
  var s = 0.0;
  for (var k = 0u; k < 48u; k++) {
    let a = mix(A[j * np.C + k], A[(j + 1u) * np.C + k], w);
    let kf = f32(k + 1u);
    if (a > 0.0) { s += a * sin(6.2831853 * fract(fract(fract(kf * np.cpb) * hi) + kf * fps * lo + kf * vph + f32(k) * 0.137)); }
  }
  for (var b = 0u; b < 16u; b++) {
    let a = mix(A[j * np.C + 48u + b], A[(j + 1u) * np.C + 48u + b], w);
    if (a <= 0.0) { continue; }
    for (var z = 0u; z < 64u; z++) {
      let nz = NZ[b * 64u + z];
      s += a * nz.w * sin(6.2831853 * fract(fract(nz.y * hi) + nz.x * lo + nz.z));
    }
  }
  outb[i] = s * np.gain;
}`;
function nsSynthJS(pl, A, a = 0, b = pl.n) {
  const { f0, sr, nF, C, vib, ns } = pl, out = new Float32Array(b - a), TAU = Math.PI * 2, wv = TAU * vib.rate;
  for (let i = a; i < b; i++) {
    const t = i / sr, fp = t * NS.FPS, j = Math.min(Math.floor(fp), nF - 2), w = fp - j, ramp = Math.max(0, Math.min(1, (t - vib.start) / 0.4));
    const ph = f0 * t + f0 * vib.depth * ramp * (1 - Math.cos(wv * Math.max(0, t - vib.start))) / wv;
    let s = 0;
    for (let k = 0; k < NS.K; k++) { const am = A[j * C + k] + (A[(j + 1) * C + k] - A[j * C + k]) * w; if (am > 0) s += am * Math.sin(TAU * ((ph * (k + 1) + k * 0.137) % 1)); }
    for (let bb = 0; bb < NS.NB; bb++) {
      const am = A[j * C + NS.K + bb] + (A[(j + 1) * C + NS.K + bb] - A[j * C + NS.K + bb]) * w; if (am <= 0) continue;
      for (let z = 0; z < NS.NOISE; z++) { const q = (bb * NS.NOISE + z) * 4; s += am * ns[q + 3] * Math.sin(TAU * ((ns[q] * i + ns[q + 2]) % 1)); }
    }
    out[i - a] = s;
  }
  return out;
}
async function gpuNeural(midi, dur, P, sr) {
  const d = GPU.dev, pl = nsPlan(midi, dur, P, sr), t0 = performance.now();
  if (!GPU.ns) {
    const mod = d.createShaderModule({ code: NSAX_WGSL });
    GPU.ns = { net: d.createComputePipeline({ layout: 'auto', compute: { module: mod, entryPoint: 'net' } }), synth: d.createComputePipeline({ layout: 'auto', compute: { module: mod, entryPoint: 'synth' } }) };
  }
  const mk = (data, usage) => { const b = d.createBuffer({ size: Math.max(16, data.byteLength), usage: usage | GPUBufferUsage.COPY_DST }); d.queue.writeBuffer(b, 0, data); return b; };
  const xb = mk(pl.X, GPUBufferUsage.STORAGE), wb = mk(nsWeights(), GPUBufferUsage.STORAGE), zb = mk(pl.ns, GPUBufferUsage.STORAGE);
  const ab = d.createBuffer({ size: pl.nF * pl.C * 4, usage: GPUBufferUsage.STORAGE });
  const ob = d.createBuffer({ size: pl.n * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const rb = d.createBuffer({ size: pl.n * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const uni = off => { const u = new ArrayBuffer(48), dv = new DataView(u); dv.setUint32(0, pl.nF, true); dv.setUint32(4, pl.C, true); dv.setUint32(8, pl.n, true); dv.setUint32(12, off, true);
    [pl.f0, sr, NS.FPS, pl.vib.rate, pl.vib.depth, pl.vib.start, 1, (pl.f0 * 4096 / sr) % 1].forEach((v, k) => dv.setFloat32(16 + k * 4, v, true)); return mk(new Uint8Array(u), GPUBufferUsage.UNIFORM); };
  const unis = [];
  const run = (pipe, entries, groups) => {
    const bg = d.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: entries.map(([i, b]) => ({ binding: i, resource: { buffer: b } })) });
    const enc = d.createCommandEncoder(), pass = enc.beginComputePass(); pass.setPipeline(pipe); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(groups); pass.end(); d.queue.submit([enc.finish()]);
  };
  { const u = uni(0); unis.push(u); run(GPU.ns.net, [[0, u], [1, xb], [2, wb], [3, ab]], Math.ceil(pl.nF * pl.C / 64)); }
  for (let off = 0; off < pl.n; off += GPU_SLICE) {
    const u = uni(off); unis.push(u);
    run(GPU.ns.synth, [[0, u], [3, ab], [4, zb], [5, ob]], Math.ceil(Math.min(GPU_SLICE, pl.n - off) / 256));
    await d.queue.onSubmittedWorkDone();
  }
  const enc = d.createCommandEncoder(); enc.copyBufferToBuffer(ob, 0, rb, 0, pl.n * 4); d.queue.submit([enc.finish()]);
  await rb.mapAsync(GPUMapMode.READ);
  const out = new Float32Array(rb.getMappedRange().slice(0)); rb.unmap();
  for (const b of [xb, wb, zb, ab, ob, rb, ...unis]) b.destroy();
  GPU.renders++; GPU.lastMs = performance.now() - t0;
  return { out, n: pl.n };
}
function gpuRequestNeural(key, midi, dur, P) {
  if (GPU.cache.has(key) || GPU.jobs.has(key)) return key;
  const job = GPU.chain = GPU.chain.then(async () => {
    if (GPU.state !== 'ok') return;
    const sr = S.ctx.sampleRate, { out, n } = await gpuNeural(midi, dur, P, sr);
    const buf = S.ctx.createBuffer(2, n, sr);
    for (let i = 0; i < n; i++) out[i] *= 0.2;
    buf.copyToChannel(out, 0); buf.copyToChannel(out, 1);
    gpuCachePut(key, buf);
  }).catch(e => { GPU.err = String(e.message || e); }).finally(() => GPU.jobs.delete(key));
  GPU.jobs.set(key, job);
  return key;
}
const gpuKey = (inst, midi, dur, P) => [inst, midi, inst === 'nsax' ? ((P.vel ?? 1) < 0.75 ? 0 : 1) + '|' + (P.vib || 0).toFixed(2) : '', inst === 'plate' || inst === 'membrane' ? 0 : Math.round(dur * 8) / 8, (P.bright ?? 0).toFixed(2), (P.detune ?? 0).toFixed(0), (P.dec ?? 0).toFixed(2), (P.width ?? 0).toFixed(2), (P.atk ?? 0).toFixed(2), (P.rel ?? 0).toFixed(2)].join('|');
function gpuCachePut(key, buf) {
  GPU.cache.set(key, buf); GPU.bytes += buf.length * 8;
  while (GPU.bytes > GPU_CACHE_BYTES && GPU.cache.size > 1) { const [k0, b0] = GPU.cache.entries().next().value; GPU.cache.delete(k0); GPU.bytes -= b0.length * 8; }
}
function gpuRequest(inst, midi, dur, P) {
  const key = gpuKey(inst, midi, dur, P);
  if (inst === 'membrane') return gpuRequestMembrane(key, midi, P);
  if (inst === 'nsax') return gpuRequestNeural(key, midi, dur, P);
  if (GPU.cache.has(key) || GPU.jobs.has(key)) return key;
  const job = GPU.chain = GPU.chain.then(async () => {
    if (GPU.state !== 'ok') return;
    const sr = S.ctx.sampleRate, v = gpuVoice(inst, midi, dur, P, sr);
    const pcm = await gpuRender(v.parts, v.np, v.n, v.o);
    const buf = S.ctx.createBuffer(2, v.n, sr);
    buf.copyToChannel(pcm.subarray(0, v.n), 0); buf.copyToChannel(pcm.subarray(v.n), 1);
    gpuCachePut(key, buf);
  }).catch(e => { GPU.err = String(e.message || e); }).finally(() => GPU.jobs.delete(key));
  GPU.jobs.set(key, job);
  return key;
}
// one render gives all four velocity layers; they are cached as key|0 .. key|3
function gpuRequestMembrane(key, midi, P) {
  if (GPU.cache.has(key + '|2') || GPU.jobs.has(key)) return key;
  const job = GPU.chain = GPU.chain.then(async () => {
    if (GPU.state !== 'ok') return;
    const sr = S.ctx.sampleRate, { out, n } = await gpuMembrane(midi, P, sr);
    let pk = 0; for (let i = 0; i < out.length; i++) pk = Math.max(pk, Math.abs(out[i]));
    const g = 0.6 / Math.max(1e-9, pk);
    for (let L = 0; L < MEM_LAYERS; L++) {
      const buf = S.ctx.createBuffer(2, n, sr), o = L * n * 2;
      const a = out.subarray(o, o + n), b = out.subarray(o + n, o + 2 * n);
      for (let i = 0; i < n; i++) { a[i] *= g; b[i] *= g; }
      buf.copyToChannel(a, 0); buf.copyToChannel(b, 1);
      gpuCachePut(key + '|' + L, buf);
    }
  }).catch(e => { GPU.err = String(e.message || e); }).finally(() => GPU.jobs.delete(key));
  GPU.jobs.set(key, job);
  return key;
}
// play GPU-rendered notes; returns the notes it could not play yet (still rendering, or no GPU)
function gpuPlay(def, notes, time, dur, P, vel, r) {
  if (GPU.state !== 'ok') return notes;
  const miss = [];
  for (const m of notes) {
    let key = gpuRequest(def.inst, m, dur, def.inst === 'nsax' ? { ...P, vel } : P), gv = vel;
    // the drum head was rendered at four strengths: take the nearest, and only trim the level around it
    if (def.inst === 'membrane') { const L = vel < 0.55 ? 0 : vel < 0.8 ? 1 : vel < 1.05 ? 2 : 3; key += '|' + L; gv = 0.85 + 0.15 * vel; }
    const buf = GPU.cache.get(key);
    if (!buf) { miss.push(m); continue; }
    GPU.cache.delete(key); GPU.cache.set(key, buf);   // keep recently used notes
    const src = S.ctx.createBufferSource(), g = S.ctx.createGain();
    src.buffer = buf; g.gain.value = P.gain * gv * (INST[def.inst].lvl || 0.5);
    src.connect(g).connect(r.input);
    src.start(time); src.onended = () => { try { g.disconnect(); } catch (e) { } };
    registerVoice(r, { filters: [], kt: 1, tEnd: time + buf.duration });
  }
  return miss;
}
// render the notes a GPU track will play over the next bars, so the first hits are ready in time
function gpuPrewarm() {
  if (GPU.state !== 'ok' || !S.ctx) return;
  const c0 = Math.floor(Math.max(0, S.playing ? S.lastC : 0));
  for (const name of S.order) {
    const def = S.tracks.get(name);
    if (!def || !GPU_INSTS[def.inst]) continue;
    for (let n = c0; n < c0 + 4; n++) for (const ev of cycleEvents(def, n)) {
      const P = evalParams(def, ev.s);
      const notes = /^r'*$/.test(ev.v) ? null : atomNotes(ev.v, S.root, S.scale, Math.round(P.oct), def.fixed ? 0 : progAt(ev.s), (ev.tr || 0) + Math.round(P.tr || 0), ev.inv);
      if (notes) for (const m of notes) gpuRequest(def.inst, m, ev.d / S.cps * P.len, P);
    }
  }
}

// ---------- modal reverb ----------
// A space is rendered as the sum of its resonant modes, each a decaying sine (the approach studied for
// GPUs and FPGAs in Frontiers in Signal Processing, 2025). Low room modes are exact for the room's size,
// so a small room booms where a real one would; higher up the modes are placed at random with just enough
// density that neighbours overlap, and every mode decays at its own frequency-dependent rate. Each mode
// gets independent left and right gains, so the tail is wide. Up to 65,536 modes and several seconds:
// minutes of work in JavaScript, well under a second on a GPU. The result loads into a ConvolverNode.
const SPACES = {
  room:      { dims: [6.4, 4.8, 3.0], t60: 0.8, air: 6000, pre: 0.004 },
  chamber:   { dims: [11, 7.6, 5.2], t60: 1.5, air: 7000, pre: 0.01 },
  hall:      { dims: [32, 21, 14], t60: 2.5, air: 6500, pre: 0.022 },
  cathedral: { dims: [72, 34, 30], t60: 5.5, air: 5000, pre: 0.04 },
  plate:     { plate: true, t60: 3.4, air: 10000, pre: 0.001 },
};
const SPACE_MAXMODES = 65536;
function modalModes(sp, sr) {
  const c = 343, rnd = rngOf(sp.t60 * 1000 | 0), fTop = Math.min(15000, sr * 0.45);
  const T = f => sp.t60 * (1 + 0.35 * Math.exp(-f / 180)) / (1 + Math.pow(f / sp.air, 1.6));
  const need = f => 3 * T(f) / 2.2;             // modes per Hz for ~3 inside each mode's bandwidth
  const modes = [];
  let fEx = 30;
  if (sp.dims) {
    // exact room modes (nx, ny, nz) below fEx
    const [X, Y, Z] = sp.dims, V = X * Y * Z;
    const phys = f => 4 * Math.PI * V * f * f / (c * c * c);
    while (fEx < 2000 && phys(fEx) < need(fEx)) fEx += 5;
    for (let a = 0; a * c / (2 * X) < fEx; a++) for (let b = 0; b * c / (2 * Y) < fEx; b++) for (let d = 0; d * c / (2 * Z) < fEx; d++) {
      if (!a && !b && !d) continue;
      const f = c / 2 * Math.hypot(a / X, b / Y, d / Z);
      if (f >= 30 && f < fEx && modes.length < 8000) modes.push(f);
    }
  }
  // above: jittered spacing at the needed density, thinned if the total would pass the cap
  let est = 0; for (let f = fEx; f < fTop; f += 10) est += need(f) * 10 * (sp.plate ? 1.4 : 1);
  const thin = Math.min(1, (SPACE_MAXMODES - modes.length) / Math.max(1, est));
  for (let f = fEx; f < fTop && modes.length < SPACE_MAXMODES;) { modes.push(f); f += (0.3 + rnd() * 1.4) / (need(f) * (sp.plate ? 1.4 : 1) * thin); }
  return { modes, T, rnd };
}
function gauss(rnd) { return Math.sqrt(-2 * Math.log(Math.max(1e-9, rnd()))) * Math.cos(2 * Math.PI * rnd()); }
async function modalIR(name, sr) {
  const sp = SPACES[name], { modes, T, rnd } = modalModes(sp, sr);
  const np = modes.length, parts = new Float32Array(np * 8);
  let s2 = 0, tMax = 0;
  for (let k = 0; k < np; k++) {
    const f = modes[k], t60 = T(f), fn = k + 1 < np ? modes[k + 1] - f : 1;
    // energy per Hz stays even: each mode's power follows the spacing to its neighbour; very low end rolled off
    const amp = Math.sqrt(Math.max(0.02, fn)) / (1 + Math.pow(45 / f, 4)), q = k * 8;
    partial(parts, k, f, sr, amp, rnd(), t60, 0);
    parts[q + 5] = gauss(rnd); parts[q + 6] = gauss(rnd);
    s2 += amp * amp; tMax = Math.max(tMax, t60);
  }
  const n = Math.round(Math.min(9, tMax * 1.05) * sr);
  const pcm = await gpuRender(parts, np, n, { atk: 0.002 * sr, gain: 0.5 / Math.sqrt(s2) });
  // early reflections: first and second-order image sources of the box (none for the plate)
  if (sp.dims) {
    const [X, Y, Z] = sp.dims, src = [X * 0.31, Y * 0.42, 1.4], mic = [X * 0.62, Y * 0.55, 1.6];
    let e2 = 0; for (let i = 0; i < Math.min(n, sr * 0.05); i++) e2 += pcm[i] * pcm[i];
    const lvl = Math.sqrt(e2 / (sr * 0.05)) * 6;
    for (let a = -2; a <= 2; a++) for (let b = -2; b <= 2; b++) for (let d = -1; d <= 1; d++) {
      const ord = Math.abs(a) + Math.abs(b) + Math.abs(d);
      if (!ord || ord > 2) continue;
      const img = [a * X + (a & 1 ? X - src[0] : src[0]), b * Y + (b & 1 ? Y - src[1] : src[1]), d * Z + (d & 1 ? Z - src[2] : src[2])];
      const dist = Math.hypot(img[0] - mic[0], img[1] - mic[1], img[2] - mic[2]), direct = Math.hypot(src[0] - mic[0], src[1] - mic[1], src[2] - mic[2]);
      const at = Math.round((dist - direct) / 343 * sr), gain = lvl * Math.pow(0.72, ord) * direct / dist * (rnd() < 0.5 ? -1 : 1);
      const side = Math.sign(img[0] - mic[0]) * 0.35;
      for (let j = 0; j < 24 && at + j < n; j++) { const w = gain * Math.exp(-j / 5); pcm[at + j] += w * (1 - side); pcm[n + at + j] += w * (1 + side); }
    }
  }
  const buf = S.ctx.createBuffer(2, n, sr);
  buf.copyToChannel(pcm.subarray(0, n), 0); buf.copyToChannel(pcm.subarray(n), 1);
  return { buf, modes: np, ms: GPU.lastMs };
}

// ---------- spectral freeze ----------
// Hold a pad and the last moment of the mix becomes a sustained chord: the browser's FFT gives the
// spectrum, its strongest peaks become ~2,000 sines with random phases, and the GPU renders 6 seconds
// with a sine window. Copies overlap by half, so their powers always add up to one.
const FRZ = { on: false, srcs: [], timer: null, g: null };
async function freezeStart() {
  if (!A.frz) return;
  const ctx = S.ctx, sr = ctx.sampleRate, L = 6;
  FRZ.on = true;
  const g = FRZ.g = ctx.createGain(); g.gain.value = 0; g.connect(A.bus); g.connect(A.rIn);
  let buf;
  if (GPU.state === 'ok') {
    const N = A.frz.fftSize, db = new Float32Array(N / 2); A.frz.getFloatFrequencyData(db);
    let mx = -Infinity; for (let k = 2; k < db.length; k++) mx = Math.max(mx, db[k]);
    const cand = [];
    for (let k = 3; k < db.length - 2; k++) { const f = k * sr / N; if (f > 16000) break; if (db[k] > mx - 66 && db[k] >= db[k - 1] && db[k] >= db[k + 1]) cand.push(k); }
    cand.sort((a, b) => db[b] - db[a]);
    const use = cand.slice(0, 2048), parts = new Float32Array(Math.max(1, use.length) * 8), rnd = rngOf(Date.now() & 0xffff);
    let s2 = 0;
    use.forEach((k, i) => { const amp = Math.pow(10, db[k] / 20); s2 += amp * amp; partial(parts, i, (k + (rnd() - 0.5) * 0.7) * sr / N, sr, amp, rnd(), 0, (rnd() * 2 - 1) * 0.5); });
    if (!use.length) { FRZ.on = false; return; }
    const n = L * sr, pcm = await gpuRender(parts, use.length, n, { win: true, gain: 0.25 / Math.sqrt(s2 / 2) });
    buf = ctx.createBuffer(2, n, sr); buf.copyToChannel(pcm.subarray(0, n), 0); buf.copyToChannel(pcm.subarray(n), 1);
  } else {
    // without a GPU: loop the last 0.68 s of the mix under the same window
    const N = A.frz.fftSize, td = new Float32Array(N); A.frz.getFloatTimeDomainData(td);
    buf = ctx.createBuffer(2, N, sr);
    for (let ch = 0; ch < 2; ch++) { const d = buf.getChannelData(ch); for (let i = 0; i < N; i++) d[i] = td[i] * Math.sin(Math.PI * i / N) * 0.9; }
  }
  if (!FRZ.on) return;
  const half = buf.duration / 2, now = ctx.currentTime + 0.01;
  g.gain.setValueAtTime(0, now); g.gain.linearRampToValueAtTime(1, now + 0.08);
  const play = (t, off) => { const s = ctx.createBufferSource(); s.buffer = buf; s.connect(g); s.start(t, off); FRZ.srcs.push(s); s.onended = () => { FRZ.srcs = FRZ.srcs.filter(x => x !== s); }; };
  play(now, half); play(now, 0);
  let next = now + half;
  FRZ.timer = setInterval(() => { while (FRZ.on && next < ctx.currentTime + 0.5) { play(next, 0); next += half; } }, 100);
}
function freezeStop() {
  FRZ.on = false; clearInterval(FRZ.timer);
  const g = FRZ.g; if (!g) return;
  const now = S.ctx.currentTime;
  g.gain.cancelScheduledValues(now); g.gain.setValueAtTime(g.gain.value, now); g.gain.linearRampToValueAtTime(0, now + 0.6);
  const srcs = FRZ.srcs; FRZ.srcs = [];
  for (const s of srcs) { try { s.stop(now + 0.65); } catch (e) { } }
  setTimeout(() => { try { g.disconnect(); } catch (e) { } }, 900);
}

// ---------- scenes: GPU backdrops behind the code ----------
// Each scene is one full-screen fragment shader fed by the same inputs: a spectrogram history kept as a
// texture (64 log-spaced bands x 160 rows, about 5 s), the transport position, the kick, and every track's
// code line with the time since it last played. Quiet by design: thin lines in the screen's ink colour,
// with the track colours only where something just happened.
//   terrain  ridgelines of the recent spectrum in perspective, nearest at the bottom
//   contour  a topographic map: spectrogram + slow drift + ripples from track hits, index line every fifth
//   dots     an LED matrix: hits send rings out from their code line, the spectrum lights columns
//   demo     a demoscene tunnel with copper bars and a sine scroller
//   sono     the whole song as one sonogram, filled in by song position as it plays
// and any `shader name <WGSL expression>` line in the code becomes a scene too (see sceneCustomWGSL).
const SCENE_BANDS = 64, SCENE_ROWS = 160, SCENE_MAXE = 24, SCENE_MODES = ['terrain', 'contour', 'dots', 'demo', 'sono'];
const SCENE_PRE = `
struct U {
  res: vec2f, time: f32, beat: f32,
  bass: f32, mid: f32, high: f32, kick: f32,
  ne: u32, head: f32, rows: f32, dpr: f32,
  ink: vec4f, accent: vec4f,
  em: array<vec4f, 24>,
  pal: array<vec4f, 10>,
  sono: vec4f,
  cuts: array<vec4f, 4>,
};
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var specT: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var txt: texture_2d<f32>;
@group(0) @binding(4) var sonoT: texture_2d<f32>;
struct VO { @builtin(position) p: vec4f, @location(0) uv: vec2f };
@vertex fn vs(@builtin(vertex_index) i: u32) -> VO {
  var o: VO;
  let x = f32((i << 1u) & 2u); let y = f32(i & 2u);
  o.p = vec4f(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0); o.uv = vec2f(x, y);
  return o;
}
// band in [0,1] (low to high), age in [0,1] (now to ~5 s ago)
fn specAt(band: f32, age: f32) -> f32 {
  let v = (u.head - clamp(age, 0.0, 1.0) * (u.rows - 2.0) + 0.5) / u.rows;
  return textureSampleLevel(specT, samp, vec2f(clamp(band, 0.0, 1.0), v), 0.0).r;
}
fn premul(c: vec3f, a: f32) -> vec4f { return vec4f(c * a, a); }
fn rot(a: f32) -> mat2x2f { let c = cos(a); let s = sin(a); return mat2x2f(c, s, -s, c); }
// helpers for shader lines written in the code
fn spec(band: f32, age: f32) -> f32 { return specAt(band, age); }
fn hash(p: vec2f) -> f32 { return fract(sin(dot(p, vec2f(127.1, 311.7))) * 43758.5453); }
fn noise(p: vec2f) -> f32 {
  let i = floor(p); let f = fract(p); let w = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash(i), hash(i + vec2f(1.0, 0.0)), w.x), mix(hash(i + vec2f(0.0, 1.0)), hash(i + vec2f(1.0, 1.0)), w.x), w.y);
}
fn fbm(p: vec2f) -> f32 { var v = 0.0; var a = 0.5; var q = p; for (var k = 0; k < 5; k++) { v += a * noise(q); q = q * 2.03 + 1.7; a *= 0.5; } return v; }
fn pal(k: i32) -> vec3f { return u.pal[u32(k) % 10u].rgb; }
// how recently track k (in code order) played: 1 on the hit, fading over about a second
fn hit(k: i32) -> f32 { if (u32(k) >= u.ne) { return 0.0; } return exp(-u.em[u32(k)].w * 3.0); }
`;
const SCENE_WGSL = SCENE_PRE + `

// one ridgeline: row k of R, at screen x
fn ridge(x: f32, kf: f32, R: f32) -> f32 {
  let z = 1.0 + kf * 0.21;
  let wx = (x - 0.5) * z * 1.1;
  let m = abs(wx) * 2.0;
  var h = 0.0;
  if (m < 1.0) {
    let e = (specAt(m * 0.85 - 0.012, kf / R) + specAt(m * 0.85, kf / R) * 2.0 + specAt(m * 0.85 + 0.012, kf / R)) * 0.25;
    h = e * e * (1.0 - m * m * m * m);
  }
  h += 0.012 * sin(wx * 23.0 + kf * 1.3 + u.time * 0.25) * (1.0 - min(m, 1.0));
  return 0.36 + 0.6 / z - h * 0.45 / z;
}
@fragment fn terrain(i: VO) -> @location(0) vec4f {
  let uv = i.uv;
  let R = 46.0;
  let px = 1.0 / u.res.x;
  for (var k = 0; k < 46; k++) {
    let kf = f32(k);
    let y = ridge(uv.x, kf, R);
    let d = uv.y - y;
    // distance to the line in pixels, measured across the slope so steep flanks stay unbroken
    let slope = (ridge(uv.x + px, kf, R) - y) * u.res.y;
    let dpx = abs(d) * u.res.y / sqrt(1.0 + slope * slope);
    let w = u.dpr * (0.55 + 0.35 / (1.0 + kf * 0.21));
    if (dpx < w + 1.0) {
      let fade = pow(1.0 - kf / R, 1.4);
      let a = (1.0 - smoothstep(w - 0.5, w + 1.0, dpx)) * (0.1 + 0.42 * fade);
      let c = mix(u.ink.rgb, u.accent.rgb, u.kick * 0.7 * max(0.0, 1.0 - kf / 4.0));
      return premul(c, a);
    }
    if (d > 0.0) { break; }
  }
  return vec4f(0.0);
}

fn height(p: vec2f, asp: f32) -> f32 {
  // the spectrogram, blurred across time and pitch so it bends the map instead of striping it
  let bx = pow(p.x, 1.3); let ay = 1.0 - p.y;
  var sp = 0.0;
  for (var j = -2; j <= 2; j++) { for (var q = -1; q <= 1; q++) { sp += specAt(bx + f32(q) * 0.05, ay + f32(j) * 0.06); } }
  var h = sp / 15.0 * 0.5;
  h += 0.32 * sin(p.x * 4.3 + u.time * 0.045) * cos(p.y * 3.7 - u.time * 0.035);
  h += 0.18 * sin((p.x * 1.3 + p.y) * 7.1 + u.time * 0.06) + 0.1 * cos(p.x * 11.0 - p.y * 5.0 - u.time * 0.05);
  for (var k = 0u; k < u.ne; k++) {
    let e = u.em[k];
    let d = distance(vec2f(p.x * asp, p.y), vec2f(e.x * asp, e.y));
    let age = e.w;
    h += 0.5 * exp(-age * 1.1) * exp(-pow((d - age * 0.3) * 9.0, 2.0));
  }
  return h;
}
@fragment fn contour(i: VO) -> @location(0) vec4f {
  let asp = u.res.x / u.res.y;
  let h = height(i.uv, asp) * 7.0;
  let w = max(fwidth(h), 1e-4);
  let f = abs(fract(h + 0.5) - 0.5) / (w * u.dpr);
  var a = (1.0 - min(f, 1.0)) * 0.2;
  let h5 = h / 5.0;
  let f5 = abs(fract(h5 + 0.5) - 0.5) / (max(fwidth(h5), 1e-4) * u.dpr * 1.4);
  a = max(a, (1.0 - min(f5, 1.0)) * 0.42);
  return premul(u.ink.rgb, a);
}

@fragment fn dots(i: VO) -> @location(0) vec4f {
  let asp = u.res.x / u.res.y;
  let cell = 15.0 * u.dpr;
  let px = i.uv * u.res;
  let c = (floor(px / cell) + 0.5) * cell;
  let d = length(px - c);
  let q = c / u.res;
  var b = 0.085;
  var col = u.ink.rgb;
  let lv = specAt(q.x, 0.0);
  let top = lv * 0.55;
  let fromBottom = 1.0 - q.y;
  if (fromBottom < top) { b += 0.34 * (1.0 - fromBottom / max(top, 0.001) * 0.6); }
  // the peak cell of each column picks up the accent, like an LED meter
  let ch = cell / u.res.y;
  if (top > ch && abs(fromBottom - top) < ch * 0.5) { b += 0.3; col = mix(col, u.accent.rgb, 0.7); }
  for (var k = 0u; k < u.ne; k++) {
    let e = u.em[k];
    let dd = distance(vec2f(q.x * asp, q.y), vec2f(e.x * asp, e.y));
    let ring = exp(-pow((dd - e.w * 0.45) * 11.0, 2.0)) * exp(-e.w * 1.5);
    if (ring > 0.02) { col = mix(col, u.pal[u32(e.z) % 10u].rgb, min(1.0, ring * 1.5)); }
    b += ring * 1.0;
  }
  let r = 1.6 * u.dpr * (0.85 + min(b, 1.0) * 0.5);
  let a = (1.0 - smoothstep(r - 0.8 * u.dpr, r + 0.4 * u.dpr, d)) * min(b, 0.9);
  return premul(col, a);
}

fn tunnel(p: vec3f) -> f32 {
  let r = 1.5 + 0.18 * sin(p.z * 0.7 + u.time * 0.6) + u.kick * 0.12;
  var q = p;
  let t = rot(p.z * 0.12 + u.time * 0.1) * q.xy; q = vec3f(t, q.z);
  let ang = atan2(q.y, q.x);
  let wall = r - length(q.xy) + 0.05 * sin(ang * 8.0 + p.z * 1.5);
  let ringz = (fract(p.z * 0.25) - 0.5) * 4.0;
  let ring = length(vec2f(length(q.xy) - r * 0.92, ringz)) - 0.05;
  return min(wall, ring);
}
@fragment fn demo(i: VO) -> @location(0) vec4f {
  let asp = u.res.x / u.res.y;
  let p = (i.uv * 2.0 - 1.0) * vec2f(asp, -1.0);
  let z0 = u.time * 1.6 + u.beat * 0.8;
  let ro = vec3f(0.0, 0.0, z0);
  var rd = normalize(vec3f(p, 1.7));
  let rr = rot(sin(u.time * 0.21) * 0.35) * rd.xy; rd = vec3f(rr, rd.z);
  var dist = 0.0; var glow = 0.0; var hit = false;
  for (var k = 0; k < 64; k++) {
    let d = tunnel(ro + rd * dist);
    glow += 0.012 / (0.02 + abs(d));
    if (d < 0.002) { hit = true; break; }
    dist += d * 0.8;
    if (dist > 28.0) { break; }
  }
  let fog = exp(-dist * 0.11);
  let pos = ro + rd * dist;
  let band = 0.5 + 0.5 * sin(pos.z * 0.6 - u.time * 1.2);
  var c = mix(u.pal[1].rgb, u.pal[3].rgb, band) * 0.35 * fog;
  if (hit) { c += u.accent.rgb * pow(fog, 2.0) * (0.25 + u.kick * 0.6) * step(0.75, fract(pos.z * 0.25 + 0.6)); }
  c += u.accent.rgb * glow * 0.012 * (0.4 + u.bass);
  // copper bars
  let cy = i.uv.y;
  for (var k = 0; k < 3; k++) {
    let yc = 0.14 + 0.06 * sin(u.time * 1.3 + f32(k) * 1.9);
    let dd = abs(cy - yc - f32(k) * 0.012);
    c += mix(u.pal[0].rgb, u.pal[4].rgb, f32(k) / 2.0) * exp(-dd * dd * 9000.0) * 0.35;
  }
  // sine scroller along the bottom
  let tsz = vec2f(textureDimensions(txt));
  let sh = 0.11;
  let base = 0.8;
  let sx = i.uv.x * asp / sh * (tsz.y / tsz.x);
  let off = fract(sx + u.time * 0.045);
  let wav = sin(i.uv.x * 9.0 + u.time * 3.0) * 0.025;
  let ty = (i.uv.y - base - wav) / sh + 0.5;
  if (ty > 0.0 && ty < 1.0) {
    let g = textureSampleLevel(txt, samp, vec2f(off, ty), 0.0).a;
    let grad = mix(u.pal[0].rgb, u.pal[1].rgb, ty);
    c = mix(c, grad, g);
  }
  return vec4f(c, 1.0);
}

// the whole song: x is the song position, y the frequency; filled in as the song plays
@fragment fn sono(i: VO) -> @location(0) vec4f {
  let x = i.uv.x;
  let fy = clamp((1.0 - i.uv.y - 0.04) / 0.92, 0.0, 1.0);
  let v = textureSampleLevel(sonoT, samp, vec2f(fy, x), 0.0).r;
  var a = pow(v, 1.8) * 0.45;
  var c = mix(u.ink.rgb, u.accent.rgb, smoothstep(0.7, 0.98, v) * 0.7);
  let bx = x * max(1.0, u.sono.y);
  let bl = 1.0 - smoothstep(0.0, fwidth(bx) * 1.2, abs(fract(bx + 0.5) - 0.5));
  a = max(a, bl * 0.05);
  for (var k = 0u; k < 16u; k++) {
    let cx = u.cuts[k / 4u][k % 4u];
    if (cx >= 0.0 && abs(x - cx) * u.res.x < 1.0 * u.dpr) { a = max(a, 0.28); c = u.ink.rgb; }
  }
  if (u.sono.z > 0.5 && abs(x - u.sono.x) * u.res.x < 1.5 * u.dpr) { return premul(u.accent.rgb, 0.85); }
  return premul(c, a);
}`;
// a shader line from the code, wrapped as a full-screen scene. p: 0..1 with y up; q: centred, square units.
function sceneCustomWGSL(expr, kind) {
  const out = kind === 'f32' ? 'premul(u.ink.rgb, clamp(v, 0.0, 1.0) * 0.55)' : 'vec4f(clamp(v, vec3f(0.0), vec3f(1.0)) * 0.6, clamp(max(v.x, max(v.y, v.z)), 0.0, 1.0) * 0.6)';
  return SCENE_PRE + `
@fragment fn custom(i: VO) -> @location(0) vec4f {
  let asp = u.res.x / u.res.y;
  let p = vec2f(i.uv.x, 1.0 - i.uv.y);
  let q = (p - 0.5) * vec2f(asp, 1.0);
  let t = u.time; let beat = u.beat; let bar = u.beat / 4.0;
  let bass = u.bass; let mid = u.mid; let high = u.high; let kick = u.kick;
  let ink = u.ink.rgb; let accent = u.accent.rgb;
  let v: ${kind} = ${expr};
  return ${out};
}`;
}
const SCENE = { ready: false, failed: false, loading: false, hits: new Map(), bins: null, rowT: 0, head: 0, custom: new Map(), sonoKey: '', sonoLast: -1 };
const SONO_ROWS = 2048, SCENE_UBYTES = 704;
// own device, as with the audio renders: a canvas problem must never take the sound with it.
// offscreen: draw into a texture (tests and headless browsers, where presenting a WebGPU canvas fails)
async function sceneInit(canvas, offscreen) {
  if (SCENE.ready || SCENE.failed || SCENE.loading) return;
  SCENE.loading = true;
  try {
    if (!navigator.gpu) throw new Error('WebGPU is not available in this browser');
    const ad = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!ad) throw new Error('no GPU adapter');
    const d = await ad.requestDevice();
    SCENE.adapter = ad; SCENE.dev = d;
    d.lost.then(i => { SCENE.ready = false; SCENE.failed = true; SCENE.err = i.message; });
    let ctx = null, fmt = 'rgba8unorm';
    if (offscreen) SCENE.offscreen = { w: offscreen.w, h: offscreen.h, tex: d.createTexture({ size: [offscreen.w, offscreen.h], format: fmt, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC }) };
    else { ctx = canvas.getContext('webgpu'); fmt = navigator.gpu.getPreferredCanvasFormat(); ctx.configure({ device: d, format: fmt, alphaMode: 'premultiplied' }); }
    const mod = d.createShaderModule({ code: SCENE_WGSL });
    const info = await mod.getCompilationInfo(); const bad = info.messages.find(m => m.type === 'error');
    if (bad) throw new Error('shader: ' + bad.message + ' at line ' + bad.lineNum);
    const uni = d.createBuffer({ size: SCENE_UBYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const tex = (w, h) => d.createTexture({ size: [w, h], format: 'r8unorm', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    const spec = tex(SCENE_BANDS, SCENE_ROWS), sono = tex(SCENE_BANDS, SONO_ROWS);
    const samp = d.createSampler({ magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'repeat', addressModeW: 'repeat' });
    // the scroller text, drawn once with the page's own font
    const tc = document.createElement('canvas'); tc.width = 4096; tc.height = 128;
    const g = tc.getContext('2d'); g.font = '700 96px "Instrument Sans", system-ui, sans-serif'; g.textBaseline = 'middle'; g.fillStyle = '#fff';
    const msg = 'TACTUS  •  LIVE CODING AT THE SPEED OF FEEL  •  RUST DSP  •  WEBGPU  •  GREETINGS TO STRUDEL, TIDALCYCLES, SONIC PI, ORCA AND EVERY ALGORAVE  •  ';
    let fs = 96; while (g.measureText(msg).width > 4000 && fs > 40) { fs -= 4; g.font = `700 ${fs}px "Instrument Sans", system-ui, sans-serif`; }
    g.fillText(msg, 8, 64);
    const txt = d.createTexture({ size: [tc.width, tc.height], format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT });
    d.queue.copyExternalImageToTexture({ source: tc }, { texture: txt }, [tc.width, tc.height]);
    // one explicit layout for every scene, so the code's own shaders can use any input
    const F = GPUShaderStage.FRAGMENT, V = GPUShaderStage.VERTEX;
    const bgl = d.createBindGroupLayout({ entries: [
      { binding: 0, visibility: F | V, buffer: { type: 'uniform' } },
      { binding: 1, visibility: F, texture: {} }, { binding: 2, visibility: F, sampler: {} },
      { binding: 3, visibility: F, texture: {} }, { binding: 4, visibility: F, texture: {} }] });
    const layout = d.createPipelineLayout({ bindGroupLayouts: [bgl] });
    const blend = { color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' }, alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' } };
    const pipe = (module, entryPoint) => d.createRenderPipeline({ layout, vertex: { module, entryPoint: 'vs' }, fragment: { module, entryPoint, targets: [{ format: fmt, blend }] }, primitive: { topology: 'triangle-list' } });
    const pipes = {};
    for (const m of SCENE_MODES) pipes[m] = pipe(mod, m);
    const bg = d.createBindGroup({ layout: bgl, entries: [{ binding: 0, resource: { buffer: uni } }, { binding: 1, resource: spec.createView() }, { binding: 2, resource: samp }, { binding: 3, resource: txt.createView() }, { binding: 4, resource: sono.createView() }] });
    Object.assign(SCENE, { ctx, fmt, canvas, uni, spec, sono, pipes, bg, pipe, ready: true });
  } catch (e) { SCENE.failed = true; SCENE.err = String(e.message || e); }
  SCENE.loading = false;
}
// a shader line from the code: compile it (as a number, then as a colour) and report the first error
async function sceneCustom(name, expr) {
  const have = SCENE.custom.get(name);
  if (have && have.expr === expr) return have;
  const ent = { expr, pipe: null, err: null, pending: true };
  SCENE.custom.set(name, ent);
  if (!SCENE.ready) { ent.pending = false; ent.err = SCENE.err || 'WebGPU'; return ent; }
  const d = SCENE.dev; let first = null;
  for (const kind of ['f32', 'vec3f']) {
    d.pushErrorScope('validation');
    const mod = d.createShaderModule({ code: sceneCustomWGSL(expr, kind) });
    const info = await mod.getCompilationInfo();
    const bad = info.messages.find(m => m.type === 'error');
    if (bad) { await d.popErrorScope(); first = first || bad.message; continue; }
    const p = SCENE.pipe(mod, 'custom');
    const ve = await d.popErrorScope();
    if (ve) { first = first || ve.message; continue; }
    ent.pipe = p; break;
  }
  if (SCENE.custom.get(name) !== ent) return ent;   // replaced while compiling
  ent.err = ent.pipe ? null : String(first || 'shader error').split('\n')[0].slice(0, 160);
  ent.pending = false;
  return ent;
}
// '#rrggbb' → [r, g, b] in 0..1
function sceneRgb(hex, fb) { const m = /^#?([0-9a-f]{6})$/i.exec(String(hex || '').trim()); if (!m) return fb; const v = parseInt(m[1], 16); return [(v >> 16 & 255) / 255, (v >> 8 & 255) / 255, (v & 255) / 255]; }
// a track played on a code line
function sceneHit(line) { SCENE.hits.set(line, performance.now() / 1000); }
let sceneKick = 0;
// opts.ink / opts.accent: [r, g, b] in 0..1, so lines follow the page theme
// opts.sono: { pos, bars, cuts, key, live } — where the song is, for the whole-song sonogram
function sceneFrame(mode, emitters, beat, opts) {
  opts = opts || {};
  // mode null: only keep the spectrum and the sonogram up to date, draw nothing
  if (!SCENE.ready) return false;
  const custom = mode && SCENE.custom.get(mode), pipe = mode && (SCENE.pipes[mode] || (custom && custom.pipe));
  if (!pipe && mode !== null) return false;
  const d = SCENE.dev, off = SCENE.offscreen, cv = SCENE.canvas, dpr = devicePixelRatio || 1;
  const w = off ? off.w : Math.max(1, Math.round(cv.clientWidth * dpr)), h = off ? off.h : Math.max(1, Math.round(cv.clientHeight * dpr));
  if (!off && (cv.width !== w || cv.height !== h)) { cv.width = w; cv.height = h; }
  const now = performance.now() / 1000, so = opts.sono;
  if (so && so.key !== SCENE.sonoKey) { SCENE.sonoKey = so.key; SCENE.sonoLast = -1; d.queue.writeTexture({ texture: SCENE.sono }, new Uint8Array(SCENE_BANDS * SONO_ROWS), { bytesPerRow: SCENE_BANDS }, [SCENE_BANDS, SONO_ROWS]); }
  // spectrum: 64 log-spaced bands, one new row every 1/30 s
  let bass = 0, mid = 0, high = 0;
  if (now - SCENE.rowT > 1 / 30) {
    SCENE.rowT = now;
    const row = new Uint8Array(SCENE_BANDS);
    if (A.an && S.playing) {
      const an = A.an; SCENE.bins = SCENE.bins || new Uint8Array(an.frequencyBinCount); an.getByteFrequencyData(SCENE.bins);
      const bin = S.ctx.sampleRate / an.fftSize;
      for (let k = 0; k < SCENE_BANDS; k++) {
        const f0 = 30 * Math.pow(16000 / 30, k / SCENE_BANDS), f1 = 30 * Math.pow(16000 / 30, (k + 1) / SCENE_BANDS);
        let m = 0;
        if (f1 - f0 < bin * 2) { const x = Math.sqrt(f0 * f1) / bin, j = Math.floor(x), fr = x - j; m = (SCENE.bins[j] || 0) * (1 - fr) + (SCENE.bins[j + 1] || 0) * fr; }
        else for (let j = Math.floor(f0 / bin); j <= Math.ceil(f1 / bin) && j < SCENE.bins.length; j++) m = Math.max(m, SCENE.bins[j]);
        const prev = SCENE.lastRow ? SCENE.lastRow[k] : 0;
        row[k] = Math.max(m, prev * 0.86);
      }
    }
    SCENE.head = (SCENE.head + 1) % SCENE_ROWS;
    d.queue.writeTexture({ texture: SCENE.spec, origin: [0, SCENE.head] }, row, { bytesPerRow: SCENE_BANDS }, [SCENE_BANDS, 1]);
    SCENE.lastRow = row;
    // the sonogram keeps the spectrum by song position, so every bar has its own place
    if (so && so.live && S.playing) {
      const r = Math.max(0, Math.min(SONO_ROWS - 1, Math.floor(so.pos * SONO_ROWS)));
      let from = r;
      if (SCENE.sonoLast >= 0 && r > SCENE.sonoLast && r - SCENE.sonoLast < 48) from = SCENE.sonoLast + 1;
      const cnt = r - from + 1, rows = new Uint8Array(SCENE_BANDS * cnt);
      for (let k = 0; k < cnt; k++) rows.set(row, k * SCENE_BANDS);
      d.queue.writeTexture({ texture: SCENE.sono, origin: [0, from] }, rows, { bytesPerRow: SCENE_BANDS }, [SCENE_BANDS, cnt]);
      SCENE.sonoLast = r;
    }
  }
  if (!pipe) return false;
  const lr = SCENE.lastRow || new Uint8Array(SCENE_BANDS), avg = (a, b) => { let s = 0; for (let k = a; k < b; k++) s += lr[k]; return s / (b - a) / 255; };
  bass = avg(0, 12); mid = avg(20, 40); high = avg(44, 60);
  sceneKick = Math.max(sceneKick * 0.9, bass > 0.62 ? (bass - 0.62) * 2.6 : 0);
  const buf = new ArrayBuffer(SCENE_UBYTES), f = new Float32Array(buf), ui = new Uint32Array(buf);
  f[0] = w; f[1] = h; f[2] = now % 1000; f[3] = beat || 0;
  f[4] = bass; f[5] = mid; f[6] = high; f[7] = Math.min(1, sceneKick);
  const em = emitters.slice(0, SCENE_MAXE);
  ui[8] = em.length; f[9] = SCENE.head; f[10] = SCENE_ROWS; f[11] = off ? 1 : dpr;
  f.set([...(opts.ink || [0.9, 0.91, 0.93]), 1], 12); f.set([...(opts.accent || [1, 0.6, 0.3]), 1], 16);
  em.forEach((e, k) => { const t0 = SCENE.hits.get(e.line); f.set([e.y, e.x, e.col, t0 ? Math.min(30, now - t0) : 30], 20 + k * 4); });
  TRACK_COLORS.forEach((hx, k) => { const v = parseInt(hx.slice(1), 16); f.set([(v >> 16 & 255) / 255, (v >> 8 & 255) / 255, (v & 255) / 255, 1], 116 + k * 4); });
  f.set([so ? so.pos : 0, so ? so.bars : 1, so && S.playing ? 1 : 0, 0], 156);
  for (let k = 0; k < 16; k++) f[160 + k] = so && so.cuts && k < so.cuts.length ? so.cuts[k] : -1;
  d.queue.writeBuffer(SCENE.uni, 0, buf);
  const enc = d.createCommandEncoder();
  const rp = enc.beginRenderPass({ colorAttachments: [{ view: off ? off.tex.createView() : SCENE.ctx.getCurrentTexture().createView(), loadOp: 'clear', clearValue: { r: 0, g: 0, b: 0, a: 0 }, storeOp: 'store' }] });
  rp.setPipeline(pipe); rp.setBindGroup(0, SCENE.bg); rp.draw(3); rp.end();
  d.queue.submit([enc.finish()]);
  return true;
}

// ---------- benchmark: the same note on the GPU and in JavaScript ----------
async function gpuBench() {
  const sr = 48000, P = Object.assign({}, BASE, INST.swarm), v = gpuVoice('swarm', 57, 2, P, sr);
  const out = { partials: v.np, seconds: v.n / sr };
  // JavaScript: time a 1/32 slice and scale up, so the page does not freeze for seconds
  const slice = Math.ceil(v.n / 32), t0 = performance.now();
  cpuRender(v.parts, v.np, v.n, v.o, 0, slice);
  out.cpuMs = (performance.now() - t0) * (v.n / slice);
  if (await gpuInit()) {
    await gpuRender(v.parts, v.np, 4096, v.o);      // compile and warm up
    const t1 = performance.now(); await gpuRender(v.parts, v.np, v.n, v.o); out.gpuMs = performance.now() - t1;
  }
  out.gpu = GPU.state === 'ok' ? GPU.name : null; out.err = GPU.err;
  return out;
}
