// ===================== Tactus GPU (WebGPU) =====================
// The audio thread cannot see the GPU (WebGPU is not exposed inside an AudioWorklet), so the GPU works
// ahead of time on the main thread: it renders whole notes and textures into buffers that the scheduler
// then plays at the exact time. That suits sounds that are too heavy for the CPU but can be prepared a
// moment early:
//   swarm  a supersaw pad of 40 detuned voices x 48 harmonics (about 1,900 sine partials per note)
//   choir  24 detuned voices with vowel formants (about 1,000 partials per note)
//   plate  a struck metal plate: every inharmonic mode split into six beating neighbours, each with its own decay
//          (a few hundred partials in the middle register, more for low notes)
//   fx ... | freeze   spectral freeze: the live mix's spectrum resynthesized as ~2,000 sines
// and it draws the "field" visualization: 131,072 particles thrown from the code lines of the tracks.
// Without WebGPU every part falls back: the GPU instruments to CPU voices, freeze to a looped grain.

const GPU = { state: 'idle', dev: null, name: '', err: '', synth: null, cache: new Map(), bytes: 0, jobs: new Map(), chain: Promise.resolve(), renders: 0, lastMs: 0 };
const GPU_INSTS = { swarm: 'supersaw', choir: 'pad', plate: 'bell' };   // and their CPU fallbacks
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
const gpuKey = (inst, midi, dur, P) => [inst, midi, inst === 'plate' ? 0 : Math.round(dur * 8) / 8, (P.bright ?? 0).toFixed(2), (P.detune ?? 0).toFixed(0), (P.dec ?? 0).toFixed(2), (P.width ?? 0).toFixed(2), (P.atk ?? 0).toFixed(2), (P.rel ?? 0).toFixed(2)].join('|');
function gpuRequest(inst, midi, dur, P) {
  const key = gpuKey(inst, midi, dur, P);
  if (GPU.cache.has(key) || GPU.jobs.has(key)) return key;
  const job = GPU.chain = GPU.chain.then(async () => {
    if (GPU.state !== 'ok') return;
    const sr = S.ctx.sampleRate, v = gpuVoice(inst, midi, dur, P, sr);
    const pcm = await gpuRender(v.parts, v.np, v.n, v.o);
    const buf = S.ctx.createBuffer(2, v.n, sr);
    buf.copyToChannel(pcm.subarray(0, v.n), 0); buf.copyToChannel(pcm.subarray(v.n), 1);
    GPU.cache.set(key, buf); GPU.bytes += v.n * 8;
    while (GPU.bytes > GPU_CACHE_BYTES && GPU.cache.size > 1) { const [k0, b0] = GPU.cache.entries().next().value; GPU.cache.delete(k0); GPU.bytes -= b0.length * 8; }
  }).catch(e => { GPU.err = String(e.message || e); }).finally(() => GPU.jobs.delete(key));
  GPU.jobs.set(key, job);
  return key;
}
// play GPU-rendered notes; returns the notes it could not play yet (still rendering, or no GPU)
function gpuPlay(def, notes, time, dur, P, vel, r) {
  if (GPU.state !== 'ok') return notes;
  const miss = [];
  for (const m of notes) {
    const key = gpuRequest(def.inst, m, dur, P), buf = GPU.cache.get(key);
    if (!buf) { miss.push(m); continue; }
    GPU.cache.delete(key); GPU.cache.set(key, buf);   // keep recently used notes
    const src = S.ctx.createBufferSource(), g = S.ctx.createGain();
    src.buffer = buf; g.gain.value = P.gain * vel * (INST[def.inst].lvl || 0.5);
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

// ---------- field: 131,072 particles thrown from the code lines ----------
const FIELD_N = 131072, FIELD_MAXE = 24;
const FIELD_COMMON = `
struct U { dt: f32, time: f32, aspect: f32, ne: u32, bass: f32, mid: f32, high: f32, size: f32, em: array<vec4f, 24> };
struct Pt { pos: vec2f, vel: vec2f, life: f32, maxl: f32, col: f32, seed: f32 };
@group(0) @binding(0) var<uniform> u: U;`;
const FIELD_COMP = FIELD_COMMON + `
@group(0) @binding(1) var<storage, read_write> pts: array<Pt>;
fn hash(x: u32) -> f32 { var h = x * 747796405u + 2891336453u; h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u; h = (h >> 22u) ^ h; return f32(h) / 4294967296.0; }
// a divergence-free swirl: the curl of a moving stream function, so particles circulate instead of clumping
fn flow(p: vec2f, t: f32) -> vec2f {
  let A = p.x * 3.1 + t * 0.7; let B = p.y * 2.3 - t * 0.5; let C = p.x * 1.7 - t * 0.3; let D = p.y * 4.1 + t * 0.9;
  let dy = -2.3 * sin(A) * sin(B) + 0.5 * 4.1 * cos(C) * cos(D);
  let dx = 3.1 * cos(A) * cos(B) - 0.5 * 1.7 * sin(C) * sin(D);
  return vec2f(dy, -dx) * 0.25;
}
@compute @workgroup_size(256)
fn step(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= arrayLength(&pts)) { return; }
  var p = pts[i];
  let seed = u32(fract(u.time * 0.137) * 1000003.0) * 1664525u + i * 22695477u;
  p.life -= u.dt;
  if (p.life <= 0.0) {
    var tot = 0.0;
    for (var k = 0u; k < u.ne; k++) { tot += u.em[k].y; }
    if (hash(seed) < tot * 0.08 * u.dt) {
      var pick = hash(seed + 7u) * tot;
      var e = u.em[0];
      for (var k = 0u; k < u.ne; k++) { e = u.em[k]; pick -= e.y; if (pick <= 0.0) { break; } }
      p.pos = vec2f(e.w + hash(seed + 11u) * 0.05, e.x + (hash(seed + 13u) - 0.5) * 0.012);
      let sp = (0.18 + hash(seed + 17u) * 0.55) * (0.6 + min(e.y, 2.0));
      let an = (hash(seed + 19u) - 0.5) * 0.9;
      p.vel = vec2f(cos(an) * sp, sin(an) * sp * 0.6);
      p.maxl = 1.4 + hash(seed + 23u) * 2.6;
      p.life = p.maxl;
      p.col = e.z;
    } else { p.life = 0.0; pts[i] = p; return; }
  }
  let f = flow(p.pos * vec2f(u.aspect, 1.0) * 1.6, u.time * (0.35 + u.mid * 1.5));
  p.vel += f * (0.25 + u.bass * 2.4) * u.dt;
  p.vel *= 1.0 - (0.9 - u.high * 0.4) * u.dt;
  p.pos += p.vel * u.dt * vec2f(1.0 / u.aspect, 1.0);
  if (p.pos.x > 1.05 || p.pos.y < -0.05 || p.pos.y > 1.05) { p.life = 0.0; }
  pts[i] = p;
}`;
const FIELD_REND = FIELD_COMMON + `
@group(0) @binding(1) var<storage, read> pts: array<Pt>;
@group(0) @binding(2) var<storage, read> pal: array<vec4f, 10>;
struct VO { @builtin(position) pos: vec4f, @location(0) col: vec4f, @location(1) uv: vec2f };
@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VO {
  let p = pts[ii];
  var o: VO;
  let corner = vec2f(f32(vi & 1u), f32(vi >> 1u)) * 2.0 - 1.0;
  let k = clamp(p.life / max(p.maxl, 0.001), 0.0, 1.0);
  let sz = u.size * (0.6 + k * 1.4);
  let c = vec2f(p.pos.x * 2.0 - 1.0, 1.0 - p.pos.y * 2.0);
  o.pos = vec4f(c + corner * vec2f(sz / u.aspect, sz), 0.0, 1.0);
  if (p.life <= 0.0) { o.pos = vec4f(2.0, 2.0, 0.0, 1.0); }
  let col = pal[u32(p.col) % 10u];
  let a = pow(k, 1.6) * 0.22;
  o.col = vec4f(col.rgb * a, a);
  o.uv = corner;
  return o;
}
@fragment
fn fs(i: VO) -> @location(0) vec4f {
  let d = dot(i.uv, i.uv);
  if (d > 1.0) { discard; }
  return i.col * (1.0 - d);
}`;
const FIELD = { ready: false, failed: false, hits: new Map(), lastT: 0 };
// The field gets its own GPU device, so a problem with the canvas can never take the audio renders down with it.
// offscreen: draw into a texture instead of the canvas (used by tests and headless browsers)
async function fieldInit(canvas, offscreen) {
  if (FIELD.ready || FIELD.failed || FIELD.loading) return;
  FIELD.loading = true;
  if (!(await gpuInit())) { FIELD.failed = true; FIELD.loading = false; return; }
  try {
    const ad = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });   // an adapter makes only one device
    if (!ad) throw new Error('no GPU adapter for the field');
    FIELD.adapter = ad;
    const d = await ad.requestDevice();
    d.lost.then(() => { FIELD.ready = false; FIELD.failed = true; });
    let ctx = null, fmt = 'rgba8unorm';
    if (offscreen) FIELD.offscreen = { w: offscreen.w, h: offscreen.h, tex: d.createTexture({ size: [offscreen.w, offscreen.h], format: fmt, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC }) };
    else { ctx = canvas.getContext('webgpu'); fmt = navigator.gpu.getPreferredCanvasFormat(); ctx.configure({ device: d, format: fmt, alphaMode: 'premultiplied' }); }
    FIELD.dev = d;
    const pts = d.createBuffer({ size: FIELD_N * 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const uni = d.createBuffer({ size: 32 + FIELD_MAXE * 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const pal = d.createBuffer({ size: 160, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const pd = new Float32Array(40);
    TRACK_COLORS.forEach((h, k) => { const v = parseInt(h.slice(1), 16); pd.set([(v >> 16 & 255) / 255, (v >> 8 & 255) / 255, (v & 255) / 255, 1], k * 4); });
    d.queue.writeBuffer(pal, 0, pd);
    const comp = d.createComputePipeline({ layout: 'auto', compute: { module: d.createShaderModule({ code: FIELD_COMP }), entryPoint: 'step' } });
    const rm = d.createShaderModule({ code: FIELD_REND });
    const blend = { color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' }, alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' } };
    const rend = d.createRenderPipeline({ layout: 'auto', vertex: { module: rm, entryPoint: 'vs' }, fragment: { module: rm, entryPoint: 'fs', targets: [{ format: fmt, blend }] }, primitive: { topology: 'triangle-strip' } });
    Object.assign(FIELD, { ctx, uni, pts, comp, rend, canvas, fmt,
      cbg: d.createBindGroup({ layout: comp.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: uni } }, { binding: 1, resource: { buffer: pts } }] }),
      rbg: d.createBindGroup({ layout: rend.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: uni } }, { binding: 1, resource: { buffer: pts } }, { binding: 2, resource: { buffer: pal } }] }),
      ready: true });
  } catch (e) { FIELD.failed = true; GPU.err = String(e.message || e); }
  FIELD.loading = false;
}
// a track hit on a code line: that line throws particles in its colour
function fieldHit(line) { FIELD.hits.set(line, Math.min(2, (FIELD.hits.get(line) || 0) + 1)); }
let fieldBins = null;
function fieldFrame(now, emitters) {
  if (!FIELD.ready) return;
  const cv = FIELD.canvas, dpr = devicePixelRatio || 1, off = FIELD.offscreen;
  const w = off ? off.w : Math.max(1, Math.round(cv.clientWidth * dpr)), h = off ? off.h : Math.max(1, Math.round(cv.clientHeight * dpr));
  if (!off && (cv.width !== w || cv.height !== h)) { cv.width = w; cv.height = h; }
  const t = performance.now() / 1000, dt = Math.min(0.05, FIELD.lastT ? t - FIELD.lastT : 0.016); FIELD.lastT = t;
  let bass = 0, mid = 0, high = 0;
  if (A.an && S.playing) {
    fieldBins = fieldBins || new Uint8Array(A.an.frequencyBinCount); A.an.getByteFrequencyData(fieldBins);
    const bin = S.ctx.sampleRate / A.an.fftSize, band = (a, b) => { let s = 0, c = 0; for (let k = Math.floor(a / bin); k <= Math.ceil(b / bin); k++, c++) s += fieldBins[k] || 0; return s / Math.max(1, c) / 255; };
    bass = band(30, 160); mid = band(300, 2000); high = band(4000, 12000);
  }
  const u = new ArrayBuffer(32 + FIELD_MAXE * 16), f = new Float32Array(u), ui = new Uint32Array(u);
  const em = emitters.slice(0, FIELD_MAXE);
  f[0] = dt; f[1] = t; f[2] = w / h; ui[3] = em.length; f[4] = bass; f[5] = mid; f[6] = high; f[7] = 2.2 * dpr / h;
  em.forEach((e, k) => { const en = FIELD.hits.get(e.line) || 0; f.set([e.y, en, e.col, e.x], 8 + k * 4); });
  for (const [k, v] of FIELD.hits) { const nv = v * Math.pow(0.02, dt); if (nv < 0.01) FIELD.hits.delete(k); else FIELD.hits.set(k, nv); }
  const d = FIELD.dev;
  d.queue.writeBuffer(FIELD.uni, 0, u);
  const enc = d.createCommandEncoder();
  const cp = enc.beginComputePass(); cp.setPipeline(FIELD.comp); cp.setBindGroup(0, FIELD.cbg); cp.dispatchWorkgroups(FIELD_N / 256); cp.end();
  const rp = enc.beginRenderPass({ colorAttachments: [{ view: off ? off.tex.createView() : FIELD.ctx.getCurrentTexture().createView(), loadOp: 'clear', clearValue: { r: 0, g: 0, b: 0, a: 0 }, storeOp: 'store' }] });
  rp.setPipeline(FIELD.rend); rp.setBindGroup(0, FIELD.rbg); rp.draw(4, FIELD_N); rp.end();
  d.queue.submit([enc.finish()]);
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
