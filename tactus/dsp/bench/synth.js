// ===================== Pulse audio-thread synths (AudioWorklet) =====================
// zither: Karplus-Strong plucked string (gayageum, koto, harp). acid: TB-303 style mono voice with
// a saturating 4-pole ladder filter, slide (glide into the next note) and accent.
// Events carry an exact sample frame, so timing is sample-accurate. The voices run in Rust compiled to
// WebAssembly (dsp/src/lib.rs); the JavaScript voices below are the fallback when WASM is unavailable,
// and if worklets can't load at all, both fall back to the WebAudio voices.
const SYN_CODE = `
const TAU = Math.PI * 2;
function blep(t, dt) {
  if (t < dt) { t /= dt; return t + t - t * t - 1; }
  if (t > 1 - dt) { t = (t - 1) / dt; return t * t + t + t + 1; }
  return 0;
}
function th(x) { if (x > 3) return 1; if (x < -3) return -1; const x2 = x * x; return x * (27 + x2) / (27 + 9 * x2); }
class Syn extends AudioWorkletProcessor {
  constructor() {
    super();
    this.q = []; this.ks = []; this.ac = null; this.x = null;
    this.port.onmessage = e => {
      const m = e.data;
      if (m.t === 'wasm') {
        try {
          const x = new WebAssembly.Instance(new WebAssembly.Module(m.bytes), {}).exports;
          x.init(sampleRate, (Math.random() * 4294967295) >>> 0);
          this.oL = new Float32Array(x.memory.buffer, x.out_l(), 128); this.oR = new Float32Array(x.memory.buffer, x.out_r(), 128);
          this.x = x; this.port.postMessage({ t: 'wasm-ok' });
        } catch (err) { this.port.postMessage({ t: 'wasm-fail', err: String(err) }); }
        return;
      }
      if (m.t === 'live') { if (this.x) { this.x.live(m.cut != null ? m.cut : -1, m.res != null ? m.res : -1); return; } if (this.ac) { if (m.cut != null) this.ac.cut = m.cut; if (m.res != null) this.ac.k = Math.min(3.9, m.res / 4); } return; }
      this.q.push(m); this.q.sort((a, b) => a.frame - b.frame);
    };
  }
  ksOn(m) {
    const sr = sampleRate, period = sr / m.f, n = Math.ceil(period * 1.04) + 4;
    const buf = new Float32Array(n), a = 0.12 + 0.85 * m.bright;
    let y = 0, mean = 0;
    for (let i = 0; i < n; i++) { y += a * ((Math.random() * 2 - 1) - y); buf[i] = y; mean += y; }
    mean /= n; let pk = 0;
    for (let i = 0; i < n; i++) { buf[i] -= mean; pk = Math.max(pk, Math.abs(buf[i])); }
    for (let i = 0; i < n; i++) buf[i] /= pk || 1;
    const p = (Math.max(-1, Math.min(1, m.pan)) + 1) * Math.PI / 4;
    if (this.ks.length >= 16) this.ks.shift();
    this.ks.push({ buf, n, w: 0, base: period - 0.5, prev: 0, amp: m.gain, pl: Math.cos(p), pr: Math.sin(p),
      g: Math.pow(10, -3 / (Math.max(0.05, m.t60) * m.f)), gRel: Math.pow(10, -3 / (0.15 * m.f)),
      off: m.frame + m.dur, start: m.frame, vib: m.vib || 0, ph: Math.random() * TAU, rate: (5 + Math.random()) / sr, life: m.frame + m.dur + Math.round(sr * Math.min(m.t60, 6)) });
  }
  acOn(m) {
    let v = this.ac;
    if (!v) v = this.ac = { ph: 0, f: m.f, target: m.f, y0: 0, y1: 0, y2: 0, y3: 0, fenv: 0, amp: 0, gate: false, off: 0 };
    const legato = v.gate && m.frame <= v.off + 64;
    v.slide = legato || (m.slide && v.amp > 0.001);
    v.target = m.f;
    if (!v.slide) { v.f = m.f; v.fenv = 1; }
    if (m.acc) v.fenv = Math.min(1.8, v.fenv + 0.7);
    v.gate = true; v.off = m.frame + m.dur;
    v.cut = m.cut; v.k = Math.min(3.9, m.res / 4); v.env = m.env; v.acc = m.acc;
    v.dec = Math.exp(-1 / (Math.max(0.02, m.dec) * (m.acc ? 0.6 : 1) * sampleRate));
    v.gain = m.gain; v.wave = m.wave; v.drive = 1 + m.drive * 3;
  }
  process(_, outputs) {
    const out = outputs[0], L = out[0], R = out[1] || out[0], N = L.length, sr = sampleRate;
    if (this.x) {
      // Rust/WASM path: render up to each event's exact sample, apply it, continue
      const x = this.x, oL = this.oL, oR = this.oR, base = currentFrame, q = this.q;
      let pos = 0;
      while (pos < N) {
        const next = q.length ? Math.max(pos, Math.min(N, q[0].frame - base)) : N;
        if (next > pos) { x.render(next - pos); for (let i = pos, j = 0; i < next; i++, j++) { L[i] = oL[j]; R[i] = oR[j]; } pos = next; }
        while (q.length && q[0].frame <= base + pos) {
          const m = q.shift();
          if (m.kind === 'ks') x.ks_on(m.f, m.gain, m.pan, m.bright, m.t60, m.vib || 0, m.dur);
          else x.acid_on(m.f, m.dur, m.slide ? 1 : 0, m.acc ? 1 : 0, m.cut, m.res, m.env, m.dec, m.gain, m.wave, m.drive);
        }
      }
      return true;
    }
    const glide = 1 - Math.exp(-1 / (0.055 * sr)), aAtk = 1 - Math.exp(-1 / (0.002 * sr)), aRel = 1 - Math.exp(-1 / (0.008 * sr));
    const fmax = sr * 0.42;
    for (let i = 0; i < N; i++) {
      const fr = currentFrame + i;
      while (this.q.length && this.q[0].frame <= fr) { const m = this.q.shift(); if (m.kind === 'ks') this.ksOn(m); else this.acOn(m); }
      let l = 0, r = 0;
      for (const v of this.ks) {
        let d = v.base;
        if (v.vib) {
          const age = (fr - v.start) / sr, depth = v.vib * Math.min(1, Math.max(0, (age - 0.2) / 0.4));
          v.ph += TAU * v.rate; d = v.base * Math.pow(2, -depth * Math.sin(v.ph) / 1200);
        }
        let rd = v.w - d; while (rd < 0) rd += v.n;
        const i0 = rd | 0, f = rd - i0, i1 = i0 + 1 === v.n ? 0 : i0 + 1;
        const y = v.buf[i0] * (1 - f) + v.buf[i1] * f;
        const s = 0.5 * (y + v.prev); v.prev = y;
        const sv = s * (fr >= v.off ? v.gRel : v.g); v.buf[v.w] = sv > -1e-20 && sv < 1e-20 ? 0 : sv;
        v.w = v.w + 1 === v.n ? 0 : v.w + 1;
        const o = y * v.amp; l += o * v.pl; r += o * v.pr;
      }
      const v = this.ac;
      if (v) {
        if (v.gate && fr >= v.off) v.gate = false;
        if (v.slide) v.f += (v.target - v.f) * glide;
        const dt = v.f / sr;
        v.ph += dt; if (v.ph >= 1) v.ph -= 1;
        let osc;
        if (v.wave > 0.5) { let p2 = v.ph + 0.5; if (p2 >= 1) p2 -= 1; osc = (v.ph < 0.5 ? 1 : -1) + blep(v.ph, dt) - blep(p2, dt); }
        else osc = 2 * v.ph - 1 - blep(v.ph, dt);
        v.fenv *= v.dec;
        const fc = Math.min(fmax, Math.max(30, v.cut * (1 + v.env * 8 * v.fenv)));
        const g = 1 - Math.exp(-TAU * fc / sr), k = v.k;
        const x = th(osc * v.drive * 0.8 - k * v.y3);
        v.y0 += g * (x - th(v.y0)); v.y1 += g * (th(v.y0) - th(v.y1)); v.y2 += g * (th(v.y1) - th(v.y2)); v.y3 += g * (th(v.y2) - th(v.y3));
        const target = v.gate ? (v.acc ? 1.35 : 1) : 0;
        v.amp += (target - v.amp) * (v.gate ? aAtk : aRel);
        const o = v.y3 * v.amp * v.gain * (1 + k * 0.35);
        l += o; r += o;
      }
      L[i] = l; R[i] = r;
    }
    if (this.ks.length) this.ks = this.ks.filter(v => v.life > currentFrame);
    return true;
  }
}
registerProcessor('pulse-synth', Syn);
`;
const WK = { ok: null, loading: null, mod: null, dsp: 'js' };
// decode the Rust DSP once; each worklet node compiles its own instance from these bytes
// (browsers do not pass compiled modules into the audio thread, so the bytes travel instead)
async function dspModule() {
  if (WK.mod !== null) return WK.mod;
  try {
    if (typeof DSP_WASM_B64 === 'undefined' || typeof WebAssembly === 'undefined' || /[?&]dsp=js\b/.test(location.search)) throw new Error('off');
    const bin = atob(DSP_WASM_B64), u = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
    if (!WebAssembly.validate(u)) throw new Error('invalid');
    WK.mod = u;
  } catch (e) { WK.mod = false; }
  return WK.mod;
}
function ensureSynthWorklet() {
  if (WK.loading) return WK.loading;
  WK.loading = (async () => {
    const urls = [URL.createObjectURL(new Blob([SYN_CODE], { type: 'text/javascript' })), 'data:text/javascript;charset=utf-8,' + encodeURIComponent(SYN_CODE)];
    dspModule();
    for (const u of urls) { try { await S.ctx.audioWorklet.addModule(u); WK.ok = true; return true; } catch (e) { } }
    WK.ok = false; return false;
  })();
  return WK.loading;
}
function wkNode(name) {
  const r = rt(name);
  if (!r.wk) {
    r.wk = new AudioWorkletNode(S.ctx, 'pulse-synth', { numberOfInputs: 0, numberOfOutputs: 1, outputChannelCount: [2] });
    r.wk.connect(r.input);
    const node = r.wk;
    node.port.onmessage = e => { if (e.data.t === 'wasm-ok') { WK.dsp = 'wasm'; S.dirty = true; } else if (e.data.t === 'wasm-fail') console.warn('Rust DSP unavailable, using JS voices:', e.data.err); };
    dspModule().then(m => { if (m) node.port.postMessage({ t: 'wasm', bytes: m.slice().buffer }); });
  }
  return r.wk;
}
// returns true when the worklet took the note
function wkPlay(def, notes, time, dur, P) {
  if (!WK.ok || (def.inst !== 'acid' && def.inst !== 'zither')) return false;
  const sr = S.ctx.sampleRate, frame = Math.round(time * sr), dFr = Math.max(1, Math.round(dur * sr));
  const node = wkNode(def.name);
  if (def.inst === 'acid') {
    node.port.postMessage({ kind: 'acid', frame, dur: dFr, f: mtof(notes[notes.length - 1]), slide: P.slide > 0.5, acc: P.acc > 0.5,
      cut: Math.max(40, P.cut), res: P.res, env: P.env, dec: P.dec, gain: P.gain * 0.27, wave: P.wave || 0, drive: (P.drive || 0) + (P.shape || 0) });
  } else {
    for (const m of notes) node.port.postMessage({ kind: 'ks', frame, dur: dFr, f: mtof(m), gain: P.gain * 1.3, pan: P.pan, bright: Math.max(0, Math.min(1, P.bright)), t60: P.dec, vib: P.vib });
  }
  return true;
}
// sounding WebAudio voices per track, so a knob turn reaches notes that are already ringing
function registerVoice(r, v) {
  if (!v) return;
  const now = S.ctx.currentTime;
  r.voices = (r.voices || []).filter(x => x.tEnd > now);
  r.voices.push(v);
}
function liveKnob(i) {
  if (!S.ctx || !S.playing) return;
  const now = S.ctx.currentTime, s = transport(now);
  for (const name of S.order) {
    const def = S.tracks.get(name);
    if (!def || def.kind !== 'synth') continue;
    const pc = def.params.cut, pr = def.params.res;
    const hitCut = pc && pc.k === 'knob' && pc.i === i, hitRes = pr && pr.k === 'knob' && pr.i === i;
    if (!hitCut && !hitRes) continue;
    const r = S.rt.get(name); if (!r) continue;
    const cut = hitCut ? valueAt(pc, s, 'cut') : null, res = hitRes ? valueAt(pr, s, 'res') : null;
    if (r.wk) r.wk.port.postMessage({ t: 'live', cut, res });
    r.voices = (r.voices || []).filter(v => v.tEnd > now);
    for (const v of r.voices) v.filters.forEach((fl, k) => {
      if (cut != null) { fl.frequency.cancelScheduledValues(now); fl.frequency.setTargetAtTime(Math.max(40, Math.min(18000, cut * v.kt)), now, 0.025); }
      if (res != null && k === 0) { fl.Q.cancelScheduledValues(now); fl.Q.setTargetAtTime(Math.max(0.1, res), now, 0.025); }
    });
  }
}
