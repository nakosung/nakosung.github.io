//! Tactus DSP: the audio-thread voices, written in Rust and compiled to WebAssembly.
//! It runs inside an AudioWorklet; the worklet hands over note events at their exact sample
//! and calls `render` for the samples in between, so timing stays sample-accurate.
//!
//! Voices
//! - Karplus-Strong plucked string (gayageum, koto, guitar, harp), up to 16 at once
//! - a TB-303 style mono voice: polyBLEP saw/square into a saturating 4-pole ladder,
//!   with slide (glide into the next note) and accent
//!
//! `no_std` and no allocator: every buffer is fixed, so nothing allocates on the audio thread.
#![no_std]

use core::f32::consts::PI;
const TAU: f32 = 2.0 * PI;
const BLOCK: usize = 128;
const KS_VOICES: usize = 16;
const KS_MAX: usize = 4096; // longest string period (about 12 Hz at 48 kHz)

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

// ---------- small math kit (no libm on the audio thread) ----------
#[inline(always)]
fn floorf(x: f32) -> f32 {
    let t = x as i32 as f32;
    if t > x { t - 1.0 } else { t }
}
/// 2^x, about 1e-7 relative error
#[inline(always)]
fn exp2f(x: f32) -> f32 {
    let x = x.clamp(-126.0, 126.0);
    let i = floorf(x);
    let f = x - i;
    let p = 1.0 + f * (0.693_147_2 + f * (0.240_226_5 + f * (0.055_504_11 + f * (0.009_618_129 + f * 0.001_333_355))));
    p * f32::from_bits(((i as i32 + 127) as u32) << 23)
}
#[inline(always)]
fn powf(b: f32, e: f32) -> f32 { exp2f(e * log2f(b)) }
fn log2f(x: f32) -> f32 {
    // split into exponent and mantissa in [1,2), then a short series for the mantissa
    let bits = x.to_bits();
    let e = ((bits >> 23) & 0xff) as i32 - 127;
    let m = f32::from_bits((bits & 0x007f_ffff) | 0x3f80_0000);
    let t = (m - 1.0) / (m + 1.0);
    let t2 = t * t;
    let ln = 2.0 * t * (1.0 + t2 * (1.0 / 3.0 + t2 * (1.0 / 5.0 + t2 * (1.0 / 7.0 + t2 / 9.0))));
    e as f32 + ln * core::f32::consts::LOG2_E
}
/// sine on any input, 7th-order odd polynomial after reduction to [-pi/2, pi/2]
#[inline(always)]
fn sinf(x: f32) -> f32 {
    let mut x = x - TAU * floorf(x / TAU + 0.5); // [-pi, pi]
    if x > PI / 2.0 { x = PI - x } else if x < -PI / 2.0 { x = -PI - x }
    let x2 = x * x;
    x * (1.0 + x2 * (-0.166_666_67 + x2 * (0.008_333_331 + x2 * (-0.000_198_409 + x2 * 0.000_002_752_6))))
}
#[inline(always)]
fn cosf(x: f32) -> f32 { sinf(x + PI / 2.0) }
/// e^x in double precision (the ladder filter feeds back on itself, so it keeps full precision like the JS voice)
#[inline(always)]
fn expd(x: f64) -> f64 {
    let x = x.clamp(-700.0, 700.0);
    let kf = x * core::f64::consts::LOG2_E + 0.5;
    let k = { let t = kf as i64 as f64; if t > kf { t - 1.0 } else { t } };
    let r = x - k * core::f64::consts::LN_2;
    let mut term = 1.0f64; let mut sum = 1.0f64;
    let mut i = 1.0f64;
    while i < 11.0 { term *= r / i; sum += term; i += 1.0; }
    sum * f64::from_bits(((k as i64 + 1023) as u64) << 52)
}
#[inline(always)]
fn thd(x: f64) -> f64 {
    if x > 3.0 { return 1.0 }
    if x < -3.0 { return -1.0 }
    let x2 = x * x;
    x * (27.0 + x2) / (27.0 + 9.0 * x2)
}
#[inline(always)]
fn blepd(mut t: f64, dt: f64) -> f64 {
    if t < dt { t /= dt; return t + t - t * t - 1.0 }
    if t > 1.0 - dt { t = (t - 1.0) / dt; return t * t + t + t + 1.0 }
    0.0
}
// ---------- state ----------
#[derive(Clone, Copy)]
struct Ks {
    on: bool, n: usize, w: usize, base: f32, prev: f32, amp: f32, pl: f32, pr: f32,
    g: f32, g_rel: f32, off: u64, start: u64, life: u64, vib: f32, ph: f32, rate: f32,
}
const KS0: Ks = Ks { on: false, n: 0, w: 0, base: 0.0, prev: 0.0, amp: 0.0, pl: 0.0, pr: 0.0, g: 0.0, g_rel: 0.0, off: 0, start: 0, life: 0, vib: 0.0, ph: 0.0, rate: 0.0 };

struct Acid {
    on: bool, ph: f64, f: f64, target: f64, y: [f64; 4], fenv: f64, amp: f64, gate: bool, off: u64, slide: bool,
    cut: f64, k: f64, env: f64, acc: bool, dec: f64, gain: f64, square: bool, drive: f64,
}

struct Dsp {
    sr: f32, now: u64, seed: u32,
    ks: [Ks; KS_VOICES], ks_next: usize,
    ac: Acid,
    out_l: [f32; BLOCK], out_r: [f32; BLOCK],
}
static mut D: Dsp = Dsp {
    sr: 48000.0, now: 0, seed: 0x9e37_79b9,
    ks: [KS0; KS_VOICES], ks_next: 0,
    ac: Acid { on: false, ph: 0.0, f: 110.0, target: 110.0, y: [0.0; 4], fenv: 0.0, amp: 0.0, gate: false, off: 0, slide: false, cut: 800.0, k: 0.0, env: 0.0, acc: false, dec: 0.999, gain: 0.0, square: false, drive: 1.0 },
    out_l: [0.0; BLOCK], out_r: [0.0; BLOCK],
};
// string delay lines live apart from the rest so they stay zero-filled memory instead of bytes in the module
static mut KS_BUF: [[f32; KS_MAX]; KS_VOICES] = [[0.0; KS_MAX]; KS_VOICES];
#[inline(always)]
fn d() -> &'static mut Dsp { unsafe { &mut *core::ptr::addr_of_mut!(D) } }
#[inline(always)]
fn ks_buf() -> &'static mut [[f32; KS_MAX]; KS_VOICES] { unsafe { &mut *core::ptr::addr_of_mut!(KS_BUF) } }
fn rnd(s: &mut u32) -> f32 {
    *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5;
    (*s as f32) * (1.0 / 4_294_967_296.0)
}

// ---------- exports ----------
#[unsafe(no_mangle)]
pub extern "C" fn init(sr: f32, seed: u32) {
    let d = d();
    d.sr = sr; d.now = 0; d.seed = seed | 1;
    for v in d.ks.iter_mut() { v.on = false }
    d.ac.on = false;
}
#[unsafe(no_mangle)]
pub extern "C" fn out_l() -> *const f32 { d().out_l.as_ptr() }
#[unsafe(no_mangle)]
pub extern "C" fn out_r() -> *const f32 { d().out_r.as_ptr() }
#[unsafe(no_mangle)]
pub extern "C" fn active() -> u32 { let d = d(); d.ks.iter().filter(|v| v.on).count() as u32 + d.ac.on as u32 }

/// pluck a string now. dur and the decay are in samples / seconds like the JS voice
#[unsafe(no_mangle)]
pub extern "C" fn ks_on(f: f32, gain: f32, pan: f32, bright: f32, t60: f32, vib: f32, dur: u32) {
    let d = d();
    let sr = d.sr;
    let period = sr / f.max(sr / (KS_MAX as f32 - 8.0));
    let n = ((period * 1.04) as usize + 5).min(KS_MAX);
    // reuse a free voice, or the oldest one when all 16 ring
    let slot = match d.ks.iter().position(|v| !v.on) { Some(i) => i, None => { let i = d.ks_next; d.ks_next = (i + 1) % KS_VOICES; i } };
    let buf = &mut ks_buf()[slot];
    let a = 0.12 + 0.85 * bright.clamp(0.0, 1.0);
    let (mut y, mut mean) = (0.0f32, 0.0f32);
    for i in 0..n { y += a * ((rnd(&mut d.seed) * 2.0 - 1.0) - y); buf[i] = y; mean += y; }
    mean /= n as f32;
    let mut pk = 0.0f32;
    for i in 0..n { buf[i] -= mean; pk = pk.max(buf[i].abs()); }
    let inv = if pk > 0.0 { 1.0 / pk } else { 1.0 };
    for i in 0..n { buf[i] *= inv; }
    let p = (pan.clamp(-1.0, 1.0) + 1.0) * PI / 4.0;
    let t60 = t60.max(0.05);
    d.ks[slot] = Ks {
        on: true, n, w: 0, base: period - 0.5, prev: 0.0, amp: gain, pl: cosf(p), pr: sinf(p),
        g: powf(10.0, -3.0 / (t60 * f)), g_rel: powf(10.0, -3.0 / (0.15 * f)),
        off: d.now + dur as u64, start: d.now, life: d.now + dur as u64 + (sr * t60.min(6.0)) as u64,
        vib, ph: rnd(&mut d.seed) * TAU, rate: (5.0 + rnd(&mut d.seed)) / sr,
    };
}

/// start or glide the acid voice
#[unsafe(no_mangle)]
pub extern "C" fn acid_on(f: f32, dur: u32, slide: u32, acc: u32, cut: f32, res: f32, env: f32, dec: f32, gain: f32, wave: f32, drive: f32) {
    let d = d();
    let v = &mut d.ac;
    let (f, cut, res, env, dec, gain, drive) = (f as f64, cut as f64, res as f64, env as f64, dec as f64, gain as f64, drive as f64);
    if !v.on { v.on = true; v.ph = 0.0; v.f = f; v.y = [0.0; 4]; v.fenv = 0.0; v.amp = 0.0; v.gate = false; v.off = 0; }
    let legato = v.gate && d.now <= v.off + 64;
    v.slide = legato || (slide != 0 && v.amp > 0.001);
    v.target = f;
    if !v.slide { v.f = f; v.fenv = 1.0; }
    v.acc = acc != 0;
    if v.acc { v.fenv = (v.fenv + 0.7).min(1.8); }
    v.gate = true; v.off = d.now + dur as u64;
    v.cut = cut; v.k = (res / 4.0).min(3.9); v.env = env;
    v.dec = expd(-1.0 / (dec.max(0.02) * if v.acc { 0.6 } else { 1.0 } * d.sr as f64));
    v.gain = gain; v.square = wave > 0.5; v.drive = 1.0 + drive * 3.0;
}

/// a knob turned while the acid line plays
#[unsafe(no_mangle)]
pub extern "C" fn live(cut: f32, res: f32) {
    let v = &mut d().ac;
    if cut >= 0.0 { v.cut = cut as f64 }
    if res >= 0.0 { v.k = (res as f64 / 4.0).min(3.9) }
}

/// render n (<= 128) samples into out_l / out_r
#[unsafe(no_mangle)]
pub extern "C" fn render(n: u32) {
    let d = d();
    let bufs = ks_buf();
    let n = (n as usize).min(BLOCK);
    let sr = d.sr;
    let srd = sr as f64;
    let glide = 1.0 - expd(-1.0 / (0.055 * srd));
    let a_atk = 1.0 - expd(-1.0 / (0.002 * srd));
    let a_rel = 1.0 - expd(-1.0 / (0.008 * srd));
    let fmax = srd * 0.42;
    for i in 0..n { d.out_l[i] = 0.0; d.out_r[i] = 0.0; }
    // strings: one voice at a time across the block, so its state stays in registers
    for (k, v) in d.ks.iter_mut().enumerate() {
        if !v.on { continue }
        let buf = &mut bufs[k][..v.n];
        let nn = v.n;
        let (mut w, mut prev, mut ph) = (v.w, v.prev, v.ph);
        let (pl, pr) = (v.amp * v.pl, v.amp * v.pr);
        // one tight loop per case, so the common no-vibrato string has no per-sample branch on it
        let g_on = v.g; let g_rel = v.g_rel;
        let off_i = if v.off > d.now { (v.off - d.now).min(n as u64) as usize } else { 0 };
        let vib = v.vib;
        for i in 0..n {
            let dl = if vib != 0.0 {
                let age = (d.now + i as u64 - v.start) as f32 / sr;
                let depth = vib * ((age - 0.2) / 0.4).clamp(0.0, 1.0);
                ph += TAU * v.rate;
                if ph > TAU { ph -= TAU }
                v.base * exp2f(-depth * sinf(ph) / 1200.0)
            } else { v.base };
            let mut rd = w as f32 - dl;
            if rd < 0.0 { rd += nn as f32 }
            if rd < 0.0 { rd += nn as f32 }
            let mut i0 = rd as usize;
            let f = rd - i0 as f32;
            if i0 >= nn { i0 -= nn }
            let i1 = if i0 + 1 == nn { 0 } else { i0 + 1 };
            let (a0, a1) = (buf[i0], buf[i1]);
            let y = a0 + (a1 - a0) * f;
            // flush tiny values: a decaying string otherwise drifts into denormals, which are very slow on the CPU
            let s = 0.5 * (y + prev) * if i >= off_i { g_rel } else { g_on };
            buf[w] = if s.abs() < 1e-20 { 0.0 } else { s };
            prev = y;
            w += 1; if w == nn { w = 0 }
            d.out_l[i] += y * pl; d.out_r[i] += y * pr;
        }
        v.w = w; v.prev = prev; v.ph = ph;
        if d.now + n as u64 >= v.life { v.on = false }
    }
    for i in 0..n {
        let fr = d.now + i as u64;
        let (mut l, mut r) = (0.0f32, 0.0f32);
        let v = &mut d.ac;
        if v.on {
            let srd = sr as f64;
            if v.gate && fr >= v.off { v.gate = false }
            if v.slide { v.f += (v.target - v.f) * glide }
            let dt = v.f / srd;
            v.ph += dt;
            if v.ph >= 1.0 { v.ph -= 1.0 }
            let osc = if v.square {
                let mut p2 = v.ph + 0.5; if p2 >= 1.0 { p2 -= 1.0 }
                (if v.ph < 0.5 { 1.0 } else { -1.0 }) + blepd(v.ph, dt) - blepd(p2, dt)
            } else { 2.0 * v.ph - 1.0 - blepd(v.ph, dt) };
            v.fenv *= v.dec;
            let fc = (v.cut * (1.0 + v.env * 8.0 * v.fenv)).clamp(30.0, fmax);
            let g = 1.0 - expd(-core::f64::consts::TAU * fc / srd);
            let k = v.k;
            let x = thd(osc * v.drive * 0.8 - k * v.y[3]);
            let t0 = thd(v.y[0]);
            v.y[0] += g * (x - t0);
            let t0 = thd(v.y[0]); let t1 = thd(v.y[1]);
            v.y[1] += g * (t0 - t1);
            let t1 = thd(v.y[1]); let t2 = thd(v.y[2]);
            v.y[2] += g * (t1 - t2);
            let t2 = thd(v.y[2]); let t3 = thd(v.y[3]);
            v.y[3] += g * (t2 - t3);
            let target = if v.gate { if v.acc { 1.35 } else { 1.0 } } else { 0.0 };
            v.amp += (target - v.amp) * if v.gate { a_atk } else { a_rel };
            let o = (v.y[3] * v.amp * v.gain * (1.0 + k * 0.35)) as f32;
            l += o; r += o;
            if !v.gate && v.amp < 1e-5 && fr > v.off + (sr as u64) { v.on = false }
        }
        d.out_l[i] += l; d.out_r[i] += r;
    }
    d.now += n as u64;
}
