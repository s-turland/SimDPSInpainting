// Engine for the GS-1 pages. Talks to `simdps app` when the page is served by
// it; otherwise search runs here and inpaint is simulated by blending the
// chosen candidates into the gap.
(function () {
  "use strict";

  const SD = (window.SD = {});
  const handlers = {};
  SD.on = (k, f) => (handlers[k] = handlers[k] || []).push(f);
  SD.emit = (k, ...a) => (handlers[k] || []).forEach((f) => f(...a));

  let ac = null;
  SD.ctx = () => ac || (ac = new (window.AudioContext || window.webkitAudioContext)());

  const S = (SD.s = {
    track: null,
    a: 0, b: 0,          // gap, seconds
    view: [0, 1],        // detail view, seconds
    cursor: 0,
    context: 2,          // seconds of context either side
    edge: "b",           // edge moved by the arrow keys
    cands: [],
    results: [],         // alternatives, each { buf, track, a, b }
    pick: 0,
    match: "latent",     // latent | audio | bands
    samples: 1,
    preset: "paper",
    method: "auto",      // auto: solo for one guide, else multi
    custom: {},          // settings changed on the display, by config section
    seed: null,
    keep: 8,             // trajectory checkpoints kept per inpaint
    from: "track",       // search the track or the bank
    live: null,          // trajectory of the running job
    busy: null,          // {what, frac}
    status: "",
  });

  // ---------- time ----------

  SD.fmt = (t, digits = 3) => {
    if (!isFinite(t)) t = 0;
    const m = Math.floor(t / 60);
    const s = t - 60 * m;
    const w = digits ? 3 + digits : 2;
    return m + ":" + s.toFixed(digits).padStart(w, "0");
  };
  SD.ms = (t) => Math.round(t * 1000) + " ms";
  SD.clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));

  SD.cli = () => {
    if (!S.track) return "";
    const name = /\s/.test(S.track.name) ? '"' + S.track.name + '"' : S.track.name;
    let s = "simdps inpaint " + name + " --gap " + SD.fmt(S.a) + " " + SD.fmt(S.b);
    if (S.preset !== "paper") s += " --config " + S.preset;
    if (S.method !== "auto") s += " --method " + S.method;
    const sets = [];
    for (const [sec, c] of Object.entries(S.custom)) for (const [k, v] of Object.entries(c)) sets.push(sec + "." + k + "=" + (v === null ? "null" : String(v)));
    return sets.length ? s + " --set " + sets.join(" ") : s;
  };

  // the ticked passages of this track, as --guide takes them
  SD.cliGuides = () => {
    const on = S.cands.filter((c) => c.on && !c.custom && c.src === S.track);
    if (!on.length) return "";
    return SD.cli() + " --guide" + on.map((c) => " " + SD.fmt(c.start)).join("");
  };

  // ---------- tracks ----------

  function makeTrack(buf, name) {
    const n = buf.length, ch = buf.numberOfChannels;
    const mono = new Float32Array(n);
    for (let c = 0; c < ch; c++) {
      const d = buf.getChannelData(c);
      for (let i = 0; i < n; i++) mono[i] += d[i] / ch;
    }
    // min/max per block of 256 samples, for drawing long spans quickly
    const B = 256, nb = Math.ceil(n / B);
    const bmin = new Float32Array(nb), bmax = new Float32Array(nb), bsq = new Float32Array(nb);
    for (let j = 0; j < nb; j++) {
      let lo = 1, hi = -1, sq = 0;
      const e = Math.min(n, (j + 1) * B);
      for (let i = j * B; i < e; i++) {
        const v = mono[i];
        if (v < lo) lo = v;
        if (v > hi) hi = v;
        sq += v * v;
      }
      bmin[j] = lo; bmax[j] = hi; bsq[j] = sq;
    }
    return { buf, name, mono, sr: buf.sampleRate, dur: buf.duration, B, bmin, bmax, bsq, feat: null };
  }
  SD.makeTrack = makeTrack;

  SD.decode = async (file) => {
    const data = await file.arrayBuffer();
    const buf = await SD.ctx().decodeAudioData(data.slice(0));
    const t = makeTrack(buf, file.name);
    t.bytes = data;
    return t;
  };

  SD.setTrack = (t) => {
    SD.stop();
    S.track = t;
    S.cands = [];
    S.results = [];
    S.pick = 0;
    S.cursor = 0;
    const L = Math.min(2, t.dur / 6);
    S.a = Math.round(10 * Math.min(t.dur * 0.45, Math.max(0, t.dur - L - 1))) / 10;
    S.b = S.a + L;
    S.view = [0, t.dur];
    S.status = "loaded " + t.name + " / " + SD.fmt(t.dur, 1) + " / " + t.sr + " Hz";
    SD.emit("track");
    SD.emit("change");
  };

  SD.loadFile = async (file) => {
    S.status = "decoding " + file.name;
    SD.emit("change");
    try {
      SD.setTrack(await SD.decode(file));
    } catch (e) {
      S.status = "could not decode " + file.name + " (try wav, flac, mp3 or ogg)";
      SD.emit("change");
    }
  };

  // ---------- peaks ----------

  // min, max and rms of the mono mix over n equal bins of [t0, t1]
  SD.peaks = (tr, t0, t1, n) => {
    const lo = new Float32Array(n), hi = new Float32Array(n), rms = new Float32Array(n);
    const sr = tr.sr, N = tr.mono.length;
    const per = ((t1 - t0) * sr) / n;
    for (let k = 0; k < n; k++) {
      let s = Math.floor(t0 * sr + k * per), e = Math.floor(t0 * sr + (k + 1) * per);
      s = SD.clamp(s, 0, N); e = SD.clamp(Math.max(e, s + 1), 0, N);
      let mn = 0, mx = 0, sq = 0, cnt = 0;
      if (e - s > tr.B * 4) {
        const bs = Math.floor(s / tr.B), be = Math.ceil(e / tr.B);
        for (let j = bs; j < be; j++) {
          if (tr.bmin[j] < mn) mn = tr.bmin[j];
          if (tr.bmax[j] > mx) mx = tr.bmax[j];
          sq += tr.bsq[j];
        }
        cnt = (be - bs) * tr.B;
      } else {
        for (let i = s; i < e; i++) {
          const v = tr.mono[i];
          if (v < mn) mn = v;
          if (v > mx) mx = v;
          sq += v * v;
        }
        cnt = e - s;
      }
      lo[k] = mn; hi[k] = mx; rms[k] = cnt ? Math.sqrt(sq / cnt) : 0;
    }
    return { lo, hi, rms };
  };

  SD.peaksOf = (buf, n) => SD.peaks(makeTrack(buf, ""), 0, buf.duration, n);

  // ---------- spectral features for search ----------

  const NFFT = 1024, NB = 24;
  let fftTables = null;
  function fftInit() {
    const rev = new Uint16Array(NFFT), bits = Math.log2(NFFT);
    for (let i = 0; i < NFFT; i++) {
      let r = 0;
      for (let b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b);
      rev[i] = r;
    }
    const cos = new Float64Array(NFFT / 2), sin = new Float64Array(NFFT / 2), win = new Float64Array(NFFT);
    for (let i = 0; i < NFFT / 2; i++) { cos[i] = Math.cos((2 * Math.PI * i) / NFFT); sin[i] = -Math.sin((2 * Math.PI * i) / NFFT); }
    for (let i = 0; i < NFFT; i++) win[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / NFFT);
    fftTables = { rev, cos, sin, win };
  }
  function fft(re, im) {
    const { rev, cos, sin } = fftTables;
    for (let i = 0; i < NFFT; i++) {
      const j = rev[i];
      if (j > i) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
    }
    for (let size = 2; size <= NFFT; size <<= 1) {
      const half = size >> 1, step = NFFT / size;
      for (let i = 0; i < NFFT; i += size) {
        for (let j = 0; j < half; j++) {
          const k = j * step, a = i + j, b = a + half;
          const tr = re[b] * cos[k] - im[b] * sin[k];
          const ti = re[b] * sin[k] + im[b] * cos[k];
          re[b] = re[a] - tr; im[b] = im[a] - ti;
          re[a] += tr; im[a] += ti;
        }
      }
    }
  }

  // log band energies, one frame per NFFT samples
  function features(tr) {
    if (tr.feat) return tr.feat;
    if (!fftTables) fftInit();
    const n = Math.floor(tr.mono.length / NFFT);
    const data = new Float32Array(n * NB);
    const edges = [];
    for (let b = 0; b <= NB; b++) {
      const f = 40 * Math.pow(tr.sr / 2 / 40, b / NB);
      edges.push(Math.max(1, Math.min(NFFT / 2, Math.round((f / tr.sr) * NFFT))));
    }
    const re = new Float64Array(NFFT), im = new Float64Array(NFFT), win = fftTables.win;
    for (let f = 0; f < n; f++) {
      const o = f * NFFT;
      for (let i = 0; i < NFFT; i++) { re[i] = tr.mono[o + i] * win[i]; im[i] = 0; }
      fft(re, im);
      for (let b = 0; b < NB; b++) {
        let e = 0;
        for (let k = edges[b]; k < Math.max(edges[b] + 1, edges[b + 1]); k++) e += re[k] * re[k] + im[k] * im[k];
        data[f * NB + b] = Math.log10(1e-8 + e);
      }
    }
    tr.feat = { data, n, hop: NFFT / tr.sr };
    return tr.feat;
  }

  // mean L1 distance between the frames around the gap and the frames around
  // a candidate window starting at frame fs of src
  function contextDistance(tf, fa, fb, sf, fs, cf) {
    const L = fb - fa;
    let d = 0, cnt = 0;
    for (let k = 0; k < cf; k++) {
      const t1 = fa - cf + k, s1 = fs - cf + k;
      if (t1 >= 0 && s1 >= 0) {
        for (let b = 0; b < NB; b++) d += Math.abs(tf.data[t1 * NB + b] - sf.data[s1 * NB + b]);
        cnt++;
      }
      const t2 = fb + k, s2 = fs + L + k;
      if (t2 < tf.n && s2 < sf.n) {
        for (let b = 0; b < NB; b++) d += Math.abs(tf.data[t2 * NB + b] - sf.data[s2 * NB + b]);
        cnt++;
      }
    }
    return cnt ? d / cnt : Infinity;
  }

  let distScale = 1, distFloor = 0; // from the last search, so scores are comparable

  function frames(tf) {
    return {
      fa: Math.round(S.a / tf.hop),
      fb: Math.round(S.b / tf.hop),
      cf: Math.max(1, Math.round(S.context / tf.hop)),
    };
  }

  function scoreAll(src) {
    const tf = features(S.track), sf = features(src);
    const { fa, fb, cf } = frames(tf);
    const L = fb - fa, out = [];
    for (let fs = 0; fs + L <= sf.n; fs++) {
      if (src === S.track && fs < fb + cf && fs + L > fa - cf) continue;
      out.push({ fs, d: contextDistance(tf, fa, fb, sf, fs, cf) });
    }
    return { out, hop: sf.hop, L };
  }

  // 0.5 at the median window, approaching 1 as the context match improves
  const sim = (d) => Math.min(1, Math.exp((-Math.LN2 * Math.max(0, d - distFloor)) / Math.max(1e-6, distScale - distFloor)));

  // srcs: tracks to search, the loaded one by default
  const localSearch = async (k, srcs = [S.track]) => {
    const all = [];
    let L = 1, hop = 1;
    for (const src of srcs) {
      const r = scoreAll(src);
      L = r.L; hop = r.hop;
      r.out.forEach((o) => all.push(Object.assign(o, { src })));
    }
    if (!all.length) throw new Error(srcs[0] === S.track ? "track too short to search outside the gap" : "bank signals too short for this gap");
    if (srcs[0] === S.track) {
      const ds = all.map((o) => o.d).sort((x, y) => x - y);
      distScale = ds[Math.floor(ds.length / 2)] || 1;
      distFloor = 0.5 * ds[0];
    }
    all.sort((x, y) => x.d - y.d);
    const minSep = Math.max(L / 2, Math.round(1 / hop));
    const picked = [];
    for (const o of all) {
      if (picked.every((p) => p.src !== o.src || Math.abs(p.fs - o.fs) >= minSep)) picked.push(o);
      if (picked.length >= k) break;
    }
    return { list: picked.map((o) => ({ start: o.fs * hop, score: sim(o.d), src: o.src })), windows: all.length };
  };

  SD.search = async (k = 8) => {
    if (!S.track || S.busy) return;
    const remote = SD.server && SD.server.ready;
    S.busy = { what: "search", frac: 0 };
    S.status = remote ? "searching (" + S.match + ")" : "analysing spectrum";
    SD.emit("change");
    await new Promise((r) => setTimeout(r, 30));
    let found;
    const bank = S.from === "bank";
    try {
      if (bank && !SD.bank.files.length) throw new Error("the bank is empty");
      if (remote) {
        const id = await upload(S.track);
        const r = await api("/api/search", { track: id, gap: [S.a, S.b], match: S.match, n: k, source: bank ? "bank" : null });
        for (const o of r.candidates) {
          const e = o.source && SD.bank.files.find((f) => f.id === o.source);
          o.src = e ? await SD.bankTrack(e) : S.track;
        }
        found = { list: r.candidates, windows: r.windows };
      } else {
        const srcs = bank ? await Promise.all(SD.bank.files.map(SD.bankTrack)) : [S.track];
        found = await localSearch(k, srcs);
      }
    } catch (e) {
      S.busy = null; S.status = "search failed: " + e.message; SD.emit("change"); return;
    }
    const keep = S.cands.filter((c) => c.custom);
    S.cands = found.list
      .map((o, i) => ({ id: "c" + (i + 1), src: o.src, srcName: o.src.name, start: o.start, sim: o.score, on: i < 3, custom: false, wt: null, bank: o.src !== S.track }))
      .concat(keep);
    S.busy = null;
    S.status = "search" + (bank ? " of the bank" : "") + ": " + found.list.length + " candidates" + (found.windows ? " from " + found.windows + " windows" : "");
    SD.emit("cands");
    SD.emit("change");
  };

  // Match at every start position of src: what copying from there would
  // score. Recomputed shortly after the gap stops changing; until then the
  // last curve is returned with stale = true.
  const curves = new Map();
  let curveTimer = null;
  const curveKey = () => S.a + ":" + S.b + ":" + S.context;
  function computeCurve(src) {
    const { out, hop } = scoreAll(src);
    if (src === S.track && out.length) {
      const ds = out.map((o) => o.d).sort((x, y) => x - y);
      distScale = ds[Math.floor(ds.length / 2)] || 1;
      distFloor = 0.5 * ds[0];
    }
    const sims = new Float32Array(features(src).n).fill(NaN);
    let lo = 1, hi = 0;
    out.forEach((o) => { const v = sim(o.d); sims[o.fs] = v; lo = Math.min(lo, v); hi = Math.max(hi, v); });
    return { hop, sims, lo, hi: Math.max(hi, lo + 1e-3), key: curveKey() };
  }
  SD.curve = (src) => {
    if (!S.track || !src) return null;
    const c = curves.get(src);
    if (c && c.key === curveKey()) return c;
    if (!c) {
      if (src !== S.track) SD.curve(S.track);
      const n = computeCurve(src);
      curves.set(src, n);
      return n;
    }
    clearTimeout(curveTimer);
    curveTimer = setTimeout(() => {
      if (src !== S.track) curves.set(S.track, computeCurve(S.track));
      curves.set(src, computeCurve(src));
      if (src === S.track) for (const k of curves.keys()) if (k !== S.track) curves.set(k, computeCurve(k));
      SD.redraw();
    }, 180);
    return Object.assign({}, c, { stale: true });
  };
  SD.curveAt = (src, t) => {
    const c = SD.curve(src);
    if (!c) return NaN;
    const v = c.sims[Math.round(t / c.hop)];
    return v === undefined ? NaN : v;
  };

  // best window in an uploaded file
  SD.bestStart = (src) => {
    const { out, hop } = scoreAll(src);
    if (!out.length) return 0;
    out.sort((x, y) => x.d - y.d);
    return out[0].fs * hop;
  };

  SD.customSim = (src, start) => {
    const tf = features(S.track), sf = features(src);
    const { fa, fb, cf } = frames(tf);
    return sim(contextDistance(tf, fa, fb, sf, Math.round(start / sf.hop), cf));
  };

  let customN = 0;
  SD.addCustom = (src, start) => {
    SD.curve(S.track);
    const c = { id: "u" + ++customN, src, srcName: src.name, start, sim: SD.customSim(src, start), on: true, custom: true, wt: null };
    S.cands.push(c);
    S.status = "added " + src.name + " @ " + SD.fmt(start) + " to the pool";
    SD.emit("cands");
    SD.emit("change");
    return c;
  };

  SD.removeCand = (c) => {
    S.cands = S.cands.filter((x) => x !== c);
    SD.emit("cands");
    SD.emit("change");
  };

  // Shares of the guidance. Automatic shares are the softmax of the scores,
  // as in the model; a manual share (c.wt, 0..1) is kept and the rest split
  // what is left.
  SD.weights = () => {
    const on = S.cands.filter((c) => c.on), w = new Map();
    S.cands.forEach((c) => w.set(c, 0));
    if (!on.length) return w;
    const fixed = on.filter((c) => c.wt != null), free = on.filter((c) => c.wt == null);
    let used = fixed.reduce((s, c) => s + c.wt, 0);
    const scale = used > 1 || !free.length ? 1 / (used || 1) : 1;
    fixed.forEach((c) => w.set(c, c.wt * scale));
    used = Math.min(1, used * scale);
    const ex = free.map((c) => Math.exp(c.sim)), tot = ex.reduce((s, v) => s + v, 0) || 1;
    free.forEach((c, i) => w.set(c, ((1 - used) * ex[i]) / tot));
    return w;
  };
  SD.setWeight = (c, v) => {
    c.wt = v == null ? null : SD.clamp(v, 0, 1);
    SD.emit("change");
  };

  // ---------- buffers ----------

  function newBuf(ch, n, sr) {
    return SD.ctx().createBuffer(ch, Math.max(1, n), sr);
  }

  SD.slice = (buf, t0, t1) => {
    const sr = buf.sampleRate, s = Math.max(0, Math.round(t0 * sr)), e = Math.min(buf.length, Math.round(t1 * sr));
    const out = newBuf(buf.numberOfChannels, e - s, sr);
    for (let c = 0; c < buf.numberOfChannels; c++) out.copyToChannel(buf.getChannelData(c).subarray(s, e), c);
    return out;
  };

  const gapLen = () => Math.round(S.b * S.track.sr) - Math.round(S.a * S.track.sr);

  SD.candBuf = (c) => {
    if (c._buf && c._key === S.a + ":" + S.b) return c._buf;
    const sr = c.src.sr, n = gapLen(), s = Math.round(c.start * sr);
    const out = newBuf(S.track.buf.numberOfChannels, n, sr);
    for (let ch = 0; ch < out.numberOfChannels; ch++) {
      const d = c.src.buf.getChannelData(Math.min(ch, c.src.buf.numberOfChannels - 1));
      out.copyToChannel(d.subarray(s, Math.min(d.length, s + n)), ch);
    }
    c._buf = out; c._key = S.a + ":" + S.b;
    return out;
  };

  // context + fill + context, crossfaded, as one buffer starting at a - context
  function splice(fill, xf = 0.03) {
    const tr = S.track, sr = tr.sr, ch = tr.buf.numberOfChannels;
    const t0 = Math.max(0, S.a - S.context), t1 = Math.min(tr.dur, S.b + S.context);
    const s0 = Math.round(t0 * sr), sa = Math.round(S.a * sr), sb = Math.round(S.b * sr), s1 = Math.round(t1 * sr);
    const out = newBuf(ch, s1 - s0, sr), nx = Math.round(xf * sr);
    for (let c = 0; c < ch; c++) {
      const src = tr.buf.getChannelData(c), dst = out.getChannelData(c);
      const f = fill ? fill[Math.min(c, fill.length - 1)] : null;
      for (let i = s0; i < s1; i++) {
        let g = 0; // weight of the fill
        if (i >= sa - nx / 2 && i < sb + nx / 2) {
          if (i < sa + nx / 2) g = (i - (sa - nx / 2)) / nx;
          else if (i >= sb - nx / 2) g = 1 - (i - (sb - nx / 2)) / nx;
          else g = 1;
        }
        g = SD.clamp(g, 0, 1);
        const gf = Math.sin((g * Math.PI) / 2), go = Math.cos((g * Math.PI) / 2);
        const fi = i - sa;
        const fv = f && fi >= 0 && fi < f.length ? f[fi] : 0;
        dst[i - s0] = go * src[i] + gf * fv;
      }
    }
    out.offset = t0;
    return out;
  }

  SD.preview = (c) => {
    const b = SD.candBuf(c), fill = [];
    for (let ch = 0; ch < b.numberOfChannels; ch++) fill.push(b.getChannelData(ch));
    return splice(fill);
  };

  // ---------- inpaint ----------

  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const stemOf = (s) => s.replace(/\.[^.]+$/, "");
  let resN = 0, layoutN = 0;

  // simulated: the chosen candidates blended into the gap, level-matched
  function simFill(on, w) {
    const n = gapLen(), ch = S.track.buf.numberOfChannels, fill = [];
    for (let c = 0; c < ch; c++) fill.push(new Float32Array(n));
    for (const c of on) {
      const b = SD.candBuf(c), wc = w[on.indexOf(c)];
      for (let k = 0; k < ch; k++) {
        const d = b.getChannelData(Math.min(k, b.numberOfChannels - 1));
        for (let i = 0; i < n; i++) fill[k][i] += wc * d[i];
      }
    }
    const rms = (arr) => Math.sqrt(arr.reduce((s, v) => s + v * v, 0) / Math.max(1, arr.length));
    const sr = S.track.sr, sa = Math.round(S.a * sr), sb = Math.round(S.b * sr), cs = Math.round(S.context * sr);
    const m = S.track.mono;
    const ctxR = (rms(m.subarray(Math.max(0, sa - cs), sa)) + rms(m.subarray(sb, Math.min(m.length, sb + cs)))) / 2;
    const g = SD.clamp(ctxR / (rms(fill[0]) || 1), 0.25, 4);
    fill.forEach((f) => f.forEach((v, i) => (f[i] = v * g)));
    return fill;
  }
  // the track with fill crossfaded in; noise (0..1) is mixed in for checkpoints
  function simFull(fill, a, b, noise = 0) {
    const sr = S.track.sr, ch = S.track.buf.numberOfChannels;
    let f = fill;
    if (noise > 0) {
      const lvl = Math.sqrt(fill[0].reduce((s, v) => s + v * v, 0) / Math.max(1, fill[0].length)) * 1.7;
      f = fill.map((x) => x.map((v) => (1 - noise) * v + noise * lvl * (Math.random() * 2 - 1)));
    }
    const keepA = S.a, keepB = S.b;
    S.a = a; S.b = b;
    const local = splice(f, 0.05);
    S.a = keepA; S.b = keepB;
    const out = newBuf(ch, S.track.buf.length, sr), off = Math.round(local.offset * sr);
    for (let k = 0; k < ch; k++) {
      const d = out.getChannelData(k);
      d.set(S.track.buf.getChannelData(k));
      d.set(local.getChannelData(k), off);
    }
    return out;
  }
  // a made-up path from near the heaviest guide to the weighted middle
  function simPath(guides, w, steps, first, prefix) {
    const pts = prefix ? prefix.map((p) => p.slice()) : [];
    const tx = guides.reduce((s, g, j) => s + w[j] * g[0], 0), ty = guides.reduce((s, g, j) => s + w[j] * g[1], 0);
    const heavy = guides[w.indexOf(Math.max(...w))];
    let x = pts.length ? pts[pts.length - 1][1] : heavy[0] + (Math.random() - 0.5) * 0.5;
    let y = pts.length ? pts[pts.length - 1][2] : heavy[1] + (Math.random() - 0.5) * 0.5;
    for (let s = first + 1; s <= steps; s++) {
      const f = 1 - s / steps, k = 0.06 + 0.2 * (1 - f);
      x += (tx - x) * k + (Math.random() - 0.5) * 0.3 * f;
      y += (ty - y) * k + (Math.random() - 0.5) * 0.3 * f;
      pts.push([s, x, y, 1.4 * Math.pow(f, 1.5) + 0.03]);
    }
    return pts;
  }
  const simKept = (steps, keep) => {
    const out = [];
    for (let j = 1; j <= keep; j++) {
      const s = Math.round((j * steps) / (keep + 1));
      if (s > 0 && s < steps && !out.some((c) => c.step === s)) out.push({ step: s, sigma: 2 * Math.pow(1 - s / steps, 3) });
    }
    return out;
  };

  async function simJob(job) {
    const steps = (job.custom && job.custom.sampler && job.custom.sampler.steps) ||
      { paper: 100, warm_start_fast: 35, warm_start_detailed: 40 }[job.preset] || 100, first = job.first || 0;
    const on = job.on, k = on.length;
    const guides = job.guides || on.map((c, j) => {
      const r = 0.4 + 0.6 * (1 - SD.clamp(c.sim, 0, 1)), t = (2 * Math.PI * j) / k + 0.4;
      return k === 1 ? [0, 0] : [r * Math.cos(t), r * Math.sin(t)];
    });
    const paths = [], fills = [];
    for (let s = 0; s < job.samples; s++) {
      // alternatives: the same blend with jittered shares
      let w = job.w.map((v) => v * (s || first ? 0.4 + 1.2 * Math.random() : 1));
      const tot = w.reduce((x, y) => x + y, 0) || 1;
      w = w.map((v) => v / tot);
      // variations keep most of what the checkpoint had
      const fresh = simFill(on, w);
      fills.push(job.fill ? fresh.map((f, c) => f.map((v, i) => 0.6 * job.fill[Math.min(c, job.fill.length - 1)][i] + 0.4 * v)) : fresh);
      paths.push(simPath(guides, w, steps, first, job.prefix));
    }
    for (let i = first; i <= steps; i++) {
      S.busy = { what: "inpaint", frac: (i - first) / Math.max(1, steps - first), step: i, steps };
      S.live = Object.assign(S.live || {}, { guides, paths: paths.map((p) => p.filter((q) => q[0] <= i)), steps, first });
      S.status = "sampling step " + i + "/" + steps + "  (simulated)";
      SD.emit("change");
      await sleep(40);
    }
    return {
      bufs: fills.map((f) => simFull(f, job.a, job.b)), steps, first, guides, paths,
      cks: simKept(steps, job.keep).filter((c) => c.step > first), sim: fills,
    };
  }

  async function serverJob(body) {
    const job = await api("/api/inpaint", body);
    S.busy = { what: "inpaint", frac: 0, step: 0, steps: 0 };
    for (;;) {
      await sleep(250);
      const st = await api("/api/job/" + job.id);
      const span = Math.max(1, st.steps - st.first);
      S.busy = { what: "inpaint", frac: SD.clamp((st.step - st.first) / span, 0, 1), step: st.step, steps: st.steps, phase: st.phase };
      S.live = Object.assign(S.live || {}, { guides: st.trace.guides, paths: st.trace.paths, steps: st.steps, first: st.first });
      S.status = st.phase === "sampling" ? "sampling step " + st.step + "/" + st.steps : st.phase || "working";
      SD.emit("change");
      if (!st.done) continue;
      if (st.error) throw new Error(st.error);
      const bufs = [];
      for (const id of st.results) bufs.push(await SD.ctx().decodeAudioData(await (await fetch("/api/result/" + id)).arrayBuffer()));
      return { bufs, steps: st.steps, first: st.first, guides: st.trace.guides, paths: st.trace.paths, cks: st.checkpoints, id: job.id };
    }
  }

  // Each checkpoint knows the job and sample it was kept from, so the ones a
  // variation inherits are the same objects as its parent's: heard, saved and
  // varied from as they were.
  function addResults(out, base, inherit = []) {
    const name = stemOf(S.track.name) + "_simdps";
    return out.bufs.map((buf, i) => {
      const label = String(++resN);
      const own = out.cks.map((c, k) => ({ step: c.step, sigma: c.sigma, job: out.id, index: k, sample: i, owner: label, sim: out.sim && out.sim[i] }));
      const cks = inherit.concat(own);
      return Object.assign({
        buf, track: makeTrack(buf, name + "_" + label + ".wav"), label, job: out.id, sample: i,
        steps: out.steps, first: out.first, guides: out.guides, path: out.paths[i] || [],
        cks, ck: cks.length,
      }, base);
    });
  }

  SD.inpaint = async () => {
    if (!S.track || S.busy) return;
    const on = S.cands.filter((c) => c.on);
    if (!on.length) {
      S.status = "select at least one candidate, or run search first";
      SD.emit("change");
      return;
    }
    SD.stop();
    const a = S.a, b = S.b, wm = SD.weights(), w = on.map((c) => wm.get(c));
    const nums = on.map((c) => S.cands.indexOf(c) + 1), remote = SD.server && SD.server.ready;
    S.live = { guides: [], paths: [], nums, w, steps: 0, first: 0, fresh: true };
    let out;
    try {
      if (remote) {
        const guides = [];
        for (let i = 0; i < on.length; i++) guides.push({ source: await upload(on[i].src), start: on[i].start, weight: w[i] });
        out = await serverJob({
          track: await upload(S.track), gap: [a, b], guides, nums, samples: S.samples, preset: S.preset,
          method: S.method === "auto" ? null : S.method, settings: S.custom, seed: S.seed, checkpoints: S.keep,
        });
      } else out = await simJob({ on, w, nums, a, b, samples: S.samples, preset: S.preset, keep: S.keep, custom: S.custom });
    } catch (e) {
      S.busy = null; S.live = null; S.status = "inpaint failed: " + e.message; SD.emit("change"); return;
    }
    resN = 0;
    S.results = addResults(out, { a, b, from: on.map((c) => c.id), nums, on, w, preset: S.preset, layout: ++layoutN });
    S.pick = 0;
    S.busy = null; S.live = null;
    S.status = "inpainted " + SD.fmt(a) + " - " + SD.fmt(b) + " from " + on.length + " guide" + (on.length > 1 ? "s" : "") +
      (out.bufs.length > 1 ? ", " + out.bufs.length + " alternatives" : "") + (remote ? "" : " (simulated)");
    SD.emit("result");
    SD.emit("change");
    SD.prefetch();
  };
  SD.result = () => S.results[S.pick] || null;

  // ---------- settings ----------

  // the settings of the next run as the server resolves them, before the
  // changes made on the display; null until known, and without a server
  const baseCache = new Map();
  SD.methodNow = () => (S.method !== "auto" ? S.method : S.cands.filter((c) => c.on).length === 1 ? "solo" : "multi");
  SD.baseSettings = () => {
    if (!(SD.server && SD.server.ready)) return null;
    const k = S.preset + ":" + SD.methodNow();
    if (baseCache.has(k)) return baseCache.get(k);
    baseCache.set(k, null);
    api("/api/settings?preset=" + S.preset + "&method=" + SD.methodNow())
      .then((b) => { baseCache.set(k, b); SD.emit("change"); })
      .catch(() => baseCache.delete(k));
    return null;
  };
  // v undefined: back to the preset's value
  SD.setCustom = (sec, k, v) => {
    const c = S.custom[sec] || (S.custom[sec] = {});
    if (v === undefined) delete c[k]; else c[k] = v;
    if (!Object.keys(c).length) delete S.custom[sec];
    SD.emit("change");
  };
  SD.customCount = () => Object.values(S.custom).reduce((n, c) => n + Object.keys(c).length, 0);

  // New fills from checkpoint j of res, with fresh seeds; kept as further
  // alternatives. pool: guide them with what is ticked in the pool now,
  // instead of the run's own guides.
  SD.vary = async (res, j, samples, seed, pool = false) => {
    if (S.busy || !res || !res.cks[j]) return;
    const c = res.cks[j], remote = SD.server && SD.server.ready && c.job;
    const on = pool ? S.cands.filter((x) => x.on) : res.on;
    if (!on.length) { S.status = "tick candidates in the pool to vary with them"; SD.emit("change"); return; }
    SD.stop();
    const wm = SD.weights(), w = pool ? on.map((x) => wm.get(x)) : res.w;
    const nums = pool ? on.map((x) => S.cands.indexOf(x) + 1) : res.nums;
    S.live = { guides: pool ? [] : res.guides, paths: [], nums, w, steps: res.steps, first: c.step, fresh: pool };
    let out;
    try {
      if (remote) {
        const body = { restart: { job: c.job, index: c.index, sample: c.sample }, samples, seed, nums };
        if (pool) {
          body.guides = [];
          for (let i = 0; i < on.length; i++) body.guides.push({ source: await upload(on[i].src), start: on[i].start, weight: w[i] });
        }
        out = await serverJob(body);
      } else out = await simJob({ on, w, nums, a: res.a, b: res.b, samples, preset: res.preset, keep: S.keep,
        first: c.step, guides: pool ? null : res.guides, prefix: pool ? null : res.path.filter((p) => p[0] <= c.step), fill: c.sim });
    } catch (e) {
      S.busy = null; S.live = null; S.status = "variation failed: " + e.message; SD.emit("change"); return;
    }
    const added = addResults(out, { a: res.a, b: res.b, from: pool ? on.map((x) => x.id) : res.from, nums, on, w,
      preset: res.preset, layout: pool ? ++layoutN : res.layout,
      parent: { label: res.label, ck: j, step: c.step, pool } }, res.cks.slice(0, j + 1));
    S.results = S.results.concat(added).slice(-8);
    S.pick = S.results.indexOf(added[0]);
    S.busy = null; S.live = null;
    S.status = added.length + " variation" + (added.length > 1 ? "s" : "") + " of " + res.label + " from checkpoint " + (j + 1) +
      " (step " + c.step + "/" + res.steps + ")" + (remote ? "" : " (simulated)");
    SD.emit("result");
    SD.emit("change");
    SD.prefetch();
  };

  // audio of checkpoint j of res, decoded on the server when first asked for
  SD.ckBuf = (res, j) => {
    const c = res.cks[j];
    if (!c) return Promise.resolve(res.buf);
    if (c.buf) return Promise.resolve(c.buf);
    if (!c.loading) {
      c.loading = (async () => {
        if (c.sim) c.buf = simFull(c.sim, res.a, res.b, SD.clamp(c.sigma / 2, 0, 0.95));
        else c.buf = await SD.ctx().decodeAudioData(await (await fetch("/api/checkpoint/" + c.job + "/" + c.index + "/" + c.sample)).arrayBuffer());
        c.track = makeTrack(c.buf, stemOf(res.track.name) + "_c" + (j + 1) + ".wav");
        SD.emit("change");
        return c.buf;
      })().catch((e) => { c.loading = null; throw e; });
    }
    return c.loading;
  };
  // checkpoints of the picked result, one at a time, while nothing else runs
  SD.prefetch = async () => {
    const res = SD.result();
    if (!res) return;
    for (let j = 0; j < res.cks.length; j++) {
      if (S.busy || SD.result() !== res) return;
      try { await SD.ckBuf(res, j); } catch (e) { return; }
    }
  };

  // ---------- bank ----------

  // a folder on the server (simdps app --bank); in memory without one
  SD.bank = { folder: null, files: [] };
  SD.bankLoad = async () => {
    if (!SD.server) return;
    try {
      const r = await api("/api/bank");
      const old = new Map(SD.bank.files.map((e) => [e.id, e]));
      SD.bank = { folder: r.folder, files: r.files.map((e) => Object.assign(old.get(e.id) || {}, e)) };
    } catch (e) { S.status = "could not read the bank: " + e.message; }
    SD.emit("change");
  };
  // keep [t0, t1) of buf; at: where in it the part lined up with a gap starts
  SD.bankAdd = async (buf, t0, t1, at, length, name, kind, origin) => {
    t0 = Math.max(0, t0); t1 = Math.min(buf.duration, t1);
    const part = SD.slice(buf, t0, t1), q = (k, v) => k + "=" + encodeURIComponent(v);
    try {
      if (SD.server) {
        const r = await fetch("/api/bank?" + [q("name", name), q("kind", kind), q("origin", origin), q("at", (at - t0).toFixed(4)), q("length", length.toFixed(4))].join("&"),
          { method: "POST", body: await SD.wav(part).arrayBuffer() });
        const j = await r.json();
        if (!r.ok || j.error) throw new Error(j.error || r.statusText);
        await SD.bankLoad();
        S.status = "banked " + j.file + (SD.bank.folder ? " in " + SD.bank.folder : "");
      } else {
        const file = stemOf(name) + "_" + String(SD.bank.files.length + 1).padStart(2, "0") + ".wav";
        const track = makeTrack(part, file);
        SD.bank.files.push({ file, id: "mem:" + file, kind, origin, at: at - t0, length, dur: part.duration, track });
        S.status = "banked " + file + " (until the page is closed)";
      }
    } catch (e) { S.status = "could not bank: " + e.message; }
    SD.emit("change");
  };
  SD.bankTrack = async (e) => {
    if (e.track) return e.track;
    const data = await (await fetch("/api/bank/" + encodeURIComponent(e.file))).arrayBuffer();
    const t = makeTrack(await SD.ctx().decodeAudioData(data.slice(0)), e.file);
    t.serverId = e.id;
    e.track = t;
    return t;
  };
  SD.bankDrop = async (e) => {
    if (SD.server && !e.id.startsWith("mem:")) {
      try { await api("/api/bank/" + encodeURIComponent(e.file) + "/drop", {}); } catch (err) { S.status = "could not drop: " + err.message; }
      await SD.bankLoad();
      S.status = "moved " + e.file + " to " + (SD.bank.folder || "the bank") + "/dropped";
    } else SD.bank.files = SD.bank.files.filter((x) => x !== e);
    SD.emit("change");
  };

  // ---------- server ----------

  SD.server = null; // { ready, model, sample_rate, max_gap, error, bank }

  async function api(path, body) {
    const r = await fetch(path, body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const j = await r.json().catch(() => ({ error: r.statusText }));
    if (!r.ok || j.error) throw new Error(j.error || r.statusText);
    return j;
  }
  SD.api = api;

  async function upload(tr) {
    if (tr.serverId) return tr.serverId;
    const body = tr.bytes || (await SD.wav(tr.buf).arrayBuffer());
    const r = await fetch("/api/audio?name=" + encodeURIComponent(tr.name), { method: "POST", body });
    const j = await r.json();
    if (!r.ok || j.error) throw new Error(j.error || r.statusText);
    tr.serverId = j.id;
    return j.id;
  }

  SD.connect = async () => {
    try {
      const r = await fetch("/api/info");
      if (!r.ok) throw new Error();
      SD.server = await r.json();
    } catch (e) {
      SD.server = null;
      S.match = "bands";
      SD.emit("change");
      return;
    }
    SD.bankLoad();
    // the model loads in the background
    while (SD.server && !SD.server.ready && !SD.server.error) {
      SD.emit("change");
      await new Promise((r) => setTimeout(r, 1000));
      try { SD.server = await (await fetch("/api/info")).json(); } catch (e) { break; }
    }
    SD.emit("change");
  };

  // ---------- playback ----------

  let P = null; // {src, t0, from, to, loop, tag, offset}
  SD.playing = () => P;

  // rebuild(): { buf, from, to, offset } for the current selection, so that
  // SD.refresh() can follow changes; at: where in the buffer to start
  SD.play = (buf, from, to, loop, tag, offset = 0, rebuild = null, at = null) => {
    SD.stop(true);
    const c = SD.ctx();
    if (c.state === "suspended") c.resume();
    from = SD.clamp(from, 0, buf.duration);
    to = SD.clamp(to, from + 0.01, buf.duration);
    at = at == null ? from : SD.clamp(at, from, to - 0.005);
    const src = c.createBufferSource();
    src.buffer = buf;
    src.connect(c.destination);
    if (loop) {
      src.loop = true; src.loopStart = from; src.loopEnd = to;
      src.start(0, at);
    } else src.start(0, at, to - at);
    P = { src, t0: c.currentTime - (at - from), from, to, loop, tag, offset, buf, rebuild };
    src.onended = () => {
      if (P && P.src === src) { P = null; SD.emit("change"); }
    };
    SD.emit("change");
  };

  SD.stop = (quiet) => {
    if (P) {
      const p = P;
      P = null;
      try { p.src.stop(); } catch (e) {}
      if (!quiet) S.cursor = SD.pos(p) ?? S.cursor;
    }
    if (!quiet) SD.emit("change");
  };

  // playhead in track time
  SD.pos = (p = P) => {
    if (!p) return null;
    const el = SD.ctx().currentTime - p.t0, len = p.to - p.from;
    const t = p.loop ? p.from + (el % len) : Math.min(p.to, p.from + el);
    return t + p.offset;
  };

  const ctxRange = () => [Math.max(0, S.a - S.context), Math.min(S.track.dur, S.b + S.context)];

  SD.toggle = () => {
    if (!S.track) return;
    if (P) return SD.stop();
    SD.play(S.track.buf, S.cursor >= S.track.dur - 0.05 ? 0 : S.cursor, S.track.dur, false, "track");
  };
  SD.playWith = (tag, rebuild) => {
    const r = rebuild();
    SD.play(r.buf, r.from, r.to, true, tag, r.offset, rebuild);
  };
  SD.loopGap = () => {
    if (!S.track) return;
    if (P && P.tag === "gap") return SD.stop();
    SD.playWith("gap", () => ({ buf: S.track.buf, from: S.a, to: S.b, offset: 0 }));
  };
  SD.loopContext = () => {
    if (!S.track) return;
    if (P && P.tag === "ctx") return SD.stop();
    SD.playWith("ctx", () => { const [t0, t1] = ctxRange(); return { buf: S.track.buf, from: t0, to: t1, offset: 0 }; });
  };
  SD.playCand = (c, inPlace = true) => {
    const tag = "cand:" + c.id + (inPlace ? ":in" : ":raw");
    if (P && P.tag === tag) return SD.stop();
    if (inPlace) SD.playWith(tag, () => { const b = SD.preview(c); return { buf: b, from: 0, to: b.duration, offset: b.offset }; });
    // raw candidate, shown at its own position
    else SD.playWith(tag, () => ({ buf: c.src.buf, from: c.start, to: c.start + (S.b - S.a), offset: 0 }));
  };

  // Rebuild what is playing after the selection changed, from the same place.
  let refreshTimer = null;
  SD.refresh = () => {
    clearTimeout(refreshTimer);
    refreshTimer = setTimeout(() => {
      if (!P || !P.rebuild) return;
      const t = SD.pos(), r = P.rebuild(), lt = t - r.offset;
      SD.play(r.buf, r.from, r.to, true, P.tag, r.offset, P.rebuild, lt >= r.from && lt < r.to ? lt : r.from);
    }, 90);
  };
  // result, looped around the gap; A/B keeps the position, and the result
  // follows the picked alternative while it plays
  SD.playResult = (which = "result") => {
    if (!SD.result()) return;
    const tag = "ab:" + which;
    if (P && P.tag === tag) return SD.stop();
    const rebuild = () => {
      const [t0, t1] = ctxRange();
      return { buf: which === "result" ? SD.result().buf : S.track.buf, from: t0, to: t1, offset: 0 };
    };
    const r = rebuild(), at = P && P.tag.startsWith("ab:") ? SD.pos() : null;
    SD.play(r.buf, r.from, r.to, true, tag, 0, rebuild, at);
  };

  // checkpoint res.ck of res (res.cks.length is the finished result), looped
  // around the gap; follows the selection while it plays
  const ckRange = (res) => [Math.max(0, res.a - S.context), Math.min(res.track.dur, res.b + S.context)];
  const ckNow = (res) => { const [t0, t1] = ckRange(res), c = res.cks[res.ck]; return { buf: (c && c.buf) || res.buf, from: t0, to: t1, offset: 0 }; };
  SD.playCk = async (res) => {
    if (P && P.tag === "ck") return SD.stop();
    const c = res.cks[res.ck];
    if (c && !c.buf) { S.status = "decoding checkpoint " + (res.ck + 1); SD.emit("change"); await SD.ckBuf(res, res.ck); }
    SD.playWith("ck", () => ckNow(SD.result()));
  };
  // pick alternative i; what is playing switches to it
  SD.pickAlt = async (i) => {
    S.pick = i;
    SD.emit("change");
    SD.prefetch();
    if (!P || !(P.tag === "ab:result" || P.tag === "ck")) return;
    const res = SD.result();
    if (P.tag === "ck" && res.cks[res.ck]) await SD.ckBuf(res, res.ck);
    if (S.pick === i) SD.refresh();
  };
  SD.selectCk = async (res, j) => {
    res.ck = SD.clamp(j, 0, res.cks.length);
    SD.emit("change");
    if (!P || P.tag !== "ck") return;
    if (res.cks[res.ck]) await SD.ckBuf(res, res.ck);
    if (res.ck === j) SD.refresh();
  };

  // ---------- selection & view ----------

  const MIN_GAP = 0.05;
  SD.setGap = (a, b, edge) => {
    if (!S.track) return;
    const d = S.track.dur;
    a = SD.clamp(a, 0, d); b = SD.clamp(b, 0, d);
    if (b - a < MIN_GAP) {
      if (edge === "a") a = b - MIN_GAP; else b = a + MIN_GAP;
    }
    S.a = SD.clamp(a, 0, d - MIN_GAP); S.b = SD.clamp(b, MIN_GAP, d);
    if (edge) S.edge = edge;
    S.cands.forEach((c) => { if (c.custom) c.sim = SD.customSim(c.src, c.start); });
    SD.refresh();
    SD.emit("change");
  };
  SD.nudge = (dt) => {
    if (S.edge === "a") SD.setGap(S.a + dt, S.b, "a");
    else SD.setGap(S.a, S.b + dt, "b");
  };

  SD.setView = (t0, t1) => {
    if (!S.track) return;
    const d = S.track.dur, span = SD.clamp(t1 - t0, 0.2, d);
    t0 = SD.clamp(t0, 0, d - span);
    S.view = [t0, t0 + span];
    SD.emit("change");
  };
  SD.zoomGap = () => {
    const L = S.b - S.a, pad = Math.max(S.context, L * 0.5);
    SD.setView(S.a - pad, S.b + pad);
  };
  SD.zoomAll = () => S.track && SD.setView(0, S.track.dur);
  SD.zoom = (f, at) => {
    const [t0, t1] = S.view;
    at = at ?? (t0 + t1) / 2;
    SD.setView(at - (at - t0) * f, at + (t1 - at) * f);
  };

  // drag-to-select on a timeline element. range() gives its [t0, t1].
  SD.bindTimeline = (el, range = () => S.view) => {
    let drag = null;
    const tAt = (x) => {
      const r = el.getBoundingClientRect(), [t0, t1] = range();
      return t0 + ((x - r.left) / r.width) * (t1 - t0);
    };
    const pxPer = () => el.getBoundingClientRect().width / (range()[1] - range()[0]);
    el.addEventListener("pointerdown", (e) => {
      if (!S.track || e.button !== 0) return;
      el.setPointerCapture(e.pointerId);
      const t = tAt(e.clientX), tol = 6 / pxPer();
      if (Math.abs(t - S.a) < tol) drag = { kind: "a" };
      else if (Math.abs(t - S.b) < tol) drag = { kind: "b" };
      else if (e.shiftKey) drag = { kind: Math.abs(t - S.a) < Math.abs(t - S.b) ? "a" : "b" };
      else if (t > S.a && t < S.b && e.altKey) drag = { kind: "move", off: t - S.a };
      else drag = { kind: "new", t0: t, x0: e.clientX };
      e.preventDefault();
    });
    el.addEventListener("pointermove", (e) => {
      const t = tAt(e.clientX);
      el.style.cursor = drag && drag.kind !== "new" ? "ew-resize"
        : Math.abs(t - S.a) < 6 / pxPer() || Math.abs(t - S.b) < 6 / pxPer() ? "ew-resize" : "text";
      SD.hover = t; SD.emit("hover", t);
      if (!drag) return;
      if (drag.kind === "a") SD.setGap(t, S.b, "a");
      else if (drag.kind === "b") SD.setGap(S.a, t, "b");
      else if (drag.kind === "move") { const L = S.b - S.a; SD.setGap(t - drag.off, t - drag.off + L); }
      else if (Math.abs(e.clientX - drag.x0) > 3) {
        drag.moved = true;
        SD.setGap(Math.min(drag.t0, t), Math.max(drag.t0, t), t < drag.t0 ? "a" : "b");
      }
    });
    el.addEventListener("pointerleave", () => { SD.hover = null; SD.emit("hover", null); });
    el.addEventListener("pointerup", (e) => {
      if (drag && drag.kind === "new" && !drag.moved) {
        S.cursor = SD.clamp(tAt(e.clientX), 0, S.track.dur);
        if (P && P.tag === "track") SD.play(S.track.buf, S.cursor, S.track.dur, false, "track");
        SD.emit("change");
      }
      drag = null;
    });
    el.addEventListener("wheel", (e) => {
      if (!S.track) return;
      e.preventDefault();
      if (e.shiftKey || Math.abs(e.deltaX) > Math.abs(e.deltaY)) {
        const [t0, t1] = S.view, d = (e.deltaX || e.deltaY) / 600 * (t1 - t0);
        SD.setView(t0 + d, t1 + d);
      } else SD.zoom(Math.exp(e.deltaY / 400), tAt(e.clientX));
    }, { passive: false });
  };

  // overview: click or drag to move the detail view
  SD.bindOverview = (el) => {
    let down = false;
    const go = (e) => {
      const r = el.getBoundingClientRect(), t = ((e.clientX - r.left) / r.width) * S.track.dur;
      const span = S.view[1] - S.view[0];
      SD.setView(t - span / 2, t + span / 2);
    };
    el.addEventListener("pointerdown", (e) => { if (!S.track) return; down = true; el.setPointerCapture(e.pointerId); go(e); });
    el.addEventListener("pointermove", (e) => down && go(e));
    el.addEventListener("pointerup", () => (down = false));
  };

  // ---------- keys ----------

  SD.keys = [
    ["space", "play / stop"],
    ["L", "loop gap"],
    ["K", "loop gap with context"],
    ["< >", "nudge edge 10 ms (shift 100, alt 1)"],
    ["tab", "switch edge"],
    ["Z / 0", "zoom to gap / whole track"],
    ["C", "copy command"],
    ["enter", "search"],
  ];
  document.addEventListener("keydown", (e) => {
    if (/INPUT|TEXTAREA|SELECT/.test(document.activeElement && document.activeElement.tagName)) return;
    if (!S.track) return;
    const k = e.key;
    if (k === " ") { e.preventDefault(); SD.toggle(); }
    else if (k === "l" || k === "L") SD.loopGap();
    else if (k === "k" || k === "K") SD.loopContext();
    else if (k === "z" || k === "Z") SD.zoomGap();
    else if (k === "0") SD.zoomAll();
    else if (k === "c" || k === "C") SD.copy(SD.cli());
    else if (k === "Enter") SD.search();
    else if (k === "Tab") { e.preventDefault(); S.edge = S.edge === "a" ? "b" : "a"; SD.emit("change"); }
    else if (k === "ArrowLeft" || k === "ArrowRight" || k === "," || k === ".") {
      e.preventDefault();
      const d = (e.shiftKey ? 0.1 : e.altKey ? 0.001 : 0.01) * (k === "ArrowLeft" || k === "," ? -1 : 1);
      SD.nudge(d);
    }
  });

  // ---------- export ----------

  SD.wav = (buf) => {
    const ch = buf.numberOfChannels, n = buf.length, sr = buf.sampleRate;
    const dv = new DataView(new ArrayBuffer(44 + n * ch * 2));
    const str = (o, s) => [...s].forEach((c, i) => dv.setUint8(o + i, c.charCodeAt(0)));
    str(0, "RIFF"); dv.setUint32(4, 36 + n * ch * 2, true); str(8, "WAVE"); str(12, "fmt ");
    dv.setUint32(16, 16, true); dv.setUint16(20, 1, true); dv.setUint16(22, ch, true);
    dv.setUint32(24, sr, true); dv.setUint32(28, sr * ch * 2, true); dv.setUint16(32, ch * 2, true);
    dv.setUint16(34, 16, true); str(36, "data"); dv.setUint32(40, n * ch * 2, true);
    const d = []; for (let c = 0; c < ch; c++) d.push(buf.getChannelData(c));
    let o = 44;
    for (let i = 0; i < n; i++) for (let c = 0; c < ch; c++) { dv.setInt16(o, SD.clamp(d[c][i], -1, 1) * 32767, true); o += 2; }
    return new Blob([dv], { type: "audio/wav" });
  };
  SD.save = (buf, name) => {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(SD.wav(buf));
    a.download = name;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
    S.status = "saved " + name;
    SD.emit("change");
  };
  const stem = (s) => s.replace(/\.[^.]+$/, "");
  SD.candName = (c) => stem(c.srcName) + "_" + c.id + "_" + SD.fmt(c.start).replace(":", "m").replace(".", "s") + ".wav";
  SD.saveCand = (c) => SD.save(SD.candBuf(c), SD.candName(c));
  SD.saveSelected = () => {
    const on = S.cands.filter((c) => c.on);
    on.forEach((c, i) => setTimeout(() => SD.saveCand(c), i * 250));
  };

  SD.copy = (text) => {
    const done = () => { S.status = "copied: " + text; SD.emit("change"); };
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(done, () => { S.status = "select the command and copy it by hand"; SD.emit("change"); });
    } else S.status = "select the command and copy it by hand";
    SD.emit("change");
  };

  // ---------- redraw loop ----------

  let pending = false;
  const draws = [];
  SD.onDraw = (f) => draws.push(f);
  const frame = () => {
    pending = false;
    draws.forEach((f) => f());
    if (P || S.busy) request();
  };
  const request = () => { if (!pending) { pending = true; requestAnimationFrame(frame); } };
  SD.redraw = request;
  SD.on("change", request);
  SD.on("hover", request);
  window.addEventListener("resize", request);

  // ---------- demo track ----------

  // 24 s electro sketch at 126 bpm, rendered offline, so the page opens loaded.
  SD.demo = async () => {
    const sr = SD.ctx().sampleRate, bpm = 126, beat = 60 / bpm, bars = 12, dur = bars * 4 * beat + 0.5;
    const oc = new OfflineAudioContext(2, Math.ceil(dur * sr), sr);
    const master = oc.createGain(); master.gain.value = 0.55; master.connect(oc.destination);
    const noise = oc.createBuffer(1, sr, sr);
    noise.getChannelData(0).forEach((_, i, d) => (d[i] = Math.random() * 2 - 1));

    const kick = (t) => {
      const o = oc.createOscillator(), g = oc.createGain();
      o.frequency.setValueAtTime(150, t); o.frequency.exponentialRampToValueAtTime(42, t + 0.12);
      g.gain.setValueAtTime(1, t); g.gain.exponentialRampToValueAtTime(0.001, t + 0.35);
      o.connect(g).connect(master); o.start(t); o.stop(t + 0.4);
    };
    const hit = (t, f, len, amp, type = "highpass") => {
      const s = oc.createBufferSource(), fl = oc.createBiquadFilter(), g = oc.createGain();
      s.buffer = noise; fl.type = type; fl.frequency.value = f; fl.Q.value = 0.9;
      g.gain.setValueAtTime(amp, t); g.gain.exponentialRampToValueAtTime(0.001, t + len);
      s.connect(fl).connect(g).connect(master); s.start(t, Math.random() * 0.5); s.stop(t + len + 0.01);
    };
    const tone = (t, f, len, amp, type, cut, pan) => {
      const o = oc.createOscillator(), fl = oc.createBiquadFilter(), g = oc.createGain(), p = oc.createStereoPanner();
      o.type = type; o.frequency.value = f; fl.type = "lowpass"; fl.Q.value = 6;
      fl.frequency.setValueAtTime(cut * 4, t); fl.frequency.exponentialRampToValueAtTime(cut, t + len);
      g.gain.setValueAtTime(0, t); g.gain.linearRampToValueAtTime(amp, t + 0.005); g.gain.exponentialRampToValueAtTime(0.001, t + len);
      p.pan.value = pan;
      o.connect(fl).connect(g).connect(p).connect(master); o.start(t); o.stop(t + len + 0.02);
    };
    const hz = (m) => 440 * Math.pow(2, (m - 69) / 12);
    const bass = [33, 33, 45, 33, 36, 33, 43, 31];
    const arp = [[57, 60, 64, 67], [55, 60, 64, 67], [53, 57, 60, 64], [52, 55, 59, 64]];
    for (let bar = 0; bar < bars; bar++) {
      const t0 = bar * 4 * beat, sec = Math.floor(bar / 4);
      for (let q = 0; q < 4; q++) {
        if (!(sec === 0 && bar < 2 && q % 2)) kick(t0 + q * beat);
        if (q % 2 === 1 && bar > 1) hit(t0 + q * beat, 1800, 0.18, 0.5, "bandpass");
      }
      for (let s = 0; s < 16; s++) {
        const t = t0 + (s * beat) / 4;
        if (bar > 0) hit(t, 7000, s % 4 === 2 ? 0.12 : 0.04, s % 4 === 2 ? 0.28 : 0.14);
        if (s % 2 === 0) tone(t, hz(bass[(s / 2) % 8] + (sec === 1 ? 5 : 0)), beat / 2.2, 0.5, "sawtooth", 220, 0);
        if (bar >= 4 && bar !== 7) {
          const ch = arp[bar % 4];
          tone(t + 0.003, hz(ch[s % 4] + 12), beat / 4.5, 0.12, "square", 1600, s % 2 ? 0.4 : -0.4);
        }
      }
    }
    const buf = await oc.startRendering();
    SD.setTrack(makeTrack(buf, "demo_126bpm.wav"));
  };

  SD.boot = () => {
    S.status = "rendering demo track";
    SD.emit("change");
    SD.connect();
    SD.demo().catch(() => { S.status = "load a recording to begin"; SD.emit("change"); });
  };
})();
