// The GS-1 front panel, drawn entirely on one 1-bit display. Layout is in
// character cells (6 x 9 px); "vfd" shows the gaps between cells and uses
// the 5x7 font only, "paged" splits the panel into pages with soft keys.
(function () {
  "use strict";
  const S = SD.s, CW = 6, COLS = 80;
  const X = (c) => c * CW;

  window.GS1 = function (cv, opts) {
    const mode = opts.mode || "panel", vfd = mode === "vfd", paged = mode === "paged";
    const CH = vfd ? 9 : 12, Y = (r) => r * CH;
    const L = new LCD(cv, COLS * CW, 30 * CH, vfd ? { cell: [CW, CH], margin: CW } : { margin: 4 });
    const ui = { page: "trim", edit: null, hoverT: null, scroll: 0, own: null };

    // ---------- hidden inputs ----------
    const fileIn = Object.assign(document.createElement("input"), { type: "file", accept: "audio/*", hidden: true });
    const ownIn = Object.assign(document.createElement("input"), { type: "file", accept: "audio/*", hidden: true });
    const inp = Object.assign(document.createElement("input"), { type: "text", id: "lcd-edit", autocomplete: "off", spellcheck: false });
    inp.setAttribute("aria-label", "value being edited on the display");
    inp.style.cssText = "position:fixed;left:-9999px;top:0;opacity:0";
    document.body.append(fileIn, ownIn, inp);
    fileIn.onchange = () => fileIn.files[0] && SD.loadFile(fileIn.files[0]);
    ownIn.onchange = async () => {
      const f = ownIn.files[0];
      if (!f) return;
      S.status = "decoding " + f.name; SD.emit("change");
      try {
        const tr = await SD.decode(f);
        ui.own = { track: tr, start: SD.bestStart(tr) };
        S.status = "windowed " + f.name + " at its closest match";
        if (paged) ui.page = "pool";
      } catch (e) { S.status = "could not decode " + f.name; }
      ownIn.value = "";
      SD.emit("change");
    };
    document.addEventListener("dragover", (e) => e.preventDefault());
    document.addEventListener("drop", (e) => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) SD.loadFile(f); });

    function edit(name, value, commit) {
      ui.edit = { name, commit };
      inp.value = value;
      inp.focus(); inp.select();
      SD.redraw();
    }
    inp.addEventListener("input", SD.redraw);
    inp.addEventListener("keydown", (e) => {
      // the key that ends an edit is not also a shortcut
      if (e.key === "Enter" || e.key === "Escape") e.stopPropagation();
      if (e.key === "Enter") { const ed = ui.edit; ui.edit = null; inp.blur(); ed && ed.commit(inp.value); SD.redraw(); }
      if (e.key === "Escape") { ui.edit = null; inp.blur(); SD.redraw(); }
    });
    inp.addEventListener("blur", () => { if (ui.edit) { const ed = ui.edit; ui.edit = null; ed.commit(inp.value); SD.redraw(); } });
    setInterval(() => ui.edit && SD.redraw(), 400);
    const parseT = (s) => s.trim().split(":").reduce((t, p) => 60 * t + parseFloat(p), 0);

    // ---------- primitives in cells ----------
    const T = (c, r, s, v = 1) => L.text(X(c), Y(r) + (vfd ? 0 : 2), s, "big", v);
    const Ts = T;
    const fit = (c, r, s, w) => L.textFit(X(c), Y(r) + (vfd ? 0 : 2), s, X(w) - 1);
    const inv = (c, r, w, h = 1) => L.invert(X(c), Y(r), X(w), Y(h));
    const lineH = (r, c0 = 0, c1 = COLS) => vfd ? T(c0, r, "-".repeat(c1 - c0)) : L.hline(X(c0), Y(r) + 4, X(c1 - c0), 2);

    // field: shows the edit buffer while editing
    function field(c, r, w, name, text, onClick, k = 1) {
      const editing = ui.edit && ui.edit.name === name;
      const s = editing ? inp.value + (Math.floor(performance.now() / 400) % 2 ? "_" : " ") : text;
      if (k === 2 && !vfd) L.text(X(c), Y(r), s, "big", 1, 2);
      else T(c, r, editing && vfd ? ">" + s : s);
      if (editing && !vfd) inv(c, r, w, k);
      L.hit(X(c), Y(r), X(w), Y(k), { click: onClick, cursor: "text" });
    }

    // key: framed on the panel, bracketed on the vfd; inverted when on
    function key(c, r, w, label, act, on, opt = {}) {
      const h = opt.h || 1, dis = !!opt.disabled;
      if (vfd) {
        // no inverse video on a VFD: the active key is marked with arrows
        const s = (dis ? " " + label + " " : (on ? ">" : "[") + label + (on ? "<" : "]")).slice(0, w);
        T(c + Math.floor((w - s.length) / 2), r + h - 1, s);
        if (opt.hint && h > 1) T(c + 1, r, opt.hint.slice(0, w - 1));
      } else {
        const x = X(c), y = Y(r), pw = X(w) - 2, ph = Y(h) - 1;
        // disabled keys keep their text and lose the solid frame
        if (dis) { L.hline(x, y, pw, 2); L.hline(x, y + ph - 1, pw, 2); L.vline(x, y, ph, 2); L.vline(x + pw - 1, y, ph, 2); }
        else L.frame(x, y, pw, ph);
        const tw = L.textWidth(label);
        const ty = h > 1 ? y + ph - 10 : y + Math.floor((ph - 7) / 2);
        L.text(x + Math.floor((pw - tw) / 2), ty, label);
        if (opt.hint && h > 1) L.text(x + 3, y + 2, opt.hint);
        if (on) L.invert(x + 1, y + 1, pw - 2, ph - 2);
      }
      if (!dis) L.hit(X(c), Y(r), X(w), Y(h), { click: act });
    }

    const gmax = (tr) => {
      if (tr._gmax) return tr._gmax;
      let m = 1e-3;
      for (let j = 0; j < tr.bmax.length; j++) m = Math.max(m, tr.bmax[j], -tr.bmin[j]);
      return (tr._gmax = m);
    };

    // 1-bit wave in a pixel box. mode(t): 0 dithered, 1 solid, 2 inverted
    function wave(tr, t0, t1, x0, y0, w, h, mode, scale) {
      const pk = SD.peaks(tr, t0, t1, w), mid = y0 + Math.floor(h / 2), amp = (h / 2 - 1) / scale;
      for (let x = 0; x < w; x++) {
        const m = mode(t0 + ((x + 0.5) / w) * (t1 - t0));
        const top = Math.round(mid - pk.hi[x] * amp), bot = Math.round(mid - pk.lo[x] * amp);
        if (m === 2) {
          L.rect(x0 + x, y0, 1, h);
          for (let y = Math.min(top, mid); y <= Math.max(bot, mid); y++) L.px(x0 + x, y, 0);
        } else {
          if (m === 0 && x % 2) continue;
          for (let y = Math.min(top, mid); y <= Math.max(bot, mid); y++) L.px(x0 + x, y);
        }
      }
    }

    function ruler(t0, t1, x0, y, w) {
      const steps = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300];
      let step = 600;
      const lab = vfd ? 60 : 40;
      for (const s of steps) if ((s / (t1 - t0)) * w >= lab + (vfd ? 0 : 20)) { step = s; break; }
      const digits = step < 0.01 ? 3 : step < 0.1 ? 2 : step < 1 ? 1 : 0;
      if (!vfd) L.hline(x0, y, w);
      for (let t = Math.ceil(t0 / step) * step; t < t1; t += step) {
        const x = x0 + Math.round(((t - t0) / (t1 - t0)) * w);
        if (vfd) { const c = Math.round(x / CW); T(c, y / CH, "'" + SD.fmt(t, digits)); }
        else { L.vline(x, y, 3); L.text(x + 2, y + 3, SD.fmt(t, digits)); }
      }
    }

    // ---------- components ----------

    function statusBar(r) {
      const t = S.track, sv = SD.server;
      T(1, r, "SIMDPS GS-1");
      if (t) fit(14, r, t.name + "  " + SD.fmt(t.dur) + "  " + Math.round(t.sr / 100) / 10 + "KHZ " + t.buf.numberOfChannels + "CH", 46);
      const st = !sv ? "SIMULATION" : sv.error ? "MODEL ERROR" : sv.ready ? (sv.model || "MODEL").toUpperCase() + " READY"
        : Math.floor(performance.now() / 500) % 2 ? "LOADING MODEL" : "";
      T(COLS - 1 - st.length, r, st);
      if (vfd) T(0, r + 1, "=".repeat(COLS)); else inv(0, r, COLS);
    }

    function fileKeys(c, r) {
      key(c, r, 7, "LOAD", () => fileIn.click(), false);
      key(c + 7, r, 7, "DEMO", () => SD.demo(), false);
    }

    function overview(c, r, w, h) {
      const tr = S.track, x0 = X(c), y0 = Y(r), pw = X(w), ph = Y(h) - 2;
      wave(tr, 0, tr.dur, x0, y0, pw, ph, (t) => (t >= S.a && t < S.b ? 2 : t >= S.view[0] && t < S.view[1] ? 1 : 0), gmax(tr));
      const v0 = x0 + Math.round((S.view[0] / tr.dur) * pw), v1 = x0 + Math.round((S.view[1] / tr.dur) * pw);
      L.rect(v0, y0 + ph, Math.max(1, v1 - v0), 1);
      L.vline(v0, y0, ph); L.vline(Math.max(v0, v1 - 1), y0, ph);
      const p = SD.playing(), px = x0 + Math.round(((p ? SD.pos() : S.cursor) / tr.dur) * pw);
      L.vline(px, y0, ph, 1, 2);
      const go = (x) => { const t = (x / pw) * tr.dur, s = S.view[1] - S.view[0]; SD.setView(t - s / 2, t + s / 2); };
      L.hit(x0, y0, pw, Y(h), { down: go, move: go, cursor: "pointer" });
    }

    let drag = null;
    function detail(c, r, w, h) {
      const tr = S.track, [t0, t1] = S.view, x0 = X(c), y0 = Y(r) + CH, pw = X(w), ph = Y(h - 3);
      const tx = (t) => x0 + Math.round(((t - t0) / (t1 - t0)) * pw);
      const ca = S.a - S.context, cb = S.b + S.context;
      wave(tr, t0, t1, x0, y0, pw, ph, (t) => (t >= S.a && t < S.b ? 2 : t >= ca && t < cb ? 1 : 0), gmax(tr));
      // top line: context extent and the gap length
      const xa = tx(S.a), xb = tx(S.b);
      const yt = Y(r);
      if (vfd) {
        const ca_c = Math.max(c, Math.round(tx(ca) / CW)), cb_c = Math.min(c + w - 1, Math.round(tx(cb) / CW));
        for (let k = ca_c; k <= cb_c; k++) T(k, r, "-");
        const ga = Math.round(xa / CW), gb = Math.max(ga + 1, Math.round(xb / CW));
        const lab = Math.round((S.b - S.a) * 1000) + "MS";
        for (let k = ga; k < gb; k++) T(k, r, "=");
        T(ga, r, "["); T(gb - 1, r, "]");
        if (gb - ga > lab.length + 3) T(Math.floor((ga + gb - lab.length) / 2), r, lab);
      } else {
        for (let x = Math.max(x0, tx(ca)); x < Math.min(x0 + pw, tx(cb)); x += 2) L.px(x, yt + 4);
        L.vline(Math.max(x0, tx(ca)), yt + 2, 5); L.vline(Math.min(x0 + pw - 1, tx(cb)), yt + 2, 5);
        L.rect(xa, yt, Math.max(1, xb - xa), 8);
        const lab = Math.round((S.b - S.a) * 1000) + " MS", lw = L.textWidth(lab);
        if (xb - xa > lw + 4) L.text(Math.round((xa + xb - lw) / 2), yt, lab, "big", 0);
      }
      // selected edge: caret under the wave
      const ex = S.edge === "a" ? xa : xb, ey = y0 + ph;
      L.rect(ex - 2, ey + 2, 5, 1); L.rect(ex - 1, ey + 1, 3, 1); L.px(ex, ey);
      const p = SD.playing(), pc = tx(p ? SD.pos() : S.cursor);
      if (pc >= x0 && pc < x0 + pw) L.vline(pc, y0, ph, p ? 1 : 2, 2);
      if (ui.hoverT != null) { const hx = tx(ui.hoverT); if (hx >= x0 && hx < x0 + pw) L.vline(hx, y0, ph, 4); }
      ruler(t0, t1, x0, Y(r + h - 2) + (vfd ? 0 : 3), pw);

      const tAt = (x) => t0 + (x / pw) * (t1 - t0);
      L.hit(x0, Y(r), pw, Y(h - 1), {
        cursor: "text",
        hover: (x) => { ui.hoverT = tAt(x); },
        down: (x, y, e) => {
          const t = tAt(x), tol = 3 * (t1 - t0) / pw;
          if (Math.abs(t - S.a) < tol) drag = { k: "a" };
          else if (Math.abs(t - S.b) < tol) drag = { k: "b" };
          else if (e.shiftKey) drag = { k: Math.abs(t - S.a) < Math.abs(t - S.b) ? "a" : "b" };
          else drag = { k: "new", t0: t };
        },
        move: (x) => {
          const t = tAt(x);
          ui.hoverT = t;
          if (!drag) return;
          if (drag.k === "a") SD.setGap(t, S.b, "a");
          else if (drag.k === "b") SD.setGap(S.a, t, "b");
          else { drag.moved = true; SD.setGap(Math.min(drag.t0, t), Math.max(drag.t0, t), t < drag.t0 ? "a" : "b"); }
        },
        up: (x, y, e, q, moved) => {
          if (drag && drag.k === "new" && !moved) {
            S.cursor = SD.clamp(tAt(x), 0, tr.dur);
            const pl = SD.playing();
            if (pl && pl.tag === "track") SD.play(tr.buf, S.cursor, tr.dur, false, "track");
          }
          drag = null; SD.redraw();
        },
        wheel: (x, y, e) => {
          if (e.shiftKey || Math.abs(e.deltaX) > Math.abs(e.deltaY)) {
            const d = ((e.deltaX || e.deltaY) / 600) * (t1 - t0);
            SD.setView(t0 + d, t1 + d);
          } else SD.zoom(Math.exp(e.deltaY / 400), tAt(x));
        },
      });
    }
    L.onHover = (q) => { if (!q || !q.hover) { if (ui.hoverT != null) { ui.hoverT = null; SD.redraw(); } } else SD.redraw(); };

    // readout column, 16 cells wide
    function readouts(c, r) {
      const k = vfd ? 1 : 2, step = vfd ? 2 : 3;
      Ts(c, r, "GAP START");
      field(c, r + 1, 16, "a", SD.fmt(S.a), () => edit("a", SD.fmt(S.a), (v) => { const t = parseT(v); if (isFinite(t)) SD.setGap(t, S.b, "a"); }), k);
      Ts(c, r + step, "GAP END");
      field(c, r + step + 1, 16, "b", SD.fmt(S.b), () => edit("b", SD.fmt(S.b), (v) => { const t = parseT(v); if (isFinite(t)) SD.setGap(S.a, t, "b"); }), k);
      Ts(c, r + 2 * step, "LENGTH");
      const len = Math.round((S.b - S.a) * 1000) + " MS";
      if (k === 2) L.text(X(c), Y(r + 2 * step + 1), len, "big", 1, 2); else T(c, r + 2 * step + 1, len);
      let y = r + 3 * step;
      Ts(c, y, "CTX"); field(c + 5, y, 6, "ctx", S.context.toFixed(1) + "S", () => edit("ctx", S.context.toFixed(1), (v) => { const x = parseFloat(v); if (isFinite(x)) { S.context = SD.clamp(x, 0.5, 10); SD.setGap(S.a, S.b); } }));
      Ts(c, y + 1, "CUR"); T(c + 5, y + 1, ui.hoverT != null && S.track ? SD.fmt(SD.clamp(ui.hoverT, 0, S.track.dur)) : "-:--.---");
      Ts(c, y + 2, "POS"); T(c + 5, y + 2, SD.fmt(SD.playing() ? SD.pos() : S.cursor));
      Ts(c, y + 3, "RES"); T(c + 5, y + 3, (((S.view[1] - S.view[0]) / X(61)) * 1000).toFixed(1) + "MS");
      key(c, y + 4, 8, "START", () => { S.edge = "a"; SD.emit("change"); }, S.edge === "a");
      key(c + 8, y + 4, 8, "END", () => { S.edge = "b"; SD.emit("change"); }, S.edge === "b");
      key(c, y + 5, 4, "<<", () => SD.nudge(-0.1)); key(c + 4, y + 5, 4, "<", () => SD.nudge(-0.01));
      key(c + 8, y + 5, 4, ">", () => SD.nudge(0.01)); key(c + 12, y + 5, 4, ">>", () => SD.nudge(0.1));
    }

    function softKeys(r, keys, h = 2) {
      const w = COLS / keys.length;
      keys.forEach((k, i) => k && key(Math.round(i * w), r, Math.round((i + 1) * w) - Math.round(i * w), k.label, k.act, k.on, { h, hint: k.hint, disabled: k.disabled }));
    }

    const tag = () => (SD.playing() ? SD.playing().tag : "");
    const trimKeys = () => [
      { label: "PLAY", hint: "SPC", act: SD.toggle, on: tag() === "track" },
      { label: "LOOP", hint: "L", act: SD.loopGap, on: tag() === "gap" },
      { label: "LOOP+CTX", hint: "K", act: SD.loopContext, on: tag() === "ctx" },
      { label: "ZOOM GAP", hint: "Z", act: SD.zoomGap },
      { label: "WHOLE", hint: "0", act: SD.zoomAll },
      { label: "COPY CMD", hint: "C", act: () => SD.copy(SD.cli()) },
      { label: "SEARCH", hint: "ENT", act: () => { SD.search(); if (paged) ui.page = "pool"; }, on: !!(S.busy && S.busy.what === "search") },
      { label: "INPAINT", hint: "", act: () => { SD.inpaint(); }, on: !!(S.busy && S.busy.what === "inpaint"), disabled: !S.cands.some((c) => c.on) },
    ];

    // command lines, real text goes to the clipboard
    function command(r) {
      fit(0, r, "$ " + SD.cli(), 71);
      key(72, r, 8, "COPY", () => SD.copy(SD.cli()));
      const g = SD.cliGuides();
      fit(0, r + 1, g ? "$ " + g : "", 71);
      key(72, r + 1, 8, "COPY", () => g && SD.copy(g), false, { disabled: !g });
    }

    const MATCHES = { latent: "LATENT", audio: "AUDIO", bands: "BANDS" };
    function poolHeader(r) {
      T(0, r, "CANDIDATE POOL");
      const on = S.cands.filter((c) => c.on).length;
      Ts(16, r, S.cands.length ? on + "/" + S.cands.length + " IN GUIDANCE" : "NONE YET");
      // match method: quiet, cycles on click
      const remote = SD.server && SD.server.ready;
      const opts = remote ? ["latent", "audio", "bands"] : ["bands"];
      const lab = "MATCH:" + MATCHES[S.match];
      Ts(45, r, lab);
      if (remote) L.hit(X(45), Y(r), X(lab.length), Y(1), { click: () => { S.match = opts[(opts.indexOf(S.match) + 1) % opts.length]; SD.emit("change"); } });
      const from = "FROM:" + (S.from === "bank" ? "BANK" : "TRACK");
      Ts(33, r, from);
      L.hit(X(33), Y(r), X(from.length), Y(1), { click: () => { S.from = S.from === "bank" ? "track" : "bank"; SD.emit("change"); } });
      key(58, r, 7, "ALL", () => { S.cands.forEach((c) => (c.on = true)); SD.emit("change"); });
      key(65, r, 7, "NONE", () => { S.cands.forEach((c) => (c.on = false)); SD.emit("change"); });
      key(72, r, 8, "SEARCH", () => SD.search(), !!(S.busy && S.busy.what === "search"));
      if (!vfd) {
        const y = Y(r + 1) + 2;
        [[2, "NO"], [5, "AT"], [15, "SOURCE"], [32, "WT"], [38, "SPLICED INTO CONTEXT"], [66, "LISTEN"]].forEach(([c, s]) => L.text(X(c), y, s));
      }
    }

    function miniTrack(c) {
      const key = S.a + ":" + S.b + ":" + S.context + ":" + c.start;
      if (!c._mini || c._mini.key !== key) c._mini = { key, track: SD.makeTrack(SD.preview(c), "") };
      return c._mini.track;
    }

    function candRow(cand, i, r, w) {
      const p = tag(), pct = Math.round(w * 100);
      const toggle = () => { cand.on = !cand.on; SD.emit("change"); };
      T(0, r, cand.on ? "#" : "-"); L.hit(0, Y(r), X(2), Y(2), { click: toggle });
      if (!vfd) { L.rect(0, Y(r) + 1, 9, 9, 0); L.frame(0, Y(r) + 1, 9, 9); if (cand.on) L.rect(2, Y(r) + 3, 5, 5); }
      T(2, r, String(i + 1).padStart(2, "0"));
      T(5, r, SD.fmt(cand.start));
      fit(15, r, (cand.custom ? (cand.src === S.track ? "PIN " : "OWN ") : cand.bank ? "BNK " : "") + cand.srcName, 16);
      const f = SD.clamp(cand.sim, 0, 1);
      if (vfd) { const n = Math.round(f * 8); T(5, r + 1, "#".repeat(n) + ".".repeat(8 - n) + " " + cand.sim.toFixed(2)); }
      else {
        T(5, r + 1, "MATCH");
        const bx = X(11), by = Y(r + 1) + 3, bw = 48;
        L.frame(bx, by, bw, 5); L.rect(bx + 1, by + 1, Math.round((bw - 2) * f), 3);
        T(20, r + 1, cand.sim.toFixed(2));
      }
      // weight: click to type, wheel to adjust, double click for automatic
      const wt = cand.on ? String(pct).padStart(3, " ") + "%" + (cand.wt != null ? "*" : "") : " --";
      field(32, r, 6, "w:" + cand.id, wt, () => cand.on && edit("w:" + cand.id, String(pct), (v) => { const x = parseFloat(v); if (isFinite(x)) SD.setWeight(cand, x / 100); }));
      Ts(32, r + 1, cand.wt != null ? "SET" : "AUTO");
      L.hit(X(32), Y(r), X(7), Y(2), {
        click: () => cand.on && edit("w:" + cand.id, String(pct), (v) => { const x = parseFloat(v); if (isFinite(x)) SD.setWeight(cand, x / 100); }),
        dbl: () => SD.setWeight(cand, null),
        wheel: (x, y, e) => { if (!cand.on) return; SD.setWeight(cand, (cand.wt != null ? cand.wt : w) - Math.sign(e.deltaY) * 0.05); },
        cursor: "ns-resize",
      });
      const mt = miniTrack(cand);
      const offT = Math.max(0, S.a - S.context);
      wave(mt, 0, mt.dur, X(40), Y(r), X(18), Y(2) - 1, (t) => (t + offT >= S.a && t + offT < S.b ? (cand.on ? 2 : 1) : 1), gmax(S.track));
      key(66, r, 7, "HEAR", () => SD.playCand(cand, true), p === "cand:" + cand.id + ":in");
      key(73, r, 7, "RAW", () => SD.playCand(cand, false), p === "cand:" + cand.id + ":raw");
      key(59, r + 1, 7, "BANK", () => bank(cand.src.buf, cand.start, S.b - S.a, stem(cand.srcName) + "_cand", "cand", cand.srcName));
      key(66, r + 1, 7, "SAVE", () => SD.saveCand(cand));
      if (cand.custom) key(73, r + 1, 7, "DROP", () => SD.removeCand(cand));
    }

    function poolRows(r, max) {
      const w = SD.weights(), n = S.cands.length;
      if (!n) {
        return 2;
      }
      const shown = Math.min(n, max || n);
      ui.scroll = SD.clamp(ui.scroll, 0, n - shown);
      for (let i = 0; i < shown; i++) {
        const k = i + ui.scroll;
        candRow(S.cands[k], k, r + 2 * i, w.get(S.cands[k]));
        if (i < shown - 1 && !vfd) L.hline(0, Y(r + 2 * i + 2) - 1, X(COLS), 3);
      }
      if (shown < n) {
        const sh = Y(2 * shown) - 2, th = Math.max(4, Math.round((sh * shown) / n)), ty = Math.round((sh - th) * ui.scroll / (n - shown));
        L.vline(X(COLS) - 1, Y(r), sh, 2); L.rect(X(COLS) - 2, Y(r) + ty, 2, th);
        L.hit(0, Y(r), X(COLS), Y(2 * shown), { wheel: (x, y, e) => { ui.scroll = SD.clamp(ui.scroll + Math.sign(e.deltaY), 0, n - shown); SD.redraw(); } });
      }
      return 2 * shown;
    }

    // match of every start position, on the same time axis as the trace
    // above it; marks are starts already in the pool
    function curveStrip(src, t0, t1, c, r, w, h, pick, marks, hover) {
      const x0 = X(c), y0 = Y(r), pw = X(w), ph = Y(h) - 2, cv = SD.curve(src);
      if (!cv) return;
      const tx = (t) => x0 + Math.round(((t - t0) / (t1 - t0)) * pw);
      L.hline(x0, y0 + ph, pw);
      for (let x = 0; x < pw; x += 4) L.px(x0 + x, y0 + Math.round((ph - 1) / 2));
      let prev = null;
      for (let x = 0; x < pw; x++) {
        const t = t0 + ((x + 0.5) / pw) * (t1 - t0), v = cv.sims[Math.round(t / cv.hop)];
        if (v === undefined || isNaN(v)) { prev = null; continue; }
        // scaled to this curve's own range, which zooming does not change
        const y = y0 + Math.round((1 - (v - cv.lo) / (cv.hi - cv.lo)) * (ph - 1));
        if (prev != null) for (let k = Math.min(prev, y); k <= Math.max(prev, y); k++) L.px(x0 + x, k);
        else L.px(x0 + x, y);
        if (!cv.stale && x % 2 === 0) for (let k = y + 2; k < y0 + ph; k += 2) L.px(x0 + x, k);
        prev = y;
      }
      if (src === S.track) {
        // starts this close would copy from the gap's own surroundings
        const xa = Math.max(x0, tx(S.a - (S.b - S.a) - S.context)), xb = Math.min(x0 + pw, tx(S.b + S.context));
        const lab = "NEAR GAP";
        if (xb - xa > L.textWidth(lab) + 8) {
          const lx = Math.round((xa + xb - L.textWidth(lab)) / 2), ly = y0 + Math.round((ph - 7) / 2);
          L.rect(lx - 2, ly - 1, L.textWidth(lab) + 4, 9, 0);
          L.text(lx, ly, lab);
        }
      }
      (marks || []).forEach((m) => {
        const x = tx(m);
        if (x >= x0 && x < x0 + pw) { L.rect(x - 2, y0, 5, 1); L.rect(x - 1, y0 + 1, 3, 1); L.px(x, y0 + 2); }
      });
      if (hover.t != null) { const hx = tx(hover.t); if (hx >= x0 && hx < x0 + pw) L.vline(hx, y0, ph, 2); }
      const tAt = (x) => t0 + (x / pw) * (t1 - t0);
      L.hit(x0, y0, pw, Y(h), {
        cursor: "crosshair",
        hover: (x) => { hover.t = tAt(x); },
        click: (x) => pick(tAt(x)),
        wheel: (x, y, e) => hover.wheel && hover.wheel(e, tAt(x)),
      });
    }

    const mainHover = { get t() { return ui.hoverT; }, set t(v) { ui.hoverT = v; },
      wheel: (e, t) => SD.zoom(Math.exp(e.deltaY / 400), t) };
    const poolStarts = (src) => S.cands.filter((c) => c.src === src).map((c) => c.start);

    // own recording: its own zoomable trace, a window the length of the gap,
    // and the match curve under it
    function ownSection(r) {
      key(0, r, 12, "ADD OWN", () => ownIn.click());
      if (!ui.own) return 1;
      const o = ui.own, tr = o.track, len = S.b - S.a;
      if (!o.view) o.view = [0, tr.dur];
      if (!o.hover) o.hover = { t: null, wheel: (e, t) => zoomOwn(o, e, t) };
      const [t0, t1] = o.view, x0 = 0, pw = X(62), y0 = Y(r + 1), ph = Y(5) - 1;
      const tx = (t) => x0 + Math.round(((t - t0) / (t1 - t0)) * pw);
      fit(13, r, tr.name + "  " + SD.fmt(tr.dur), 24);
      key(38, r, 7, "BANK", () => bank(tr.buf, o.start, len, stem(tr.name) + "_own", "own", tr.name));
      key(45, r, 7, "FIT", () => { o.view = [o.start - S.context - len, o.start + 2 * len + S.context]; clampOwn(o); SD.redraw(); });
      key(52, r, 7, "FULL", () => { o.view = [0, tr.dur]; SD.redraw(); });
      key(59, r, 7, "BEST", () => { o.start = SD.bestStart(tr); SD.refresh(); SD.redraw(); });
      key(66, r, 7, "HEAR", () => {
        if (tag() === "own") return SD.stop();
        SD.playWith("own", () => { const b = SD.preview({ id: "own", src: tr, start: o.start }); return { buf: b, from: 0, to: b.duration, offset: b.offset }; });
      }, tag() === "own");
      key(73, r, 7, "ADD", () => { SD.addCustom(tr, o.start); if (tag() === "own") SD.stop(); ui.own = null; });

      const ca = o.start - S.context, cb = o.start + len + S.context;
      wave(tr, t0, t1, x0, y0, pw, ph, (t) => (t >= o.start && t < o.start + len ? 2 : t >= ca && t < cb ? 1 : 0), gmax(tr));
      poolStarts(tr).forEach((m) => { const x = tx(m); if (x >= x0 && x < x0 + pw) L.vline(x, y0, ph, 3); });
      const p = SD.playing();
      if (p && p.tag === "own") { const pc = tx(SD.pos() - S.a + o.start); if (pc >= x0 && pc < x0 + pw) L.vline(pc, y0, ph, 1, 2); }
      if (o.hover.t != null) { const hx = tx(o.hover.t); if (hx >= x0 && hx < x0 + pw) L.vline(hx, y0, ph, 4); }
      const tAt = (x) => t0 + (x / pw) * (t1 - t0);
      const place = (t) => { o.start = SD.clamp(t, 0, Math.max(0, tr.dur - len)); SD.refresh(); SD.redraw(); };
      let grab = null;
      L.hit(x0, y0, pw, ph, {
        cursor: "grab",
        hover: (x) => { o.hover.t = tAt(x); },
        down: (x) => { const t = tAt(x); grab = t >= o.start && t < o.start + len ? t - o.start : len / 2; place(t - grab); },
        move: (x) => { o.hover.t = tAt(x); if (grab != null) place(tAt(x) - grab); },
        up: () => { grab = null; },
        wheel: (x, y, e) => zoomOwn(o, e, tAt(x)),
      });
      ruler(t0, t1, x0, Y(r + 6), pw);
      curveStrip(tr, t0, t1, 0, r + 7, 62, 2, place, poolStarts(tr), o.hover);

      T(64, r + 1, "WINDOW");
      field(64, r + 2, 16, "own", SD.fmt(o.start), () => edit("own", SD.fmt(o.start), (v) => { const t = parseT(v); if (isFinite(t)) place(t); }));
      T(64, r + 3, "TO " + SD.fmt(o.start + len));
      T(64, r + 4, "MATCH " + SD.customSim(tr, o.start).toFixed(2));
      const hv = o.hover.t != null ? SD.curveAt(tr, o.hover.t) : NaN;
      T(64, r + 5, "CUR " + (isNaN(hv) ? "--" : hv.toFixed(2)));
      return 9;
    }
    function clampOwn(o) {
      const d = o.track.dur, span = SD.clamp(o.view[1] - o.view[0], 0.2, d), t0 = SD.clamp(o.view[0], 0, d - span);
      o.view = [t0, t0 + span];
    }
    function zoomOwn(o, e, at) {
      const [t0, t1] = o.view;
      if (e.shiftKey || Math.abs(e.deltaX) > Math.abs(e.deltaY)) {
        const d = ((e.deltaX || e.deltaY) / 600) * (t1 - t0);
        o.view = [t0 + d, t1 + d];
      } else {
        const f = Math.exp(e.deltaY / 400);
        o.view = [at - (at - t0) * f, at + (t1 - at) * f];
      }
      clampOwn(o);
      SD.redraw();
    }

    function ownRow(r) {
      key(0, r, 12, "ADD OWN", () => ownIn.click());
      if (!ui.own) return 1;
      const o = ui.own, tr = o.track, len = S.b - S.a, x0 = X(13), pw = X(45);
      wave(tr, 0, tr.dur, x0, Y(r), pw, Y(2) - 1, (t) => (t >= o.start && t < o.start + len ? 2 : 0), gmax(tr));
      const go = (x) => { o.start = SD.clamp((x / pw) * tr.dur - len / 2, 0, Math.max(0, tr.dur - len)); SD.refresh(); SD.redraw(); };
      L.hit(x0, Y(r), pw, Y(2), { down: go, move: go, cursor: "ew-resize" });
      Ts(0, r + 1, "M " + SD.customSim(tr, o.start).toFixed(2));
      key(59, r, 7, "BEST", () => { o.start = SD.bestStart(tr); SD.redraw(); });
      key(66, r, 7, "HEAR", () => {
        if (tag() === "own") return SD.stop();
        SD.playWith("own", () => { const b = SD.preview({ id: "own", src: tr, start: o.start }); return { buf: b, from: 0, to: b.duration, offset: b.offset }; });
      }, tag() === "own");
      key(73, r, 7, "ADD", () => { SD.addCustom(tr, o.start); if (tag() === "own") SD.stop(); ui.own = null; });
      fit(59, r + 1, SD.fmt(o.start) + " " + tr.name, 21);
      return 2;
    }

    const PRESETS = { paper: "PAPER", warm_start_detailed: "WARM_START (DETAILED)", warm_start_fast: "WARM_START (FAST)" };
    function presetRow(r) {
      Ts(0, r, "PRESET");
      key(7, r, 24, PRESETS[S.preset], () => { const k = Object.keys(PRESETS); S.preset = k[(k.indexOf(S.preset) + 1) % k.length]; SD.emit("change"); });
      if (paged) return;
      // quiet, like MATCH: and FROM:; opens the settings below
      const n = SD.customCount(), lab = "SETTINGS" + (n ? " " + n + "*" : "");
      T(33, r, lab);
      if (ui.settings) inv(33, r, lab.length);
      L.hit(X(33), Y(r), X(lab.length), Y(1), { click: () => { ui.settings = !ui.settings; SD.redraw(); } });
    }

    // ---------- settings ----------

    // columns of [title, config section, keys]
    const SET_COLS = [
      [["SAMPLER", "sampler", ["steps", "sigma_max", "sigma_min", "rho", "churn", "s_noise", "order", "start"]]],
      [["CANDIDATES", "candidates", ["weight", "combine", "distance", "cutoff", "taper", "stop", "fade"]]],
      [["SEARCH", "search", ["n", "context", "crossfade", "metric", "interp", "ramp"]]],
      [["OBSERVATION", "observation", ["zeta"]], ["GUIDANCE", "guidance", ["project", "jacobian", "joint"]]],
    ];
    const SET_H = 11;
    const CHOICES = {
      "sampler.order": [1, 2], "sampler.start": ["candidate", "noise"],
      "candidates.combine": ["weighted", "solo", "min", "mean"], "candidates.distance": ["lowpass", "magnitude", "l2"],
      "search.metric": ["l2", "l1", "cosine", "pearson"], "search.interp": ["slerp", "lerp"],
      "search.ramp": [true, false], "guidance.project": [true, false], "guidance.jacobian": [true, false],
      "guidance.joint": [true, false],
    };
    const METHODS = ["auto", "multi", "solo"];
    const fmtV = (v) => v == null ? "--" : typeof v === "boolean" ? (v ? "ON" : "OFF")
      : typeof v === "number" ? String(+v.toPrecision(4)).toUpperCase() : String(v).toUpperCase();

    function setting(c, r, sec, k, base) {
      const id = sec + "." + k, has = !!S.custom[sec] && k in S.custom[sec];
      const was = base ? (base[sec] || {})[k] : undefined, v = has ? S.custom[sec][k] : was;
      const set = (x) => SD.setCustom(sec, k, base && x === was ? undefined : x);
      const s = (v === undefined ? "--" : fmtV(v)) + (has ? "*" : "");
      T(c + 1, r, k.toUpperCase());
      const ch = CHOICES[id];
      if (ch) {
        T(c + 11, r, s);
        L.hit(X(c + 11), Y(r), X(9), Y(1), { click: () => set(ch[(ch.indexOf(v) + 1) % ch.length]) });
        return;
      }
      field(c + 11, r, 9, "set:" + id, s, () => edit("set:" + id, v == null ? "" : String(v), (t) => {
        t = t.trim();
        if (t === "") return SD.setCustom(sec, k, undefined);
        if (/^(--|none|null)$/i.test(t)) return set(null);
        const x = Number(t);
        if (isFinite(x)) set(x);
      }));
    }

    function settingsSection(r) {
      const base = SD.baseSettings();
      T(0, r, "SETTINGS");
      const m = "METHOD:" + S.method.toUpperCase() + (S.method === "auto" ? " (" + SD.methodNow().toUpperCase() + ")" : "");
      Ts(10, r, m);
      L.hit(X(10), Y(r), X(m.length), Y(1), { click: () => { S.method = METHODS[(METHODS.indexOf(S.method) + 1) % METHODS.length]; SD.emit("change"); } });
      key(64, r, 8, "RESET", () => { S.custom = {}; SD.emit("change"); }, false, { disabled: !SD.customCount() });
      key(72, r, 8, "CLOSE", () => { ui.settings = false; SD.redraw(); });
      SET_COLS.forEach((groups, i) => {
        let rr = r + 2;
        groups.forEach(([title, sec, keys]) => {
          T(i * 20, rr++, title);
          keys.forEach((k) => setting(i * 20, rr++, sec, k, base));
          rr++;
        });
      });
      return SET_H;
    }
    function inpaintBar(r, go = true) {
      key(0, r, 10, "SAVE SEL", SD.saveSelected, false, { disabled: !S.cands.some((c) => c.on) });
      Ts(11, r, "ALT");
      key(15, r, 3, "-", () => { S.samples = Math.max(1, S.samples - 1); SD.emit("change"); });
      T(19, r, String(S.samples));
      key(21, r, 3, "+", () => { S.samples = Math.min(4, S.samples + 1); SD.emit("change"); });
      Ts(25, r, "SEED");
      field(30, r, 6, "seed", S.seed == null ? "RND" : String(S.seed), () => edit("seed", S.seed == null ? "" : String(S.seed), (v) => { const x = parseInt(v, 10); S.seed = isFinite(x) ? x : null; }));
      if (go) { presetRow(r - 1); goKey(56, r - 1, 24); }
    }

    // the preset, the bar, checkpoints, and INPAINT beside all three
    function inpaintArea(r) {
      presetRow(r);
      inpaintBar(r + 1, false);
      checkpointRow(r + 2);
      goKey(56, r, 24, 3);
      return 3;
    }

    // the one solid key on the display; fills with progress while running
    function goKey(c, r, w, h = 2) {
      const busy = S.busy && S.busy.what === "inpaint", ready = S.cands.some((x) => x.on);
      const f = busy ? SD.clamp(S.busy.frac || 0, 0, 1) : 1;
      const label = busy ? (S.busy.steps ? "STEP " + S.busy.step + "/" + S.busy.steps : "WORKING") : "INPAINT";
      if (vfd) {
        const n = Math.round(f * w);
        T(c, r, busy ? "#".repeat(n) + "=".repeat(w - n) : "=".repeat(w));
        const s = !ready && !busy ? "[INPAINT]" : busy ? label : ">> INPAINT <<";
        T(c + Math.floor((w - s.length) / 2), r + 1, s);
      } else {
        const x = X(c), y = Y(r), pw = X(w) - 2, ph = Y(h) - 1;
        if (!ready && !busy) {
          L.frame(x, y, pw, ph);
          L.text(x + Math.floor((pw - L.textWidth(label)) / 2), y + Math.floor((ph - 7) / 2), label);
        } else {
          L.frame(x - 2, y - 2, pw + 4, ph + 4);
          if (busy) L.frame(x, y, pw, ph);
          L.rect(x, y, Math.round(pw * f), ph);
          L.text(x + Math.floor((pw - L.textWidth(label, "big", 2)) / 2), y + Math.floor((ph - 14) / 2), label, "big", 2, 2);
        }
      }
      if (ready || busy) L.hit(X(c), Y(r), X(w), Y(h), { click: () => SD.inpaint() });
    }

    function results(r) {
      const res = SD.result();
      if (!res) return 0;
      T(0, r, "RESULT");
      const pa = res.parent;
      fit(8, r, pa ? "ALT " + res.label + ": " + (pa.pool ? "FROM ALT " + pa.label + " AT C" + (pa.ck + 1) + ", NEW GUIDES"
        : "VARIED FROM ALT " + pa.label + " AT C" + (pa.ck + 1) + ", STEP " + pa.step)
        : SD.fmt(res.a) + " - " + SD.fmt(res.b) + "  FROM " + res.from.join(" ").toUpperCase(), 39);
      Ts(48, r, "ALT");
      S.results.forEach((x, i) => key(52 + i * 3, r, 3, x.label, () => SD.pickAlt(i), S.pick === i));
      const tr = res.track, t0 = Math.max(0, res.a - S.context), t1 = Math.min(tr.dur, res.b + S.context);
      const x0 = 0, y0 = Y(r + 2), pw = X(COLS), ph = Y(5);
      const tx = (t) => x0 + Math.round(((t - t0) / (t1 - t0)) * pw);
      L.clip = [x0, y0, pw, ph];
      wave(tr, t0, t1, x0, y0, pw, ph, () => 1, gmax(S.track));
      L.clip = null;
      const xa = tx(res.a), xb = tx(res.b);
      for (let y = y0; y < y0 + ph; y += 2) { L.px(xa, y); L.px(xb, y); }
      if (vfd) { const ga = Math.round(xa / CW), gb = Math.round(xb / CW); for (let k = ga; k < gb; k++) T(k, r + 1, "="); T(ga, r + 1, "["); T(gb - 1, r + 1, "]"); T(ga + 2, r + 1, "GENERATED".slice(0, Math.max(0, gb - ga - 4))); }
      else { L.rect(xa, Y(r + 1), xb - xa, CH - 1); L.text(xa + 3, Y(r + 1) + 1, "GENERATED", "big", 0); }
      const p = SD.playing();
      if (p && p.tag.startsWith("ab:")) L.vline(tx(SD.pos()), y0, ph, 1, 2);
      ruler(t0, t1, x0, Y(r + 7) + (vfd ? 0 : 1), pw);
      const rr = r + 9;
      key(0, rr, 12, "A ORIGINAL", () => SD.playResult("orig"), tag() === "ab:orig");
      key(12, rr, 12, "B RESULT", () => SD.playResult("result"), tag() === "ab:result");
      key(46, rr, 10, "TO BANK", () => bank(res.buf, res.a, res.b - res.a, stem(res.track.name) + "_fill", "fill", S.track.name));
      key(56, rr, 12, "SAVE TRACK", () => SD.save(res.buf, res.track.name));
      key(68, rr, 12, "SAVE WIN", () => SD.save(SD.slice(res.buf, res.a, res.b), res.track.name.replace(".wav", "_window.wav")));
      return 10;
    }

    function statusLine(r) {
      fit(0, r, "> " + (S.status || ""), COLS);
    }

    // ---------- checkpoints ----------

    function checkpointRow(r) {
      T(0, r, "CHECKPOINTS");
      key(12, r, 3, "-", () => { S.keep = Math.max(0, S.keep - 1); SD.emit("change"); });
      T(S.keep > 9 ? 16 : 17, r, String(S.keep));
      key(19, r, 3, "+", () => { S.keep = Math.min(16, S.keep + 1); SD.emit("change"); });
    }

    // ---------- bank ----------

    const KINDS = { fill: "FILL", cand: "CAND", ckpt: "CKPT", own: "OWN", sel: "SEL", file: "FILE" };
    const stem = (s) => s.replace(/\.[^.]+$/, "");
    // [t0, t1) of buf with context either side of a part lined up with a gap at [a, a + len)
    const bank = (buf, a, len, name, kind, origin) =>
      SD.bankAdd(buf, a - S.context, a + len + S.context, a, len, name, kind, origin);

    function bankSection(r) {
      const files = SD.bank.files, n = files.length, shown = Math.min(n, 6);
      T(0, r, "BANK");
      const where = SD.server ? SD.bank.folder || "NO FOLDER" : "IN THIS PAGE ONLY";
      fit(6, r, (n ? n + (n > 1 ? " SIGNALS" : " SIGNAL") : "EMPTY") + "  " + where, 48);
      key(56, r, 14, "+ SELECTION", () => bank(S.track.buf, S.a, S.b - S.a, stem(S.track.name) + "_sel", "sel", S.track.name));
      key(70, r, 10, "RESCAN", () => SD.bankLoad(), false, { disabled: !SD.server });
      if (!n) return 1;
      ui.bankScroll = SD.clamp(ui.bankScroll || 0, 0, n - shown);
      for (let i = 0; i < shown; i++) {
        const k = i + ui.bankScroll, e = files[k], y = r + 1 + i, tg = "bank:" + e.id;
        T(0, y, String(k + 1).padStart(2, "0"));
        fit(3, y, e.file, 20);
        T(24, y, KINDS[e.kind] || "FILE");
        fit(29, y, e.origin ? "< " + e.origin : "", 12);
        T(42, y, SD.fmt(e.dur, 1));
        if (e.track) wave(e.track, 0, e.track.dur, X(49), Y(y), X(9), CH - 1, (t) => (e.length && t >= e.at && t < e.at + e.length ? 2 : 1), gmax(e.track));
        else { SD.bankTrack(e).catch(() => {}); L.hline(X(49), Y(y) + 6, X(9), 3); }
        key(59, y, 7, "HEAR", async () => {
          if (tag() === tg) return SD.stop();
          const tr = await SD.bankTrack(e);
          SD.playWith(tg, () => ({ buf: tr.buf, from: 0, to: tr.dur, offset: 0 }));
        }, tag() === tg);
        key(66, y, 7, "USE", async () => {
          const tr = await SD.bankTrack(e);
          ui.own = { track: tr, start: SD.bestStart(tr) };
          S.status = "windowed " + e.file + " at its closest match; ADD puts it in the pool";
          SD.emit("change");
        });
        key(73, y, 7, "DROP", () => SD.bankDrop(e));
      }
      if (shown < n) {
        const sh = Y(shown) - 2, th = Math.max(4, Math.round((sh * shown) / n)), ty = Math.round((sh - th) * ui.bankScroll / (n - shown));
        L.vline(X(COLS) - 1, Y(r + 1), sh, 2); L.rect(X(COLS) - 2, Y(r + 1) + ty, 2, th);
        L.hit(0, Y(r + 1), X(59), Y(shown), { wheel: (x, y, e) => { ui.bankScroll = SD.clamp(ui.bankScroll + Math.sign(e.deltaY), 0, n - shown); SD.redraw(); } });
      }
      return 1 + shown;
    }

    // ---------- trajectory ----------

    // a pixel line; step > 1 dots it
    function line(x0, y0, x1, y1, step = 1) {
      x0 = Math.round(x0); y0 = Math.round(y0); x1 = Math.round(x1); y1 = Math.round(y1);
      const dx = Math.abs(x1 - x0), dy = -Math.abs(y1 - y0), sx = x0 < x1 ? 1 : -1, sy = y0 < y1 ? 1 : -1;
      let err = dx + dy, i = 0;
      for (;;) {
        if (i++ % step === 0) L.px(x0, y0);
        if (x0 === x1 && y0 === y1) break;
        const e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
      }
    }
    // what the graphic shows: the guides, every alternative's path, and the
    // running job's paths
    function trajData() {
      const res = SD.result(), busy = S.busy && S.busy.what === "inpaint", live = busy ? S.live : null;
      const fresh = live && live.fresh;
      const src = live && live.guides && live.guides.length ? live : res;
      const d = { guides: (src && src.guides) || [], nums: (live || res || {}).nums || [], w: (live || res || {}).w || [], paths: [], res, live, marks: [] };
      if (!fresh) S.results.forEach((x) => { if (!res || x.layout === res.layout) d.paths.push({ pts: x.path, sel: x === res, res: x }); });
      if (live) (live.paths || []).forEach((p) => d.paths.push({ pts: p, live: true }));
      // the server sends the layout centred and scaled, with the noise level as height
      d.c = [0, 0]; d.R = 1;
      d.h = (h) => Math.min(h, 1.4);
      return d;
    }

    function trajGraphic(x0, y0, pw, ph) {
      const d = trajData();
      L.frame(x0, y0, pw, ph);
      if (!d.guides.length) return d;
      L.clip = [x0 + 1, y0 + 1, pw - 2, ph - 2];
      isoView(d, x0 + 1, y0 + 1, pw - 2, ph - 2);
      L.clip = null;
      const res = d.res;
      L.hit(x0, y0, pw, ph, {
        cursor: "grab",
        down: (x) => { ui.spin = { x, az: ui.az }; },
        move: (x) => { if (ui.spin) { ui.az = ui.spin.az + (x - ui.spin.x) / 60; SD.redraw(); } },
        up: (x, y, e, q, moved) => {
          ui.spin = null;
          if (moved) return;
          // nearest checkpoint mark
          let best = null, bd = 64;
          d.marks.forEach((m) => { const dd = (m.x - x0 - x) ** 2 + (m.y - y0 - y) ** 2; if (dd < bd) { bd = dd; best = m; } });
          if (best && res) SD.selectCk(best.res, best.j);
          if (best && best.res !== res) SD.pickAlt(S.results.indexOf(best.res));
        },
      });
      return d;
    }

    // guide marker, sized by its share
    function guideMark(x, y, share) {
      const s = share > 0.4 ? 4 : share > 0.15 ? 3 : 2;
      L.rect(x - s - 1, y - s - 1, 2 * s + 3, 2 * s + 3, 0);
      L.frame(x - s, y - s, 2 * s + 1, 2 * s + 1);
      L.rect(x - s + 2, y - s + 2, 2 * s - 3, 2 * s - 3);
    }
    // numbers beside the markers; markers this close share one label, and a
    // label that would overlap another is moved to the next free side
    function guideLabels(tops, nums, w) {
      const groups = [];
      tops.forEach(([x, y], j) => {
        const g = groups.find((q) => Math.abs(q.x - x) <= 8 && Math.abs(q.y - y) <= 8);
        if (g) g.nums.push(nums[j]); else groups.push({ x, y, nums: [nums[j]], s: (w[j] || 0) > 0.4 ? 4 : (w[j] || 0) > 0.15 ? 3 : 2 });
      });
      const boxes = [];
      groups.forEach((g) => {
        const s = g.nums.map((v) => String(v).padStart(2, "0")).join("+"), tw = L.textWidth(s);
        const tries = [[g.s + 3, -3], [-g.s - 3 - tw, -3], [-tw / 2, -g.s - 10], [-tw / 2, g.s + 3]];
        const fits = ([dx, dy]) => !boxes.some((b) => g.x + dx < b[2] && g.x + dx + tw > b[0] && g.y + dy < b[3] && g.y + dy + 7 > b[1]);
        const [dx, dy] = tries.find(fits) || tries[0];
        const bx = Math.round(g.x + dx), by = Math.round(g.y + dy);
        L.rect(bx - 1, by - 1, tw + 2, 9, 0);
        L.text(bx, by, s);
        boxes.push([bx - 1, by - 1, bx + tw + 1, by + 8]);
      });
    }
    // checkpoint marks along a path; the selected one inverted and labelled
    function ckMarks(d, p, proj) {
      if (!p.res) return;
      const r = p.res;
      r.cks.forEach((c, j) => {
        const q = p.pts.find((t) => t[0] === c.step);
        if (!q) return;
        const [x, y] = proj(q);
        d.marks.push({ x, y, res: r, j });
        if (!p.sel) return;
        if (j === r.ck) { L.rect(x - 3, y - 3, 7, 7, 0); L.frame(x - 3, y - 3, 7, 7); L.px(x, y); L.text(x + 6, y + 2, "C" + (j + 1)); }
        else L.frame(x - 1, y - 1, 3, 3);
      });
      const last = p.pts[p.pts.length - 1];
      if (last) {
        const [x, y] = proj(last);
        d.marks.push({ x, y, res: r, j: r.cks.length });
        if (p.sel && r.ck === r.cks.length) { L.rect(x - 3, y - 3, 7, 7, 0); L.frame(x - 3, y - 3, 7, 7); L.text(x + 6, y + 2, "END"); }
      }
    }
    function pathLine(pts, proj, step) {
      for (let i = 1; i < pts.length; i++) {
        const [ax, ay] = proj(pts[i - 1]), [bx, by] = proj(pts[i]);
        line(ax, ay, bx, by, step);
      }
    }

    // a floor seen from above at an angle, turning while it runs; guides
    // stand on it, paths come down from their height to where they land
    function isoView(d, x0, y0, pw, ph) {
      if (ui.az == null) ui.az = 0.6;
      if (S.busy && !ui.spin) ui.az += 0.004;
      const az = ui.az, el = 0.55, ca = Math.cos(az), sa = Math.sin(az);
      const K = Math.min(pw / 2.7, ph / 1.6), cx = x0 + pw / 2, cy = y0 + ph * 0.7;
      const P3 = (x, y, z) => {
        const xr = ca * x - sa * y, yr = sa * x + ca * y;
        return [Math.round(cx + xr * K), Math.round(cy + yr * K * Math.sin(el) - 0.6 * z * K * Math.cos(el))];
      };
      const n = (q) => [(q[1] - d.c[0]) / d.R, (q[2] - d.c[1]) / d.R];
      for (let v = -1.2; v <= 1.21; v += 0.4) {
        const [a1, b1] = P3(v, -1.2, 0), [a2, b2] = P3(v, 1.2, 0), [a3, b3] = P3(-1.2, v, 0), [a4, b4] = P3(1.2, v, 0);
        line(a1, b1, a2, b2, 4); line(a3, b3, a4, b4, 4);
      }
      const tops = d.guides.map((g, j) => {
        const [gx, gy] = n([0, g[0], g[1]]), [bx, by] = P3(gx, gy, 0), [tx, ty] = P3(gx, gy, 0.25 + 0.5 * (d.w[j] || 0));
        line(bx, by, tx, ty);
        guideMark(tx, ty, d.w[j] || 0);
        return [tx, ty];
      });
      d.paths.forEach((p) => {
        const proj = (q) => { const [x, y] = n(q); return P3(x, y, d.h(q[3])); };
        const shadow = (q) => { const [x, y] = n(q); return P3(x, y, 0); };
        if (p.sel || p.live) {
          p.pts.forEach((q, i) => { if (i % 2 === 0) { const [x, y] = shadow(q); L.px(x, y); } });
          if (p.res) p.res.cks.forEach((c) => { const q = p.pts.find((t) => t[0] === c.step); if (q) { const [ax, ay] = proj(q), [bx, by] = shadow(q); line(ax, ay, bx, by, 2); } });
        }
        pathLine(p.pts, proj, p.sel || p.live ? 1 : 3);
        const q = p.pts[p.pts.length - 1];
        if (q && (p.sel || p.live)) { const [x, y] = proj(q); L.rect(x - 1, y - 2, 3, 5); L.rect(x - 2, y - 1, 5, 3); }
        ckMarks(d, p, proj);
      });
      // labels last, so paths do not cross them out
      guideLabels(tops, d.nums, d.w);
    }

    // checkpoints of the picked result as a strip of cells, and what to do with one
    function ckBrowser(c, r, res) {
      const busy = S.busy && S.busy.what === "inpaint";
      if (!res) {
        T(c, r, "SAMPLING");
        if (S.busy && S.busy.steps) T(c, r + 1, "STEP " + S.busy.step + "/" + S.busy.steps);
        return;
      }
      const total = res.cks.length + 1, m = Math.min(total, 10), cw = Math.min(10, Math.floor(40 / m));
      T(c, r, "CHECKPOINTS");
      T(c + 12, r, res.cks.length ? res.cks.length + " OF ALT " + res.label : "NONE KEPT");
      ui.ckScroll = SD.clamp(ui.ckScroll || 0, Math.max(0, res.ck - m + 1), Math.min(res.ck, total - m));
      const len = res.b - res.a, t0 = Math.max(0, res.a - 0.25 * len), t1 = Math.min(res.track.dur, res.b + 0.25 * len);
      for (let i = 0; i < m; i++) {
        const j = i + ui.ckScroll, ck = res.cks[j], x = X(c + i * cw), y = Y(r + 1), w = X(cw) - 2, h = Y(3) + CH - 2;
        const lab = ck ? "C" + (j + 1) : "END";
        L.text(x + Math.floor((w - L.textWidth(lab)) / 2), y + 2, lab);
        const tr = ck ? ck.track : res.track, wy = y + CH, wh = Y(3) - 2;
        if (tr) { L.clip = [x + 1, wy, w - 2, wh]; wave(tr, t0, t1, x + 1, wy, w - 2, wh, (t) => (t >= res.a && t < res.b ? 1 : 0), gmax(S.track)); L.clip = null; }
        else {
          // not decoded yet: stipple as dense as the noise left
          const f = SD.clamp(ck.sigma / 2, 0.05, 1);
          for (let yy = 0; yy < wh; yy += 2) for (let xx = (yy / 2) % 2; xx < w - 2; xx += 2) if (((xx * 7 + yy * 13) % 10) / 10 < f) L.px(x + 1 + xx, wy + yy);
        }
        if (ck && ck.owner !== res.label) { L.hline(x, y, w, 2); L.hline(x, y + h - 1, w, 2); L.vline(x, y, h, 2); L.vline(x + w - 1, y, h, 2); }
        else L.frame(x, y, w, h);
        if (j === res.ck) L.invert(x + 1, y + 1, w - 2, CH - 1);
        L.hit(x, y, X(cw), h, { click: () => SD.selectCk(res, j) });
      }
      L.hit(X(c), Y(r + 1), X(40), Y(4), { wheel: (x, y, e) => SD.selectCk(res, res.ck + Math.sign(e.deltaY)) });
      const ck = res.cks[res.ck];
      T(c, r + 5, ck ? "C" + (res.ck + 1) + "  STEP " + ck.step + "/" + res.steps + "  SIGMA " + ck.sigma.toFixed(3) + (ck.owner !== res.label ? "  AS ALT " + ck.owner : "")
        : "END  STEP " + res.steps + "/" + res.steps + "  FINISHED");
      key(c, r + 6, 4, "<", () => SD.selectCk(res, res.ck - 1));
      key(c + 4, r + 6, 4, ">", () => SD.selectCk(res, res.ck + 1));
      key(c + 9, r + 6, 8, "HEAR", () => SD.playCk(res), tag() === "ck");
      key(c + 17, r + 6, 8, "SAVE", async () => { const b = await SD.ckBuf(res, res.ck); SD.save(SD.slice(b, res.a, res.b), stem(res.track.name) + (ck ? "_c" + (res.ck + 1) : "") + "_window.wav"); });
      key(c + 25, r + 6, 8, "BANK", async () => { const b = await SD.ckBuf(res, res.ck); bank(b, res.a, res.b - res.a, stem(res.track.name) + (ck ? "_c" + (res.ck + 1) : ""), ck ? "ckpt" : "fill", S.track.name); });

      const v = ui.vary || (ui.vary = { n: 2, seed: null, pool: false });
      T(c, r + 8, ck ? "VARY FROM C" + (res.ck + 1) : "VARY");
      const gl = "GUIDES:" + (v.pool ? "POOL" : "AS RUN");
      T(c + 40 - gl.length, r + 8, gl);
      L.hit(X(c + 40 - gl.length), Y(r + 8), X(gl.length), CH, { click: () => { v.pool = !v.pool; SD.redraw(); } });
      const ticked = S.cands.filter((x) => x.on).length;
      T(c, r + 9, "ALT");
      key(c + 4, r + 9, 3, "-", () => { v.n = Math.max(1, v.n - 1); SD.redraw(); });
      T(c + 8, r + 9, String(v.n));
      key(c + 10, r + 9, 3, "+", () => { v.n = Math.min(4, v.n + 1); SD.redraw(); });
      T(c + 14, r + 9, "SEED");
      field(c + 19, r + 9, 5, "vseed", v.seed == null ? "RND" : String(v.seed), () => edit("vseed", v.seed == null ? "" : String(v.seed), (s) => { const x = parseInt(s, 10); v.seed = isFinite(x) ? x : null; }));
      key(c + 25, r + 9, 15, busy ? "VARYING" : "VARY", () => SD.vary(res, res.ck, v.n, v.seed, v.pool), busy,
        { disabled: !ck || (busy && !S.live) || (v.pool && !ticked) });
    }

    function trajectory(r) {
      const res = SD.result(), d0 = S.live && S.busy ? S.live : res;
      T(0, r, "TRAJECTORY");
      if (d0) {
        const k = (d0.nums || []).length;
        const alts = S.results.length;
        fit(12, r, (res ? "ALT " + res.label + " OF " + alts + "  " : "") + (d0.steps || "--") + " STEPS  " + k + " GUIDE" + (k === 1 ? "" : "S"), 44);
      }
      trajGraphic(0, Y(r + 1) + 2, X(38), Y(12) - 4);
      ckBrowser(40, r + 1, res);
      return 13;
    }

    // ---------- layouts ----------

    function panel() {
      const n = S.cands.length, poolH = n ? 2 * n : 2, ownH = ui.own ? 9 : 1, resH = SD.result() ? 11 : 0;
      const nb = SD.bank.files.length, bankH = nb ? 1 + Math.min(6, nb) : 2;
      const trajH = SD.result() || (S.busy && S.busy.what === "inpaint") ? 14 : 0;
      const setH = ui.settings ? 1 + SET_H : 0;
      const rows = 1 + 1 + 20 + 1 + 2 + 1 + 2 + 1 + 2 + poolH + 1 + ownH + 1 + bankH + 1 + 1 + 3 + 1 + setH + resH + trajH + 2;
      L.resize(Y(rows)); L.clear();
      let r = 0;
      statusBar(r); r += 2;
      if (!S.track) { T(2, r + 2, "NO RECORDING"); fileKeys(2, r + 4); return; }
      overview(0, r, 62, 2);
      detail(0, r + 2, 62, 16);
      readouts(64, r);
      fileKeys(64, r + 16);
      curveStrip(S.track, S.view[0], S.view[1], 0, r + 18, 62, 2, (t) => SD.addCustom(S.track, t), poolStarts(S.track), mainHover);
      const hv = ui.hoverT != null ? SD.curveAt(S.track, ui.hoverT) : NaN;
      T(64, r + 18, "MATCH " + (isNaN(hv) ? "--" : hv.toFixed(2)));
      r += 20;
      lineH(r); r += 1;
      softKeys(r, trimKeys()); r += 2;
      lineH(r); r += 1;
      command(r); r += 2;
      lineH(r); r += 1;
      poolHeader(r); r += 2;
      r += poolRows(r);
      lineH(r); r += 1;
      r += ownSection(r);
      lineH(r); r += 1;
      r += bankSection(r);
      lineH(r); r += 1;
      r += 1; r += inpaintArea(r);
      if (ui.settings) { lineH(r); r += 1; r += settingsSection(r); }
      if (SD.result() || (S.busy && S.busy.what === "inpaint")) { lineH(r); r += 1; r += trajectory(r); }
      if (SD.result()) { lineH(r); r += 1; r += results(r); }
      statusLine(rows - 1);
    }

    // fixed 80 x 27 cells, three pages, soft keys on the bottom two rows
    function pagedLayout() {
      const rows = 27;
      L.resize(Y(rows)); L.clear();
      statusBar(0);
      const pages = [["trim", "TRIM"], ["pool", "POOL"], ["result", "RESULT"]];
      pages.forEach(([id, lab], i) => {
        const c = 1 + i * 10;
        T(c + 1, 1, lab);
        if (ui.page === id) { if (vfd) T(c, 1, ">"); else inv(c, 1, 9); }
        L.hit(X(c), Y(1), X(9), Y(1), { click: () => { ui.page = id; SD.redraw(); } });
      });
      fit(32, 1, "> " + (S.status || ""), 48);
      if (!S.track) { T(2, 4, "NO RECORDING"); fileKeys(2, 6); return softKeys(rows - 2, []); }
      let keys;
      if (ui.page === "trim") {
        overview(0, 3, 62, 2);
        detail(0, 5, 62, 16);
        readouts(64, 3);
        curveStrip(S.track, S.view[0], S.view[1], 0, 21, 62, 2, (t) => SD.addCustom(S.track, t), poolStarts(S.track), mainHover);
        command(23);
        keys = trimKeys();
        keys[7] = { label: "POOL >", act: () => { ui.page = "pool"; SD.redraw(); } };
      } else if (ui.page === "pool") {
        poolHeader(3);
        poolRows(5, 6);
        ownRow(18);
        inpaintBar(21);
        keys = [
          { label: "LOAD", act: () => fileIn.click() },
          { label: "ADD OWN", act: () => ownIn.click() },
          { label: "ALL", act: () => { S.cands.forEach((c) => (c.on = true)); SD.emit("change"); } },
          { label: "NONE", act: () => { S.cands.forEach((c) => (c.on = false)); SD.emit("change"); } },
          { label: "SEARCH", act: () => SD.search(), on: !!(S.busy && S.busy.what === "search") },
          { label: "SAVE SEL", act: SD.saveSelected },
          { label: "INPAINT", act: async () => { await SD.inpaint(); if (SD.result()) ui.page = "result"; SD.redraw(); }, on: !!(S.busy && S.busy.what === "inpaint") },
          { label: "< TRIM", act: () => { ui.page = "trim"; SD.redraw(); } },
        ];
      } else {
        if (!results(3)) T(2, 5, "NOTHING YET");
        keys = [
          { label: "A ORIG", act: () => SD.playResult("orig"), on: tag() === "ab:orig" },
          { label: "B RESULT", act: () => SD.playResult("result"), on: tag() === "ab:result" },
          null, null, null, null, null,
          { label: "< POOL", act: () => { ui.page = "pool"; SD.redraw(); } },
        ];
      }
      softKeys(rows - 2, keys);
      ui.keys = keys;
    }

    SD.on("track", () => SD.zoomGap());
    SD.onDraw(() => {
      if (paged) pagedLayout(); else panel();
      L.paint(opts.look);
      if (opts.after) opts.after(ui);
    });

    // physical keys under the paged display press the soft key above them
    GS1.press = (i) => { const k = ui.keys && ui.keys[i]; if (k && !k.disabled) k.act(); };
    document.addEventListener("keydown", (e) => {
      if (e.target === inp) return;
      if (/^F[1-8]$/.test(e.key) && paged) { e.preventDefault(); GS1.press(+e.key.slice(1) - 1); }
    });

    SD.boot();
    return { ui, lcd: L };
  };
})();
