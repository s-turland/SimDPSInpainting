"""Web interface, started with ``simdps app``.

A small HTTP server on localhost serves the page in ``simdps/web`` and runs
searches and inpainting for it. It needs nothing beyond the standard library.

    GET  /api/info                model state
    POST /api/audio?name=...      upload a recording, returns its id
    POST /api/search              {track, gap, match, n, source}; source "bank" searches the bank
    POST /api/inpaint             {track, gap, guides, samples, preset, method, settings, seed, checkpoints},
                                  or {restart: {job, index, sample}, samples, seed}; returns a job id
    GET  /api/job/<id>            progress, trajectory, checkpoints, and result ids when done
    GET  /api/result/<id>         a result as WAV
    GET  /api/checkpoint/<job>/<index>/<sample>    a checkpoint, spliced in, as WAV
    GET  /api/settings?preset=&method=   the settings a run would use, by config section
    GET  /api/bank                files in the bank
    POST /api/bank?name=&kind=&origin=&at=&length= add a WAV to the bank
    GET  /api/bank/<file>         a bank file
    POST /api/bank/<file>/drop    move a file to the bank's dropped/ folder

Bank files can be used as guides with the id ``bank:<file>``.
"""

import io
import json
import mimetypes
import os
import re
import shutil
import threading
import time
import traceback
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
import soundfile as sf
import torch

import simdps
from simdps.audio import resample
from simdps.config import SECTIONS
from simdps.inpainter import AUDIO_EXTENSIONS, Checkpoint

WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
# "paper" runs the settings of the paper; the others are configs in configs/speed
PRESETS = ("paper", "warm_start_detailed", "warm_start_fast")


def _bank():
    # bank/ beside the code in a source checkout, else in the home folder
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.isfile(os.path.join(root, "pyproject.toml")):
        return os.path.join(root, "bank")
    return os.path.join(os.path.expanduser("~"), "simdps-bank")


BANK = _bank()


def launch(model="acestep", device=None, port=7860, browser=True, bank=BANK):
    """Load ``model`` in the background and serve the interface on ``port``."""
    state = State(bank=bank)
    threading.Thread(target=state.load, args=(model, device), daemon=True).start()
    server = serve(state, port)
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"SimDPS on {url}  (ctrl+c to stop)")
    if browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def serve(state, port=0):
    """HTTP server for ``state`` on localhost; ``port=0`` picks a free one."""
    handler = type("Handler", (Handler,), {"state": state})
    return ThreadingHTTPServer(("127.0.0.1", port), handler)


def _settings(given):
    # settings sent by the page, limited to the keys of the config sections
    out = {}
    for section, values in (given or {}).items():
        if section not in SECTIONS or not isinstance(values, dict):
            raise ValueError(f"unknown settings section '{section}'")
        for k, v in values.items():
            if k not in SECTIONS[section] or not (v is None or isinstance(v, (bool, int, float, str))):
                raise ValueError(f"unknown setting '{section}.{k}'")
        out[section] = dict(values)
    return out or None


def _wav(y, sr):
    buf = io.BytesIO()
    sf.write(buf, y.T.numpy(), sr, format="WAV", subtype="FLOAT")
    return buf.getvalue()


class Trajectory:
    """Paths of the samples among the guides, step by step, for the display.

    At each step the guides and the samples' current estimates are laid out by
    classical scaling of their distances, turned and scaled to line up with
    the previous layout, and every earlier estimate is placed in it. The paths
    part as the samples do, and end where the results lie. Height is the noise
    level. Coordinates are centred on the guides and lie roughly in [-1, 1].
    """

    def __init__(self, order=None, prefix=None, sigma_min=1e-3, ease=0.15):
        self.order = order  # request index of each guide, as the inpainter has them
        self.prefix = prefix or []  # (step, sigma, estimate) of the sample a variation starts from
        self.sigma_min = sigma_min
        self.ease = ease
        self.steps, self.sigmas, self.hist = [], [], []  # hist: (samples, D) per step
        self.guides, self.paths = [], []
        self._g = self._prev = self._scale = None

    def __call__(self, i, sigma, x0, guides):
        x = x0.flatten(1).float().cpu().numpy()
        if self._g is None:
            g = guides if guides is not None else torch.zeros_like(x0[:1])
            self._g = g.flatten(1).float().cpu().numpy()
            for step, s, e in self.prefix:
                self.steps.append(step)
                self.sigmas.append(s)
                self.hist.append(np.repeat(e[None], len(x), 0))
        self.steps.append(i + 1)
        self.sigmas.append(float(sigma))
        self.hist.append(x)
        self._layout(x)

    def history(self, sample, step):
        """(step, sigma, estimate) of ``sample`` up to ``step``, to start variations from."""
        return [(t, s, h[sample]) for t, s, h in zip(self.steps, self.sigmas, self.hist) if t <= step]

    def _layout(self, x):
        g, k = self._g, len(self._g)
        A = np.concatenate([g, x])
        P, place = _scaling(_sq_dist(A, A))
        if self._prev is not None:
            # the guides should move least
            w = np.r_[np.full(k, 3.0), np.ones(len(x))]
            P, place = _aligned(P, place, self._prev, w)
        self._prev = P
        H = np.stack(self.hist)  # (steps, samples, D)
        T, S = H.shape[:2]
        Y = place(_sq_dist(H.reshape(T * S, -1), A)).reshape(T, S, 2)
        for t in range(1, T):
            # early estimates jump about; the path eases after them
            Y[t] = Y[t - 1] + self.ease * (Y[t] - Y[t - 1])
        s = np.log(np.maximum(self.sigmas, self.sigma_min) / self.sigma_min)
        h = 1.2 * np.clip(s / max(s[0], 1e-9), 0, 1)
        c = P[:k].mean(0)
        R = max(np.hypot(*(P - c).T).max(), np.hypot(*(Y - c).reshape(-1, 2).T).max(), 1e-9)
        # the zoom follows slowly, so the picture does not pump
        self._scale = R if self._scale is None else max(R, 0.9 * self._scale)
        G, Y = (P[:k] - c) / self._scale, (Y - c) / self._scale
        back = list(range(k))
        for j, r in enumerate(self.order or range(k)):
            back[r] = j
        self.guides = [[round(float(v), 4) for v in G[j]] for j in back]
        self.paths = [[[step, round(float(Y[t, s_, 0]), 4), round(float(Y[t, s_, 1]), 4), round(float(h[t]), 4)]
                       for t, step in enumerate(self.steps)] for s_ in range(S)]

    def view(self):
        return {"guides": self.guides, "paths": self.paths}


def _sq_dist(X, A):
    # mean squared distances between rows
    d = (X ** 2).sum(1)[:, None] + (A ** 2).sum(1)[None] - 2 * X @ A.T
    return np.maximum(d, 0) / X.shape[1]


def _scaling(D2):
    # classical scaling to two dimensions, and the map placing new points by their distances
    k = len(D2)
    J = np.eye(k) - 1 / k
    lam, V = np.linalg.eigh(-0.5 * J @ D2 @ J)
    lam, V = lam[::-1][:2], V[:, ::-1][:, :2]
    if len(lam) < 2:
        lam, V = np.r_[lam, 0.0], np.c_[V, np.zeros(k)]
    on = lam > 1e-6 * max(abs(lam).max(), 1e-12)
    P = V * np.sqrt(np.clip(lam, 0, None))
    proj = np.where(on, 0.5 / np.sqrt(np.clip(lam, 1e-12, None)), 0.0) * V
    mean = D2.mean(0)
    return P, lambda d2: (mean - d2) @ proj


def _aligned(P, place, Q, w):
    # rotation or reflection and scale taking P onto Q, weighted by w
    w = w / w.sum()
    pc, qc = w @ P, w @ Q
    U, s, Vt = np.linalg.svd(((P - pc) * w[:, None]).T @ (Q - qc))
    R = U @ Vt
    scale = s.sum() / max((w * ((P - pc) ** 2).sum(1)).sum(), 1e-12)
    def f(X):
        return (X - pc) @ R * scale + qc

    return f(P), lambda d2: f(place(d2))


class State:
    """Server state: the model, uploaded recordings, jobs and the bank."""

    def __init__(self, inpainter=None, name=None, bank=None):
        self.inpainter = inpainter
        self.name = name
        self.error = None
        self.bank = bank
        self.audio = {}  # id -> (Tensor (channels, length), sample rate, name)
        self.jobs = {}
        self.runs = {}  # job id -> request, Result and input, for checkpoints and restarts
        self.results = {}  # id -> WAV bytes
        self.lock = threading.Lock()  # one model call at a time

    def load(self, model, device=None):
        try:
            self.inpainter = simdps.load(model, device=device)
            self.name = model
        except Exception as e:  # shown on the page
            traceback.print_exc()
            self.error = f"{type(e).__name__}: {e}"

    def info(self):
        inp = self.inpainter
        out = {"ready": inp is not None, "model": self.name, "error": self.error, "bank": self.bank}
        if inp is not None:
            p = inp.prior
            out.update(sample_rate=p.sample_rate, max_window=p.length * p.hop_length / p.sample_rate)
        return out

    def add(self, data, name):
        x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        key = uuid.uuid4().hex[:12]
        self.audio[key] = (torch.from_numpy(x.T.copy()), sr, name)
        return key

    def track(self, key):
        if key not in self.audio and key.startswith("bank:"):
            x, sr = sf.read(self._bank_path(key[5:]), dtype="float32", always_2d=True)
            self.audio[key] = (torch.from_numpy(x.T.copy()), sr, key[5:])
        if key not in self.audio:
            raise ValueError("unknown recording; reload the page")
        return self.audio[key]

    def settings(self, preset="paper", method=None):
        """Settings a run with ``preset`` and ``method`` would use, by section."""
        if preset not in PRESETS:
            raise ValueError(f"unknown preset '{preset}'")
        if method not in (None, "multi", "solo"):
            raise ValueError(f"unknown method '{method}'")
        cfg = self.inpainter.settings(method, None if preset == "paper" else preset)
        return {"method": cfg.get("method"), **{s: cfg.get(s) or {} for s in SECTIONS}}

    def search(self, req):
        x, sr, _ = self.track(req["track"])
        src, names = None, {}
        if req.get("source") == "bank":
            files = [e["file"] for e in self.bank_list()]
            if not files:
                raise ValueError("the bank is empty")
            src = [self._bank_path(f) for f in files]
            names = {p: "bank:" + f for p, f in zip(src, files)}
        elif req.get("source"):
            s, ssr, _ = self.track(req["source"])
            src = resample(s, ssr, sr)
        with self.lock:
            found = self.inpainter.search(x, sr, tuple(req["gap"]), source=src, n=int(req.get("n", 8)),
                                          match=req.get("match", "latent"))
        out = []
        for t, c, where in found:
            key = None if where is None else names.get(where, req.get("source"))
            out.append({"start": t, "score": c, "source": key})
        return {"candidates": out}

    def start(self, req):
        restart = parent = None
        if req.get("restart"):
            r = req["restart"]
            run = self.runs.get(r["job"])
            if run is None:
                raise ValueError("unknown job to restart")
            c = run["result"].checkpoints[int(r["index"])]
            s = int(r.get("sample", 0))
            restart = Checkpoint(c.step, c.sigma, c.latents[s : s + 1])
            # the run's own guides unless new ones are given
            same = "guides" not in req
            parent = {"job": r["job"], "index": int(r["index"]), "sample": s, "step": c.step, "same_guides": same}
            new = {"guides": req["guides"]} if "guides" in req else {}
            req = {**run["req"], "samples": req.get("samples", 1), "seed": req.get("seed"), **new}
        x, sr, _ = self.track(req["track"])
        guides = []
        for g in req.get("guides", []):
            src = None
            if g.get("source") and g["source"] != req["track"]:
                s_, ssr, _ = self.track(g["source"])
                src = resample(s_, ssr, sr)
            guides.append((float(g["start"]), src, g.get("weight")))
        if not guides:
            raise ValueError("choose at least one candidate")
        key = uuid.uuid4().hex[:12]
        preset = req.get("preset", "paper")
        if preset not in PRESETS:
            raise ValueError(f"unknown preset '{preset}'")
        method = req.get("method") or ("solo" if len(guides) == 1 else None)
        if method not in (None, "multi", "solo"):
            raise ValueError(f"unknown method '{method}'")
        config = [None if preset == "paper" else preset, _settings(req.get("settings"))]
        n = max(1, min(4, int(req.get("samples", 1))))
        weights = [w for _, _, w in guides]
        # the inpainter puts weighted guides heaviest first
        order = sorted(range(len(guides)), key=lambda j: -weights[j]) if None not in weights else None
        prefix = None
        if restart is not None and parent["same_guides"]:
            # variations continue the parent's path; new guides change the layout,
            # so the old path would not fit
            prefix = run["trace"].history(s, restart.step)
        trace = Trajectory(order, prefix)
        first = restart.step if restart is not None else 0
        job = {"phase": "waiting", "step": first, "first": first,
               "steps": self.inpainter.schedule(config),
               "done": False, "error": None, "results": [], "checkpoints": [], "trace": trace.view(),
               "parent": parent}
        self.jobs[key] = job

        def run_job():
            def progress(i, *_):
                job["phase"], job["step"] = "sampling", i + 1
                job["trace"] = trace.view()

            try:
                with self.lock:
                    job["phase"] = "encoding"
                    seed = req.get("seed")
                    out = self.inpainter.inpaint(
                        x, sr, tuple(req["gap"]), guide=[(t, src) for t, src, _ in guides],
                        weights=weights, n_samples=n, config=config, method=method,
                        seed=None if seed is None else int(seed), callback=progress,
                        checkpoints=int(req.get("checkpoints", 0)), restart=restart, trace=trace)
                    self.runs[key] = {"req": req, "result": out, "x": x, "sr": sr, "trace": trace}
                for y in out.audio:
                    rid = uuid.uuid4().hex[:12]
                    self.results[rid] = _wav(y, out.sample_rate)
                    job["results"].append(rid)
                job["checkpoints"] = [{"step": c.step, "sigma": c.sigma} for c in out.checkpoints]
                job["trace"] = trace.view()
                job["phase"] = "done"
            except Exception as e:
                traceback.print_exc()
                job["error"] = str(e)
            job["done"] = True

        threading.Thread(target=run_job, daemon=True).start()
        return {"id": key}

    def checkpoint(self, job, index, sample):
        name = f"{job}/{index}/{sample}"
        if name not in self.results:
            run = self.runs.get(job)
            if run is None:
                raise ValueError("unknown job")
            c = run["result"].checkpoints[index]
            with self.lock:
                y = self.inpainter.decode(c.latents[sample : sample + 1], run["x"], run["sr"], run["result"])
            self.results[name] = _wav(y[0], self.inpainter.prior.sample_rate)
        return self.results[name]

    # ---------- bank: a folder of audio files, with what is known of each in bank.json ----------

    def _index(self):
        try:
            with open(os.path.join(self.bank, "bank.json"), encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError, TypeError):
            return {}

    def _write_index(self, meta):
        with open(os.path.join(self.bank, "bank.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=1)

    def _bank_path(self, name):
        if not self.bank or os.path.basename(name) != name or not name.lower().endswith(AUDIO_EXTENSIONS):
            raise ValueError("not a bank file")
        path = os.path.join(self.bank, name)
        if not os.path.isfile(path):
            raise ValueError("no such bank file")
        return path

    def bank_list(self):
        if not self.bank or not os.path.isdir(self.bank):
            return []
        meta, out = self._index(), []
        for f in os.listdir(self.bank):
            path = os.path.join(self.bank, f)
            if not os.path.isfile(path) or not f.lower().endswith(AUDIO_EXTENSIONS):
                continue
            try:
                info = sf.info(path)
            except RuntimeError:
                continue
            m = meta.get(f, {})
            out.append({"file": f, "id": "bank:" + f, "kind": m.get("kind", "file"), "origin": m.get("origin", ""),
                        "at": m.get("at", 0.0), "length": m.get("length"), "dur": info.duration,
                        "made": m.get("made", os.path.getmtime(path))})
        return sorted(out, key=lambda e: e["made"])

    def bank_add(self, data, name, kind="file", origin="", at=0.0, length=None):
        if not self.bank:
            raise ValueError("no bank folder")
        sf.info(io.BytesIO(data))  # raises if this is not audio
        os.makedirs(self.bank, exist_ok=True)
        stem = re.sub(r"[^\w.-]+", "_", os.path.splitext(name)[0]).strip("_") or "signal"
        n = 1
        while os.path.exists(os.path.join(self.bank, f"{stem}_{n:02d}.wav")):
            n += 1
        f = f"{stem}_{n:02d}.wav"
        with open(os.path.join(self.bank, f), "wb") as out:
            out.write(data)
        meta = self._index()
        meta[f] = {"kind": kind, "origin": origin, "at": at, "length": length, "made": time.time()}
        self._write_index(meta)
        return f

    def bank_drop(self, name):
        path = self._bank_path(name)
        gone = os.path.join(self.bank, "dropped")
        os.makedirs(gone, exist_ok=True)
        shutil.move(path, os.path.join(gone, name))
        meta = self._index()
        if meta.pop(name, None) is not None:
            self._write_index(meta)
        self.audio.pop("bank:" + name, None)


class Handler(BaseHTTPRequestHandler):
    state = None

    def log_message(self, *args):
        pass

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, e, code=400):
        self._send(code, {"error": str(e)})

    def _same_origin(self):
        # stop other web pages from driving the server, directly (Origin) or
        # through a domain that resolves to localhost (Host)
        host, origin = self.headers.get("Host", ""), self.headers.get("Origin")
        local = urlparse("//" + host).hostname in ("127.0.0.1", "localhost")
        return local and (origin is None or urlparse(origin).netloc == host)

    def do_GET(self):
        if not self._same_origin():
            return self._error("forbidden", 403)
        path = unquote(urlparse(self.path).path)
        st = self.state
        try:
            if path == "/api/info":
                return self._send(200, st.info())
            if path.startswith("/api/job/"):
                job = st.jobs.get(path.rsplit("/", 1)[1])
                return self._send(200, job) if job else self._error("unknown job", 404)
            if path.startswith("/api/result/"):
                data = st.results.get(path.rsplit("/", 1)[1])
                return self._send(200, data, "audio/wav") if data else self._error("unknown result", 404)
            if path.startswith("/api/checkpoint/"):
                job, index, sample = path.split("/")[3:6]
                return self._send(200, st.checkpoint(job, int(index), int(sample)), "audio/wav")
            if path == "/api/bank":
                return self._send(200, {"folder": st.bank, "files": st.bank_list()})
            if path == "/api/settings":
                if st.inpainter is None:
                    return self._error(st.error or "the model is still loading", 503)
                q = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
                return self._send(200, st.settings(q.get("preset", "paper"), q.get("method") or None))
            if path.startswith("/api/bank/"):
                with open(st._bank_path(path[len("/api/bank/"):]), "rb") as f:
                    return self._send(200, f.read(), mimetypes.guess_type(f.name)[0] or "audio/wav")
        except (ValueError, IndexError, KeyError) as e:
            return self._error(e, 404)
        name = "index.html" if path in ("/", "") else path.lstrip("/")
        full = os.path.normpath(os.path.join(WEB, name))
        if not full.startswith(WEB + os.sep) or not os.path.isfile(full):
            return self._error("not found", 404)
        with open(full, "rb") as f:
            self._send(200, f.read(), mimetypes.guess_type(full)[0] or "application/octet-stream")

    def do_POST(self):
        url = urlparse(self.path)
        path = unquote(url.path)
        st = self.state
        data = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if not self._same_origin():
            return self._error("forbidden", 403)
        try:
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            if path == "/api/audio":
                return self._send(200, {"id": st.add(data, q.get("name", "audio"))})
            if path == "/api/bank":
                length = q.get("length")
                f = st.bank_add(data, q.get("name", "signal"), q.get("kind", "file"), q.get("origin", ""),
                                float(q.get("at", 0)), None if length is None else float(length))
                return self._send(200, {"file": f, "id": "bank:" + f})
            if path.startswith("/api/bank/") and path.endswith("/drop"):
                st.bank_drop(path[len("/api/bank/"):-len("/drop")])
                return self._send(200, {"ok": True})
            if st.inpainter is None:
                return self._error(st.error or "the model is still loading", 503)
            req = json.loads(data or b"{}")
            if path == "/api/search":
                return self._send(200, st.search(req))
            if path == "/api/inpaint":
                return self._send(200, st.start(req))
            self._error("not found", 404)
        except (ValueError, KeyError, IndexError, RuntimeError) as e:  # bad requests and undecodable audio
            self._error(e)
