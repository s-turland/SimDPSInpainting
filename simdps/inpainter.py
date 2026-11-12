"""Gap inpainting with a pretrained prior: search, guided sampling, splicing."""

import hashlib
import math
import os
from collections import OrderedDict
from dataclasses import dataclass, field

import torch

from .audio import crossfade, load_audio, resample
from .config import load_config, merge, with_options
from .guidance import Candidates, Guidance, Observation, gap_mask
from .sampler import EDMSampler
from .search import BandFeature, ChromaFeature, SearchResult, STFTFeature, latent_search, waveform_search

# Config of each method.
METHODS = {
    "multi": "ismir26_multi",
    "solo": "ismir26_solo",
    "dps": "ismir26_dps",
    "search": "ismir26_search",
}

AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg", ".aif", ".aiff")


@dataclass
class Result:
    """Output of :meth:`Inpainter.inpaint`.

    Attributes
    ----------
    audio : Tensor
        ``(n_samples, channels, length)``: the input with the gap filled, or
        the generated window if ``raw=True``.
    sample_rate : int
    window : tuple of int
        Samples ``[start, end)`` of the input covered by the generated window.
    gap : tuple of int
        Gap samples ``[start, end)`` in the input.
    candidates : SearchResult or None
        Search result, cropped to the window, in the prior's domain.
    latents : Tensor
        Generated window in the prior's domain, before decoding.
    checkpoints : list of Checkpoint
        Estimates kept along the way, if ``checkpoints`` was given.
    """

    audio: torch.Tensor
    sample_rate: int
    window: tuple
    gap: tuple
    candidates: SearchResult = None
    latents: torch.Tensor = field(default=None, repr=False)
    checkpoints: list = field(default_factory=list, repr=False)


@dataclass
class Checkpoint:
    """Estimate of the finished window part way through sampling.

    Attributes
    ----------
    step : int
        Steps taken; ``restart`` resumes from here.
    sigma : float
        Noise level after those steps.
    latents : Tensor
        ``(n_samples, channels, frames)`` estimate, padding included. Decode
        it with :meth:`Inpainter.decode`.
    """

    step: int
    sigma: float
    latents: torch.Tensor = field(repr=False)


class Inpainter:
    """Fills gaps in recordings with a diffusion prior guided by similarity search.

    Parameters
    ----------
    prior : Prior
        Pretrained model, e.g. :class:`simdps.priors.ACEStep`.
    config : optional
        Settings of every call, as taken by :func:`simdps.config.load_config`.
        Anything that is not a whole experiment goes on top of SimDPS-Multi,
        with the noise range of the prior.
    buffer : int, optional
        Frames regenerated on either side of the gap, which absorb the coarse
        time resolution of latent models. 0 for waveform models.
    context, crossfade : int, optional
        Context and crossfade lengths of the latent search, in frames.
        Waveform models (``hop_length == 1``) use
        :func:`~simdps.search.waveform_search` instead.
    zeta : float, optional
        Weight of the observation term.
    cache_dir : str or None
        Where encoded tracks are kept between sessions. Defaults to
        ``$SIMDPS_CACHE`` or ``~/.cache/simdps``; ``None`` keeps them in
        memory only.
    """

    def __init__(self, prior, config=None, buffer=None, context=None, crossfade=None, zeta=None, cache_dir="default"):
        self.prior = prior
        self.config = load_config(config)
        if "method" not in self.config:
            base = load_config(METHODS["multi"])
            del base["sampler"]["sigma_min"], base["sampler"]["sigma_max"]
            if prior.hop_length == 1:
                base["buffer"] = 0
            self.config = merge(base, self.config)
        given = dict(buffer=buffer, context=context, crossfade=crossfade, zeta=zeta)
        self.config = with_options(self.config, {k: v for k, v in given.items() if v is not None})
        if cache_dir == "default":
            cache_dir = os.environ.get("SIMDPS_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "simdps"))
        self.cache_dir = cache_dir
        self._cache = OrderedDict()

    def encode(self, audio, seed=None):
        """Encode ``(channels, samples)`` audio, reusing cached results.

        The search needs the whole track in the prior's domain, which is slow
        for long recordings. ``seed`` is part of the cache key since encoders
        may sample.
        """
        return self._encode(hashlib.sha1(audio.numpy().tobytes()), seed, lambda: audio)

    def encode_file(self, path, seed=None):
        """Same as :meth:`encode` for an audio file, which is only read on a cache miss."""
        st = os.stat(path)
        h = hashlib.sha1(f"{os.path.abspath(path)}/{st.st_size}/{st.st_mtime_ns}".encode())
        return self._encode(h, seed, lambda: load_audio(path, self.prior.sample_rate)[0])

    def _encode(self, h, seed, read):
        cache_id = getattr(self.prior, "cache_id", None)
        h.update(f"{type(self.prior).__name__}/{cache_id}/{getattr(self.prior, 'sample_latent', '')}/{seed}".encode())
        key = h.hexdigest()
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        # only on disk if the prior can tell its autoencoder apart from others
        path = os.path.join(self.cache_dir, key + ".pt") if self.cache_dir and cache_id else None
        if path and os.path.exists(path):
            z = torch.load(path, map_location=self.device, weights_only=True)
        else:
            with torch.no_grad():
                z = self.prior.encode(read()[None].to(self.device))[0]
            if path:
                os.makedirs(self.cache_dir, exist_ok=True)
                torch.save(z.cpu(), path)
        self._cache[key] = z
        while len(self._cache) > 32:
            self._cache.popitem(last=False)
        return z

    @property
    def buffer(self):
        return self.config.get("buffer", 0)

    @property
    def device(self):
        dev = getattr(self.prior, "device", None)
        if dev is None:
            p = next(self.prior.parameters(), None)
            dev = p.device if p is not None else torch.device("cpu")
        return torch.device(dev)

    def inpaint(
        self,
        audio,
        sample_rate,
        gap,
        method=None,
        config=None,
        source=None,
        guide=None,
        candidates=None,
        weights=None,
        n_samples=1,
        steps=None,
        seed=None,
        raw=False,
        blend=0.1,
        callback=None,
        checkpoints=0,
        restart=None,
        trace=None,
        **options,
    ):
        """Fill ``gap`` in ``audio``.

        Parameters
        ----------
        audio : Tensor
            ``(channels, length)`` recording.
        sample_rate : int
        gap : tuple of float
            Start and end of the gap, in seconds.
        method : {"multi", "solo", "dps", "search"}, optional
            SimDPS-Multi, SimDPS-Solo, plain DPS, or the search candidates
            without diffusion, best first, one per sample. Defaults to the
            method of the inpainter's config.
        config : optional
            Settings for this call on top of the inpainter's, as taken by
            :func:`simdps.config.load_config`: the name of a config, a path to
            one, a dict, ``"section.key=value"``, or a list of these.
        source : optional
            What to search instead of ``audio``: a ``(channels, length)``
            tensor at ``sample_rate``, an audio file, a folder of them, or a
            list of any of these.
        guide : float or list, optional
            Times in seconds, in ``source`` or ``audio``, of passages to use
            as the candidates instead of searching. Each is aligned with the
            start of the gap. Items may also be ``(time, source)`` pairs, for
            passages from different recordings, with ``None`` for ``audio``.
        candidates : list of int, optional
            Ranks of the search candidates to keep, e.g. ``[0, 3]``, as
            returned by ``method="search"`` with the same ``seed``.
        weights : list, optional
            Guidance share of each ``guide``, in the same order. ``None``
            entries split the remainder by score, which is the default for all.
        n_samples : int
            Number of independent fills, returned as a batch.
        steps : int, optional
            Sampler steps, overriding the config.
        seed : int, optional
        raw : bool
            Return the generated window as is, instead of splicing it into
            ``audio``.
        blend : float
            Crossfade at the gap edges when splicing, in seconds.
        callback : callable, optional
            Passed to :meth:`EDMSampler.sample`.
        checkpoints : int
            Number of evenly spaced estimates of the result to keep, in
            ``Result.checkpoints``.
        restart : Checkpoint, optional
            Resume from a checkpoint of an earlier call with the same
            arguments, re-noised with ``seed``, to get variations of that run.
            Each sample starts from the first sample in the checkpoint. Pass
            ``Checkpoint(c.step, c.sigma, c.latents[k:k + 1])`` to use another.
        trace : callable, optional
            Called as ``trace(i, sigma, x0, guides)`` after each step, with
            the estimate and the candidates over the regenerated frames, in
            the prior's domain. ``guides`` is ``None`` without candidates.
        **options
            Any single setting of the config by its name, e.g. ``weight``,
            ``stop`` or ``cutoff`` (see :class:`simdps.guidance.Candidates`),
            ``sigma_max``, ``order`` or ``start`` (:class:`simdps.EDMSampler`),
            ``n`` or ``ramp`` (:func:`simdps.search.latent_search`), ``project``
            or ``jacobian`` (:class:`simdps.guidance.Guidance`), ``window``
            (seconds generated around the gap) and ``pad`` (extra frames).
        """
        cfg = self.settings(method, config, steps=steps, **options)
        method = cfg["method"]
        if method == "dps" and (guide is not None or candidates is not None):
            raise ValueError("dps uses no candidates")
        if weights is not None and (guide is None or candidates is not None):
            raise ValueError("weights are given per guide")
        if restart is not None and method == "search":
            raise ValueError("search has nothing to restart")
        # seed the global RNG too, as encoders may sample from it; fork_rng restores it after
        devices = [self.device] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices, enabled=seed is not None):
            if seed is not None:
                torch.manual_seed(seed)
            return self._inpaint(audio, sample_rate, gap, cfg, source=source, guide=guide, picks=candidates,
                                 weights=weights, n_samples=n_samples, seed=seed, raw=raw, blend=blend,
                                 callback=callback, checkpoints=checkpoints, restart=restart, trace=trace)

    def settings(self, method=None, config=None, steps=None, **options):
        """Merge the inpainter's config with ``method``, ``config`` and ``options``, in that order."""
        cfg = self.config
        if method is not None:
            if method not in METHODS:
                raise ValueError(f"unknown method '{method}'")
            m = load_config(METHODS[method])
            # only what defines the method, not the schedule or the model
            cfg = merge(cfg, {"method": method, "candidates": m.get("candidates"),
                              "search": {"n": m.get("search", {}).get("n", 0)},
                              "sampler": {"start": m.get("sampler", {}).get("start", "noise")}})
        if config is not None:
            cfg = merge(cfg, load_config(config))
        if steps is not None:
            options["steps"] = steps
        return with_options(cfg, options)

    def schedule(self, config=None):
        """Return the number of sampler steps under ``config``."""
        if self.prior.sigmas is not None:
            return len(self.prior.sigmas)
        return self.settings(config=config)["sampler"]["steps"]

    def _inpaint(self, audio, sample_rate, gap, cfg, *, source, guide, picks, weights, n_samples, seed, raw, blend,
                 callback, checkpoints, restart, trace):
        prior, hop = self.prior, self.prior.hop_length
        method = cfg["method"]
        settings = dict(cfg.get("search") or {})
        sampling = dict(cfg.get("sampler") or {})
        window = cfg.get("window")
        N = prior.length if window is None else round(window * prior.sample_rate / hop)
        pad = cfg.get("pad", prior.pad_frames)
        start_from = sampling.pop("start", "noise")
        sampling.setdefault("sigma_min", prior.sigma_min)
        sampling.setdefault("sigma_max", prior.sigma_max)
        sampling.setdefault("sigmas", prior.sigmas)
        dev = self.device
        g = torch.Generator(dev).manual_seed(seed) if seed is not None else None

        sr = prior.sample_rate
        x = resample(torch.as_tensor(audio, dtype=torch.float32), sample_rate, sr)
        length = x.shape[-1]
        a, b = round(gap[0] * sr), round(gap[1] * sr)
        if not 0 <= a < b <= length:
            raise ValueError("gap must lie inside the audio")

        W = N * hop
        x = torch.nn.functional.pad(x, (0, max(0, W - length)))
        ws = min(max((a + b) // 2 - W // 2, 0), x.shape[-1] - W) // hop * hop
        fs = max(0, (a - ws) // hop - self.buffer)
        fe = min(N, math.ceil((b - ws) / hop) + self.buffer)
        if fs == 0 or fe == N:
            raise ValueError(f"gap too long for the model's {W / sr:.1f} s window")

        with torch.no_grad():
            z_win = prior.encode(x[None, :, ws : ws + W].to(dev))
        mask = gap_mask(N, fs, fe, device=dev)

        if method == "dps":
            n = 0
        elif method == "search":
            n = settings.get("n") or n_samples
        else:
            n = settings.get("n", 1)
        if picks is not None:
            n = max(n, max(picks) + 1)

        cands = shares = None
        if n > 0:
            # keep candidates apart when they are listed or picked by hand
            spacing = (fe - fs) * hop if method == "search" or picks is not None else 0
            search = self._searcher(settings, n, spacing)
            lead = (a - ws) // hop - fs  # candidates start this many frames before the gap
            cands, shares = self._candidates(search, n, x, z_win, sample_rate, seed, source, guide, weights,
                                             lead, ws, fs, fe, N)
            if picks is not None:
                cands = cands.select(picks)

        kept = []
        if method == "search":
            z = cands.signals[:n_samples]
        else:
            terms = [Observation(mask * z_win, mask, **(cfg.get("observation") or {}))]
            if cands is not None:
                limits = (prior.sigma_min, prior.sigma_max)
                terms.append(Candidates(cands.signals, cands.scores, (fs, fe), sigma_range=limits,
                                        weights=shares, **(cfg.get("candidates") or {})))
            shape = (n_samples, z_win.shape[1], N + pad)
            start = torch.zeros(shape, device=dev)
            if start_from == "candidate":
                if cands is None:
                    raise ValueError("a warm start needs candidates")
                start[..., :N] = cands.signals[0]
            noise = torch.randn(shape, generator=g, device=dev)
            guidance = Guidance(terms, length=N, **(cfg.get("guidance") or {}))
            sampler = EDMSampler(**sampling)
            first, levels = 0, sampler.sigmas(dev)
            if restart is not None:
                first = int(restart.step)
                if not 0 < first < sampler.steps:
                    raise ValueError("restart is not a checkpoint of this schedule")
                start = restart.latents[:1].to(dev).expand(shape)
            keep = {round(j * sampler.steps / (checkpoints + 1)) for j in range(1, checkpoints + 1)}
            guides = None if cands is None else cands.signals[..., fs:fe]

            def step(i, sigma, xi, x0):
                if i + 1 in keep and i + 1 < sampler.steps:
                    kept.append(Checkpoint(i + 1, float(sigma), x0.detach().cpu()))
                if trace is not None:
                    trace(i, sigma, x0[..., fs:fe].detach(), guides)
                if callback is not None:
                    callback(i, sigma, xi, x0)

            z = sampler.sample(prior.denoise, start + levels[first] * noise, guidance, g, step, first)[..., :N]

        out = self._splice(z, x, ws, (a, b), length, raw, blend)
        return Result(out, sr, (ws, ws + W), (a, b), cands, z.detach().cpu(), kept)

    def _candidates(self, search, n, x, z_win, sample_rate, seed, source, guide, weights, lead, ws, fs, fe, N):
        # candidates over the window, and their shares of the guidance if set by hand
        hop, sr = self.prior.hop_length, self.prior.sample_rate
        pairs = _pairs(guide)
        if pairs is not None and (weights is not None or any(s is not None for _, s in pairs)):
            if source is not None:
                raise ValueError("give each guide its own source, or source=, not both")
            return self._guided(pairs, weights, x, z_win, sample_rate, seed, search, lead, ws, fs, fe, N)
        starts = None if pairs is None else [round(t * sr / hop) - lead for t, _ in pairs]
        if source is None:
            lw = ws // hop
            return search(self.encode(x, seed), lw + fs, lw + fe, starts=starts).crop(lw, N), None
        names, zs = self._sources(source, sample_rate, seed)
        if starts is not None and len(zs) > 1:
            raise ValueError("guide needs a single source")
        results, found = [], []
        for name, z in zip(names, zs):
            try:
                results.append(search(z_win[0], fs, fe, source=z, starts=starts))
                found.append(name)
            except ValueError:
                if len(zs) == 1:
                    raise
        if not results:
            raise ValueError("every source is too short for this gap")
        return SearchResult.merge(results, found, None if starts is not None else n), None

    def _searcher(self, settings, n, spacing):
        # search function for the prior's domain, latent or waveform
        hop, sr = self.prior.hop_length, self.prior.sample_rate
        if hop == 1:
            kw = {k: settings[k] for k in ("refine", "crossfade") if k in settings}
            if "rate" in settings:
                kw["search_rate"] = settings["rate"]
            features = []
            if settings.get("stft"):
                features.append(STFTFeature(**settings["stft"]))
            if settings.get("chroma"):
                features.append(ChromaFeature(**settings["chroma"]))
            return lambda t, a, b, **more: waveform_search(t, a, b, sr, n=n, spacing=spacing,
                                                           features=features or None, **kw, **more)
        kw = {k: settings[k] for k in ("context", "crossfade", "metric", "interp", "ramp") if k in settings}
        return lambda t, a, b, **more: latent_search(t, a, b, n=n, spacing=spacing // hop, **kw, **more)

    def _splice(self, z, x, ws, gap, length, raw, blend):
        # decode z and splice it into x, unless raw
        prior, sr = self.prior, self.prior.sample_rate
        W = z.shape[-1] * prior.hop_length
        with torch.no_grad():
            y = torch.cat([prior.decode(zi[None].to(self.device)) for zi in z]).cpu()[..., :W]
        if y.shape[1] != x.shape[0]:
            y = y.mean(1, keepdim=True).expand(-1, x.shape[0], -1)
        if raw:
            return y
        a, b = gap
        out = x.expand(len(y), -1, -1).clone()
        out[..., ws : ws + W] = crossfade(out[..., ws : ws + W], y, a - ws, b - ws, round(blend * sr))
        return out[..., :length]

    def decode(self, latents, audio, sample_rate, result, raw=False, blend=0.1):
        """Decode ``latents``, e.g. from a :class:`Checkpoint` of ``result``, and
        splice them into ``audio`` as :meth:`inpaint` did."""
        sr = self.prior.sample_rate
        x = resample(torch.as_tensor(audio, dtype=torch.float32), sample_rate, sr)
        length, (ws, we) = x.shape[-1], result.window
        x = torch.nn.functional.pad(x, (0, max(0, we - length)))
        N = (we - ws) // self.prior.hop_length
        return self._splice(latents[..., :N], x, ws, result.gap, length, raw, blend)

    def search(self, audio, sample_rate, gap, source=None, n=8, match="latent", seed=None):
        """Search for candidates for ``gap`` without sampling, best first.

        Parameters
        ----------
        audio : Tensor
            ``(channels, length)`` recording.
        sample_rate : int
        gap : tuple of float
            Start and end of the gap, in seconds.
        source : optional
            What to search instead of ``audio``, as in :meth:`inpaint`.
        n : int
        match : {"latent", "audio", "bands"}
            Compare contexts in the prior's domain, as :meth:`inpaint` does;
            through log spectra and chroma
            (:func:`~simdps.search.waveform_search`); or through band energies
            only (:class:`~simdps.search.BandFeature`).

        Returns
        -------
        list of (float, float, object)
            Time in seconds of each candidate, aligned with the gap's start as
            ``guide`` takes it, its score, and where it is: ``None`` for
            ``audio``, else the file or tensor it came from.
        """
        if match not in ("latent", "audio", "bands"):
            raise ValueError(f"unknown match '{match}'")
        prior, hop, sr = self.prior, self.prior.hop_length, self.prior.sample_rate
        x = resample(torch.as_tensor(audio, dtype=torch.float32), sample_rate, sr)
        a, b = round(gap[0] * sr), round(gap[1] * sr)
        if not 0 <= a < b <= x.shape[-1]:
            raise ValueError("gap must lie inside the audio")
        if match == "latent" and hop > 1:
            # the frames inpaint() regenerates: the gap and its buffer
            x = torch.nn.functional.pad(x, (0, max(0, prior.length * hop - x.shape[-1])))
            fs, fe = a // hop - self.buffer, -(-b // hop) + self.buffer
            if fs <= 0:
                raise ValueError("gap too close to the start")
            search = self._searcher(self.config.get("search") or {}, n, (fe - fs) * hop)
            items = [(None, None)]
            if source is not None:
                items = list(zip(_flat(source), self._sources(source, sample_rate, seed)[1]))
            z = self.encode(x, seed)
            found = []
            for where, enc in items:
                r = search(z, fs, fe) if where is None else search(z, fs, fe, source=enc)
                found += [((f + a // hop - fs) * hop / sr, c, where) for f, c in _listed(r)]
        else:
            features = [BandFeature()] if match == "bands" else None
            items = [(None, None)]
            if source is not None:
                items = list(zip(_flat(source), self._waveforms(source, sample_rate)))
            found = []
            for where, w in items:
                kw = {} if where is None else {"source": w}
                r = waveform_search(x, a, b, sr, n=n, features=features, spacing=b - a, **kw)
                found += [(f / sr, c, where) for f, c in _listed(r)]
        return sorted(found, key=lambda f: -f[1])[:n]

    def _waveforms(self, source, sample_rate):
        # everything in ``source`` as waveforms at the prior's rate, ordered as in _flat
        out = []
        for s in _flat(source):
            if isinstance(s, (str, os.PathLike)):
                out.append(load_audio(s, self.prior.sample_rate)[0])
            else:
                out.append(resample(torch.as_tensor(s, dtype=torch.float32), sample_rate, self.prior.sample_rate))
        return out

    def _guided(self, pairs, weights, x, z_win, sample_rate, seed, search, lead, ws, fs, fe, N):
        """One candidate per guide, each from its own source, heaviest first."""
        hop, sr = self.prior.hop_length, self.prior.sample_rate
        lw, parts = ws // hop, []
        for t, src in pairs:
            at = [round(t * sr / hop) - lead]
            if src is None:
                r = search(self.encode(x, seed), lw + fs, lw + fe, starts=at).crop(lw, N)
                parts.append((r, None))
            else:
                names, zs = self._sources(src, sample_rate, seed)
                if len(zs) > 1:
                    raise ValueError("each guide needs a single source")
                parts.append((search(z_win[0], fs, fe, source=zs[0], starts=at), names[0]))
        cands = SearchResult(torch.cat([r.signals for r, _ in parts]), torch.cat([r.scores for r, _ in parts]),
                             torch.cat([r.starts for r, _ in parts]), [name for _, name in parts])
        shares = _shares(cands.scores, list(weights) if weights is not None else [None] * len(pairs))
        if shares is None:
            return cands, None
        order = torch.argsort(shares, descending=True, stable=True).tolist()
        return cands.select(order), shares[order]

    def _sources(self, source, sample_rate, seed):
        # names and encodings of everything in ``source``
        if isinstance(source, (list, tuple)):
            names, zs = [], []
            for s in source:
                n, z = self._sources(s, sample_rate, seed)
                names, zs = names + n, zs + z
            return names, zs
        if isinstance(source, (str, os.PathLike)):
            if not os.path.isdir(source):
                return [str(source)], [self.encode_file(source, seed)]
            files = _audio_files(source)
            if not files:
                raise ValueError(f"no audio files in {source}")
            return files, [self.encode_file(f, seed) for f in files]
        s = resample(torch.as_tensor(source, dtype=torch.float32), sample_rate, self.prior.sample_rate)
        return ["source"], [self.encode(s, seed)]


def _pairs(guide):
    # guide as a list of (time, source or None)
    if guide is None:
        return None
    items = guide if isinstance(guide, (list, tuple)) and not _is_pair(guide) else [guide]
    return [(float(g[0]), g[1]) if _is_pair(g) else (float(g), None) for g in items]


def _is_pair(g):
    return (isinstance(g, (list, tuple)) and len(g) == 2 and isinstance(g[0], (int, float))
            and not isinstance(g[1], (int, float)))


def _flat(source):
    # the files, folders' files and tensors in ``source``, in search order
    if isinstance(source, (list, tuple)):
        return [s for x in source for s in _flat(x)]
    if isinstance(source, (str, os.PathLike)) and os.path.isdir(source):
        return _audio_files(source)
    return [source]


def _audio_files(folder):
    files = (os.path.join(d, f) for d, _, names in os.walk(folder) for f in names)
    return sorted(f for f in files if f.lower().endswith(AUDIO_EXTENSIONS))


def _listed(result):
    return zip(result.starts.tolist(), result.scores.tolist())


def _shares(scores, given):
    """Guidance share of each candidate. Given shares are kept and the remainder
    is split by the softmax of the other scores. ``None`` if no share is given."""
    if all(w is None for w in given):
        return None
    fixed = torch.tensor([w is not None for w in given])
    w = torch.tensor([0.0 if g is None else float(g) for g in given])
    used = w[fixed].sum()
    if used > 1 or fixed.all():
        w[fixed] = w[fixed] / used.clamp(min=1e-8)
        used = torch.tensor(1.0)
    free = ~fixed
    if free.any():
        w[free] = (1 - used) * torch.softmax(scores[free].float().cpu(), 0)
    return w.to(scores.device)
