"""Similarity search for candidate gap fillers."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class SearchResult:
    """Result of a similarity search.

    Attributes
    ----------
    signals : Tensor
        ``(K, C, L)`` copies of the target with the gap filled by each candidate.
    scores : Tensor
        ``(K,)`` similarity scores ``1 / (1 + J)`` (Sec. 3.2), in decreasing order.
    starts : Tensor
        ``(K,)`` start frame of each candidate in the source.
    sources : list, optional
        Source of each candidate, when several were searched.
    """

    signals: torch.Tensor
    scores: torch.Tensor
    starts: torch.Tensor
    sources: list = None

    def __len__(self):
        return len(self.scores)

    def crop(self, start, length):
        """Result restricted to frames ``[start, start + length)``."""
        return SearchResult(self.signals[..., start : start + length], self.scores, self.starts, self.sources)

    def select(self, index):
        """Candidates ``index``, in that order."""
        index = list(index)
        sources = None if self.sources is None else [self.sources[i] for i in index]
        return SearchResult(self.signals[index], self.scores[index], self.starts[index], sources)

    @staticmethod
    def merge(results, sources, n=None):
        """Best ``n`` candidates of several results, labelled by ``sources``."""
        names = [s for r, s in zip(results, sources) for _ in range(len(r))]
        out = SearchResult(
            torch.cat([r.signals for r in results]),
            torch.cat([r.scores for r in results]),
            torch.cat([r.starts.to(results[0].starts.device) for r in results]),
            names,
        )
        order = torch.argsort(out.scores, descending=True, stable=True)[:n]
        return out.select(order.tolist())


@torch.no_grad()
def latent_search(
    target,
    start,
    end,
    source=None,
    n=15,
    context=5,
    crossfade=5,
    metric="l2",
    interp="slerp",
    ramp=False,
    starts=None,
    spacing=0,
):
    """Exhaustive search for latent segments whose context matches the gap's (Eqs. 11-12).

    Parameters
    ----------
    target : Tensor
        ``(C, L)`` latent sequence containing the gap.
    start, end : int
        Gap frames ``[start, end)``.
    source : Tensor, optional
        ``(C, L')`` sequence to search. Defaults to ``target``, in which case
        candidates overlapping the gap are discarded.
    n : int
        Number of candidates to return.
    context : int
        Frames compared on each side of the gap.
    crossfade : int
        Frames over which each candidate is blended into the target context.
    metric : {"l2", "l1", "cosine", "pearson"}
        Context distance.
    interp : {"slerp", "lerp"}
        Interpolation used for the crossfades.
    ramp : bool
        Weight the context linearly from 0 at its far end to 1 at the gap
        (Eq. 13). Otherwise all context frames count equally.
    starts : sequence of int, optional
        Source frames to score instead of searching, e.g. a passage chosen
        by hand. They are used even if they overlap the gap.
    spacing : int
        Minimum distance in frames between candidates. Neighbouring frames
        of a good match are usually good matches too; a spacing of the gap
        length returns distinct passages.

    Returns
    -------
    SearchResult
    """
    if metric not in ("l2", "l1", "cosine", "pearson"):
        raise ValueError(f"unknown metric '{metric}'")
    if interp not in ("slerp", "lerp"):
        raise ValueError(f"unknown interpolation '{interp}'")
    external = source is not None
    source = target if source is None else source
    C, L = target.shape
    Ls = source.shape[-1]
    gap = end - start

    lc = min(context, start)
    rc = min(context, L - end)
    xl = min(crossfade, start)
    xr = min(crossfade, L - end)

    lo = max(lc, xl)
    hi = Ls - gap - max(rc, xr)
    if hi < lo:
        raise ValueError("source is too short for this gap")
    given = starts is not None
    if given:
        starts = torch.as_tensor(starts, device=target.device).long().reshape(-1)
        if ((starts < lo) | (starts > hi)).any():
            raise ValueError(f"candidate starts must lie in [{lo}, {hi}]")
    else:
        starts = torch.arange(lo, hi + 1, device=target.device)

    dist = torch.zeros(len(starts), device=target.device)
    if lc > 0:
        cands = source.unfold(1, lc, 1)[:, starts - lc]
        w = torch.linspace(1 / lc, 1, lc, device=target.device) if ramp else None
        dist += lc * _distance(cands, target[:, start - lc : start], metric, w)
    if rc > 0:
        cands = source.unfold(1, rc, 1)[:, starts + gap]
        w = torch.linspace(1, 1 / rc, rc, device=target.device) if ramp else None
        dist += rc * _distance(cands, target[:, end : end + rc], metric, w)
    dist /= max(1, lc + rc)

    if not external and not given:
        overlap = (starts - xl < end) & (starts + gap + xr > start)
        dist[overlap] = float("inf")
    k = len(starts) if given else min(n, int(torch.isfinite(dist).sum()))
    if k == 0:
        raise ValueError("no candidates outside the gap; the sequence is too short")
    if spacing > 0 and not given:
        idx = _spaced(starts, -dist, k, spacing)
        best = dist[idx]
    else:
        best, idx = torch.topk(dist, k, largest=False)
    starts = starts[idx]

    out = target.expand(k, C, L).clone()
    fade_l = torch.linspace(0, 1, xl, device=target.device)
    fade_r = torch.linspace(0, 1, xr, device=target.device)
    for i, s in enumerate(starts.tolist()):
        seg = source[:, s - xl : s + gap + xr].clone()
        if xl > 0:
            seg[:, :xl] = interpolate(target[:, start - xl : start], seg[:, :xl], fade_l, interp)
        if xr > 0:
            seg[:, -xr:] = interpolate(seg[:, -xr:], target[:, end : end + xr], fade_r, interp)
        out[i, :, start - xl : end + xr] = seg
    return SearchResult(out, 1 / (1 + best), starts)


def interpolate(a, b, t, mode="slerp"):
    """Frame-wise interpolation from ``a`` (t = 0) to ``b`` (t = 1).

    ``slerp`` takes the angle between the channel vectors of each frame, falling
    back to ``lerp`` where they are nearly parallel.
    """
    lerp = a * (1 - t) + b * t
    if mode == "lerp":
        return lerp
    an = a / (a.norm(dim=0, keepdim=True) + 1e-8)
    bn = b / (b.norm(dim=0, keepdim=True) + 1e-8)
    omega = torch.acos((an * bn).sum(0, keepdim=True).clamp(-0.9999, 0.9999))
    so = torch.sin(omega)
    slerp = (torch.sin(omega - omega * t) * a + torch.sin(omega * t) * b) / (so + 1e-8)
    return torch.where(so < 1e-5, lerp, slerp)


def _distance(cands, ctx, metric, weight=None):
    # cands (C, N, W), ctx (C, W) -> (N,)
    C, N, W = cands.shape
    c = cands.permute(1, 0, 2)
    x = ctx.unsqueeze(0)
    if metric in ("l2", "l1"):
        err = (c - x) ** 2 if metric == "l2" else (c - x).abs()
        if weight is None:
            return err.reshape(N, -1).mean(1)
        return (err * weight).sum((1, 2)) / (C * weight.sum())
    if weight is not None:
        c, x = c * weight.sqrt(), x * weight.sqrt()
    c, x = c.reshape(N, -1), x.reshape(1, -1)
    if metric == "pearson":
        c = c - c.mean(1, keepdim=True)
        x = x - x.mean(1, keepdim=True)
    return 1 - F.cosine_similarity(c, x, dim=1)


class STFTFeature:
    """Log-magnitude STFT, for short-range continuity.

    Parameters
    ----------
    context : float
        Seconds compared on each side of the gap.
    weight : float
        Weight of this feature in the search cost.
    scale : float
        Per-frame normaliser of the squared distance.
    """

    def __init__(self, n_fft=1024, hop_length=256, context=2.0, weight=1.0, scale=12500.0):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.context = context
        self.weight = weight
        self.scale = scale

    def __call__(self, x, sample_rate):
        return torch.log1p(_stft(x, self.n_fft, self.hop_length).abs())


class ChromaFeature(STFTFeature):
    """Chromagram in dB, for harmonic context."""

    def __init__(self, n_fft=1024, hop_length=256, context=5.0, weight=1.5, scale=76800.0, top_db=80.0):
        super().__init__(n_fft, hop_length, context, weight, scale)
        self.top_db = top_db

    def __call__(self, x, sample_rate):
        fb = chroma_filterbank(sample_rate, self.n_fft // 2 + 1).to(x.device)
        c = (_stft(x, self.n_fft, self.hop_length).abs().T @ fb).T
        db = 20 * torch.log10(c.clamp(min=1e-10))
        return torch.maximum(db, db.max() - self.top_db)


class BandFeature(STFTFeature):
    """Log energy in logarithmically spaced bands, for overall timbre and level.

    Coarser than :class:`STFTFeature`, so it tolerates small differences in
    pitch and arrangement between the gap's context and a candidate's.
    """

    def __init__(self, n_fft=1024, hop_length=256, context=2.0, weight=1.0, scale=12.0, n_bands=24, fmin=40.0):
        super().__init__(n_fft, hop_length, context, weight, scale)
        self.n_bands = n_bands
        self.fmin = fmin

    def __call__(self, x, sample_rate):
        fb = band_filterbank(sample_rate, self.n_fft // 2 + 1, self.n_bands, self.fmin).to(x.device)
        power = _stft(x, self.n_fft, self.hop_length).abs() ** 2
        return torch.log10(fb.T @ power + 1e-8)


@torch.no_grad()
def waveform_search(
    target,
    start,
    end,
    sample_rate,
    source=None,
    n=5,
    features=None,
    search_rate=16000,
    refine=256,
    crossfade=0.1,
    starts=None,
    spacing=0,
):
    """Search for audio segments whose context matches the gap's.

    Contexts are compared through ``features`` at ``search_rate`` over every
    feature frame, then the best candidates are moved by up to ``refine / 2``
    samples to match the samples at the gap boundaries.

    Parameters
    ----------
    target : Tensor
        ``(channels, samples)`` audio containing the gap.
    start, end : int
        Gap samples ``[start, end)``.
    sample_rate : int
    source : Tensor, optional
        ``(channels, samples)`` audio to search. Defaults to ``target``, in
        which case candidates overlapping the gap are discarded.
    n : int
        Number of candidates.
    features : list, optional
        Defaults to ``[STFTFeature(), ChromaFeature()]``.
    search_rate : int
        Sample rate of the coarse search.
    refine : int
        Width in samples of the boundary refinement.
    crossfade : float
        Equal-power crossfade into the target context, in seconds.
    starts : sequence of int, optional
        Source samples to score instead of searching. They are still refined.
    spacing : int
        Minimum distance in samples between candidates.

    Returns
    -------
    SearchResult
        ``starts`` are in samples of ``source``.
    """
    from .audio import resample

    features = features or [STFTFeature(), ChromaFeature()]
    external = source is not None
    source = target if source is None else source
    tgt, src = target.mean(0), source.mean(0)
    r = search_rate / sample_rate
    t_lo = resample(tgt, sample_rate, search_rate)
    s_lo = t_lo if not external else resample(src, sample_rate, search_rate)

    hop = min(f.hop_length for f in features)
    ctx_max = max(f.context for f in features)
    ctx_samples = int(ctx_max * search_rate)
    a, b = int(start * r), int(end * r)
    pre = t_lo[max(0, a - ctx_samples) : a]
    post = t_lo[b : b + ctx_samples]

    feats = [(f, f(pre, search_rate), f(post, search_rate), f(s_lo, search_rate)) for f in features]
    total = feats[0][3].shape[-1]
    gap = (b - a) // hop
    lo = int(ctx_max * search_rate) // hop
    given = None if starts is None else torch.as_tensor(starts, device=target.device).long().reshape(-1)
    if given is not None:
        cands = (given * r).long() // hop
        if ((cands < lo) | (cands >= total - gap - lo)).any():
            raise ValueError("candidate starts too close to the ends of the source")
    else:
        if total - gap - lo <= lo:
            raise ValueError("source is too short for this gap and context")
        cands = torch.arange(lo, total - gap - lo, device=target.device)
        if not external:
            cands = cands[~((cands < b // hop) & (cands + gap > a // hop))]
    if len(cands) == 0:
        raise ValueError("source is too short for this gap")

    cost = torch.zeros(len(cands), device=target.device)
    for f, f_pre, f_post, f_src in feats:
        ctx = int(f.context * search_rate) // hop
        if ctx == 0:
            continue
        windows = f_src.unfold(-1, ctx, 1)
        w = torch.linspace(0.5, 1.0, ctx, device=target.device)
        d_pre = _feature_distance(f_pre[..., -ctx:], windows[:, cands - ctx], w, f.scale)
        d_post = _feature_distance(f_post[..., :ctx], windows[:, cands + gap], w.flip(0), f.scale)
        cost += (d_pre + d_post) / 2 * f.weight

    score = 1 / (1 + cost.sqrt())
    if spacing > 0 and given is None:
        idx = _spaced(cands, score, min(n, len(score)), max(1, int(spacing * r) // hop))
        score = score[idx]
    else:
        score, idx = torch.topk(score, len(score) if given is not None else min(n, len(score)))
    starts = given[idx] if given is not None else (cands[idx] * hop * (sample_rate / search_rate)).long()

    # sample-level alignment with the samples either side of the gap
    length = end - start
    before = tgt[start - 1] if start > 0 else tgt.new_zeros(())
    after = tgt[end] if end < len(tgt) else tgt.new_zeros(())
    offsets = torch.arange(-refine // 2, max(refine // 2, 1), device=target.device)
    pos = starts[:, None] + offsets
    ok = (pos > 0) & (pos + length < len(src))
    p = pos.clamp(0, len(src) - length)
    err = (before - src[p]).abs() + (after - src[p + length - 1]).abs()
    err[~ok] = float("inf")
    starts = starts + torch.where(ok.any(1), offsets[err.argmin(1)], 0)

    out = target.expand(len(starts), *target.shape).clone()
    xf = int(crossfade * sample_rate)
    for i, s in enumerate(starts.tolist()):
        seg = source[:, s : s + length]
        out[i, :, start : start + seg.shape[-1]] = seg
        out[i, :, start + seg.shape[-1] : end] = 0
        k = min(xf, start, s)
        if k > 0:
            fo, fi = _equal_power(k, target.device)
            out[i, :, start - k : start] = target[:, start - k : start] * fo + source[:, s - k : s] * fi
        k = min(xf, target.shape[-1] - end, source.shape[-1] - (s + length))
        if k > 0:
            fo, fi = _equal_power(k, target.device)
            out[i, :, end : end + k] = target[:, end : end + k] * fi + source[:, s + length : s + length + k] * fo
    return SearchResult(out, score, starts)


def chroma_filterbank(sample_rate, n_freqs, n_chroma=12, ctroct=5.0, octwidth=2.0):
    """``(n_freqs, n_chroma)`` frequency-to-chroma matrix, starting at C, as in librosa."""
    freqs = torch.linspace(0, sample_rate // 2, n_freqs)[1:]
    bins = n_chroma * torch.log2(freqs / (440.0 / 16))
    bins = torch.cat([bins[:1] - 1.5 * n_chroma, bins])
    widths = torch.cat([torch.clamp(bins[1:] - bins[:-1], min=1.0), torch.ones(1)])
    d = bins[:, None] - torch.arange(n_chroma)
    half = round(n_chroma / 2)
    d = torch.remainder(d + half, n_chroma) - half
    fb = torch.exp(-0.5 * (2 * d / widths[:, None]) ** 2)
    fb = F.normalize(fb, p=2, dim=1)
    fb = fb * torch.exp(-0.5 * ((bins[:, None] / n_chroma - ctroct) / octwidth) ** 2)
    return torch.roll(fb, -3 * (n_chroma // 12), dims=1)


def band_filterbank(sample_rate, n_freqs, n_bands=24, fmin=40.0):
    """``(n_freqs, n_bands)`` matrix summing STFT bins into log-spaced bands up to Nyquist."""
    freqs = torch.linspace(0, sample_rate / 2, n_freqs)
    edges = fmin * (sample_rate / 2 / fmin) ** (torch.arange(n_bands + 1) / n_bands)
    fb = ((freqs[:, None] >= edges[:-1]) & (freqs[:, None] < edges[1:])).float()
    # bands narrower than a bin take the nearest bin
    empty = fb.sum(0) == 0
    if empty.any():
        centre = (edges[:-1] + edges[1:]) / 2
        nearest = (freqs[:, None] - centre[empty]).abs().argmin(0)
        fb[nearest, empty.nonzero()[:, 0]] = 1.0
    return fb


def _spaced(positions, score, k, spacing):
    # best k by score, greedily skipping positions within spacing of one kept
    kept = []
    for i in torch.argsort(score, descending=True, stable=True).tolist():
        if not torch.isfinite(score[i]):
            break
        if all(abs(positions[i] - positions[j]) >= spacing for j in kept):
            kept.append(i)
            if len(kept) == k:
                break
    return torch.tensor(kept, dtype=torch.long, device=score.device)


def _stft(x, n_fft, hop):
    window = torch.hann_window(n_fft, device=x.device)
    return torch.stft(x, n_fft, hop, window=window, center=True, pad_mode="reflect", return_complex=True)


def _feature_distance(ctx, windows, weight, scale):
    # ctx (F, W), windows (F, N, W) -> (N,)
    m = min(ctx.shape[-1], windows.shape[-1])
    err = ((windows[..., :m] - ctx[:, None, :m]) ** 2).sum(0)
    if scale > 0:
        err = err / (m * scale + 1e-9)
    return (err * weight[:m]).sum(-1)


def _equal_power(n, device):
    t = torch.linspace(0, torch.pi / 2, n, device=device)
    return torch.cos(t), torch.sin(t)
