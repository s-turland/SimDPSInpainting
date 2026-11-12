"""Settings of an experiment described in YAML files.

``configs/experiment`` holds one complete file per experiment of the paper, and
``configs/speed`` the warm starts, which list only what they change. Any of them
can be copied and edited, or adjusted on the fly:

    load_config("ismir26_solo", "candidates.distance=lowpass")
    load_config("ismir26_multi", "warm_start_fast")
    load_config("my_experiment.yaml")
"""

import copy
import os
import re

import yaml

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
GROUPS = ("experiment", "speed")

# section of each keyword argument of Inpainter.inpaint
SECTIONS = {
    "observation": ("zeta", "loss", "compression"),
    "search": ("n", "context", "crossfade", "metric", "interp", "ramp", "rate", "refine", "stft", "chroma"),
    "candidates": ("weight", "combine", "distance", "cutoff", "taper", "stop", "fade"),
    "guidance": ("project", "jacobian", "joint"),
    "sampler": ("steps", "sigma_min", "sigma_max", "rho", "churn", "churn_range", "s_noise", "order", "sigmas",
                "start"),
}
TOP = ("method", "prior", "buffer", "window", "pad")

_FLOAT = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)[eE][+-]?\d+$")


def available():
    """List the packaged configs by group."""
    out = {}
    for group in GROUPS:
        files = sorted(os.listdir(os.path.join(ROOT, group)))
        out[group] = [f[:-5] for f in files if f.endswith(".yaml")]
    return out


def load_config(*configs):
    """Load configs and merge them left to right.

    Parameters
    ----------
    *configs : str, dict or list
        The name of a config of the package, a path to a YAML file, a dict,
        ``"section.key=value"``, or a list of these.

    Returns
    -------
    dict
    """
    out = {}
    for c in configs:
        if c is None:
            continue
        if isinstance(c, (list, tuple)):
            out = merge(out, load_config(*c))
        elif isinstance(c, dict):
            out = merge(out, _numbers(c))
        elif isinstance(c, str) and "=" in c:
            out = _assign(out, c)
        else:
            out = merge(out, _read(c))
    return out


def merge(base, new):
    """Recursively merge ``new`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def with_options(config, options):
    """Return a copy of ``config`` with flat keyword options moved to their sections."""
    out = copy.deepcopy(config)
    for k, v in options.items():
        if k in TOP:
            out[k] = v
            continue
        for section, keys in SECTIONS.items():
            if k in keys:
                out[section] = out.get(section) or {}
                out[section][k] = v
                break
        else:
            raise ValueError(f"unknown option '{k}'")
    return out


def _read(name):
    path = os.fspath(name)
    if not os.path.isfile(path):
        for group in GROUPS:
            candidate = os.path.join(ROOT, group, path + ".yaml")
            if os.path.isfile(candidate):
                path = candidate
                break
        else:
            names = ", ".join(n for group in available().values() for n in group)
            raise ValueError(f"no config '{name}'; the package has {names}")
    with open(path, encoding="utf-8") as f:
        return _numbers(yaml.safe_load(f) or {})


def _assign(config, item):
    key, value = item.split("=", 1)
    parts = key.strip().split(".")
    out = copy.deepcopy(config)
    d = out
    for p in parts[:-1]:
        d[p] = d.get(p) or {}
        d = d[p]
    d[parts[-1]] = _numbers(yaml.safe_load(value))
    return out


def _numbers(x):
    # YAML 1.1 reads 1e-3 as a string
    if isinstance(x, dict):
        return {k: _numbers(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_numbers(v) for v in x]
    if isinstance(x, str) and _FLOAT.match(x):
        return float(x)
    return x
