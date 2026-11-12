import importlib
from functools import partial

from .acestep import ACEStep
from .base import DenoiserPrior, EDMPrior, FlowPrior, Prior
from .cqtdiff import CQTDiff

__all__ = ["Prior", "EDMPrior", "FlowPrior", "DenoiserPrior", "ACEStep", "CQTDiff", "PRIORS", "build_prior"]

PRIORS = {
    "acestep": ACEStep,
    "acestep-turbo": partial(ACEStep, "turbo"),
    "cqtdiff": CQTDiff,
}


def build_prior(settings, device=None, dtype=None):
    """Build a prior from the ``prior`` section of a config.

    ``name`` is one of :data:`PRIORS`, and ``target`` (``"package.module:function"``)
    any function returning a :class:`Prior`. The other keys are its arguments.
    """
    settings = dict(settings)
    target, name = settings.pop("target", None), settings.pop("name", None)
    if target is not None:
        module, _, attr = target.partition(":")
        make = getattr(importlib.import_module(module), attr)
    elif name in PRIORS:
        make = PRIORS[name]
    else:
        raise ValueError(f"unknown model '{name}'")
    return make(device=device, dtype=dtype, **settings)
