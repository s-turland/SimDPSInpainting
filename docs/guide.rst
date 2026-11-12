How it works
============

Diffusion posterior sampling (DPS) solves inpainting by adding the gradient of
a reconstruction loss on the observed audio to each step of a pretrained
diffusion sampler. Over long gaps, more than a few hundred milliseconds, the
missing part is only weakly constrained by this loss and the result tends to
drift from the music around it.

SimDPS adds an auxiliary guide :math:`\tilde{X}` for the gap. The rest of the
track (or another recording) is searched for segments whose context matches
the gap's context, and the best ones are added to the likelihood:

.. math::

   \nabla \log p(\tilde{Y} \mid X_\tau) \approx
   -\nabla \left( \tfrac{1}{\sigma_Y^2} \| M (Y - \hat{X}_0) \|^2
   + \tfrac{1}{\sigma_{\tilde{X}}^2} \mathcal{L}_{\tilde{X}}(\hat{X}_0) \right)

where :math:`M` masks the gap and :math:`\hat{X}_0` is the denoiser's estimate
of the clean signal. Each term's gradient is rescaled to a fixed size, which is
set by its weight.

Methods
-------

``"multi"`` (SimDPS-Multi, default)
   Uses the 15 best candidates. Each is weighted by the softmax of its
   similarity score and compared with the estimate through a low-pass
   distance, so only the coarse structure of the candidates is followed. The
   guidance stops at noise level 0.5, which leaves the details to the model.

``"solo"`` (SimDPS-Solo)
   Uses the best candidate only, with a higher weight and through the plain
   squared error, up to the end of sampling. The result stays close to the
   match.

``"dps"``
   No candidates: plain DPS.

``"search"``
   Returns the best candidate, crossfaded into the gap, without diffusion.
   This is fast and is a useful reference.

Configs
-------

All the settings of an experiment are in one YAML file, in
``simdps/configs/experiment``. The file for SimDPS-Multi is
``ismir26_multi.yaml``, and its sections follow the method:

``observation``
   The likelihood of the observed context. ``zeta`` is its weight.
``search``
   The similarity search. ``n`` is the number of candidates kept.
``candidates``
   The guidance from the candidates: its ``weight``, how to ``combine`` the
   candidates, and how to measure the ``distance`` to them.
``guidance``
   How the terms are combined with the model's direction.
``sampler``
   The noise schedule and the sampler's steps.

A config is chosen by name or path, and changes can be stacked on top of it,
later ones winning:

.. code-block:: python

   model = simdps.load("acestep", config="ismir26_solo")
   model.inpaint(audio, sr, gap)

   model.inpaint(audio, sr, gap, config="my_experiment.yaml")
   model.inpaint(audio, sr, gap, config=["ismir26_solo", "candidates.distance=lowpass"])
   model.inpaint(audio, sr, gap, weight=0.2, stop=0.1)

and the same from the command line::

   simdps inpaint take.wav --gap 62.5 64.5 --config ismir26_solo --set candidates.distance=lowpass

SimDPS-Solo can switch to the low-pass distance of SimDPS-Multi like this, so
that it follows only the coarse structure of its match. ``solo_lowpass.yaml``
has the other settings tuned for that.

The settings most worth trying are:

``candidates.weight``
   Strength of the candidate guidance (0.1 for multi, 0.3 for solo).
``candidates.stop``
   Noise level below which the candidates are ignored. Lower values follow
   them into finer detail.
``candidates.cutoff``
   Number of frequency bins kept by the low-pass filter. Higher values follow
   faster changes in the candidates.

Speed
-----

The settings of the paper take about a minute per gap on a laptop with a commercial GPU (tested on an RTX 3070 Ti (8 GB)). 
Two configs in ``simdps/configs/speed`` act on top of any experiment and are faster via the use of a warm start, 
but are not part of the paper and included for demonstration and efficiency purposes. They start from the best candidate at a low noise
level instead of from pure noise, and take fewer steps. ``warm_start_fast`` also
takes the guidance gradients with respect to the denoised estimate only, demonstrating warm start "cut-and-paste" similarity 
diffusion without any backwards passes.

=========================  =====  =======  ========  ======
config                     steps  start σ  Jacobian  time
=========================  =====  =======  ========  ======
``warm_start_fast``        35     1        no        7 s
``warm_start_detailed``    40     1        yes       17 s
(the paper's settings)     100    20       yes       55 s
=========================  =====  =======  ========  ======

Times are for one 5 s gap with ``"multi"`` on a laptop RTX 3070 Ti (8 GB), once
the track has been encoded. ``"dps"`` has no candidate to start from, so it
refuses them.

Windows
-------

The model works on windows of 15 s (ACE-Step) centred on the gap. ``buffer``
frames (0.8 s) on each side of the gap are regenerated as well, and the gap and
its buffers must leave some context inside the window. The paper evaluates
gaps of 2 to 5 s, but any length can be used, as long as the window is long 
enough to contain it and its context.

Choosing the material
---------------------

By default the candidates come from the rest of the recording. ``source``
can be also set to search a tensor, an audio file, a folder of files (searched
recursively), or a list of these. The replacement is then selected from there
while matching the context of the input, for example a phrase from another
take, or a sample library for sound design. If searching via latents, instead of audio, encoding a 
full track may be slow on a laptop. Each file is encoded once and cached, 
so later searches of the same folder can be refreshed quickly.  
The cache is kept in ``~/.cache/simdps``, or in ``$SIMDPS_CACHE`` if set;
``Inpainter(prior, cache_dir=None)`` keeps it in memory only.

``method="search"`` returns the candidates themselves, best first, one per
sample to listen to before choosing:

.. code-block:: python

   matches = model.inpaint(audio, sr, gap, method="search", n_samples=8, seed=0)
   matches.candidates.starts   # latent frames in the source
   matches.candidates.sources  # which file, when several were searched

   out = model.inpaint(audio, sr, gap, candidates=[0, 3], seed=0)

``candidates`` keeps those ranks only. With the same ``seed`` the ranking is
equivalent. ``guide`` skips the search and uses the passages at the given times
instead, each aligned with the start of the gap:

.. code-block:: python

   out = model.inpaint(audio, sr, gap, guide=95.0)                # from the input
   out = model.inpaint(audio, sr, gap, source="riff.wav", guide=[2.0, 6.5])

For an exact copy, use ``method="search"``.

The same is available from the command line (``--source``, ``--candidates``,
counted from 1 as the files written by ``--method search`` are, and ``--guide``)
and in the app. There, a single guide or kept candidate is followed with
SimDPS-Solo unless a method is given, e.g. ``--method multi``.
