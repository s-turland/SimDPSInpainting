Using your own model
====================

SimDPS treats the diffusion model as a :class:`~simdps.priors.Prior`: a
denoiser in the variance-exploding parameterisation of Karras et al. (2022),

.. math::

   x = x_0 + \sigma n, \qquad \texttt{denoise}(x, \sigma) \approx E[x_0 \mid x],

plus ``encode`` and ``decode`` functions for latent models. A prior also
declares its sample rate, its window length in frames, and the number of audio
samples per frame.

Adapters
--------

Most models fit one of three adapters.

:class:`~simdps.priors.EDMPrior`, for networks trained with the EDM
preconditioning:

.. code-block:: python

   from simdps.priors import EDMPrior

   prior = EDMPrior(net, sigma_data=0.063, sample_rate=44100, length=2**18,
                    sigma_min=1e-5, sigma_max=10)

:class:`~simdps.priors.FlowPrior`, for rectified flow (velocity prediction),
here on the latents of an autoencoder:

.. code-block:: python

   from simdps.priors import FlowPrior

   prior = FlowPrior(net, sample_rate=44100, length=512, hop_length=1024,
                     encode=vae.encode, decode=vae.decode)

:class:`~simdps.priors.DenoiserPrior`, for anything that already gives a VE
denoiser:

.. code-block:: python

   from simdps.priors import DenoiserPrior

   prior = DenoiserPrior(lambda x, sigma: model(x, sigma), sample_rate=16000, length=64000)

Then:

.. code-block:: python

   from simdps import Inpainter

   model = Inpainter(prior)
   out = model.inpaint(audio, sr, gap=(4.0, 5.5))

For latent models the search runs on the latents
(:func:`~simdps.search.latent_search`). Encoded tracks are kept in memory, and on
disk if the prior is given a ``cache_id`` naming its autoencoder. Waveform models, with
``hop_length=1``, are searched on STFT and chroma features of the audio
(:func:`~simdps.search.waveform_search`); use ``Inpainter(prior, buffer=0)``
for them, as there is no frame rounding to absorb.

For other parameterisations, subclass :class:`~simdps.priors.Prior` and
implement :meth:`~simdps.priors.Prior.denoise`. :class:`simdps.priors.ACEStep`
is a complete example.

Define the model with a config
------------------------------

A function that builds the prior can be named in a config, which makes the model
an additional experiment that is usable from Python and the command line:

.. code-block:: python

   # my_package/models.py
   def make_prior(checkpoint, device=None, dtype=None):
       net = MyNetwork()
       net.load_state_dict(torch.load(checkpoint))
       return EDMPrior(net.to(device), sigma_data=0.5, sample_rate=44100, length=2**18,
                       hop_length=1, sigma_min=1e-3, sigma_max=10)

.. code-block:: yaml

   # my_model.yaml
   method: solo
   prior:
     target: my_package.models:make_prior
     checkpoint: weights.pt
   buffer: 0
   observation: {zeta: 0.5}
   search: {n: 1}
   candidates: {weight: 0.3, combine: solo, distance: l2, taper: null, stop: 1e-3, fade: 0.1}
   guidance: {project: true, jacobian: true}
   sampler: {steps: 50, rho: 7, churn: 10, start: noise}

::

   simdps inpaint take.wav --gap 2 4 --config my_model.yaml

Keys of ``prior`` other than ``target`` are arguments of the function, along with
``device`` and ``dtype``. Leave ``sigma_min`` and ``sigma_max`` out of ``sampler``
to use those of the prior.

The piano model
---------------

The waveform-domain model of the paper is the ``cqtdiff`` prior, which needs the
``piano`` extra and its checkpoint:

.. code-block:: python

   model = simdps.load("ismir26_piano_strict", checkpoint="piano.pt")

A checkpoint can be a plain state dict, or a training checkpoint with ``ema``,
``ema_model`` or ``network``. If the shape of your network differs from the
paper's, ``prior.network`` changes any of
:data:`simdps.priors.cqtdiff.NETWORK`.

Lower-level use
---------------

The pieces of :class:`~simdps.Inpainter` can be used on their own, for
instance on signals that are not audio:

.. code-block:: python

   from simdps import EDMSampler
   from simdps.guidance import Candidates, Guidance, Observation, gap_mask
   from simdps.search import latent_search

   z = prior.encode(audio[None])                 # (1, C, T)
   start, end = 120, 180
   mask = gap_mask(z.shape[-1], start, end)

   found = latent_search(z[0], start, end, n=15)
   guide = Guidance([
       Observation(mask * z, mask, zeta=0.5),
       Candidates(found.signals, found.scores, (start, end), weight=0.1),
   ])
   sampler = EDMSampler(steps=100)
   x = found.signals[:1] + sampler.sigma_max * torch.randn_like(z)
   z_hat = sampler.sample(prior.denoise, x, guide)

Guidance terms are plain objects with ``weight(sigma)`` and ``loss(x0)``, so
new ones can be added to the list.
