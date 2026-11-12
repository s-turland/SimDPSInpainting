API reference
=============

.. autofunction:: simdps.load

.. autoclass:: simdps.Inpainter
   :members: inpaint

.. autoclass:: simdps.Result

Configs
-------

.. automodule:: simdps.config
   :members: load_config, merge, available

Audio
-----

.. automodule:: simdps.audio
   :members: load_audio, save_audio, resample, crossfade

Search
------

.. automodule:: simdps.search
   :members: latent_search, waveform_search, SearchResult, STFTFeature, ChromaFeature, chroma_filterbank, interpolate

Guidance
--------

.. automodule:: simdps.guidance
   :members: Guidance, Observation, Candidates, gap_mask, tukey

Sampler
-------

.. autoclass:: simdps.EDMSampler
   :members: sigmas, sample

Priors
------

.. automodule:: simdps.priors.base
   :members: Prior, EDMPrior, FlowPrior, DenoiserPrior

.. autofunction:: simdps.priors.build_prior

.. autoclass:: simdps.priors.acestep.ACEStep

.. autoclass:: simdps.priors.cqtdiff.CQTDiff
