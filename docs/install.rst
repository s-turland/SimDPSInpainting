Installation
============

SimDPS needs Python 3.10 or later and PyTorch 2.6 or later. Install PyTorch
for your platform first (see `pytorch.org <https://pytorch.org>`_), then::

   pip install "simdps[acestep] @ git+https://github.com/s-turland/SimDPSInpainting"

The ``acestep`` extra pulls in ``diffusers``, ``transformers`` and ``accelerate`` for the
ACE-Step model. The web interface needs nothing further.

The ``piano`` extra adds what the waveform-domain model of the paper needs.

Without extras, only the core (sampler, guidance, search) is installed, which
is enough to use your own model.

Weights
-------

The ACE-Step 1.5 transformer and VAE (about 5 GB) are downloaded from the
Hugging Face Hub the first time a model is loaded, and cached in
``~/.cache/huggingface``. Set ``HF_HOME`` to keep them elsewhere.

Development
-----------

::

   git clone https://github.com/s-turland/SimDPSInpainting
   cd SimDPSInpainting
   pip install -e ".[acestep,piano,test]"
   pytest

``pytest -m slow`` also runs the tests that download the model.
