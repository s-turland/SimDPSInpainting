Getting started
===============

A notebook version of this page runs in
`Colab <https://colab.research.google.com/github/s-turland/SimDPSInpainting/blob/main/examples/quickstart.ipynb>`_.

Command line
------------

Mark the gap with its start and end, in seconds or ``m:ss``::

   simdps inpaint take.wav --gap 1:02.5 1:04.5

This writes ``take_simdps.wav``, which is the input with the gap replaced.
Everything outside the gap, apart from a 0.1 s crossfade at each edge, is left
as it was. Other options::

   simdps inpaint take.wav --gap 62.5 64.5 --samples 4 --seed 0   # four alternatives
   simdps inpaint take.wav --gap 62.5 64.5 --method solo          # follow the best match closely
   simdps inpaint take.wav --gap 62.5 64.5 --source other.wav     # take material from another file
   simdps inpaint take.wav --gap 62.5 64.5 --config warm_start_fast   # much faster
   simdps inpaint --help

Web interface (Experimental)
----------------------------

::

   simdps app

opens a page on ``localhost`` where you can experiment with diffusion-based editing locally. Current 
functionality includes the ability to mark the gap on the waveform, loop the gap
with its context, find suitable matches via search, choose which will guide the model and
with what weighting, add passages from other recordings, and compare the
results with the original. A curve under each waveform shows how well a
candidate starting there would match. Intermediate diffusion estimates along the trajectory
can be heard and sampled again from with multiple seeds. Results, candidates and
checkpoints can be kept in a bank, a folder (``bank/`` in the repository,
``~/simdps-bank`` for an installed package, or ``--bank``) that later tracks
can search or take guides from, to build up a library of iterative audio tracks. ``--port``
and ``--no-browser`` change how it starts.

Python
------

.. code-block:: python

   import simdps

   model = simdps.load("acestep")          # downloads the weights once
   audio, sr = simdps.load_audio("take.wav")

   out = model.inpaint(audio, sr, gap=(62.5, 64.5), n_samples=2, seed=0)
   for i, y in enumerate(out.audio):
       simdps.save_audio(f"take_{i}.wav", y, out.sample_rate)

:meth:`~simdps.Inpainter.inpaint` returns a :class:`~simdps.Result`. Besides
the audio, it holds the window that was generated and the search candidates:

.. code-block:: python

   out.window                 # (start, end) samples of the generated window
   out.candidates.scores      # similarity of each candidate
   out.candidates.starts      # where they were found, in latent frames

The output sample rate is the model's (48 kHz for ACE-Step). Mono input gives
mono output, and stereo input gives stereo.
