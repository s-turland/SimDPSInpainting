Settings of the paper
=====================

Each method of the paper has a config in ``simdps/configs/experiment``, which
holds all of its settings. Scripts that rerun the paper's evaluation will be
added to this repository.

General music (Sec. 4.1)
------------------------

====================  ==================
In the paper          Config
====================  ==================
SimDPS-Multi          ``ismir26_multi``
SimDPS-Solo           ``ismir26_solo``
DPS                   ``ismir26_dps``
Latent-Sim            ``ismir26_search``
====================  ==================

These use ACE-Step 1.5. ``ismir26_multi`` is the default::

   simdps inpaint take.wav --gap 62.5 64.5 --config ismir26_solo

Piano (Sec. 4.2)
----------------

=================  ==========================
In the paper       Config
=================  ==========================
SimDPS-Strict      ``ismir26_piano_strict``
SimDPS-Relaxed     ``ismir26_piano_relaxed``
DPS                ``ismir26_piano_dps``
Sim                ``ismir26_piano_search``
=================  ==========================

These use the waveform-domain piano model, whose checkpoint is given with
``prior.checkpoint``::

   simdps inpaint take.wav --gap 2 4 --config ismir26_piano_strict --set prior.checkpoint=piano.pt

Editing an experiment
----------------------

Settings can be edited for a run, via the command line or from Python::

   simdps inpaint take.wav --gap 62.5 64.5 --config ismir26_solo --set candidates.distance=lowpass

.. code-block:: python

   model.inpaint(audio, sr, gap, config=["ismir26_multi", "candidates.weight=0.2"])

Experiments of your own can be added as new files, or passed
to ``simdps inpaint --config`` as a path. See :doc:`guide` for what the
settings do.
