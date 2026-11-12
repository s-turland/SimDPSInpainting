SimDPS
======

SimDPS fills long gaps in music recordings. It searches the rest of the track
for passages whose surroundings match the gap, and uses them to guide a
pretrained diffusion model through a modified likelihood. No training is
needed, and any diffusion model, in the waveform or latent domain, can be used.

.. code-block:: python

   import simdps

   model = simdps.load("acestep")
   audio, sr = simdps.load_audio("take.wav")
   out = model.inpaint(audio, sr, gap=(62.5, 64.5))
   simdps.save_audio("take_simdps.wav", out.audio[0], out.sample_rate)

Audio examples are on the `project page <https://s-turland.github.io/SimDPSInpainting/>`_.

.. toctree::
   :maxdepth: 2

   install
   quickstart
   guide
   models
   paper
   api

Citation
--------

.. code-block:: bibtex

   @inproceedings{turland2026simdps,
     title     = {Music Inpainting with Similarity-Guided Latent Diffusion},
     author    = {Turland, Sean and Moliner, Eloi and V{\"a}lim{\"a}ki, Vesa},
     booktitle = {Proc. Int. Society for Music Information Retrieval Conf. (ISMIR)},
     year      = {2026}
   }
