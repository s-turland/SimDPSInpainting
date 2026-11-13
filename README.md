# SimDPS

Similarity-guided diffusion posterior sampling, for inpainting and editing
long gaps in music recordings. This is the public code for the paper *Music Inpainting with
Similarity-Guided Latent Diffusion*, presented at ISMIR 2026.

SimDPS fills a missing segment of a recording by searching an audio source for
sections whose surrounding context matches that of the gap. The source can be the same
recording, an unrelated track or a corpus of signals. The chosen sections
guide a pretrained diffusion model without any additional training and can be applied to
any diffusion models in the waveform or latent domain.

[Paper](https://s-turland.github.io/SimDPSInpainting/) |
[Audio examples](https://s-turland.github.io/SimDPSInpainting/) |
[Documentation](docs/) |
[Colab](https://colab.research.google.com/github/s-turland/SimDPSInpainting/blob/main/examples/quickstart.ipynb)

## Installation

```
pip install "simdps[acestep] @ git+https://github.com/s-turland/SimDPSInpainting"
```

SimDPS needs Python 3.10 or later and at least PyTorch 2.6. The ACE-Step weights
(about 5 GB) are downloaded on first use. Experiments were conducted on a slurm-based HPC system. If using on a personal machine, we recommend using a GPU with at least 8 GB of memory.

## Usage

From the command line:

```
simdps inpaint take.wav --gap 1:02.5 1:04.5
simdps inpaint take.wav --gap 62.5 64.5 --samples 4 --seed 0
simdps inpaint take.wav --gap 62.5 64.5 --source different_song.wav
simdps inpaint take.wav --gap 62.5 64.5 --source samples/
simdps inpaint take.wav --gap 62.5 64.5 --method search --samples 8
simdps inpaint take.wav --gap 62.5 64.5 --candidates 2 5
simdps inpaint take.wav --gap 62.5 64.5 --config warm_start_fast
simdps app
```

From Python:

```python
import simdps

model = simdps.load("acestep")
audio, sr = simdps.load_audio("take.wav")
out = model.inpaint(audio, sr, gap=(62.5, 64.5))
simdps.save_audio("take_simdps.wav", out.audio[0], out.sample_rate)
```

`method` selects SimDPS-Multi (`"multi"`, default), SimDPS-Solo (`"solo"`),
plain DPS (`"dps"`) or the search candidates without diffusion
(`"search"`). With `source`, the candidates come from another recording or a
folder of audio, which can be used to steer a gap towards different material.
`method="search"` lists the matches to choose from with `candidates`, and
`guide` selects a passage by time instead.

## Configs

Every setting of an experiment is in one YAML file, in
[`simdps/configs`](simdps/configs). Edit a config file, or change single
settings:

```
simdps inpaint take.wav --gap 62.5 64.5 --config my_experiment.yaml
simdps inpaint take.wav --gap 62.5 64.5 --config ismir26_solo --set candidates.distance=lowpass
```

```python
model.inpaint(audio, sr, gap, config="ismir26_solo")
model.inpaint(audio, sr, gap, config=["ismir26_solo", "candidates.weight=0.2"])
model.inpaint(audio, sr, gap, weight=0.2)
```

The defaults are the settings of the paper, about 55 s per gap on an 8 GB laptop
GPU. `warm_start_detailed` (17 s) and `warm_start_fast` (7 s) are faster as they
start from a renoised match at a low noise level. These speed setting work alongside any experiment config.

## Using another model

Any denoiser can be used through `simdps.priors`. For a model trained with
rectified flow on the latents of some autoencoder:

```python
from simdps import Inpainter
from simdps.priors import FlowPrior

prior = FlowPrior(net, sample_rate=44100, length=512,
                  hop_length=1024, encode=vae.encode, decode=vae.decode)
model = Inpainter(prior)
```

`EDMPrior` covers networks trained with the EDM preconditioning, and
`DenoiserPrior` any model that already provides `denoise(x, sigma)`. A config can
define a custom function that builds the prior (`prior: {target:
"my_package.models:make_prior"}`), so that a new model is one more YAML file. See
the documentation for details.

## Web interface (experimental)

```
simdps app
```

opens an experimental interface in the browser, served from a local server. 
The app demonstrates the joint search and diffusion editing capabilities 
alongside an audio editor, for creative, iterative use.
Select the gap on the waveform, search the recording for matching candidates,
choose which ones guide the model and with what weighting distribution, and compare the
results against the original. Intermediate estimates along the way can be heard
and sampled again with new seeds. Any audio can be banked and saved for use in further inpainting edits.
Results vary widely, depending on the input and settings used. See the
[quickstart](docs/quickstart.rst) for more information.

## Citation

```bibtex
@inproceedings{turland2026simdps,
  title     = {Music Inpainting with Similarity-Guided Latent Diffusion},
  author    = {Turland, Sean and Moliner, Eloi and V{\"a}lim{\"a}ki, Vesa},
  booktitle = {Proc. Int. Society for Music Information Retrieval Conf. (ISMIR)},
  year      = {2026}
}
```

## Acknowledgements

This work was supported by Nokia Solutions and Networks, the Research Council of
Finland (Grant no. 371845, REMUS), and the HUCE infrastructure of the Aalto School
of Electrical Engineering. We acknowledge the computational resources provided by
the Aalto University "Science-IT" project.

## Licence

MIT. `simdps/_vendor/acestep` is from
[ACE-Step 1.5](https://huggingface.co/ACE-Step/acestep-v15-base) under the
Apache License 2.0. `simdps/_vendor/cqtdiff` is based on
[CQTdiff](https://github.com/eloimoliner/CQTdiff), under the MIT license.


