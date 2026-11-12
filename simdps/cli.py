"""Command line interface.

    simdps inpaint take.wav --gap 31 33
    simdps inpaint take.wav --gap 1:02.5 1:04 --source other.wav --samples 4
    simdps inpaint take.wav --gap 31 33 --method search --samples 8
    simdps inpaint take.wav --gap 31 33 --candidates 2 5
    simdps inpaint take.wav --gap 31 33 --source samples/
    simdps inpaint take.wav --gap 31 33 --config warm_start_fast
    simdps inpaint take.wav --gap 31 33 --config ismir26_solo --set candidates.distance=lowpass
    simdps app
"""

import argparse
import os

import torch


def seconds(s):
    """Parse ``12.5``, ``1:02.5`` or ``1:01:02.5``."""
    t = 0.0
    for part in s.split(":"):
        t = 60 * t + float(part)
    return t


def main(argv=None):
    p = argparse.ArgumentParser(prog="simdps", description="Inpaint long gaps in music recordings.")
    sub = p.add_subparsers(dest="command", required=True)

    q = sub.add_parser("inpaint", help="fill a gap in an audio file")
    q.add_argument("input")
    q.add_argument("--gap", nargs=2, type=seconds, required=True, metavar=("START", "END"),
                   help="gap in seconds or m:ss")
    q.add_argument("-o", "--output", help="output file (default: <input>_simdps.wav)")
    q.add_argument("--method", choices=["multi", "solo", "dps", "search"], help="default: multi")
    q.add_argument("--config", nargs="+", metavar="CONFIG",
                   help="configs of the package or YAML files, later ones on top of earlier ones")
    q.add_argument("--set", nargs="+", default=[], metavar="SECTION.KEY=VALUE", dest="settings",
                   help="change single settings, e.g. candidates.weight=0.2")
    q.add_argument("--source", nargs="+", help="search these files or folders instead of the input")
    q.add_argument("--guide", nargs="+", type=seconds, metavar="TIME",
                   help="use the passages starting here (in the source) instead of searching")
    q.add_argument("--candidates", nargs="+", type=int, metavar="RANK",
                   help="keep these matches, numbered as by --method search")
    q.add_argument("--match", default="latent", choices=["latent", "audio", "bands"], help=argparse.SUPPRESS)
    q.add_argument("--samples", type=int, default=1, help="number of alternative fills")
    q.add_argument("--steps", type=int, help="sampler steps, overriding the config")
    q.add_argument("--seed", type=int)
    q.add_argument("--raw", action="store_true", help="save the generated window only")
    q.add_argument("--model", default="acestep", choices=["acestep", "acestep-turbo"])
    q.add_argument("--device")
    q.add_argument("--fp32", action="store_true", help="run the model in float32")

    a = sub.add_parser("app", help="open the web interface")
    a.add_argument("--model", default="acestep", choices=["acestep", "acestep-turbo"])
    a.add_argument("--device")
    a.add_argument("--port", type=int, default=7860)
    a.add_argument("--no-browser", action="store_true", help="do not open a browser")
    a.add_argument("--bank", help="folder of saved signals (default bank/ in the repository, or ~/simdps-bank)")

    args = p.parse_args(argv)
    if args.command == "app":
        from .app import BANK, launch

        return launch(args.model, args.device, port=args.port, browser=not args.no_browser,
                      bank=args.bank or BANK)
    inpaint(args)


def inpaint(args):
    import simdps

    dtype = torch.float32 if args.fp32 else None
    settings = simdps.load_config(args.config, args.settings)
    if "prior" in settings:
        model = simdps.load(settings, device=args.device, dtype=dtype)
    else:
        model = simdps.load(args.model, device=args.device, dtype=dtype, config=settings or None)
    audio, sr = simdps.load_audio(args.input)
    source = args.source[0] if args.source and len(args.source) == 1 else args.source
    candidates = args.candidates and [i - 1 for i in args.candidates]
    guide = args.guide
    method = args.method or model.config.get("method", "multi")
    picked = guide if guide is not None else candidates
    single = args.method is None and "method" not in settings and picked is not None and len(picked) == 1
    if single:
        method = "solo"  # one chosen passage is followed closely, as by SimDPS-Solo
    if args.match != "latent" and guide is None and method != "dps":
        # match by audio features, then guide with what was found
        n = 15 if method == "multi" else max(1, args.samples)
        found = model.search(audio, sr, tuple(args.gap), source=source, n=n, match=args.match)
        guide, source = [(t, where) for t, _, where in found], None
    out = model.inpaint(
        audio, sr, tuple(args.gap), method="solo" if single else args.method, source=source, guide=guide,
        candidates=candidates,
        n_samples=args.samples, steps=args.steps, seed=args.seed, raw=args.raw,
    )

    stem = os.path.splitext(args.output or os.path.splitext(args.input)[0] + "_simdps.wav")
    for i, y in enumerate(out.audio):
        path = stem[0] + (f"_{i + 1}" if len(out.audio) > 1 else "") + (stem[1] or ".wav")
        simdps.save_audio(path, y, out.sample_rate)
        if method == "search":
            # where the match starts, as --guide takes it
            c = out.candidates
            t = (c.starts[i].item() + model.buffer) * model.prior.hop_length / out.sample_rate
            where = os.path.basename(c.sources[i]) + " " if c.sources else ""
            print(f"{path}  ({where}{t:.1f} s)")
        else:
            print(path)


if __name__ == "__main__":
    main()
