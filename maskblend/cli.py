"""Command-line paste-back: apply a generated patch through a mask."""

from __future__ import annotations

import argparse
from PIL import Image

from .core import DEFAULT_THRESHOLD, paste_back


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Paste a generated image into the original through a mask, "
        "with optional growth and a distance ramp."
    )
    parser.add_argument("--original", required=True)
    parser.add_argument("--generated", required=True)
    parser.add_argument("--mask", required=True, help="0 replaces, 255 keeps")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--box",
        help="left,top,right,bottom of the generated patch; default is the whole original",
    )
    parser.add_argument("--grow", type=int, default=0)
    parser.add_argument("--ramp", type=int, default=12)
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD)
    parser.add_argument("--feather", type=float, default=0.6)
    args = parser.parse_args(argv)

    original = Image.open(args.original).convert("RGB")
    generated = Image.open(args.generated).convert("RGB")
    mask = Image.open(args.mask).convert("L")

    if mask.size != original.size:
        parser.error(f"mask is {mask.size}, original is {original.size}")
    box = (
        tuple(int(value) for value in args.box.split(","))
        if args.box
        else (0, 0, original.width, original.height)
    )
    left, top, right, bottom = box
    if generated.width < right - left or generated.height < bottom - top:
        parser.error("generated image is smaller than the box")

    out = paste_back(
        original,
        generated,
        box,
        mask,
        grow=args.grow,
        ramp=args.ramp,
        threshold=args.threshold,
        feather=args.feather,
    )
    out.save(args.out)
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
