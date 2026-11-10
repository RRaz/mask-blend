# mask-blend

Mask growth, crop planning and distance-ramp paste-back, extracted from the
retired **LanPaint for Affinity** experiment. This is the reusable fragment, not
the plugin: no model, no pipeline, no UI.

The full story is in
[RRaz/LanPaint-diffusers](https://github.com/RRaz/LanPaint-diffusers), including
the [post-mortem](https://github.com/RRaz/LanPaint-diffusers/blob/main/POST-MORTEM.md).

## Convention

**0 = replace, 255 = keep.** Grey counts as selected: a diffusion model takes a
binary mask, so anything short of fully opaque is inside the edit area and the
ramp handles the transition at the edge.

## What is here

| Function | Purpose |
|---|---|
| `mask_box` | smallest rectangle holding every selected pixel |
| `binary_mask` | soft/grey mask to 0/255 |
| `grow_mask` | grow the editable area by `n` pixels |
| `plan_crop` | grow a selection into a working rectangle, inside the frame and on the VAE grid |
| `pad_to_grid` | pad right/bottom to the grid, with an optional edge margin |
| `paste_back` | replace the grown mask, then fade inward over a distance ramp |
| `fill_alpha` | the old `keep_mask=0` rule: make partly transparent pixels opaque |

## Example

```python
from PIL import Image
from maskblend import mask_box, binary_mask, grow_mask, plan_crop, paste_back

original = Image.open("before.png").convert("RGB")
mask = Image.open("mask.png").convert("L")       # 0 = edit, 255 = keep
generated = Image.open("patch.png").convert("RGB")

box = plan_crop(mask_box(mask), original.size, selection_limit=629145)
work_mask = grow_mask(binary_mask(mask), 8)
out = paste_back(original, generated, box, mask, grow=8, ramp=4)
out.save("after.png")
```

`generated` must be aligned with the top-left of `box` and at least as large as
the box. A full-size generated image also works.

## Command line

```console
python -m maskblend.cli --original before.png --generated patch.png \
    --mask mask.png --out after.png --grow 8 --ramp 4
```

Add `--box left,top,right,bottom` when `patch.png` is a crop rather than a
full-size image.

## Install and test

```console
pip install numpy pillow scipy
python -m unittest discover -s tests -v
```

The tests cover soft and grey masks, irregular and multiple regions, a mask that
touches the frame, crop clamping and grid alignment, and the ramp.

## Notes

- `grow_mask` and `paste_back(grow=...)` grow the **edit** area; they do not
  shrink a selection.
- `ramp=0` gives a hard replacement at the grown mask edge; `feather` smooths
  the alpha by 0.6 px by default.
- `fill_alpha` applies to the whole image, not only the mask. A layer with
  intentional transparency elsewhere should pass `keep=True` and clear its
  alpha separately.
- The 250 threshold is the old working value: grey below it is edited. It is a
  parameter everywhere for a reason.

## Licence

GPL-3.0, as the project it came from.
