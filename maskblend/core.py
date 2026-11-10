"""Mask growth, crop planning and distance-ramp paste-back.

The convention throughout is **0 = replace, 255 = keep**. Grey counts as
selected: a diffusion model takes a binary mask, so anything short of fully
opaque is inside the edit area and the ramp handles the transition at the edge.

Extracted from the LanPaint-for-Affinity experiment. The model, the pipeline and
the plugin are deliberately absent; this is only the masking and compositing
part.
"""

from __future__ import annotations

from PIL import Image, ImageFilter
import numpy as np

DEFAULT_THRESHOLD = 250
DEFAULT_GRID = 16


def mask_box(mask: Image.Image, threshold: int = DEFAULT_THRESHOLD):
    """Smallest rectangle holding every selected pixel.

    Returns ``(left, top, right, bottom)`` with inclusive edges, or ``None``
    when nothing is selected.
    """
    arr = np.asarray(mask.convert("L"), dtype=np.uint8)
    painted = np.argwhere(arr < threshold)
    if painted.size == 0:
        return None
    top, left = painted.min(axis=0)
    bottom, right = painted.max(axis=0)
    return int(left), int(top), int(right), int(bottom)


def binary_mask(mask: Image.Image, threshold: int = DEFAULT_THRESHOLD) -> Image.Image:
    """Turns a soft mask into the 0/255 mask a model expects."""
    return mask.convert("L").point(lambda v: 0 if v < threshold else 255)


def grow_mask(mask: Image.Image, grow: int) -> Image.Image:
    """Grows the **editable** area (the 0 pixels) by ``grow`` pixels."""
    if grow <= 0:
        return mask.copy()
    return mask.filter(ImageFilter.MinFilter(2 * int(grow) + 1))


def _pads(width: int, height: int, grid: int, margin: int) -> tuple[int, int]:
    return ((-width) % grid) + max(0, int(margin)), ((-height) % grid) + max(0, int(margin))


def pad_to_grid(
    image: Image.Image,
    mask: Image.Image,
    grid: int = DEFAULT_GRID,
    margin: int = 0,
):
    """Pads the right and bottom to a multiple of ``grid`` plus ``margin``.

    The image is extended by edge replication. The mask is extended by its own
    edge rather than with 255, so a selection that runs to the frame stays
    editable in the padding; a constant "keep" band at the picture edge is what
    produced a blurred strip in the original experiment.

    Returns ``(padded_image, padded_mask, (width, height))`` where the last item
    is the size to crop back to after generation.
    """
    width, height = image.size
    pad_w, pad_h = _pads(width, height, grid, margin)
    if not pad_w and not pad_h:
        return image, mask.convert("L"), (width, height)

    pixels = np.pad(np.asarray(image), ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    kept = np.pad(np.asarray(mask.convert("L")), ((0, pad_h), (0, pad_w)), mode="edge")
    return (
        Image.fromarray(pixels, image.mode),
        Image.fromarray(kept, "L"),
        (width, height),
    )


def plan_crop(
    box,
    size,
    selection_limit: int | None = None,
    target_area: int = 1048576,
    grid: int = DEFAULT_GRID,
    min_context: int = 32,
):
    """Grows a selection into the working rectangle sent to a model.

    The result always contains the whole selection and always stays inside the
    picture. It gains about half the short side of context, then grows towards
    ``target_area``. Sides are kept on the VAE ``grid`` when the selection still
    fits, which avoids a second internal crop shifting the patch.

    ``selection_limit`` is optional: pass ``None`` for a whole-layer run where
    the working area is the picture itself.
    """
    left, top, right, bottom = box
    image_w, image_h = size
    width, height = right - left + 1, bottom - top + 1

    if selection_limit is not None and width * height > selection_limit:
        raise ValueError(
            f"the selection covers {width * height / 1e6:.2f} MP, over the "
            f"{selection_limit / 1e6:.2f} MP limit"
        )
    if image_w < 32 or image_h < 32:
        raise ValueError(f"the picture is {image_w}x{image_h}; at least 32x32 is needed")

    pad = max(min_context, int(round(0.5 * min(width, height))))
    crop_w = min(image_w, width + 2 * pad)
    crop_h = min(image_h, height + 2 * pad)

    area = crop_w * crop_h
    if area < target_area:
        scale = (target_area / float(area)) ** 0.5
        crop_w = min(image_w, int(round(crop_w * scale)))
        crop_h = min(image_h, int(round(crop_h * scale)))

    crop_w = min(max(crop_w, width), image_w)
    crop_h = min(max(crop_h, height), image_h)

    grid_w = (crop_w // grid) * grid
    grid_h = (crop_h // grid) * grid
    if grid_w >= width:
        crop_w = grid_w
    if grid_h >= height:
        crop_h = grid_h

    def place(low: int, high: int, length: int, limit: int) -> int:
        lowest = max(0, high - length + 1)
        highest = min(low, limit - length)
        centred = int(round((low + high) / 2.0 - length / 2.0))
        return max(lowest, min(centred, highest))

    crop_left = place(left, right, crop_w, image_w)
    crop_top = place(top, bottom, crop_h, image_h)
    return crop_left, crop_top, crop_left + crop_w, crop_top + crop_h


def paste_back(
    original: Image.Image,
    generated: Image.Image,
    box,
    mask: Image.Image,
    grow: int = 0,
    ramp: int = 12,
    threshold: int = DEFAULT_THRESHOLD,
    feather: float = 0.6,
) -> Image.Image:
    """Pastes ``generated`` into ``original`` through the grown mask and ramp.

    Everything inside the *grown* mask is replaced outright. The weight then
    fades inward over ``ramp`` pixels from zero at the grown edge, which hides
    the model's own seam while still removing a rim of the old object just
    outside the painted selection.

    ``generated`` is aligned with the top-left of ``box`` and may be a crop of
    the same size as the box, or a full-size image.
    """
    from scipy import ndimage

    left, top, right, bottom = box
    keep = mask.crop(box).convert("L")

    edit = np.asarray(keep, dtype=np.uint8) < threshold
    if grow > 0:
        edit = ndimage.maximum_filter(edit, size=2 * int(grow) + 1)

    if ramp > 0:
        inside = ndimage.distance_transform_edt(edit)
        weight = np.clip(inside / float(ramp), 0.0, 1.0)
    else:
        weight = edit.astype(np.float32)

    alpha = Image.fromarray((weight * 255.0).astype(np.uint8))
    if feather > 0:
        alpha = alpha.filter(ImageFilter.GaussianBlur(feather))

    out = original.copy()
    out.paste(generated, (left, top), alpha)
    return out


def fill_alpha(image: Image.Image, keep: bool = False) -> Image.Image:
    """Makes deliberately transparent pixels opaque, as the plugin did.

    This is the ``keep_mask=0`` rule: every pixel with alpha below 255 becomes
    fully opaque. It applies to the **whole image**, not only the mask, so a
    layer with intentional transparency elsewhere should use ``keep=True`` and
    clear the alpha separately.
    """
    if keep:
        return image.copy()
    arr = np.array(image.convert("RGBA"))
    arr[:, :, 3] = np.where(arr[:, :, 3] < 255, 255, arr[:, :, 3])
    return Image.fromarray(arr, "RGBA")
