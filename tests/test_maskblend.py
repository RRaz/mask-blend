import unittest

import numpy as np
from PIL import Image, ImageDraw

from maskblend import (
    binary_mask,
    fill_alpha,
    grow_mask,
    mask_box,
    pad_to_grid,
    paste_back,
    plan_crop,
)


class MaskTests(unittest.TestCase):
    def test_grey_is_selected(self):
        mask = Image.new("L", (100, 80), 255)
        ImageDraw.Draw(mask).rectangle([20, 10, 60, 40], fill=200)
        self.assertEqual(mask_box(mask), (20, 10, 60, 40))

    def test_nothing_painted(self):
        self.assertIsNone(mask_box(Image.new("L", (10, 10), 255)))

    def test_binary_mask_turns_grey_into_edit(self):
        mask = Image.new("L", (20, 20), 255)
        ImageDraw.Draw(mask).rectangle([4, 4, 15, 15], fill=200)
        binary = binary_mask(mask)
        self.assertEqual(binary.getpixel((10, 10)), 0)
        self.assertEqual(binary.getpixel((0, 0)), 255)

    def test_grow_expands_the_edit_area(self):
        mask = Image.new("L", (40, 40), 255)
        ImageDraw.Draw(mask).rectangle([18, 18, 22, 22], fill=0)
        grown = grow_mask(mask, 3)
        self.assertEqual(grown.getpixel((16, 20)), 0)
        self.assertEqual(grown.getpixel((10, 20)), 255)
        self.assertEqual(grown.getpixel((20, 20)), 0)


class CropTests(unittest.TestCase):
    def test_crop_contains_selection_and_stays_on_grid(self):
        box = (500, 400, 900, 700)
        crop = plan_crop(box, (2048, 1536), 629145, 1048576)
        left, top, right, bottom = crop
        self.assertTrue(left <= 500 and top <= 400)
        self.assertTrue(right >= 900 and bottom >= 700)
        self.assertEqual((right - left) % 16, 0)
        self.assertEqual((bottom - top) % 16, 0)

    def test_selection_limit_is_enforced(self):
        with self.assertRaises(ValueError):
            plan_crop((0, 0, 1000, 1000), (2048, 1536), 629145, 1048576)

    def test_whole_layer_has_no_selection_limit(self):
        crop = plan_crop((0, 0, 699, 499), (700, 500), None, 1048576)
        self.assertEqual(crop, (0, 0, 700, 500))


class PaddingTests(unittest.TestCase):
    def test_mask_that_touches_the_edge_pads_as_editable(self):
        image = Image.new("RGB", (70, 50), (10, 20, 30))
        mask = Image.new("L", (70, 50), 255)
        for y in range(50):
            mask.putpixel((69, y), 0)
        padded, padded_mask, size = pad_to_grid(image, mask, margin=16)
        self.assertEqual(size, (70, 50))
        self.assertEqual(padded.width % 16, 0)
        self.assertEqual(padded.height % 16, 0)
        self.assertEqual(padded_mask.getpixel((padded.width - 1, 0)), 0)

    def test_keep_edge_pads_as_keep(self):
        image = Image.new("RGB", (70, 50), (10, 20, 30))
        mask = Image.new("L", (70, 50), 255)
        padded, padded_mask, _ = pad_to_grid(image, mask, margin=16)
        self.assertEqual(padded_mask.getpixel((padded.width - 1, 0)), 255)


class PasteTests(unittest.TestCase):
    def test_two_regions_are_replaced_and_the_gap_is_not(self):
        original = Image.new("RGB", (400, 200), (20, 20, 20))
        mask = Image.new("L", (400, 200), 255)
        draw = ImageDraw.Draw(mask)
        draw.ellipse([60, 60, 140, 140], fill=0)
        draw.ellipse([260, 60, 340, 140], fill=0)
        generated = Image.new("RGB", (400, 200), (180, 180, 180))

        out = paste_back(original, generated, (0, 0, 400, 200), mask, grow=8, ramp=4)
        self.assertEqual(out.getpixel((100, 100))[0], 180)
        self.assertEqual(out.getpixel((300, 100))[0], 180)
        self.assertEqual(out.getpixel((200, 100))[0], 20)

    def test_ramp_fades_at_the_mask_edge(self):
        original = Image.new("RGB", (80, 80), (0, 0, 0))
        mask = Image.new("L", (80, 80), 255)
        ImageDraw.Draw(mask).rectangle([20, 20, 59, 59], fill=0)
        generated = Image.new("RGB", (80, 80), (255, 255, 255))

        out = paste_back(original, generated, (0, 0, 80, 80), mask, grow=0, ramp=8, feather=0)
        self.assertEqual(out.getpixel((40, 40))[0], 255)
        edge = out.getpixel((20, 40))[0]
        self.assertGreater(edge, 0)
        self.assertLess(edge, 255)
        self.assertEqual(out.getpixel((19, 40))[0], 0)


class AlphaTests(unittest.TestCase):
    def test_fill_alpha(self):
        image = Image.new("RGBA", (2, 2), (10, 20, 30, 128))
        filled = fill_alpha(image)
        self.assertEqual(filled.getpixel((0, 0))[3], 255)
        kept = fill_alpha(image, keep=True)
        self.assertEqual(kept.getpixel((0, 0))[3], 128)


if __name__ == "__main__":
    unittest.main()
