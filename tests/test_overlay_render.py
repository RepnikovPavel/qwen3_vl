"""L0: parse + draw must change image pixels (spatial skills are not chat-only)."""

from __future__ import annotations

import unittest

from PIL import Image

from demo.grounding_3d_viz import draw_3d_bboxes, generate_camera_params, parse_bbox_3d_from_text
from demo.grounding_viz import draw_grounding, parse_grounding
from qwen3_vl.skill_parsers import parse_skill


def _pixel_delta(a: Image.Image, b: Image.Image) -> int:
    pa, pb = a.convert("RGB").load(), b.convert("RGB").load()
    w, h = a.size
    n = 0
    for y in range(h):
        for x in range(w):
            if pa[x, y] != pb[x, y]:
                n += 1
    return n


class OverlayRenderTest(unittest.TestCase):
    def test_draw_boxes_changes_pixels(self):
        img = Image.new("RGB", (200, 100), (20, 20, 20))
        parsed = [{"bbox_2d": [100, 100, 800, 800], "label": "car"}]
        # draw_grounding expects 0..1000 coords scaled to image
        out = draw_grounding(img, parsed)
        self.assertGreater(_pixel_delta(img, out), 50)

    def test_draw_points_changes_pixels(self):
        img = Image.new("RGB", (200, 100), (20, 20, 20))
        parsed = [{"point_2d": [500, 500], "label": "p"}]
        out = draw_grounding(img, parsed)
        self.assertGreater(_pixel_delta(img, out), 10)

    def test_draw_3d_partial_corners_still_paints(self):
        img = Image.new("RGB", (640, 480), (30, 30, 30))
        cam = generate_camera_params(img, fov=60.0)
        # Box roughly in front of camera
        items = [
            {
                "bbox_3d": [0.0, 0.5, 8.0, 2.0, 1.5, 4.0, 0.0, 10.0, 0.0],
                "label": "car",
            }
        ]
        out = draw_3d_bboxes(img, cam, items)
        self.assertGreater(_pixel_delta(img, out), 20)

    def test_parse_point_and_bbox_from_cookbook_shaped_text(self):
        text = (
            '[{"bbox_2d": [10, 20, 100, 200], "label": "plate"}, '
            '{"point_2d": [500, 250], "label": "person", "role": "player"}]'
        )
        parsed = parse_grounding(text)
        kinds = set()
        for item in parsed:
            if "bbox_2d" in item:
                kinds.add("box")
            if "point_2d" in item:
                kinds.add("point")
        self.assertEqual(kinds, {"box", "point"})

    def test_parse_3d_cookbook_json(self):
        text = (
            '[{"bbox_3d":[1,2,10,1.5,1.6,4.0,0,5,0],"label":"car"},'
            '{"bbox_3d":[-1,2,12,1.4,1.5,3.8,0,0,0],"label":"car"}]'
        )
        parsed = parse_bbox_3d_from_text(text)
        self.assertGreaterEqual(len(parsed), 2)
        self.assertEqual(len(parsed[0]["bbox_3d"]), 9)


class BuildSkillOverlaysTest(unittest.TestCase):
    """_build_skill_overlays without a real session returns None (needs media meta).

    Full SSE path is covered in test_overlay_contract with a store mock.
    """

    def test_parse_skill_detection_and_points(self):
        det = parse_skill(
            "detection_2d",
            '[{"class":"vehicle","bbox_2d":[10,20,100,200]}]',
        )
        self.assertGreaterEqual(len(det), 1)
        pts = parse_skill(
            "spatial_understanding",
            '[{"point_2d":[400,300],"label":"sign"}]',
        )
        self.assertEqual(pts[0]["point_2d"], [400, 300])


if __name__ == "__main__":
    unittest.main()
