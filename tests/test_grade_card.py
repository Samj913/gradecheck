from pathlib import Path

import numpy as np
from PIL import Image

from grade_card import TagLikeCardGrader


def _make_synthetic_card(path: Path, off_x: int = 0, off_y: int = 0, edge_whitening: bool = False) -> None:
    h, w = 400, 300
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # blue-ish border region
    img[:, :] = (30, 70, 180)

    # artwork window (variable centering)
    top, left = 60 + off_y, 45 + off_x
    bottom, right = h - 60 + off_y, w - 45 + off_x
    top = max(30, min(top, h - 170))
    left = max(30, min(left, w - 130))
    bottom = max(top + 120, min(bottom, h - 30))
    right = max(left + 90, min(right, w - 30))

    img[top:bottom, left:right] = (160, 90, 60)

    # subtle texture noise
    noise = np.random.default_rng(7).integers(0, 18, size=(bottom - top, right - left, 3), dtype=np.uint8)
    img[top:bottom, left:right] = np.clip(img[top:bottom, left:right] + noise, 0, 255)

    if edge_whitening:
        img[:8, :, :] = 245
        img[-8:, :, :] = 245

    Image.fromarray(img).save(path)


def test_grader_scores_and_ordering(tmp_path: Path) -> None:
    clean = tmp_path / "clean.png"
    damaged = tmp_path / "damaged.png"

    _make_synthetic_card(clean, off_x=0, off_y=0, edge_whitening=False)
    _make_synthetic_card(damaged, off_x=12, off_y=-8, edge_whitening=True)

    grader = TagLikeCardGrader()

    clean_result = grader.grade(clean)
    damaged_result = grader.grade(damaged)

    assert 1.0 <= clean_result.overall_score <= 10.0
    assert clean_result.overall_score > damaged_result.overall_score
    assert clean_result.edge_score >= damaged_result.edge_score
