from types import SimpleNamespace

import pytest

from app.services import analyzer


def test_bbox_iou_identical_boxes_is_one():
    assert analyzer._bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0


def test_bbox_iou_disjoint_boxes_is_zero():
    assert analyzer._bbox_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_bbox_iou_partial_overlap():
    # 50px overlap, union 150 -> 1/3
    assert analyzer._bbox_iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(50 / 150)


def _motor(bbox):
    return SimpleNamespace(bbox=bbox)


def test_match_motor_picks_highest_iou_and_skips_used():
    vehicle = [0, 0, 100, 100]
    motors = [_motor([0, 0, 100, 100]), _motor([0, 0, 50, 50])]

    # Best overlap is the full box (index 0).
    assert analyzer._match_motor_to_vehicle(vehicle, motors, set()) == 0
    # Once index 0 is used, index 1 still overlaps enough (IoU 0.25? -> below 0.3) => None
    assert analyzer._match_motor_to_vehicle(vehicle, motors, {0}) is None


def test_match_motor_returns_none_below_min_iou():
    vehicle = [0, 0, 100, 100]
    motors = [_motor([90, 90, 200, 200])]  # tiny overlap
    assert analyzer._match_motor_to_vehicle(vehicle, motors, set()) is None


def _plate(bbox, text, text_conf, det_conf=0.5):
    return SimpleNamespace(
        bbox=bbox,
        text=text,
        text_confidence=text_conf,
        detection_confidence=det_conf,
    )


def test_match_plate_prefers_higher_text_confidence():
    vehicle = [0, 0, 100, 100]
    plates = [
        _plate([10, 10, 30, 20], "B1", 0.5),
        _plate([40, 40, 60, 50], "B2", 0.9),
    ]
    # Both centers inside the vehicle; the higher-confidence plate (index 1) wins.
    assert analyzer._match_plate_to_vehicle(vehicle, plates, set()) == 1


def test_match_plate_ignores_textless_and_outside_plates():
    vehicle = [0, 0, 100, 100]
    plates = [
        _plate([10, 10, 30, 20], None, 0.9),  # no text -> skipped
        _plate([200, 200, 230, 220], "B2", 0.9),  # outside vehicle -> skipped
    ]
    assert analyzer._match_plate_to_vehicle(vehicle, plates, set()) is None


def test_bbox_contains_point():
    assert analyzer._bbox_contains_point([0, 0, 10, 10], 5, 5) is True
    assert analyzer._bbox_contains_point([0, 0, 10, 10], 15, 5) is False
