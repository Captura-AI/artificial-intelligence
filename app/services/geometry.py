def _bbox_contains_point(bbox: list[int], x: float, y: float) -> bool:
    return bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]


def _plate_center(bbox: list[int]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _bbox_iou(box_a: list[int], box_b: list[int]) -> float:
    inter_x1 = max(box_a[0], box_b[0])
    inter_y1 = max(box_a[1], box_b[1])
    inter_x2 = min(box_a[2], box_b[2])
    inter_y2 = min(box_a[3], box_b[3])

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    if intersection == 0:
        return 0.0

    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _expand_bbox(
    bbox: list[int], image_size: tuple[int, int], padding_px: int
) -> list[int]:
    width, height = image_size
    x1, y1, x2, y2 = bbox
    return [
        max(0, int(x1) - padding_px),
        max(0, int(y1) - padding_px),
        min(width, int(x2) + padding_px),
        min(height, int(y2) + padding_px),
    ]
