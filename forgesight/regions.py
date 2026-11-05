"""Pixel geometry of provisional anomaly-response support, not physical damage."""

import math


def response_regions(score_map, threshold: float) -> dict:
    import cv2
    import numpy as np

    values = np.asarray(score_map)
    if (values.ndim != 2 or not values.size or not np.isfinite(values).all()
            or not math.isfinite(threshold)):
        raise ValueError('Expected a finite nonempty 2-D score map and threshold.')
    support = (values > threshold).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(support, connectivity=8)
    regions = []
    for component in range(1, count):
        x, y, width, height, area = (int(v) for v in stats[component])
        regions.append({'region_id': component, 'area_px': area,
                        'bbox_xywh_px': [x, y, width, height],
                        'bbox_diagonal_px': math.hypot(width, height)})
    return {'units': 'pixels', 'measurement': 'thresholded_model_response_not_physical_damage',
            'threshold': threshold, 'pixel_threshold_validated': False,
            'connectivity': 8, 'morphology': 'none',
            'source_size_px': [values.shape[1], values.shape[0]],
            'total_area_px': int(support.sum()), 'regions': regions,
            'length_definition': 'bounding-box diagonal proxy; not measured scratch length'}
