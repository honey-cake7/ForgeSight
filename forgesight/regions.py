"""Pixel geometry of provisional anomaly-response support, not physical damage."""

import math


def response_regions(score_map, threshold: float, *, mvtec_nut_layout=False) -> dict:
    import cv2
    import numpy as np

    values = np.asarray(score_map)
    if (values.ndim != 2 or not values.size or not np.isfinite(values).all()
            or not math.isfinite(threshold)):
        raise ValueError('Expected a finite nonempty 2-D score map and threshold.')
    support = (values > threshold).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(support, connectivity=8)
    zone_counts = None
    zone_names = ('central opening', 'inner rim', 'annular face',
                  'outer perimeter or nearby background')
    if mvtec_nut_layout:
        if values.shape != (700, 700):
            raise ValueError('Approximate nut areas require the original 700x700 MVTec layout.')
        # Image-centered radial bands are descriptive only: no nut segmentation or
        # pixel-level defect threshold has been validated for these labels.
        yy, xx = np.ogrid[:700, :700]
        radius = np.hypot(xx - 349.5, yy - 349.5) / 350.0
        band = np.digitize(radius, (0.23, 0.34, 0.69))
        zone_counts = np.bincount((labels[support != 0] * 4 + band[support != 0]),
                                  minlength=count * 4).reshape(count, 4)
    regions = []
    for component in range(1, count):
        x, y, width, height, area = (int(v) for v in stats[component])
        region = {'region_id': component, 'area_px': area,
                        'bbox_xywh_px': [x, y, width, height],
                        'bbox_diagonal_px': math.hypot(width, height)}
        if zone_counts is not None:
            band_counts = zone_counts[component]
            dominant = int(np.argmax(band_counts))
            second = int(np.argsort(band_counts)[-2])
            if area > values.size * 0.25:
                zone = 'area undetermined; response spans multiple areas'
            elif band_counts[dominant] / area >= 0.75:
                zone = zone_names[dominant]
            elif abs(dominant - second) == 1 and (band_counts[dominant] + band_counts[second]) / area >= 0.90:
                zone = ' / '.join(zone_names[index] for index in sorted((dominant, second)))
            else:
                zone = 'area undetermined; response spans multiple areas'
            region['approximate_nut_area'] = zone
        regions.append(region)
    regions.sort(key=lambda region: (-region['area_px'], region['region_id']))
    primary = regions[0] if regions else None
    return {'units': 'pixels', 'measurement': 'thresholded_model_response_not_physical_damage',
            'threshold': threshold, 'pixel_threshold_validated': False,
            'connectivity': 8, 'morphology': 'none',
            'source_size_px': [values.shape[1], values.shape[0]],
            'total_area_px': int(support.sum()), 'regions': regions,
            'primary_region_id': primary['region_id'] if primary else None,
            'additional_region_count': max(0, len(regions) - 1),
            'area_reference': 'full_image_not_nut_or_physical_damage',
            'zone_method': ('approximate_centered_mvtec_radial_bands_v1' if mvtec_nut_layout
                            else 'unavailable_for_this_source'),
            'length_definition': 'bounding-box diagonal proxy; not measured scratch length'}
