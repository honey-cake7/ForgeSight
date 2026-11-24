"""Opt-in benchmark defect features, with separate nut and screw geometry."""

import json
import math
import re
from time import perf_counter

from forgesight.defect_severity import assess_severity


FEATURE_NAMES = ("elongation", "chroma_difference")
LABELS = ("scratch", "discoloration", "unknown")
FOUR_TYPES = ("bent", "flip", "scratch", "discoloration")
FOREST_FEATURES = FEATURE_NAMES + (
    "response_fraction", "primary_fraction", "radial_offset", "peak_score",
    "outline_solidity", "outline_aspect", "face_brightness", "rim_brightness",
    "face_contrast", "face_chroma", "response_brightness",
)
BENCHMARK_MODES = ("SAVED_IMAGE_REPLAY", "USER_DECLARED_MVTEC_METAL_NUT_UPLOAD")
SCREW_TYPES = ("manipulated_front", "scratch_head", "scratch_neck", "thread_side", "thread_top")
SCREW_FEATURES = (
    "object_aspect", "head_tip_width_ratio", "response_fraction", "response_axial_mean",
    "response_axial_std", "response_transverse_mean", "response_transverse_std",
    "peak_axial", "peak_transverse", "head_response_fraction", "neck_response_fraction",
    "thread_response_fraction", "tip_response_fraction", "head_texture", "neck_texture",
    "thread_texture", "response_contrast", "peak_score",
)


def load_profile(path):
    with open(path, encoding="utf-8") as stream:
        profile = json.load(stream)
    kind = profile.get("kind") if isinstance(profile, dict) else None
    features = SCREW_FEATURES if kind == "screw_decision_tree" else FOREST_FEATURES if kind == "four_type_forest" else FEATURE_NAMES
    if (not isinstance(profile, dict) or profile.get("schema_version") != 1
            or profile.get("feature_names") != list(features)
            or not isinstance(profile.get("profile_id"), str) or not profile["profile_id"]
            or not isinstance(profile.get("model_sha256"), str)
            or not re.fullmatch(r"[a-f0-9]{64}", profile["model_sha256"])
            or not _finite(profile.get("image_threshold"))):
        raise ValueError("Invalid defect-type profile.")
    if profile.get("kind") == "preview_rules":
        if not _finite(profile.get("min_chroma_difference")) or profile["min_chroma_difference"] <= 0:
            raise ValueError("Invalid preview rule thresholds.")
    elif profile.get("kind") == "decision_tree":
        _validate_node(profile.get("tree"), depth=0)
    elif kind == "screw_decision_tree":
        if profile.get("category") != "screw" or profile.get("labels") != list(SCREW_TYPES):
            raise ValueError("Invalid screw classifier domain/labels.")
        _validate_node(profile.get("tree"), depth=0, screw=True)
    elif profile.get("kind") == "four_type_forest":
        trees = profile.get("trees")
        if (not isinstance(trees, list) or len(trees) != 31
                or type(profile.get("min_votes")) is not int or not 16 <= profile["min_votes"] <= 31):
            raise ValueError("Invalid four-type voting profile.")
        for tree in trees:
            _validate_node(tree, depth=0, forest=True)
    else:
        raise ValueError("Unsupported defect-type profile kind.")
    return profile


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _validate_node(node, *, depth, forest=False, screw=False):
    if not isinstance(node, dict):
        raise ValueError("Invalid defect-type tree node.")
    labels = SCREW_TYPES + ("unknown",) if screw else FOUR_TYPES + ("unknown",) if forest else LABELS
    features = SCREW_FEATURES if screw else FOREST_FEATURES if forest else FEATURE_NAMES
    if set(node) == {"label"} and node["label"] in labels:
        return
    if (depth >= (4 if screw else 5 if forest else 3) or set(node) != {"feature", "threshold", "left", "right"}
            or node["feature"] not in features or not _finite(node["threshold"])):
        raise ValueError("Invalid defect-type tree split.")
    _validate_node(node["left"], depth=depth + 1, forest=forest, screw=screw)
    _validate_node(node["right"], depth=depth + 1, forest=forest, screw=screw)


def predict_type(features, profile):
    if features is None or not all(_finite(features.get(name)) for name in profile.get("feature_names", FEATURE_NAMES)):
        return "unknown"
    if profile["kind"] == "screw_decision_tree":
        import struct
        try:
            values = {name: struct.unpack('f', struct.pack('f', features[name]))[0] for name in SCREW_FEATURES}
        except (OverflowError, struct.error):
            return "unknown"
        if not all(_finite(v) for v in values.values()):
            return "unknown"
        return _tree_label(profile["tree"], values)
    if profile["kind"] == "preview_rules":
        if features["chroma_difference"] >= profile["min_chroma_difference"]:
            return "discoloration"
        return "unknown"
    if profile["kind"] == "four_type_forest":
        from collections import Counter
        votes = Counter(_tree_label(tree, features) for tree in profile["trees"])
        label, count = votes.most_common(1)[0]
        return label if count >= profile["min_votes"] else "unknown"
    return _tree_label(profile["tree"], features)


def _tree_label(node, features):
    while "label" not in node:
        node = node["left" if features[node["feature"]] <= node["threshold"] else "right"]
    return node["label"]


def assess_response(record, image, score_map, profile):
    """Attach an explicitly provisional assessment to eligible vision records."""
    modes = (("SAVED_IMAGE_REPLAY", "USER_DECLARED_MVTEC_SCREW_UPLOAD")
             if profile is not None and profile.get("kind") == "screw_decision_tree" else BENCHMARK_MODES)
    if (profile is None or record["candidate"] != "ANOMALY_CANDIDATE"
            or record["source"]["mode"] not in modes):
        return
    import cv2

    started = perf_counter()
    features = outline = None
    try:
        model = record["model"]
        matches_profile = (model["export_sha256"] == profile["model_sha256"]
                           and model["image_threshold"] == profile["image_threshold"])
        if matches_profile:
            if profile["kind"] == "four_type_forest":
                features, outline = _four_type_features_and_outline(image, score_map, record["geometry"])
            elif profile["kind"] == "screw_decision_tree":
                if record["geometry"].get("threshold") == profile["image_threshold"]:
                    features = screw_features(image, score_map, record["geometry"])
            else:
                features = response_features(image, score_map, record["geometry"])
    except (cv2.error, ValueError, ArithmeticError):
        features = None
    label = predict_type(features, profile)
    record["assessment"] = {"schema_version": 1, "profile_id": profile["profile_id"],
                            "method": profile["kind"], "scope": "PROVISIONAL_BENCHMARK_IMAGE",
                            "defect_type": label, "features": features}
    record["timings"]["defect_type_ms"] = (perf_counter() - started) * 1000
    if profile["kind"] == "four_type_forest":
        started = perf_counter()
        record["assessment"]["severity"] = assess_severity(label, record["geometry"], outline)
        record["timings"]["defect_severity_ms"] = (perf_counter() - started) * 1000


def response_features(image, score_map, geometry):
    """Measure the largest predicted component using original RGB pixels."""
    import cv2
    import numpy as np

    regions = geometry["regions"]
    if not regions:
        return None
    region = regions[0]
    x, y, width, height = region["bbox_xywh_px"]
    padding = 12
    left, top = max(0, x - padding), max(0, y - padding)
    right = min(image.width, x + width + padding)
    bottom = min(image.height, y + height + padding)
    support = (score_map[top:bottom, left:right] > geometry["threshold"]).astype(np.uint8)
    _, components, stats, _ = cv2.connectedComponentsWithStats(support, connectivity=8)
    expected = [x - left, y - top, width, height, region["area_px"]]
    matches = np.flatnonzero(np.all(stats[1:] == expected, axis=1))
    if len(matches) != 1 or region["area_px"] < 16:
        return None

    response = components == matches[0] + 1
    coordinates = np.column_stack(np.nonzero(response)).astype(np.float64)
    eigenvalues = np.linalg.eigvalsh(np.cov(coordinates, rowvar=False, bias=True))
    # Add one pixel of variance so a single-pixel-wide line stays finite.
    elongation = float(np.sqrt((eigenvalues[1] + 1) / (eigenvalues[0] + 1)))

    component = response.astype(np.uint8)
    outer = cv2.dilate(component, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    inner = cv2.dilate(component, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    local_reference = (outer != 0) & (inner == 0) & (support == 0)
    if np.count_nonzero(local_reference) < 32:
        return None
    rgb = np.asarray(image.crop((left, top, right, bottom)), dtype=np.int16)
    # Red–green and blue–green remove a shared brightness offset and avoid
    # the cost of initializing an OpenCV Lab conversion in each cold replay.
    chroma = rgb[:, :, (0, 2)] - rgb[:, :, 1, None]
    response_color = np.median(chroma[response], axis=0)
    reference_color = np.median(chroma[local_reference], axis=0)
    return {"elongation": float(np.float32(elongation)),
            "chroma_difference": float(np.float32(np.linalg.norm(response_color - reference_color)))}


def four_type_features(image, score_map, geometry):
    """Combine the anomaly response with the outline and face of a benchmark nut."""
    features, _ = _four_type_features_and_outline(image, score_map, geometry)
    return features


def _four_type_features_and_outline(image, score_map, geometry):
    import cv2
    import numpy as np

    features = response_features(image, score_map, geometry)
    if features is None:
        return None, None
    rgb = cv2.resize(np.asarray(image), (96, 96), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # The benchmark uses a dark background visible at the image border.
    border = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
    mask = (gray > np.median(border) + 12).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    (cx, cy), sides, _ = cv2.minAreaRect(contour)
    if area < 100 or min(sides) <= 0:
        return None, None
    yy, xx = np.ogrid[:96, :96]
    radius = np.hypot(xx - cx, yy - cy) / (sum(sides) / 4)
    face = (radius > 0.5) & (radius < 0.85) & (mask != 0)
    rim = (radius > 0.85) & (radius < 1.05) & (mask != 0)
    if not np.any(face) or not np.any(rim):
        return None, None
    response = cv2.resize(score_map, (96, 96), interpolation=cv2.INTER_AREA)
    peak = np.unravel_index(response.argmax(), response.shape)
    color = rgb[face].astype(np.int16)
    features = dict(features, response_fraction=geometry["total_area_px"] / score_map.size,
                primary_fraction=geometry["regions"][0]["area_px"] / score_map.size,
                radial_offset=float(radius[peak]), peak_score=float(response[peak]),
                outline_solidity=area / cv2.contourArea(cv2.convexHull(contour)),
                outline_aspect=min(sides) / max(sides), face_brightness=float(np.mean(gray[face])),
                rim_brightness=float(np.mean(gray[rim])), face_contrast=float(np.std(gray[face])),
                face_chroma=float(np.percentile(color.max(axis=1) - color.min(axis=1), 95)),
                response_brightness=float(gray[peak]))
    # The benchmark is square. Reuse the outline from subtype extraction.
    scale = image.width / gray.shape[1]
    outline = (sum(sides) / 2 * scale, area * scale ** 2)
    return features, outline


def screw_features(image, score_map, geometry):
    """Locate a predicted response along the screw axis; never use annotation masks.

    A dark-object silhouette supplies a principal axis; the wider end is the head.
    Head/shank/thread bands are approximate benchmark regions, not measured geometry.
    Feature computation uses a fixed 256px grid for matching fit and inference paths.
    """
    import cv2
    import numpy as np

    values = np.asarray(score_map)
    if (values.shape != (image.height, image.width) or not np.isfinite(values).all()
            or geometry.get("source_size_px") != [image.width, image.height]
            or not _finite(geometry.get("threshold"))):
        return None
    rgb = np.asarray(image.convert("RGB").resize((256, 256), resample=2))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    count, components, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count < 2:
        return None
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    body = components == component
    area = int(body.sum())
    if not 128 <= area <= gray.size * .65:
        return None
    coordinates = np.column_stack(np.nonzero(body))[:, ::-1].astype(np.float64)
    center = coordinates.mean(axis=0)
    eigenvalues, vectors = np.linalg.eigh(np.cov(coordinates, rowvar=False, bias=True))
    if eigenvalues[0] <= 0 or eigenvalues[1] / eigenvalues[0] < 2:
        return None
    axis = vectors[:, 1]
    perpendicular = np.array([-axis[1], axis[0]])
    axial = (coordinates - center) @ axis
    transverse = (coordinates - center) @ perpendicular
    lower, upper = np.percentile(axial, [1, 99])
    span = upper - lower
    if span <= 0:
        return None
    normalized = (axial - lower) / span
    def end_width(low, high):
        group = transverse[(normalized >= low) & (normalized <= high)]
        return float(np.percentile(group, 95) - np.percentile(group, 5)) if len(group) >= 8 else 0.
    first, last = end_width(.02, .22), end_width(.78, .98)
    if min(first, last) <= 0:
        return None
    if first < last:
        axis = -axis
        perpendicular = -perpendicular
        lower, upper = -upper, -lower
        first, last = last, first
    yy, xx = np.mgrid[:256, :256]
    grid = np.stack((xx - center[0], yy - center[1]), axis=-1)
    along = (grid @ axis - lower) / span
    across = (grid @ perpendicular) / max(first / 2, 1)
    response = cv2.resize(values.astype(np.float32), (256, 256), interpolation=cv2.INTER_LINEAR)
    support = response > geometry["threshold"]
    if int(support.sum()) < 4:
        return None
    peak = response == response.max()
    weights = np.maximum(response[support] - geometry["threshold"], 1e-8)
    weights = weights / weights.sum()
    axial_mean = float(np.sum(along[support] * weights))
    transverse_mean = float(np.sum(np.abs(across[support]) * weights))
    bands = (along < .2, (along >= .2) & (along < .5),
             (along >= .5) & (along < .9), along >= .9)
    fractions = [float(np.sum(weights[band[support]])) for band in bands]
    textures = [float(np.std(gray[body & band])) if np.any(body & band) else 0. for band in bands[:3]]
    background = float(np.median(gray[~body]))
    features = dict(zip(SCREW_FEATURES, (
        float(np.sqrt(eigenvalues[1] / eigenvalues[0])), first / last,
        float(support.sum()) / area, axial_mean,
        float(np.sqrt(np.sum((along[support] - axial_mean) ** 2 * weights))),
        transverse_mean, float(np.sqrt(np.sum((np.abs(across[support]) - transverse_mean) ** 2 * weights))),
        float(np.mean(along[peak])), float(np.mean(np.abs(across[peak]))), *fractions, *textures,
        background - float(np.mean(gray[support])), float(response.max()),
    )))
    return features if all(_finite(v) for v in features.values()) else None
