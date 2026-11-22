"""Versioned geometry rules for provisional severity on MVTec nut images."""

import math


POLICY_ID = "mvtec-metal-nut-severity-v1"
LEVELS = ("LOW", "MEDIUM", "HIGH")
# Cutoffs describe response extent in benchmark images.
RULES = {
    "scratch": ("response_span_over_nut_diameter", 0.15, 0.35),
    "discoloration": ("response_area_over_nut_outline_area", 0.02, 0.08),
    "bent": ("response_area_over_nut_outline_area", 0.05, 0.15),
}
ZONES = {"central opening", "inner rim", "annular face", "outer perimeter or nearby background"}


def assess_severity(defect_type, geometry, outline):
    """Grade a localized anomaly response using its type, extent and approximate zone."""
    result = {"policy_id": POLICY_ID, "level": "UNASSESSED"}
    if defect_type == "flip":
        return dict(result, reason="ORIENTATION_ISSUE")
    if defect_type not in RULES:
        return dict(result, reason="UNKNOWN_DEFECT_TYPE")
    if outline is None or not geometry["regions"]:
        return dict(result, reason="LOCALIZATION_UNAVAILABLE")

    region = geometry["regions"][0]
    location = region.get("approximate_nut_area")
    zones = set(location.split(" / ")) if location else set()
    if not zones or not zones <= ZONES or zones == {"central opening"}:
        return dict(result, reason="LOCALIZATION_UNCERTAIN")

    metric, medium_at, high_at = RULES[defect_type]
    nut_diameter_px, nut_outline_area_px = outline
    if defect_type == "scratch":
        response_size, reference_size = region["bbox_diagonal_px"], nut_diameter_px
    else:
        response_size, reference_size = region["area_px"], nut_outline_area_px
    ratio = response_size / reference_size if reference_size > 0 else 0
    if not math.isfinite(ratio) or not 0 < ratio <= 1:
        return dict(result, reason="RESPONSE_EXTENT_UNCERTAIN")

    size_band = int(ratio >= medium_at) + int(ratio >= high_at)
    rim_escalation = defect_type in ("scratch", "bent") and "inner rim" in zones and size_band < 2
    return dict(result, level=LEVELS[size_band + rim_escalation], reason="RESPONSE_GEOMETRY",
                evidence={"region_id": region["region_id"], "location": location, "metric": metric,
                          "response_size": response_size, "reference_size": reference_size,
                          "ratio": ratio, "medium_at": medium_at, "high_at": high_at,
                          "size_band": LEVELS[size_band], "rim_escalation": rim_escalation})
