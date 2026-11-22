"""Evidence and reviewed check choices for the benchmark vision advisory."""

import math

from forgesight.defect_severity import LEVELS, POLICY_ID, ZONES, assess_severity
from forgesight.defect_type import BENCHMARK_MODES, FOUR_TYPES


VISION_GUIDE_ID = "G3"
VISION_PROMPT_VERSION = "forgeassist-vision-choice-v1"
VISION_OUTPUT_POLICY = "forgeassist-reviewed-vision-choice-v1"
TYPE_LABELS = {"bent": "possible bent nut", "flip": "possible flipped nut",
               "scratch": "possible scratch", "discoloration": "possible discoloration",
               "unknown": "unknown"}
CHECKS = {
    "bent": ("Compare the highlighted edge with the outline of a good nut.",
             "View the nut from the side to check for distortion."),
    "flip": ("Compare the visible face with a correctly oriented reference.",
             "Check which side is up before taking another image."),
    "scratch": ("Check the highlighted line under angled light.",
                "Compare the highlighted surface with a good nut."),
    "discoloration": ("Compare the highlighted color with a good nut under the same lighting.",
                      "Change the light angle and check whether the color patch persists."),
    "unknown": ("Inspect the highlighted area under even lighting.",
                "Take a sharp image with the entire nut in view."),
}


def vision_context(record):
    source = record["source"]
    assessment = record.get("assessment")
    if (source.get("mode") not in BENCHMARK_MODES or source.get("physical_part") is not False
            or source.get("size_px") != [700, 700] or not isinstance(assessment, dict)
            or assessment.get("method") != "four_type_forest" or "severity" not in assessment):
        return None
    defect_type, severity = assessment["defect_type"], assessment["severity"]
    if (assessment.get("schema_version") != 1 or assessment.get("scope") != "PROVISIONAL_BENCHMARK_IMAGE"
            or defect_type not in (*FOUR_TYPES, "unknown") or severity["policy_id"] != POLICY_ID):
        raise ValueError("Unsupported vision assessment.")

    geometry = record["geometry"]
    region = geometry["regions"][0]
    extent = severity.get("evidence")
    if severity["level"] in LEVELS:
        reference = extent["reference_size"] if isinstance(extent, dict) else None
        if (type(reference) not in (int, float) or not math.isfinite(reference) or reference <= 0
                or assess_severity(defect_type, geometry, (reference, reference)) != severity):
            raise ValueError("Severity measurements do not match the policy and response region.")
    elif (severity["level"] != "UNASSESSED" or extent is not None
          or severity["reason"] not in {"UNKNOWN_DEFECT_TYPE", "ORIENTATION_ISSUE",
                                       "LOCALIZATION_UNAVAILABLE", "LOCALIZATION_UNCERTAIN",
                                       "RESPONSE_EXTENT_UNCERTAIN"}
          or (defect_type == "unknown") != (severity["reason"] == "UNKNOWN_DEFECT_TYPE")
          or (defect_type == "flip") != (severity["reason"] == "ORIENTATION_ISSUE")):
        raise ValueError("Unsupported unassessed severity reason.")

    location = region.get("approximate_nut_area", "")
    zones = location.split(" / ")
    if not set(zones) <= ZONES:
        location = "uncertain"
    else:
        location = location.replace("outer perimeter or nearby background", "outer edge/background")
    if defect_type == "flip":
        location = "whole nut orientation"

    actions = CHECKS[defect_type]
    if defect_type in ("scratch", "bent") and "inner rim" in zones and location != "uncertain":
        actions = ("Check the thread opening beside the highlighted area.", actions[0])

    if extent:
        unit = "nut diameter" if defect_type == "scratch" else "nut outline area"
        measurement = f"Response extent {extent['ratio']:.1%} of {unit}."
    else:
        measurement = f"Response region {region['area_px']} pixels."
    facts = {"type": defect_type, "severity": severity["level"], "severity_reason": severity["reason"],
             "location": location, "response_region_area_px": region["area_px"], "extent": extent}
    prompt = (f"Image evidence: {TYPE_LABELS[defect_type]}. Severity {severity['level'].lower()}. "
              f"Location {location}. {measurement}\nA. {actions[0]}\nB. {actions[1]}\n"
              "Which is the best first check for this image? Answer with A or B only.")
    return facts, dict(zip(("A", "B"), actions)), prompt


def vision_note(choice, actions):
    return f"{actions[choice]} [E1][{VISION_GUIDE_ID}]"
