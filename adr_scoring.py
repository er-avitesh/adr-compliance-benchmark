"""Exact rubric arithmetic shared by collection and offline analysis."""
from fractions import Fraction
import json
import hashlib

SCORING_VERSION = "exact-rational-v1"
SC_WEIGHTS = {f"C{i}": Fraction(w, 100) for i, w in enumerate((10, 15, 15, 15, 20, 15, 10), 1)}
DQ_WEIGHTS = {f"Q{i}": Fraction(w, 100) for i, w in enumerate((10, 15, 15, 10, 25, 15, 10), 1)}


def exact(value):
    """Interpret decimal configuration values without binary float artifacts."""
    return value if isinstance(value, Fraction) else Fraction(str(value))


def structural_score(checks, weights=None):
    weights = SC_WEIGHTS if weights is None else weights
    total = Fraction(0)
    for key, weight in weights.items():
        value = checks[key]
        if value not in (False, True, 0, 1):
            raise ValueError(f"{key} must be a binary criterion value")
        total += exact(weight) * int(value)
    return 100 * total


def quality_score(scores, weights=None):
    weights = DQ_WEIGHTS if weights is None else weights
    total = Fraction(0)
    for key, weight in weights.items():
        value = exact(scores[key])
        if value.denominator != 1 or not 0 <= value <= 3:
            raise ValueError(f"{key} must be an integer from 0 to 3")
        total += exact(weight) * value
    return 100 * total / 3


def composite_score(sc, dq, sc_weight=Fraction(2, 5)):
    weight = exact(sc_weight)
    return weight * exact(sc) + (1 - weight) * exact(dq)


def dimension_class(value):
    value = exact(value)
    if value >= 85:
        return "Fully_Compliant"
    if value >= 70:
        return "Mostly_Compliant"
    if value >= 45:
        return "Partially_Compliant"
    return "Not_Compliant"


def overall_class(sc, dq, sc_weight=Fraction(2, 5), structural_floor=40):
    sc, dq = exact(sc), exact(dq)
    cs = composite_score(sc, dq, sc_weight)
    if cs < 45 or sc < exact(structural_floor):
        return "Not_Compliant"
    if cs >= 85 and sc >= 80 and dq >= 80:
        return "Fully_Compliant"
    if cs >= 70 and sc >= 60 and dq >= 60:
        return "Mostly_Compliant"
    return "Partially_Compliant"


def evaluate_criteria(checks, scores=None, *, sc_weights=None, dq_weights=None,
                      sc_weight=Fraction(2, 5), structural_floor=40):
    """Classify exact scores before converting numeric outputs for JSON storage."""
    scores = checks if scores is None else scores
    sc = structural_score(checks, sc_weights)
    dq = quality_score(scores, dq_weights)
    cs = composite_score(sc, dq, sc_weight)
    return {
        "scoring_version": SCORING_VERSION,
        "sc_score": float(sc), "dq_score": float(dq), "composite_score": float(cs),
        "sc_class": dimension_class(sc), "dq_class": dimension_class(dq),
        "overall": overall_class(sc, dq, sc_weight, structural_floor),
        "exact_scores": {"sc": str(sc), "dq": str(dq), "composite": str(cs)},
    }


def prediction_fields(checks, scores):
    """The derived fields that must agree across collection and offline scoring."""
    result = evaluate_criteria(checks, scores)
    fields = {
        "scoring_version": SCORING_VERSION,
        "predicted_exact_scores": result["exact_scores"],
        "predicted": result["overall"],
        "predicted_overall_from_criteria": result["overall"],
    }
    for dimension in ("sc", "dq"):
        fields[f"predicted_{dimension}_class"] = result[f"{dimension}_class"]
    for key in ("sc_score", "dq_score", "composite_score"):
        fields[f"predicted_{key}"] = result[key]
        fields[f"predicted_{key}_from_criteria"] = result[key]
    return fields


def require_exact_run(run_dir):
    """Reject archived or incomplete runs before writing derived analysis."""
    manifest = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    if manifest.get("scoring_version") != SCORING_VERSION:
        raise ValueError("This run uses archived scoring. Rescore into a new run ID first.")
    if manifest.get("analysis_only") and not manifest.get("rescore_complete"):
        raise ValueError("Offline rescoring is incomplete; analysis is blocked.")
    for filename, key in (("eval_set.json", "evaluation_set_sha256"),
                          ("human_ground_truth.json", "ground_truth_sha256")):
        with (run_dir.parents[1] / filename).open("rb") as handle:
            sha = hashlib.file_digest(handle, "sha256").hexdigest()
        if sha != manifest.get(key):
            raise ValueError(f"Frozen analysis input changed: {filename}")
    return manifest
