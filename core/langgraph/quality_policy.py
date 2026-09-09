"""Content-bound quality evidence; author exceptions never override strict gates."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal, cast

import structlog
from pydantic import BaseModel, ConfigDict, Field

import config
from core.langgraph.content_manager import require_project_dir
from core.langgraph.state import NarrativeState
from utils.file_io import ContainedFiles

SCORE_FIELDS = ("coherence_score", "prose_quality_score", "plot_advancement_score", "pacing_score", "tone_consistency_score")


class QualityPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    schema_version: Literal[1] = 1
    identity: Literal["author", "strict"]
    graph_quality: Literal["advisory", "mandatory"]
    minimum_score: float = Field(ge=0, le=1, allow_inf_nan=False)
    consistency_enabled: bool
    graph_quality_enabled: bool
    graph_quality_frequency: int = Field(gt=0)
    contradictory_traits_enabled: bool


class CheckEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    source: dict[str, Any]
    status: Literal["completed", "failed", "skipped"]
    reason: str
    details: dict[str, Any]


def configured_policy() -> dict[str, Any]:
    return QualityPolicy(
        identity=config.settings.QUALITY_ACCEPTANCE_POLICY,
        graph_quality=config.settings.QA_ACCEPTANCE_POLICY,
        minimum_score=config.settings.MIN_QUALITY_THRESHOLD,
        consistency_enabled=config.settings.validation.ENABLE_VALIDATION,
        graph_quality_enabled=config.settings.ENABLE_QA_CHECKS,
        graph_quality_frequency=config.settings.QA_CHECK_FREQUENCY,
        contradictory_traits_enabled=config.settings.QA_CHECK_CONTRADICTORY_TRAITS,
    ).model_dump()


def policy_for(state: NarrativeState) -> QualityPolicy:
    if not state.get("quality_policy"):
        raise ValueError("Quality policy missing; legacy checkpoint requires explicit revalidation")
    return QualityPolicy.model_validate(state["quality_policy"])


def quality_source(state: NarrativeState) -> dict[str, Any]:
    return {
        "project_id": state.get("graph_project_id", state.get("project_id")),
        "chapter_number": state.get("current_chapter"),
        "iteration_count": state.get("iteration_count", 0),
        "artifacts": {name: cast(dict[str, Any], state.get(name) or {}).get("checksum") for name in ("draft_ref", "scene_drafts_ref", "extracted_entities_ref", "extracted_relationships_ref")},
    }


def record_check(state: NarrativeState, name: str, status: Literal["completed", "failed", "skipped"], reason: str = "", details: dict[str, Any] | None = None) -> dict[str, Any]:
    evidence = CheckEvidence(source=quality_source(state), status=status, reason=reason, details={} if details is None else details)
    return {**state.get("quality_checks", {}), name: evidence.model_dump()}


def validation_decision(state: NarrativeState) -> dict[str, Any]:
    """Require observed checks, even for force-continue; never infer them from scores."""
    policy = policy_for(state)
    if state.get("has_fatal_error") or state.get("revision_rollback_failure") is not None:
        raise ValueError("Quality acceptance cannot bypass a fatal or compensation barrier")
    source = quality_source(state)
    if not source["artifacts"]["draft_ref"]:
        raise ValueError("Quality acceptance requires a draft identity")
    checks = {name: CheckEvidence.model_validate(value) for name, value in state.get("quality_checks", {}).items()}
    exceptions: list[str] = []
    force = state.get("force_continue", False)
    iteration = state.get("iteration_count", 0)
    maximum = state.get("max_iterations", 3)
    if type(force) is not bool or type(iteration) is not int or type(maximum) is not int or iteration < 0 or maximum < 0:
        raise ValueError("Invalid quality revision policy controls")
    for name in ("consistency", "evaluation", "contradictions"):
        if name not in checks or checks[name].source != source:
            raise ValueError(f"Quality validation evidence missing or stale: {name}")
        check = checks[name]
        if check.status != "completed":
            if name == "consistency" and check.status == "skipped" and not policy.consistency_enabled and policy.identity == "author":
                exceptions.append("consistency_disabled")
            elif name == "evaluation" and check.status == "failed" and force and policy.identity == "author":
                exceptions.append("force_continue_incomplete_evaluation")
            else:
                raise ValueError(f"Mandatory quality validation incomplete: {name}: {check.reason}")
    scores: dict[str, Any] = {name: state.get(name) for name in SCORE_FIELDS}
    if checks["evaluation"].status == "completed":
        if any(type(value) not in (int, float) or not 0 <= value <= 1 for value in scores.values()):
            raise ValueError("Quality evaluation scores incomplete or invalid")
        if checks["evaluation"].details.get("scores") != scores:
            raise ValueError("Quality scores differ from evaluation evidence")
    findings = [item.model_dump(mode="json") for item in state.get("contradictions", [])]
    if checks["contradictions"].details.get("findings") != findings:
        raise ValueError("Quality findings differ from completed validation evidence")
    has_issues = any(item["severity"] in {"critical", "major"} for item in findings)
    if checks["evaluation"].status == "completed":
        has_issues = has_issues or sum(scores[name] for name in SCORE_FIELDS[:3]) / 3 < policy.minimum_score
    if has_issues:
        if policy.identity == "author" and force:
            exceptions.append("force_continue_findings")
        elif policy.identity == "author" and iteration >= maximum:
            exceptions.append("max_revisions_findings")
        else:
            raise ValueError("Mandatory quality gate failed: unresolved findings")
    return {"schema_version": 1, "policy": policy.model_dump(), "source": source, "checks": {name: value.model_dump() for name, value in checks.items()},
            "scores": scores, "feedback": state.get("quality_feedback"), "findings": findings, "force_continue": force,
            "iteration_count": iteration, "max_iterations": maximum, "exceptions": exceptions,
            "status": "accepted_with_exceptions" if exceptions else "accepted"}


def acceptance_decision(state: NarrativeState) -> dict[str, Any]:
    decision = validation_decision(state)
    policy = policy_for(state)
    check = CheckEvidence.model_validate(state.get("graph_quality_check"))
    if check.source != quality_source(state):
        raise ValueError("Graph quality evidence stale")
    decision["graph_quality_check"] = check.model_dump()
    if check.status != "completed" or check.details.get("issues_found", 0):
        if policy.graph_quality == "mandatory":
            raise ValueError(f"Mandatory graph quality gate failed: {check.reason}")
        decision["exceptions"].append(f"advisory_graph_quality:{check.reason or 'findings'}")
    decision["status"] = "accepted_with_exceptions" if decision["exceptions"] else "accepted"
    return decision


def announce_acceptance(decision: dict[str, Any], receipt_path: str) -> None:
    """Show exceptions only after publication has been acknowledged and verified."""
    if decision["exceptions"]:
        structlog.get_logger(__name__).warning("Chapter accepted_with_exceptions", policy=decision["policy"]["identity"], exceptions=decision["exceptions"], receipt=receipt_path)


def verify_decision(decision: dict[str, Any], state: NarrativeState) -> None:
    """Replay retained evidence under its recorded policy, not today's configuration."""
    from core.langgraph.state import Contradiction

    retained: NarrativeState = {
        **state, "quality_policy": decision["policy"], "quality_checks": decision["checks"],
        "graph_quality_check": decision["graph_quality_check"], "force_continue": decision["force_continue"],
        "max_iterations": decision["max_iterations"], "quality_feedback": decision["feedback"],
        "contradictions": [Contradiction.model_validate(item) for item in decision["findings"]], **decision["scores"],
        "has_fatal_error": False,
    }
    if decision != acceptance_decision(retained):
        raise ValueError("Retained quality decision mismatch")
    if state.get("quality_policy") != decision["policy"]:
        raise ValueError("Checkpoint/acceptance quality policy mismatch")


def verify_checkpoint_quality(state: NarrativeState, decision: dict[str, Any]) -> None:
    if state.get("quality_policy") != decision["policy"]:
        raise ValueError("Checkpoint/acceptance quality policy mismatch")
    if state.get("has_fatal_error") and state.get("error_node") != "finalize":
        raise ValueError("Mandatory quality acceptance cannot bypass a fatal checkpoint")
    for name, evidence in state.get("quality_checks", {}).items():
        if evidence != decision["checks"].get(name):
            raise ValueError("Checkpoint/acceptance quality evidence mismatch")
    if "evaluation" in state.get("quality_checks", {}) and {name: state.get(name) for name in SCORE_FIELDS} != decision["scores"]:
        raise ValueError("Checkpoint/acceptance quality scores mismatch")
    if "contradictions" in state.get("quality_checks", {}) and [item.model_dump(mode="json") for item in state.get("contradictions", [])] != decision["findings"]:
        raise ValueError("Checkpoint/acceptance quality findings mismatch")


def retain_maintenance(state: NarrativeState, operation: str, outcome: dict[str, Any]) -> None:
    """Append content-addressed advisory observations without rewriting acceptance."""
    if operation not in {"healing", "graph_quality"}:
        raise ValueError("Unknown maintenance operation")
    from core.langgraph.chapter_lifecycle import ChapterLifecycle
    from core.langgraph.manuscript import ManuscriptStore

    if "lifecycle_version" in state:
        lifecycle = ChapterLifecycle(state).stage()
        if not lifecycle.files.exists(lifecycle.phase_path("published")):
            raise ValueError("Advisory maintenance requires a published attempt")
        identity: dict[str, Any] = {"attempt_id": lifecycle.manifest.attempt_id}
        directory = f".saga/attempts/{lifecycle.manifest.attempt_id}/maintenance"
    elif state.get("quality_policy") and state.get("draft_ref"):
        manuscript = ManuscriptStore(Path(require_project_dir(state))).accepted(state["current_chapter"])
        if manuscript is None or manuscript.body_sha256 != quality_source(state)["artifacts"]["draft_ref"]:
            raise ValueError("Advisory maintenance requires the accepted manuscript")
        identity = {"manuscript": manuscript.model_dump()}
        directory = manuscript.artifact_path.removesuffix(".md") + ".maintenance"
    else:
        # Standalone legacy diagnostics have no publication identity; they cannot
        # certify acceptance. All publication entrypoints now require a policy.
        return
    receipt = {"schema_version": 1, **identity, "policy": "advisory", "operation": operation, "outcome": outcome,
               "status": "accepted_with_exceptions" if outcome.get("errors") or outcome.get("warnings") or outcome.get("issues_found") else "completed"}
    encoded = json.dumps(receipt, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    checksum = hashlib.sha256(encoded).hexdigest()
    files = ContainedFiles(Path(require_project_dir(state)), durable=True)
    path = f"{directory}/{operation}-{checksum}.json"
    if not files.exists(path):
        files.write_bytes(path, encoded)
    if files.read_bytes(path) != encoded:
        raise ValueError("Maintenance receipt readback mismatch")
