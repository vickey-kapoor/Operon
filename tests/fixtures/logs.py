"""Shared log artifact path builders."""

from __future__ import annotations

from operon.models.logs import ModelDebugArtifacts


def model_debug_artifacts(run_id: str = "run-1", step: int = 1, stage: str = "policy") -> ModelDebugArtifacts:
    parsed_name = "policy_decision.json" if stage == "policy" else f"{stage}_parsed.json"
    return ModelDebugArtifacts(
        prompt_artifact_path=f"runs/{run_id}/step_{step}/{stage}_prompt.txt",
        raw_response_artifact_path=f"runs/{run_id}/step_{step}/{stage}_raw.txt",
        parsed_artifact_path=f"runs/{run_id}/step_{step}/{parsed_name}",
    )

