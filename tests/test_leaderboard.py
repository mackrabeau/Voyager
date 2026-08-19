from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from voyager.benchmark.eval_cli import _create_policy
from voyager.benchmark.leaderboard import (
    load_submissions,
    render_generated_tables,
    update_readme,
    validate_submission,
)
from voyager.training.island_evaluation import LegalRandomIslandPolicy

ROOT = Path(__file__).resolve().parents[1]
LEADERBOARD = ROOT / "leaderboard"


def test_official_leaderboard_submissions_validate_and_readme_is_current() -> None:
    submissions = load_submissions(
        LEADERBOARD / "submissions",
        LEADERBOARD / "schema.json",
    )

    assert {submission["submission_id"] for submission in submissions} == {
        "feed-forward-ppo",
        "legal-random",
        "recurrent-ppo",
        "scripted-oracle",
    }
    readme = (LEADERBOARD / "README.md").read_text(encoding="utf-8")
    assert update_readme(readme, render_generated_tables(submissions)) == readme


def test_validator_rejects_a_score_that_does_not_match_episode_records() -> None:
    schema = json.loads((LEADERBOARD / "schema.json").read_text(encoding="utf-8"))
    submission = json.loads(
        (LEADERBOARD / "submissions/legal-random.json").read_text(encoding="utf-8")
    )
    tampered = copy.deepcopy(submission)
    tampered["results"]["aggregate_score"]["value"] += 0.01

    with pytest.raises(ValueError, match="does not recompute"):
        validate_submission(tampered, schema)


def test_eval_factory_supports_a_seeded_single_policy() -> None:
    policy = _create_policy(
        LegalRandomIslandPolicy,
        training_seed=None,
        evaluation_seed=123,
    )

    assert isinstance(policy, LegalRandomIslandPolicy)
