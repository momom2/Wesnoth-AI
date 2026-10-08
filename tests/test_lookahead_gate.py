"""The look-ahead gate's path (scripts/lookahead_gate_box.sh): a look-ahead
side through the match driver's persistent workers and shared inference
server, its telemetry pooled per match, and a critic arm's checkpoint
fetched and checked (tools/lookahead_gate.py)."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.lookahead_gate import ensure, summarize  # noqa: E402
from wesnoth_ai import game_core as gc  # noqa: E402
from wesnoth_ai.lookahead_config import config_record, load_config  # noqa: E402

GATE_CONFIG = Path(__file__).parent.parent / "configs" / "lookahead_material_gate.json"


@pytest.mark.skipif(gc.game_core_class() is None, reason="wesnoth_core.GameCore not available")
def test_a_look_ahead_side_plays_through_the_workers_and_the_shared_server(tmp_path):
    """Two 2-turn games of the gate's configuration on side A against the
    raw player, through two persistent workers and one inference server:
    every result records the configuration and the operator's telemetry,
    the operator expanded on the workers' simulators, and the match's
    summary pools what the games recorded."""
    from test_eval_inference_server import _tiny_checkpoint
    from tools.run_elo_batch import EXIT_GUARD_SPENT, main
    spec = _tiny_checkpoint(tmp_path / "tiny.pt")
    out = tmp_path / "games"
    rc = main(["x", "--label-a", "A", "--spec-a", spec, "--label-b", "B", "--spec-b", spec,
               "--outdir", str(out), "--games", "2", "--mcts-sims", "0",
               "--raw-temperature-a", "0", "--raw-temperature-b", "0",
               "--lookahead-a", str(GATE_CONFIG), "--max-turns", "2", "--device", "cpu",
               "--jobs", "2", "--persistent-workers", "--shared-inference",
               "--max-extra-games", "0", "--time-budget-min", "5", "--min-free-mb", "0"])
    assert rc == EXIT_GUARD_SPENT, "2 turns decide nothing and no replacement is allowed"
    record = config_record(load_config(GATE_CONFIG))
    results = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(out.glob("game_*.json"))]
    assert len(results) == 2
    for r in results:
        assert r["procedure_a"] == "la:material:k8c1s0.1" and r["procedure_b"] == "raw:t0"
        assert r["lookahead_a"] == record and r["lookahead_b"] is None
        assert r["shared_inference"] is True
        tel = r["lookahead_telemetry_a"]
        assert tel["operated"] > 0 and tel["states"] > 0, "the operator never expanded a candidate"
        assert tel["failed"] == {}, tel["failed"]
        assert r["lookahead_telemetry_b"] is None
    summary = summarize(out)
    assert summary["games"] == 2 and summary["games_without_telemetry"] == 0
    assert summary["lookahead"] == record and summary["procedure"] == "la:material:k8c1s0.1"
    assert summary["decisions"] == sum(r["lookahead_telemetry_a"]["decisions"] for r in results)
    assert 0 < summary["operated_share"] <= 1 and summary["states_per_operated"] > 0


def _game(path: Path, outcome: str, telemetry=None) -> None:
    path.write_text(json.dumps({"outcome_a": outcome, "procedure_a": "la:x", "lookahead_a": {"k": 8},
                                "lookahead_telemetry_a": telemetry}), encoding="utf-8")


def _telemetry(decisions: int, attack_flips: int, failed=None) -> dict:
    kinds = dict.fromkeys(("attack", "move", "recruit", "end_turn"), 0)
    return {"decisions": decisions, "operated": decisions, "states": 3 * decisions, "states_max": 5,
            "terminal_states": 0, "forwards": decisions, "seconds": 0.1 * decisions,
            "seconds_prior": 0.0, "seconds_expand": 0.0, "seconds_evaluate": 0.0, "seconds_max": 0.2,
            "by_kind": {**kinds, "attack": decisions}, "flips_by_kind": {**kinds, "attack": attack_flips},
            "flips_to_kind": {**kinds, "move": attack_flips}, "candidates_by_kind": {**kinds, "attack": decisions},
            "failed": failed or {}}


def test_rates_are_pooled_over_decisions_not_averaged_over_games(tmp_path):
    """A game of 10 attacks with 5 flips and one of 90 without: the flip
    rate is 5 of 100, not the games' mean of 0.25; failures add up by
    reason; a game without telemetry is counted apart."""
    _game(tmp_path / "game_A_B_s1_1.json", "win", _telemetry(10, 5, {"attack_leaves": 1}))
    _game(tmp_path / "game_A_B_s2_2.json", "loss", _telemetry(90, 0, {"attack_leaves": 2, "strike_record": 1}))
    _game(tmp_path / "game_A_B_s1_3.json", "timeout")
    s = summarize(tmp_path)
    assert (s["games"], s["games_without_telemetry"], s["decisions"]) == (2, 1, 100)
    assert s["flip_rate"] == s["flip_rate_by_kind"]["attack"] == pytest.approx(0.05)
    assert s["flip_rate_by_kind"]["move"] is None, "no decision had a move as its argmax"
    assert s["flips_to_kind"]["move"] == 5
    assert s["failed"] == {"attack_leaves": 3, "strike_record": 1} and s["games_with_failures"] == 2
    assert s["outcomes_a"] == {"loss": 1, "timeout": 1, "win": 1}
    assert s["seconds_per_decision"]["total"] == pytest.approx(0.1)


def test_a_critic_arm_takes_only_the_checkpoint_its_configuration_records(tmp_path, monkeypatch):
    """The critic is fetched from the path the configuration names and kept
    only when its SHA-256 is the recorded one; the material arm fetches
    nothing."""
    monkeypatch.chdir(tmp_path)
    host = tmp_path / "host"
    host.mkdir()
    (host / "critic.pt").write_bytes(b"the critic")
    (host / "other.pt").write_bytes(b"another critic")
    fetched = []

    def download(repo, path):
        fetched.append((repo, path))
        return str(host / path)

    def config(name: str, hf_path: str, sha: str) -> Path:
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({"k": 8, "c": 1.0, "sigma": 0.1, "evaluator": {
            "name": "critic", "view": "obs", "checkpoint": f"ckpt/{name}.pt", "checkpoint_hf": hf_path,
            "checkpoint_sha256": sha}}), encoding="utf-8")
        return path

    sha = hashlib.sha256(b"the critic").hexdigest()
    assert ensure(config("good", "critic.pt", sha), -1.5, "repo", download) == "la:critic.obs:k8c1s0.1+eo-1.5"
    assert (tmp_path / "ckpt" / "good.pt").read_bytes() == b"the critic"
    assert fetched == [("repo", "critic.pt")]
    with pytest.raises(ValueError, match="not kept"):
        ensure(config("bad", "other.pt", sha), -1.5, "repo", download)
    assert not list((tmp_path / "ckpt").glob("bad.pt*"))
    with pytest.raises(ValueError, match="checkpoint_hf"):
        ensure(config("unnamed", "", sha), -1.5, "repo", download)
    assert ensure(GATE_CONFIG, -1.5, "repo", download) == "la:material:k8c1s0.1+eo-1.5"
    assert len(fetched) == 2, "the material arm fetches nothing"
