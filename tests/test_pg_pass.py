"""The gradient pass (tools/pg_pass.py, tools/pg_schedule.py): the probe
gradients are the logs' sums (P1), the splits add to the total, the
accumulated windows are one big batch's gradient, and a shadow row reads
the memory from before it and writes nothing."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from wesnoth_ai import game_core as gc  # noqa: E402
from wesnoth_ai.pg_records import decisions_of, read_labels  # noqa: E402
from wesnoth_ai.sequence_streams import GameSide, Step  # noqa: E402
from wesnoth_ai.server_priors import KIND_ATTACK, KIND_END_TURN, KIND_RECRUIT  # noqa: E402

needs_core = pytest.mark.skipif(gc.game_core_class() is None, reason="wesnoth_core.GameCore not available")
CPU = torch.device("cpu")
# By the suite's names (helpers.pg_games.SUITE); `named` turns them into the stems.
SPLITS = {"g0": 0, "g1": 3, "g2": 5}
R = {("g0", 1): 0.7, ("g0", 2): -0.4, ("g1", 1): -0.9, ("g1", 2): 0.3, ("g2", 1): 0.5, ("g2", 2): -1.1}


def _labels(tmp_path, splits=SPLITS):
    from helpers.pg_games import named, played_games, write_labels
    games = played_games()
    return games, read_labels(write_labels(tmp_path / "labels.jsonl", named(games, splits), named(games, R)))


def _run(tmp_path, splits=SPLITS, records=None, **config):
    from tools.pg_pass import GradientPass, PassConfig
    games, labels = _labels(tmp_path, splits)
    return games, GradientPass(games.checkpoint, records or games.records, labels, CPU, PassConfig(**config)).run()


@needs_core
def test_the_corrections_weight_each_decision_and_the_pass_is_linear_in_them(tmp_path):
    """With corrections (module docstring) decision t weighs R - alpha v_t -
    gamma beta^T x_t. The sum is linear in the weights: the corrected pass is
    the plain pass less alpha times the pass weighing v_t and gamma times
    the one weighing beta^T x_t, both run on R = 0 labels with coefficient
    -1. Zero corrections are the plain pass, bit for bit."""
    from helpers.pg_games import named, played_games, write_labels
    from tools.match_luck import COLUMNS
    from tools.pg_pass import GradientPass, PassConfig
    games = played_games()
    beta = tuple(0.1 * (k + 1) for k in range(len(COLUMNS)))

    def total(name, R_of, **corrections):
        labels = read_labels(write_labels(tmp_path / f"{name}.jsonl", named(games, SPLITS), named(games, R_of)))
        return GradientPass(games.checkpoint, games.records, labels, CPU,
                            PassConfig(streams=2, floor_dtype="off", **corrections)).run()["total"]
    zero_R = {k: 0.0 for k in R}
    plain = total("plain", R)
    assert all(torch.equal(t, plain[n]) for n, t in total("zero", R, alpha=0.0, gamma=0.0).items())
    alpha, gamma = 0.6, -0.9
    corrected = total("corrected", R, alpha=alpha, gamma=gamma, luck_beta=beta)
    by_v = total("v", zero_R, alpha=-1.0)
    by_luck = total("luck", zero_R, gamma=-1.0, luck_beta=beta)
    num = sum(float(((corrected[n] - (plain[n] - alpha * by_v[n] - gamma * by_luck[n])) ** 2).sum()) for n in plain)
    den = sum(float((corrected[n] ** 2).sum()) for n in plain)
    assert den > 0 and num <= 1e-8 * den
    assert sum(float((by_luck[n] ** 2).sum()) for n in plain) > 0        # the luck term reached the sum


def _log_sums(games, R_of):
    """The logs' sum over every decision of R (1[c] - p(c)), by class: for
    attacks p(attack | the sampled actor), no term where its actor has
    no attack."""
    from tools.pg_positions import read_record
    sums = {"end_turn": 0.0, "attack": 0.0, "recruit": 0.0}
    for game, path in games.records.items():
        for d in decisions_of(read_record(path)):
            r = R_of[(game, d.side)]
            sums["end_turn"] += r * ((d.kind == KIND_END_TURN) - d.p_end)
            sums["recruit"] += r * ((d.kind == KIND_RECRUIT) - d.p_rec)
            if not math.isnan(d.p_att_actor):
                sums["attack"] += r * ((d.kind == KIND_ATTACK) - d.p_att_actor)
    return {c: v / (2 * len(games.records)) for c, v in sums.items()}


@needs_core
def test_the_probe_gradients_are_the_logs_sums_and_fail_on_a_shifted_label(tmp_path):
    """P1 end to end: the trainer's gradients at the three probe scalars
    equal the logs' (1/2N) sum R (1[c] - p(c)); with R moved by one
    game-side they do not. The three games share one split on one stream,
    so the one slot takes all six game-sides in turn: a join of R on the
    slot (a seat) in place of the game-side's side gives both sides of a
    game one R, which misses. The trainer's log pi of every logged action
    is the logged one (the replay identity, fp32)."""
    from helpers.pg_games import named
    from tools.pg_positions import read_record
    games, result = _run(tmp_path, splits={g: 0 for g in SPLITS}, streams=1, window=4)
    labels = named(games, R)
    want = _log_sums(games, labels)
    sides = sorted(labels)
    shifted = _log_sums(games, {sides[i]: labels[sides[(i + 1) % len(sides)]] for i in range(len(sides))})
    tol = 2e-5
    for c in want:
        assert result["probes"][c] == pytest.approx(want[c], abs=tol), c
    assert max(abs(result["probes"][c] - shifted[c]) for c in want) > 50 * tol
    assert abs(want["attack"]) > 50 * tol and abs(want["recruit"]) > 50 * tol
    assert result["replay"]["max_abs"] < 1e-4 and abs(result["replay"]["kl"]) < 1e-6
    assert result["shadow"] == 2
    assert result["decisions"] == sum(len(decisions_of(read_record(p))) for p in games.records.values())
    print(f"pass: {result['seconds'] / result['decisions'] * 1e3:.2f} ms per position on the CPU "
          f"({result['decisions']} decisions, {result['windows']} windows)")


def _reference_gradient(games):
    """sum over game-sides of R sum_t grad log pi, each game-side walked
    alone with its memory's gradient never cut, by code of its own: the
    network's forward per decision, a shadow row reading the memory before
    it and writing nothing. Divided by 2N, as the pass divides."""
    from helpers.pg_games import named, side_forwards
    from tools.pg_pass import load_network
    from tools.pg_positions import positions_of, read_record
    from wesnoth_ai.param_groups import named_model_parameters
    from wesnoth_ai.pg_loss import action_log_probs
    encoder, model = load_network(games.checkpoint, CPU)
    labels = named(games, R)
    total = torch.zeros(())
    for game, path in games.records.items():
        pos = positions_of(read_record(path), *games.vocab, game=game)
        for side, rows in pos.sides.items():
            for out, batch in side_forwards(encoder, model, rows):
                total = total + labels[(game, side)] * action_log_probs(out, batch)[0]
    named = named_model_parameters(model, encoder)
    grads = torch.autograd.grad(total, [p for _, p in named], allow_unused=True)
    return {n: (torch.zeros_like(p) if g is None else g) / (2 * len(games.records)) for (n, p), g in zip(named, grads)}


def _flat(d, names):
    return torch.cat([d[n].flatten() for n in names])


@needs_core
def test_the_splits_add_to_the_total_and_the_windows_to_one_big_batch(tmp_path):
    """With windows longer than every game-side, the pass's total is the
    gradient of the whole sum taken at once; the eight split sums add to
    it, each split holds its games' part only, and a split of one window
    holds that window's square as its second moment. With short windows
    the memory's gradient is cut and the total moves. The head slopes are
    the total's derivative along each head's own weights."""
    from tools.pg_pass import load_network
    from wesnoth_ai.pg_loss import HEAD_PARAMETERS, HEADS
    games, result = _run(tmp_path, streams=2, window=64)
    names = result["names"]
    total = _flat(result["total"], names)
    ref = _flat(_reference_gradient(games), names)
    assert ref.norm() > 0
    assert torch.allclose(total, ref, rtol=1e-4, atol=1e-6 * float(ref.abs().max()))
    splits = [_flat(s, names) for s in result["splits"]]
    assert torch.allclose(sum(splits), total, atol=1e-7)
    assert [i for i, s in enumerate(splits) if s.norm() > 0] == sorted(SPLITS.values())
    assert result["split_games"] == [1, 0, 0, 1, 0, 1, 0, 0]
    one_window = [i for i, k in enumerate(result["split_windows"]) if k == 1]
    assert one_window
    for i in one_window:
        g_w = splits[i] * 2 * result["games"]
        assert torch.allclose(_flat(result["sq"][i], names), g_w * g_w, rtol=1e-5, atol=1e-12)
    params = {n: p.detach() for n, p in load_network(games.checkpoint, CPU)[1].named_parameters()}
    for head in HEADS:
        along = sum(float((result["total"][n] * params[n[len("model."):]]).sum()) for n in HEAD_PARAMETERS[head])
        assert result["head_slopes"][head] == pytest.approx(along, rel=1e-3, abs=1e-7), head
    _, cut = _run(tmp_path, streams=2, window=2)
    assert not torch.allclose(_flat(cut["total"], names), total, rtol=1e-3, atol=0.0)


@needs_core
def test_the_floors_size_changes_only_the_floor(tmp_path):
    """Prereg departure 46: a floor over fewer game-sides reads fewer of
    them, and the pass's sums (the total, the splits, their second moments,
    the head slopes and the replay identity) are the same, bit for bit."""
    games, every = _run(tmp_path, streams=2, floor_sides=1000)
    _, few = _run(tmp_path, streams=2, floor_sides=2)
    assert every["floor"]["game_sides"] == 6 and few["floor"]["game_sides"] < 6
    names = every["names"]
    assert torch.equal(_flat(few["total"], names), _flat(every["total"], names))
    for part in ("splits", "sq"):
        assert all(torch.equal(_flat(a, names), _flat(b, names)) for a, b in zip(few[part], every[part])), part
    assert few["head_slopes"] == every["head_slopes"] and few["decisions"] == every["decisions"]
    assert few["replay"]["kl"] == every["replay"]["kl"]


@needs_core
def test_a_shadow_row_reads_the_memory_from_before_it_and_writes_nothing():
    """Along one slot: a decision, a refused one, a decision. The third
    reads what the first wrote, through the shadow; the shadow's own write
    reaches nothing. A game-side that starts with a shadow reads the
    initial memory twice."""
    from helpers.pg_games import tiny_network
    from tools.pg_schedule import PGMemoryRouter
    _, _, model = tiny_network()
    router = PGMemoryRouter(model, CPU, {})
    g = GameSide("g", 1)
    steps = [[Step(slot=0, game_side=g, offset=t, k=64, starts=t == 0)] for t in range(3)]
    written = [torch.randn(1, 64, model.d_model, requires_grad=True) for _ in range(3)]
    reads = []
    for t, shadow in enumerate((False, True, False)):
        reads.append(router.batch(steps[t]).padded)
        router.wrote(steps[t], written[t], [shadow])
    assert torch.equal(reads[1], written[0].detach()) and torch.equal(reads[2], written[0].detach())
    reads[2].sum().backward()
    assert written[0].grad is not None and written[0].grad.abs().sum() > 0
    assert written[1].grad is None
    assert torch.equal(router.memories()[0], written[2][0])
    router = PGMemoryRouter(model, CPU, {})
    first = router.batch(steps[0]).padded
    router.wrote(steps[0], written[0], [True])
    second = router.batch(steps[1]).padded
    assert torch.equal(first, second) and torch.equal(first[0], model.initial_memory(64).detach())
    assert np.isfinite(float(second.detach().sum()))


def _copied_games(tmp_path, games):
    """The suite's records and match results copied to a directory of
    their own, by game."""
    import shutil
    dst = tmp_path / "copied"
    dst.mkdir()
    for f in games.games.iterdir():
        shutil.copy(f, dst / f.name)
    return dst, {g: dst / p.name for g, p in games.records.items()}


def _rewrite(path, rec):
    from tools.game_record import GameRecordLog
    path.unlink()
    GameRecordLog(path).write(rec)


@needs_core
@pytest.mark.parametrize("where, key, value", [
    ("result", "checkpoint_sha256_b", "0" * 64),
    ("player", "procedure", "raw:t0+eo-1.5"),
    ("player", "memory", 0),
])
def test_a_game_side_another_player_played_is_refused(tmp_path, where, key, value):
    """A labelled game-side its match played with another checkpoint, under
    another procedure or at another memory size stops the pass and the
    evaluator before they replay anything."""
    from helpers.pg_games import named, write_labels
    from tools.pg_eval import EvalConfig, EvalContext
    from tools.pg_pass import GradientPass, PassConfig
    from tools.pg_positions import ForeignGameError, read_record, result_path
    games, labels = _labels(tmp_path)
    _, records = _copied_games(tmp_path, games)
    foreign = games.names["g1"]
    path = records[foreign]
    if where == "result":
        res = json.loads(result_path(path).read_text(encoding="utf-8"))
        res[key] = value
        result_path(path).write_text(json.dumps(res), encoding="utf-8")
    else:
        rec = read_record(path)
        rec["players"]["b"][key] = value
        _rewrite(path, rec)
    with pytest.raises(ForeignGameError, match=f"{foreign} side 2"):
        GradientPass(games.checkpoint, records, labels, CPU, PassConfig())
    splits = named(games, SPLITS)
    held = read_labels(write_labels(tmp_path / "held.jsonl", splits, named(games, R), heldout=list(splits),
                                    halves={g: "A" for g in splits}))
    with pytest.raises(ForeignGameError, match=f"{foreign} side 2"):
        EvalContext(games.checkpoint, records, held, CPU, EvalConfig())


def _scaled_actor(scale: float, packed: bool):
    """An injected forward: the fp32 law of the packed (or padded) trunk with
    its actor logits scaled, a difference of known sign and of a size set by
    the scale."""
    import dataclasses

    def forward(encoder, model, staged, memory):
        out = model.forward_embedded(encoder.embed_staged(staged), memory=memory, packed=packed).float32()
        return dataclasses.replace(out, actor_logits=out.actor_logits * scale)
    return forward


@needs_core
def test_the_replay_limit_rises_by_twice_the_floors_upper_bound(tmp_path):
    """The floor read at an injected serving precision (rounding against the
    packed trunk in fp32) lifts the limit to 1e-4 + 2 (floor + 2 SE): logs
    shifted to a KL estimate inside it fail without the floor and pass with
    it, past it fail either way. The real trunks agree in fp32 (the path
    check). A floor whose bound lifts the limit past 1e-3 is refused, and so
    is a packed trunk that disagrees with the padded one; pass.pt records the
    floor, its SE, its bound and the limit."""
    from tools.pg_floor import FLOOR_LIMIT_CAP, PATH_KL_LIMIT
    from tools.pg_pass import FloorRefused, GradientPass, PassConfig, PathMismatch, ReplayMismatch
    from tools.pg_positions import read_record
    games, labels = _labels(tmp_path)
    floor_config = dict(streams=2, floor_sides=1000)
    warm = _scaled_actor(1.05, packed=True)
    clean = GradientPass(games.checkpoint, games.records, labels, CPU, PassConfig(**floor_config),
                         serving_forward=warm).run()
    floor = clean["floor"]
    assert floor["decisions"] == clean["decisions"] and floor["game_sides"] == 6 and floor["games"] == 3
    assert 0 < floor["se"] < floor["kl"] and floor["upper"] == pytest.approx(floor["kl"] + 2 * floor["se"])
    # One row a forward reads the same floor, to fp32 batch rounding (the RX 6600 bounds its padded forwards).
    chunked = GradientPass(games.checkpoint, games.records, labels, CPU, PassConfig(floor_chunk=1, **floor_config),
                           serving_forward=warm).run()["floor"]
    assert chunked["decisions"] == floor["decisions"] and chunked["kl"] == pytest.approx(floor["kl"], rel=1e-3)
    assert chunked["path_kl"] < PATH_KL_LIMIT
    assert floor["path_kl"] < PATH_KL_LIMIT
    limit = clean["replay"]["limit"]
    assert limit == pytest.approx(1e-4 + 2 * floor["upper"]) and 2e-4 < limit < FLOOR_LIMIT_CAP
    dst, records = _copied_games(tmp_path, games)
    shifted = records[games.names["g2"]]
    original = read_record(shifted)
    share = len(original["decisions"]) / clean["decisions"]

    def run(kl, serving=warm, packed=None, **config):
        rec = json.loads(json.dumps(original))
        for d in rec["decisions"]:
            d["logp"] += kl / share
        _rewrite(shifted, rec)
        return GradientPass(games.checkpoint, records, labels, CPU, PassConfig(**config), serving_forward=serving,
                            packed_forward=packed).run()

    inside = (1e-4 + limit) / 2
    assert run(inside, **floor_config)["replay"]["kl"] == pytest.approx(inside, rel=1e-3)
    with pytest.raises(ReplayMismatch):
        run(inside, streams=2, floor_dtype="off")
    with pytest.raises(ReplayMismatch):
        run((limit + FLOOR_LIMIT_CAP) / 2, **floor_config)
    with pytest.raises(FloorRefused, match="stage 1 should serve in fp32"):
        run(0.0, serving=_scaled_actor(1.3, packed=True), **floor_config)
    with pytest.raises(PathMismatch, match="packed and padded trunks disagree"):
        run(0.0, packed=_scaled_actor(1.02, packed=True), **floor_config)
    # The default reading, bf16 autocast on this device's packed trunk.
    bf16 = GradientPass(games.checkpoint, games.records, labels, CPU, PassConfig(**floor_config)).run()["floor"]
    assert bf16["dtype"] == "bfloat16" and bf16["decisions"] == clean["decisions"]
    assert math.isfinite(bf16["kl"]) and bf16["kl"] > -1e-7 and bf16["path_kl"] < PATH_KL_LIMIT


@needs_core
def test_a_cut_pass_takes_its_written_splits_and_reads_the_rest(tmp_path, monkeypatch):
    """Each split's sums are written as it completes; a pass cut in its
    third split, started again, takes the two written splits and reads the
    rest, and its sums are the uncut pass's. A written split of other labels
    is read again."""
    from tools import pg_pass
    from tools.pg_pass import GradientPass, PassConfig
    games, labels = _labels(tmp_path)
    config = PassConfig(streams=2, floor_sides=1000)
    whole = GradientPass(games.checkpoint, games.records, labels, CPU, config).run()
    partial = tmp_path / "splits"
    run_split = pg_pass.GradientPass._run_split
    done = []

    def cut(self, i, schedule):
        if len(done) == 2:
            raise KeyboardInterrupt("the box's cut")
        run_split(self, i, schedule)
        done.append(i)

    monkeypatch.setattr(pg_pass.GradientPass, "_run_split", cut)
    with pytest.raises(KeyboardInterrupt):
        GradientPass(games.checkpoint, games.records, labels, CPU, config, partial_dir=partial).run()
    assert sorted(p.name for p in partial.iterdir()) == [f"split{i}.pt" for i in sorted(done)]
    monkeypatch.setattr(pg_pass.GradientPass, "_run_split", run_split)
    read = []
    monkeypatch.setattr(pg_pass.GradientPass, "_run_split",
                        lambda self, i, schedule: (read.append(i), run_split(self, i, schedule)))
    resumed = GradientPass(games.checkpoint, games.records, labels, CPU, config, partial_dir=partial).run()
    assert set(read).isdisjoint(done) and sorted(read + done) == list(range(8))
    # Its rate is the windows it timed, never the splits it took: too few here, so unread, and unpriced.
    from tools.stage_c_plan import UnreadRate, pass_rate
    assert resumed["rate"] is None
    with pytest.raises(UnreadRate):
        pass_rate({k: resumed[k] for k in ("rate", "decisions", "seconds")})
    names = whole["names"]
    assert torch.allclose(_flat(resumed["total"], names), _flat(whole["total"], names), rtol=1e-6, atol=1e-9)
    for key in ("probes", "head_slopes"):
        assert resumed[key] == pytest.approx(whole[key], rel=1e-6, abs=1e-9), key
    assert resumed["replay"] == pytest.approx(whole["replay"]) and resumed["decisions"] == whole["decisions"]
    assert resumed["floor"]["kl"] == pytest.approx(whole["floor"]["kl"])
    assert resumed["split_windows"] == whole["split_windows"] and resumed["shadow"] == whole["shadow"]
    from wesnoth_ai.pg_records import read_labels
    from helpers.pg_games import named, write_labels
    other = read_labels(write_labels(tmp_path / "other.jsonl", named(games, SPLITS),
                                     {k: -v for k, v in named(games, R).items()}))
    read.clear()
    GradientPass(games.checkpoint, games.records, other, CPU, config, partial_dir=partial).run()
    assert sorted(read) == sorted(SPLITS.values())


@needs_core
def test_the_command_line_trains_on_the_first_games_when_limited(tmp_path):
    """--train-limit N (stage C's C0): the first N labelled training games,
    in the labels' order, and every decision of theirs. Beside pass.pt, its
    summary; the rate counts the decisions past the warm-up windows."""
    from helpers.pg_games import named, write_labels
    from tools.pg_pass import WARMUP_WINDOWS, main
    from tools.pg_positions import read_record
    games, _ = _labels(tmp_path)
    labels = write_labels(tmp_path / "cli.jsonl", named(games, SPLITS), named(games, R))
    out = tmp_path / "c0_pass.pt"
    assert main(["--checkpoint", str(games.checkpoint), "--games", str(games.games), "--labels", str(labels),
                 "--out", str(out), "--device", "cpu", "--streams", "2", "--window", "2", "--train-limit", "2"]) == 0
    result = torch.load(out, weights_only=True)
    first = [games.names["g0"], games.names["g1"]]
    assert result["games"] == 2
    assert result["decisions"] == sum(len(decisions_of(read_record(games.records[g]))) for g in first)
    summary = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert {k: summary[k] for k in ("games", "decisions", "replay", "probes")} == \
        {k: result[k] for k in ("games", "decisions", "replay", "probes")}
    assert result["windows"] > WARMUP_WINDOWS
    rate = result["rate"]
    assert 0 < rate["decisions"] < result["decisions"] and rate["seconds"] < result["seconds"]
    assert rate["positions_per_second"] == pytest.approx(rate["decisions"] / rate["seconds"])
    # The floor reads every game-side of so few games: its forwards are kept out of the rate.
    assert rate["floor_seconds"] > 0
    assert rate["seconds"] + rate["floor_seconds"] == pytest.approx(rate["wall_seconds"])


@needs_core
def test_a_log_the_trainer_does_not_reproduce_fails_the_pass(tmp_path):
    """Logged log probabilities 0.05 above the network's on one game: the
    replay's KL estimate is 0.05 times that game's share of the decisions,
    above the limit, and the command line keeps the sums aside and exits 3
    without writing pass.pt."""
    from helpers.pg_games import named, write_labels
    from tools.pg_pass import EXIT_REPLAY, ReplayMismatch, main
    from tools.pg_positions import read_record
    games, _ = _labels(tmp_path)
    dst, records = _copied_games(tmp_path, games)
    shifted = records[games.names["g2"]]
    rec = read_record(shifted)
    for d in rec["decisions"]:
        d["logp"] += 0.05
    _rewrite(shifted, rec)
    total = sum(len(decisions_of(read_record(p))) for p in records.values())
    with pytest.raises(ReplayMismatch) as caught:
        _run(tmp_path, records=records, streams=2)
    assert caught.value.result["replay"]["kl"] == pytest.approx(0.05 * len(rec["decisions"]) / total, rel=1e-3)
    labels = write_labels(tmp_path / "cli.jsonl", named(games, SPLITS), named(games, R))
    out = tmp_path / "pass.pt"
    code = main(["--checkpoint", str(games.checkpoint), "--games", str(dst), "--labels", str(labels),
                 "--out", str(out), "--device", "cpu", "--streams", "2"])
    assert code == EXIT_REPLAY and not out.exists()
    kept = torch.load(tmp_path / "pass.replay_failed.pt", weights_only=True)
    assert kept["replay"]["kl"] > kept["replay"]["limit"] and kept["games"] == 3


@needs_core
def test_an_fp16_window_that_overflows_is_run_again_and_sums_as_fp32(tmp_path, monkeypatch):
    """--fp16 off a GPU runs no autocast but keeps its loss scale, and scaling
    by a power of two is exact in fp32. A scale so large that the windows'
    gradients overflow fp32 is halved until they do not, grown again every
    three finite windows (so it overflows again), and each overflowed window
    is run again with its readings undone: the sums, probes, readings, floor
    and counts are still the plain fp32 pass's."""
    from tools import pg_pass
    from tools.pg_loss_scale import LossScale
    from tools.pg_pass import GradientPass, PassConfig
    games, labels = _labels(tmp_path)
    config = dict(streams=2, window=8, floor_sides=1000)
    plain = GradientPass(games.checkpoint, games.records, labels, CPU, PassConfig(**config)).run()
    assert plain["loss_scale"] is None
    monkeypatch.setattr(pg_pass, "LossScale", lambda: LossScale(scale=2.0 ** 127, growth_interval=3))
    scaled = GradientPass(games.checkpoint, games.records, labels, CPU, PassConfig(fp16=True, **config)).run()
    ls = scaled["loss_scale"]
    assert ls["overflows"] > ls["growths"] >= 1 and ls["smallest"] < 2.0 ** 127
    names = plain["names"]
    assert torch.allclose(_flat(scaled["total"], names), _flat(plain["total"], names), rtol=1e-5, atol=1e-9)
    for i in range(8):
        assert torch.allclose(_flat(scaled["sq"][i], names), _flat(plain["sq"][i], names), rtol=1e-4, atol=1e-12)
    for key in ("probes", "head_slopes"):
        assert scaled[key] == pytest.approx(plain[key], rel=1e-5, abs=1e-9), key
    assert torch.allclose(torch.tensor(scaled["head_fisher"]), torch.tensor(plain["head_fisher"]), rtol=1e-9)
    assert scaled["replay"] == pytest.approx(plain["replay"], rel=1e-9)
    assert scaled["floor"]["kl"] == pytest.approx(plain["floor"]["kl"], rel=1e-9)
    for key in ("decisions", "windows", "split_windows", "shadow"):
        assert scaled[key] == plain[key], key


@needs_core
def test_a_resumed_fp16_pass_keeps_the_loss_scale_of_its_written_splits(tmp_path, monkeypatch):
    """Each split file holds the split's loss-scale counts and its end
    scale; a pass cut after four splits (two of them with windows) and
    started again takes them, carries the scale on from the last split
    taken, and records the uncut pass's loss scale: its overflows and
    growths over every split, its smallest and its end scale."""
    from tools import pg_pass
    from tools.pg_loss_scale import LossScale
    from tools.pg_pass import GradientPass, PassConfig
    games, labels = _labels(tmp_path)
    config = PassConfig(fp16=True, streams=2, window=8, floor_sides=1000)
    monkeypatch.setattr(pg_pass, "LossScale", lambda: LossScale(scale=2.0 ** 127, growth_interval=3))
    whole = GradientPass(games.checkpoint, games.records, labels, CPU, config).run()
    assert whole["loss_scale"]["overflows"] > whole["loss_scale"]["growths"] >= 1
    partial = tmp_path / "splits"
    run_split = pg_pass.GradientPass._run_split
    done = []

    def cut(self, i, schedule):
        if len(done) == 4:
            raise KeyboardInterrupt("the box's cut")
        run_split(self, i, schedule)
        done.append(i)
    monkeypatch.setattr(pg_pass.GradientPass, "_run_split", cut)
    with pytest.raises(KeyboardInterrupt):
        GradientPass(games.checkpoint, games.records, labels, CPU, config, partial_dir=partial).run()
    assert done == [0, 1, 2, 3]
    written = [torch.load(partial / f"split{i}.pt", weights_only=True)["loss_scale"] for i in done]
    assert sum(w["overflows"] for w in written) > 0
    monkeypatch.setattr(pg_pass.GradientPass, "_run_split", run_split)
    resumed = GradientPass(games.checkpoint, games.records, labels, CPU, config, partial_dir=partial).run()
    assert resumed["loss_scale"] == whole["loss_scale"]
    names = whole["names"]
    assert torch.allclose(_flat(resumed["total"], names), _flat(whole["total"], names), rtol=1e-6, atol=1e-9)


def test_the_loss_scale_stops_at_its_floor():
    """Halving below MIN_SCALE would not help an overflow fp16 itself makes:
    the pass stops there."""
    from tools.pg_loss_scale import MIN_SCALE, LossScale
    scale = LossScale(scale=2 * MIN_SCALE, growth_interval=2)
    assert scale.smallest == 2 * MIN_SCALE                 # the scale it starts from, not INITIAL_SCALE
    scale.overflowed()
    assert scale.scale == MIN_SCALE and scale.overflows == 1
    with pytest.raises(FloatingPointError):
        scale.overflowed()
    scale.finite()
    scale.finite()
    assert scale.scale == 2 * MIN_SCALE and scale.growths == 1


def _served(root: Path, games, **fields):
    """The suite's records copied under `root`, their match results naming
    the serving precision of `fields` (elo_eval_game's infer_* fields)."""
    from tools.pg_positions import result_path
    root.mkdir()
    dst, records = _copied_games(root, games)
    for path in records.values():
        res = json.loads(result_path(path).read_text(encoding="utf-8"))
        res.update(fields)
        result_path(path).write_text(json.dumps(res), encoding="utf-8")
    return dst, records


@needs_core
def test_the_floor_reads_the_precision_the_games_were_served_in(tmp_path):
    """auto reads the floor at the precision the games' results name: float16
    for games served in fp16 (the local RX 6600's stage 1), bfloat16 for a
    box's, the very floor and limit a named bfloat16 reads (a box's records
    pass as before), and the path alone for games served in float32, at the
    base limit. A named precision the games were not served in is refused
    before anything is replayed, and so are games of both precisions in one
    pass. The command line reads auto by default, and its summary names the
    floor read."""
    from helpers.pg_games import named, write_labels
    from tools.pg_floor import PATH_KL_LIMIT, PATH_ONLY, FloorDtypeError
    from tools.pg_pass import GradientPass, PassConfig, main
    games, labels = _labels(tmp_path)
    config = dict(streams=2, floor_sides=1000)

    def construct(records, dtype):
        return GradientPass(games.checkpoint, records, labels, CPU, PassConfig(floor_dtype=dtype, **config))

    fp16_dir, fp16 = _served(tmp_path / "fp16", games, infer_fp16=True, infer_bf16=False)
    half = construct(fp16, "auto").run()
    assert half["floor_dtype"] == "float16" and half["floor"]["dtype"] == "float16"
    assert half["floor"]["decisions"] == half["decisions"] and half["floor"]["kl"] > -1e-7
    with pytest.raises(FloorDtypeError, match="served in float16"):
        construct(fp16, "bfloat16")
    _, bf16 = _served(tmp_path / "bf16", games, infer_bf16=True)
    auto, explicit = construct(bf16, "auto").run(), construct(bf16, "bfloat16").run()
    assert auto["floor_dtype"] == explicit["floor_dtype"] == "bfloat16"
    assert auto["floor"] == explicit["floor"] and auto["replay"] == explicit["replay"]
    with pytest.raises(FloorDtypeError, match="served in bfloat16"):
        construct(bf16, "float16")
    fp32 = construct(games.records, "auto").run()
    assert fp32["floor_dtype"] == fp32["floor"]["dtype"] == PATH_ONLY and fp32["floor"]["kl"] is None
    assert fp32["floor"]["decisions"] == fp32["decisions"] and fp32["floor"]["path_kl"] < PATH_KL_LIMIT
    assert fp32["replay"]["limit"] == fp32["replay"]["base_limit"]
    mixed = dict(fp16)
    mixed[games.names["g2"]] = bf16[games.names["g2"]]
    with pytest.raises(FloorDtypeError, match="bfloat16 and float16"):
        construct(mixed, "auto")
    cli = write_labels(tmp_path / "cli.jsonl", named(games, SPLITS), named(games, R))
    out = tmp_path / "pass.pt"
    assert main(["--checkpoint", str(games.checkpoint), "--games", str(fp16_dir), "--labels", str(cli),
                 "--out", str(out), "--device", "cpu", "--streams", "2", "--floor-sides", "1000"]) == 0
    summary = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert summary["floor_dtype"] == "float16" and summary["config"]["floor_dtype"] == "auto"
    assert summary["floor"]["dtype"] == "float16"


@needs_core
def test_games_served_in_float32_keep_the_path_check_under_auto(tmp_path):
    """auto on games served in float32 (a stage 1 re-served in fp32, as C0's
    exit 4 advises) reads no rounding, and still checks the packed trunk
    against the padded one: a packed trunk that departs fails the pass
    (exit 5) as it does under a reduced-precision floor. Games served in
    float32 beside games served in float16 are refused: an fp16 floor would
    lift the limit for decisions that carry no rounding."""
    from tools.pg_floor import FloorDtypeError
    from tools.pg_pass import GradientPass, PassConfig, PathMismatch
    games, labels = _labels(tmp_path)
    config = PassConfig(floor_dtype="auto", streams=2, floor_sides=1000)
    with pytest.raises(PathMismatch, match="packed and padded trunks disagree") as caught:
        GradientPass(games.checkpoint, games.records, labels, CPU, config,
                     packed_forward=_scaled_actor(1.02, packed=True)).run()
    floor = caught.value.result["floor"]
    assert caught.value.result["floor_dtype"] == "float32" and floor["kl"] is None
    assert floor["decisions"] == caught.value.result["decisions"]
    _, fp16 = _served(tmp_path / "fp16", games, infer_fp16=True, infer_bf16=False)
    mixed = dict(games.records)
    mixed[games.names["g1"]] = fp16[games.names["g1"]]
    with pytest.raises(FloorDtypeError, match="float16 and float32"):
        GradientPass(games.checkpoint, mixed, labels, CPU, config)


@needs_core
def test_games_served_on_other_numerics_are_refused_by_the_floor(tmp_path):
    """Games served with the bf16 residual stream (or the fused ReLU, or fp8
    products) carry another rounding than the plain bf16 autocast the floor
    reads: the pass refuses them before replaying anything, unless the
    floor is off."""
    from tools.pg_floor import FloorDtypeError
    from tools.pg_pass import GradientPass, PassConfig
    games, labels = _labels(tmp_path)
    _, residual = _served(tmp_path / "residual", games, infer_bf16=True, infer_bf16_residual=True)
    for dtype in ("auto", "bfloat16"):
        with pytest.raises(FloorDtypeError, match="bfloat16\\+infer_bf16_residual"):
            GradientPass(games.checkpoint, residual, labels, CPU, PassConfig(floor_dtype=dtype))
    assert GradientPass(games.checkpoint, residual, labels, CPU, PassConfig(floor_dtype="off")).floor_dtype == "off"


def test_a_step_with_no_drawn_game_side_reads_nothing():
    """The floor reader on a step whose rows are none of its drawn
    game-sides runs no forward, chunked or not (unchunked, an empty step
    once gave range() a step of 0)."""
    from types import SimpleNamespace
    from tools.pg_floor import PATH_ONLY, FloorReader
    steps = [SimpleNamespace(game_side=SimpleNamespace(file="g", side=1))]
    for chunk in (None, 8):
        reader = FloorReader(None, None, CPU, {("other", 2)}, PATH_ONLY, chunk=chunk)
        reader._read_rows = lambda *a: pytest.fail("a forward ran on no drawn row")
        reader.read(steps, [], None, None)
        assert reader.sums == {}


@needs_core
def test_a_heads_only_pass_is_the_full_pass_on_the_head_maps(tmp_path):
    """--heads-only freezes every parameter but the head maps: their sums are
    the full pass's, every other sum is zero, the forward's readings are
    unchanged, the heads_only and probe_span directions are the full pass's,
    and the other arms refuse such a pass."""
    from tools.pg_pass import load_network
    from wesnoth_ai.param_groups import named_model_parameters
    from wesnoth_ai.pg_step import ARMS, HEADS_ONLY_ARMS, direction, head_of
    games, full = _run(tmp_path, streams=2, window=8)
    _, heads = _run(tmp_path, streams=2, window=8, heads_only=True)
    names = full["names"]
    head_names = [n for n in names if head_of(n)]
    assert head_names and len(head_names) < len(names) and heads["config"]["heads_only"]
    pairs = [(full["total"], heads["total"])] + list(zip(full["splits"], heads["splits"])) + list(zip(full["sq"], heads["sq"]))
    for a, b in pairs:
        # A tensor's own scale is no yardstick: the actor head's bias moves
        # every unit's logit alike, so its gradient is rounding around 0.
        scale = max(float(a[n].abs().max()) for n in head_names)
        for n in names:
            if head_of(n):
                torch.testing.assert_close(b[n], a[n], rtol=1e-5, atol=1e-6 * scale + 1e-12)
            else:
                assert not b[n].any(), n
    assert any(full["total"][n].any() for n in names if not head_of(n))
    assert heads["split_windows"] == full["split_windows"] and heads["decisions"] == full["decisions"]
    # A frozen trunk may take other kernels (attention's backend, say), so
    # the forward differs by rounding: sums that cancel to a small value
    # are held to their largest term's scale.
    for key in ("head_slopes", "probes"):
        scale = max(abs(v) for v in full[key].values())
        for k, v in full[key].items():
            assert heads[key][k] == pytest.approx(v, rel=1e-4, abs=1e-4 * scale), (key, k)
    fisher = np.asarray(full["head_fisher"])
    assert np.allclose(heads["head_fisher"], fisher, rtol=1e-4, atol=1e-4 * np.abs(fisher).max())
    assert heads["replay"]["kl"] == pytest.approx(full["replay"]["kl"], abs=1e-6)   # 1% of REPLAY_KL_LIMIT
    params = {n: p.detach() for n, p in named_model_parameters(*reversed(load_network(games.checkpoint, CPU)))}
    # The actor head's bias shifts every actor's logit alike: its gradient is
    # rounding (about 1e-7 of the weights'), which reduced AdamW's g / sqrt(v)
    # makes a unit-sized entry in either pass. It moves no probability.
    g_scale = max(float(full["total"][n].abs().max()) for n in head_names)
    rounding = {n for n in head_names if float(full["total"][n].abs().max()) < 1e-5 * g_scale}
    assert rounding <= {"model.actor_head.bias"}, rounding
    for arm in HEADS_ONLY_ARMS:
        a, b = direction(full, arm, params), direction(heads, arm, params)
        scale = max(float(t.abs().max()) for t in a.values())
        for n in set(names) - rounding:
            torch.testing.assert_close(b[n], a[n], rtol=1e-4, atol=1e-5 * scale)
    for arm in set(ARMS) - set(HEADS_ONLY_ARMS):
        with pytest.raises(ValueError, match="heads-only pass"):
            direction(heads, arm, params)
