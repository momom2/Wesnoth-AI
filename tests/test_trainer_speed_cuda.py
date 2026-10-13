"""The sequence trainer's staging on CUDA (docs/trainer_speed_20261013.md,
item P3); skipped without a GPU (CI, the laptop), run by
scripts/trainer_speed_box.sh. The pinned non-blocking staging puts on the
device what the blocking copies it replaced put there, and a window's
forward loop, backward pass and clip never wait for the device: the one
wait is the read after them, with own game-sides in the window or not."""
from __future__ import annotations

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")

CUDA = torch.device("cuda")
QUIET = ["--probe-every", "1000000", "--barrier-positions", "1000000", "--signal-every", "1000000"]


def _trainer(pass_inputs, tmp_path, *extra):
    from tools import sequence_train
    args = sequence_train.parse_args([*pass_inputs, *QUIET, "--out", str(tmp_path / "arm.pt"),
                                      "--device", "cuda", *extra])
    return sequence_train.build_trainer(args)


def _blocking(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a)).to(CUDA)


def test_the_pinned_staging_equals_the_blocking_copies(tmp_path, pass_inputs):
    """On the fixture's batches: the belief targets, the padded and packed
    trunk indices; on synthetic own rows: the masks and the targets."""
    from test_own_loss import T, W, _masks, _target
    from wesnoth_ai.memory import stream_row_index
    from wesnoth_ai.own_loss import stage_own_batch
    from wesnoth_ai.packed_trunk import padded_gather_index
    from wesnoth_ai.padded_streams import padded_trunk_index, stream_widths
    from wesnoth_ai.sequence_loss import step_labels
    from wesnoth_ai.server_priors import mask_layout, write_masks
    trainer = _trainer(pass_inputs, tmp_path)
    n_steps = 0
    for steps in trainer._window():
        positions = [trainer.loader.sides(s.game_side)[s.offset] for s in steps]
        staged = trainer.encoder.stage_raws([p.raw for p in positions], device=CUDA)
        sizes, sightings, ks = staged.sizes, staged.sighting_counts, [s.k for s in steps]
        labels = step_labels(positions, steps, sizes, trainer.games, seed=1, value_states_per_game=16,
                             value_weight=1.0, pin=True)
        target, mask = labels.belief_on(CUDA)
        torch.cuda.synchronize()
        assert torch.equal(target, _blocking(labels.belief_target.numpy()))
        assert torch.equal(mask, _blocking(labels.belief_mask.numpy()))
        H_max, U_max, R_max, S_max, K_max = stream_widths(sizes, sightings, ks)
        padded = trainer.model.trunk_index(sizes, sightings, ks, CUDA, packed=False)
        pad, actor, _ = padded_trunk_index(sizes, H_max, U_max, R_max, sightings=sightings, S_max=S_max,
                                           memory=ks, K_max=K_max)
        gather = padded_gather_index(sizes, H_max, U_max, R_max, sightings=sightings, S_max=S_max)
        packed = trainer.model.trunk_index(sizes, sightings, ks, CUDA, packed=True)
        torch.cuda.synchronize()
        assert torch.equal(padded.pad_mask, _blocking(pad)) and torch.equal(padded.actor, _blocking(actor))
        assert torch.equal(padded.gather, _blocking(gather))
        for name in ("src", "kind", "actor", "hex", "glob"):
            assert torch.equal(getattr(packed.index, name), _blocking(getattr(packed.layout, name))), name
        if K_max:
            assert torch.equal(packed.memory_rows, _blocking(stream_row_index(ks, K_max)))
        n_steps += 1
    assert n_steps > 0
    rng = np.random.default_rng(5)
    shapes = [(3, 1, 9), (1, 0, 4), (2, 2, 13)]
    packs = [_masks(rng, U, R, H, b == 2) for b, (U, R, H) in enumerate(shapes)]
    targets = [_target(rng, n, U + R + 1, H) for n, (U, R, H) in zip((5, 1, 3), shapes)]
    on_device = stage_own_batch([0, 2, 3], packs, targets, 6, 13, T, W, CUDA)
    on_host = stage_own_batch([0, 2, 3], packs, targets, 6, 13, T, W, torch.device("cpu"))
    layout = mask_layout(3, 6, 13, T, W, type_bias=True, attack_bias=True)
    reference = np.zeros(layout.nbytes, np.uint8)
    write_masks(layout.numpy_views(reference), packs)
    torch.cuda.synchronize()
    for name, v in layout.numpy_views(reference).items():
        assert torch.equal(on_device.views[name], _blocking(v)), name
    for name in ("rows", "actor", "kind", "target", "weapon", "prob"):
        assert torch.equal(getattr(on_device, name), _blocking(getattr(on_host, name).numpy())), name


@pytest.mark.parametrize("own", [False, True], ids=["corpus", "own"])
@pytest.mark.parametrize("extra", [(), ("--activation-budget-gb", "100"), ("--packed-trunk",)],
                         ids=["padded", "kept activations", "packed"])
def test_a_window_waits_for_the_device_once(tmp_path, request, monkeypatch, extra, own):
    """Under torch.cuda.set_sync_debug_mode("error") from the window's start
    to its read: staging, forwards, the backward pass with its
    recomputations, and the clip raise on any synchronizing call; on the
    corpus alone, and with own game-sides on half the streams (their staging
    and soft-target loss), as production runs."""
    from tools import sequence_train
    inputs = request.getfixturevalue("own_pass_inputs" if own else "pass_inputs")
    trainer = _trainer(inputs, tmp_path, "--window", "1", *extra)
    assert (trainer.own is not None) == own
    assert trainer.train_window()                # the first window builds what is built once
    real = sequence_train.read_window
    reads = []

    def read(*a, **kw):
        torch.cuda.set_sync_debug_mode(0)
        reads.append(1)
        return real(*a, **kw)

    monkeypatch.setattr(sequence_train, "read_window", read)
    torch.cuda.synchronize()
    try:
        torch.cuda.set_sync_debug_mode("error")
        sums = trainer.train_window()
    finally:
        torch.cuda.set_sync_debug_mode(0)
    assert sums and reads == [1]
    assert (sums.get("own_n", 0) > 0) == own
