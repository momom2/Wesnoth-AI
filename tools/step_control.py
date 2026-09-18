"""Step-size control for the minimal self-play loop.

The optimizer proposes an update; the applied update is the largest
fraction alpha in {1, 1/2, 1/4, ...} of that proposal that lowers
the loss on a held-out fifth of the iteration's games (Armijo
backtracking with zero sufficient-decrease slope). The rule has no
learning-rate role: a fresh-moment Adam step (lr * sign(g) on every
parameter) or any other overshoot is shrunk until the held-out
games agree it helps. If no fraction helps, the step is skipped
and the weights restored.

Also reports how far the policy moved: KL(pi_old || pi_new), total
variation, and end_turn prior mass on real held-out states.
"""
from __future__ import annotations

import logging
import math
import statistics

import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

log = logging.getLogger("step_control")


@dataclass
class StepResult:
    alpha: float
    trials: int
    skipped: bool
    held_before: Dict[str, float]
    held_after: Dict[str, float]
    shift: Dict[str, float] = field(default_factory=dict)
    n_train: int = 0
    n_held: int = 0


def split_holdout(exps: List, frac: float, rng) -> Tuple[List, List]:
    """Hold out a fraction of GAMES (never of positions: positions
    of one game share its outcome label)."""
    gids = sorted({getattr(e, "game_id", "") for e in exps})
    if len(gids) < 2 or frac <= 0:
        return list(exps), []
    rng.shuffle(gids)
    n_held = max(1, int(round(frac * len(gids))))
    held = set(gids[:n_held])
    return ([e for e in exps if e.game_id not in held],
            [e for e in exps if e.game_id in held])


class _StubbedOptimizerStep:
    def __init__(self, trainer):
        self.trainer = trainer

    def __enter__(self):
        self.real = self.trainer.optimizer.step
        self.trainer.optimizer.step = lambda *a, **k: None
        return self

    def __exit__(self, *exc):
        self.trainer.optimizer.step = self.real


def held_loss(base, exps: List) -> Dict[str, float]:
    """Policy CE + value loss on `exps` through the production loss
    path, with the optimizer step stubbed out (weights untouched).

    Runs the loss WITHOUT gradients: this probe only reads two scalars,
    and the backward plus clip it used to pay for are 13-32% of a step.
    The stubbed optimizer stays as a second guard on the weights."""
    tr = base._trainer
    with torch.no_grad(), _StubbedOptimizerStep(tr):
        st = tr.step_mcts(list(exps), no_grad=True)
    tr.optimizer.zero_grad(set_to_none=True)
    return {"policy_ce": float(st.policy_loss),
            "value_loss": float(st.value_loss),
            "total": float(st.policy_loss) + float(st.value_loss)}


def held_loss_by_game(base, exps: List) -> Dict[str, Dict[str, float]]:
    """`held_loss` per held-out game (paired comparison unit: a
    game's positions share its outcome label, so games, not
    positions, are the independent samples)."""
    by_game: Dict[str, List] = {}
    for e in exps:
        by_game.setdefault(getattr(e, "game_id", ""), []).append(e)
    return {g: held_loss(base, es) for g, es in by_game.items()}


def _pooled(per_game: Dict[str, Dict[str, float]], exps: List) -> Dict[str, float]:
    """Experience-weighted pool of per-game losses (what one
    `held_loss` over all of them would report)."""
    n_of = {}
    for e in exps:
        g = getattr(e, "game_id", "")
        n_of[g] = n_of.get(g, 0) + 1
    tot = sum(n_of.values()) or 1
    out = {}
    for k in ("policy_ce", "value_loss", "total"):
        out[k] = sum(v[k] * n_of[g] for g, v in per_game.items()) / tot
    return out


def not_significantly_worse(before: Dict[str, Dict[str, float]],
                            after: Dict[str, Dict[str, float]],
                            z: float = 2.0) -> Tuple[bool, float, float]:
    """Paired test over held-out games on total loss: accept unless
    the mean per-game increase exceeds z standard errors. With one
    game there is no error estimate: require a strict decrease.
    Returns (ok, mean_delta, se)."""
    deltas = [after[g]["total"] - before[g]["total"] for g in before if g in after]
    if not deltas:
        return False, math.nan, math.nan
    mean = statistics.fmean(deltas)
    if len(deltas) < 2:
        return mean < 0.0, mean, math.nan
    se = statistics.stdev(deltas) / math.sqrt(len(deltas))
    return mean <= z * se, mean, se


def action_priors(base, e) -> Tuple[Dict, Dict, float]:
    """Normalized prior over the legal actions of `e`'s state and the
    state's value, from the INFERENCE model (the one search
    consults); plus action category per key."""
    import torch
    from wesnoth_ai.action_sampler import enumerate_legal_actions_with_priors
    with torch.no_grad():
        enc = base._inference_encoder.encode(e.game_state)
        out = base._inference_model(enc)
        legal = enumerate_legal_actions_with_priors(
            enc, out, e.game_state,
            decision_step=int(getattr(e, "decision_step", 0)))
        value = float(out.value.reshape(-1)[0].item())
    pri, cat = {}, {}
    for la in legal:
        key = (la.actor_idx, la.target_idx, la.weapon_idx,
               getattr(la, "type_idx", None))
        pri[key] = float(la.prior)
        cat[key] = la.action.get("type", "?")
    z = sum(pri.values()) or 1e-12
    return {k: v / z for k, v in pri.items()}, cat, value


def policy_shift(base, states: List, old: List[Tuple[Dict, Dict, float]]) -> Dict[str, float]:
    """How far the network's OUTPUTS moved on `states`: KL(old||new)
    and TV of the action prior, end_turn prior mass, and the value
    head's level shift (mean and mean-absolute V_new - V_old). A
    level shift is what tips search between acting and ending the
    turn when the head is flat within a turn."""
    kls, tvs, et, dvs = [], [], [], []
    for e, (p_old, cat, v_old) in zip(states, old):
        p_new, _, v_new = action_priors(base, e)
        kl = tv = 0.0
        for k, po in p_old.items():
            pn = max(p_new.get(k, 0.0), 1e-12)
            if po > 0:
                kl += po * math.log(po / pn)
            tv += abs(po - pn)
        kls.append(kl)
        tvs.append(0.5 * tv)
        et.append(sum(p for k, p in p_new.items() if cat[k] == "end_turn"))
        dvs.append(v_new - v_old)
    if not kls:
        return {}
    return {"kl_mean": statistics.fmean(kls),
            "kl_median": statistics.median(kls),
            "tv_mean": statistics.fmean(tvs),
            "end_turn_prior_mean": statistics.fmean(et),
            "dv_mean": statistics.fmean(dvs),
            "dv_abs_mean": statistics.fmean(abs(d) for d in dvs),
            "n": len(kls)}


_MODEL_PREFIX = "model."
_ENCODER_PREFIX = "encoder."


def publish_weights(base, theta: Dict) -> None:
    """Load `theta` (a flat dict from `_clone_weights`: model and
    encoder parameters under their prefixes) into the trainer's model
    AND encoder, then into the inference snapshots, the same contract
    as TransformerPolicy._snapshot_inference_weights. The encoder is
    trained by the same optimizer as the model; a backtrack that
    restored the model alone left the encoder at the full step."""
    base._model.load_state_dict(
        {k[len(_MODEL_PREFIX):]: v for k, v in theta.items() if k.startswith(_MODEL_PREFIX)})
    base._encoder.load_state_dict(
        {k[len(_ENCODER_PREFIX):]: v for k, v in theta.items() if k.startswith(_ENCODER_PREFIX)})
    with base._lock, base.serve_gate_exclusive():
        inf = getattr(base, "_inference_base", base._inference_model)
        inf.load_state_dict(base._model.state_dict())
        inf.eval()
        base._inference_encoder.load_state_dict(base._encoder.state_dict())
        base._inference_encoder.eval()


def _clone_weights(base) -> Dict:
    theta = {_MODEL_PREFIX + k: v.detach().clone()
             for k, v in base._model.state_dict().items()}
    theta.update({_ENCODER_PREFIX + k: v.detach().clone()
                  for k, v in base._encoder.state_dict().items()})
    return theta


def backtracking_step(base, take_step, train_exps: List, held_exps: List,
                      kl_states: Optional[List] = None, *,
                      shrink: float = 0.5, max_trials: int = 7,
                      max_level_shift: Optional[float] = None,
                      select: str = "first") -> StepResult:
    """Apply `take_step()` (the production update on `train_exps`,
    already queued by the caller), then shrink the applied move
    until BOTH hold: the held-out games are not significantly worse
    off (paired over games, `not_significantly_worse`), and (when
    `max_level_shift` is set) the value head's mean shift on
    `kl_states` stays within it.

    Why "not significantly worse" rather than "lower": the held-out
    fifth is ~5 games whose labels are one outcome each, so its loss
    is noisy at the size of a small step's effect; a strict-decrease
    test skipped every step once the value level had been corrected
    (leg az3, iteration 6). What the test must still catch is the
    systematic harm of an overshoot, and that shows on every game.

    Why the level cap on top of the loss test: a level shift that
    overshoots the label mean to the other side costs the same
    squared error, so held-out loss accepts it, while search turns
    any mover-frame level error b into a 2b act-vs-end_turn bias
    (step-scale measurement, 2026-09-03).

    `select`: "first" takes the largest passing fraction (Armijo);
    "best" evaluates every fraction down to shrink**(max_trials-1)
    and takes the passing one with the lowest held-out loss (an
    exact line search on the held-out games; costs max_trials
    evaluations every step).

    `take_step` must perform exactly one optimizer update on the
    trainer model and return its stats; the weights it leaves are
    the full proposal (alpha = 1)."""
    import torch
    theta0 = _clone_weights(base)
    old_priors = ([action_priors(base, e) for e in kl_states]
                  if kl_states else [])
    before_by_game = held_loss_by_game(base, held_exps) if held_exps else {}
    before = _pooled(before_by_game, held_exps) if held_exps else {"total": math.inf}
    take_step()
    theta1 = _clone_weights(base)
    # Publish the full proposal ourselves: the shift is read from the
    # inference snapshot, and a bare trainer step leaves it stale.
    publish_weights(base, theta1)
    delta = {k: theta1[k] - v for k, v in theta0.items()
             if torch.is_floating_point(v)}
    if not held_exps:
        shift = policy_shift(base, kl_states, old_priors) if kl_states else {}
        return StepResult(alpha=1.0, trials=0, skipped=False,
                          held_before={}, held_after={}, shift=shift,
                          n_train=len(train_exps), n_held=0)

    def _level_ok(shift: Dict) -> bool:
        if max_level_shift is None or not shift:
            return True
        return abs(shift["dv_mean"]) <= max_level_shift

    alpha, after, trials, shift = 1.0, None, 0, {}
    mean_d = se_d = math.nan
    best = None   # (held total, alpha, after, shift, mean_d, se_d) under "best"
    for t in range(max_trials):
        alpha = shrink ** t
        trials = t + 1
        if t > 0:
            publish_weights(base, {k: (v + alpha * delta[k] if k in delta else v)
                                   for k, v in theta0.items()})
        shift = policy_shift(base, kl_states, old_priors) if kl_states else {}
        if not _level_ok(shift):
            after = None
            continue
        after_by_game = held_loss_by_game(base, held_exps)
        after = _pooled(after_by_game, held_exps)
        ok, mean_d, se_d = not_significantly_worse(before_by_game, after_by_game)
        if ok and select == "first":
            break
        if ok and (best is None or after["total"] < best[0]):
            best = (after["total"], alpha, after, shift, mean_d, se_d)
    else:
        if best is not None:
            _, alpha, after, shift, mean_d, se_d = best
            publish_weights(base, {k: (v + alpha * delta[k] if k in delta else v)
                                   for k, v in theta0.items()})
            after = dict(after, delta_mean=mean_d, delta_se=se_d)
            log.info(f"step alpha={alpha:.4g} (best of {trials}) held-out "
                     f"{before['total']:.4f} -> {after['total']:.4f} "
                     f"(per-game delta {mean_d:+.4f} se {se_d:.4f}) | "
                     f"KL_med {shift.get('kl_median', float('nan')):.4f} "
                     f"dV {shift.get('dv_mean', float('nan')):+.4f}")
            return StepResult(alpha=alpha, trials=trials, skipped=False,
                              held_before=before, held_after=after, shift=shift,
                              n_train=len(train_exps), n_held=len(held_exps))
        publish_weights(base, theta0)
        shift = policy_shift(base, kl_states, old_priors) if kl_states else {}
        log.warning(f"step skipped: no alpha down to {alpha:.4g} passed "
                    f"(held-out {before['total']:.4f}, last delta "
                    f"{mean_d:+.4f} se {se_d:.4f}, level cap {max_level_shift})")
        return StepResult(alpha=0.0, trials=trials, skipped=True,
                          held_before=before, held_after=after or {},
                          shift=shift, n_train=len(train_exps),
                          n_held=len(held_exps))
    after = dict(after, delta_mean=mean_d, delta_se=se_d)
    log.info(f"step alpha={alpha:.4g} ({trials} trial(s)) held-out "
             f"{before['total']:.4f} -> {after['total']:.4f} "
             f"(per-game delta {mean_d:+.4f} se {se_d:.4f}) | "
             f"KL_med {shift.get('kl_median', float('nan')):.4f} "
             f"dV {shift.get('dv_mean', float('nan')):+.4f}")
    return StepResult(alpha=alpha, trials=trials, skipped=False,
                      held_before=before, held_after=after, shift=shift,
                      n_train=len(train_exps), n_held=len(held_exps))
