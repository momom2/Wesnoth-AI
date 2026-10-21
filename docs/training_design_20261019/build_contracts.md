# Build contracts for stages 0, 1 and C (2026-10-21)

The code of `current_design.md` stages 0, 1 and C is built on three topic branches in parallel. This file fixes what crosses between them, so each branch can be written and tested alone. None of them merges into `main` before the user approves the design; they meet on an integration branch, `feature/pg-integration`, where an end-to-end CPU test runs on real logs.

## Branches and file ownership

| Branch | Purpose | Owns (creates or edits) |
|---|---|---|
| `feature/t1-selfplay-logs` (A) | Stage 1: the decision log, label rows, offline readouts, the box script | `tools/raw_player.py`, `tools/elo_eval_game.py`, `tools/run_elo_batch.py`, `tools/game_record.py` (only the `decisions` field), new `tools/decision_log.py`, `tools/selfplay_rows.py`, `tools/pg_labels.py`, `tools/selfplay_readout.py`, `scripts/selfplay_t1_box.sh` |
| `feature/pg-pass` (B) | Stage C, model side: the gradient pass over logged games and the network evaluator | new `wesnoth_ai/pg_records.py`, `wesnoth_ai/pg_loss.py`, `tools/pg_positions.py`, `tools/pg_schedule.py`, `tools/pg_pass.py`, `tools/pg_eval.py` |
| `feature/pg-step` (C) | Stage C, step math: directions, KL search, the gain estimators, the ladder, the critical batch, the toy game | new `wesnoth_ai/pg_step.py`, `wesnoth_ai/pg_protocol.py`, `tools/pg_stage_c.py`, `tests/helpers/toy_game.py` |

A branch that needs a change in a file it does not own stops and reports. No branch changes the Rust core.

## 1. The decision log (A writes it; B reads it)

A game record (`tools/game_record.py`) gains one field, `decisions`: a list with one entry per decision of either side, in play order, including refused ones. Each entry is a dict:

| Key | Type | Meaning |
|---|---|---|
| `side` | int | 1 or 2, the side that decided |
| `cmd` | int | index in the record's command list of the first command this decision emitted; -1 when the simulator refused it |
| `refused` | bool | the simulator refused the action (it emitted no command); the decision still counts (I-count) |
| `action` | list | the sampled action's key as `wesnoth_ai/own_loss.py` keys actions: `[actor, kind, target, weapon]` with the same index conventions as `OwnTarget` |
| `n` | int | the number of legal actions |
| `mass` | float | the sum of the network's joint priors over the legal list, before normalisation |
| `logp` | float | log probability of the sampled action under the network's raw law: prior / mass, no temperature, no offset |
| `H` | float | entropy of that law, in nats |
| `p_end`, `p_att`, `p_rec` | float | the law's total mass on end_turn, on attacks, on recruits |
| `p_att_actor` | float | p(attack \| the sampled actor), the type head's attack probability for that actor; NaN when the actor has no attack |
| `cdf` | [float, float] | cumulative law mass before and after the sampled action, in the legal list's order |
| `v` | float | the value head's expected value at the decision, in the deciding side's frame |

The log is written whatever the decode; at temperature 1 with no offset the played law is the logged law, and I-sample checks it.

## 2. Label rows (A writes `labels.jsonl`; B and C read it)

One JSON object per game-side:

```
{"game": str, "side": 1|2, "split": 0..7, "heldout": bool, "half": "A"|"B"|null,
 "z": float, "capped": bool, "cell": str, "S": [7 floats],
 "b": float, "beta_S": float, "R": float}
```

- `z` is the result in the side's frame: +1 win, -1 loss, 0 for a game that reached the turn cap.
- `S` holds the luck walk's seven sums (`tools/match_luck.py`) in the side's frame; `b` and `beta_S` are the baseline and luck terms subtracted; `R = z - b - beta_S`.
- `split` is the game's index modulo 8 (the eight split accumulators); `half` is set for held-out game-sides only.
- b and β are fitted on games other than the one labelled (cross-fitted).

## 3. The gradient pass's output (B writes `pass.pt`; C reads it)

A `torch.save` dict, loaded with `weights_only=True`:

```
{"names": [str], "total": {name: Tensor}, "splits": [8 x {name: Tensor}],
 "sq": [8 x {name: Tensor}], "windows": int, "games": int, "decisions": int,
 "split_games": [8 ints], "probes": {"end_turn": float, "attack": float, "recruit": float},
 "checkpoint_sha256": str}
```

- `total` is the sum of R · ∇log π over every decision, divided by 2 × games. `splits[i]` is the same sum over split i's games, divided by the same 2 × games, so `sum(splits) == total`; `split_games[i]` lets a reader renormalise a split to its own size.
- `sq[i]` is the mean over split i's windows of the squared window gradient (the reduced AdamW's second moment).
- `probes` are the gradients of the three probe scalars (design §3.4, P1).

## 4. The Evaluator protocol (C defines it in `wesnoth_ai/pg_protocol.py`; B's `NetworkEvaluator` and C's toy game implement it)

```python
class Evaluator(Protocol):
    def game_sides(self) -> Sequence[GameSide]: ...
    def base_logp(self) -> np.ndarray: ...
    def logp(self, s: float) -> np.ndarray: ...
    def mean_kl(self, s: float) -> float: ...

@dataclass(frozen=True)
class GameSide:
    game: str
    side: int
    R: float
    half: Optional[str]
    start: int      # this game-side's decisions are [start, stop) in the logp arrays
    stop: int
```

- The evaluator is bound to one direction d at construction; `logp(s)` is log π at θ_k + s·d for every held-out decision, in log order; `base_logp()` is `logp(0.0)` as the trainer recomputes it (not the logged value).
- `mean_kl(s)` is the mean per-decision KL(π_k ‖ π_{θ_k + s·d}) over the evaluator's KL subsample.

## Rules for the three agents

- Branch from `origin/main` at or after the commit that adds this file. Push freely; never merge into `main`.
- Local tests run only through the lock wrapper, never bare pytest, so that no two pytest processes run at once on the laptop: `python C:/Users/amaur/AppData/Local/Temp/claude/C--Users-amaur-Desktop-Perso-projects-Wesnoth-AI/7c28dcf1-1cd4-4ca7-8ace-2b7bb12fc907/scratchpad/pytest_locked.py -q <test files>`.
- Do not build or install the Rust wheel; the installed one is current for `main`.
- CI is the gate: a branch is done when its tip's CI run is green.
