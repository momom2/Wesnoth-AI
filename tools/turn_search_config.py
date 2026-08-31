"""TurnSearchConfig and its knob-provenance helpers, in a module
with NO torch/sim imports (round-37 C3: the batch DRIVER reads TCS
knobs for its estimand pre-scan and timeout artifacts on every TCS
run, and importing them through turn_search pulled torch + the sim
stack (+177 MB RSS) into the process the memory guard sizes --
tools/eval_procedure.py exists for the same reason on the procedure
axis). tools/turn_search.py re-exports everything here, so search
code keeps its imports."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TurnSearchConfig:
    """TCS knobs (sigma/damping constants come from MCTSConfig so the
    target transform stays byte-shared with the Gumbel path)."""
    n_alt:          int = 4      # alternatives per coordinate/round
    rounds:         int = 3      # hill-climb rounds on full turns
    fast_rounds:    int = 1      # rounds on cheap (no-target) turns
    reval_salts:    int = 3      # fresh salts in acceptance stage 2
    min_delta:      float = 0.01  # accept floor (float-jitter guard)
    max_spine:      int = 40     # hard cap on spine length
    turn_full_prob: float = 0.25  # playout-cap analog, per TURN
    # Multi-turn projection at the boundary (docs/tcs_spec.md par.3;
    # user directive 2026-08-17, generalizing the opponent-reply arm).
    # Candidate turns are graded by the value `project_halfturns`
    # half-turns PAST our boundary, each half-turn played closed-loop
    # by the same policy -- one line, no branching, so cost is LINEAR
    # in depth. This is the guard against value-head tempo blindness
    # (the leg-3 turn-collapse mechanism): passing early stops looking
    # free once the evaluated state shows the opponent's free reply.
    # Placement:
    #   none  -- grade at our own boundary (status quo). DEFAULT OFF.
    #   reval -- projection gates stage-2 acceptance only: the climb
    #            proposes by the cheap boundary objective, the gate
    #            re-grades both sides of the pairing with projection.
    #   all   -- projection also drives stage-1 selection and the
    #            distill targets (the search and the training signal
    #            both optimize the projected objective; costlier).
    project:             str = "none"   # none | reval | all
    project_halfturns:   int = 1        # depth past our boundary
    project_max_actions: int = 40       # per-half-turn action cap
    # Target link function (user ruling 2026-08-17): "random draw
    # among the evaluated actions should not push their probability
    # up" -- evaluation EXPOSURE must carry no expected mass gain
    # under an uninformative grader.
    #   linear -- target = prior^lam * max(0, 1 + beta*(q - LOO
    #            mean of the other evaluated q)); linear in q, so
    #            symmetric judge error cancels to first order and
    #            E[target] ~ prior regardless of how often an action
    #            is evaluated. DEFAULT (leg-4 ruling: the grader is
    #            fresh/unproven; noise-robustness beats the exp
    #            link's concentration).
    #   exp    -- the AlphaZero/Gumbel mirror-descent tilt (sigma
    #            transform shared byte-for-byte with the MCTS path).
    #            Concentrates faster under a KNOWN-GOOD grader, but
    #            convex in q: under noise, evaluated actions gain
    #            expected mass in proportion to evaluation frequency
    #            (the leg-3 R2 end_turn exposure ratchet).
    target_link:         str = "linear"  # linear | exp
    target_beta:         float = 5.0     # linear-link advantage gain
    #   beta=5: an action 5 C51 atoms (0.20) below its evaluated
    #   peers' mean clips to zero mass; 2 atoms (0.08, the probe's
    #   median accepted delta) above gains +40% before renorm.
    #   Derivation in docs/design_constants.md.
    # Boundary evaluation frame (2026-08-21 fog finding, leg-4
    # postmortem): the post-end_turn boundary state's acting side is
    # the OPPONENT, and the encoder is acting-side-framed -- so the
    # grader saw only the opponent's fogged view of the mover's
    # turn. On no-contact fogged turns EVERY candidate graded
    # bit-identically (measured: 4 different candidate turns, one
    # value to 16 digits; fogless control spread 0.24-0.63).
    #   opponent -- post-flip state, sign-flipped (status quo;
    #               assumes fog symmetry that does not exist).
    #   mover    -- the PRE-end_turn state, mover still acting: the
    #               mover's own information set. Terminal flips
    #               still grade by exact outcome.
    # Default stays "opponent" until the A/B probes re-baseline;
    # leg-5 config must assert this explicitly.
    #   mover_mp0 -- mover frame with the boundary NEUTRALIZED
    #               (2026-08-31 collapse-probe finding): the pre-flip
    #               state shows each candidate's UNSPENT MP /
    #               un-acted units, so truncated plans parade latent
    #               "potential" and end_turn alternatives won 30-50%
    #               of accepted gates (the K-collapse door; under
    #               the opponent frame the same heads accepted 0).
    #               mp0 zeroes the mover's current_moves and sets
    #               has_attacked at the boundary encode -- the turn
    #               is over, spent or not -- so plans compare on
    #               POSITION.
    boundary_frame:      str = "opponent"  # opponent | mover | mover_mp0


TS_KNOB_KEYS = ("n_alt", "rounds", "fast_rounds", "reval_salts",
                "min_delta", "max_spine", "turn_full_prob",
                "project", "project_halfturns",
                "project_max_actions", "target_link", "target_beta",
                "boundary_frame")


def turn_knobs_dict(cfg: "TurnSearchConfig") -> dict:
    """Estimand provenance of a TCS player (round-32 C3: an eval
    that silently played dataclass defaults while the leg trained
    boundary_frame=mover was a different estimand the 'tcs:sims'
    procedure tag could not see)."""
    return {k: getattr(cfg, k) for k in TS_KNOB_KEYS}


# Choice-valued TCS knobs, shared by elo_eval_game's argparse and
# run_elo_batch's --ts-args pre-scan so the validators cannot drift
# (round-33 C3).
TS_CHOICES = {
    "--turn-project": ("none", "reval", "all"),
    "--turn-target-link": ("linear", "exp"),
    "--turn-boundary-frame": ("opponent", "mover", "mover_mp0"),
}

# --turn-* eval flag -> TurnSearchConfig field (+ cast; the batch
# pre-scan feeds string tokens, round-32 C3).
TS_FLAG_FIELDS = (
    ("turn_alt", "n_alt", int), ("turn_rounds", "rounds", int),
    ("turn_fast_rounds", "fast_rounds", int),
    ("turn_reval_salts", "reval_salts", int),
    ("turn_min_delta", "min_delta", float),
    ("turn_max_spine", "max_spine", int),
    ("turn_full_prob", "turn_full_prob", float),
    ("turn_project", "project", str),
    ("turn_project_halfturns", "project_halfturns", int),
    ("turn_project_max_actions", "project_max_actions", int),
    ("turn_target_link", "target_link", str),
    ("turn_target_beta", "target_beta", float),
    ("turn_boundary_frame", "boundary_frame", str),
)


def ts_config_from_args(args) -> "TurnSearchConfig":
    """TurnSearchConfig for eval: explicit --turn-* knobs override
    the dataclass defaults so a TCS match plays the SAME config the
    leg trained with (round-32 C3)."""
    kw = {}
    for flag, field, typ in TS_FLAG_FIELDS:
        v = getattr(args, flag, None)
        if v is not None:
            kw[field] = typ(v)
    return TurnSearchConfig(**kw)
