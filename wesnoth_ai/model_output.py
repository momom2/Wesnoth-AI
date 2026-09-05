"""The model's interface types: what WesnothModel's outputs are made of.

- The fixed vocabulary: TokenKind (stream tags), ActorKind (actor
  slots), UnitActionType (a unit actor's sub-decision), MAX_ATTACKS
  (the weapon head's width) and the value head's categorical support
  (VALUE_N_ATOMS, VALUE_V_MIN, VALUE_V_MAX).
- ModelOutput: one state's head outputs, what the sampler consumes.
- PaddedOutput: one batch's head outputs ([B, ...], padded) plus the
  per-sample sizes; its samples are ModelOutput views.

wesnoth_ai/model.py (the nn.Module) re-exports every name here, so
`from wesnoth_ai.model import ModelOutput` keeps working.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch


# How many attack slots per unit the weapon head predicts. Wesnoth
# units have 1–4 attacks typically; MAX_ATTACKS=4 covers them.
MAX_ATTACKS = 4

# Distributional value head (C51 / categorical-K). The head emits a
# softmax over K bins on a fixed support [V_MIN, V_MAX]; the scalar
# value the rest of the codebase reads is the distribution's mean,
# and `cliffness = std(Z(s))` is the same distribution's spread.
# Both come "for free" from one head with one categorical-CE loss
# (training is implicit: noisy returns → wider distribution →
# higher std). See trainer.py::_project_returns_to_atoms for the
# loss-side projection.
#
# The support is fixed to match the trainer's existing `value_clip`
# range. If you raise value_clip above 1.0, widen V_MIN/V_MAX
# accordingly or returns clip to the support edges and lose
# resolution.
#
# K=51 is the Bellemare et al. (2017) default. With our [-1, +1]
# range that's 0.04 / bin — plenty of resolution.
VALUE_N_ATOMS = 51
VALUE_V_MIN   = -1.0
VALUE_V_MAX   = +1.0

# Token-kind tags added to every stream so the transformer can tell
# "this is a hex" from "this is a recruit" during self-attention.
class TokenKind:
    HEX      = 0
    UNIT     = 1
    RECRUIT  = 2
    GLOBAL   = 3
    END_TURN = 4
    COUNT    = 5


# Mutually-exclusive categorization for ACTOR tokens. Sampler uses this
# to decide what to sample next (target hex? end the turn? weapon?).
class ActorKind:
    UNIT     = 0
    RECRUIT  = 1
    END_TURN = 2
    COUNT    = 3


# Action sub-type for UNIT actors. Recruit / end_turn actors don't
# need this sub-decision -- their action type is implicit in the
# actor kind. The type head only fires for UNIT actors and chooses
# between attacking an enemy or moving to an empty hex (HOLD is not
# modeled in v1; see CLAUDE.md design discussion).
class UnitActionType:
    ATTACK = 0
    MOVE   = 1
    COUNT  = 2



def _compute_marginal_type_logits(
    actor_logits: torch.Tensor,
    type_logits:  torch.Tensor,
    U: int, R: int,
    device,
) -> torch.Tensor:
    """Aggregate per-leaf-action-type probabilities by marginalizing
    over the actor distribution. Returns log-probabilities (for
    consistency with the rest of the model output). Layout
    [1, T+2] = [ATTACK, MOVE, RECRUIT, END_TURN].

    Done WITHOUT the legality mask -- the sampler's `_masked_*`
    helpers compose the legality on top. This is a debug / coarse-
    prior signal, not an exact distribution.
    """
    # Softmax over actors (joint actor probability).
    actor_probs = torch.softmax(actor_logits, dim=-1)   # [1, A]
    # Softmax over type axis (per-actor type prior).
    type_probs = torch.softmax(type_logits, dim=-1)     # [1, A, T]
    # Layout-dependent slicing: actor index 0..U-1 = UNIT,
    # U..U+R-1 = RECRUIT, U+R = END_TURN.
    if U > 0:
        unit_actor_p = actor_probs[:, :U]                   # [1, U]
        unit_type_p  = type_probs[:, :U, :]                  # [1, U, T]
        # Sum P(actor) * P(type | actor) over UNIT actors.
        attack_mass = (unit_actor_p * unit_type_p[:, :, UnitActionType.ATTACK]).sum(dim=-1)
        move_mass   = (unit_actor_p * unit_type_p[:, :, UnitActionType.MOVE]).sum(dim=-1)
    else:
        attack_mass = torch.zeros(1, device=device)
        move_mass   = torch.zeros(1, device=device)
    if R > 0:
        recruit_mass = actor_probs[:, U:U + R].sum(dim=-1)   # [1]
    else:
        recruit_mass = torch.zeros(1, device=device)
    end_turn_mass = actor_probs[:, U + R]                    # [1]

    # Stack and convert to log-probs. Clamp the inner sum to avoid
    # log(0) when an entire actor-class has zero mass.
    masses = torch.stack([attack_mass, move_mass, recruit_mass, end_turn_mass],
                         dim=-1)                              # [1, 4]
    return torch.log(masses.clamp_min(1e-10))


@dataclass
class ModelOutput:
    """Everything the sampler needs from one forward pass.

    Shapes (batch dim = 1 in Phase 3.1):
      A = num_units + num_recruits + 1   — # of actor slots, incl. end_turn
      H = # hex tokens
      T = UnitActionType.COUNT (= 2: ATTACK, MOVE)
    """
    actor_logits:  torch.Tensor  # [1, A]            pick an actor
    actor_kind:    torch.Tensor  # [1, A] long       UNIT / RECRUIT / END_TURN
    type_logits:   torch.Tensor  # [1, A, T]         per-actor ATTACK/MOVE
                                 #                   prior. Meaningful only
                                 #                   for UNIT slots; sampler
                                 #                   ignores T-axis for
                                 #                   non-UNIT actors.
    target_logits: torch.Tensor  # [1, A, H]         pick a hex per actor
    weapon_logits: torch.Tensor  # [1, A, MAX_ATTACKS]
    value:         torch.Tensor  # [1, 1]            mean of value distribution
    # Distributional value head outputs (categorical over K atoms).
    # Trainer's value loss reads `value_logits`; rollout / MCTS read
    # `value` (the mean). `cliffness` is the std of the predicted
    # distribution at this state -- a heteroscedastic uncertainty
    # estimate that comes for free from the categorical head and
    # marks states where small perturbations imply big value swings.
    value_logits:  torch.Tensor  # [1, K]            raw softmax logits
    cliffness:     torch.Tensor  # [1, 1]            std(Z(s))
    num_units:     int
    num_recruits:  int
    # Optional auxiliary prediction head (KataGo §3.5): the predicted
    # final MATERIAL margin from the acting side's perspective, tanh-
    # bounded to (-1, +1). `None` when the model was built without the
    # aux head (default). A denser training signal than win/loss z;
    # see trainer aux_coef + draw_tiebreak.material_margin.
    aux_score:     Optional[torch.Tensor] = None  # [1, 1] or None
    # Optional moves-left head (Lc0-style, 2026-07-04): predicted
    # FRACTION of the turn budget still to be played from this state,
    # sigmoid-bounded to (0, 1) (fraction of MOVES_LEFT_NORM_TURNS).
    # Trains as a dense regression alongside z; intended search-side
    # consumer (prefer shorter wins / longer losses) is wired
    # SEPARATELY and default-off pending calibration. `None` when the
    # model was built without the head (default).
    moves_left:    Optional[torch.Tensor] = None  # [1, 1] or None

    # GBC event-supervision tap (2026-08-14, docs/archive/gbc_spec.md):
    # contextualized token slices, populated ONLY when the model was
    # built with `gbc=True` and only on the single-state forward
    # path (the trainer's loss path). References to tensors already
    # computed in the forward -- zero extra compute; None by default
    # so eval/search paths pay nothing.
    unit_ctx:      Optional[torch.Tensor] = None  # [1, U, d] or None
    hex_ctx:       Optional[torch.Tensor] = None  # [1, H, d] or None
    global_ctx:    Optional[torch.Tensor] = None  # [1, 1, d] or None
    # Server-side priors (wesnoth_ai/server_priors.py): when set, the
    # legal actions with priors were computed on the inference server
    # from actor-shipped masks; enumerate_legal_actions_with_priors
    # unpacks them instead of reading the logits (which the server
    # then ships as placeholders).
    legal_compact: Optional[object] = None

    # Diagnostic: marginal-over-actors probability of each action
    # type. Layout [1, T+2]:
    #   0: ATTACK   (sum_unit_actors P(unit) * P(attack | unit))
    #   1: MOVE     (sum_unit_actors P(unit) * P(move   | unit))
    #   2: RECRUIT  (sum_recruit_actors P(recruit_slot))
    #   3: END_TURN (P(end_turn_slot))
    #
    # LAZY (optimization #2, 2026-06-14): this was a per-forward field
    # costing ~5-9% of every forward (two extra softmaxes + reductions),
    # but the ONLY reader is test_action_type_head.py -- self-play and
    # the trainer never touch it. It is now computed on demand from the
    # logits already stored on this dataclass, so the hot path pays
    # nothing. Recompute-on-access (no cache) is fine: the sole reader
    # touches it a handful of times.
    @property
    def marginal_type_logits(self) -> torch.Tensor:  # [1, T+2]
        return _compute_marginal_type_logits(
            self.actor_logits, self.type_logits,
            self.num_units, self.num_recruits,
            self.actor_logits.device,
        )


@dataclass
class PaddedOutput:
    """Batched head outputs ([B, ...], padded) plus per-sample sizes.
    `samples()` yields the per-sample ModelOutputs as views; `to_cpu()`
    moves each batched field ONCE (one transfer per field per batch)
    and yields CPU views -- the inference server's path."""
    actor_logits: torch.Tensor          # [B, A_max]
    actor_kind: torch.Tensor            # [B, A_max] long, CPU
    type_logits: torch.Tensor           # [B, A_max, T]
    target_logits: torch.Tensor         # [B, A_max, H_max]
    weapon_logits: torch.Tensor         # [B, A_max, W]
    value: torch.Tensor                 # [B, 1]
    value_logits: torch.Tensor          # [B, K]
    cliffness: torch.Tensor             # [B, 1]
    aux_score: Optional[torch.Tensor]
    moves_left: Optional[torch.Tensor]
    sizes: List[Tuple[int, int, int]]   # (U_b, R_b, H_b)
    unit_ctx: Optional[torch.Tensor] = None
    hex_ctx: Optional[torch.Tensor] = None
    global_ctx: Optional[torch.Tensor] = None

    _TENSOR_FIELDS = ("actor_logits", "type_logits", "target_logits", "weapon_logits",
                      "value", "value_logits", "cliffness", "aux_score", "moves_left",
                      "unit_ctx", "hex_ctx", "global_ctx")

    def to_cpu(self) -> "PaddedOutput":
        kw = {f: getattr(self, f) for f in ("actor_kind", "sizes")}
        for f in self._TENSOR_FIELDS:
            v = getattr(self, f)
            kw[f] = v.cpu() if v is not None else None
        return PaddedOutput(**kw)

    def float32(self) -> "PaddedOutput":
        """Cast bf16 autocast outputs back to float32 (numpy consumers
        never see bf16, as in `forward`)."""
        kw = {f: getattr(self, f) for f in ("actor_kind", "sizes")}
        for f in self._TENSOR_FIELDS:
            v = getattr(self, f)
            kw[f] = (v.float() if v is not None and v.dtype == torch.bfloat16 else v)
        return PaddedOutput(**kw)

    def sample(self, b: int) -> ModelOutput:
        U_b, R_b, H_b = self.sizes[b]
        A_b = U_b + R_b + 1
        return ModelOutput(
            actor_logits=self.actor_logits[b:b + 1, :A_b],
            actor_kind=self.actor_kind[b:b + 1, :A_b].to(self.actor_logits.device),
            type_logits=self.type_logits[b:b + 1, :A_b],
            target_logits=self.target_logits[b:b + 1, :A_b, :H_b],
            weapon_logits=self.weapon_logits[b:b + 1, :A_b],
            value=self.value[b:b + 1],
            value_logits=self.value_logits[b:b + 1],
            cliffness=self.cliffness[b:b + 1],
            num_units=U_b, num_recruits=R_b,
            aux_score=self.aux_score[b:b + 1] if self.aux_score is not None else None,
            moves_left=self.moves_left[b:b + 1] if self.moves_left is not None else None,
            unit_ctx=self.unit_ctx[b:b + 1, :U_b] if self.unit_ctx is not None else None,
            hex_ctx=self.hex_ctx[b:b + 1, :H_b] if self.hex_ctx is not None else None,
            global_ctx=self.global_ctx[b:b + 1] if self.global_ctx is not None else None)

    def samples(self) -> List[ModelOutput]:
        return [self.sample(b) for b in range(len(self.sizes))]
