"""One Elo-ladder evaluation game between two player specs; result to
a JSON file. Designed to be launched N-way parallel (each game is an
independent process — the pattern that saturated a 4090 where the
central-server pool could not; see BACKLOG 2026-07-03).

Usage:
    python tools/elo_eval_game.py LABEL_A SPEC_A LABEL_B SPEC_B \
        SIDE_A SEED OUTDIR [--max-turns 200] [--mcts-sims 32]

SPEC is a checkpoint .pt path or the literal 'dummy' (scripted
baseline). Checkpoint players play through MCTS at --mcts-sims
(training-matched, 32) unless 0 (raw policy). Maps come from the
LADDER-ONLY default `random_setup` (pinned by test_elo_ladder_maps).

The result file records BOTH the outcome and the final material
margin from A's perspective, so the collector can fit Elo under the
PURE (primary -- decisive games only; a capped game is a no-result
absence, not a draw, user 2026-08-17; material advantage is a
training crutch and does not factor into evaluation, user
2026-07-11) and material-sign (diagnostic) conventions from one set
of games.
Eval search likewise runs WITHOUT the material shapers
(draw_tiebreak, aux_value_bonus) regardless of training config.

Combat luck is PER GAME (2026-09-13): the sim's synced-RNG stream is
salted with this game's seed, so two games roll independent dice.
Before that every eval game shared one luck vector (the unsalted
`request_seed(k)` stream), which made an N-game match N draws against
one vector while the standard error assumed N independent ones. The
result records `combat_stream`; `--shared-combat-stream` restores the
old behavior for reproducing pre-2026-09-13 numbers only.

Shared inference (`--inference-address-a/-b`, tools/
eval_inference_server.py): a checkpoint side at sims 0 with a raw
temperature plays through a server that owns the model; this process
keeps the game loop, a RemoteEncoder with server-side priors and the
raw player. The result records `shared_inference`, `infer_bf16` and
`infer_packed_trunk` (the server's precision path, passed explicitly
on argv by the driver and checked against the server's hello), and
an outdir never mixes shared and per-process games. The procedure
tag stays `raw:t0`; before a shared-inference gate is quoted, re-pin
raw:t0 against itself once (the 20-game determinism check of
docs/box_specs.md), because batched bf16 numerics vary with batch
composition and argmax can flip on near-ties. One known divergence
from the per-process path: a unit type absent from the checkpoint's
vocab is aliased to the overflow bucket here (frozen-vocab
semantics, warned once per name), where the per-process encoder
grows its vocab with an untrained row.
"""

from __future__ import annotations

import argparse
import json
import os
import logging
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from wesnoth_ai.constants import OBSERVATION_EPOCH
from tools.draw_tiebreak import DrawTiebreakConfig, material_margin
from tools.elo_ladder import _ScriptedAdapter
from tools.eval_sim import (_PolicyPair, _load_policy,
                            _play_one_eval_game, peek_checkpoint_arch)
from tools.inference_seam import RemoteEncoder
from tools.run_elo_batch import basis_refusal, terrain_refusal
from tools.scenario_pool import build_scenario_gamestate, random_setup
from tools.wesnoth_sim import WesnothSim

log = logging.getLogger("elo_eval_game")


def _search_policy_cls(turn_search: bool, plan_tournament: bool = False):
    """Deployment sampling matches the training default (user ruling
    2026-08-26). --plan-a/--plan-b selects the plan-tournament
    procedure (proposition 1); otherwise TCS unless --no-turn-search
    opts down to per-decision Gumbel MCTS."""
    if plan_tournament:
        from tools.plan_tournament import PlanTournamentPolicy
        return PlanTournamentPolicy
    if turn_search:
        from tools.turn_policy import TurnCommitPolicy
        return TurnCommitPolicy
    from tools.mcts_policy import MCTSPolicy
    return MCTSPolicy


from tools.eval_procedure import procedure_of as _procedure_of  # noqa: E402


# The knob machinery lives in tools/turn_search_config (torch-free,
# round-37 C3) so the batch driver can read it without pulling the
# sim stack.
from tools.turn_search_config import (  # noqa: E402
    TS_CHOICES, ts_config_from_args as _ts_config,
)


def _pt_config(args):
    """TournamentConfig for eval: explicit --pt-* knobs override the
    code defaults so a match can play the SAME config the leg
    trained with (review C16)."""
    from tools.plan_tournament import PT_KNOB_KEYS, config_from_args
    from types import SimpleNamespace
    ns = SimpleNamespace(plan_tournament=True)
    for key in PT_KNOB_KEYS:          # single source (round-11 C2)
        k = "pt_" + key
        v = getattr(args, k, None)
        if v is not None:
            setattr(ns, k, v)
    return config_from_args(ns)


class _CountingModel:
    """Transparent proxy counting net forwards for a player -- the
    step-1 mandate is EQUAL MEASURED FORWARDS (round-12 C5).
    Installed BEFORE any search wrapper is constructed, because
    MCTSPolicy caches base._inference_model at __init__ (round-13
    C0: attach-after-construction left the MCTS arm counting a
    fabricated zero). forward_batch counts per SAMPLE (round-13 C1:
    __getattr__ passthrough silently omitted every batched boundary
    forward from the TCS arm's count)."""

    def __init__(self, inner):
        self._inner = inner
        self.n_forwards = 0
        # Cumulative wall seconds inside the model -- recorded per
        # result so ms/forward stays measured under every precision/
        # compile config (user 2026-08-28: the compile+bf16 default
        # must remain reviewable from the result files alone).
        self.fwd_secs = 0.0

    def __call__(self, *a, **k):
        self.n_forwards += 1
        t0 = time.perf_counter()
        try:
            return self._inner(*a, **k)
        finally:
            self.fwd_secs += time.perf_counter() - t0

    def forward_batch(self, encs, *a, **k):
        self.n_forwards += len(encs)
        t0 = time.perf_counter()
        try:
            return self._inner.forward_batch(encs, *a, **k)
        finally:
            self.fwd_secs += time.perf_counter() - t0

    def __getattr__(self, name):
        return getattr(self._inner, name)


# Worker mode (tools/eval_workers.py): loaded policies are kept
# across games so a worker pays checkpoint load, CUDA init and
# compile once. Keyed by everything that shapes the loaded object
# AND by the player label: the two sides of a same-spec match get
# distinct objects, as they do in one-process mode (a shared object
# would share the decision counter, the pending queue and the
# forward-counting proxy -- side A's counter read 0 on 2026-09-04).
_WORKER_MODE = False
_POLICY_CACHE: dict = {}
# Shared inference: one connection per server address (the hello
# handshake is paid once; across games in worker mode).
_CLIENT_CACHE: dict = {}
# The basis a checkpoint carries (peeked once per spec in worker mode).
_BASIS_CACHE: dict = {}


def _policy_for(spec, device, label, infer_bf16, infer_compile,
                relevant_set: bool = False):
    """`relevant_set` forces the relevant-hex-subset basis on the
    encoder whatever the checkpoint carries. It is part of the cache
    key: a persistent worker must never hand the mutated encoder to a
    full-board game."""
    # spec None is the random-init reference: a fresh draw per game
    # in one-process mode, so never cached (2026-09-04 review).
    cacheable = _WORKER_MODE and spec is not None
    key = (spec, label, str(device), bool(infer_bf16), bool(infer_compile),
           bool(relevant_set))
    if cacheable and key in _POLICY_CACHE:
        return _POLICY_CACHE[key]
    policy = _load_policy(Path(spec) if spec else None, device,
                          label=label, infer_bf16=infer_bf16,
                          infer_compile=infer_compile)
    if relevant_set:
        policy._inference_encoder.relevant_set_hexes = True
    if cacheable:
        _POLICY_CACHE[key] = policy
    return policy


_TERRAIN_CACHE: dict = {}


def _effective_terrain(spec, inference_address) -> str:
    """The terrain view this side's encoder plays in (run_elo_batch.
    TERRAIN_VIEWS): 'set' when the checkpoint carries terrain_multi_hot
    or the shared inference server's hello says so, else 'class'; a
    fresh net ('random') is 'set', 'dummy' has no encoder. Recorded as
    terrain_a/terrain_b and guarded per outdir like the basis."""
    if spec == "dummy":
        return "class"
    if inference_address is not None:
        return ("set" if _shared_client(inference_address).hello.get("terrain_multi_hot")
                else "class")
    if spec in (None, "random"):
        return "set"
    view = _TERRAIN_CACHE.get(spec) if _WORKER_MODE else None
    if view is None:
        view = ("set" if peek_checkpoint_arch(Path(spec), spec).get("terrain_multi_hot")
                else "class")
        if _WORKER_MODE:
            _TERRAIN_CACHE[spec] = view
    return view


def _effective_basis(spec, relevant_set: bool, inference_address) -> str:
    """The hex basis this side's encoder plays in (run_elo_batch.BASES):
    'relset' when the CLI flag, the checkpoint's relevant_set_hexes or
    the shared inference server's hello says so, else 'full'. Recorded
    in the result as basis_a/basis_b and guarded per outdir."""
    if spec == "dummy":
        return "full"                   # no encoder, no basis
    if relevant_set:
        return "relset"
    if inference_address is not None:
        return ("relset" if _shared_client(inference_address).hello["relevant_set"]
                else "full")
    if spec in (None, "random"):
        return "full"
    basis = _BASIS_CACHE.get(spec) if _WORKER_MODE else None
    if basis is None:
        basis = ("relset" if peek_checkpoint_arch(Path(spec), spec)
                 .get("relevant_set_hexes") else "full")
        if _WORKER_MODE:
            _BASIS_CACHE[spec] = basis
    return basis


def worker_loop() -> int:
    """`elo_eval_game.py --worker`: one JSON argv list per stdin line,
    `__DONE__ <rc>` per game on stdout (see tools/eval_workers.py)."""
    import json as _json
    import traceback as _tb
    global _WORKER_MODE
    _WORKER_MODE = True
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            rc = main(_json.loads(line))
        except SystemExit as e:
            # main refuses with SystemExit("<reason>"): keep the reason
            # visible in the worker's stderr as one-process mode does.
            if e.code is not None and not isinstance(e.code, int):
                print(str(e.code), file=sys.stderr, flush=True)
            rc = e.code if isinstance(e.code, int) else 1
        except Exception:                            # noqa: BLE001
            _tb.print_exc(file=sys.stderr)
            rc = 1
        print(f"__DONE__ {int(rc or 0)}", flush=True)
    return 0


def _shared_client(address: str):
    from tools.eval_inference_server import EvalInferenceClient
    client = _CLIENT_CACHE.get(address)
    if client is None or client.broken:
        client = EvalInferenceClient(address)
        _CLIENT_CACHE[address] = client
    return client


def _remote_player(address: str, raw_temperature: float, raw_seed,
                   relevant_set: bool, infer_bf16: bool, infer_packed_trunk: bool,
                   raw_end_turn: str = "joint", raw_end_turn_offset: float = 0.0):
    """The raw player over a shared inference server: a RemoteEncoder
    on the server's vocab with server-side priors, a RemoteModel
    behind the forward-counting proxy (its `fwd_secs` is the round
    trip: queue wait, batch, transport)."""
    import threading
    from types import SimpleNamespace
    from tools.inference_seam import RemoteModel
    from tools.raw_player import RawPolicyPlayer
    client = _shared_client(address)
    h = client.hello
    got = (bool(h["infer_bf16"]), bool(h["packed_trunk"]))
    if got != (bool(infer_bf16), bool(infer_packed_trunk)):
        raise SystemExit(
            f"the inference server at {address} serves (bf16, packed_trunk)="
            f"{got} but this game would record {(infer_bf16, infer_packed_trunk)}: "
            f"pass the server's precision (--infer-bf16/--no-infer-bf16, "
            f"--infer-packed-trunk/--no-infer-packed-trunk) so the result "
            f"file says what ran")
    counter = _CountingModel(RemoteModel(client))
    encoder = _VocabCheckedRemoteEncoder(
        h["type_to_id"], h["faction_to_id"], device=torch.device("cpu"),
        relevant_set=bool(h["relevant_set"]) or bool(relevant_set),
        server_priors=True,
        fog_hides_enemy_villages=bool(h.get("fog_hides_enemy_villages", False)),
        terrain_multi_hot=bool(h.get("terrain_multi_hot", False)))
    base = SimpleNamespace(_inference_model=counter, _inference_encoder=encoder,
                           _lock=threading.Lock(), _decision_step=0)
    return RawPolicyPlayer(base, raw_temperature, seed=raw_seed,
                           end_turn_rule=raw_end_turn,
                           end_turn_offset=raw_end_turn_offset), counter


class _VocabCheckedRemoteEncoder(RemoteEncoder):
    """RemoteEncoder that warns once per unit type absent from the
    server's vocab (encode_raw aliases it to the overflow bucket
    silently; the per-process encoder would grow its vocab)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._warned_names: set = set()

    def encode(self, game_state):
        for u in game_state.map.units:
            if u.name not in self._type_to_id and u.name not in self._warned_names:
                self._warned_names.add(u.name)
                log.warning("unit type %r is not in the server's vocab; "
                            "aliased to the overflow bucket", u.name)
        return super().encode(game_state)


def combat_salt(seed: int, shared_stream: bool = False) -> str:
    """The sim's combat-luck salt for the eval game on `seed`.

    An unsalted sim draws its k-th synced-RNG seed from
    `wesnoth_sim.request_seed(k)` = sha256("sim_to_replay:k")[:8], a
    pure function of the request counter (`WesnothSim._next_seed`, the
    no-salt branch). Every eval game therefore used to share one luck
    vector: the scenario and factions varied per game, the dice did
    not, so an N-game match was N draws against one vector while the
    standard error assumed N independent ones. Salting off the
    per-game seed makes the luck independent across games and keeps
    each game reproducible from its slot.

    `shared_stream=True` returns the empty salt, i.e. the
    pre-2026-09-13 behavior, for reproducing old numbers only."""
    return "" if shared_stream else f"elo:{int(seed)}"


def _build_player(spec: str, label: str, sims: int, device,
                  turn_search: bool = True,
                  plan_tournament: bool = False, pt_cfg=None,
                  ts_cfg=None, batch_size: int = 1,
                  infer_bf16: bool = False,
                  infer_compile: bool = False,
                  value_center: float = 0.0,
                  raw_temperature=None, raw_seed=None,
                  gumbel_root: bool = True,
                  relevant_set: bool = False,
                  inference_address=None,
                  infer_packed_trunk: bool = False,
                  raw_end_turn: str = "joint",
                  raw_end_turn_offset: float = 0.0):
    """`raw_temperature`: sims == 0 only -- the joint-temperature raw
    player (tools/raw_player.py; 0 = argmax). None = the legacy
    factored sampler, the pre-2026-09-04 'raw' procedure.
    `relevant_set`: encode this side with the relevant hex subset
    whatever the checkpoint carries (main passes the EFFECTIVE basis,
    `_effective_basis`). `inference_address`: play the raw player
    through a shared inference server (main() has checked sims == 0,
    a temperature and a checkpoint spec)."""
    if spec == "random":
        # Deliberate random-init reference (round-24 C8: reaching
        # random init through a nonexistent PATH is how a typo
        # produced a catalog edge against noise; the literal is the
        # only sanctioned route now).
        spec = None
    if spec == "dummy":
        from wesnoth_ai.dummy_policy import DummyPolicy
        return _ScriptedAdapter(DummyPolicy()), None
    if inference_address is not None:
        return _remote_player(inference_address, raw_temperature, raw_seed,
                              relevant_set, infer_bf16, infer_packed_trunk,
                              raw_end_turn, raw_end_turn_offset)
    policy = _policy_for(spec, device, label, infer_bf16, infer_compile,
                         relevant_set)
    inner = policy._inference_model
    if isinstance(inner, _CountingModel):      # cached policy: fresh counter
        inner = inner._inner
    counter = _CountingModel(inner)
    policy._inference_model = counter
    if sims > 0:
        from tools.mcts import MCTSConfig
        import os
        # EVALUATION CONTRACT (user, 2026-07-11): valuing material
        # advantage is a TRAINING crutch, not part of what policy
        # performance means -- so the material-based search shapers
        # (draw_tiebreak, aux_value_bonus) are OFF here regardless of
        # what the checkpoint trained with. Eval search sees the real
        # game: win +1, loss -1, draw 0. moves_left_utility (time
        # preference among equal outcomes, no material content) stays
        # env-configurable.
        cls = _search_policy_cls(turn_search, plan_tournament)
        mc = MCTSConfig(
            n_simulations=sims,
            gumbel_root=bool(gumbel_root),
            batch_size=max(1, int(batch_size)),
            moves_left_utility=float(
                os.environ.get("ELO_MOVES_LEFT_UTILITY", "0") or 0),
            # Part of the player, not a material shaper: the level
            # the checkpoint's loop centered its search on.
            value_center=float(value_center))
        if plan_tournament:
            return cls(policy, mc, tournament_config=pt_cfg), counter
        if turn_search:
            # No config = dataclass defaults = a DIFFERENT estimand
            # than the leg trained (round-32 C3: boundary_frame
            # defaults to "opponent" while leg 5+ trains "mover").
            return cls(policy, mc, turn_config=ts_cfg), counter
        return cls(policy, mc), counter
    if raw_temperature is not None:
        from tools.raw_player import RawPolicyPlayer
        return RawPolicyPlayer(policy, raw_temperature, seed=raw_seed,
                               end_turn_rule=raw_end_turn,
                               end_turn_offset=raw_end_turn_offset), counter
    return policy, counter


def _check_shared_inference_args(args, sims_a: int, sims_b: int) -> bool:
    """Whether this game plays through shared inference servers, after
    refusing the combinations that would mislabel the result: a
    served side must be a checkpoint at sims 0 with a raw temperature
    (the certified surface), both checkpoint sides must be served or
    neither (one game, one numerics path), and the server's precision
    must be stated explicitly on argv (main checks it against the
    server's hello when connecting)."""
    sides = (("a", args.inference_address_a, args.spec_a, sims_a, args.raw_temperature_a),
             ("b", args.inference_address_b, args.spec_b, sims_b, args.raw_temperature_b))
    shared = any(addr is not None for _, addr, _, _, _ in sides)
    for side, addr, spec, sims, temp in sides:
        if addr is not None:
            if spec in ("dummy", "random"):
                raise SystemExit(
                    f"--inference-address-{side} needs a checkpoint spec, not "
                    f"{spec!r}: a server has nothing to serve for 'dummy', and one "
                    f"fixed random-init net for a whole match is a different "
                    f"estimand than a fresh draw per game.")
            if sims > 0 or temp is None:
                raise SystemExit(
                    f"--inference-address-{side} serves the raw player only: that "
                    f"side needs sims 0 and --raw-temperature-{side}. Search and "
                    f"the legacy sampler are not certified through the server.")
        elif shared and spec != "dummy":
            raise SystemExit(
                f"side {side} would play per-process while the other side plays "
                f"through a server: one game, one numerics path. Serve both "
                f"checkpoint sides or neither.")
    if not shared:
        if args.infer_packed_trunk is not None:
            raise SystemExit("--infer-packed-trunk is shared-inference provenance "
                             "(an --inference-address-* side); the per-process "
                             "path never runs the packed trunk.")
        return False
    if args.infer_compile:
        raise SystemExit("--infer-compile with a shared server: the server runs "
                         "eager kernels; drop the flag.")
    if args.infer_bf16 is None or args.infer_packed_trunk is None:
        raise SystemExit("shared inference needs the server's precision stated: "
                         "--infer-bf16/--no-infer-bf16 and --infer-packed-trunk/"
                         "--no-infer-packed-trunk (the driver copies them from the "
                         "server's __INFO__ line; the result file records them).")
    return True


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("label_a")
    ap.add_argument("spec_a")
    ap.add_argument("label_b")
    ap.add_argument("spec_b")
    ap.add_argument("side_a", type=int, choices=(1, 2))
    ap.add_argument("seed", type=int)
    ap.add_argument("outdir", type=Path)
    ap.add_argument("--max-turns", type=int, default=200)
    ap.add_argument("--mcts-sims", type=int, default=32)
    ap.add_argument("--mcts-sims-a", type=int, default=None,
                    help="Player A's sims budget (default: --mcts-sims). "
                         "0 = raw policy, so one match can play "
                         "search-vs-no-search on the SAME weights (the "
                         "does-search-help-at-all engine test).")
    ap.add_argument("--mcts-sims-b", type=int, default=None,
                    help="Player B's sims budget (see --mcts-sims-a).")
    ap.add_argument("--value-center-a", type=float, default=0.0,
                    help="MCTSConfig.value_center for player A (the "
                         "centering its training loop used; 0 = off).")
    ap.add_argument("--value-center-b", type=float, default=0.0)
    ap.add_argument("--shared-combat-stream", action="store_true",
                    help="Play on the unsalted combat stream every "
                         "eval game shared before 2026-09-13 (the "
                         "k-th roll of every game identical). "
                         "Reproduce pre-2026-09-13 numbers only; the "
                         "result records combat_stream=shared and the "
                         "catalog refuses to pool it with per-game "
                         "streams.")
    ap.add_argument("--raw-temperature-a", type=float, default=None,
                    help="Player A at sims 0 plays the joint-temperature "
                         "raw player (tools/raw_player.py): 0 = argmax "
                         "of the policy's joint prior, 1 = the legacy "
                         "sampler in distribution. Default None = the "
                         "legacy factored sampler ('raw'). Recorded in "
                         "the procedure tag ('raw:t0'); estimands never "
                         "mix within an outdir.")
    ap.add_argument("--raw-temperature-b", type=float, default=None,
                    help="Player B (see --raw-temperature-a).")
    ap.add_argument("--raw-end-turn-a", choices=("joint", "actor"), default="joint",
                    help="How the raw player A decides end_turn: 'joint' = the "
                         "joint argmax/sample over every legal action; 'actor' = "
                         "end_turn only when its actor mass leads every actor "
                         "marginal, else the joint choice among non-end actions "
                         "(tools/raw_player.py; procedure tag '+endm').")
    ap.add_argument("--raw-end-turn-b", choices=("joint", "actor"), default="joint",
                    help="Player B (see --raw-end-turn-a).")
    ap.add_argument("--raw-end-turn-offset-a", type=float, default=0.0,
                    help="Offset added to the raw player A's end_turn actor logit "
                         "before its choice (negative = against passing; procedure "
                         "tag '+eo<x>').")
    ap.add_argument("--raw-end-turn-offset-b", type=float, default=0.0,
                    help="Player B (see --raw-end-turn-offset-a).")
    ap.add_argument("--relevant-set-a", action="store_true",
                    help="Encode side A's states with the relevant hex subset "
                         "(encoder relevant_set_hexes) whatever the checkpoint "
                         "was trained with: the zero-training probe of the "
                         "token-count study (docs/model_cost_study_20260905.md). "
                         "The effective basis is recorded per side (basis_a/"
                         "basis_b) and an outdir holds one basis per side.")
    ap.add_argument("--relevant-set-b", action="store_true",
                    help="Side B (see --relevant-set-a).")
    ap.add_argument("--gumbel-root-a", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Root procedure of side A's plain search: Gumbel "
                         "root (default, procedure 'mcts:<sims>') or "
                         "--no-gumbel-root-a for plain PUCT ('puct:<sims>', "
                         "what tools/az_loop.py trains with).")
    ap.add_argument("--gumbel-root-b", action=argparse.BooleanOptionalAction,
                    default=True, help="Side B (see --gumbel-root-a).")
    ap.add_argument("--mcts-batch-size", type=int, default=1,
                    help="Leaf-evaluation batch (virtual-loss batching, "
                         "both players). 1 = sequential, the canonical "
                         "protocol and the CPU optimum (2026-04 "
                         "measurement, mcts.py header). On GPU, 8-32 "
                         "amortizes launch overhead (5-10x per the same "
                         "header). Recorded in the result file: batched "
                         "search explores differently, so B must never "
                         "mix within an outdir.")
    ap.add_argument("--infer-bf16", action=argparse.BooleanOptionalAction,
                    default=None,
                    help="bfloat16 inference, both players. Default "
                         "AUTO: ON on cuda, OFF on cpu (user ruling "
                         "2026-08-28: compile+bf16 is the default -- "
                         "bench_infer measured 2.0x together, ~1x "
                         "each alone). Different logits = a "
                         "different measured object: the EFFECTIVE "
                         "value is recorded per result and never "
                         "mixes within an outdir. Explicitly forcing "
                         "it ON with a cpu device is refused (it "
                         "would silently no-op and mislabel).")
    ap.add_argument("--infer-compile", action=argparse.BooleanOptionalAction,
                    default=None,
                    help="torch.compile the inference model. Default "
                         "AUTO: ON on cuda, OFF on cpu. See "
                         "--infer-bf16 for the ruling/provenance "
                         "contract; ~10-14s compile per shape bucket "
                         "per process, amortized via the shared "
                         "TORCHINDUCTOR_CACHE_DIR kernel cache.")
    ap.add_argument("--inference-address-a", default=None,
                    help="Play side A through the shared inference server at "
                         "this address (tools/eval_inference_server.py; the "
                         "driver's --shared-inference). Sims 0 with a raw "
                         "temperature and a checkpoint spec only; requires "
                         "explicit --infer-bf16/--no-infer-bf16 and "
                         "--infer-packed-trunk/--no-infer-packed-trunk "
                         "(the server's precision, recorded in the result).")
    ap.add_argument("--inference-address-b", default=None,
                    help="Side B (see --inference-address-a).")
    ap.add_argument("--infer-packed-trunk", action=argparse.BooleanOptionalAction,
                    default=None,
                    help="Shared-inference provenance: whether the server runs "
                         "the packed varlen trunk. Recorded per result; never "
                         "mixes within an outdir.")
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"),
                    help="'auto' (default) uses CUDA when visible. PREFER "
                         "cuda when a GPU exists: profiled 2026-08-28, "
                         "86-90%% of a CPU game is the model forward, and "
                         "the same game ran 10x faster on a 3060 at "
                         "~420MB VRAM per game process (6 concurrent = "
                         "2.5/12GB -- a 12GB card fits ~20 games). 'cpu' "
                         "remains for GPU-less boxes.")
    ap.add_argument("--no-turn-search", action="store_true",
                    help="BOTH players use per-decision Gumbel MCTS "
                         "instead of TCS. Default is TCS -- deployment "
                         "sampling matches the training default (user "
                         "ruling 2026-08-26). Pre-2026-08-26 catalog "
                         "numbers were measured with this flag's "
                         "behavior.")
    ap.add_argument("--no-turn-search-a", action="store_true",
                    help="Player A only plays MCTS (per-checkpoint "
                         "deployment: each side plays the sampling it "
                         "was trained for -- e.g. an imitation seed is "
                         "an MCTS-native checkpoint).")
    ap.add_argument("--no-turn-search-b", action="store_true",
                    help="Player B only plays MCTS (see "
                         "--no-turn-search-a).")
    ap.add_argument("--plan-a", action="store_true",
                    help="Player A plays the plan-tournament "
                         "procedure (proposition 1, 2026-08-26).")
    ap.add_argument("--plan-b", action="store_true",
                    help="Player B plays the plan-tournament "
                         "procedure.")
    # Plan-tournament knobs: the eval must be able to play the SAME
    # config the leg trained with (review C16: a knob-less eval
    # silently measures code defaults, a different estimand).
    ap.add_argument("--pt-challengers", type=int, default=None)
    ap.add_argument("--pt-depths", type=str, default=None)
    ap.add_argument("--pt-redraws", type=int, default=None)
    ap.add_argument("--pt-cert-depth", type=int, default=None)
    ap.add_argument("--pt-cert-redraws", type=int, default=None)
    ap.add_argument("--pt-budget-forwards", type=int, default=None)
    ap.add_argument("--pt-margin-band", type=float, default=None)
    ap.add_argument("--pt-beta-max", type=float, default=None)
    ap.add_argument("--pt-margin-ref", type=float, default=None)
    # TCS knobs (round-32 C3): the eval must be able to play the
    # SAME turn-search config the leg trained with; None = the
    # TurnSearchConfig dataclass default.
    ap.add_argument("--turn-alt", type=int, default=None)
    ap.add_argument("--turn-rounds", type=int, default=None)
    ap.add_argument("--turn-fast-rounds", type=int, default=None)
    ap.add_argument("--turn-reval-salts", type=int, default=None)
    ap.add_argument("--turn-min-delta", type=float, default=None)
    ap.add_argument("--turn-max-spine", type=int, default=None)
    ap.add_argument("--turn-full-prob", type=float, default=None)
    ap.add_argument("--turn-project", default=None,
                    choices=(None,) + TS_CHOICES["--turn-project"])
    ap.add_argument("--turn-project-halfturns", type=int,
                    default=None)
    ap.add_argument("--turn-project-max-actions", type=int,
                    default=None)
    ap.add_argument("--turn-target-link", default=None,
                    choices=(None,)
                    + TS_CHOICES["--turn-target-link"])
    ap.add_argument("--turn-target-beta", type=float, default=None)
    ap.add_argument("--turn-boundary-frame", default=None,
                    choices=(None,)
                    + TS_CHOICES["--turn-boundary-frame"])
    ap.add_argument("--log-level", default="WARNING")
    args = ap.parse_args(argv[1:])
    if args.label_a == args.label_b:
        raise SystemExit("--label-a and --label-b must differ (result files "
                         "and the workers' per-side policy cache are keyed "
                         "by label)")
    for _side, _flag, _plan, _nts in (
            ("a", args.gumbel_root_a, args.plan_a,
             args.no_turn_search or args.no_turn_search_a),
            ("b", args.gumbel_root_b, args.plan_b,
             args.no_turn_search or args.no_turn_search_b)):
        if not _flag and (_plan or not _nts):
            raise SystemExit(f"--no-gumbel-root-{_side} applies to a plain "
                             f"search only (--no-turn-search, no --plan-{_side}): "
                             f"the turn-search and plan-tournament procedures "
                             f"never read it")
    sims_a = (args.mcts_sims if args.mcts_sims_a is None
              else args.mcts_sims_a)
    sims_b = (args.mcts_sims if args.mcts_sims_b is None
              else args.mcts_sims_b)
    if (args.plan_a and sims_a <= 0) or (args.plan_b and sims_b <= 0):
        raise SystemExit(
            "--plan-a/--plan-b require that side's sims > 0 (sims 0 "
            "is the raw-policy player; a silently ignored procedure "
            "flag would mislabel the measured object).")
    if ((args.raw_temperature_a is not None and sims_a > 0)
            or (args.raw_temperature_b is not None and sims_b > 0)):
        raise SystemExit(
            "--raw-temperature-a/-b apply to the raw player only (that "
            "side's sims must be 0); a silently ignored temperature "
            "would mislabel the measured object.")
    for _n, _spec in (("spec_a", args.spec_a),
                      ("spec_b", args.spec_b)):
        if _spec not in ("dummy", "random") \
                and not Path(_spec).exists():
            raise SystemExit(
                f"{_n}={_spec!r} does not exist. A missing path "
                f"would silently play a RANDOM-INIT net under a "
                f"checkpoint's label and record a full Elo edge "
                f"against noise (round-24 C8). Pass the literal "
                f"'random' for a deliberate random-init player.")
    shared = _check_shared_inference_args(args, sims_a, sims_b)
    logging.basicConfig(level=getattr(logging, args.log_level))
    basis_a = _effective_basis(args.spec_a, args.relevant_set_a,
                               args.inference_address_a)
    basis_b = _effective_basis(args.spec_b, args.relevant_set_b,
                               args.inference_address_b)
    terrain_a = _effective_terrain(args.spec_a, args.inference_address_a)
    terrain_b = _effective_terrain(args.spec_b, args.inference_address_b)

    torch.set_num_threads(2)
    if shared:
        # This process holds no model: encoding and masks run on CPU
        # and the server's precision path is what the result records.
        device = None
        inf_bf16 = bool(args.infer_bf16)
        # The server's compiled packed loop is a numerics path of its
        # own; the record carries it as infer_compile so an outdir
        # never mixes compiled and eager games.
        _hello = _shared_client(args.inference_address_a or args.inference_address_b).hello
        inf_compile = bool(_hello.get("compile_packed", False))
        inf_packed = bool(args.infer_packed_trunk)
        logging.getLogger("elo_eval_game").warning(
            "shared inference: bf16=%s packed_trunk=%s compile_packed=%s",
            inf_bf16, inf_packed, inf_compile)
    else:
        if args.device == "cpu":
            device = None
        elif args.device == "cuda":
            if not torch.cuda.is_available():
                raise SystemExit("--device cuda requested but no CUDA device is "
                                 "visible; refusing to silently fall back to CPU "
                                 "(an eval that quietly changes device is an "
                                 "eval whose timings mean nothing).")
            device = torch.device("cuda")
        else:
            device = (torch.device("cuda") if torch.cuda.is_available() else None)

        # Precision/compile resolution (user ruling 2026-08-28:
        # compile+bf16 is the DEFAULT on cuda -- measured 2.0x together
        # on the real shape stream, ~1x each alone). On cpu both
        # default OFF; forcing them ON there is refused because they
        # would silently no-op and the result would be mislabeled.
        _cuda = device is not None and device.type == "cuda"
        inf_bf16 = _cuda if args.infer_bf16 is None else args.infer_bf16
        inf_compile = (_cuda if args.infer_compile is None
                       else args.infer_compile)
        inf_packed = False           # the per-process forward is single-sample
        if (inf_bf16 or inf_compile) and not _cuda:
            raise SystemExit(
                "--infer-bf16/--infer-compile require a cuda device: on "
                "cpu they no-op silently, so the result file would claim "
                "a precision that never ran.")
        logging.getLogger("elo_eval_game").warning(
            "inference config: bf16=%s compile=%s device=%s",
            inf_bf16, inf_compile, "cuda" if _cuda else "cpu")

    args.outdir.mkdir(parents=True, exist_ok=True)
    out_path = args.outdir / (
        f"game_{args.label_a}_{args.label_b}_s{args.side_a}"
        f"_{args.seed}.json")
    if out_path.exists():
        # Procedure guard (review C14 round 3): a result produced
        # under a DIFFERENT decision procedure must never be
        # silently reused -- estimands don't mix.
        try:
            prev = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 -- unreadable = replay it
            prev = None
        if prev is not None:
            want_a = _procedure_of(
                sims_a, args.plan_a,
                args.no_turn_search or args.no_turn_search_a,
                args.raw_temperature_a, args.gumbel_root_a,
                raw_end_turn=args.raw_end_turn_a,
                raw_end_turn_offset=args.raw_end_turn_offset_a)
            want_b = _procedure_of(
                sims_b, args.plan_b,
                args.no_turn_search or args.no_turn_search_b,
                args.raw_temperature_b, args.gumbel_root_b,
                raw_end_turn=args.raw_end_turn_b,
                raw_end_turn_offset=args.raw_end_turn_offset_b)
            got_a = prev.get("procedure_a")
            got_b = prev.get("procedure_b")
            got_mt = prev.get("max_turns")
            # Absent field = 1: every pre-flag result was B=1.
            if prev.get("mcts_batch", 1) != args.mcts_batch_size:
                raise SystemExit(
                    f"{out_path.name} was played at leaf-batch "
                    f"B={prev.get('mcts_batch', 1)} but this run "
                    f"uses B={args.mcts_batch_size}: batched search "
                    f"explores differently, refusing to mix. Use a "
                    f"fresh outdir.")
            # Absent fields = False: pre-flag results were fp32 eager.
            if bool(prev.get("infer_bf16", False)) != inf_bf16:
                raise SystemExit(
                    f"{out_path.name} was played with infer_bf16="
                    f"{bool(prev.get('infer_bf16', False))} but this "
                    f"run uses {inf_bf16}: precision changes "
                    f"the logits, refusing to mix. Use a fresh "
                    f"outdir.")
            if bool(prev.get("infer_compile", False)) != inf_compile:
                raise SystemExit(
                    f"{out_path.name} was played with infer_compile="
                    f"{bool(prev.get('infer_compile', False))} but "
                    f"this run uses {inf_compile}: compiled kernels "
                    f"may reorder float ops, refusing to mix. Use a "
                    f"fresh outdir.")
            # Absent = per-process, padded single-sample forward.
            if bool(prev.get("shared_inference", False)) != shared:
                raise SystemExit(
                    f"{out_path.name} was played with shared_inference="
                    f"{bool(prev.get('shared_inference', False))} but "
                    f"this run uses {shared}: batched forwards have "
                    f"different numerics, refusing to mix. Use a fresh "
                    f"outdir.")
            if bool(prev.get("infer_packed_trunk", False)) != inf_packed:
                raise SystemExit(
                    f"{out_path.name} was played with infer_packed_trunk="
                    f"{bool(prev.get('infer_packed_trunk', False))} but "
                    f"this run uses {inf_packed}: the packed and padded "
                    f"trunks are different kernels, refusing to mix. Use "
                    f"a fresh outdir.")
            # Combat-luck regime; absent = the pre-2026-09-13 stream
            # every eval game shared.
            _prev_epoch = int(prev.get("observation_epoch", 1))
            if _prev_epoch != int(OBSERVATION_EPOCH):
                raise SystemExit(
                    f"{out_path.name} was played under observation epoch {_prev_epoch} "
                    f"but this sim is {OBSERVATION_EPOCH}: the players saw different "
                    f"games, refusing to mix (constants.OBSERVATION_EPOCH). Use a fresh "
                    f"outdir.")
            _want_cs = ("shared" if args.shared_combat_stream
                        else "per_game")
            if prev.get("combat_stream", "shared") != _want_cs:
                raise SystemExit(
                    f"{out_path.name} was played on combat_stream="
                    f"{prev.get('combat_stream', 'shared')} but this "
                    f"run uses {_want_cs}: the shared stream gives "
                    f"every game the same luck vector, so the two are "
                    f"different estimands. Use a fresh outdir.")
            _want_vc = (args.value_center_a if sims_a > 0 else None,
                        args.value_center_b if sims_b > 0 else None)
            if (prev.get("value_center_a"),
                    prev.get("value_center_b")) != _want_vc \
                    and "value_center_a" in prev:
                raise SystemExit(
                    f"{out_path.name} was played at value_center "
                    f"({prev.get('value_center_a')},"
                    f"{prev.get('value_center_b')}) but this run uses "
                    f"{_want_vc}: it centers the search, refusing to "
                    f"mix. Use a fresh outdir.")
            _want_mlu = (float(os.environ.get(
                "ELO_MOVES_LEFT_UTILITY", "0") or 0)
                if (sims_a > 0 or sims_b > 0) else None)
            if "moves_left_utility" in prev \
                    and prev.get("moves_left_utility") != _want_mlu:
                raise SystemExit(
                    f"{out_path.name} was played at "
                    f"moves_left_utility="
                    f"{prev.get('moves_left_utility')} but this run "
                    f"uses {_want_mlu} (ELO_MOVES_LEFT_UTILITY): it "
                    f"changes the search's time preference, refusing "
                    f"to mix. Use a fresh outdir.")
            # The hex basis (effective per side; absent = full board).
            _why = basis_refusal(out_path.name, prev, (basis_a, basis_b))
            if _why is not None:
                raise SystemExit(_why)
            _why = terrain_refusal(out_path.name, prev, (terrain_a, terrain_b))
            if _why is not None:
                raise SystemExit(_why)
            if (got_a, got_b, got_mt) != (want_a, want_b,
                                          args.max_turns):
                raise SystemExit(
                    f"{out_path.name} exists with procedure/horizon "
                    f"({got_a},{got_b},max_turns={got_mt}) but this "
                    f"run wants ({want_a},{want_b},max_turns="
                    f"{args.max_turns}): refusing to mix estimands "
                    f"in one outdir. Use a fresh outdir.")
            if "pt_config" in prev and (args.plan_a or args.plan_b):
                from tools.plan_tournament import pt_knobs_dict
                cur = _pt_config(args)
                cur_knobs = None if cur is None else pt_knobs_dict(cur)
                if prev.get("pt_config") != cur_knobs:
                    raise SystemExit(
                        f"{out_path.name} was played under a "
                        f"different --pt-* config: refusing to mix "
                        f"(round-4 C12). Use a fresh outdir.")
            _any_tcs = (
                (sims_a > 0 and not args.plan_a
                 and not (args.no_turn_search
                          or args.no_turn_search_a))
                or (sims_b > 0 and not args.plan_b
                    and not (args.no_turn_search
                             or args.no_turn_search_b)))
            if _any_tcs or "turn_config" in prev:
                from tools.turn_search import turn_knobs_dict
                _want_tc = (turn_knobs_dict(_ts_config(args))
                            if _any_tcs else None)
                if prev.get("turn_config") != _want_tc:
                    raise SystemExit(
                        f"{out_path.name} was played under a "
                        f"different turn-search config "
                        f"({prev.get('turn_config')} vs {_want_tc})"
                        f": refusing to mix estimands (round-32 "
                        f"C3). Use a fresh outdir.")
            print(f"exists, skipping: {out_path.name}")
            return 0

    pt_cfg = _pt_config(args) if (args.plan_a or args.plan_b) else None
    ts_cfg = _ts_config(args)
    pa, cnt_a = _build_player(
        args.spec_a, args.label_a, sims_a, device,
        turn_search=not (args.no_turn_search or args.no_turn_search_a),
        plan_tournament=args.plan_a, pt_cfg=pt_cfg, ts_cfg=ts_cfg,
        batch_size=args.mcts_batch_size, infer_bf16=inf_bf16,
        infer_compile=inf_compile, value_center=args.value_center_a,
        raw_temperature=args.raw_temperature_a,
        raw_seed=2 * args.seed, gumbel_root=args.gumbel_root_a,
        relevant_set=basis_a == "relset",
        inference_address=args.inference_address_a,
        infer_packed_trunk=inf_packed,
        raw_end_turn=args.raw_end_turn_a,
        raw_end_turn_offset=args.raw_end_turn_offset_a)
    pb, cnt_b = _build_player(
        args.spec_b, args.label_b, sims_b, device,
        turn_search=not (args.no_turn_search or args.no_turn_search_b),
        plan_tournament=args.plan_b, pt_cfg=pt_cfg, ts_cfg=ts_cfg,
        batch_size=args.mcts_batch_size, infer_bf16=inf_bf16,
        infer_compile=inf_compile, value_center=args.value_center_b,
        raw_temperature=args.raw_temperature_b,
        raw_seed=2 * args.seed + 1, gumbel_root=args.gumbel_root_b,
        relevant_set=basis_b == "relset",
        inference_address=args.inference_address_b,
        infer_packed_trunk=inf_packed,
        raw_end_turn=args.raw_end_turn_b,
        raw_end_turn_offset=args.raw_end_turn_offset_b)

    rng = random.Random(args.seed)
    setup = random_setup(rng)
    gs = build_scenario_gamestate(setup)
    sim = WesnothSim(gs, scenario_id=setup.scenario_id,
                     max_turns=args.max_turns)
    # PER-GAME COMBAT LUCK (2026-09-13): see combat_salt for why an
    # unsalted eval stream is wrong. Safe for replay export --
    # sim_to_replay._seed_from_recorded reads the seed back out of the
    # recorded command instead of re-deriving it from the counter, so
    # a salted seed reaches [random_seed] unchanged.
    sim._seed_salt = combat_salt(args.seed, args.shared_combat_stream)
    game_label = out_path.stem
    t0 = time.time()
    r = _play_one_eval_game(
        sim,
        _PolicyPair(policy=pa, label=args.label_a, side=args.side_a),
        _PolicyPair(policy=pb, label=args.label_b, side=3 - args.side_a),
        game_label=game_label)
    margin_a = material_margin(sim.gs, args.side_a,
                               DrawTiebreakConfig(cap=0.3))
    if pt_cfg is None:
        pt_knobs = None
    else:
        from tools.plan_tournament import pt_knobs_dict
        pt_knobs = pt_knobs_dict(pt_cfg)
    _tcs_played = (
        (sims_a > 0 and not args.plan_a
         and not (args.no_turn_search or args.no_turn_search_a))
        or (sims_b > 0 and not args.plan_b
            and not (args.no_turn_search or args.no_turn_search_b)))
    if _tcs_played:
        from tools.turn_search import turn_knobs_dict
        ts_knobs = turn_knobs_dict(ts_cfg)
    else:
        ts_knobs = None
    result = {
        "label_a": args.label_a, "label_b": args.label_b,
        "pt_config": pt_knobs,
        "turn_config": ts_knobs,
        # Procedure provenance (review C19): result files from
        # different estimands must never be silently mergeable.
        "procedure_a": _procedure_of(
            sims_a, args.plan_a,
            args.no_turn_search or args.no_turn_search_a,
            args.raw_temperature_a, args.gumbel_root_a,
            raw_end_turn=args.raw_end_turn_a,
            raw_end_turn_offset=args.raw_end_turn_offset_a),
        "procedure_b": _procedure_of(
            sims_b, args.plan_b,
            args.no_turn_search or args.no_turn_search_b,
            args.raw_temperature_b, args.gumbel_root_b,
            raw_end_turn=args.raw_end_turn_b,
            raw_end_turn_offset=args.raw_end_turn_offset_b),
        # The CLI probe flags as given; TCS and plan-tournament arms
        # never read them (2026-09-04 review).
        "relevant_set_a": bool(args.relevant_set_a),
        "relevant_set_b": bool(args.relevant_set_b),
        # The EFFECTIVE hex basis per side (flag, checkpoint or server;
        # run_elo_batch.BASES): an estimand field, guarded per outdir.
        "basis_a": basis_a,
        "basis_b": basis_b,
        # The EFFECTIVE terrain view per side (checkpoint or server;
        # run_elo_batch.TERRAIN_VIEWS): an estimand field, guarded per outdir.
        "terrain_a": terrain_a,
        "terrain_b": terrain_b,
        "gumbel_root_a": (bool(args.gumbel_root_a) if sims_a > 0 and not args.plan_a
                          and (args.no_turn_search or args.no_turn_search_a) else None),
        "gumbel_root_b": (bool(args.gumbel_root_b) if sims_b > 0 and not args.plan_b
                          and (args.no_turn_search or args.no_turn_search_b) else None),
        "raw_temperature_a": args.raw_temperature_a,
        "raw_temperature_b": args.raw_temperature_b,
        "raw_end_turn_a": args.raw_end_turn_a,
        "raw_end_turn_b": args.raw_end_turn_b,
        "raw_end_turn_offset_a": args.raw_end_turn_offset_a,
        "raw_end_turn_offset_b": args.raw_end_turn_offset_b,
        # The horizon decides decisive-vs-absence, the quantity
        # the PURE fit is built on (round-24 C9).
        "max_turns": args.max_turns,
        # Leaf-batch provenance: batched (virtual-loss) search is a
        # slightly different explorer than sequential B=1.
        "mcts_batch": args.mcts_batch_size,
        # Precision/compile provenance (EFFECTIVE values): bf16
        # logits differ from fp32's; compiled kernels may reorder
        # float ops.
        "infer_bf16": inf_bf16,
        "infer_compile": inf_compile,
        # Shared inference (tools/eval_inference_server.py): batched
        # forwards on a server process; the packed varlen trunk is
        # its kernel choice. Both False for the per-process path.
        "shared_inference": shared,
        "infer_packed_trunk": inf_packed,
        # Combat-luck regime: "per_game" salts the sim's synced-RNG
        # stream with this game's seed; "shared" is the pre-2026-09-13
        # stream every eval game had in common. An estimand -- a
        # number from the shared stream must not pool with one from
        # per-game streams.
        # The SIM's observation semantics AT MEASUREMENT TIME. Two
        # raw:t0 dirs measured either side of an epoch bump would
        # declare otherwise identical estimands and pool with no
        # warning; today alone produced two bumps. This is the
        # sim's epoch, not the checkpoints' training epoch.
        "observation_epoch": int(OBSERVATION_EPOCH),
        "combat_stream": ("shared" if args.shared_combat_stream
                          else "per_game"),
        # Search knobs that change the PLAYER (2026-09-13: neither
        # reached a result field, so neither could be compared). None
        # when no side searched, so raw dirs stay unconstrained.
        "value_center_a": (args.value_center_a if sims_a > 0 else None),
        "value_center_b": (args.value_center_b if sims_b > 0 else None),
        "moves_left_utility": (
            float(os.environ.get("ELO_MOVES_LEFT_UTILITY", "0") or 0)
            if (sims_a > 0 or sims_b > 0) else None),
        "side_a": args.side_a, "seed": args.seed,
        "scenario_id": setup.scenario_id,
        "outcome_a": r.outcome,          # win/loss/draw/timeout from A
        "margin_a": float(margin_a),     # final material, A's view
        "turns": sim.gs.global_info.turn_number,
        # Measured forward counts (round-12 C5): the step-1 equal-
        # compute mandate is verified from these, per side. Per-
        # side-turn = forwards / turns (each side moves once/turn).
        "forwards_a": (cnt_a.n_forwards if cnt_a else None),
        "forwards_b": (cnt_b.n_forwards if cnt_b else None),
        # Wall seconds spent inside the model per side, so
        # ms/forward stays measured under whatever precision/
        # compile/device config -- the standing review record for
        # the compile+bf16 default (user 2026-08-28). Under shared
        # inference: the round trip to the server.
        "fwd_secs_a": (round(cnt_a.fwd_secs, 2) if cnt_a else None),
        "fwd_secs_b": (round(cnt_b.fwd_secs, 2) if cnt_b else None),
        "ended_by": sim.ended_by,
        "secs": round(time.time() - t0, 1),
    }
    # Atomic publish (round-24 C11): a kill mid-write must never
    # leave a truncated file occupying the slot.
    _tmp = out_path.with_suffix(".json.tmp")
    _tmp.write_text(json.dumps(result), encoding="utf-8")
    os.replace(_tmp, out_path)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    if "--worker" in sys.argv[1:]:
        sys.exit(worker_loop())
    sys.exit(main(sys.argv))
