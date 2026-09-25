"""Tunables for Wesnoth AI.

Deliberately small. Training hyperparameters, model sizes, and scenario
reference data from the earlier (deleted) ML code are gone — Phase 3
reintroduces them alongside the policy code that actually uses them.
"""

from pathlib import Path

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------

# Wesnoth executable. Steam install on Windows by default; the
# WESNOTH_EXE env var overrides (2026-08-04: lets the export-fidelity
# sweep run a source-built 1.18.4 on a headless Linux eval box).
import os as _os
WESNOTH_PATH = Path(
    _os.environ.get("WESNOTH_EXE")
    or r"C:\Program Files (x86)\Steam\steamapps\common\wesnoth\wesnoth.exe"
)

# Wesnoth userdata — where add-ons live, where Wesnoth writes its logs,
# and where our Lua side reads the action.lua file via the add-on
# directory junction.
WESNOTH_USERDATA_PATH = Path.home() / "Documents" / "My Games" / "Wesnoth1.18"
WESNOTH_LOGS_PATH = WESNOTH_USERDATA_PATH / "logs"

# Project layout — all source files of the add-on live here. Wesnoth
# sees them via the junction installed by main.install_addon().
#
# BASE_PATH is the REPO ROOT, not this package's dir: `add-ons/`,
# `training/` and `logs/` all live at the root. This file used to sit at
# the root, so `Path(__file__).parent` was the root -- the 2026-07-23
# package reorg moved it into `wesnoth_ai/` and silently pushed every
# derived path one level too deep (`wesnoth_ai/add-ons/...`,
# `wesnoth_ai/training/checkpoints`). It went unnoticed because the
# training entry points pass explicit paths; it broke `--check-setup` and
# the live-Wesnoth RCA eval, which resolve the add-on through here.
BASE_PATH       = Path(__file__).resolve().parent.parent
LOGS_PATH       = BASE_PATH / "logs"
ADDONS_PATH     = BASE_PATH / "add-ons" / "wesnoth_ai"
SCENARIOS_PATH  = ADDONS_PATH / "scenarios"
LUA_PATH        = ADDONS_PATH / "lua"
GAMES_PATH      = ADDONS_PATH / "games"

# Installed-add-on location; a directory junction / symlink back to
# ADDONS_PATH, managed by main.install_addon().
ADDON_INSTALL_PATH = WESNOTH_USERDATA_PATH / "data" / "add-ons" / "wesnoth_ai"

# Artifacts from training runs. Phase 1 doesn't produce checkpoints;
# kept so the layout is ready for Phase 3.
CHECKPOINTS_PATH = BASE_PATH / "training" / "checkpoints"
REPLAYS_PATH     = BASE_PATH / "training" / "replays"

# ----------------------------------------------------------------------
# Run configuration
# ----------------------------------------------------------------------

NUM_PARALLEL_GAMES   = 4      # Each game is a separate Wesnoth process.
                              # Lua generates a random per-process game_id
                              # so parallel games don't collide on the
                              # IPC directory. See training_scenario.cfg
                              # preload.
# Was 2000 — with a 16-way actor pool (units + recruits + end_turn),
# ~1/16 of random actions end the turn, so 2000 actions ≈ 60 turns.
# That's far longer than necessary, and a random policy never finds a
# leader kill before hitting the cap — meaning every game terminated
# on TIMEOUT (terminal reward 0) and the ±1 training signal never
# fired. 500 actions ≈ 15-30 turns, still enough room for proper
# engagement but produces way more terminations per hour and raises
# the probability of a stumbled-upon kill registering in the queue.
MAX_ACTIONS_PER_GAME = 500

# How often to emit aggregated stats / save checkpoints. Only
# checkpoints fire when the policy is trainable; stats always.
LOG_FREQUENCY        = 10
CHECKPOINT_FREQUENCY = 100


# ----------------------------------------------------------------------
# IPC (see wesnoth_interface.py)
# ----------------------------------------------------------------------
#
# Lua → Python (state): Lua std_print()s a framed block; it lands in
#   <userdata>/logs/wesnoth-*.out.log; Python tails the file.
# Python → Lua (action): Python atomically writes a Lua chunk to
#   <game_dir>/action.lua; Lua reads via wesnoth.read_file; a monotonic
#   `seq` field lets Lua distinguish fresh from stale.

ACTION_FILE_NAME       = "action.lua"
STATE_TIMEOUT_SECONDS  = 30.0
ACTION_TIMEOUT_SECONDS = 30.0
STATE_POLL_INTERVAL    = 0.01   # lowered from 0.05 — Python's 50 ms tick
                                # added an avg 25 ms tail to every
                                # read_state. 10 ms cuts that to ~5 ms
                                # and matches the Lua-side POLL_MS.

# ----------------------------------------------------------------------
# Encoder feature normalization
# ----------------------------------------------------------------------
# Each scalar feature is divided by its NORM constant before being
# concatenated into the unit / global token. The NORMs are chosen so
# default-era values land in the [0, ~1] range (the transformer's
# input scale). Override at config-time to support custom eras whose
# unit costs / HP exceed these (a cost-200 unit feeds in as 2.5 with
# the default COST_NORM=80; clipping is fine but you might want to
# tune COST_NORM up so the spread of cost values is more uniform).

HP_NORM       = 80.0    # full Walking-Corpse's HP =~22; ladder Drake max =~70
MOVES_NORM    = 10.0    # ladder caps around 7 (Wolf Rider with quick)
EXP_NORM      = 150.0   # 4-level units (e.g. Lich) need ~150 XP to AMLA
COST_NORM     = 80.0    # ladder caps around 60 (Yeti, Lich)
GOLD_NORM     = 500.0   # default-era 2p starts at 100; 500 covers
                        # late-game gold accumulation
INCOME_NORM   = 50.0
VILLAGES_NORM = 30.0    # large 4p maps; 2p ladder caps ~15
TURN_NORM     = 60.0    # default 2p ladder turn limit ~30, 60 for safety

# Combat-oracle biases. Two channels:
#
#   - TARGET_ALPHA: scales the per-target attack-bias added to the
#     hex-target logits when the policy chose ATTACK. Raises priors
#     for high expected-damage targets. 0.1 = moderate; 0 = off.
#   - TYPE_ALPHA: scales the per-actor "raise P(ATTACK | actor)"
#     bias added to the type_logits[ATTACK] when ANY reachable
#     enemy gives positive expected net damage. The aggregator is
#     `max_j(net_score[actor, j])` -- "you have at least one
#     profitable attack available, consider attacking."
#
# Both biases anneal multiplicatively over training: at the start
# of supervised training they're at full strength, by horizon end
# they're at 10% strength (configurable floor below). The policy's
# learned logits dominate after horizon; the oracle is just a
# warm-start prior. See action_sampler.combat_alphas_at().
# RETIRED until further notice (user 2026-07-16): with the
# behavior-cloning pass giving the policy real preferences, the
# oracle crutch comes off -- "time for the policy to learn to walk
# on its own legs". Machinery (bias computation + anneal schedule)
# stays intact; alphas 0.0 make every bias exactly zero at any
# decision_step. Restore by setting the old values (0.1 / 0.1).
# PRIOR HARDCODED BIAS (renamed from "combat oracle", user order
# 2026-08-06): a general facility for hand-placed prior nudges.
# POLICY: every instance defaults OFF (0.0) and is activated only in
# specific situations on the user's explicit order (pinned by
# tests/test_mcts.py::test_prior_hardcoded_bias_defaults_off).
# Instances:
#   - attack-target / attack-type bias (the original oracle shape;
#     retired 2026-08-05, defaults stay 0.0)
#   - end_turn bias on mini-category games (2026-08-06): activated
#     per-run via WESNOTH_PRIOR_BIAS_END_TURN_MINI=<float> (negative
#     = against passing); env-inherited so the pool's actors and the
#     trainer re-forward see the identical bias (symmetry contract).
COMBAT_TARGET_ALPHA = 0.0   # prior-bias instance: attack target
COMBAT_TYPE_ALPHA   = 0.0   # prior-bias instance: attack type
PRIOR_BIAS_END_TURN_MINI_DEFAULT = 0.0
# Backwards-compat alias (used by the rare external caller); the
# canonical names are the two above.
COMBAT_LOGIT_ALPHA = COMBAT_TARGET_ALPHA

# Anneal horizon (per-Python-decision count) over which the alphas
# decay from their configured value to ANNEAL_FLOOR_FRACTION × the
# configured value. 1M decisions ≈ a significant chunk of supervised
# training (5000 self-play games × 200 decisions/game), so the
# annealing reaches the floor late in training. Set to 0 to disable
# annealing (alphas stay at the configured value forever).
COMBAT_ANNEAL_HORIZON     = 1_000_000
# Minimum multiplier on the configured alphas at horizon end. 0.1 =
# the bias persists at 10% strength forever (a small nudge that's
# still useful late-training when the model has its own preferences).
COMBAT_ANNEAL_FLOOR_FRACTION = 0.1

# Pre-seeded faction vocab. The encoder pre-binds these to specific
# embedding rows so cross-replay supervised training stays
# consistent (a state that happens to feed Rebels first wouldn't
# alone determine its row). Empty string is reserved for
# "unknown / unset" -> id 0; the six default-era factions follow.
# Era mods can extend the list (the encoder's `MAX_FACTIONS=32`
# bound leaves headroom). Order MATTERS for backwards compatibility
# with saved checkpoints; appending is safe, reordering is not.
DEFAULT_FACTIONS = (
    "",
    "Drakes",
    "Knalgan Alliance",
    "Rebels",
    "Loyalists",
    "Northerners",
    "Undead",
)


# ----------------------------------------------------------------------
# Observation semantics epoch
# ----------------------------------------------------------------------
# Encoded observations are cached on disk: the pre-encoded corpora
# (tools/preencode_corpus.py) and the rehearsal cache
# (tools/policy_anchor.py). Both carry a fingerprint so a run refuses a
# cache built under a different vocab, hex basis or fog gate. Neither
# covered the SIM's OWN rules about what a player sees, so a cache
# built before such a rule changed would be mixed, silently, with
# encodings made after it.
#
# Bump this when a change alters WHAT A PLAYER SEES: the fog or shroud
# gate, which units are hidden, the hide abilities' terrain cover, the
# meaning of an encoder feature. Do NOT bump it for speed work that
# leaves the observation identical.
#
#   1  (up to 2026-09-12) the rules as they stood.
#   2  (2026-09-13) hide cover is the engine's [hides] terrain globs
#      (tools/terrain_resolver.hides_cover) instead of a defense-key
#      table, so ambush, concealment and submerge hide units on a
#      different set of hexes (docs/wesnoth_rules.md, "Hide cover is a
#      terrain-CODE filter").
#   3  (2026-09-13) a scenario [effect] identifies an ability and a
#      weapon special by its `id=`, not by the tag carrying it, and
#      apply_to=new_ability is applied at all. Silverhead Crossing's
#      Tentacle now has the submerge its scenario grants it, so it is
#      hidden on the deep water it stands on, and its evil-eye attack
#      carries `magical`, which SETS chance-to-hit to 70 (tools/scenario_events.py).
#   4  (2026-09-22) the global features carry this turn's and next turn's
#      lawful bonus (GLOBAL_FEAT_DIM 6 -> 8).
#   5  (2026-09-23) generation reads a side's declared fog (three minis
#      play without it), and the statues of Caves of the Basilisk, Sullas
#      Ruins and Thousand Stings Garrison carry their own modifications
#      (1 hp, no moves) in generation and reconstruction alike
#      (scenario_events.apply_side_unit_modifications).
#   6  (2026-09-24) a side sees its fog as the engine keeps it: each
#      unit's vision is what it could reach with its full movement plus
#      the ring around that, and what the side cleared during its turn
#      stays clear until the turn ends (visibility.py, docs/wesnoth_rules.md
#      "Vision and fog"); it was a disc of radius max_moves around each
#      unit's current hex.
#   7  (2026-09-24) replay reconstruction hides a side's revealed hiders
#      again at its turn start after turn 1 (unit::new_turn), as the
#      simulator did; the simulator no longer does it at turn 1, where
#      the engine does not either.
#   8  (2026-09-24) the simulator ends the neutral side's turn (the
#      tentacles of six mini maps) through the end_turn applier, as the
#      engine does for every side: a slowed tentacle's slow expires, and
#      a tentacle pinned at 0 MP loses `resting` and heals by
#      regeneration alone (docs/wesnoth_rules.md "End of a side's turn").
#      Replay reconstruction already applied it, so the corpora it
#      builds are unchanged; generated games on those maps change.
#   9  (2026-09-25) a declared zero village gold or village support is
#      paid as zero, not as the default (16 of 17,019 corpus games declare
#      one; no pool map does), and a unit that levels up keeps its
#      movement up to its new maximum with its traits, as unit::advance_to
#      clamps after re-applying them (a quick defender kept 5 of 6 moves).
#      The defender's counter weapon also follows the engine's level-up
#      scoring, which changes play, not what a side observes.
OBSERVATION_EPOCH = 9
