"""Policy contract for the Wesnoth AI.

A Policy decides what action to take given a game state. This module
defines the minimum interface plus a small registry so the `--policy`
CLI flag can resolve names to implementations without game_manager
having to know about every policy class.

Flavors a policy can be:
  - **Scripted** — no learning, just rules (e.g., `DummyPolicy`).
  - **Learned** — neural-net / tree-search policies trained via
    self-play or imitation (Phase 3).
  - **Composite** — one policy wrapping another. Example use case the
    user asked for: a scripted opener that delegates to a learned
    policy after N turns.

## Required contract

    select_action(game_state: GameState) -> Dict

The returned dict is in *internal* format — Position objects for
coords, 0-indexed. `game_manager.convert_action_to_json` transforms
it into the 1-indexed flat Wesnoth wire format.

Valid shapes:
  - ``{'type': 'end_turn'}``
  - ``{'type': 'move', 'start_hex': Position, 'target_hex': Position}``
  - ``{'type': 'attack', 'start_hex': Position, 'target_hex': Position,
        'attack_index': int}``
  - ``{'type': 'recruit', 'unit_type': str, 'target_hex': Position}``
  - ``{'type': 'recall', 'unit_id': str, 'target_hex': Position}``

## Optional hooks (duck-typed; no enforcement)

  - ``trainable: bool = False`` — whether `game_manager.run_training`
    should call `train_step`, save checkpoints, etc.
  - ``train_step() -> None`` — called once per batch of games.
  - ``save_checkpoint(path: pathlib.Path) -> None``
  - ``load_checkpoint(path: pathlib.Path) -> None``
  - ``observe(game_state, action, reward) -> None`` — called after each
    accepted action if the policy wants a replay-buffer feed.

These are intentionally not part of the Protocol: scripted policies
don't need them, and Phase 3 will pin down the exact training hooks
when there's a concrete learned policy to design against.

## Registering a new policy

    import policy
    policy.register("my_name", lambda: MyPolicy(some_arg))

Then `python main.py --policy my_name` picks it up.
"""

from typing import Callable, Dict, List, Protocol

from classes import GameState


class Policy(Protocol):
    """The one method every policy MUST provide.

    ``game_label`` is a caller-assigned string identifying which
    game/episode this call belongs to — trainable policies use it to
    bucket per-episode bookkeeping (rollout buffers, per-side
    trajectories). Scripted policies can ignore it. Required as a
    keyword-only argument so a call always reads unambiguously.
    """

    def select_action(
        self,
        game_state: GameState,
        *,
        game_label: str = "default",
    ) -> Dict: ...


# Name → zero-arg factory. Factories (not instances) so we don't
# eagerly construct policies the user won't use — important once
# Phase 3 learned policies are in here and construction means loading
# a checkpoint into memory.
_REGISTRY: Dict[str, Callable[[], Policy]] = {}


def register(name: str, factory: Callable[[], Policy]) -> None:
    """Associate a name with a policy factory. Later calls win."""
    _REGISTRY[name] = factory


def get_policy(name: str) -> Policy:
    """Construct a fresh policy instance by registered name."""
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown policy {name!r}. Available: {available()}"
        )
    return _REGISTRY[name]()


def available() -> List[str]:
    """Registered policy names, sorted for stable CLI --help output."""
    return sorted(_REGISTRY)


# ---------------------------------------------------------------------
# Built-in registrations. Imported at module load so `get_policy` and
# `available` work without callers having to remember to trigger
# registration.
# ---------------------------------------------------------------------

def _register_builtins() -> None:
    from dummy_policy import DummyPolicy
    register("dummy", DummyPolicy)
    # transformer_policy self-registers on import.
    import transformer_policy  # noqa: F401


_register_builtins()
