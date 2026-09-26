"""Loading pickles written before a module moved.

A pickle names each object's class by its module path. The records the
project keeps on disk (pre-encoded corpora, holdout sidecars,
policy-anchor caches, benchmark experiences, graphed-serve failure dumps)
hold project classes, so a record written before a module moved names a
module that no longer exists. Every loader of such a pickle reads through
`load` / `loads` here, which look each module path up in MOVED_MODULES
first.

The module is also a valid `pickle_module` for `torch.load`, which
subclasses `Unpickler`. It imports nothing but the standard library.
"""
import io
import pickle
from typing import Any, BinaryIO, Dict

# Old module path -> the path the module moved to; one entry per move.
# tools/dev/move_module.py --apply adds the entry for the module it moves.
MOVED_MODULES: Dict[str, str] = {
    "tools.scenarios": "wesnoth_ai.rules.scenarios",
    "tools.terrain_resolver": "wesnoth_ai.rules.terrain_resolver",
    "tools.wml_state": "wesnoth_ai.rules.wml_state",
}


def current_module(module: str) -> str:
    """Where `module` lives now: its entries in MOVED_MODULES followed
    to a path that did not move."""
    seen = []
    while module in MOVED_MODULES:
        if module in seen:
            raise ValueError(f"MOVED_MODULES has a cycle: {' -> '.join(seen + [module])}")
        seen.append(module)
        module = MOVED_MODULES[module]
    return module


class Unpickler(pickle.Unpickler):
    """pickle.Unpickler that finds a class under its module's current path."""

    def find_class(self, module: str, name: str) -> Any:
        return super().find_class(current_module(module), name)


def load(file: BinaryIO) -> Any:
    return Unpickler(file).load()


def loads(data: bytes) -> Any:
    return Unpickler(io.BytesIO(data)).load()
