"""Pub/sub event system for decoupled communication between game systems."""

from __future__ import annotations

import logging
from typing import Any, Callable
from collections import defaultdict

logger = logging.getLogger("dungeon_builder.events")


class EventBus:
    """Simple pub/sub event bus. Subscribers receive keyword arguments."""

    def __init__(self) -> None:
        self._listeners: dict[str, list[Callable[..., None]]] = defaultdict(list)

    def subscribe(self, event_type: str, callback: Callable[..., None]) -> None:
        self._listeners[event_type].append(callback)

    def unsubscribe(self, event_type: str, callback: Callable[..., None]) -> None:
        try:
            self._listeners[event_type].remove(callback)
        except ValueError:
            pass

    def publish(self, event_type: str, **kwargs: Any) -> None:
        for cb in self._listeners[event_type]:
            try:
                cb(**kwargs)
            except Exception:
                logger.exception(
                    "Error in event handler for '%s': %s", event_type, cb
                )

    def clear(self) -> None:
        self._listeners.clear()
