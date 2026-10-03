"""A weight publication made while games are in flight, for the tests of
continuous generation (tools/actor_stream.py) on the real pool."""
from __future__ import annotations

# Publications made before giving up on seeing a game in flight outlive one.
PUBLICATION_ATTEMPTS = 3


def collect_across_publication(stream, publish, *, timeout: float = 600.0,
                               attempts: int = PUBLICATION_ATTEMPTS):
    """Publish (through `publish()`) while every actor is inside a game,
    then collect until those games have completed. Returns the games
    collected, the games that were in flight at the publication (by
    index) and the stream's own stamp of it.

    The stream's view of the games in flight lags the actors by the time
    their reports take to arrive: an actor's start report can wait behind
    another's experiences in the shared queue, so a game in the view may
    have ended already, and the test games last under a second. Every game
    in the view can then end before the publication is stamped (CI
    2026-10-03: both did, 0.42 and 0.07 s before it). Such an attempt shows
    nothing; the next one waits until every game in flight started after
    that stamp, so a game straddles at most the publication it is judged
    against, and publishes again, up to `attempts` publications."""
    after = float("-inf")
    for _ in range(attempts):
        flight = stream.wait_in_flight(timeout=120.0)
        while any(t_start <= after for _, t_start in flight.values()):
            stream.collect(1, timeout=timeout)           # the games older than the last stamp end first
            flight = stream.wait_in_flight(timeout=120.0)
        targets = {g for g, _ in flight.values()}
        publish()
        t_pub = stream.publications()[-1]
        games = []
        while not targets <= {g.index for g in games}:
            assert len(games) < 4 * len(targets), \
                f"the games in flight at the publication ({sorted(targets)}) must complete"
            games.extend(stream.collect(1, timeout=timeout).games)
        if any(g.index in targets and g.t_end > t_pub for g in games):
            return games, targets, t_pub
        after = t_pub
    raise AssertionError(f"no game in flight outlived any of {attempts} publications")


def assert_publication_straddled(games, targets, t_pub) -> None:
    """A game inside which the publication fell counts it once; a game
    that started after it counts nothing. A game that ended in the
    instant between the wait's return and the stamp is not judged."""
    checked = 0
    for g in games:
        if g.index in targets and g.t_end > t_pub:
            assert g.straddled == 1, f"game {g.index} was in flight at the publication"
            checked += 1
        elif g.index not in targets and g.t_start > t_pub:
            assert g.straddled == 0, f"game {g.index} started after the publication"
    timings = [(g.index, round(g.t_start - t_pub, 3), round(g.t_end - t_pub, 3), g.straddled)
               for g in games]
    assert checked >= 1, ("no in-flight game outlived the publication: "
                          f"targets {sorted(targets)}, (index, start, end, straddled) "
                          f"relative to the stamp {timings}")
