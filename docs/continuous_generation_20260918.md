# Continuous generation on the actor pool (2026-09-18)

User order (2026-09-18): "Measure form A if you want, but work on
implementing form B." Form A is more games than actors per iteration,
which the pool already runs (`--games-per-iter` above `--actors`);
form B is this: no iteration barrier at all.

## The tail

An iteration hands every actor one game and ends with the longest
one. Games finish around a median of 200 s while the longest takes
300-430 s, so for the second half of every iteration fewer than half
the actors are alive and the server starves. The gap between the
iteration rate and the saturated rate is that tail:

| host | iteration leaves/s | saturated leaves/s | tail cost |
|---|---|---|---|
| post-review box, 24 cores (2026-09-14) | 1,288 | 1,759 | 1.37x |
| shared 128-thread host (2026-09-14) | 905 | 1,744 | 1.93x |
| Core Ultra 9 285K, pair a (2026-09-18) | 1,434 | 2,634 | 1.84x |
| Core Ultra 9 285K, pair b (2026-09-18) | 1,706 | 2,728 | 1.60x |

Nothing else left in the generation path is that large, and it costs
no compute.

## The design

`tools/actor_stream.py`, on the started pool (`ActorPool.stream()`):

- **Tickets.** The game queue is kept topped up: one game in hand per
  live actor and `tickets_ahead` (one per actor by default) waiting,
  one new ticket posted per completed game. Game indices are global
  to the stream, so every game's seed is its own.
- **Windows.** `collect(G)` blocks until G more games have completed
  and returns them, serving going on underneath. The pool's `last_*`
  readbacks describe the window's serving: the serve threads register
  their live stats dicts at their start now, and a window is the
  difference between two snapshots (the serve processes answer a
  STATS command without stopping).
- **Publication.** After the learner's step, `publish()` ships the
  weights to the serve processes and the new value center and anneal
  counter to the actors (an UPDATE command they apply between games),
  and dates the publication. The in-process server already serves the
  new snapshot: every load into the inference copies (the policy's
  own snapshot after a step, step control's trial publishes) goes
  through the server's `ServeGate`, whose exclusive side waits for
  the batches in flight and excludes the next ones. A batch never
  forwards through half a state_dict. The serve processes load under
  their own gate. During a backtracking step's trials the servers
  serve each candidate for the seconds of its held-out evaluation;
  with `--step-select first` and a full step passing, that is the
  final weights at once.
- **Straddling.** A game that started before a publication and ended
  after it lived through it. Its experiences mix the two policies'
  decisions; its label is the game's outcome, as always. Each game
  reports the count; each window its mean, maximum and the share of
  games that straddled at all. With as many actors as games per window
  a game straddles about one publication on average; with twice as
  many actors, about two. That ratio is the learner-side knob.
- **Stopping.** `stop()` sends DRAIN: every actor finishes the game in
  hand and reports done; the games that complete meanwhile are
  returned. Past the grace the rest are abandoned and logged.
- **Failures.** A serve process that dies or fails a command aborts
  the stream as it aborts an iteration; an actor killed before its
  `finally` aborts loudly; one that exits clean is dropped and the
  window says so. A window past its timeout returns what it has when
  that is at least `min_games`, and raises otherwise: a stream on which
  no game completes for that long is broken.

The learner (`tools/az_loop.py --stream`) collects a window, steps on
it exactly as on an iteration, computes the value center, publishes.
Three columns record the regime (`straddle_mean`, `straddle_max`,
`straddled_share`) and one what the in-process server served while the
learner stepped (`step_leaves_per_s`: the step's Python competes with
the serve threads for the GIL and the GPU). `tools/bench_pool.py
--stream --rounds R --step-seconds S` is the throughput instrument:
R windows of `--games` games with S seconds of idling between them in
place of a step, then the drain.

The barrier iteration is unchanged and stays the default; the two
modes share the serving threads, the stats and the liveness scan.

Rejected: a trainer process of its own, so the step's Python never
touches the serve threads' GIL. Measure `step_leaves_per_s` first; the
step is 4% of an iteration, and if serving holds through it the
separate process buys nothing.

## The measurement (pre-registered)

`scripts/stream_box.sh`, on a single-tenant 4090 host of the
2026-09-18 class, the committed bf16 packed configuration, 32
evaluations, 48 actors:

| arm | what | games |
|---|---|---|
| barrier | `bench_pool --actors 48 --games 48` | 48 |
| form A | `bench_pool --actors 48 --games 96` | 96 |
| stream | `bench_pool --actors 48 --games 48 --stream --rounds 4 --step-seconds 20` | 192 plus the drain |

Twice each, interleaved. The number is games per hour: the stream's
over its windows, each plus the 20 s idle that stands in for a step
(the drain left out, a campaign pays it once); the barrier's over its
iteration plus the same 20 s, since its step runs with the actors
idle. The saturated rate is the roof. Expectation: the stream's games
per hour reads within 15% of the barrier's saturated rate divided by
the barrier's leaves per game; form A sits between. Kill: the stream
under 1.25x the barrier on games per hour, or a straddle mean outside
0.7-1.5 at 48 actors and 48 games per window, or any window timing
out.

What the run cannot decide is whether a learner minds the straddling.
That needs a learner that improves on the prior, run both ways, which
is phase 2's business; until then `--stream` is opt-in.

## Measured (2026-09-18, docs/box_specs.md "Continuous generation against the barrier")

FAIL under the rule above, by a hair in both pairs: pair 1 reads
1.302x on games per hour and a straddle mean of 0.69 over all four
windows (the first window precedes any publication, so the rule's own
averaging pulled it under 0.70); pair 2 reads a straddle of 0.70 and
1.248x. Steady windows 2-4: 1.49x and 1.39x at a straddle mean of
0.92-0.93, the server at its GPU roof. Form A: 1.32x and 1.10x.
`--stream` stays opt-in. A rule written again would exclude the first
window from the straddle mean and read the rate on the steady windows;
that is a different rule, and it was not registered.
