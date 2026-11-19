# Pre-registration: the loop's network against Wesnoth's default AI (2026-11-19)

**Status:** registered 2026-11-19. The trial game was won at 18:19 that
day and the 100 games started at 18:19:53: see "Results".

## Why

The user, 2026-11-19: "Let's see how it does against RCA AI."

Every strength number so far is against `own2`, a network of our own. This
is the first reading against an opponent from outside the project. The
default (RCA) AI is weak by players' standards, so the reading is a floor:
a network that does not beat it nearly always is far from a strong
player, and one that does has shown little more than that.

## What runs

- **The player:** the simple loop's network after round 128
  (`wesnoth_runs\simple_loop\round_0128\pi.pt`, SHA-256
  `6bdcae0947c9e81514e4ad62d7efaa0768a671e789be8e844319bea53b39d3aa`), the
  last one a match has read (+107.4 ± 12.9 against `own2` when sampling,
  `docs/simple_loop_prereg_20261117.md`). It plays at the reference's
  decode (`raw:t0+eo-1.5`) with 64 memory slots, on the CPU.
- **The opponent:** the engine's default AI, with no configuration of
  ours.
- **The games:** `tools/live_vs_rca.py --checkpoint <that file> --label
  s0128 --unwatched --seed 20261119`, from `feature/live-rca-checkpoint`
  at 156901ee (the `--checkpoint` option; `main` could only play the
  reference). Each game is a live Wesnoth game, one at a time, in a
  minimized window. It draws a map of the Ladder pool and both factions
  uniformly (`random_setup(category="ladder")`), with the settings a game
  hosted in the lobby gets, and our side alternates.
- **First one trial game,** to check the bridge on this machine. Then
  **100 games.**
- **Beside the training run:** on the CPU at Idle priority, 4 threads at
  most. The GPU stays the run's.

## What is measured

- **The win rate over the games that finish,** with its binomial standard
  error, and the count of each outcome: win, loss, draw (the turn limit)
  and stopped.
- **A stopped game is a fidelity defect, not a result.** The tool compares
  the engine's board with the simulator's at every decision and stops the
  game on a difference (Principle 4). Each one is recorded and its root
  cause looked for.
- **No Elo is fitted and nothing is adopted.** This is not a strength
  claim in the sense of CLAUDE.md, "Strength claims".

## The number that decides

Whether the default AI can serve as an outside yardstick from here on:

- **At 95 wins or more of 100 finished games** it is saturated, and it
  cannot track further progress. An outside yardstick then needs a
  stronger opponent or a handicap, which is the user's to choose.
- **Under 95,** it stays in use: the same 100 games are replayed for a
  later network, and the losses' records are read for what they have in
  common.

## Kill criteria

1. **The trial game stops at its first decision** (the engine plays
   another experience modifier): the build is wrong. Nothing more is
   launched.
2. **Three stopped games in the first ten:** the series ends there and the
   difference is investigated first.
3. **A game past 30 minutes** is inspected.
4. **The training run slows by more than 10%** in decisions a second over
   a pass, or the display driver resets (System event 4101): the games
   end.

## The estimate

- A decision takes 0.06 s on the CPU beside the run (120 decisions at Idle
  priority on 4 threads, 2026-11-19), so about 15 s of a game is the
  network's.
- The engine's share is not known before the trial game. The guess is 2 to
  5 minutes a game, 3 to 8 hours for the 100. Past 1.5 times the trial
  game's rate, the series is inspected.

## What blocked the launch, and how it was lifted

- **The patched build was not on the desktop.**
  `Desktop\Perso\games\wesnoth-1.18.8-patched\wesnoth.exe` did not exist,
  and no copy was found under the user's folder or on `D:`. With the Steam
  build the tool refuses to play, by design: a command-line game there
  plays 100% experience where a hosted game plays 70%.
- **GitHub no longer held it.** The workflow last built it on 2026-10-07
  (run 37600290795, on `main`), and its artifact was kept 30 days.
- **It was rebuilt on 2026-11-19** by run 44551964120, from
  `fix/patched-wesnoth-retention` at c3541c94 (the same patch; the
  artifact is now kept 90 days). The user signed in to GitHub's CLI, and
  the artifact (458 MB) was downloaded at 18:14: 20,827 files, 582 MB.
  `wesnoth.exe`'s SHA-256 is the one in the build's `BUILD_INFO.txt`
  (`429028a6…8859e`).
- **The add-on link** was made by `python main.py --check-setup`, run from
  `..\wesnoth_ai-rca`: `Documents\My Games\Wesnoth1.18\data\add-ons\wesnoth_ai`
  points at that worktree's `add-ons\wesnoth_ai`.

## The launch

- **The launcher** is `wesnoth_runs\rca_games\play_s0128.cmd <games>
  <records directory>`. It sets `OMP_NUM_THREADS` and `MKL_NUM_THREADS` to
  4 and starts the tool with `start /LOW`, so the tool and the Wesnoth it
  starts both run at Idle priority.
- **The records** go to `wesnoth_runs\rca_games\<records directory>`, and
  the console to the file of that name with `.console.log`.

## Departures

1. **Kill criterion 4 is read over three passes or more** (written
   2026-11-19 18:21, before any such reading). One pass's rate varies on
   its own: over rounds 97 to 144 it has a median of 67.2 decisions a
   second and a standard deviation of 4.1 (51.4 to 73.5). The games end
   if the mean over the passes beside them, three at least, falls under
   60.5.
2. **The number that decides is read as a rate: 95% or more of the
   finished games** (written 2026-11-19 18:55, at 9 wins of 9 finished
   games). "95 wins of 100 finished games" assumed all 100 would finish.
   They will not: an Aethermaw game cannot get past turn 4 on this bridge
   (Results, "Game 6"), and Aethermaw is one map in 21.

## Results

Filled in as the games go.

- **The trial game** (2026-11-19 18:15 to 18:19,
  `wesnoth_runs\rca_games\trial_s0128`): a win on turn 31, on Swamp of
  Dread as Undead on side 1 against Rebels, in 361 decisions and 4 min
  1 s.
  - The engine's board and the simulator's agreed at every decision. The
    check let through 247 `unit.moves` differences of the kind it
    understands: units the default AI stopped, whose movement the engine
    zeroes outside the replay.
  - The engine reported twice that it could not open the action file: a
    read caught during the file's replacement, which is tried again at
    the next poll (`lua/live_stage.lua`).
  - The training pass it ran beside (round 145's, groups 4 to 7) read
    67.0 decisions a second, against the median of 67.2.
- **The 100 games** started at 18:19:53
  (`wesnoth_runs\rca_games\games_s0128`), with a hidden console. At the
  trial game's rate they take about 7 hours.
- **After 10 games** (18:52): 9 wins and 1 stopped, no loss.
  - The nine wins came on turns 12 to 18, on eight maps, five as side 1
    and four as side 2. A finished game takes about 2.5 minutes, so the
    series should end near 23:00.
  - The boards agreed at every decision of the nine. The check let
    through 32 to 269 `unit.moves` differences a game, all of the kind it
    understands.
- **Game 6 stopped on a dialog, not on a board difference.** It is
  Aethermaw (Knalgan against Knalgan, our side 2), and it ended with "no
  frame in 600 s" after 22 decisions.
  - The engine's log ends at "changing current turn number from 3 to 4".
    Aethermaw's `side 1 turn 4` event opens with a narrator `[message]`,
    "The Aethermaw growls."
    (`wesnoth_src/data/multiplayer/scenarios/2p_Aethermaw.cfg:329`), and
    nobody is there to click it.
  - That the dialog is what held the game is read from where the log
    ends; the engine's code for it has not been checked.
  - Aethermaw is the only map of the 21 whose scenario has a `[message]`.
    Each of its games will stop this way, about 5 of the 100, at 10
    minutes each.
  - The bridge is not changed during the series: its add-on is the one
    the running games load.
- **After 40 games** (20:06): 38 wins, 1 loss and 1 stopped. That is 38
  of 39 finished games, 97.4% ± 2.5%.
  - The wins came on turns 9 to 19 (median 14), on 16 maps, 20 as side 1
    and 18 as side 2.
  - The boards agreed at every decision of the 39. The only differences
    let through are `unit.moves` ones, 32 to 378 a game.
  - No game has stopped since game 6. Aethermaw has been drawn once in
    40.
  - Forty games took 106 minutes, so the series should still end near
    23:00.
- **Game 20 is the loss:** Swamp of Dread, Undead against Undead, our
  side 2, our leader a Dark Sorcerer, killed on turn 18.
  - The default AI recruited 29 units and ours 17 (10 Dark Adepts, 5
    Ghosts, a Walking Corpse and a Vampire Bat).
  - Our leader fought away from its keep from turn 5 (an attack from five
    hexes off it), was attacked on six of the turns from 9 on, and fell
    at second watch to a Ghoul and a Skeleton Archer, off its keep.
  - This is one game's record, read from its commands with the leader
    followed by its position. Nothing is concluded from it.
- **After 73 games** (21:20): 67 wins, 2 losses and 4 stopped. That is 67
  of 69 finished games, 97.1% ± 2.0%.
  - By the faction our side drew: Undead 7 wins of 9, the five others 60
    of 60.
  - The finished games cover 20 of the 21 maps.
- **Game 48 is the second loss:** Thousand Stings Garrison, our Undead on
  side 2 (leader a Deathblade) against Loyalists. Our leader was killed
  on turn 8, in the morning, by a Mage and a Heavy Infantryman, on a hex
  it had not recruited from.
  - Both losses are Undead games in which our leader died away from its
    keep. Two games are not a pattern; the records are there to read if
    more come.
- **Games 49 and 51 stopped on a difference the comparison made
  itself.** In both, a Walking Corpse of ours killed a unit by plague (a
  Saurian Skirmisher, a Merman Hunter), and the corpse it raised read
  `Walking Corpse` on the engine's board and `Walking Corpse:saurian`
  (`:swimmer`) on the simulator's.
  - The engine holds a unit's variation beside its type, and the board
    report sent the type alone. The simulator puts the variation in the
    type's name.
  - The two units were the same. Every other field agreed, and in game 49
    that includes the corpse's 16 hit points, the saurian variation's (a
    plain corpse has 18). The engine's plague code asks for the victim's
    variation (`src/actions/attack.cpp:1298-1306`), as the simulator
    does.
  - So the simulator is not at fault here. The check was blind to one
    field, and it stopped two games it should have let through.
- **Game 50 is the second Aethermaw game,** stopped at turn 4 like game
  6.
- **Of the first four stopped games none is a rules mismatch:** two are
  the dialog, two the comparison's missing field. (Game 75, below, is
  the first that is the simulator's.)
- **The sound.** The user, 2026-11-19: "It's minimized, but sound is at
  full power." The games were started with no sound switch. The running
  series is left as it is, on the user's word.
- **The fixes** are on `fix/live-unattended` at 1b740064, cut from
  `feature/live-rca-checkpoint`, pushed, not merged:
  - the board report sends the variation and the comparison has a
    `unit.variation` field;
  - an unwatched game starts with `--nosound`, and the launcher puts the
    user's preferences file back afterwards, since that switch turns the
    sound preferences off and Wesnoth saves them as it exits;
  - in an unwatched game each scenario message sets its event's
    skip flag first, the one Escape sets, so the dialog is passed over;
  - `--scenario` plays one map every game, to check a map.
  The rules are in that branch's `docs/wesnoth_rules.md`. Its tests pass
  on the desktop (17). The Lua has not run in an engine yet: that check
  waits for the series to end, because the running games load the add-on.
- **After 84 games** (21:42): 76 wins, 2 losses and 6 stopped, 76 of 78
  finished games.
- **Game 82 stopped like games 49 and 51:** our Walking Corpse killed a
  Saurian Augur and the corpse's variation was the only difference.
- **Game 75 stopped on a defect of the simulator.** It is Hellhole, our
  Knalgan on side 1 against Loyalists. At decision 54 of turn 7 the
  Gryphon Rider we had just recruited read `uncovered` on the
  simulator's board and not on the engine's.
  - The simulator keeps the engine's uncovered state as a list of unit
    ids, cleared at a side's turn start for its living units. A unit that
    dies while uncovered stays in the list.
  - A new unit is numbered one past the largest id on the board. When the
    dead unit held the largest id, the next recruit takes it and reads
    as uncovered.
  - The game's record replayed in the simulator shows it: our Dwarvish
    Ulfserker `u18` attacked on turn 7 (an attack uncovers its attacker),
    died in that fight, and the recruit that followed was numbered `u18`.
  - What it changes in play: nothing for a unit with no hiding ability,
    since the state is only read for hiders. A hider recruited onto such
    an id would be seen through its cover until its side's next turn.
    This is read from the code, not measured.
  - The fix is one line (the id leaves the list when the unit leaves the
    board), on `fix/uncovered-dead-id` at 84aeebff, cut from `main`,
    pushed, not merged. Its test fails on the wheel built before the fix,
    as it should. No wheel can be installed beside the training run, so
    the fixed Rust is for CI to certify.
- **CI is red on every branch pushed on 2026-11-19,** for two failures
  the branches did not cause: the runner's shellcheck reports SC2329 on
  the box scripts, and `tests/test_boxlib.py::
  test_a_watched_step_ends_when_silent_or_once_its_line_appears` reads an
  exit status of 124 where it expects 143. Both also failed on
  `fix/patched-wesnoth-retention`, which changed one workflow file. On
  `fix/live-unattended` the rest of the suite passed (2263 passed, 119
  skipped). The last green run is `main`'s of 2026-10-27.
- **The training run beside the games:** its passes read 67.0, 66.8,
  65.1, 67.9, 61.3, 67.0, 67.5, 68.9, 66.0, 72.6 and 70.0 decisions a
  second (rounds 145 to 155), a mean of 67.3 against the median of 67.2
  and the limit of 60.5. No display driver reset is logged.
