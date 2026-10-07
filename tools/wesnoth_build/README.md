# Patched Wesnoth build

Wesnoth 1.18.8 with the project's patches, built for Windows by GitHub
Actions (`.github/workflows/patched-wesnoth.yml`) with Wesnoth's own MinGW
cross-build, the build behind the official Windows installer.

## Quickstart

1. Run the workflow: push a change under `tools/wesnoth_build/`, or run it
   by hand.

       gh workflow run patched-wesnoth.yml --ref <branch>

2. Download its artifact into the games folder, which is where
   `wesnoth_ai/constants.py` looks for it.

       gh run download <run id> -n wesnoth-1.18.8-patched-win64 -D "%USERPROFILE%\Desktop\Perso\games\wesnoth-1.18.8-patched"

3. Check it: `python main.py --check-setup` names the executable in use.
   `BUILD_INFO.txt`, next to `wesnoth.exe`, gives the upstream commit, the
   run that built it, and the checksums of the patches and executables.

The build uses the user data of the Steam install (`Documents/My Games/
Wesnoth1.18`): same preferences, add-ons and saves. Setting `WESNOTH_EXE`
selects another executable.

## Patches

- `commandline_game_settings_1.18.8.patch`: a game started with
  `--multiplayer` gets the experience modifier, village gold, village
  support, fog, shroud and random start time that a game created in the
  lobby gets: the scenario's own values, the lobby's defaults otherwise.
  The stock build writes none of them into the scenario, so its
  command-line games play 100% experience and 1 gold per village
  (docs/wesnoth_rules.md, "A command-line `--multiplayer` start skips the
  lobby's parameter writes"). An upstream pull request for `master` is
  drafted, not posted.
