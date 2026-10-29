# Setting up a new machine

The repository holds the code and the records. The private Hugging Face
repository `momom2/wesnoth-model-checkpoints` holds the checkpoints, the
replay corpus and the match records. The files that lived only on the
first laptop were uploaded on 2026-10-27 to `tier-b/handover_20261027/`
in that repository; its `MANIFEST.json` lists every file and where it
came from.

## Quickstart

Steps 1 to 3 come before the first Claude Code session that touches a key.

1. **Clone and install.** Follow README's Quickstart, steps 1 and 2.

2. **Credentials.** Enter each key in your own terminal, never through
   Claude:

   ```bash
   huggingface-cli login
   vastai set api-key <key>
   gh auth login
   ```

   The Hugging Face token needs write access to the checkpoint repository
   only. The Vast key needs offers and instances only. `vastai` stores its
   key in `~/.config/vastai/vast_api_key`.

   Found on the Windows desktop, 2026-10-28:
   - huggingface_hub 1.x and later name the CLI `hf`: `hf auth login`,
     `hf download`. `huggingface-cli` is gone (hf 2.3.0).
   - `gh` is not installed with Git for Windows: `winget install GitHub.cli`.
   - Git's Credential Manager signs in through the browser, as whichever
     GitHub account the browser is signed in to. Sign the browser in as
     the repository's owner first, or pick the device-code option in its
     window. A refused push stores nothing, so the next push asks again.
   - `vastai` (unpinned in requirements.txt) pulls in 37 packages,
     `requests`, `urllib3` and `pillow` among them. On the desktop it lives
     in a venv of its own, `%USERPROFILE%\venvs\wesnoth_vast` (`vastai`
     and `huggingface_hub`, 2026-10-29), which runs `scripts/rent_box.py`;
     the key goes in with that venv's `vastai set api-key`.

3. **Secret guard.** It redacts credentials from everything a tool
   returns to Claude:

   ```bash
   python tools/secret_guard/install.py --write-settings
   ```

   Re-enter by hand any value the old machine listed in
   `~/.claude/secret_guard/extra_secrets.txt`. That file holds
   credentials, so it was not uploaded.

4. **Claude's state.** Download the archive:

   ```bash
   huggingface-cli download momom2/wesnoth-model-checkpoints tier-b/handover_20261027/claude_state.tar.gz --local-dir handover
   ```

   Then put its contents in place:

   | In the archive | Goes to |
   |---|---|
   | `memory/` | `~/.claude/projects/<key>/memory/`. Open one Claude Code session in the repository first: it creates the `<key>` folder, named after the repository's path. |
   | `home_claude/CLAUDE.md` | `~/.claude/CLAUDE.md`, the user-wide instructions. |
   | `home_claude/settings.json` | Copy its other entries, such as `autoMode`, into `~/.claude/settings.json`. Step 3 already wrote the guard's entries for this machine's paths. |
   | `repo_claude/settings.local.json` | `.claude/settings.local.json`, the project's permission allowlist. |
   | `scratchpad/` | The monitorability board's generator (`board/`), the pytest lock wrapper for parallel agents (`pytest_locked.py`) and the handover scripts. |

   The board itself is a claude.ai artifact, reachable from any device:
   https://claude.ai/artifact/7AsxHBu6CN1DhrQY3M2qp2.

5. **Rust core.** On Windows, Defender's ASR rules block new unsigned
   executables in an unexcluded folder. Exclude the cargo target folder
   from an administrator PowerShell:

   ```powershell
   Add-MpPreference -ExclusionPath "$env:USERPROFILE\.cargo-target"
   ```

   Then build and install the wheel with the command in CLAUDE.md,
   Testing. `python tools/kernel_status.py` should then read "current".

   A Windows machine without Visual Studio has no MSVC linker. The GNU
   toolchain links on its own (rustc 1.99.1, maturin 1.15.0 on 2026-10-28):

   ```bash
   rustup default stable-x86_64-pc-windows-gnu
   cd rust/wesnoth_core
   CARGO_TARGET_DIR="$USERPROFILE/.cargo-target" maturin build --release -i python --out "$USERPROFILE/.cargo-target/wheels"
   pip install --force-reinstall --no-deps "$USERPROFILE"/.cargo-target/wheels/wesnoth_core-*.whl
   ```

6. **Wesnoth** is needed only for the live tools and the engine oracles.
   - Install 1.18.x from Steam.
   - `python main.py --check-setup` links the add-on into Wesnoth's user
     data.
   - Copy the rest of `wesnoth_src/data/` from the install with the
     robocopy command in CLAUDE.md, "Wesnoth data provenance".
   - The patched build comes from CI: see `tools/wesnoth_build/README.md`.

7. **Local data.** Download `records.tar.gz` and `datasets.tar.gz` the
   same way as step 4, and unpack both at the repository root: their
   paths are relative to it.
   - `records.tar.gz`: the old logs and records (`training/logs/`,
     `eval_games/`, `logs/`, Kaggle outputs). It includes the replay
     dispositions ledger, `training/logs/replay_dispositions.jsonl.gz`,
     whose builder is in no commit.
   - `datasets.tar.gz`: the replay datasets built on the laptop
     (`replays_dataset_imitation/` and its duplicates and quarantine,
     `replays_dataset/`). The tests that read replay data use them.
   - **Checkpoints.** `python tools/reference_player.py --ensure` fetches
     the reference player. Nine older checkpoints that existed only on
     the laptop are under `tier-b/handover_20261027/training/checkpoints/`
     and `tier-b/handover_20261027/kaggle/`.
   - **Raw replays.** `replays_raw/` (37 GB, 470,206 files to
     2026-08-06) is the full download. `python tools/download_replays.py
     2024-03-18 2026-08-06` fetches it again, since replays.wesnoth.org
     serves every day since 2024-03-18 (three days compared on
     2026-10-27, the server holding as many files as the laptop or a few
     more). The corpus every pipeline reads is
     `tier-b/corpus_v3/raw_corpus_20260929.tar`.

8. **An AMD GPU on Windows** (the desktop's RX 6600, gfx1032; records in
   `training/metrics/local_gpu_20261028/`). Put the GPU build of torch in
   a second venv outside the repository, and keep the CPU `.venv`: the
   ROCm build's CPU kernels are much slower.

   ```bash
   python -m venv "$USERPROFILE/venvs/wesnoth_rocm"
   "$USERPROFILE/venvs/wesnoth_rocm/Scripts/python" -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "torch[device-gfx1032]==2.14.0+rocm10.1.0"
   ```

   Then install the repository's requirements and the Rust wheel into it.
   - **Pick the GPU.** Set `HIP_VISIBLE_DEVICES` to its index: 1 there,
     since device 0 is the CPU's integrated GPU.
   - **Kernels.** ROCm on Windows has no fused attention for RDNA2.
     `wesnoth_ai/hip_kernels.py` compiles its own kernels at first use,
     with the hiprtc that ships in the wheel.
   - **Matches.** Pass `--infer-fp16` with `--shared-inference`: this card
     runs bf16 matmuls at a fifth of fp16's rate.
   - **Stage 1** runs with `scripts/selfplay_t1_local.py`.
   - **Strength matches** run with `scripts/match_local.py` (README,
     Quickstart, "On the local Windows machine").
   - **The display.** The card drives the screen, and a GPU stall past 2 s
     resets the display driver (System event 4101). Both scripts stop on
     one.

Copy by hand anything else you want: the Claude Code session transcripts
(`~/.claude/projects/<key>/*.jsonl`, 1.6 GB) and the Kaggle credential
(`kaggle/kaggle_token.txt`).
