# Overnight process death on the Modern Standby laptop — corrected fix design

Status: design corrected after adversarial review of F1–F4. Nothing below is confirmed by a live overnight test yet; §4 lists what still needs one.

## 1. Corrected mechanism story

Three separate mechanisms were conflated. Only one of them is even capable of killing a process.

**(a) The nightly hibernates were DC policy working correctly — not an AC-policy violation.**
`powercfg /q SCHEME_CURRENT SUB_SLEEP` (verified 2026-08-10): `STANDBYIDLE` AC=`0x0`, DC=`0xb4` (180 s); `HIBERNATEIDLE` AC=`0x0`, DC=`0x5460` (21600 s = 6 h). Every cited event-42 (`TargetState=5 EffectiveState=5 Reason=6` — "Hibernate from Sleep — Fixed Timeout", genuine S4) fired ~6 h after a Kernel-Power 105 `AcOnline=false`:

| AC lost | Hibernate | Δ |
|---|---|---|
| 08/03 01:27:47 | 08/03 07:27:54 | 6 h 00 m |
| 08/06 23:01:57 | 08/07 05:05:24 | 6 h 03 m |
| 08/02 02:00:00 | 08/02 08:01:36 | 6 h 01 m |

The machine was **on battery**. The quoted AC values never applied. The 105 log also shows the charger flapping AC true/false repeatedly in the evenings — a loose connector/dock is the precondition for all three.

**(b) Modern Standby entry is display-off-driven, not idle-timer-driven.** MS enters when the display turns off (idle-out, lid, power button). On DC the 180 s `STANDBYIDLE` gets there fast; the AC "never" values do not gate S0 entry at all. This is why 506/507 appear despite sleep-after=0.

**(c) DAM suspends, it does not kill.** In MS, the Desktop Activity Moderator puts session-1+ processes in a job object subject to suspension — all threads frozen, **process memory preserved, same PID on resume**. DAM fully explains an overnight *stall* (zero progress, wall-clock/uptime skew). It cannot explain "the process dies or orphans".

**So the likely proximate killer is not DAM.** Two live candidates, in order: (i) the S4 hibernate in (a) not resuming cleanly / being followed by a cold boot; (ii) **connection-lifetime death across the freeze** — the CLI's TLS/websocket to the API and any SSH sessions to the rented box are torn down by peer/NAT over hours; the client exits on resume and its watcher children orphan. Mechanism (ii) is consistent with everything observed and requires no hibernate at all.

## 2. Fixes that survive review

Use alias GUIDs, not localized names (French locale).

**2.1 Correct the DC timers (this is the real F1 lever).**
```
powercfg /setdcvalueindex SCHEME_CURRENT SUB_SLEEP HIBERNATEIDLE 0
powercfg /setdcvalueindex SCHEME_CURRENT SUB_SLEEP STANDBYIDLE 0
powercfg /setactive SCHEME_CURRENT
```

**2.2 Fix the AC supply, physically.** Verified-stable charger/dock before any overnight run; confirm with no `AcOnline=false` in Kernel-Power 105 across the run window.

**2.3 Keep hibernation ENABLED.** `BATACTIONCRIT` = 2 (hibernate) on **both** AC and DC with `BATLEVELCRIT`=2%. Disabling it removes the only graceful low-battery save path and kills Fast Startup, with no S3 to fall back to (`powercfg /a`: S1/S2/S3 all unavailable under S0 low-power idle).

**2.4 If a laptop-side process must survive at all, keep the display on.** Display-off is the documented MS trigger:
```
powercfg /change monitor-timeout-ac 0
```
Accept a lit screen. **Lid close still voids it** (user-initiated sleep terminates all power requests).

**2.5 F4's architecture, with four amendments** — this is the actual fix for the reported problem:
- **Periodic escrow, not completion-chained.** 30 min is the established cadence (`docs/tier_b_runbook.md`, `hf_upload_loop.py`). Completion markers are an *additional* signal, never the primary one.
- **Timestamped heartbeat** so session-start reconstruction can distinguish finished / died / still-running. The HF `trainer_history_local.csv` already is this channel; designate it.
- **Box-side watchdog must be able to end billing.** Its terminal action stops/destroys its own instance (restricted `instance_api_key`, or on-box `vastai stop instance`), **plus** an independent hard wall-clock self-destruct as backstop. Vast has no scheduled auto-stop; without this, F4 converts "run dies silently" into "run dies *and burns credit* silently".
- **Stall threshold is a parameter**, derived from the run's observed logging cadence, not a hardcoded 20 min.

## 3. Fixes that were wrong

| Proposal | Verdict | Replacement |
|---|---|---|
| **F1** `powercfg /h off` | Prediction right, reasoning and instrument wrong. It would stop the event-42s (S4 becomes unavailable) but the premise "hibernate-after=never yet it hibernates" is false — those were DC timers. With hibernation gone the machine still enters MS at 180 s on DC and now drains to an ungraceful shutdown instead of a resumable hibernate. | §2.1 + §2.2 + §2.3. `/h off` only as a blunt fallback *if* an event 42 still fires ~6 h after AC loss with `HIBERNATEIDLE`-DC=0. |
| **F2** helper process holding `SetThreadExecutionState(ES_CONTINUOUS\|ES_SYSTEM_REQUIRED)` | Refuted three ways. Wrong trigger (ES_SYSTEM_REQUIRED resets the *system idle timer*; MS entry is display-off). Wrong layer (no documented exemption from DAM once standby begins; MS's own remedy is *notification*, `RegisterSuspendResumeNotification`). Wrong architecture (`PowerRequestExecutionRequired` is scoped to **the calling process** — a helper cannot shield the session or its children). Also: `ES_AWAYMODE_REQUIRED` is S3-only, dead here. | §2.4 (prevent display-off) and/or **each** long-running process holding its own `PowerCreateRequest` + `PowerSetRequest(PowerRequestExecutionRequired)`. Describe the effect as "keeps the *system* out of MS — AC only, void on lid close", never as "exempts apps from S0 suspension". |
| **F3** "DAM suspension is what kills the session" | Mechanism half correct, causal half wrong. Suspension preserves state and PID. Also a category error: event 42 is a kernel power-transition record, not a DAM signal. And DAM engages at *every* screen-off, which does not match a failure localized to 05:05/07:27/08:01. | Split it: keep "DAM freezes session-1+ processes ⇒ no overnight progress". Drop "kills". Attribute death to hibernate-without-clean-resume or to socket/handshake teardown across the freeze. |
| **F4** "laptop-side watchers only for sub-hour horizons" | Wrong axis. MS entry is state-driven, not time-driven: a 40-min watcher on an idle machine dies exactly like an 8-h one, while on AC a held power request blocks the No-CS phase indefinitely and can carry a 12-h watcher. | State-based rule: laptop-side watchers are valid only while **display on + lid open + (on AC) a power request held**, and are **never the sole custodian** of anything not reconstructible from artifacts, at any horizon. Best-effort, never load-bearing. |

## 4. Residual uncertainty — needs a live overnight test

1. **The discriminating test (cheap, run first).** Record the session PID, leave the machine idle overnight on stable AC, check after resume:
   - PID alive, connections dead → mechanism (ii); F3's "kills" is simply wrong and the fix is reconnect/resume handling, not power config.
   - PID gone with a normal resume → neither DAM nor a clean hibernate explains it; look for a non-resuming hibernate, a cold boot, or an OOM/terminator.
2. **Does `PowerRequestExecutionRequired` actually exempt a session-1 process from DAM's suspension job on 24H2?** MS docs say "PLM"; DAM is a distinct component. Only positive evidence is one single-machine report (PowerToys #48965). Treat as unverified until measured.
3. **Power-request lifetime in practice**: terminated on lid close / power button / Start-menu Sleep; terminated after 5 min on DC. Untested whether the request survives workstation lock here.
4. **Whether AC is genuinely stable overnight** after the connector fix — verify by absence of Kernel-Power 105 `AcOnline=false`.
5. **Whether §2.1 alone removes the event-42s**, or an override outside the power scheme exists.
6. **Wake timers**: `RTCWAKE` AC=1, DC=0; wake-timer reliability on MS is contested in the field. Do not build a laptop-side scheduled re-arm on it without measuring.
7. **Stall-watchdog threshold**: 20 min is plausible but unverified against the current imitation arms' logging cadence.

Instrumentation to collect alongside the test: `powercfg /sleepstudy` (Screen Off vs Sleep segments, blockers), `powercfg /requests` while the session runs, and the **full French message text + TargetState/EffectiveState** of any event 42 (the numeric Reason field is community-decoded, not authoritative).