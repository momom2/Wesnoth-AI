# Cleanup and structure inventory (2026-09-25, commit 90ff321)

A read-only crawl of the tree at 0.6.7 for the refactor plan
(docs/refactor_plan_20260925.md) and the deletion candidates (BACKLOG.md
"Decision: deletions"). Figures are at commit 90ff321; the helper scripts
that produced them were one-offs and are not kept.

**Method.** I built an AST import graph over all 423 tracked `.py` files. It covers module-level and function-level imports and `importlib`. I tokenized every tracked text file for `tools/x.py`, `tools.x` and `x.py`, and counted references to every top-level def and method. I also scanned argparse flags and dataclass fields. "Live docs" means `docs/` outside `docs/archive`, plus CLAUDE.md, BACKLOG.md and README.md.

Short prefixes: `t.` = tools/, `ta.` = tools/analysis/, `w.` = wesnoth_ai/, `sp.` = signal_profiler/, `s.` = scripts/, `b.` = benchmarks/.

# (a) System map

The tree holds 221 production modules plus the package `__init__`: 97.7k lines of Python. Tests are 199 files and 38.2k lines. The Rust crate is 12 files and 4.2k lines.

| system | scope | modules | lines |
|---|---|---:|---:|
| RULES | unit and terrain data, WML, scenario building and its detectors | 10 | 6,980 |
| SIM | game state, rules engine, simulator, Rust-core adapter | 14 | 11,762 |
| CORPUS | replay download, filtering, extraction, datasets, replay export | 19 | 7,781 |
| FIDELITY | simulator against Wesnoth: oracles, diffs, strict-sync validation | 23 | 5,862 |
| MODEL | encoder, network, shared constants, unit vocabulary | 8 | 4,132 |
| INFER | inference seam, server priors, wire format, CUDA graphs, devices | 6 | 1,989 |
| SEARCH | action sampling, policies, MCTS, raw player, TCS and plan tournament | 13 | 9,872 |
| TRAIN | trainer, imitation training and pre-encoding, value-head tools | 15 | 7,964 |
| SELFPLAY | az_loop, actor pool, the legacy sim_self_play loop and its mechanisms | 14 | 11,004 |
| EVAL | match driver, per-game runner, ratings, catalog, reference player, phase-2 gap | 16 | 9,229 |
| BRIDGE | live Wesnoth: setup CLI, IPC, state conversion, RCA eval | 7 | 3,094 |
| BENCH | benchmarks and profilers | 18 | 4,736 |
| OPS | box and repo operations | 13 | 2,750 |
| PROBE | one-off analyses, probes, signal_profiler | 45 | 10,575 |

**Legend for the tables below.**
- **kind:** E = has a `__main__` block, L = imported by production code, EL = both, `-` = neither.
- **P/T:** number of production / test modules that import it.
- **named by:** sh = box shell scripts, doc = live docs, arch = docs/archive or quarantine/, cfg = configs, str = other Python files naming its path in a string, tsrc = test files naming its path.
- **Flags:** (Q) = a quarantined mechanism in quarantine/INVENTORY.md; (D) = a dead candidate.

### RULES (10 modules, 6980 lines)
| module | lines | kind | P/T | production importers | named by |
|---|---:|---|---|---|---|
| t.scenario_events | 2328 | L | 9/15 | t.neutral_ai, t.replay_builder, t.replay_dataset +6 | sh1 cfg doc3 arch1 tsrc1 |
| t.scrape_unit_stats | 1171 | E | 0/0 |  | doc1 str3 |
| t.scenario_pool | 896 | L | 39/56 | t.actor_worker, t.bench_infer, t.bench_pipeline +36 | cfg doc2 arch2 tsrc1 |
| t.terrain_resolver | 640 | L | 11/5 | w.game_core, w.state_converter, w.visibility +8 | sh1 doc1 |
| t.wml_state | 506 | L | 8/2 | t.check_replay_consistency, t.dump_savestate, t.replay_builder +5 | doc1 tsrc1 |
| t.build_scenario_templates | 408 | EL | 1/1 | t.scenario_init_oracle | doc1 str1 |
| t.scrape_terrain | 378 | E | 0/0 |  | doc1 str1 |
| ta.expansion_diff | 304 | EL | 2/3 | t.unit_vocab, ta.scenario_surface | doc3 |
| ta.scenario_surface | 259 | EL | 1/2 | t.unit_vocab | doc2 tsrc1 |
| t.scenarios | 90 | L | 5/0 | t.build_value_corpus, t.check_first_cmd_anomaly, t.diff_replay +2 | doc1 str1 |

### SIM (14 modules, 11762 lines)
| module | lines | kind | P/T | production importers | named by |
|---|---:|---|---|---|---|
| t.replay_dataset | 3081 | EL | 54/30 | w.action_sampler, w.encoder, w.game_core +51 | doc5 arch1 |
| t.wesnoth_sim | 1799 | EL | 44/41 | w.action_sampler, w.rewards, t.actor_worker +41 | doc5 arch5 |
| w.combat | 1035 | L | 10/4 | w.game_core, t.combat_outcomes, t.diff_combat_strike +7 | doc4 |
| t.combat_outcomes | 879 | L | 7/5 | t.gbc_rung0, t.luck_probe, t.mcts +4 | doc4 arch2 |
| w.game_core | 862 | L | 4/5 | t.bench_core, t.bench_leaf, t.diff_core +1 | doc5 |
| t.pathfind_sim | 760 | L | 12/12 | w.action_sampler, w.encoder, w.game_core +9 | doc3 str1 |
| w.visibility | 613 | L | 17/16 | w.action_sampler, w.encoder, w.gbc +14 | doc4 arch2 |
| w.classes | 609 | L | 43/37 | w.action_sampler, w.combat_oracle, w.dummy_policy +40 | doc2 arch1 tsrc1 |
| t.traits | 478 | L | 3/1 | t.replay_dataset, t.scenario_events, t.scenario_init_oracle | doc3 str1 tsrc1 |
| t.game_record | 434 | L | 6/3 | t.actor_worker, t.elo_eval_game, t.eval_sim +3 | doc2 str2 |
| w.observe | 373 | L | 4/4 | w.action_sampler, w.encoder, w.game_core +1 | doc2 |
| t.abilities | 327 | L | 12/13 | w.action_sampler, w.dummy_policy, w.observe +9 | doc3 |
| t.neutral_ai | 304 | L | 1/2 | t.wesnoth_sim | doc2 arch2 |
| t.engagement_stats | 208 | L | 3/1 | w.game_core, t.replay_dataset, t.wesnoth_sim | - |

### CORPUS (19 modules, 7781 lines)
| module | lines | kind | P/T | production importers | named by |
|---|---:|---|---|---|---|
| t.replay_extract | 2106 | EL | 13/15 | t.build_imitation_dataset, t.build_scenario_templates, t.build_value_corpus +10 | cfg doc2 arch1 str1 tsrc2 |
| t.sim_to_replay | 1566 | EL | 5/8 | t.ladder_anatomy, t.replay_builder, t.sim_demo_game +2 | doc3 tsrc2 |
| t.player_ratings | 513 | E | 0/1 |  | doc3 tsrc1 |
| t.replay_builder (D) | 407 | - | 0/0 |  | doc1 tsrc1 |
| t.sim_demo_game | 377 | E | 0/1 |  | doc2 arch1 str1 |
| t.sort_replays | 365 | E | 0/0 |  | - |
| t.build_value_corpus | 336 | E | 0/1 |  | tsrc1 |
| ta.corpus_census | 303 | E | 0/1 |  | doc4 |
| t.purge_mod_replays | 299 | E | 0/0 |  | doc1 |
| t.filter_replays | 255 | EL | 2/0 | t.build_value_corpus, t.player_ratings | - |
| t.build_imitation_dataset | 239 | EL | 1/1 | t.dedup_corpus | cfg doc2 arch1 str1 |
| t.flag_replays_with_recalls | 183 | E | 0/0 |  | str2 |
| t.download_replays | 152 | E | 0/0 |  | arch2 |
| t.purge_corrupt_gold_replays | 129 | E | 0/0 |  | - |
| ta.decisions_per_side_turn | 123 | E | 0/0 |  | doc2 |
| t.annotate_corpus_fog | 121 | E | 0/1 |  | doc2 |
| t.replay_manifest | 118 | E | 0/0 |  | - |
| t.check_replay_consistency | 95 | E | 0/0 |  | arch1 |
| t.dedup_corpus | 94 | E | 0/1 |  | doc2 |

### FIDELITY (23 modules, 5862 lines)
| module | lines | kind | P/T | production importers | named by |
|---|---:|---|---|---|---|
| t.dump_savestate | 667 | E | 0/0 |  | doc1 tsrc1 |
| t.diff_replay | 642 | EL | 5/0 | t.build_value_corpus, t.check_first_cmd_anomaly, t.purge_corrupt_gold_replays +2 | sh3 doc5 |
| t.scenario_init_oracle | 421 | E | 0/1 |  | doc3 |
| t.validate_replay_wesnoth | 416 | EL | 1/1 | t.run_validation_batch | doc1 |
| t.hidden_units_oracle | 415 | E | 0/1 |  | doc3 |
| t.diff_move_final_hex | 281 | E | 0/0 |  | - |
| t.validation_exports | 252 | L | 2/3 | t.az_loop, t.sim_self_play | doc1 |
| t.diff_combat_strike | 246 | E | 0/1 |  | - |
| t.verify_trailer_drop | 245 | E | 0/0 |  | - |
| t.diff_unit_counter | 241 | E | 0/0 |  | - |
| t.verify_mp_checkup | 239 | EL | 1/2 | t.diff_combat_strike | - |
| ta.vision_rule_census | 232 | E | 0/0 |  | doc2 |
| t.validate_replay | 212 | E | 0/1 |  | - |
| t.dump_unit_states | 182 | E | 0/0 |  | - |
| ta.hider_rule_sample | 162 | E | 0/0 |  | sh1 doc3 |
| t.run_validation_batch | 161 | E | 0/0 |  | arch2 |
| ta.counter_weapon_census | 151 | E | 0/0 |  | doc1 |
| t.mask_sim_fuzz | 150 | E | 0/0 |  | doc1 |
| t.check_mask_coverage | 121 | E | 0/0 |  | - |
| t.diff_core | 115 | E | 0/1 |  | sh5 doc6 |
| t.make_strict_replay | 106 | E | 0/0 |  | - |
| t.playback_verdict | 103 | E | 0/0 |  | doc1 |
| ta.hide_cover_census | 102 | E | 0/0 |  | sh1 doc2 |

### MODEL (8 modules, 4132 lines; plus the 19-line package `__init__`, whose 10 importers use `from wesnoth_ai import <module>`)
| module | lines | kind | P/T | production importers | named by |
|---|---:|---|---|---|---|
| w.encoder | 2024 | L | 26/30 | w.action_sampler, w.game_core, w.graphed_serve +23 | sh6 doc8 arch5 |
| w.packed_trunk | 682 | L | 8/4 | w.encoder, w.graphed_serve, w.imitation_loss +5 | doc3 str1 tsrc1 |
| w.model | 665 | L | 14/14 | w.action_sampler, w.graphed_serve, w.server_priors +11 | sh1 doc8 arch5 |
| w.model_output | 266 | L | 2/0 | w.model, w.padded_streams | - |
| w.constants | 251 | L | 19/16 | w.action_sampler, w.encoder, w.transformer_policy +16 | sh2 doc5 arch1 str1 |
| t.unit_vocab | 108 | EL | 2/2 | t.policy_anchor, t.supervised_train | sh1 doc2 |
| w.padded_streams | 101 | L | 1/0 | w.model | - |
| w.material | 35 | L | 3/1 | w.encoder, w.model, ta.value_head_by_phase | doc1 str1 |

### INFER (6 modules, 1989 lines)
| module | lines | kind | P/T | production importers | named by |
|---|---:|---|---|---|---|
| t.inference_seam | 654 | L | 11/15 | w.trainer, t.actor_pool, t.actor_worker +8 | doc1 |
| w.graphed_serve | 496 | L | 4/1 | t.actor_pool, t.bench_serve_graph, t.eval_inference_server +1 | sh2 doc4 str4 |
| w.server_priors | 467 | L | 11/9 | w.action_sampler, w.graphed_serve, w.leaf_wire +8 | doc2 |
| w.leaf_wire | 149 | L | 4/1 | t.actor_worker, t.bench_serve_graph, t.eval_inference_server +1 | doc2 |
| t.device_select | 137 | L | 7/1 | t.diagnose_selfplay, t.elo_ladder, t.eval_sim +4 | - |
| w.device | 86 | L | 2/0 | w.trainer, w.transformer_policy | - |

### SEARCH (13 modules, 9872 lines)
| module | lines | kind | P/T | production importers | named by |
|---|---:|---|---|---|---|
| w.action_sampler | 1918 | L | 18/28 | w.server_priors, w.trainer, w.transformer_policy +15 | doc8 arch2 |
| t.mcts | 1857 | L | 25/36 | t.az_loop, t.bench_leaf, t.bench_pool +22 | doc7 arch9 str1 tsrc1 |
| t.mcts_policy | 1619 | L | 18/27 | t.actor_worker, t.az_loop, t.bench_pool +15 | doc3 arch6 |
| t.plan_tournament (Q) | 1199 | L | 4/1 | t.actor_worker, t.elo_eval_game, t.run_elo_batch +1 | doc1 str1 tsrc2 |
| w.transformer_policy | 1045 | L | 28/71 | w.policy, t.box_bench, t.build_human_anchor +25 | doc4 arch1 |
| t.turn_search (Q) | 958 | L | 13/5 | t.crn_kill_probe, t.elo_eval_game, t.gbc_rung0 +10 | doc4 arch8 tsrc1 |
| t.turn_policy (Q) | 338 | L | 4/5 | t.actor_worker, t.elo_eval_game, t.sim_self_play +1 | doc1 arch8 |
| t.raw_player | 191 | L | 2/3 | t.elo_eval_game, t.turn_gap | sh1 doc7 str2 |
| w.dummy_policy | 179 | L | 8/11 | w.policy, t.bench_infer, t.elo_eval_game +5 | - |
| t.draw_tiebreak | 166 | L | 9/9 | t.elo_eval_game, t.ladder_anatomy, t.mcts +6 | cfg doc2 arch4 |
| t.turn_search_config (Q) | 147 | L | 5/1 | t.elo_eval_game, t.run_elo_batch, t.tcs_collapse_probe +2 | - |
| w.combat_oracle (Q) | 139 | L | 1/0 | w.action_sampler | doc1 |
| w.policy (D) | 116 | L | 1/0 | w.transformer_policy | doc1 |

### TRAIN (15 modules, 7964 lines)
| module | lines | kind | P/T | production importers | named by |
|---|---:|---|---|---|---|
| t.supervised_train | 2845 | EL | 3/8 | t.policy_anchor, t.policy_shape_probe, s.check_mover_sign | sh11 cfg doc8 arch4 str1 tsrc1 |
| w.trainer | 2203 | L | 7/22 | w.transformer_policy, t.az_loop, t.bench_train_step +4 | doc5 arch10 tsrc1 |
| t.value_head_fit | 353 | E | 0/0 |  | doc2 arch1 |
| t.value_pretrain | 319 | EL | 1/0 | t.value_head_fit | doc2 arch1 |
| t.step_control | 316 | L | 2/1 | t.az_loop, sp.run_step_scale | doc1 arch3 str1 |
| t.net2net | 267 | EL | 1/1 | t.measure_warm_start | doc1 arch4 tsrc1 |
| t.value_finetune | 267 | E | 0/0 |  | doc2 arch1 |
| t.value_corpus | 240 | EL | 5/1 | t.build_human_anchor, t.probe_teacher_advantage, t.probe_value_head +2 | arch1 |
| t.preencode_corpus | 228 | EL | 3/3 | t.supervised_train, s.check_mover_sign, s.count_epoch_pairs | sh7 doc1 str1 |
| t.compute_action_type_weights | 199 | E | 0/0 |  | cfg doc1 str1 |
| w.imitation_loss | 185 | L | 1/1 | t.supervised_train | sh1 doc2 |
| t.encode_worker | 165 | L | 2/2 | t.preencode_corpus, t.supervised_train | doc1 |
| t.signal_telemetry | 165 | L | 3/0 | t.az_loop, t.mcts_policy, sp.run_label_calibration | - |
| w.train_perf | 107 | L | 4/2 | w.trainer, w.transformer_policy, t.az_loop +1 | tsrc1 |
| sp.target_amplitude | 105 | L | 3/0 | t.az_loop, sp.run_profile, sp.run_profile_v2 | - |

### SELFPLAY (14 modules, 11004 lines)
| module | lines | kind | P/T | production importers | named by |
|---|---:|---|---|---|---|
| t.sim_self_play (Q) | 4088 | EL | 14/15 | t.actor_worker, t.az_loop, t.bench_pool +11 | sh2 doc5 arch8 str4 tsrc4 |
| w.rewards (Q) | 1518 | L | 5/6 | t.diagnose_selfplay, t.profile_selfplay, t.sim_dummy_smoke +2 | doc3 arch2 |
| t.actor_pool | 1213 | L | 8/8 | t.actor_stream, t.az_loop, t.bench_pool +5 | doc5 arch3 tsrc2 |
| t.az_loop | 824 | E | 0/1 |  | sh1 doc6 arch3 str1 tsrc2 |
| t.actor_worker | 582 | L | 3/4 | t.actor_pool, t.actor_stream, t.serve_worker | tsrc2 |
| t.actor_stream | 489 | L | 1/1 | t.actor_pool | doc3 str2 |
| t.serve_worker | 470 | L | 1/2 | t.actor_pool | - |
| w.gbc (Q) | 326 | L | 5/2 | w.model, w.trainer, t.gbc_heads +2 | doc1 arch3 |
| t.host_resources | 308 | L | 2/1 | t.az_loop, t.run_elo_batch | sh1 arch1 |
| t.policy_anchor (Q) | 290 | EL | 1/2 | t.sim_self_play | sh2 arch2 str1 |
| t.mp_teardown | 259 | L | 5/2 | t.actor_pool, t.actor_worker, t.encode_worker +2 | doc1 |
| t.value_grounding (Q) | 259 | L | 7/2 | t.mcts_policy, t.signal_telemetry, t.sim_self_play +4 | doc1 arch1 |
| t.build_human_anchor (Q) | 201 | EL | 1/1 | t.sim_self_play | sh1 doc2 arch1 str1 |
| t.midgame_starts (Q) | 177 | L | 4/4 | t.actor_worker, t.crn_kill_probe, t.probe_teacher_advantage +1 | doc2 arch1 |

### EVAL (16 modules, 9229 lines)
| module | lines | kind | P/T | production importers | named by |
|---|---:|---|---|---|---|
| t.run_elo_batch | 1502 | EL | 3/10 | t.elo_collect, t.elo_eval_game, t.eval_inference_server | sh20 doc8 arch3 str2 tsrc5 |
| t.turn_gap | 1315 | EL | 1/1 | ta.turn_gap_verdict | sh1 doc5 tsrc1 |
| t.elo_catalog | 1269 | EL | 1/5 | t.elo_collect | doc2 arch1 str1 |
| t.elo_eval_game | 1146 | EL | 1/13 | t.run_elo_batch | sh5 doc4 arch2 str1 tsrc3 |
| t.eval_inference_server | 731 | EL | 3/4 | t.elo_eval_game, t.run_elo_batch, t.turn_gap | sh3 doc3 str3 |
| t.elo_ladder | 701 | EL | 5/6 | t.bench_infer, t.elo_catalog, t.elo_collect +2 | doc1 arch2 |
| t.eval_sim | 573 | EL | 18/12 | t.az_loop, t.bench_infer, t.bench_model_cost +15 | doc2 arch2 str2 |
| t.elo_collect | 475 | E | 0/8 |  | sh9 doc5 arch1 str1 |
| t.whr | 317 | E | 0/1 |  | doc1 arch1 |
| ta.value_head_by_phase | 294 | EL | 1/0 | ta.value_head_compare | sh6 doc1 |
| ta.value_head_compare | 234 | E | 0/0 |  | doc3 |
| t.eval_workers | 209 | L | 1/1 | t.run_elo_batch | doc1 str1 |
| ta.endturn_readout | 191 | E | 0/0 |  | sh3 doc2 |
| ta.turn_gap_verdict | 122 | E | 0/1 |  | sh1 doc2 |
| t.reference_player | 94 | EL | 1/0 | t.turn_gap | sh4 cfg doc3 |
| t.eval_procedure | 56 | L | 3/2 | t.elo_eval_game, t.run_elo_batch, t.turn_gap | doc1 |

### BRIDGE (7 modules, 3094 lines)
| module | lines | kind | P/T | production importers | named by |
|---|---:|---|---|---|---|
| t.eval_daily | 577 | E | 0/0 |  | - |
| w.state_converter | 547 | L | 2/1 | t.eval_runner, t.eval_vs_builtin | doc2 |
| w.wesnoth_interface | 498 | L | 3/2 | t.eval_runner, t.hidden_units_oracle, t.scenario_init_oracle | doc2 arch3 |
| t.eval_vs_builtin | 477 | E | 0/2 |  | doc4 arch4 str2 tsrc1 |
| t.eval_scenarios | 399 | L | 1/0 | t.eval_vs_builtin | - |
| main | 310 | E | 0/0 |  | doc2 arch3 tsrc1 |
| t.eval_runner | 286 | L | 1/1 | t.eval_vs_builtin | doc2 arch1 |

### BENCH (18 modules, 4736 lines)
| module | lines | kind | P/T | production importers | named by |
|---|---:|---|---|---|---|
| t.bench_train_step | 783 | E | 0/7 |  | sh1 doc1 |
| t.profile_selfplay | 630 | E | 0/0 |  | - |
| t.bench_pipeline | 569 | EL | 4/3 | t.bench_model_cost, t.bench_serve_graph, t.bench_train_step +1 | sh1 doc6 str2 |
| t.bench_serve_graph | 454 | E | 0/0 |  | sh1 doc1 |
| t.bench_pool | 358 | E | 0/1 |  | sh9 doc4 |
| t.profile_rollout | 303 | E | 0/0 |  | doc2 arch3 |
| t.bench_infer | 219 | E | 0/2 |  | arch1 |
| t.bench_model_cost | 215 | E | 0/1 |  | doc3 |
| t.box_bench | 172 | E | 0/0 |  | doc2 arch2 |
| b.bench_mcts_tt (D) | 164 | E | 0/0 |  | - |
| t.bench_leaf | 161 | E | 0/0 |  | sh1 doc1 |
| b.bench_sim_throughput (D) | 149 | E | 0/0 |  | - |
| t.bench_core | 137 | E | 0/0 |  | sh1 doc2 |
| w.profiling (D) | 123 | - | 0/0 |  | doc1 |
| t.pyspy_summary | 90 | E | 0/0 |  | sh6 |
| b.bench_enum (D) | 84 | E | 0/0 |  | - |
| ta.eval_jobs_sweep | 74 | E | 0/0 |  | - |
| ta.turn_gap_timing | 51 | E | 0/0 |  | - |

### OPS (13 modules, 2750 lines)
| module | lines | kind | P/T | production importers | named by |
|---|---:|---|---|---|---|
| t.cleanup | 644 | E | 0/0 |  | tsrc1 |
| s.hf_upload_loop | 357 | E | 0/1 |  | sh3 arch1 |
| s.leg_daemons (Q) | 345 | E | 0/0 |  | sh1 tsrc1 |
| s.holdout_probe_loop (Q) | 293 | E | 0/1 |  | sh1 arch4 |
| t.stage_code | 219 | E | 0/0 |  | sh1 |
| s.rent_box | 168 | E | 0/0 |  | sh1 |
| t.kernel_status | 158 | EL | 1/1 | t.sim_self_play | sh10 doc1 |
| t.leg_config (Q) | 146 | E | 0/1 |  | sh1 |
| s.box_stop_on_abort | 142 | E | 0/0 |  | sh3 arch1 |
| s.stall_watchdog | 109 | E | 0/0 |  | sh3 arch1 |
| t.pull_box_records | 70 | E | 0/0 |  | - |
| s.leg_table (Q) | 50 | E | 0/0 |  | arch1 |
| s.probe_escrow_loop | 49 | E | 0/0 |  | sh2 |

### PROBE (45 modules, 10575 lines)
| module | lines | kind | P/T | production importers | named by |
|---|---:|---|---|---|---|
| t.swap_detector (Q) | 1151 | E | 0/1 |  | arch1 |
| t.turn_counterfactual_probe (Q) | 578 | EL | 3/1 | t.gbc_rung0, t.projection_sign_probe, t.target_channel_icc | arch4 |
| t.diagnose_selfplay | 506 | E | 0/0 |  | - |
| t.gbc_rung0 (Q) | 418 | E | 0/0 |  | arch1 |
| t.eval_mcts_vs_reinforce | 389 | E | 0/0 |  | - |
| t.measure_warm_start | 374 | E | 0/0 |  | doc1 arch2 |
| t.recruit_prior_drift (Q) | 364 | E | 0/1 |  | arch1 |
| t.plot_training_curves | 349 | E | 0/0 |  | - |
| t.metrics_viz | 347 | E | 0/0 |  | - |
| t.sim_dummy_smoke | 330 | E | 0/0 |  | - |
| t.mini_anatomy | 306 | E | 0/0 |  | arch2 |
| t.target_channel_icc (Q) | 292 | E | 0/0 |  | arch1 |
| sp.gradient_tree (Q) | 289 | L | 5/1 | sp.aleatoric_probe, sp.run_profile, sp.run_profile_v2 +2 | - |
| t.luck_probe (Q) | 277 | E | 0/0 |  | arch2 |
| t.ladder_anatomy | 267 | E | 0/0 |  | - |
| ta.turn_gap_pregrader | 258 | E | 0/0 |  | doc3 |
| ta.turn_gap_audit (D) | 257 | - | 0/0 |  | - |
| t.crn_kill_probe (Q) | 243 | E | 0/0 |  | arch1 |
| t.gbc_labels (Q) | 237 | EL | 1/1 | t.gbc_rung0 | arch1 |
| t.policy_shape_probe (Q) | 226 | E | 0/0 |  | arch1 |
| sp.run_step_scale (Q) | 221 | E | 0/0 |  | arch1 |
| t.check_first_cmd_anomaly | 219 | E | 0/0 |  | - |
| t.projection_sign_probe (Q) | 208 | E | 0/0 |  | - |
| sp.update_tree (Q) | 199 | L | 1/0 | sp.run_profile_v2 | - |
| sp.run_label_calibration (Q) | 185 | E | 0/0 |  | arch2 |
| t.tcs_collapse_probe (Q) | 179 | E | 0/0 |  | - |
| t.probe_value_head | 170 | E | 0/0 |  | - |
| t.diagnose_value_drift | 165 | E | 0/0 |  | - |
| t.probe_teacher_advantage | 147 | E | 0/0 |  | doc1 |
| t.gbc_heads (Q) | 134 | L | 1/1 | t.gbc_rung0 | doc1 arch1 |
| sp.run_profile_v2 (Q) | 130 | E | 0/0 |  | sh1 str1 |
| t.analyze_teacher_advantage | 128 | E | 0/0 |  | doc1 arch1 |
| t.noprogress_report (Q) | 122 | E | 0/1 |  | - |
| sp.experience_harvest (Q) | 112 | L | 4/0 | sp.run_aleatoric, sp.run_label_calibration, sp.run_profile +1 | - |
| sp.aleatoric_probe (Q) | 107 | L | 1/0 | sp.run_aleatoric | - |
| t.raw_policy_smoke | 100 | E | 0/0 |  | - |
| ta.turn_gap_audit2 (D) | 98 | - | 0/0 |  | - |
| sp.value_splits (Q) | 89 | L | 1/0 | sp.run_profile_v2 | - |
| sp.run_profile (Q) | 84 | E | 0/0 |  | - |
| t.measure_gold_hoarding | 80 | E | 0/0 |  | - |
| sp.run_aleatoric (Q) | 77 | E | 0/0 |  | - |
| sp.consult_capture (Q) | 60 | L | 1/0 | sp.run_profile_v2 | - |
| s.count_epoch_pairs (D) | 42 | - | 0/0 |  | - |
| s.check_mover_sign (D) | 31 | - | 0/0 |  | - |
| sp.render (Q) | 30 | L | 2/1 | sp.run_profile, sp.run_profile_v2 | - |

### Rust crate `rust/wesnoth_core/src` (`__phase__` 14)
| module | lines | what it exports | Python callers | system |
|---|---:|---|---|---|
| lib.rs | 542 | Module registry, plus the reach kernels: Dijkstra, `unit_reach_arrays`, `reach_rows`, `rows_from_reach`, `enumerate_moves` | w.visibility, t.pathfind_sim, w.observe, w.action_sampler, t.kernel_status | SIM |
| combat.rs | 535 | `resolve_attack`; `random_int` | w.combat; `random_int` is used only by tests/test_rust_combat.py | SIM |
| encode.rs | 513 | `encode_raw_streams` | w.encoder | MODEL |
| observe.rs | 266 | `observe_side` | w.observe | SIM |
| core.rs | 789 | `GameCore`: the records plus its Python methods. Off by default (`WESNOTH_RUST_CORE` = 0). | w.game_core, through t.wesnoth_sim.core_enabled; t.bench_core, t.diff_core | SIM |
| core_step / core_move / core_attack / core_fog / core_observe / core_encode / core_sim | 243 / 313 / 334 / 192 / 220 / 203 / 94 | GameCore internals: commands, move, attack, fog, observation, encoding, queries from the simulator | through core.rs | SIM |

Rust doc comments name Python paths 22 times (for example `tools/replay_dataset._apply_command`). Moving Python code does not change the build.

# (b) Dead code, with evidence

These extend BACKLOG's "Decision: deletions" item. Every deletion below is a candidate for the owner, not a recommendation to act.

**b1. Modules nothing imports.**
- **`w.policy`:** its one "importer" is transformer_policy.py:1041 registering itself, and policy.py:113 imports transformer_policy back, which makes a cycle. `get_policy()` and `available()` have 0 callers.
- **`w.profiling`:** 0 importers. Its docstring says it is "used by game_manager", a module that no longer exists.
- **`t.replay_builder`:** 0 importers. The only reference is tests/test_wml_state.py:303, which lists the file in a source scan; that list needs editing if the module is deleted.
- **New:**
  - `ta.turn_gap_audit` and `ta.turn_gap_audit2` hardcode `ROOT = Path(r"C:/Users/amaur/...")`.
  - `s.check_mover_sign` and `s.count_epoch_pairs` have no `__main__` block and nothing references them.
  - The whole `benchmarks/` directory (3 files, 397 lines) is named nowhere.

**b2. Entry points nobody runs: 48 files, 10,111 lines.** Each has no importer, no box script, no live-doc mention, no test path and no Python string reference.
- **Probes whose result lives in archive docs:** leg_table, sp.run_label_calibration, sp.run_step_scale, crn_kill_probe, gbc_rung0, luck_probe, mini_anatomy, policy_shape_probe, target_channel_icc.
- **Corpus-rebuild tools** (needed only to rebuild the corpus from raw replays): download_replays, sort_replays, purge_corrupt_gold_replays, replay_manifest, check_first_cmd_anomaly, check_replay_consistency.
- **Fidelity oracles** worth keeping for the next fidelity bug: diff_move_final_hex, diff_unit_counter, dump_unit_states, verify_trailer_drop, make_strict_replay, check_mask_coverage, run_validation_batch.
- **Legacy loops, evals, dashboards and smokes:**
  - eval_daily (577), eval_mcts_vs_reinforce (389), profile_selfplay (630), diagnose_selfplay (506), diagnose_value_drift (165).
  - sim_dummy_smoke (330), raw_policy_smoke (100), measure_gold_hoarding (80), probe_value_head (170), ladder_anatomy (267).
  - metrics_viz (347), plot_training_curves (349), tcs_collapse_probe (179), projection_sign_probe (208).
  - sp.run_aleatoric, sp.run_profile, benchmarks ×3.
- **September one-offs:** ta.eval_jobs_sweep, ta.turn_gap_audit, ta.turn_gap_audit2, ta.turn_gap_timing, s.check_mover_sign, s.count_epoch_pairs. Also pull_box_records (09-21); it is probably used by hand, so check with the user.

**b3. Functions and classes with no caller anywhere: 23 defs, 306 lines.**
- rewards: `_min_dist_to_enemy_leader` (66) and `RewardFn` (4).
- replay_builder: `export_scenario_replay` (60).
- scrape_unit_stats: `_scan_attack_special_macros` (42).
- value_head_by_phase: `head_values` (17).
- supervised_train: `_loss_for_pair` (17).
- sim_self_play: `_is_ladder_map` (12). A comment at scenario_pool.py:66 still names it as a consumer.
- wesnoth_sim: `_move_cost` (11).
- replay_dataset: `iter_dataset` (11).
- swap_detector: `compare_state_distributions_lex` (10).
- policy: `get_policy` (7).
- encoder: `_modifier_flags` and `_clamp_pos`.
- replay_extract: `_safe_int`.
- scenario_events: `unknown_macro_counts` and `reset_unknown_macros` (named only in `__all__`).
- combat_outcomes: `fallback_counter_weapon_count`.
- game_record: `add_outcomes` and `recording`.
- abilities: `_units_at`. dump_unit_states: `_find_unit_by_id`. wml_state: `side_numbers`. sp.gradient_tree: `_norm`.

Dead methods (76 lines):
- `StateConverter.convert_hex` (33) and `.forget_game`.
- `Map.deep_clone` (15; mentioned only in comments).
- `Profiler.throughput`, `WesnothGame.check_game_over`, `WorkerPool.dead_worker_logs`, `ImitationLossParts.log_values`, `Comparison.is_improvement`.

Code only tests reach (22 defs, 724 lines; 6 methods, 38 lines):
- `trainer._mcts_factored_policy_loss_reference` (293) and `pathfind_sim._unit_reach_reference` (74) are parity oracles; they belong in `tests/oracles`.
- `action_sampler.predict_priors` (91) has no production caller.
- swap_detector's `compare_states`, `reconstruct_side_turn_dist` and `compare_state_distributions` (85).
- gbc_heads' `TrunkTap`, `goal_token` and `ece`.
- sim_to_replay's `_scrape_scenario_metadata` and `_load_map_data`.
- `server_priors._legal_capacity`.
- `game_record.read_records` and `rebuild`. **Keep `read_records`:** both open exp branches use it.

**b4. Flags, fields, constants and configs.**
- **argparse flags no caller ever passes** (no script, test, doc or Python string):
  - az_loop: `--search-probe-every-pins`, `--rng-seed`.
  - sim_self_play: `--midgame-dataset`, `--turn-reply-max-actions`, `--mcts-temperature`, `--mcts-temperature-decisions`, `--mcts-playout-cap-prob`, `--mcts-playout-cap-fast-sims`, `--mcts-moves-left-coef`, `--value-ground-capture-prob`. Another 24 of its 132 flags appear only in docs.
  - supervised_train: `--all-scenarios`, `--max-starting-units`, `--gc-every-files`, `--prefetch-factor`, `--no-fog-hides-enemy-villages`, `--no-terrain-multi-hot`.
  - bench_train_step: 8 flags (`--masks`, `--experiences-in`, `--compile`, `--parity-n`, `--exps-per-game`, `--loop-sims`, `--leaves-per-s`, `--iteration-leaves-per-s`).
  - run_elo_batch: `--pt-args`. turn_gap: `--raw-end-turn`.
- **Flags every caller sets the same way:**
  - run_elo_batch has 21 invocations in 20 box scripts. `--persistent-workers` is on in 18/21 and `--shared-inference` in 17/21, and both default off. `--mcts-sims 0` appears in 19/20 and `--raw-temperature-a/b 0` in 19/21.
  - bench_pool: `--server-priors --infer-bf16 --packed-trunk` in 8/10 invocations. All three default off there but on in az_loop.
  - supervised_train: `--lr 1e-4` in 9/9 (it is already the default); `--imitation-config configs/imitation.json` in 9/9 (default None).
- **Dataclass fields no production caller sets:**
  - MCTSConfig: `dirichlet_alpha`, `dirichlet_eps`, `time_budget`, `virtual_loss`, `gumbel_c_visit`, `gumbel_c_scale`, `gumbel_rescale_q`.
  - TrainerConfig: `gamma`, `entropy_coef`, `normalize_advantages`, `value_clip`, `trust_delta`.
  - GroundingConfig: `rollout_max_halfturns`, `rollout_max_actions`, `rollouts_per_state`.
  - TournamentConfig: `min_challengers`.
- **Always-default parameter:** every live caller passes `play_one_game(reward_fn=_zero_reward)` (actor_worker:513, the anatomy scripts).
- **Unread constants** in constants.py: `NUM_PARALLEL_GAMES`, `LOG_FREQUENCY`, `CHECKPOINT_FREQUENCY`, `COMBAT_LOGIT_ALPHA`.
- **Configs:**
  - No reader at all: configs/replay_map_whitelist.txt, map_whitelist_1v1.json, vendored_addon_ids.txt.
  - Only passed by hand to quarantined legs: reward_selfplay.json, leg_l4.json, leg_l5.json.

**b5. Code paths only quarantined mechanisms reach.** az_loop.py:446-452 builds `MCTSConfig(gumbel_root=False, tree_reuse=False, playout_cap_randomization=False, draw_tiebreak=None)` and `MCTSPolicy(replay_config=disabled, holdout_size=0, gbc_labels=False)`, and passes midgame 0, fogless 0 and aux/gbc/moves_left coefficients 0. So the live loop never runs:
- In `MCTSPolicy`: holdout (L923-1071, about 150 lines), value memory (L1072-1142), the VG2 trust region (L1260-1380) and the replay-buffer branch.
- In the trainer: REINFORCE `Trainer.step` (254 lines, only under `--reinforce`), `_trainer_step_value_from_raw`, and the trust and consistency terms.
- GBC heads and labels (`w.gbc`, 326).
- The combat-oracle bias (`w.combat_oracle`, alphas 0).
- In mcts: tree reuse, playout cap, draw-tiebreak terminal, aux bonus, moves-left utility, hierarchical and classic roots. The Gumbel root stays live for eval `mcts:` players.
- `rewards.compute_delta`: the game loop skips it because `MCTSPolicy.uses_step_rewards` is False.
- sim_self_play L986-4067 (`run_iteration` 854, `_TrainerHistoryCSV` 282, `main` 1,895).

Live code still branches into quarantined modules:
- `actor_worker._actor_loop` imports PlanTournamentPolicy, TurnCommitPolicy and midgame_starts.
- run_elo_batch reads plan-tournament and TCS knobs.
- elo_eval_game's searched players **default to TCS** (`_search_policy_cls(True)`; `--no-turn-search` opts out), which is the 2026-08-26 ruling pinned by test_eval_sampling. Removing TCS means changing that default first.

**b6. Tests that only test dead or quarantined code.**
- **Already named in BACKLOG** (10 files, 3,160 lines): test_rewards, test_plan_tournament, test_swap_detector, test_holdout_tripwire, test_gbc_heads, test_gbc_labels, test_gbc_training, test_vg2_mixture, test_vg3_continuation, test_value_grounding.
- **New** (29 files, about 4,600 lines; inventory item in brackets):
  - test_policy_anchor [4.1]; test_anchor_cache_gate [4.1/6.17, mixed: it also checks the pre-encode epoch gate]; test_probe_tripwires [3.4]; test_leg_config [3.9]; test_leg_daemons [3.8].
  - test_turn_policy [5.4, slow]; test_turn_project [6.21]; test_turn_target_link [5.5]; test_turn_probe [2.13].
  - test_value_memory [5.6]; test_moves_left and test_aux_targets [6.10, slow]; test_playout_cap [6.8]; test_draw_tiebreak [6.11, mostly; elo_eval_game reads `material_margin`].
  - test_no_progress_rule and test_noprogress_report [5.11]; test_fogless_mixing and test_midgame_starts [6.18, slow]; test_replay_buffer [6.15]; test_combat_anneal_mcts [6.25]; test_policy_weight [plan-tournament only]; test_holdout_persistence [2.1, slow].
  - test_recruit_prior_drift and test_net2net (tier-a tools with no user).
  - test_parallel_rollouts [1.7, slow] and test_sim_self_play_smoke [1.9].
  - test_eval_sampling: reads source text to pin anchor defaults and the TCS default.
  - test_predict_priors: tests production code with no production caller.
  - signal_profiler/tests/test_gradient_tree [2.13]. Its docstring says it sits outside the suite, but pytest.ini has no `testpaths` and nothing ignores it, so the default run probably collects it (not verified by a collection run).
- **Checked and still live** (not candidates): test_value_weight (`draw_value_weight` censoring runs in az_loop), test_boundary_telemetry (the pool harvests boundary pairs), test_value_loss_form (az_loop uses `mse_mean`).
- **Blockers on open branches:**
  - exp/xod-dominance imports `swap_detector.enumerate_children_via_sim`, `rewards.hex_distance` and `game_record._replay_steps`.
  - feature/signal-telemetry's new tests/imitation_helpers.py imports `bench_infer`, and its `signal_telemetry.py` imports `value_grounding`.

# (c) Structure problems

**c1. About 50k lines of library code live in tools/.** tools/ holds 165 files and 76.1k lines:
- 39 library-only modules (20.9k lines);
- 34 that are both library and CLI (29.1k lines);
- 89 entry-only (25.4k lines);
- 3 that are neither.

The simulator, MCTS, the actor pool and the eval stack all live there. The de facto public API is private names:

| private name | prod / test importers |
|---|---|
| `replay_dataset._apply_command` | 24 / 9 |
| `replay_dataset._build_initial_gamestate` | 24 / 8 |
| `replay_dataset._setup_scenario_events` | 22 / 4 |
| `replay_dataset._stats_for` | 16 / 6 |
| `eval_sim._load_policy` | 18 / 7 |
| `sim_self_play._recruit_cost_lookup` | 10 / 11 |

Production code imports 116 private names across modules; the actor and serve protocol constants in actor_worker and serve_worker account for 36 of them.

- **The live generation path runs through the legacy CLI.** actor_worker:338 imports `_play_one_game_safe` from the 4,088-line sim_self_play. az_loop, bench_pool, eval_sim, elo_ladder and turn_gap import helpers from it too.
- **Benchmarks double as libraries.** Tests and turn_gap import helpers from benchmark scripts: `bench_train_step.configure_trainer_like_az_loop` (6 tests), `bench_pipeline.load_states` (turn_gap and 2 tests), `bench_infer.harvest_states` (2 tests).

**c2. Layering and cycles.**
- wesnoth_ai imports tools in 23 module pairs (35 statements). All are deferred imports except dummy_policy → abilities at module level. The "library" package cannot run without tools/ on sys.path.
- There is no module-level cycle. Five cycles close through deferred imports:
  1. A 19-module simulator, replay and encoder tangle: replay_dataset, wesnoth_sim, pathfind_sim, scenario_events, scenario_pool, combat_outcomes, abilities, neutral_ai, game_record, engagement_stats, replay_extract, and w.encoder, game_core, observe, visibility, model, gbc, material, dummy_policy.
  2. action_sampler ↔ server_priors.
  3. policy ↔ transformer_policy (the dead registry).
  4. actor_pool / actor_stream / actor_worker / serve_worker / sim_self_play. It shrinks to actor_pool ↔ actor_stream once the game loop leaves sim_self_play.
  5. elo_eval_game → run_elo_batch at module level (the provenance helpers). run_elo_batch → `elo_eval_game._pt_config` and eval_inference_server → `run_elo_batch.file_sha256` close it.

**c3. Files over 600 lines: 42 Python files.** The size comes mostly from 26 functions of 250 lines or more (13.5k lines in all):
- `sim_self_play.main` 1,895; `replay_extract.extract_replay` 1,211; `supervised_train.train` 1,099; `run_elo_batch.main` 947; `sim_self_play.run_iteration` 854; `replay_dataset._apply_command` 783.
- `elo_eval_game.main` 640; `az_loop.main` 583; `elo_catalog.update_from_games` 556; `_build_legality_masks` 404; `play_one_game` 376; `build_scenario_gamestate` 340; `_trainer_step_mcts` 314.
- Splitting files without splitting these functions only moves the problem.

Proposed splits:

| file | lines | proposed modules |
|---|---:|---|
| t.sim_self_play | 4088 | `selfplay/game_loop.py` for L1-983 (GameOutcome, play_one_game, `_play_one_game_safe`, `_worker_loop`, `k_median_of`, `_roll_max_turns`, the recruit-cost and bounce helpers, `_leader_of`). The legacy trainer (L986-4067) stays as the quarantined CLI. |
| t.replay_dataset | 3081 | `rules/unit_db.py` (L55-150), `rules/map_data.py` (L150-406), `sim/units.py` (L406-706 and `_to_combat_unit`), `sim/setup.py` (L708-845, L2899-2946), `sim/advance.py` (L1227-1650), `sim/attack_context.py`, `sim/apply_*.py` (`_apply_command` split into init_side and healing, move, attack, recruit, end_turn), `sim/plague_capture.py`, `replays/dataset.py` (`ActionIndices`, `_action_indices`, manifest split) |
| t.supervised_train | 2845 | `training/imitation/{streams, loss (with w.imitation_loss), evaluate, checkpoint, train}.py`; `train` itself split into setup, epoch loop, batched step and per-pair step. The CLI stays. |
| t.scenario_events | 2328 | `rules/preprocessor.py` (L1-490), `rules/scenario_cfg.py` (L495-697), `sim/events.py` (L700-1476), `sim/effects.py` (L1482-2157), `sim/event_dispatch.py` (fire_event) |
| w.trainer | 2203 | `training/{experience, config, trainer, reinforce (quarantined), policy_loss, step_mcts, value_loss, value_metrics}.py`; the reference loss moves to tests/oracles |
| t.replay_extract | 2106 | `rules/wml.py` (`parse_wml`, `WMLNode`, plus-forms), `replays/initial_state.py`, `replays/extract.py` with per-command handlers |
| w.encoder | 2024 | `model/encoded.py` (EncodedState, RawEncoded), `model/encoder.py` (GameStateEncoder, 743), `model/encode_raw.py`, `model/features.py` |
| w.action_sampler | 1918 | `policy/{sampling, legal, masks, combat_bias}.py`; the reference enumerator and `predict_priors` move to tests/oracles |
| t.mcts | 1857 | `search/{mcts_config, mcts_tree, mcts_select, gumbel, mcts}.py` |
| t.wesnoth_sim | 1799 | `sim/simulator.py`, `sim/commands.py` (`_action_to_command`, 282), `sim/invariants.py`, `sim/rng.py`, the lookups at L159-428 into `rules/unit_db.py` |
| t.mcts_policy | 1619 | `search/mcts_policy.py` core; the holdout and value-memory/VG2 code (about 430 lines) to legacy |
| t.sim_to_replay | 1566 | `replays/{wml_commands, save_splice, save_compose, export}.py` |
| t.run_elo_batch | 1502 | `eval/provenance.py`, `eval/slots.py`, `eval/batch.py` (main split into launch, monitor, collect); the CLI stays |
| t.turn_gap / t.elo_catalog | 1315 / 1269 | `eval/turn_gap/{config, play, grade, summary}.py`; `eval/catalog/{edges, refit, update, render}.py` |
| t.actor_pool | 1213 | `selfplay/pool.py`, `selfplay/serve_manager.py`, `selfplay/protocol.py` (queue message constants and `ActorFatalError`) |
| t.elo_eval_game / w.transformer_policy | 1146 / 1045 | `eval/players.py` and `eval/game.py`; `policy/checkpoint_io.py` (`load_checkpoint` 272, `save_checkpoint`) |

- The quarantined large files (w.rewards 1518, t.plan_tournament 1199, t.swap_detector 1151) are quarantine or delete, not split.
- Leave t.scrape_unit_stats (1171, the pinned scraper) and w.combat (1035, the bit-exact port) as they are.
- 18 more files sit between 600 and 960 lines: turn_search, scenario_pool, combat_outcomes, game_core, az_loop, bench_train_step, pathfind_sim, eval_inference_server, elo_ladder, packed_trunk, dump_savestate, model, inference_seam, cleanup, diff_replay, terrain_resolver, profile_selfplay, visibility, classes.
- Rust: move the reach kernels out of lib.rs into `reach.rs`; split core.rs into `core_records.rs` (L27-257) and the Python methods.

**c4. Duplicated helpers.**
- **unit_stats.json has 5 loaders:**
  - `replay_dataset._load_unit_db`/`_stats_for` (generic fallback);
  - `wesnoth_sim._unit_stats_data`/`_recruit_cost_for` (cost 14 fallback; action_sampler imports it);
  - `sim_self_play._recruit_cost_lookup` (its own `json.load`, cost 14);
  - sim_to_replay:336, which re-reads the 400 KB file for each new unit type;
  - `replay_extract._unit_stats` (`_DEFAULT_UNIT` fallback).
  They fall back differently, so merging them must keep each call site's fallback.
- **The az trainer recipe is copied.** az_loop.py:420-431 matches `bench_train_step.configure_trainer_like_az_loop` (L300-316) line for line, and 6 tests use the copy. It can drift.
- **Checkpoint → policy loaders: 6.** `eval_sim._load_policy`, `eval_vs_builtin._load_checkpoint`, `eval_mcts_vs_reinforce._load_policy`, `gbc_rung0._load_policy`, `measure_gold_hoarding.load_policy`, `turn_counterfactual_probe.load_policy`.
- **AUC: 8 copies:**
  - holdout_probe_loop `_auc`; value_head_by_phase `pooled_auc`; value_head_compare `rank_auc`; analyze_teacher_advantage `auc`;
  - diagnose_value_drift `_auc`; gbc_heads `auc`; probe_value_head `auc`; supervised_train `_auc` (nested).
- **Mean ± SE: 4 copies** (turn_gap, value_head_by_phase, supervised_train, turn_gap_pregrader). **Wilson interval: 2** (eval_mcts_vs_reinforce, eval_vs_builtin).
- **Identical after normalization:**
  - `_leader_of` in diagnose_selfplay, diff_replay and sim_self_play;
  - `_hex_dist` plus `cube` in ladder_anatomy and mini_anatomy, next to `rewards.hex_distance`;
  - `replay_dataset._find_unit_at`, `replay_extract._find_unit_at` (0.91 similar) and `diff_replay._unit_at`;
  - `gbc_labels._village_hexes` and `gbc.village_hexes`;
  - `value_grad` in two signal_profiler files;
  - `drop_last_pending` and `drain_distill_stats` in mcts_policy, turn_policy and plan_tournament.
- **Tests:** `_u` is defined in 8 test files, `_tiny_policy` in 7, `_gs` in 6, `_unit` in 4. 12 tests import `_gs`/`_u` from test_inference_snapshot.py. Six other test files serve as fixture libraries: test_actor_pool_watchdog, test_batched_policy_loss, test_root_masks_shipped, test_actor_pool_smoke, test_eval_inference_server, test_sim_to_replay_from_scratch.
- **Shell:** `upload()` is defined in 13 box scripts, `progress()` in 8, `match()` in 7, `timed_match()` and `decisive_results()` in 6. Each retrain script is 61-75% identical to its predecessor (unit_vocab ← observation ← terrain_arm ← seed2_relset). scripts/ mixes live infrastructure with about 45 one-shot run records.

**c5. Hazards already in the tree that a move must handle.**
- **Paths computed from `__file__` depth:** there are 178 such computations in production code, 24 of them in the 15 modules that would move. Moving a module one directory deeper silently retargets `parent.parent` (unit_stats.json, terrain_db.json, wesnoth_src, add-ons).
- **Persisted pickles name module paths.**
  - Pre-encoded corpus records (preencode_corpus.py:91) pickle `wesnoth_ai.encoder.RawEncoded` and `tools.replay_dataset.ActionIndices`.
  - Holdout sidecars and bench experience pickles hold `MCTSExperience`; anchor caches hold their own objects.
- **93 `monkeypatch.setattr(module, name)` calls in tests** patch a module's namespace. A function that moves out of that module is no longer patched: in the worst case the test loses the patch silently. Heavy spots:
  - supervised_train: 8 names;
  - turn_gap: 7;
  - host_resources: 11 patches of one name;
  - scenario_pool's imported `_build_initial_gamestate`;
  - `mcts.enumerate_legal_actions_with_priors`;
  - `pathfind_sim._RUST`.
- **15 test files read source by path or glob:**
  - test_wml_state, test_eval_sampling, test_plan_tournament, test_end_turn_rule, test_raw_player, test_server_priors, test_terrain_multi_hot, test_elo_collect, test_probe_tripwires, test_time_of_day_features, test_cli_help, test_train_perf, test_no_dual_imports, test_addon_events, test_mcts.
  - The last two use paths relative to the working directory.
  - test_train_perf's glob and test_no_dual_imports' globs would shrink silently if files move.
- **tests/data/scenario_surface.json** names 75 readers as `tools/<file>.py:symbol`.
- **sys.path boilerplate:** 615 `sys.path.insert` lines in 307 files. 184 of the 199 test files repeat what tests/conftest.py already does. Putting `tools/` on sys.path is what makes the dual-import hazard possible; with only CLIs left in tools/, it could come off.
- **Torch-free drivers:** run_elo_batch, eval_procedure, turn_search_config and host_resources must stay torch-free, so new package `__init__.py` files must import nothing.

**c6. Documentation drift that hurts navigation.**
- CLAUDE.md's "Run:" line says training runs through sim_self_play.py.
- CLAUDE.md Architecture describes sim_self_play as the training entry, and its "Encapsulation" section names `GameManager`, which no longer exists.
- The `wesnoth_ai/__init__.py` docstring says "tools/ holds the scripts".
- The main.py docstring lists "Launch training".
- The tests/conftest.py docstring cites `import mcts`.
- requirements.txt pins torch-directml and omits scipy, psutil, pyyaml and huggingface_hub, which CI installs.
- CLAUDE.md is 1,264 lines, of which 780 are the "Current status" log; the Architecture map is 85 lines.

# (d) Target layout

Rule: all library code lives under `wesnoth_ai/<system>/`, and no library module imports tools/. `tools/` keeps every current filename as a thin CLI (argparse, then a call into the package), so every command line in docs and box scripts keeps working. tools/analysis and scripts/ keep their paths.

```
wesnoth_ai/
  paths.py        NEW: repo root and data files (replaces the __file__-depth lookups)
  constants.py    model constants, OBSERVATION_EPOCH (Steam paths move to bridge/settings.py)
  rules/          unit_db (5 loaders merged), terrain (terrain_resolver), map_data, wml, wml_state,
                  preprocessor, scenario_cfg, scenario_pool (+ scenarios)
  sim/            state (classes)*, combat*, combat_outcomes, abilities, traits, units, setup, advance,
                  apply_* (_apply_command), events, effects, pathfind, visibility*, observe*,
                  simulator + commands (wesnoth_sim), core (game_core)*, neutral_ai, engagement_stats, game_record
  replays/        dataset (ActionIndices, labels, manifest), extract, export + save_* (sim_to_replay),
                  value_corpus, midgame_starts
  model/          encoder (+ encoded, encode_raw)*, model*, model_output*, padded_streams*, packed_trunk*, material*, unit_vocab
  inference/      seam (inference_seam), server_priors*, leaf_wire*, graphed_serve*, device (+ device_select)
  policy/         sampling, legal, masks, combat_bias (action_sampler)*, transformer_policy* + checkpoint_io, dummy_policy*, raw_player
  search/         mcts (split), mcts_policy, draw_tiebreak, tcs/ (turn_search, config, turn_policy)
  training/       trainer (split)*, train_perf*, step_control, signal_telemetry, az_config (NEW),
                  imitation/ (streams, loss, evaluate, checkpoint, train, encode_worker, preencode)
  selfplay/       game_loop (from sim_self_play), protocol (NEW), pool + serve_manager (actor_pool),
                  actor_stream, actor_worker, serve_worker, mp_teardown, host_resources, validation_exports
  eval/           provenance, procedure, checkpoints (_load_policy), match (eval_sim), players, workers,
                  inference_server, ratings (elo_ladder math), catalog, whr, reference, turn_gap/
  bridge/         interface (wesnoth_interface)*, state_converter*, eval_runner, eval_scenarios, settings
  legacy/         only on the owner's ruling: rewards, gbc, value_grounding, policy_anchor, plan_tournament,
                  the sim_self_play trainer
tools/            CLIs only, same filenames; tools/analysis unchanged
scripts/          same paths; NEW scripts/lib/box.sh for upload/progress/match/stop_self
tests/            NEW tests/helpers (shared fixtures), tests/oracles (reference implementations)
rust/.../src      NEW reach.rs (from lib.rs), core_records.rs (from core.rs)
```
`*` marks an existing flat wesnoth_ai module; moving those is optional (cost below).

**Size of the change.**

| part | modules | import statements | files touched |
|---|---:|---:|---:|
| Minimal: library code out of tools/ | 41 moved whole (28.1k lines) | about 1,021 (488 production, 533 tests) | 260 (121 production, 139 tests) |
| Extractions from the 29 other library+CLI modules | about 15 extractions | up to about 630 (today 248 production and 380 test statements name those modules) | included above |
| Optional: flat wesnoth_ai modules into subpackages | 29 | about 763 (335 production, 428 tests) | 324 total with the minimal part |

Other edits in the minimal scope:
- 24 path lookups;
- 15 path-reading test files;
- 75 manifest readers in scenario_surface.json;
- about 29 thin CLI wrappers;
- live docs: 136 distinct Python paths, 788 mentions. Archive and dated docs stay as records, with a move table.

An AST codemod keeps the import rewrites mechanical.

**Order of steps.** Each step is one branch and must end with ruff clean, the fast tier green locally and CI's full run green.
0. **Make moves safe (no moves yet).**
   - Add `paths.py` and switch the 24 lookups to it.
   - Make the 15 source-scanning tests resolve files through the module or a recursive package walk.
   - Move shared fixtures into tests/helpers.
   - Give `preencode_corpus.read_record` and the other pickle loaders an Unpickler that maps old module paths to new ones.
   - Add the codemod script.
1. **Structural extractions (no directory moves).**
   - a. Game loop out of sim_self_play.
   - b. `eval_sim._load_policy` and match helpers into `eval/`.
   - c. Provenance helpers and `_pt_config` out of run_elo_batch and elo_eval_game. This breaks cycle 5.
   - d. Shared `configure_az_trainer`, removing the bench mirror.
   - e. One unit_db.
   - f. `harvest_states` and `load_states` out of the bench scripts.
   - g. `selfplay/protocol.py`, which breaks the actor_pool ↔ actor_stream cycle.
2. **Owner-approved deletions**, so nothing dead gets moved.
3. `rules/`. 4. `sim/` (plus the replay_dataset split and the manifest readers). 5. `replays/`. 6. `model/` and `inference/`. 7. `policy/`, `search/`, `training/` and `selfplay/`. 8. `eval/` and `bridge/`.
9. **Function splits,** one per commit, as pure moves guarded by the parity, sweep and snapshot tests.
10. **Docs:** CLAUDE.md "Run" line and Architecture, README Layout, package docstrings.

**Open branches, relative to 90ff321.**

| branch | touches | conflicts with |
|---|---|---|
| exp/turn-value (running on box 52605483) | modifies `tools/game_record.py` (+46/-15) and `tools/turn_gap.py` (+185/-39); adds playout_reads, turn_value, turn_value_data, turn_value_fit (1,820 lines importing game_record, replay_dataset, scenario_pool, wesnoth_sim, eval_sim, bench_pipeline, raw_player, sim_self_play) and a 789-line box script | steps 1a/1b/1f (turn_gap's imports), 3-8 (game_record move, replay_dataset split). Uses the test-only `game_record.read_records`, `turn_starts` and `walk`. |
| exp/xod-dominance | new files only: combat_dominance, dominance_rewrites, analysis/dominance_count, a test, and 14 `.py` files under training/metrics/xod_20260924/ (code in a records directory) | no textual conflict, but every move of replay_dataset, combat_outcomes, game_record, pathfind_sim or abilities needs a rewrite pass at merge; deleting swap_detector or rewards breaks it |
| feature/signal-telemetry | `tools/supervised_train.py` (+101/-52), `tools/signal_telemetry.py` (+241/-15), `w.imitation_loss`, two signal_profiler files, new tests/imitation_helpers.py (imports bench_infer) | the supervised_train split, the signal_telemetry move, deleting bench_infer or value_grounding |
| land/shutdown-drains-results | BACKLOG.md and `__init__.py` only | none. It is a superseded duplicate landing: its tip ef75d57 has the same two parents as 7e5e5e4, which is on main. |

All other local branches (claude/*, worktree-agent-*, fix/*, docs/audit-corrections, land/audit-batch) are already merged into 90ff321.

**Decisions for the owner, and where I push back.**
- **The quarantine rule blocks moving quarantined code.** quarantine/README.md says quarantined code "stays where it is". The tag `pre-restart-20260903` preserves the tree INVENTORY reads, so relocating or deleting that code needs a ruling that lifts the rule. Until then, the refactor can move live code around it; quarantined modules can stay in tools/ and import from the new packages.
- **Timing.** Two experiment branches are adding about 3,000 lines of new library code to tools/. I would land or park them before step 3; steps 0-2 are small and can go first.
- **Scope.** Moving the flat wesnoth_ai modules costs 763 statements for a moderate navigation gain. I would do it only for the SIM group (classes, combat, visibility, observe, game_core), in step 4.
- **Probe scripts.** Moving the 48 unused entry points into a subdirectory would break command lines in archive docs (82 distinct paths). Either leave them in tools/ or delete them.
