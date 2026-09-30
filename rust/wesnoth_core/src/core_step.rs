//! Phase 4: commands applied on the Rust-owned state (docs/
//! rust_port_plan.md). Each method transcribes the branch of
//! `tools/replay_dataset._apply_command` it replaces, which stays the
//! oracle (tools/diff_core.py replays the corpus through both and
//! compares after every command). Rules cite the Python, which cites
//! the engine.
//!
//! The scenario's turn events fire at the engine's moments (events.rs;
//! docs/wesnoth_rules.md "Turn events"): at side 1's init_side the end
//! of the turn before, then the side's turn forms, after the refresh,
//! healing and income the four refresh forms; at end_turn the four end
//! forms.

use pyo3::prelude::*;

use crate::core::{GameCore, DEFAULT_CYCLE, TOD_NAMES};

pub const REST_HEAL_AMOUNT: i64 = 2;

/// Engagement telemetry kinds of `last_heal_events` rows.
pub const HEAL_EVENT: i64 = 0;
pub const POISON_CURED_EVENT: i64 = 1;
pub const POISON_DAMAGE_EVENT: i64 = 2;

/// `scenario_events.side_turn_event_names`.
fn side_turn_names(side: i64, turn: i64, new_turn: bool) -> Vec<String> {
    let mut names = if new_turn { vec![format!("turn {turn}"), "new turn".to_string()] } else { Vec::new() };
    names.extend([
        "side turn".to_string(), format!("side {side} turn"), format!("side turn {turn}"),
        format!("side {side} turn {turn}"),
    ]);
    names
}

/// `scenario_events.turn_refresh_event_names`.
fn turn_refresh_names(side: i64, turn: i64) -> Vec<String> {
    vec!["turn refresh".to_string(), format!("side {side} turn refresh"), format!("turn {turn} refresh"),
         format!("side {side} turn {turn} refresh")]
}

/// `scenario_events.side_turn_end_event_names`.
fn side_turn_end_names(side: i64, turn: i64) -> Vec<String> {
    vec!["side turn end".to_string(), format!("side {side} turn end"), format!("side turn {turn} end"),
         format!("side {side} turn {turn} end")]
}

/// `scenario_events.turn_end_event_names`.
fn turn_end_names(turn: i64) -> Vec<String> {
    vec!["turn end".to_string(), format!("turn {turn} end")]
}
pub const REGENERATE_AMOUNT: i64 = 8;
pub const POISON_AMOUNT: i64 = 8;

impl GameCore {
    /// `_tod_cycle_index`: the phase of the default cycle for a turn.
    /// The start slot wraps as the engine wraps `current_time`
    /// (`fix_time_index`, a modulo that is never negative).
    pub fn tod_index(&self, turn: i64) -> usize {
        (turn.max(1) - 1 + self.global.tod_start_offset).rem_euclid(6) as usize
    }

    /// `_lawful_bonus_at`: the hex's lawful bonus for a turn, from its
    /// time area's cycle or the default one, then the terrain light
    /// (`terrain_resolver.terrain_light_bonus`, the engine's bounded_add).
    /// An area's cycle is never empty (`game_core.map_static` registers
    /// only non-empty ones) and comes phased to turn 1: an area keeps its
    /// own slot, and the board's start slot moves the default cycle only.
    pub fn lawful_bonus_at(&self, hex: i64, turn: i64) -> i64 {
        let map = &self.map;
        let base = if hex >= 0 && map.area_cycle[hex as usize] >= 0 {
            let cyc = &map.cycles[map.area_cycle[hex as usize] as usize];
            let idx = (turn.max(1) - 1).rem_euclid(cyc.len() as i64) as usize;
            cyc[idx]
        } else {
            DEFAULT_CYCLE[self.tod_index(turn)]
        };
        if hex < 0 || map.has_light[hex as usize] == 0 {
            return base;
        }
        let h = hex as usize;
        let light = map.light_mod[h];
        if light >= 0 {
            (base + light).min(base.max(map.light_max[h]))
        } else {
            (base + light).max(base.min(map.light_min[h]))
        }
    }

    /// The unit's level from its type (`_stats_for(u.name)["level"]`, 1 unknown).
    pub fn unit_level(&self, idx: usize) -> i64 {
        let t = self.units[idx].type_idx;
        if t < 0 {
            return 1;
        }
        self.types.read().unwrap()[t as usize].level
    }

    /// The unit's abilities: its own set plus its type's (as the
    /// healing branch unions them for regeneration).
    fn has_ability_or_type(&self, idx: usize, name: &str) -> bool {
        let u = &self.units[idx];
        if u.has_ability(name) {
            return true;
        }
        let t = u.type_idx;
        t >= 0 && self.types.read().unwrap()[t as usize].abilities.iter().any(|a| a == name)
    }

    /// Indices of the units on the six neighbours of `hex`.
    pub fn adjacent_units(&self, hex: i64, occupant: &[i64]) -> Vec<usize> {
        let mut out = Vec::new();
        if hex < 0 {
            return out;
        }
        let h = hex as usize;
        for &nb in &self.map.nbrs[h * 6..h * 6 + 6] {
            if nb >= 0 && occupant[nb as usize] >= 0 {
                out.push(occupant[nb as usize] as usize);
            }
        }
        out
    }

    /// Unit index per hex (-1 empty); the god-view board.
    pub fn occupancy(&self) -> Vec<i64> {
        let mut occ = vec![-1i64; self.map.h];
        for (i, u) in self.units.iter().enumerate() {
            if u.hex >= 0 {
                occ[u.hex as usize] = i as i64;
            }
        }
        occ
    }
}

#[pymethods]
impl GameCore {
    /// `_apply_command(["init_side", side])`: at side 1 the end of the
    /// turn before (its events), the side to move, the recruit rejections
    /// cleared, the turn counter and time of day at side 1, the side's
    /// turn events, healing (heal.cpp::calculate_healing as the Python
    /// transcribes it, its telemetry in `last_heal_events`), the move
    /// refresh, income and upkeep (play_controller.cpp:524-534), the
    /// side's revealed hiders hidden again after turn 1, the refresh
    /// events, the side's fog recalculated, and what each side sees
    /// recorded (core_sight.rs).
    fn apply_init_side(&mut self, side: i64) -> PyResult<()> {
        if side == 1 && self.global.turn_number >= 1 {
            self.fire_all(&turn_end_names(self.global.turn_number))?;
        }
        self.global.current_side = side;
        self.recruit_rejected = vec![0; self.map.h];
        self.pending_vision.clear();            // undo_list::new_side_turn, undo.cpp:243-262
        if side == 1 {
            self.global.turn_number += 1;
            self.global.time_of_day = TOD_NAMES[self.tod_index(self.global.turn_number)].to_string();
        }
        let turn = self.global.turn_number;
        self.fire_all(&side_turn_names(side, turn, side == 1))?;
        self.last_heal_events.clear();
        let first_turn = turn <= 1;
        let do_healing = self.global.did_first_init_side;
        self.global.did_first_init_side = true;
        if !(first_turn && !do_healing) {
            let occ = self.occupancy();
            let n = self.units.len();
            // Facts of the snapshot before any unit changes (the Python
            // reads `gs.map.units` as it was).
            let mut plan: Vec<(usize, i64, bool, bool)> = Vec::new(); // (idx, new_hp, cure, refresh)
            for i in 0..n {
                let u = &self.units[i];
                if u.side != side || u.has_status("petrified") {
                    continue;
                }
                let terrain_heal = if u.hex >= 0 { self.map.heal[u.hex as usize] } else { 0 };
                let has_regen = self.has_ability_or_type(i, "regenerate");
                let mut healer_amt = 0;
                let mut has_curer = false;
                for j in self.adjacent_units(u.hex, &occ) {
                    let a = &self.units[j];
                    if a.side != u.side || a.has_status("petrified") {
                        continue;
                    }
                    if a.has_ability("cures") {
                        has_curer = true;
                    }
                    if a.has_ability("cures") || a.has_ability("heals_8") || a.has_ability("heals+8") {
                        healer_amt = healer_amt.max(8);
                    } else if a.has_ability("heals_4") || a.has_ability("heals+4") {
                        healer_amt = healer_amt.max(4);
                    }
                }
                if u.has_ability("regenerate") {
                    healer_amt = 0;               // healer_heal_amount: regeneration is self-healing
                }
                let poisoned = u.has_status("poisoned");
                let rest_eligible = u.has_status("resting") || u.has_trait("healthy");
                let mut healing = if rest_eligible { REST_HEAL_AMOUNT } else { 0 };
                let mut cure = false;
                let mut main = 0;
                if !poisoned {
                    if terrain_heal > 0 {
                        main = main.max(terrain_heal);
                    }
                    if has_regen {
                        main = main.max(REGENERATE_AMOUNT);
                    }
                    main = main.max(healer_amt);
                    healing += main;
                } else if terrain_heal > 0 || has_regen || has_curer {
                    cure = true;
                } else if healer_amt > 0 {
                    // slowed poison: no damage, no healing
                } else {
                    healing -= POISON_AMOUNT;
                }
                let max_heal = (u.max_hp - u.current_hp).max(0);
                let min_heal = (1 - u.current_hp).min(0);
                healing = healing.clamp(min_heal, max_heal);
                // The engagement telemetry of the applied heal: rest first,
                // the rest to the village or the ability that gave the most.
                if healing > 0 {
                    let rest = (if rest_eligible { REST_HEAL_AMOUNT } else { 0 }).min(healing);
                    let main_applied = healing - rest;
                    let village = main_applied > 0 && terrain_heal > 0 && terrain_heal == main;
                    self.last_heal_events.push((HEAL_EVENT, u.side, if village { main_applied } else { 0 },
                                                if village { 0 } else { main_applied }, rest));
                }
                if cure {
                    self.last_heal_events.push((POISON_CURED_EVENT, u.side, 0, 0, 0));
                } else if poisoned && healing < 0 {
                    self.last_heal_events.push((POISON_DAMAGE_EVENT, u.side, -healing, 0, 0));
                }
                plan.push((i, u.current_hp + healing, cure, !first_turn));
            }
            for (i, new_hp, cure, refresh) in plan {
                let u = &mut self.units[i];
                u.current_hp = new_hp;
                if cure {
                    u.drop_status("poisoned");
                }
                u.add_status("resting");
                if refresh {
                    u.current_moves = u.max_moves;
                    u.has_attacked = false;
                }
            }
        }
        if !first_turn {
            // unit::new_turn clears STATE_UNCOVERED (unit.cpp:1277)
            // inside board_.new_turn's turn() > 1 gate.
            let own: Vec<String> = self.units.iter().filter(|u| u.side == side).map(|u| u.id.clone()).collect();
            self.uncovered.retain(|id| !own.contains(id));
        }
        if side >= 1 && (side as usize) <= self.sides.len() && turn > 1 {
            let owned = self.sides[side as usize - 1].nb_villages;
            // A declared 0 is paid as 0 (team.cpp:236 and :239-244 default
            // only a missing value); the state always carries a value, the
            // default having been taken where the record was read.
            let village_gold = self.global.village_gold;
            let village_support = self.global.village_upkeep;
            let income = self.sides[side as usize - 1].base_income + owned * village_gold;
            let mut upkeep = 0;
            for i in 0..self.units.len() {
                let u = &self.units[i];
                if u.side != side || u.is_leader || u.has_trait("loyal") {
                    continue;
                }
                upkeep += self.unit_level(i);
            }
            let net_upkeep = (upkeep - owned * village_support).max(0);
            let s = &mut self.sides[side as usize - 1];
            s.current_gold += income - net_upkeep;
        }
        self.fire_all(&turn_refresh_names(side, turn))?;
        self.refog(side);                       // play_controller.cpp:524-525
        self.note_sightings();
        Ok(())
    }

    /// The last init_side's engagement telemetry: (kind, side, a, b, c)
    /// rows, kind 0 a heal (village, ability, rest), 1 a cured poison, 2
    /// a poison's damage (a).
    fn heal_events(&self) -> Vec<(i64, i64, i64, i64, i64)> {
        self.last_heal_events.clone()
    }

    /// `_apply_command(["end_turn"])`: the ending side's units lose
    /// `slowed`, and `resting` when they moved; its end events fire; then
    /// its fog is recalculated and its sighting record starts over.
    fn apply_end_turn(&mut self) -> PyResult<()> {
        let side = self.global.current_side;
        self.clear_undo_stack();                // play_controller.cpp:576-577
        for u in self.units.iter_mut() {
            if u.side != side {
                continue;
            }
            u.drop_status("slowed");
            if u.current_moves != u.max_moves {
                u.drop_status("resting");
            }
        }
        self.fire_all(&side_turn_end_names(side, self.global.turn_number))?;
        self.refog(side);                       // play_controller.cpp:582-590
        self.clear_sightings(side);
        self.note_sightings();
        Ok(())
    }

    /// The lawful bonus of a hex (x, y) for a turn: the test oracle
    /// against `_lawful_bonus_at`.
    fn lawful_bonus(&self, x: i64, y: i64, turn: i64) -> i64 {
        let hex = self.map.pos_index.get(&(x, y)).map(|&i| i as i64).unwrap_or(-1);
        self.lawful_bonus_at(hex, turn)
    }
}
