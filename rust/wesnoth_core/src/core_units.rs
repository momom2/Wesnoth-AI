//! Units made and changed on the core: a recruit
//! (`_apply_command(["recruit", ...])`), the pick-advance command, a
//! plague corpse (`_spawn_plague_corpse`) and advancement
//! (`_maybe_advance_unit` / `_advance_unit_once`: the choice queue, the
//! pick-advance narrowing, the uniform self-play draw, AMLA, variation
//! persistence, the feeding bonus, traits and [object] effects re-applied
//! in their order). The Python applier is the oracle
//! (tests/test_rust_units.py, tools/diff_core.py).

use pyo3::prelude::*;
use std::sync::Arc;

use crate::combat::Mt19937;
use crate::core::{sorted, DefTable, GameCore, UnitRec};
use crate::effects::apply_effect;
use crate::units::{
    apply_traits, attacks_from_type, build_plague_corpse, build_recruit_unit, defenses_of, numeric_uid,
    resistances_of, scaled_max_exp, seed_int_of, sha256_hex,
};

/// Statuses advancement clears (advancement.cpp:319-326 heal_fully and
/// the three set_state calls; stunned with them in the Python applier).
const CLEARED_ON_ADVANCE: [&str; 4] = ["poisoned", "slowed", "petrified", "stunned"];

/// `pickadvance`'s type list: comma-separated, blanks and "null" dropped.
fn split_types(s: &str) -> Vec<String> {
    s.split(',').map(|x| x.trim()).filter(|x| !x.is_empty() && *x != "null").map(String::from).collect()
}

/// AMLA_DEFAULT (data/core/macros/amla.cfg): +3 maximum hit points and a
/// full heal, the experience cap raised 20% (compounding, div100rounded),
/// poison and slow cured.
pub(crate) fn amla(u: &mut UnitRec) {
    let new_max = u.max_hp + 3;
    let new_exp = u.max_exp + (u.max_exp * 20 + 50).div_euclid(100);
    u.current_exp = (u.current_exp - u.max_exp).max(0);
    u.max_hp = new_max;
    u.current_hp = new_max;
    u.max_exp = new_exp;
    u.drop_status("poisoned");
    u.drop_status("slowed");
}

impl GameCore {
    /// One past the largest `u<n>` id on the board (the applier's uid
    /// for a new unit).
    pub fn next_uid(&self) -> i64 {
        self.units.iter().filter_map(|u| numeric_uid(&u.id)).max().unwrap_or(0) + 1
    }

    /// `_draw_uniform_advance`: the self-play advancement draw on its own
    /// counter, salted like combat's seeds.
    fn draw_uniform_advance(&mut self, n: usize) -> usize {
        self.global.advance_counter += 1;
        let counter = self.global.advance_counter;
        let base = if self.global.advance_salt.is_empty() {
            format!("sim_advance:{counter}")
        } else {
            format!("sim_advance:{}:{counter}", self.global.advance_salt)
        };
        let mut rng = Mt19937::new(seed_int_of(&sha256_hex(&base, 8)), 0);
        rng.next_u32() as usize % n
    }

    fn pop_choice(&mut self) -> Option<i64> {
        if self.advance_choices.is_empty() { None } else { Some(self.advance_choices.remove(0)) }
    }

    /// The game-wide pick-advance list of (side, type), when non-empty.
    fn game_pick(&self, side: i64, unit_type: &str) -> Option<Vec<String>> {
        self.pickadvance_game.iter()
            .find(|(s, t, l)| *s == side && t == unit_type && !l.is_empty())
            .map(|(_, _, l)| l.clone())
    }

    /// The types `u` is offered when it advances: its type's list,
    /// narrowed by its pick-advance list to the types still in it. Empty
    /// for a unit at its last level (AMLA).
    pub(crate) fn advance_targets(&self, u: &UnitRec) -> Vec<String> {
        let targets = self.db.get(&u.name).advances_to.clone();
        let pick: Vec<String> = u.pickadvance.as_ref()
            .map(|p| p.iter().filter(|x| targets.contains(x)).cloned().collect())
            .unwrap_or_default();
        if pick.is_empty() { targets } else { pick }
    }

    /// `_advance_unit_once` on a detached record: the choice (the queue,
    /// else the self-play draw, else the first type) and its event, then
    /// the advance.
    fn advance_once(&mut self, u: &mut UnitRec) {
        let targets = self.advance_targets(u);
        if targets.is_empty() {
            // AMLA's [choose] still pops the queue, and the recorded
            // value is what its event reports.
            let choice = self.pop_choice().unwrap_or(0);
            self.last_advance_events.push((u.side, choice));
            amla(u);
            return;
        }
        let k = match self.pop_choice() {
            Some(v) if v >= 0 && (v as usize) < targets.len() => v as usize,
            Some(_) => 0,
            None if targets.len() > 1 && self.global.advance_uniform => self.draw_uniform_advance(targets.len()),
            None => 0,
        };
        let index = targets.iter().position(|x| *x == targets[k]).unwrap_or(0) as i64;
        self.last_advance_events.push((u.side, index));
        self.advance_to(u, &targets[k]);
    }

    /// One advance of `u` to `new_type` (one of `advance_targets`): the
    /// new type's statistics at full health, the experience past the cap
    /// kept, the cap scaled by the game's modifier, the moves left kept,
    /// the feeding bonus, the traits in their rolled order and the
    /// [object] effects re-applied, the statuses advancement cures gone.
    pub(crate) fn advance_to(&self, u: &mut UnitRec, new_type: &str) {
        // The advanced unit re-initializes under the pick-advance mod.
        u.pickadvance = self.game_pick(u.side, new_type);
        let mut new_type = new_type.to_string();
        if let Some((_, var)) = u.name.split_once(':') {
            if !var.is_empty() {
                let candidate = format!("{new_type}:{var}");
                if self.db.contains(&candidate) {
                    new_type = candidate;
                }
            }
        }
        let nt = self.db.get(&new_type);
        let moves_left = u.current_moves;
        let feeding = u.feeding_count.unwrap_or(0);
        let old_traits = u.traits.clone();
        u.name = new_type;
        u.max_hp = nt.hitpoints + feeding;
        u.current_hp = u.max_hp;
        u.current_exp = (u.current_exp - u.max_exp).max(0);
        u.max_exp = scaled_max_exp(nt.experience, self.global.experience_modifier);
        u.max_moves = nt.moves;
        u.current_moves = moves_left;
        u.cost = nt.cost;
        u.alignment = nt.alignment;
        u.levelup_names = nt.advances_to.clone();
        u.attacks = attacks_from_type(&nt);
        u.resistances = resistances_of(&nt);
        u.defenses = defenses_of(&nt);
        u.abilities = sorted(nt.abilities.clone());
        u.traits = Vec::new();
        u.statuses.retain(|s| !CLEARED_ON_ADVANCE.contains(&s.as_str()));
        let mut table: DefTable = nt.defense.clone();
        // Traits in the order they were rolled, the ones the order lacks
        // after it (sorted: the Python walks a set there).
        let trait_ids: Vec<String> = match &u.trait_order {
            Some(order) if !order.is_empty() => {
                let mut ids: Vec<String> = order.iter().filter(|x| old_traits.contains(x)).cloned().collect();
                ids.extend(old_traits.iter().filter(|x| !order.contains(x)).cloned());
                ids
            }
            _ => old_traits,
        };
        if !trait_ids.is_empty() {
            apply_traits(u, &trait_ids, nt.level, &mut table);
            u.trait_order = Some(trait_ids);
        }
        u.def_table = Some(Arc::new(table));
        for eff in u.object_effects.clone() {
            apply_effect(u, &eff);
        }
        u.current_moves = u.max_moves.min(moves_left);
    }

    /// `_maybe_advance_unit` on unit `i` then, when it advanced, its
    /// vision from its hex (advancement.cpp:397-399). Returns whether it
    /// advanced.
    pub fn advance_unit(&mut self, i: usize) -> bool {
        let mut u = self.units[i].clone();
        let mut advanced = false;
        while u.current_exp >= u.max_exp {
            self.advance_once(&mut u);
            advanced = true;
        }
        if !advanced {
            return false;
        }
        self.place_unit_facts(&mut u);
        self.units[i] = u;
        let hex = self.units[i].hex;
        if hex >= 0 {
            self.clear_fog_from(i, &[hex as usize]);
        }
        true
    }

    /// The corpse a plague kill of `dead_name` raises for `side` on
    /// (x, y), with the next unit id.
    pub fn spawn_corpse(&mut self, x: i64, y: i64, side: i64, dead_name: &str) -> PyResult<()> {
        let uid = self.next_uid();
        let u = build_plague_corpse(&self.db, dead_name, side, x, y, uid, &self.game_id,
                                    self.global.experience_modifier);
        self.insert_unit(u)?;
        self.global.next_uid_counter += 1;
        Ok(())
    }

    pub fn unit_pos(&self, id: &str) -> Option<usize> {
        self.unit_index.get(id).copied()
    }
}

#[pymethods]
impl GameCore {
    /// `_apply_command(["recruit", type, x, y, seed])`: the recruit with
    /// its rolled traits, unable to move or attack this turn, the game's
    /// pick-advance list for its type, its vision cleared, the uid counter
    /// advanced and its cost spent. Returns the new unit's id.
    #[pyo3(signature = (unit_type, x, y, seed=""))]
    fn apply_recruit(&mut self, unit_type: &str, x: i64, y: i64, seed: &str) -> PyResult<String> {
        let side = self.global.current_side;
        let uid = self.next_uid();
        let mut u = build_recruit_unit(&self.db, unit_type, side, x, y, uid, &self.game_id, seed,
                                       self.global.experience_modifier);
        u.current_moves = 0;
        u.has_attacked = true;
        if let Some(pick) = self.game_pick(side, unit_type) {
            u.pickadvance = Some(pick);
        }
        let id = u.id.clone();
        let i = self.insert_unit(u)?;
        let hex = self.units[i].hex;
        if hex >= 0 {
            self.clear_fog_from(i, &[hex as usize]);
        }
        self.global.next_uid_counter += 1;
        let cost = self.db.get(unit_type).cost;
        self.spend_gold(side, cost);
        Ok(id)
    }

    /// `_apply_command(["pickadvance", x, y, unit_override, game_override,
    /// is_unit, is_game])` (data/modifications/pick_advance/main.lua:124-148):
    /// the unit's own list, and for a game override the side's list for
    /// the type, set on every unit of it (an empty list clears theirs).
    fn apply_pickadvance(&mut self, x: i64, y: i64, unit_override: &str, game_override: &str, is_unit: bool,
                         is_game: bool) {
        let i = match self.unit_at(x, y) {
            Some(i) => i,
            None => return,
        };
        let side = self.units[i].side;
        let base = self.units[i].name.split(':').next().unwrap_or("").to_string();
        let u_list = split_types(unit_override);
        if is_unit && !u_list.is_empty() {
            self.units[i].pickadvance = Some(u_list.clone());
        }
        if is_game {
            let g_list = split_types(game_override);
            match self.pickadvance_game.iter_mut().find(|(s, t, _)| *s == side && *t == base) {
                Some(entry) => entry.2 = g_list,
                None => self.pickadvance_game.push((side, base.clone(), g_list)),
            }
            for u in self.units.iter_mut() {
                if u.side == side && u.name.split(':').next().unwrap_or("") == base {
                    if !u_list.is_empty() {
                        u.pickadvance = Some(u_list.clone());
                    } else {
                        u.pickadvance = None;
                    }
                }
            }
        }
    }

    /// `_maybe_advance_unit` on a unit of this core (the differential
    /// tests' handle). Returns whether it advanced.
    fn advance_unit_id(&mut self, id: &str) -> PyResult<bool> {
        let i = self.unit_pos(id).ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(id.to_string()))?;
        Ok(self.advance_unit(i))
    }
}
