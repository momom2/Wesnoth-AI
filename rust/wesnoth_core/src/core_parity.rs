//! The parity observation (docs/parity_memory_design_20260929.md "What
//! the network observes"): what `GameCore.encode_streams` adds under
//! `observation_parity` to the unit, recruit, hex and global rows, the
//! recruit options' base values, the sighting stream and the relevant hex
//! set version 2. The column layout is encoder.py's ("The parity
//! observation"); `parity_layout` hands the vocabularies and widths to the
//! adapter, which refuses a wheel whose layout differs from the encoder's.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::combat::combat_modifier;
use crate::core::{GameCore, UnitRec};
use crate::core_attack::{apply_illumination, dt_index};
use crate::db::{UnitType, DAMAGE_TYPES};
use crate::terrain::{FUNGUS, REEF};
use crate::units::scaled_max_exp;

/// Weapon specials with a column, in column order (encoder.PARITY_SPECIALS).
pub const SPECIALS: [&str; 10] = [
    "magical", "poison", "slow", "marksman", "firststrike", "drains", "backstab", "charge", "plague", "berserk",
];
/// Traits with a column (encoder.PARITY_TRAITS).
pub const TRAITS: [&str; 13] = [
    "strong", "quick", "intelligent", "resilient", "healthy", "dextrous", "weak", "slow", "dim", "fearless",
    "undead", "feral", "elemental",
];
/// Abilities with a column (encoder.PARITY_ABILITIES).
pub const ABILITIES: [&str; 13] = [
    "leadership", "skirmisher", "regenerate", "submerge", "cures", "ambush", "heals_4", "steadfast", "feeding",
    "illuminates", "nightstalk", "concealment", "teleport",
];
/// The first attacks of a unit, in its attack order (the weapon head's index).
pub const WEAPON_SLOTS: usize = 3;
/// One weapon slot: present, damage, strikes, ranged, the damage type
/// (one-hot), the specials (multi-hot).
pub const WEAPON_COLS: usize = 4 + DAMAGE_TYPES.len() + SPECIALS.len();
const RESIST_AT: usize = WEAPON_SLOTS * WEAPON_COLS;
const TRAITS_AT: usize = RESIST_AT + DAMAGE_TYPES.len();
const ABILITIES_AT: usize = TRAITS_AT + TRAITS.len();
const POISONED_AT: usize = ABILITIES_AT + ABILITIES.len();
const SLOWED_AT: usize = POISONED_AT + 1;
const TOD_AT: usize = SLOWED_AT + 1;
const LEADERSHIP_AT: usize = TOD_AT + 1;
/// The columns a unit or recruit row gains after its first 13.
pub const UNIT_EXTRA: usize = LEADERSHIP_AT + 1;
/// The hex row gains: the side sees the hex now; the hex's lawful bonus
/// minus the board's.
pub const HEX_EXTRA: usize = 2;
/// The global row gains: village gold, village support, own net income,
/// fog on, and with fog off the enemy's gold, net income and upkeep.
pub const GLOBAL_EXTRA: usize = 7;
/// A sighting token's features: hp / max hp, max hp / HP_NORM.
pub const SIGHT_FEAT_DIM: usize = 2;
/// The relevant set version 2 keeps every hex the side does not see
/// within this hex distance of one of its units.
pub const FOG_RADIUS: i64 = 6;
/// The recruit row's has_attacked column (encoder.UNIT_NUMERIC_FEATS 8).
pub const HAS_ATTACKED_COL: usize = 8;
pub const NUM_PARITY_NORMS: usize = 6;

/// The divisors of the parity columns, in `parity_norms` order
/// (encoder.py's module values).
pub(crate) struct ParityNorms {
    pub damage: f64,
    pub strikes: f64,
    pub lawful_bonus: f64,
    pub leadership: f64,
    pub village_gold: f64,
    pub village_support: f64,
}

impl ParityNorms {
    pub fn from_array(a: [f64; NUM_PARITY_NORMS]) -> Self {
        ParityNorms {
            damage: a[0], strikes: a[1], lawful_bonus: a[2], leadership: a[3], village_gold: a[4],
            village_support: a[5],
        }
    }
}

/// Hex distance between two offset coordinates of the map (columns with
/// an odd x sit half a hex lower, `observe::neighbours`), through axial
/// coordinates.
pub(crate) fn hex_distance(a: (i64, i64), b: (i64, i64)) -> i64 {
    let axial = |(x, y): (i64, i64)| (x, y - (x - (x & 1)) / 2);
    let ((q1, r1), (q2, r2)) = (axial(a), axial(b));
    let (dq, dr) = (q1 - q2, r1 - r2);
    (dq.abs() + dr.abs() + (dq + dr).abs()) / 2
}

fn flag(on: bool) -> f32 {
    if on { 1.0 } else { 0.0 }
}

/// Each name of `vocab` present in `names`, as 0/1 columns.
fn name_bits(names: &[String], vocab: &[&str], out: &mut [f32]) {
    for (k, v) in vocab.iter().enumerate() {
        out[k] = flag(names.iter().any(|n| n == v));
    }
}

/// One weapon slot's columns.
fn weapon_cols(damage: i64, strikes: i64, ranged: bool, damage_type: Option<usize>, specials: &[String],
               pn: &ParityNorms, out: &mut [f32]) {
    out[0] = 1.0;
    out[1] = (damage as f64 / pn.damage) as f32;
    out[2] = (strikes as f64 / pn.strikes) as f32;
    out[3] = flag(ranged);
    if let Some(t) = damage_type {
        out[4 + t] = 1.0;
    }
    name_bits(specials, &SPECIALS, &mut out[4 + DAMAGE_TYPES.len()..]);
}

/// Resistances as (100 - damage percent) / 100, `DAMAGE_TYPES` order.
fn resist_cols(damage_percent: &[i64; 6], out: &mut [f32]) {
    for (k, &r) in damage_percent.iter().enumerate() {
        out[k] = ((100 - r) as f64 / 100.0) as f32;
    }
}

fn illuminates(u: &UnitRec) -> bool {
    u.has_ability("illuminates") && !u.has_status("petrified")
}

/// A recruit option's parity columns from its type's base values: its
/// attacks, resistances and abilities. Its traits are rolled when it is
/// recruited, and it has no status, time of day or leadership yet.
pub(crate) fn recruit_extra(t: &UnitType, pn: &ParityNorms, out: &mut [f32]) {
    for (k, a) in t.attacks.iter().take(WEAPON_SLOTS).enumerate() {
        weapon_cols(a.damage, a.number, a.ranged, dt_index(&a.type_name), &a.specials, pn,
                    &mut out[k * WEAPON_COLS..(k + 1) * WEAPON_COLS]);
    }
    resist_cols(&t.resist, &mut out[RESIST_AT..TRAITS_AT]);
    name_bits(&t.abilities, &ABILITIES, &mut out[ABILITIES_AT..POISONED_AT]);
}

/// `rows` rows of `base_cols` then `extra_cols` columns, row-major.
pub(crate) fn widen(base: &[f32], base_cols: usize, extra: &[f32], extra_cols: usize) -> Vec<f32> {
    let rows = if base_cols == 0 { 0 } else { base.len() / base_cols };
    let mut out = Vec::with_capacity(rows * (base_cols + extra_cols));
    for r in 0..rows {
        out.extend_from_slice(&base[r * base_cols..(r + 1) * base_cols]);
        out.extend_from_slice(&extra[r * extra_cols..(r + 1) * extra_cols]);
    }
    out
}

impl GameCore {
    /// Unit `i`'s parity columns after its first 13: its weapons (damage
    /// after traits and objects; damage type and specials of the type's
    /// attack at that index with the unit's own specials added, as the
    /// fight reads them), resistances, traits, abilities, poisoned and
    /// slowed, the combat modifier the time of day gives it at its hex and
    /// the leadership bonus it fights with now.
    pub(crate) fn unit_extra(&self, i: usize, pn: &ParityNorms, out: &mut [f32]) {
        let u = &self.units[i];
        if !u.attacks.is_empty() {
            for (k, w) in self.weapons_of(i).iter().take(WEAPON_SLOTS).enumerate() {
                weapon_cols(w.damage, w.number, w.ranged, w.type_idx, &w.specials, pn,
                            &mut out[k * WEAPON_COLS..(k + 1) * WEAPON_COLS]);
            }
        }
        if u.type_idx >= 0 {
            resist_cols(&self.type_rec(u.type_idx).resist, &mut out[RESIST_AT..TRAITS_AT]);
        }
        name_bits(&u.traits, &TRAITS, &mut out[TRAITS_AT..ABILITIES_AT]);
        name_bits(&u.abilities, &ABILITIES, &mut out[ABILITIES_AT..POISONED_AT]);
        out[POISONED_AT] = flag(u.has_status("poisoned"));
        out[SLOWED_AT] = flag(u.has_status("slowed"));
        out[TOD_AT] = (self.time_of_day_modifier(i) as f64 / pn.lawful_bonus) as f32;
        out[LEADERSHIP_AT] = (self.leadership(i) as f64 / pn.leadership) as f32;
    }

    /// The engine's combat modifier for unit `i` at its hex
    /// (attack.cpp `combat_modifier`): its alignment and fearless trait
    /// under the illuminated time of day there, as its fights read it
    /// (`fight_inputs`, the alignment of its type) and the interface
    /// shows it (reports.cpp `attack_info`).
    fn time_of_day_modifier(&self, i: usize) -> i64 {
        let u = &self.units[i];
        let alignment = if u.type_idx >= 0 { self.type_rec(u.type_idx).alignment } else { 1 };
        let bonus = apply_illumination(self.lawful_bonus_at(u.hex, self.global.turn_number), self.illuminated(i));
        combat_modifier(alignment, bonus, u.has_trait("fearless"))
    }

    /// A recruit option's (max hp, max moves, max experience, cost,
    /// alignment) as the recruited unit has them before its traits: the
    /// experience scaled by the game's modifier, the alignment coded as
    /// board units code it (`classes.Alignment`).
    pub(crate) fn recruit_base_stats(&self, name: &str) -> [f64; 5] {
        let t = self.db.get(name);
        [t.hitpoints as f64, t.moves as f64,
         scaled_max_exp(t.experience, self.global.experience_modifier) as f64, t.cost as f64,
         t.alignment as f64]
    }

    /// The hex columns in slot order: whether the side sees the hex now
    /// (every hex with fog off), and the hex's lawful bonus minus the
    /// board's as the interface shows it (`get_visible_time_of_day_at`,
    /// src/reports.cpp:100-112, 1.18.4): on a fogged hex its time area's
    /// alone; on a seen hex with the terrain's light and the illumination
    /// of every unit on or next to it, seen or not
    /// (`get_illuminated_time_of_day`, src/tod_manager.cpp:221-262).
    pub(crate) fn hex_extra(&self, slots: &[usize], seen: &[u8], pn: &ParityNorms) -> Vec<f32> {
        let m = &self.map;
        let mut lit = vec![false; m.h];
        for u in self.units.iter() {
            if u.hex < 0 || !illuminates(u) {
                continue;
            }
            let h = u.hex as usize;
            lit[h] = true;
            for &nb in &m.nbrs[h * 6..h * 6 + 6] {
                if nb >= 0 {
                    lit[nb as usize] = true;
                }
            }
        }
        let turn = self.global.turn_number;
        let board = self.lawful_bonus_at(-1, turn);
        let mut out = vec![0f32; slots.len() * HEX_EXTRA];
        for (t, &mh) in slots.iter().enumerate() {
            let sees = !self.global.fog_on || seen[mh] != 0;
            out[t * HEX_EXTRA] = flag(sees);
            let here = if sees {
                apply_illumination(self.lawful_bonus_at(mh as i64, turn), lit[mh])
            } else {
                self.area_lawful_bonus(mh as i64, turn)
            };
            out[t * HEX_EXTRA + 1] = ((here - board) as f64 / pn.lawful_bonus) as f32;
        }
        out
    }

    /// A side's economy as the status table shows it
    /// (src/display_context.cpp `team_data`): its upkeep (the levels of
    /// its units but leaders and loyal ones) and its net income (base
    /// income plus village gold, minus the upkeep its villages do not
    /// support).
    pub(crate) fn side_economy(&self, side: i64) -> (i64, i64) {
        if side < 1 || side as usize > self.sides.len() {
            return (0, 0);
        }
        let upkeep: i64 = (0..self.units.len())
            .filter(|&i| {
                let u = &self.units[i];
                u.side == side && !u.is_leader && !u.has_trait("loyal")
            })
            .map(|i| self.unit_level(i))
            .sum();
        let s = &self.sides[side as usize - 1];
        let expenses = (upkeep - s.nb_villages * self.global.village_upkeep).max(0);
        (upkeep, s.base_income + s.nb_villages * self.global.village_gold - expenses)
    }

    /// The global row's parity values: village gold, village support, the
    /// side's net income, fog on, and with fog off the enemy's gold, net
    /// income and upkeep (under fog the status table shows none of them,
    /// docs/wesnoth_rules.md "Enemy side statistics under fog or shroud").
    pub(crate) fn global_extra(&self, side: i64, them: i64, gold_norm: f64, income_norm: f64, pn: &ParityNorms)
        -> [f32; GLOBAL_EXTRA] {
        let g = &self.global;
        let (_, own_net) = self.side_economy(side);
        let mut out = [
            (g.village_gold as f64 / pn.village_gold) as f32,
            (g.village_upkeep as f64 / pn.village_support) as f32,
            (own_net as f64 / income_norm) as f32,
            flag(g.fog_on),
            0.0, 0.0, 0.0,
        ];
        if !g.fog_on && them >= 1 && them as usize <= self.sides.len() {
            let (their_upkeep, their_net) = self.side_economy(them);
            out[4] = (self.sides[them as usize - 1].current_gold as f64 / gold_norm) as f32;
            out[5] = (their_net as f64 / income_norm) as f32;
            out[6] = (their_upkeep as f64 / income_norm) as f32;
        }
        out
    }

    /// The relevant hex set version 2 (map space, in place): today's set,
    /// the six neighbours of each of the side's units, and with fog on
    /// every hex the side does not see within `FOG_RADIUS` of one of them.
    pub(crate) fn widen_relevant(&self, side: i64, relevant: &mut [u8], seen: &[u8]) {
        let m = &self.map;
        let own: Vec<usize> = (0..self.units.len())
            .filter(|&i| self.units[i].side == side && self.units[i].hex >= 0 && !self.is_scenery(i))
            .collect();
        for &i in &own {
            let h = self.units[i].hex as usize;
            for &nb in &m.nbrs[h * 6..h * 6 + 6] {
                if nb >= 0 {
                    relevant[nb as usize] = 1;
                }
            }
        }
        if !self.global.fog_on {
            return;
        }
        for h in 0..m.h {
            if relevant[h] != 0 || seen[h] != 0 {
                continue;
            }
            let near = own.iter().any(|&i| {
                let u = &self.units[i];
                hex_distance((u.x, u.y), (m.hx[h], m.hy[h])) <= FOG_RADIUS
            });
            if near {
                relevant[h] = 1;
            }
        }
    }

    /// The sighting stream of `side`: each unit of its record it does not
    /// see now, at the last hex it saw it on, in the unit stream's order
    /// (y, x, id). Returns (vocab ids, x, y (clamped to `map_limit`),
    /// features [S * SIGHT_FEAT_DIM]).
    #[allow(clippy::type_complexity)]
    pub(crate) fn sighting_stream(&self, side: i64, visible: &[u8], type_vocab: &[i64], map_limit: i64,
                                  hp_norm: f64) -> (Vec<i64>, Vec<i64>, Vec<i64>, Vec<f32>) {
        let overflow = type_vocab.iter().copied().max().unwrap_or(0);
        let clamp = |v: i64| v.clamp(0, map_limit);
        let mut rows = self.sightings_not_visible(side, visible);
        rows.sort_by(|a, b| (a.1.y, a.1.x, a.0).cmp(&(b.1.y, b.1.x, b.0)));
        let (mut ids, mut xs, mut ys) = (Vec::new(), Vec::new(), Vec::new());
        let mut feats = Vec::with_capacity(rows.len() * SIGHT_FEAT_DIM);
        for (_id, r) in rows {
            ids.push(type_vocab.get(self.type_idx(&r.type_name)).copied().unwrap_or(overflow));
            xs.push(clamp(r.x));
            ys.push(clamp(r.y));
            feats.push((r.hp as f64 / r.max_hp.max(1) as f64) as f32);
            feats.push((r.max_hp as f64 / hp_norm) as f32);
        }
        (ids, xs, ys, feats)
    }
}

/// The parity observation's layout as the core builds it: the
/// vocabularies in column order and the widths (the adapter compares them
/// with encoder.py's before the first parity encode).
#[pyfunction]
pub fn parity_layout(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("damage_types", DAMAGE_TYPES.to_vec())?;
    d.set_item("specials", SPECIALS.to_vec())?;
    d.set_item("traits", TRAITS.to_vec())?;
    d.set_item("abilities", ABILITIES.to_vec())?;
    d.set_item("weapon_slots", WEAPON_SLOTS)?;
    d.set_item("weapon_cols", WEAPON_COLS)?;
    d.set_item("unit_extra", UNIT_EXTRA)?;
    d.set_item("hex_extra", HEX_EXTRA)?;
    d.set_item("global_extra", GLOBAL_EXTRA)?;
    d.set_item("sight_feat_dim", SIGHT_FEAT_DIM)?;
    d.set_item("fog_radius", FOG_RADIUS)?;
    d.set_item("terrain_classes", REEF + 1)?;
    d.set_item("fungus", FUNGUS)?;
    d.set_item("reef", REEF)?;
    Ok(d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observe::neighbours;

    #[test]
    fn every_neighbour_is_at_distance_one() {
        for &(x, y) in &[(0i64, 0i64), (1, 0), (4, 7), (5, 7), (-1, 3)] {
            for nb in neighbours(x, y) {
                assert_eq!(hex_distance((x, y), nb), 1, "{:?} -> {:?}", (x, y), nb);
            }
            assert_eq!(hex_distance((x, y), (x, y)), 0);
        }
    }
}
