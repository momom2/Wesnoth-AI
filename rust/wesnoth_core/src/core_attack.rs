//! Phase 4: the attack command on the Rust-owned state, a transcription
//! of the attack branch of `tools/replay_dataset._apply_command` and of
//! `build_attack_context` / `_to_combat_unit` (the snapshots, the
//! terrain defense, the lawful bonus with illumination, leadership,
//! backstab) over the combat kernel (combat.rs). The kernel applies the
//! outcome to the two units (hit points, experience, statuses, feeding,
//! deaths) and reports what stays Python's: the advancement of a
//! survivor at its experience cap and the corpse a plague kill raises
//! (`wesnoth_ai.game_core.CoreState._apply_attack`).

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::combat::{resolve_fight, UNIT_FLAGS, UNIT_INTS};
use crate::core::{BaseAttack, GameCore};
use crate::observe::neighbours;

/// `combat.DAMAGE_TYPES`: the resistance order of a unit type.
const DAMAGE_TYPES: [&str; 6] = ["blade", "pierce", "impact", "fire", "cold", "arcane"];
/// `combat._UNIT_FLAGS`: the kernel's flag order.
const UNIT_FLAG_NAMES: [&str; UNIT_FLAGS] = [
    "slowed", "poisoned", "petrified", "invulnerable", "fearless", "undrainable", "unpoisonable",
    "steadfast", "magical", "marksman", "deflect", "backstab", "charge", "swarm", "drains",
    "plague", "poison", "slow", "petrifies", "firststrike", "berserk",
];
/// `replay_dataset._UNPLAGUEABLE_TRAITS`: their macros add the
/// unplagueable, undrainable and unpoisonable statuses.
const UNPLAGUEABLE_TRAITS: [&str; 3] = ["undead", "mechanical", "elemental"];
/// The [illuminates] ability's value and cap (tod_manager.cpp:265-281).
const ILLUMINATION: i64 = 25;

/// `combat.Weapon` as `_to_combat_unit` builds it: the type's attack
/// by index (damage type, specials, accuracy, parry) with the unit's
/// numbers and its own specials unioned in.
struct WeaponView {
    type_idx: Option<usize>,         // into DAMAGE_TYPES; None = a type no resistance names
    damage: i64,
    number: i64,
    accuracy: i64,
    parry: i64,
    specials: Vec<String>,
}

fn dt_index(name: &str) -> Option<usize> {
    DAMAGE_TYPES.iter().position(|d| *d == name)
}

/// `apply_unit_illumination`: bounded_add(base, 25, max 25, min 0), positive branch.
pub(crate) fn apply_illumination(base: i64, illuminated: bool) -> i64 {
    if illuminated { (base + ILLUMINATION).min(base.max(ILLUMINATION)) } else { base }
}

impl GameCore {
    fn weapons_of(&self, i: usize) -> Vec<WeaponView> {
        let u = &self.units[i];
        let types = self.types.read().unwrap();
        let base: &[BaseAttack] = if u.type_idx >= 0 { &types[u.type_idx as usize].attacks } else { &[] };
        let mut out = Vec::with_capacity(u.attacks.len());
        for (k, a) in u.attacks.iter().enumerate() {
            let (type_idx, mut specials, accuracy, parry) = if k < base.len() {
                (dt_index(&base[k].type_name), base[k].specials.clone(), base[k].accuracy, base[k].parry)
            } else {
                let t = if (0..6).contains(&a.type_id) { a.type_id as usize } else { 0 };
                (Some(t), Vec::new(), 0, 0)
            };
            for s in &a.specials {
                if !specials.contains(s) {
                    specials.push(s.clone());
                }
            }
            out.push(WeaponView { type_idx, damage: a.damage, number: a.strikes, accuracy, parry, specials });
        }
        if out.is_empty() {
            // `Weapon("none", 1, 1, "melee", "blade", [])`
            out.push(WeaponView { type_idx: Some(0), damage: 1, number: 1, accuracy: 0, parry: 0, specials: Vec::new() });
        }
        out
    }

    /// `combat._unit_arrays(me, opp, weapon)`: the kernel's integers
    /// and flags for one combatant.
    fn combat_arrays(&self, i: usize, opp: usize, weapon: Option<&WeaponView>, defense_pct: i64)
        -> ([i64; UNIT_INTS], [u8; UNIT_FLAGS]) {
        let u = &self.units[i];
        let o = &self.units[opp];
        let types = self.types.read().unwrap();
        let (level, alignment) = if u.type_idx >= 0 {
            let t = &types[u.type_idx as usize];
            (t.level, t.alignment)
        } else {
            (1, 1)
        };
        let opp_resist = match weapon.and_then(|w| w.type_idx) {
            Some(t) if o.type_idx >= 0 => types[o.type_idx as usize].resist[t],
            _ => 100,
        };
        let ints = [
            u.current_hp, u.max_hp, level, u.current_exp, u.max_exp, alignment, defense_pct, opp_resist,
            weapon.map_or(0, |w| w.damage), weapon.map_or(0, |w| w.number),
            weapon.map_or(0, |w| w.accuracy), weapon.map_or(0, |w| w.parry),
        ];
        let unplagueable_trait = u.traits.iter().any(|t| UNPLAGUEABLE_TRAITS.contains(&t.as_str()));
        let mut flags = [0u8; UNIT_FLAGS];
        for (k, name) in UNIT_FLAG_NAMES.iter().enumerate() {
            let on = match *name {
                "slowed" => u.has_status("slowed"),
                "poisoned" => u.has_status("poisoned"),
                "petrified" | "invulnerable" => false,      // the snapshot never carries them
                "fearless" => u.has_trait("fearless"),
                "undrainable" => u.has_status("undrainable") || unplagueable_trait,
                "unpoisonable" => u.has_status("unpoisonable") || unplagueable_trait,
                "steadfast" => u.has_ability("steadfast"),
                special => weapon.map_or(false, |w| w.specials.iter().any(|s| s == special)),
            };
            flags[k] = on as u8;
        }
        (ints, flags)
    }

    /// `_terrain_def_pct` on the unit's hex: its movement class's
    /// defense percentage there.
    fn defense_at(&self, i: usize) -> i64 {
        let u = &self.units[i];
        if u.hex < 0 || u.class_id < 0 {
            return 50;
        }
        self.classes.read().unwrap()[u.class_id as usize].defense_pct[u.hex as usize]
    }

    /// `abilities.leadership_bonus`: 25 per level an adjacent,
    /// higher-level, unpetrified leader of the same side has over
    /// the unit; the best one, not cumulative.
    fn leadership(&self, i: usize) -> i64 {
        let u = &self.units[i];
        let level = self.unit_level(i);
        let adj = neighbours(u.x, u.y);
        let mut best = 0;
        for j in 0..self.units.len() {
            let a = &self.units[j];
            if j == i || a.side != u.side || !adj.contains(&(a.x, a.y)) || a.has_status("petrified")
                || !a.has_ability("leadership") {
                continue;
            }
            let ally_level = self.unit_level(j);
            if level < ally_level {
                best = best.max(25 * (ally_level - level));
            }
        }
        best
    }

    /// `abilities.illuminate_step`: the unit or any adjacent unit of
    /// any side illuminates and is not petrified.
    pub(crate) fn illuminated(&self, i: usize) -> bool {
        let u = &self.units[i];
        if u.has_ability("illuminates") && !u.has_status("petrified") {
            return true;
        }
        let adj = neighbours(u.x, u.y);
        self.units.iter().any(|o| adj.contains(&(o.x, o.y)) && o.has_ability("illuminates") && !o.has_status("petrified"))
    }

    /// `abilities.is_backstab_active`: an unpetrified enemy of the
    /// defender on the hex opposite the attacker.
    fn backstab_active(&self, attacker: usize, defender: usize) -> bool {
        let a = &self.units[attacker];
        let d = &self.units[defender];
        let nb = neighbours(d.x, d.y);
        let idx = match nb.iter().position(|&p| p == (a.x, a.y)) {
            Some(k) => k,
            None => return false,
        };
        let opp = nb[(idx + 3) % 6];
        self.units.iter().any(|f| (f.x, f.y) == opp && f.side != d.side && !f.has_status("petrified"))
    }

    /// `_is_unplagueable`: the status, or a trait whose macro adds it.
    fn unplagueable(&self, i: usize) -> bool {
        let u = &self.units[i];
        u.has_status("unplagueable") || u.traits.iter().any(|t| UNPLAGUEABLE_TRAITS.contains(&t.as_str()))
    }

    /// `_spawn_plague_corpse`'s eligibility: a plagueable victim whose
    /// type has an undead variation, killed off a village.
    fn plague_eligible(&self, i: usize) -> bool {
        if self.unplagueable(i) {
            return false;
        }
        let u = &self.units[i];
        if u.type_idx >= 0 {
            let types = self.types.read().unwrap();
            if types[u.type_idx as usize].undead_variation.to_lowercase() == "null" {
                return false;
            }
        }
        !(u.hex >= 0 && self.map.village_terrain[u.hex as usize] != 0)
    }
}

#[pymethods]
impl GameCore {
    /// `_apply_command(["attack", ax, ay, dx, dy, a_weapon, d_weapon,
    /// seed, choices])`. The choices join the advancement queue first;
    /// no unit on either hex or no seed (an attack aborted mid-way)
    /// ends the command there. Returns None then, else the facts the
    /// wrapper finishes with: ids, names, sides, costs, positions
    /// before the fight, who lives, who feeds, who advances, whether
    /// a plague corpse rises on either hex, the damage each took.
    #[pyo3(signature = (ax, ay, dx, dy, a_weapon, d_weapon, seed, has_seed, choices))]
    #[allow(clippy::too_many_arguments)]
    fn apply_attack<'py>(&mut self, py: Python<'py>, ax: i64, ay: i64, dx: i64, dy: i64, a_weapon: i64,
                         d_weapon: i64, seed: u32, has_seed: bool, choices: Vec<i64>)
        -> PyResult<Option<Bound<'py, PyDict>>> {
        self.advance_choices.extend(choices);
        let (a, d) = match (self.unit_at(ax, ay), self.unit_at(dx, dy)) {
            (Some(a), Some(d)) => (a, d),
            _ => return Ok(None),
        };
        if !has_seed {
            return Ok(None);
        }
        let att_id = self.units[a].id.clone();
        let dfd_id = self.units[d].id.clone();
        let (att_side, dfd_side) = (self.units[a].side, self.units[d].side);
        self.track_side(att_side);
        self.track_side(dfd_side);
        let dfd_was_slowed = self.units[d].has_status("slowed");
        let dfd_was_petrified = self.units[d].has_status("petrified");
        self.uncover(&att_id);                  // attack.cpp:1378
        let aw = self.weapons_of(a);
        let dw = self.weapons_of(d);
        let a_idx = if a_weapon >= aw.len() as i64 {
            0
        } else if a_weapon < 0 {
            let k = aw.len() as i64 + a_weapon;
            if k < 0 {
                return Err(pyo3::exceptions::PyIndexError::new_err("attacker weapon"));
            }
            k as usize
        } else {
            a_weapon as usize
        };
        let mut d_idx = d_weapon;
        if d_idx >= 0 && d_idx >= dw.len() as i64 {
            d_idx = 0;
        }
        if self.units[d].has_status("petrified") {
            d_idx = -1;
        }
        let d_has = d_idx >= 0;
        let d_wv = if d_has { Some(&dw[d_idx as usize]) } else { None };
        let (a_ints, a_flags) = self.combat_arrays(a, d, Some(&aw[a_idx]), self.defense_at(a));
        let (d_ints, d_flags) = self.combat_arrays(d, a, d_wv, self.defense_at(d));
        let turn = self.global.turn_number;
        let a_lawful = apply_illumination(self.lawful_bonus_at(self.units[a].hex, turn), self.illuminated(a));
        let d_lawful = apply_illumination(self.lawful_bonus_at(self.units[d].hex, turn), self.illuminated(d));
        let (out, record) = resolve_fight(
            &a_ints, &a_flags, &d_ints, &d_flags, d_has, a_lawful, d_lawful,
            self.leadership(a), self.leadership(d), self.backstab_active(a, d), self.backstab_active(d, a),
            seed, 0,
        );
        self.last_checkup_strikes = record;
        let (a_hp, d_hp, a_xp, d_xp) = (out[0], out[1], out[2], out[3]);
        let (a_alive, d_alive) = (a_hp > 0, d_hp > 0);
        let att_feed = !d_alive && self.units[a].has_ability("feeding") && !self.unplagueable(d);
        let dfd_feed = !a_alive && self.units[d].has_ability("feeding") && !self.unplagueable(a);
        let plague_forward = !d_alive && out[10] != 0 && a_alive && self.plague_eligible(d);
        let plague_reverse = !a_alive && out[11] != 0 && self.plague_eligible(a);
        let r = PyDict::new(py);
        {
            let (att, dfd) = (&self.units[a], &self.units[d]);
            r.set_item("att_id", &att_id)?;
            r.set_item("dfd_id", &dfd_id)?;
            r.set_item("att_name", &att.name)?;
            r.set_item("dfd_name", &dfd.name)?;
            r.set_item("att_side", att.side)?;
            r.set_item("dfd_side", dfd.side)?;
            r.set_item("att_cost", att.cost)?;
            r.set_item("dfd_cost", dfd.cost)?;
            r.set_item("att_x", att.x)?;
            r.set_item("att_y", att.y)?;
            r.set_item("dfd_x", dfd.x)?;
            r.set_item("dfd_y", dfd.y)?;
            r.set_item("att_alive", a_alive)?;
            r.set_item("dfd_alive", d_alive)?;
            r.set_item("att_feed", att_feed)?;
            r.set_item("dfd_feed", dfd_feed)?;
            r.set_item("att_advances", a_alive && a_xp >= att.max_exp)?;
            r.set_item("dfd_advances", d_alive && d_xp >= dfd.max_exp)?;
            r.set_item("plague_forward", plague_forward)?;
            // attack.cpp:1456-1458: the defender's side refogs when the
            // defender died, was slowed or was petrified in the fight;
            // the wrapper does it after the corpses rise.
            r.set_item("dfd_refog", !d_alive || (out[7] != 0 && !dfd_was_slowed)
                                    || (out[9] != 0 && !dfd_was_petrified))?;
            r.set_item("plague_reverse", plague_reverse)?;
            r.set_item("dmg_to_defender", (dfd.current_hp - d_hp).max(0))?;
            r.set_item("dmg_to_attacker", (att.current_hp - a_hp).max(0))?;
        }
        {
            let u = &mut self.units[a];
            u.drop_status("resting");           // attack.cpp:1279-80
            if out[4] != 0 { u.add_status("slowed"); }
            if out[5] != 0 { u.add_status("poisoned"); }
            if out[6] != 0 { u.add_status("petrified"); }
            if a_alive {
                u.has_attacked = true;
                u.current_moves = 0;            // attack.cpp:1372, movement_used = everything
                u.max_hp += att_feed as i64;
                u.current_hp = a_hp + att_feed as i64;
                u.current_exp = a_xp;
            }
        }
        {
            let u = &mut self.units[d];
            u.drop_status("resting");
            if out[7] != 0 { u.add_status("slowed"); }
            if out[8] != 0 { u.add_status("poisoned"); }
            if out[9] != 0 { u.add_status("petrified"); }
            if d_alive {
                u.max_hp += dfd_feed as i64;
                u.current_hp = d_hp + dfd_feed as i64;
                u.current_exp = d_xp;
            }
        }
        if !a_alive {
            self.remove_unit(&att_id)?;
        }
        if !d_alive {
            self.remove_unit(&dfd_id)?;
        }
        Ok(Some(r))
    }
}
