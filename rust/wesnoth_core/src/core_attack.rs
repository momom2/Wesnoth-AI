//! Phase 4: the attack command on the Rust-owned state, a transcription
//! of the attack branch of `tools/replay_dataset._apply_command` and of
//! `build_attack_context` / `_to_combat_unit` (the snapshots, the
//! terrain defense, the lawful bonus with illumination, leadership,
//! backstab) over the combat kernel (combat.rs). The command applies the
//! outcome to the two units (hit points, experience, statuses, deaths),
//! then in the applier's order the attacker's feeding and advancement or
//! the corpse its death raises, the defender's feeding, its side's refog
//! and its advancement, or the corpse and the refog of its death
//! (core_units.rs); it reports the fight for the engagement telemetry.
//! The fight's setup (`fight_inputs`) and the combatants' write-back
//! (`write_back`) serve the outcome enumeration too (outcomes.rs).

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::combat::{resolve_fight_with, FightInputs, Mt19937, ScriptedRng, StrikeRng, UNIT_FLAGS, UNIT_INTS};
use crate::core::{BaseAttack, GameCore, UnitRec};
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
/// by index (damage type, range, specials, accuracy, parry) with the
/// unit's numbers and its own specials unioned in.
pub(crate) struct WeaponView {
    pub type_idx: Option<usize>,     // into DAMAGE_TYPES; None = a type no resistance names
    pub ranged: bool,
    pub damage: i64,
    pub number: i64,
    accuracy: i64,
    parry: i64,
    pub specials: Vec<String>,
}

pub(crate) fn dt_index(name: &str) -> Option<usize> {
    DAMAGE_TYPES.iter().position(|d| *d == name)
}

/// An attacker's weapon index as the applier reads it: past the end is
/// weapon 0, a negative one counts from the end (Python indexing).
pub(crate) fn attacker_weapon(n: usize, a_weapon: i64) -> PyResult<usize> {
    if a_weapon >= n as i64 {
        return Ok(0);
    }
    if a_weapon >= 0 {
        return Ok(a_weapon as usize);
    }
    let k = n as i64 + a_weapon;
    if k < 0 {
        return Err(pyo3::exceptions::PyIndexError::new_err("attacker weapon"));
    }
    Ok(k as usize)
}

/// A combatant's record after the fight, as the attack command writes it
/// back before any advancement: "resting" gone (attack.cpp:1279-80), the
/// statuses the fight gave added, and for a survivor its hit points and
/// experience, with a fed kill's +1 hit point, +1 maximum and one more on
/// the feeding count advancement adds back.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_back(u: &mut UnitRec, hp: i64, xp: i64, slowed: bool, poisoned: bool, petrified: bool,
                         fed: bool) {
    u.drop_status("resting");
    if slowed { u.add_status("slowed"); }
    if poisoned { u.add_status("poisoned"); }
    if petrified { u.add_status("petrified"); }
    if hp > 0 {
        u.max_hp += fed as i64;
        u.current_hp = hp + fed as i64;
        u.current_exp = xp;
        if fed {
            u.feeding_count = Some(u.feeding_count.unwrap_or(0) + 1);
        }
    }
}

/// `apply_unit_illumination`: bounded_add(base, 25, max 25, min 0), positive branch.
pub(crate) fn apply_illumination(base: i64, illuminated: bool) -> i64 {
    if illuminated { (base + ILLUMINATION).min(base.max(ILLUMINATION)) } else { base }
}

impl GameCore {
    pub(crate) fn weapons_of(&self, i: usize) -> Vec<WeaponView> {
        let u = &self.units[i];
        let types = self.types.read().unwrap();
        let base: &[BaseAttack] = if u.type_idx >= 0 { &types[u.type_idx as usize].attacks } else { &[] };
        let mut out = Vec::with_capacity(u.attacks.len());
        for (k, a) in u.attacks.iter().enumerate() {
            let (type_idx, ranged, mut specials, accuracy, parry) = if k < base.len() {
                let b = &base[k];
                (dt_index(&b.type_name), b.ranged, b.specials.clone(), b.accuracy, b.parry)
            } else {
                let t = if (0..6).contains(&a.type_id) { a.type_id as usize } else { 0 };
                (Some(t), a.ranged, Vec::new(), 0, 0)
            };
            for s in &a.specials {
                if !specials.contains(s) {
                    specials.push(s.clone());
                }
            }
            out.push(WeaponView { type_idx, ranged, damage: a.damage, number: a.strikes, accuracy, parry, specials });
        }
        if out.is_empty() {
            // `Weapon("none", 1, 1, "melee", "blade", [])`
            out.push(WeaponView {
                type_idx: Some(0), ranged: false, damage: 1, number: 1, accuracy: 0, parry: 0, specials: Vec::new(),
            });
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
                special => weapon.is_some_and(|w| w.specials.iter().any(|s| s == special)),
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
    pub(crate) fn leadership(&self, i: usize) -> i64 {
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

    /// `build_attack_context`: the fight's inputs for unit `a` attacking
    /// `d` with weapon `a_weapon` (`attacker_weapon`) and `d` answering
    /// with `d_weapon` (-1: no answer; past its weapons: weapon 0; never
    /// when petrified, whose attacks the engine strips).
    pub(crate) fn fight_inputs(&self, a: usize, d: usize, a_weapon: i64, d_weapon: i64) -> PyResult<FightInputs> {
        let aw = self.weapons_of(a);
        let dw = self.weapons_of(d);
        let a_idx = attacker_weapon(aw.len(), a_weapon)?;
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
        Ok(FightInputs {
            a_ints,
            a_flags,
            d_ints,
            d_flags,
            d_has_weapon: d_has,
            a_lawful_bonus: apply_illumination(self.lawful_bonus_at(self.units[a].hex, turn), self.illuminated(a)),
            d_lawful_bonus: apply_illumination(self.lawful_bonus_at(self.units[d].hex, turn), self.illuminated(d)),
            a_leadership_bonus: self.leadership(a),
            d_leadership_bonus: self.leadership(d),
            a_backstab_active: self.backstab_active(a, d),
            d_backstab_active: self.backstab_active(d, a),
        })
    }

    /// `_is_unplagueable`: the status, or a trait whose macro adds it.
    pub(crate) fn unplagueable(&self, i: usize) -> bool {
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
    /// ends the command there. Returns None then, else the fight's facts
    /// for the telemetry: ids, names, sides, costs, positions before the
    /// fight, who lives, who fed, who reached its experience cap, whether
    /// a plague corpse rose on either hex, the damage each took. What each
    /// side sees afterwards is recorded (core_sight.rs).
    #[pyo3(signature = (ax, ay, dx, dy, a_weapon, d_weapon, seed, has_seed, choices))]
    #[allow(clippy::too_many_arguments)]
    fn apply_attack<'py>(&mut self, py: Python<'py>, ax: i64, ay: i64, dx: i64, dy: i64, a_weapon: i64,
                         d_weapon: i64, seed: u32, has_seed: bool, choices: Vec<i64>)
        -> PyResult<Option<Bound<'py, PyDict>>> {
        self.advance_choices.extend(choices);
        let (a, d) = match (self.unit_at(ax, ay), self.unit_at(dx, dy)) {
            (Some(a), Some(d)) => (a, d),
            _ => {
                crate::effects::warn_once(format!(
                    "{}: an attack from ({ax}, {ay}) on ({dx}, {dy}) misses a unit; the command is skipped",
                    self.game_id));
                return Ok(None);
            }
        };
        if !has_seed {
            // Aborted before its first draw; the engine's handler has already
            // cleared the stack (synced_commands.cpp:228).
            self.clear_undo_stack();
            return Ok(None);
        }
        let out = self.attack(py, a, d, a_weapon, d_weapon, &mut Mt19937::new(seed, 0))?;
        self.note_sightings();
        Ok(Some(out))
    }

    /// The attack command with its strike draws scripted
    /// (`swap_detector._apply_attack_scripted`): `prefix` forces the
    /// first strikes to hit (true) or miss (false), every later one
    /// hits. Non-empty `choices` become the advancement queue. Returns
    /// the number of draws the fight took, or None when a hex holds no
    /// unit.
    #[allow(clippy::too_many_arguments)]
    fn apply_attack_scripted(&mut self, py: Python<'_>, ax: i64, ay: i64, dx: i64, dy: i64, a_weapon: i64,
                             d_weapon: i64, prefix: Vec<bool>, choices: Vec<i64>) -> PyResult<Option<u64>> {
        if !choices.is_empty() {
            self.advance_choices = choices;
        }
        let (a, d) = match (self.unit_at(ax, ay), self.unit_at(dx, dy)) {
            (Some(a), Some(d)) => (a, d),
            _ => return Ok(None),
        };
        let mut rng = ScriptedRng::new(prefix);
        self.attack(py, a, d, a_weapon, d_weapon, &mut rng)?;
        self.note_sightings();
        Ok(Some(rng.calls()))
    }
}

impl GameCore {
    /// The attack of unit `a` on unit `d` with the strike draws of `rng`,
    /// and the fight's facts for the telemetry.
    #[allow(clippy::too_many_arguments)]
    fn attack<'py>(&mut self, py: Python<'py>, a: usize, d: usize, a_weapon: i64, d_weapon: i64,
                   rng: &mut dyn StrikeRng) -> PyResult<Bound<'py, PyDict>> {
        let att_id = self.units[a].id.clone();
        let dfd_id = self.units[d].id.clone();
        let (att_side, dfd_side) = (self.units[a].side, self.units[d].side);
        self.track_side(att_side);
        self.track_side(dfd_side);
        // The fight's first random draw makes the turn's actions final,
        // which commits a delaying side's pending vision
        // (synced_context.cpp:277-285, core_shroud.rs).
        self.clear_undo_stack();
        let dfd_was_slowed = self.units[d].has_status("slowed");
        let dfd_was_petrified = self.units[d].has_status("petrified");
        self.uncover(&att_id);                  // attack.cpp:1378
        let inputs = self.fight_inputs(a, d, a_weapon, d_weapon)?;
        let (out, record) = resolve_fight_with(&inputs, rng);
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
            r.set_item("plague_reverse", plague_reverse)?;
            r.set_item("dmg_to_defender", (dfd.current_hp - d_hp).max(0))?;
            r.set_item("dmg_to_attacker", (att.current_hp - a_hp).max(0))?;
        }
        write_back(&mut self.units[a], a_hp, a_xp, out[4] != 0, out[5] != 0, out[6] != 0, att_feed);
        if a_alive {
            let u = &mut self.units[a];
            u.has_attacked = true;
            u.current_moves = 0;                // attack.cpp:1372, movement_used = everything
        }
        write_back(&mut self.units[d], d_hp, d_xp, out[7] != 0, out[8] != 0, out[9] != 0, dfd_feed);
        let att_advances = a_alive && a_xp >= self.units[a].max_exp;
        let dfd_advances = d_alive && d_xp >= self.units[d].max_exp;
        let dfd_refog = !d_alive || (out[7] != 0 && !dfd_was_slowed) || (out[9] != 0 && !dfd_was_petrified);
        let (att_name, dfd_name) = (self.units[a].name.clone(), self.units[d].name.clone());
        let (att_pos, dfd_pos) = ((self.units[a].x, self.units[a].y), (self.units[d].x, self.units[d].y));
        if !a_alive {
            self.remove_unit(&att_id)?;
        }
        if !d_alive {
            self.remove_unit(&dfd_id)?;
        }
        if a_alive {
            if att_advances {
                let i = self.unit_pos(&att_id).expect("the attacker lives");
                self.advance_unit(i);
            }
        } else if plague_reverse {
            self.spawn_corpse(att_pos.0, att_pos.1, dfd_side, &att_name)?;
        }
        if d_alive {
            // attack.cpp:1150-1185 and 1456-1458: the defender's side
            // refogs when the defender was slowed or petrified, before
            // its advancement.
            if dfd_refog {
                self.refog(dfd_side);
            }
            if dfd_advances {
                let i = self.unit_pos(&dfd_id).expect("the defender lives");
                self.advance_unit(i);
            }
        } else {
            if plague_forward {
                self.spawn_corpse(dfd_pos.0, dfd_pos.1, att_side, &dfd_name)?;
            }
            self.refog(dfd_side);
        }
        Ok(r)
    }
}
