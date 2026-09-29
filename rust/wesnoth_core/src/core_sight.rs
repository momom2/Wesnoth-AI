//! What each player side saw of the other sides' units
//! (docs/parity_memory_design_20260929.md "The watched turn").
//!
//! Per player side, two records:
//! - the sighting record: every unit of another side the side saw since
//!   its last end_turn, by id, with its type, hit points and maximum hit
//!   points and the last hex it was seen on. Cleared at the side's
//!   end_turn; a unit that leaves the board leaves it.
//! - the seen types: every (side, unit type) the side has seen in the
//!   game, the faction posterior's evidence. Never cleared.
//!
//! Both follow every command (the units the side sees afterwards) and,
//! during another side's move, each hex of the path the side sees: a
//! watching player's display draws the mover at each step where
//! `unit::is_visible_to_team` holds, its hex not fogged for the watcher
//! and the mover not hidden there by its hide ability
//! (docs/wesnoth_rules.md "A watching player sees a mover on every hex of
//! its path"). Scenery is never recorded: it is always visible.

use pyo3::prelude::*;

use crate::core::{GameCore, UnitRec};
use crate::core_attack::apply_illumination;
use crate::observe::neighbours;

/// Sides 1 and 2 keep records (`classes.PLAYER_SIDES`); the sides a
/// scenario declares beyond them are scenery or a neutral AI.
pub const RECORD_SIDES: usize = 2;

/// One unit as a side last saw it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SightRec {
    pub type_name: String,
    pub hp: i64,
    pub max_hp: i64,
    pub x: i64,
    pub y: i64,
}

/// A sighting as Python reads it: (unit id, type, hp, max hp, x, y).
pub type SightRow = (String, String, i64, i64, i64, i64);

/// The record index of a side that keeps records.
fn record_index(side: i64) -> Option<usize> {
    (side >= 1 && side as usize <= RECORD_SIDES).then(|| side as usize - 1)
}

fn record_side(side: i64) -> PyResult<usize> {
    record_index(side).ok_or_else(|| pyo3::exceptions::PyValueError::new_err(format!(
        "side {side} keeps no sighting record (the players' sides 1 and 2 do)")))
}

fn illuminates(u: &UnitRec) -> bool {
    u.has_ability("illuminates") && !u.has_status("petrified")
}

impl GameCore {
    /// Unit `i` seen by `side` on (x, y): its sighting and its type.
    fn record_sighting(&mut self, side: i64, i: usize, x: i64, y: i64) {
        let Some(k) = record_index(side) else { return };
        let u = &self.units[i];
        let rec = SightRec { type_name: u.name.clone(), hp: u.current_hp, max_hp: u.max_hp, x, y };
        self.seen_types[k].insert((u.side, u.name.clone()));
        self.sightings[k].insert(u.id.clone(), rec);
    }

    /// After a command: each player side records the units of the other
    /// sides it sees now, and a unit gone from the board leaves every
    /// record.
    pub(crate) fn note_sightings(&mut self) {
        for side in 1..=RECORD_SIDES as i64 {
            let visible = self.visible_to(side);
            for (i, &seen) in visible.iter().enumerate() {
                let u = &self.units[i];
                if !seen || u.side == side || u.hex < 0 || self.is_scenery(i) {
                    continue;
                }
                let (x, y) = (u.x, u.y);
                self.record_sighting(side, i, x, y);
            }
        }
        let index = &self.unit_index;
        for record in self.sightings.iter_mut() {
            record.retain(|id, _| index.contains_key(id));
        }
    }

    /// During the move of unit `i` along the map hexes `path` (its start
    /// hex, then every hex it entered): each other player side records it
    /// on the last of them it could see it on.
    pub(crate) fn note_path_sightings(&mut self, i: usize, path: &[usize]) {
        if self.is_scenery(i) {
            return;
        }
        let mover_side = self.units[i].side;
        for side in 1..=RECORD_SIDES as i64 {
            if side == mover_side {
                continue;
            }
            let seen = self.global.fog_on.then(|| self.seen_by(side));
            let last = path.iter().rev().copied().find(|&h| {
                seen.as_ref().map_or(true, |s| s[h] != 0) && !self.hidden_on_path(i, h)
            });
            if let Some(h) = last {
                let (x, y) = (self.map.hx[h], self.map.hy[h]);
                self.record_sighting(side, i, x, y);
            }
        }
    }

    /// The side's end_turn empties its sighting record.
    pub(crate) fn clear_sightings(&mut self, side: i64) {
        if let Some(k) = record_index(side) {
            self.sightings[k].clear();
        }
    }

    /// The entries of the side's record whose unit it does not see now
    /// (`visible`, one flag per unit), in id order.
    pub(crate) fn sightings_not_visible(&self, side: i64, visible: &[u8]) -> Vec<(&str, &SightRec)> {
        let Some(k) = record_index(side) else { return Vec::new() };
        self.sightings[k].iter()
            .filter(|(id, _)| self.unit_index.get(id.as_str()).is_some_and(|&i| visible[i] == 0))
            .map(|(id, rec)| (id.as_str(), rec))
            .collect()
    }

    /// Whether unit `i`, stepping on map hex `h`, is hidden there by its
    /// hide ability (`unit::invisible`): not uncovered, its cover active
    /// on the hex (the [hides] terrain globs; nightstalk at a dark
    /// illuminated time of day), and no unit of another side that is not
    /// scenery adjacent to the hex (`discovered_by_adjacency`'s rule).
    fn hidden_on_path(&self, i: usize, h: usize) -> bool {
        let u = &self.units[i];
        if self.is_uncovered(&u.id) {
            return false;
        }
        let m = &self.map;
        let dark = || {
            let bonus = self.lawful_bonus_at(h as i64, self.global.turn_number);
            apply_illumination(bonus, self.illuminated_on(i, h)) < 0
        };
        let cover = (u.has_ability("ambush") && m.hides_ambush[h] != 0)
            || (u.has_ability("concealment") && m.hides_concealment[h] != 0)
            || (u.has_ability("submerge") && m.hides_submerge[h] != 0)
            || (u.has_ability("nightstalk") && dark());
        if !cover {
            return false;
        }
        let adj = neighbours(m.hx[h], m.hy[h]);
        !(0..self.units.len()).any(|j| {
            let o = &self.units[j];
            j != i && o.side != u.side && !self.is_scenery(j) && adj.contains(&(o.x, o.y))
        })
    }

    /// `illuminated` for unit `i` standing on map hex `h`: it or a unit on
    /// or next to the hex illuminates and is not petrified.
    fn illuminated_on(&self, i: usize, h: usize) -> bool {
        if illuminates(&self.units[i]) {
            return true;
        }
        let at = (self.map.hx[h], self.map.hy[h]);
        let adj = neighbours(at.0, at.1);
        self.units.iter().enumerate().any(|(j, o)| {
            j != i && ((o.x, o.y) == at || adj.contains(&(o.x, o.y))) && illuminates(o)
        })
    }
}

#[pymethods]
impl GameCore {
    /// The side's sighting record as (unit id, type, hp, max hp, x, y)
    /// rows in id order.
    fn sightings_export(&self, side: i64) -> PyResult<Vec<SightRow>> {
        let k = record_side(side)?;
        Ok(self.sightings[k].iter()
            .map(|(id, r)| (id.clone(), r.type_name.clone(), r.hp, r.max_hp, r.x, r.y))
            .collect())
    }

    /// Replace the side's sighting record (a core built from a view).
    /// Each type is registered, so the encoding's vocabulary covers it.
    fn set_sightings(&mut self, side: i64, rows: Vec<SightRow>) -> PyResult<()> {
        let k = record_side(side)?;
        for row in &rows {
            self.type_idx(&row.1);
        }
        self.sightings[k] = rows.into_iter()
            .map(|(id, type_name, hp, max_hp, x, y)| (id, SightRec { type_name, hp, max_hp, x, y }))
            .collect();
        Ok(())
    }

    /// The unit types of side `other` that `side` has seen in the game,
    /// sorted.
    fn seen_types(&self, side: i64, other: i64) -> PyResult<Vec<String>> {
        let k = record_side(side)?;
        Ok(self.seen_types[k].iter().filter(|(s, _)| *s == other).map(|(_, t)| t.clone()).collect())
    }

    /// Every (side, unit type) `side` has seen in the game, sorted.
    fn seen_types_export(&self, side: i64) -> PyResult<Vec<(i64, String)>> {
        let k = record_side(side)?;
        Ok(self.seen_types[k].iter().cloned().collect())
    }

    /// Replace what `side` has seen (a core built from a view).
    fn set_seen_types(&mut self, side: i64, rows: Vec<(i64, String)>) -> PyResult<()> {
        let k = record_side(side)?;
        self.seen_types[k] = rows.into_iter().collect();
        Ok(())
    }
}
