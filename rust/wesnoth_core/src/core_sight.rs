//! What each player side saw of the other sides' units
//! (docs/parity_memory_design_20260929.md "The watched turn").
//!
//! Per player side, two records:
//! - the sighting record: every unit of another side the side saw since
//!   its last end_turn, by id, with its type, hit points and maximum hit
//!   points and the last hex it was seen on. Cleared at the side's
//!   end_turn. A unit that leaves the board leaves it when the side saw
//!   its hex at that moment; otherwise it stays until the end_turn, as a
//!   player who did not see the unit go still believes it where it was
//!   last seen (a neutral side can kill it in the side's fog).
//! - the seen types: every (side, unit type) the side has seen in the
//!   game, the faction posterior's evidence. Never cleared.
//!
//! Both follow every command (the units the side sees afterwards) and,
//! during another side's move, the path: a watching player's display
//! animates each step from a hex where the mover is visible to it (not
//! fogged, not hidden by its hide ability) toward the next hex, so the
//! player last sees the mover entering the hex after the last one it was
//! visible on (docs/wesnoth_rules.md "A watching player sees a mover on
//! every hex of its path"). A fight the side's own unit defends is watched
//! too: the side records what it sees before the fight refogs it. Scenery
//! is never recorded: it is always visible.

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
            self.note_sightings_of(side);
        }
        let index = &self.unit_index;
        for (record, gone) in self.sightings.iter_mut().zip(self.sightings_gone.iter()) {
            record.retain(|id, _| index.contains_key(id) || gone.contains(id));
        }
    }

    /// What `side` sees now, into its record (`note_sightings`, one side).
    pub(crate) fn note_sightings_of(&mut self, side: i64) {
        if record_index(side).is_none() {
            return;
        }
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

    /// Unit `i` is about to leave the board. Each side whose record holds
    /// it and that does not see its hex keeps the entry, marked gone.
    pub(crate) fn note_departure(&mut self, i: usize) {
        let (id, hex) = (self.units[i].id.clone(), self.units[i].hex);
        for side in 1..=RECORD_SIDES as i64 {
            let k = side as usize - 1;
            if !self.sightings[k].contains_key(&id) {
                continue;
            }
            let unseen = self.global.fog_on && (hex < 0 || self.seen_by(side)[hex as usize] == 0);
            if unseen {
                self.sightings_gone[k].insert(id.clone());
            }
        }
    }

    /// During the move of unit `i` along the map hexes `path` (its start
    /// hex, then every hex it entered): each other player side records it
    /// on the hex after the last one it could see it on (the step out of
    /// view is animated from that hex toward the next, udisplay.cpp:141-148,
    /// drawn while the mover is visible there, drawer.cpp:200), or on that
    /// hex when it is the route's last.
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
            let last = (0..path.len()).rev().find(|&k| {
                seen.as_ref().map_or(true, |s| s[path[k]] != 0) && !self.hidden_on_path(i, path[k])
            });
            if let Some(k) = last {
                let h = path[(k + 1).min(path.len() - 1)];
                let (x, y) = (self.map.hx[h], self.map.hy[h]);
                self.record_sighting(side, i, x, y);
            }
        }
    }

    /// The side's end_turn empties its sighting record.
    pub(crate) fn clear_sightings(&mut self, side: i64) {
        if let Some(k) = record_index(side) {
            self.sightings[k].clear();
            self.sightings_gone[k].clear();
        }
    }

    /// The entries of the side's record whose unit it does not see now
    /// (`visible`, one flag per unit), the gone ones included, in id order.
    pub(crate) fn sightings_not_visible(&self, side: i64, visible: &[u8]) -> Vec<(&str, &SightRec)> {
        let Some(k) = record_index(side) else { return Vec::new() };
        let gone = &self.sightings_gone[k];
        self.sightings[k].iter()
            .filter(|(id, _)| gone.contains(id.as_str())
                    || self.unit_index.get(id.as_str()).is_some_and(|&i| visible[i] == 0))
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

    /// The ids of the side's record whose unit left the board where the
    /// side did not see it go, sorted.
    fn sightings_gone_export(&self, side: i64) -> PyResult<Vec<String>> {
        let k = record_side(side)?;
        Ok(self.sightings_gone[k].iter().cloned().collect())
    }

    /// Replace the side's gone ids (a core built from a view); each must
    /// name an entry of its record.
    fn set_sightings_gone(&mut self, side: i64, ids: Vec<String>) -> PyResult<()> {
        let k = record_side(side)?;
        if let Some(id) = ids.iter().find(|id| !self.sightings[k].contains_key(id.as_str())) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "side {side}: gone id {id} has no entry in its sighting record")));
        }
        self.sightings_gone[k] = ids.into_iter().collect();
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
