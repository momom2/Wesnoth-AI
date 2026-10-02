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
//! Both follow every command: the units the side sees afterwards, as a
//! player with move animations off sees the enemy's turn (user ruling
//! 2026-10-01), so a unit that crosses the side's view during a move is
//! not seen, and one that walks out of view is remembered where it stood.
//! A fight the side's own unit defends is watched too: the side records
//! what it sees before the fight refogs it. Scenery is never recorded: it
//! is always visible.

use pyo3::prelude::*;

use crate::core::GameCore;

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

impl GameCore {
    /// Unit `i` seen by `side` on (x, y): its sighting, and its type unless
    /// the scenario placed it.
    fn record_sighting(&mut self, side: i64, i: usize, x: i64, y: i64) {
        let Some(k) = record_index(side) else { return };
        let u = &self.units[i];
        let rec = SightRec { type_name: u.name.clone(), hp: u.current_hp, max_hp: u.max_hp, x, y };
        if !self.scenario_unit_ids.contains(&u.id) {
            self.seen_types[k].insert((u.side, u.name.clone()));
        }
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

    /// The units the scenario placed, whose types are not seen types.
    fn set_scenario_unit_ids(&mut self, ids: Vec<String>) {
        self.scenario_unit_ids = ids.into_iter().collect();
    }

    fn scenario_unit_ids(&self) -> Vec<String> {
        self.scenario_unit_ids.iter().cloned().collect()
    }
}
