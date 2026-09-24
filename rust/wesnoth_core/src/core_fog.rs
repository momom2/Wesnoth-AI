//! Fog of war over the core (docs/wesnoth_rules.md "Vision and fog:
//! what a side sees"): a unit's vision area, each side's cleared hexes
//! as the engine keeps them, and the hooks the commands call.
//! `wesnoth_ai.visibility` is the oracle (tests/test_game_core.py
//! compares the two after every command).
//!
//! A side's cleared hexes are tracked once the side has been refogged
//! or has cleared something; until then the side sees its units'
//! vision from where they stand, which is what the engine clears for
//! every side at the start of the game.

use pyo3::prelude::*;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::core::GameCore;

impl GameCore {
    /// Marks in `out` the hexes unit `i` sees from map hex `at`
    /// (`visibility.unit_vision`): every hex it could reach spending
    /// its vision points, its maximum movement, at its movement costs
    /// (doubled when slowed), other units ignored, plus every
    /// neighbour of those.
    pub fn mark_vision(&self, i: usize, at: usize, out: &mut [u8]) {
        let u = &self.units[i];
        let class = if u.has_status("slowed") { u.class_slowed_id } else { u.class_id };
        let classes = self.classes.read().unwrap();
        if class < 0 || class as usize >= classes.len() {
            return;
        }
        let mcost = &classes[class as usize].mcost;
        let budget = u.max_moves.max(0);
        let nbrs = &self.map.nbrs;
        let mut spent = vec![i64::MAX; self.map.h];
        spent[at] = 0;
        let mut frontier = BinaryHeap::new();
        frontier.push(Reverse((0i64, at)));
        while let Some(Reverse((cost, cur))) = frontier.pop() {
            if cost > spent[cur] {
                continue;
            }
            out[cur] = 1;
            for &nb in &nbrs[cur * 6..cur * 6 + 6] {
                if nb < 0 {
                    continue;
                }
                let nb = nb as usize;
                out[nb] = 1;
                let next = cost + mcost[nb];
                if next <= budget && next < spent[nb] {
                    spent[nb] = next;
                    frontier.push(Reverse((next, nb)));
                }
            }
        }
    }

    /// The union of the side's units' vision from where they stand
    /// (`visibility.side_vision`).
    pub fn side_vision(&self, side: i64) -> Vec<u8> {
        let mut out = vec![0u8; self.map.h];
        for i in 0..self.units.len() {
            let u = &self.units[i];
            if u.side == side && u.hex >= 0 {
                self.mark_vision(i, u.hex as usize, &mut out);
            }
        }
        out
    }

    fn cleared_of(&self, side: i64) -> Option<&Vec<u8>> {
        if side < 1 {
            return None;
        }
        self.fog_cleared.get(side as usize - 1).filter(|v| !v.is_empty())
    }

    fn set_cleared(&mut self, side: i64, cleared: Vec<u8>) {
        let k = side as usize - 1;
        if self.fog_cleared.len() <= k {
            self.fog_cleared.resize(k + 1, Vec::new());
        }
        self.fog_cleared[k] = cleared;
    }

    /// What `side` sees (`visibility.visible_hexes_for`): its cleared
    /// hexes when tracked, else its units' vision.
    pub fn seen_by(&self, side: i64) -> Vec<u8> {
        match self.cleared_of(side) {
            Some(v) => v.clone(),
            None => self.side_vision(side),
        }
    }

    /// `visibility.track_side`: start tracking the side's cleared hexes
    /// from its units' vision, before a command changes them.
    pub fn track_side(&mut self, side: i64) {
        if !self.global.fog_on || side < 1 || self.cleared_of(side).is_some() {
            return;
        }
        let v = self.side_vision(side);
        self.set_cleared(side, v);
    }

    /// `visibility.refog`: the side's fog recalculated from where its
    /// units stand (turn start, turn end, the defender after a fight
    /// that killed, slowed or petrified it).
    pub fn refog(&mut self, side: i64) {
        if !self.global.fog_on || side < 1 {
            return;
        }
        let v = self.side_vision(side);
        self.set_cleared(side, v);
    }

    /// `visibility.clear_fog`: unit `i`'s vision from each of `hexes`
    /// added to its side's cleared hexes (a move's entered hexes, a
    /// recruit's or an advanced unit's hex).
    pub fn clear_fog_from(&mut self, i: usize, hexes: &[usize]) {
        let side = self.units[i].side;
        if !self.global.fog_on || side < 1 || hexes.is_empty() {
            return;
        }
        let mut v = self.seen_by(side);
        for &hex in hexes {
            self.mark_vision(i, hex, &mut v);
        }
        self.set_cleared(side, v);
    }
}

#[pymethods]
impl GameCore {
    /// `visibility.refog`: the wrapper calls it for the defender's side
    /// after a fight that killed, slowed or petrified the defender.
    fn refog_side(&mut self, side: i64) {
        self.refog(side);
    }

    /// The unit's vision from its hex added to its side's cleared hexes:
    /// the wrapper calls it after placing a recruit or an advanced unit.
    fn clear_unit_fog(&mut self, id: &str) -> PyResult<()> {
        let i = match self.units.iter().position(|u| u.id == id) {
            Some(i) => i,
            None => return Err(pyo3::exceptions::PyKeyError::new_err(id.to_string())),
        };
        let hex = self.units[i].hex;
        if hex >= 0 {
            self.clear_fog_from(i, &[hex as usize]);
        }
        Ok(())
    }

    /// Every tracked side's cleared hexes as (side, [(x, y)]).
    fn fog_cleared_export(&self) -> Vec<(i64, Vec<(i64, i64)>)> {
        let mut out = Vec::new();
        for (k, v) in self.fog_cleared.iter().enumerate() {
            if v.is_empty() {
                continue;
            }
            let hexes: Vec<(i64, i64)> =
                (0..self.map.h).filter(|&j| v[j] != 0).map(|j| (self.map.hx[j], self.map.hy[j])).collect();
            out.push((k as i64 + 1, hexes));
        }
        out
    }

    /// Replace the tracked sides' cleared hexes; hexes off the map are
    /// dropped, a side left out is untracked.
    fn set_fog_cleared(&mut self, sides: Vec<(i64, Vec<(i64, i64)>)>) -> PyResult<()> {
        self.fog_cleared.clear();
        for (side, hexes) in sides {
            if side < 1 {
                return Err(pyo3::exceptions::PyValueError::new_err(format!("side {side}")));
            }
            let mut v = vec![0u8; self.map.h];
            for (x, y) in hexes {
                if let Some(&j) = self.map.pos_index.get(&(x, y)) {
                    v[j] = 1;
                }
            }
            self.set_cleared(side, v);
        }
        Ok(())
    }

    /// What `side` sees, map space (`visibility.visible_hexes_for`).
    fn seen_export<'py>(&self, py: Python<'py>, side: i64) -> Bound<'py, numpy::PyArray1<u8>> {
        use numpy::IntoPyArray;
        self.seen_by(side).into_pyarray(py)
    }
}
