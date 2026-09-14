//! Phase 4: the move command on the Rust-owned state, a transcription
//! of `tools/pathfind_sim.walk_move_path` and of the move branch of
//! `tools/replay_dataset._apply_command` (the landing, the village
//! capture), with the visibility rules they consume from
//! `wesnoth_ai/visibility.py`: the hide cover, discovery by adjacency,
//! the sight disc and the units a side can see. tools/diff_core.py
//! replays the corpus through both and is the oracle.

use pyo3::prelude::*;

use crate::core::GameCore;
use crate::core_attack::apply_illumination;
use crate::observe::{hex_distance, neighbours};

/// `pathfind_sim.UNREACHABLE` (movetype.hpp).
pub const UNREACHABLE: i64 = 99;
const HIDE_ABILITIES: [&str; 4] = ["ambush", "concealment", "submerge", "nightstalk"];

/// `pathfind_sim.MoveOutcome`: where the walk stopped, the movement
/// left, the hiders it revealed and why it stopped.
pub struct MoveOutcome {
    pub final_idx: usize,
    pub mp_left: i64,
    pub uncovered: Vec<String>,
    pub reason: &'static str,
}

impl GameCore {
    /// `visibility.is_scenery_unit`: petrified, or an attackless unit
    /// of a side other than 1 and 2.
    pub fn is_scenery(&self, i: usize) -> bool {
        let u = &self.units[i];
        u.has_status("petrified") || (u.side != 1 && u.side != 2 && u.attacks.is_empty())
    }

    /// `visibility._hide_cover_active`: a hide ability whose cover the
    /// unit's hex provides (the [hides] terrain globs baked into the
    /// map flags; for nightstalk the ILLUMINATED time of day below
    /// zero, an [illuminates] unit on or next to the hex counted).
    pub fn hide_cover_active(&self, i: usize) -> bool {
        let u = &self.units[i];
        if !u.abilities.iter().any(|a| HIDE_ABILITIES.contains(&a.as_str())) {
            return false;
        }
        if u.hex >= 0 {
            let h = u.hex as usize;
            if u.has_ability("ambush") && self.map.hides_ambush[h] != 0 {
                return true;
            }
            if u.has_ability("concealment") && self.map.hides_concealment[h] != 0 {
                return true;
            }
            if u.has_ability("submerge") && self.map.hides_submerge[h] != 0 {
                return true;
            }
        }
        u.has_ability("nightstalk")
            && apply_illumination(self.lawful_bonus_at(u.hex, self.global.turn_number), self.illuminated(i)) < 0
    }

    /// `visibility._discovered_by_adjacency`: an enemy of the unit,
    /// not scenery, not petrified, on an adjacent hex.
    pub fn discovered_by_adjacency(&self, i: usize) -> bool {
        let u = &self.units[i];
        let adj = neighbours(u.x, u.y);
        (0..self.units.len()).any(|j| {
            let o = &self.units[j];
            o.side != u.side && !self.is_scenery(j) && !o.has_status("petrified")
                && adj.contains(&(o.x, o.y))
        })
    }

    pub fn is_uncovered(&self, id: &str) -> bool {
        self.uncovered.iter().any(|u| u == id)
    }

    /// The persistent reveal (STATE_UNCOVERED, move.cpp:870).
    pub fn uncover(&mut self, id: &str) {
        if !self.is_uncovered(id) {
            self.uncovered.push(id.to_string());
            self.uncovered.sort();
        }
    }

    /// `visibility.visible_hexes_for`: the union of the side's units'
    /// sight discs, radius max(max_moves, 1).
    pub fn vision_disc(&self, side: i64) -> Vec<u8> {
        let h = self.map.h;
        let mut disc = vec![0u8; h];
        for u in &self.units {
            if u.side != side {
                continue;
            }
            let r = u.max_moves.max(1);
            for j in 0..h {
                if disc[j] == 0 && hex_distance(u.x, u.y, self.map.hx[j], self.map.hy[j]) <= r {
                    disc[j] = 1;
                }
            }
        }
        disc
    }

    /// `visibility.units_visible_to(side)` as one flag per unit: own
    /// units and scenery always; a covered hider only when uncovered
    /// or discovered by adjacency; the rest inside the sight disc
    /// when fog is on.
    pub fn visible_to(&self, side: i64) -> Vec<bool> {
        let n = self.units.len();
        let mut out = vec![false; n];
        let mut disc: Option<Vec<u8>> = None;
        for i in 0..n {
            let u = &self.units[i];
            if u.side == side || self.is_scenery(i) {
                out[i] = true;
                continue;
            }
            if self.hide_cover_active(i) && !self.is_uncovered(&u.id) && !self.discovered_by_adjacency(i) {
                continue;
            }
            if !self.global.fog_on {
                out[i] = true;
                continue;
            }
            let d = disc.get_or_insert_with(|| self.vision_disc(side));
            if u.hex >= 0 && d[u.hex as usize] != 0 {
                out[i] = true;
            }
        }
        out
    }

    /// `walk_move_path`'s hidden hider: an enemy of `side` with its
    /// cover active, not uncovered, not petrified, with no enemy of
    /// its own adjacent (would_be_discovered, pre-move snapshot).
    fn is_hidden_hider(&self, i: usize, side: i64) -> bool {
        let u = &self.units[i];
        u.side != side
            && !self.is_uncovered(&u.id)
            && !u.has_status("petrified")
            && self.hide_cover_active(i)
            && !self.discovered_by_adjacency(i)
    }

    /// The first unit standing on (x, y) in unit order (`_find_unit_at`).
    pub fn unit_at(&self, x: i64, y: i64) -> Option<usize> {
        self.units.iter().position(|u| u.x == x && u.y == y)
    }

    /// `pathfind_sim.walk_move_path` for unit `i` along the ordered
    /// path (xs[0], ys[0]) = its hex: blocked by an enemy on a path
    /// hex (revealed, stop before it, movement kept), ambush by a
    /// hidden hider adjacent to an entered hex (stop there, movement
    /// zeroed), zone of control of the units the side can see (stop,
    /// movement zeroed unless a skirmisher), the backtrack off
    /// occupied hexes, the capturable-village landing. Movement is
    /// charged per entered hex from the unit's movement class; with
    /// `enforce_budget` off (reconstruction) an overrun clamps to 0.
    pub fn walk_move_path(&self, i: usize, xs: &[i64], ys: &[i64], enforce_budget: bool) -> MoveOutcome {
        let u = &self.units[i];
        let side = u.side;
        let budget = u.current_moves;
        let skirmisher = u.has_ability("skirmisher");
        let n = self.units.len();
        let h = self.map.h;
        let occ = self.occupancy();
        let other_at = |hex: Option<usize>| -> Option<usize> {
            match hex {
                Some(hi) if occ[hi] >= 0 && occ[hi] as usize != i => Some(occ[hi] as usize),
                _ => None,
            }
        };
        let hidden: Vec<usize> = (0..n).filter(|&j| self.is_hidden_hider(j, side)).collect();
        let mut hider_adjacent = vec![0u8; h];
        for &j in &hidden {
            for (x, y) in neighbours(self.units[j].x, self.units[j].y) {
                if let Some(&hi) = self.map.pos_index.get(&(x, y)) {
                    hider_adjacent[hi] = 1;
                }
            }
        }
        let mut zoc = vec![0u8; h];
        if !skirmisher {
            let visible = self.visible_to(side);
            for j in 0..n {
                let o = &self.units[j];
                if !visible[j] || o.side == side || o.has_status("petrified") || self.unit_level(j) < 1 {
                    continue;
                }
                for (x, y) in neighbours(o.x, o.y) {
                    if let Some(&hi) = self.map.pos_index.get(&(x, y)) {
                        zoc[hi] = 1;
                    }
                }
            }
        }
        let class = if u.has_status("slowed") { u.class_slowed_id } else { u.class_id };
        let classes = self.classes.read().unwrap();
        let mcost = &classes[class as usize].mcost;
        let m = xs.len();
        let mut uncovered: Vec<String> = Vec::new();
        let mut cum: Vec<i64> = vec![0];
        let mut final_idx = 0usize;
        let mut reason = "end";
        for j in 1..m {
            let hex = self.map.pos_index.get(&(xs[j], ys[j])).copied();
            if let Some(b) = other_at(hex) {
                if self.units[b].side != side {
                    uncovered.push(self.units[b].id.clone());
                    reason = "blocked";
                    break;
                }
            }
            let step_cost = match hex {
                Some(hi) => mcost[hi],
                None => UNREACHABLE,
            };
            let last = *cum.last().unwrap();
            if enforce_budget && (step_cost >= UNREACHABLE || last + step_cost > budget) {
                break;
            }
            cum.push(last + step_cost);
            final_idx = j;
            if hex.map_or(false, |hi| hider_adjacent[hi] != 0) {
                for &k in &hidden {
                    if neighbours(self.units[k].x, self.units[k].y).contains(&(xs[j], ys[j])) {
                        uncovered.push(self.units[k].id.clone());
                    }
                }
                reason = "ambush";
                break;
            }
            if !skirmisher && hex.map_or(false, |hi| zoc[hi] != 0) && j < m - 1 {
                reason = "zoc";
                break;
            }
        }
        while final_idx > 0 && other_at(self.map.pos_index.get(&(xs[final_idx], ys[final_idx])).copied()).is_some() {
            final_idx -= 1;
        }
        if final_idx == 0 {
            return MoveOutcome { final_idx: 0, mp_left: budget, uncovered, reason };
        }
        let fh = self.map.pos_index.get(&(xs[final_idx], ys[final_idx])).copied();
        let mut mp_left = (budget - cum[final_idx]).max(0);
        if reason == "ambush" {
            mp_left = 0;
        } else if let Some(fh) = fh {
            if !skirmisher && zoc[fh] != 0 {
                mp_left = 0;
            } else if self.map.village_terrain[fh] != 0 && self.village_owner[fh] != side {
                mp_left = 0;
            }
        }
        MoveOutcome { final_idx, mp_left, uncovered, reason }
    }

    /// `_capture_village`: ownership and the sides' village counts; a
    /// revisit by the owner changes nothing.
    pub fn capture_village(&mut self, hex: usize, side: i64) {
        let prev = self.village_owner[hex];
        if prev == side {
            return;
        }
        self.village_owner[hex] = side;
        let n = self.sides.len() as i64;
        if prev >= 1 && prev <= n {
            let s = &mut self.sides[prev as usize - 1];
            s.nb_villages = (s.nb_villages - 1).max(0);
        }
        if side >= 1 && side <= n {
            self.sides[side as usize - 1].nb_villages += 1;
        }
    }
}

#[pymethods]
impl GameCore {
    /// `_apply_command(["move", xs, ys, from_side])`: the unit on the
    /// source hex (of `from_side` when given) walks the path; the walk
    /// record, the reveals, the landing (position, movement, resting
    /// dropped) and the village capture follow the Python branch.
    #[pyo3(signature = (xs, ys, from_side=0, enforce_budget=false))]
    fn apply_move(&mut self, xs: Vec<i64>, ys: Vec<i64>, from_side: i64, enforce_budget: bool) -> PyResult<()> {
        if xs.is_empty() || xs.len() != ys.len() {
            return Err(pyo3::exceptions::PyValueError::new_err("empty or uneven path"));
        }
        let i = match self.units.iter().position(|u| {
            u.x == xs[0] && u.y == ys[0] && (from_side == 0 || u.side == from_side)
        }) {
            Some(i) => i,
            None => return Ok(()),
        };
        let u = &self.units[i];
        let class = if u.has_status("slowed") { u.class_slowed_id } else { u.class_id };
        if class < 0 || class as usize >= self.classes.read().unwrap().len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!("unit {} has no movement class", u.id)));
        }
        let out = self.walk_move_path(i, &xs, &ys, enforce_budget);
        let m = xs.len();
        self.last_move_walk = Some((xs[m - 1], ys[m - 1], xs[out.final_idx], ys[out.final_idx], out.reason.to_string()));
        for id in &out.uncovered {
            self.uncover(id);
        }
        if out.final_idx < 1 {
            return Ok(());
        }
        let (tx, ty) = (xs[out.final_idx], ys[out.final_idx]);
        let hex = self.map.pos_index.get(&(tx, ty)).map(|&i| i as i64).unwrap_or(-1);
        let side = {
            let u = &mut self.units[i];
            u.x = tx;
            u.y = ty;
            u.hex = hex;
            u.current_moves = out.mp_left;
            u.drop_status("resting");
            u.side
        };
        if hex >= 0 && self.map.village_terrain[hex as usize] != 0 {
            self.capture_village(hex as usize, side);
        }
        Ok(())
    }
}
