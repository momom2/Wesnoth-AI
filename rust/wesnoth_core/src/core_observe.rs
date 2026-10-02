//! Phase 4: the observation over the core's own arrays (docs/
//! rust_port_plan.md), what `wesnoth_ai.observe.observe(state, side,
//! reach)` computes from a Python state: the per-unit facts straight
//! from the unit records into `observe_slices` (observe.rs), the
//! acting units' landable rows through `landable_rows` (lib.rs) over
//! the movement classes, and the relevant hex set. The same side view
//! answers the move-order planner (`unit_reach`, `side_context`: what
//! `pathfind_sim.ReachContext.for_side` and `unit_reach` compute) and
//! the units a side sees (`visible_ids`, `visibility.units_visible_to`).
//! The Python modules stay the oracle (tests/test_game_core.py,
//! tests/test_rust_moves.py).

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

use crate::core::GameCore;
use crate::observe::{observe_slices, SideView, UnitFacts};
use crate::{dijkstra_reach, landable_rows};

const HIDE_ABILITIES: [&str; 4] = ["ambush", "concealment", "submerge", "nightstalk"];

/// `observe.Observation`'s arrays: the side's view plus, with reach,
/// the acting units (own, not petrified, on the map, moves left or an
/// attack left), their flags, their landable rows [N*H] and the
/// relevant hex set [H].
pub(crate) struct CoreObservation {
    pub view: SideView,
    pub unit_hex: Vec<i64>,
    pub acting: Vec<u8>,
    pub can_move: Vec<u8>,
    pub can_attack: Vec<u8>,
    pub landable: Vec<u8>,
    pub relevant: Vec<u8>,
}

/// A unit's reach per hex: movement points spent, route cost, predecessor.
type ReachArrays<'py> = (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<i64>>);

impl GameCore {
    /// What `side` sees of the board (`observe_slices`) and each unit's
    /// hex index (-1 off the map).
    pub(crate) fn side_view(&self, side: i64) -> (SideView, Vec<i64>) {
        let n = self.units.len();
        let mut ux = vec![0i64; n];
        let mut uy = vec![0i64; n];
        let mut uhex = vec![-1i64; n];
        let mut uside = vec![0i64; n];
        let mut uscenery = vec![0u8; n];
        let mut upetrified = vec![0u8; n];
        let mut uleader = vec![0u8; n];
        let mut uhider = vec![0u8; n];
        let mut uzoc = vec![0u8; n];
        for i in 0..n {
            let u = &self.units[i];
            ux[i] = u.x;
            uy[i] = u.y;
            uhex[i] = u.hex;
            uside[i] = u.side;
            uscenery[i] = self.is_scenery(i) as u8;
            upetrified[i] = u.has_status("petrified") as u8;
            uleader[i] = u.is_leader as u8;
            let hides = u.abilities.iter().any(|a| HIDE_ABILITIES.contains(&a.as_str()));
            uhider[i] = (hides && self.hide_cover_active(i) && !self.is_uncovered(&u.id)) as u8;
            uzoc[i] = (self.unit_level(i) >= 1) as u8;      // the type's zoc=, level > 0 by default
        }
        let facts = UnitFacts {
            ux: &ux, uy: &uy, uhex: &uhex, uside: &uside, uscenery: &uscenery,
            upetrified: &upetrified, uleader: &uleader, uhider: &uhider, uzoc: &uzoc,
        };
        let m = &self.map;
        let view = observe_slices(&m.nbrs, &m.castle_or_keep, &m.keep, &self.recruit_rejected,
                                  self.seen_by(side), &facts, side, self.global.fog_on);
        (view, uhex)
    }

    /// `observe.observe(state, side, reach=...)` over the core.
    pub(crate) fn observe_core(&self, side: i64, reach: bool) -> PyResult<CoreObservation> {
        let n = self.units.len();
        let h = self.map.h;
        let m = &self.map;
        let (view, uhex) = self.side_view(side);
        let upetrified: Vec<u8> = self.units.iter().map(|u| u.has_status("petrified") as u8).collect();
        let uleader: Vec<u8> = self.units.iter().map(|u| u.is_leader as u8).collect();
        let mut out = CoreObservation {
            view, unit_hex: uhex, acting: Vec::new(), can_move: Vec::new(), can_attack: Vec::new(),
            landable: Vec::new(), relevant: Vec::new(),
        };
        if !reach {
            return Ok(out);
        }
        // `observe._add_reach`: the acting units' rows over the stack
        // of their distinct movement classes.
        let mut acting = vec![0u8; n];
        let mut can_move = vec![0u8; n];
        let mut can_attack = vec![0u8; n];
        let mut unit_hexidx = vec![-1i64; n];
        let mut unit_type = vec![0i64; n];
        let mut budget = vec![0i64; n];
        let mut skirm = vec![0u8; n];
        let classes = self.classes.read().unwrap();
        let mut row_of: HashMap<i64, usize> = HashMap::new();
        let mut tm: Vec<i64> = Vec::new();
        let mut td: Vec<i64> = Vec::new();
        for i in 0..n {
            let u = &self.units[i];
            if u.side != side || upetrified[i] != 0 || out.unit_hex[i] < 0 {
                continue;
            }
            let moves = u.current_moves > 0;
            if !(moves || !u.has_attacked) {
                continue;
            }
            acting[i] = 1;
            can_move[i] = moves as u8;
            can_attack[i] = (!u.has_attacked) as u8;
            unit_hexidx[i] = out.unit_hex[i];
            budget[i] = u.current_moves;
            skirm[i] = u.has_ability("skirmisher") as u8;
            if moves {
                let c = if u.has_status("slowed") { u.class_slowed_id } else { u.class_id };
                if c < 0 || c as usize >= classes.len() {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!("unit {} has no movement class", u.id)));
                }
                let next = tm.len() / h;
                let row = *row_of.entry(c).or_insert_with(|| {
                    tm.extend_from_slice(&classes[c as usize].mcost);
                    td.extend_from_slice(&classes[c as usize].dsub);
                    next
                });
                unit_type[i] = row as i64;
            }
        }
        let landable = landable_rows(&m.nbrs, &tm, &td, &unit_hexidx, &unit_type, &budget, &skirm, &can_move,
                                     &out.view.zoc, &out.view.enemy, &out.view.ally, &out.view.occupied)?;
        let mut relevant = vec![0u8; h];
        for j in 0..h {
            if m.village_terrain[j] != 0 || m.castle_or_keep[j] != 0 || out.view.network[j] != 0
                || out.view.occupied[j] != 0 {
                relevant[j] = 1;
            }
        }
        if let Some(l) = (0..n).find(|&i| self.units[i].side == side && uleader[i] != 0) {
            if out.unit_hex[l] >= 0 {
                relevant[out.unit_hex[l] as usize] = 1;
            }
        }
        for i in 0..n {
            if can_move[i] == 0 {
                continue;
            }
            for j in 0..h {
                if landable[i * h + j] != 0 {
                    relevant[j] = 1;
                }
            }
        }
        out.acting = acting;
        out.can_move = can_move;
        out.can_attack = can_attack;
        out.landable = landable;
        out.relevant = relevant;
        Ok(out)
    }
}

/// The observation as a dict of numpy arrays (`CoreState.observe`
/// wraps it into `observe.Observation`).
pub(crate) fn observation_dict<'py>(py: Python<'py>, core: &GameCore, obs: CoreObservation, side: i64)
    -> PyResult<Bound<'py, PyDict>> {
    let n = core.units.len();
    let h = core.map.h;
    let d = PyDict::new(py);
    d.set_item("side", side)?;
    d.set_item("fog_on", core.global.fog_on)?;
    d.set_item("unit_ids", core.units.iter().map(|u| u.id.clone()).collect::<Vec<_>>())?;
    d.set_item("unit_hex", obs.unit_hex.into_pyarray(py))?;
    let v = obs.view;
    d.set_item("seen", v.seen.into_pyarray(py))?;
    d.set_item("visible", v.visible.into_pyarray(py))?;
    d.set_item("zoc", v.zoc.into_pyarray(py))?;
    d.set_item("enemy", v.enemy.into_pyarray(py))?;
    d.set_item("ally", v.ally.into_pyarray(py))?;
    d.set_item("occupied", v.occupied.into_pyarray(py))?;
    d.set_item("inert", v.inert.into_pyarray(py))?;
    d.set_item("recruit_row", v.recruit_row.into_pyarray(py))?;
    d.set_item("network", v.network.into_pyarray(py))?;
    d.set_item("leader_on_keep", v.leader_on_keep)?;
    if !obs.relevant.is_empty() {
        d.set_item("acting", obs.acting.into_pyarray(py))?;
        d.set_item("unit_can_move", obs.can_move.into_pyarray(py))?;
        d.set_item("unit_can_attack", obs.can_attack.into_pyarray(py))?;
        let landable: Bound<'py, PyArray2<u8>> = Array2::from_shape_vec((n, h), obs.landable)
            .expect("landable rows sized N*H")
            .into_pyarray(py);
        d.set_item("landable", landable)?;
        d.set_item("relevant", obs.relevant.into_pyarray(py))?;
    }
    Ok(d)
}

#[pymethods]
impl GameCore {
    /// `pathfind_sim.unit_reach` for the unit on (x, y) with `budget`
    /// movement points, against its side's observable context
    /// (`ReachContext.for_side`: the visible units, the enemies' zones of
    /// control): per hex the movement points spent (-1 unreached), the
    /// route cost (movement plus the defense and ally subcosts, inf
    /// unreached) and the predecessor on the preferred route (-1). The
    /// unit's own hex counts as its side's, which no route re-enters.
    /// None when no unit stands there on the map.
    fn unit_reach<'py>(&self, py: Python<'py>, x: i64, y: i64, budget: i64) -> PyResult<Option<ReachArrays<'py>>> {
        let Some(i) = self.unit_at(x, y) else { return Ok(None) };
        let u = &self.units[i];
        if u.hex < 0 {
            return Ok(None);
        }
        let (view, _) = self.side_view(u.side);
        let c = if u.has_status("slowed") { u.class_slowed_id } else { u.class_id };
        let classes = self.classes.read().unwrap();
        if c < 0 || c as usize >= classes.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!("unit {} has no movement class", u.id)));
        }
        let class = &classes[c as usize];
        let h = self.map.h;
        let (mut mp, mut cost, mut prev) = (vec![-1i64; h], vec![f64::INFINITY; h], vec![-1i64; h]);
        dijkstra_reach(&self.map.nbrs, &class.mcost, &class.dsub, &view.zoc, &view.enemy, &view.ally,
                       u.hex as usize, budget, u.has_ability("skirmisher"), &mut mp, &mut cost, &mut prev);
        Ok(Some((mp.into_pyarray(py), cost.into_pyarray(py), prev.into_pyarray(py))))
    }

    /// `ReachContext.for_side(state, side)` as hex flags [H]: occupied by
    /// a unit the side sees, by a seen enemy (scenery included), by one
    /// of its own, and in a seen enemy's zone of control.
    #[allow(clippy::type_complexity)]
    fn side_context<'py>(&self, py: Python<'py>, side: i64)
        -> (Bound<'py, PyArray1<u8>>, Bound<'py, PyArray1<u8>>, Bound<'py, PyArray1<u8>>, Bound<'py, PyArray1<u8>>) {
        let (view, _) = self.side_view(side);
        (view.occupied.into_pyarray(py), view.enemy.into_pyarray(py), view.ally.into_pyarray(py),
         view.zoc.into_pyarray(py))
    }

    /// `visibility.units_visible_to(state, side)`: the ids of the units
    /// the side sees.
    fn visible_ids(&self, side: i64) -> Vec<String> {
        let (view, _) = self.side_view(side);
        self.units.iter().zip(&view.visible).filter(|(_, &v)| v != 0).map(|(u, _)| u.id.clone()).collect()
    }

    /// `observe.observe(state, side, reach)` as a dict of arrays: side,
    /// fog_on, unit_ids (unit order), unit_hex, seen, visible, zoc,
    /// enemy, ally, occupied, inert, recruit_row, network,
    /// leader_on_keep; with reach also acting, unit_can_move,
    /// unit_can_attack, landable [N, H] and relevant.
    #[pyo3(signature = (side, reach=false))]
    fn observe<'py>(&self, py: Python<'py>, side: i64, reach: bool) -> PyResult<Bound<'py, PyDict>> {
        let obs = self.observe_core(side, reach)?;
        observation_dict(py, self, obs, side)
    }

    /// The registered unit-type names by index (the vocab lookup's key).
    fn type_names(&self) -> Vec<String> {
        self.types.read().unwrap().iter().map(|t| t.name.clone()).collect()
    }

    /// One side's fields: (player, recruits, gold, base income,
    /// villages, faction), or None off the side list.
    fn side_export(&self, side: i64) -> Option<(String, Vec<String>, i64, i64, i64, String)> {
        if side >= 1 && (side as usize) <= self.sides.len() {
            let s = &self.sides[side as usize - 1];
            Some((s.player.clone(), s.recruits.clone(), s.current_gold, s.base_income, s.nb_villages, s.faction.clone()))
        } else {
            None
        }
    }

    /// The whole recruit-row array as a convenience for callers that
    /// keep the Python observation: `PyArray1<u8>` of length H.
    fn recruit_rejected_export<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u8>> {
        self.recruit_rejected.clone().into_pyarray(py)
    }
}
