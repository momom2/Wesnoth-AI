//! Phase 2c: the observable state of one side in one call (docs/
//! rust_port_plan.md; the 2026-09-11 worker profile). Given the map
//! geometry and every unit as flat arrays, computes what the Python
//! originals compute separately and rebuild every decision:
//!
//!   the vision disc        `visibility.visible_hexes_for`
//!   unit visibility        `visibility.units_visible_to` (rules 1-3,
//!                          the hide-cover gate with adjacency discovery)
//!   the reach context      `action_sampler._build_legality_masks`
//!                          (occupied / ally / enemy / ZoC flags)
//!   the recruit row        `visibility.leader_castle_network` +
//!                          `action_sampler._recruit_hex_mask`
//!
//! Map space throughout: hex index = position in `gs.map.hexes`
//! (pathfind_sim's ordering); `nbrs` columns follow
//! `tools.abilities.hex_neighbors` (N, NE, SE, S, SW, NW). Every
//! rule is a transcription of the Python, which stays the diff
//! oracle (tests/test_rust_observe.py). `observe_slices` is the
//! computation over slices; `observe_side` is its numpy entry and
//! the core (core_observe.rs) calls it over its own arrays.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use std::collections::VecDeque;

/// Wesnoth hex distance, odd-q offset (`visibility._hex_distance`,
/// map_location.cpp::distance_between).
#[inline]
pub(crate) fn hex_distance(ax: i64, ay: i64, bx: i64, by: i64) -> i64 {
    let hd = (ax - bx).abs();
    let a_even = ax & 1 == 0;
    let b_even = bx & 1 == 0;
    let vpenalty =
        if (a_even && !b_even && ay <= by) || (b_even && !a_even && by <= ay) { 1 } else { 0 };
    hd.max((ay - by).abs() + hd / 2 + vpenalty)
}

/// The six neighbours of (x, y), `tools.abilities.hex_neighbors` order.
#[inline]
pub(crate) fn neighbours(x: i64, y: i64) -> [(i64, i64); 6] {
    if x % 2 == 0 {
        [(x, y - 1), (x + 1, y - 1), (x + 1, y), (x, y + 1), (x - 1, y), (x - 1, y - 1)]
    } else {
        [(x, y - 1), (x + 1, y), (x + 1, y + 1), (x, y + 1), (x - 1, y + 1), (x - 1, y)]
    }
}

/// Per unit i (N, `gs.map.units` order): ux, uy, uhex (map index or
/// -1), uside, uradius (sight), uscenery, upetrified, uleader, uhider
/// (hide-cover active and not uncovered), uzoc (level >= 1).
pub(crate) struct UnitFacts<'a> {
    pub ux: &'a [i64],
    pub uy: &'a [i64],
    pub uhex: &'a [i64],
    pub uside: &'a [i64],
    pub uradius: &'a [i64],
    pub uscenery: &'a [u8],
    pub upetrified: &'a [u8],
    pub uleader: &'a [u8],
    pub uhider: &'a [u8],
    pub uzoc: &'a [u8],
}

/// What a side observes, map space: the vision disc, the units it
/// sees, the reach context, the recruit row and the castle network.
pub(crate) struct SideView {
    pub disc: Vec<u8>,
    pub visible: Vec<u8>,
    pub zoc: Vec<u8>,
    pub enemy: Vec<u8>,
    pub ally: Vec<u8>,
    pub occupied: Vec<u8>,
    pub inert: Vec<u8>,
    pub recruit_row: Vec<u8>,
    pub network: Vec<u8>,
    pub leader_on_keep: bool,
}

/// The observation over slices (see the module doc). Per hex h (H):
/// hx, hy, nbrs[h*6..], castle_or_keep, keep, recruit_rej. `network`
/// is the leader's castle network itself (the BFS closure without
/// the keep, occupied hexes included), what
/// `visibility.leader_castle_network` returns.
#[allow(clippy::too_many_arguments)]
pub(crate) fn observe_slices(
    hx: &[i64],
    hy: &[i64],
    nbrs: &[i64],
    castle_or_keep: &[u8],
    keep: &[u8],
    recruit_rej: &[u8],
    u: &UnitFacts<'_>,
    side: i64,
    fog_on: bool,
) -> SideView {
    let h = hx.len();
    let n = u.ux.len();
    let (ux, uy, uhex, uside, uradius) = (u.ux, u.uy, u.uhex, u.uside, u.uradius);
    let (uscenery, upetrified, uleader, uhider, uzoc) =
        (u.uscenery, u.upetrified, u.uleader, u.uhider, u.uzoc);

    // 1. The vision disc: the union of the own units' sight discs.
    let mut disc = vec![0u8; h];
    for i in 0..n {
        if uside[i] != side {
            continue;
        }
        let r = uradius[i];
        for j in 0..h {
            if disc[j] == 0 && hex_distance(ux[i], uy[i], hx[j], hy[j]) <= r {
                disc[j] = 1;
            }
        }
    }

    // 2. Unit visibility (units_visible_to). A hider is discovered
    // while any unit of another side, not scenery, not petrified,
    // stands on an adjacent hex (would_be_discovered).
    let discovered = |i: usize| -> bool {
        let adj = neighbours(ux[i], uy[i]);
        (0..n).any(|j| {
            uside[j] != uside[i]
                && uscenery[j] == 0
                && upetrified[j] == 0
                && adj.iter().any(|&(x, y)| x == ux[j] && y == uy[j])
        })
    };
    let mut visible = vec![0u8; n];
    for i in 0..n {
        if uside[i] == side || uscenery[i] != 0 {
            visible[i] = 1;
            continue;
        }
        if uhider[i] != 0 && !discovered(i) {
            continue;
        }
        if !fog_on {
            visible[i] = 1;
            continue;
        }
        if uhex[i] >= 0 && disc[uhex[i] as usize] != 0 {
            visible[i] = 1;
        }
    }

    // 3. The reach context of the visible units, map space. `inert`
    // marks hexes of visible scenery (statues, attackless side>=3
    // furniture): occupied, never an attack target (the mask
    // builder's occupancy code 3).
    let mut zoc = vec![0u8; h];
    let mut enemy = vec![0u8; h];
    let mut ally = vec![0u8; h];
    let mut occupied = vec![0u8; h];
    let mut inert = vec![0u8; h];
    for i in 0..n {
        if visible[i] == 0 || uhex[i] < 0 {
            continue;
        }
        let hi = uhex[i] as usize;
        occupied[hi] = 1;
        if uscenery[i] != 0 {
            inert[hi] = 1;
        }
        if uside[i] == side {
            ally[hi] = 1;
            continue;
        }
        enemy[hi] = 1;
        if uscenery[i] != 0 || upetrified[i] != 0 || uzoc[i] == 0 {
            continue;
        }
        for &nb in &nbrs[hi * 6..hi * 6 + 6] {
            if nb >= 0 {
                zoc[nb as usize] = 1;
            }
        }
    }

    // 4. The recruit row: the castle network of the side's first
    // leader on a keep (BFS over castle/keep hexes, the keep itself
    // excluded), minus visibly occupied and rejected hexes.
    let mut recruit_row = vec![0u8; h];
    let mut network = vec![0u8; h];
    let mut leader_on_keep = false;
    if let Some(l) = (0..n).find(|&i| uside[i] == side && uleader[i] != 0) {
        if uhex[l] >= 0 && keep[uhex[l] as usize] != 0 {
            leader_on_keep = true;
            let start = uhex[l] as usize;
            let mut seen = vec![false; h];
            seen[start] = true;
            let mut queue = VecDeque::new();
            queue.push_back(start);
            while let Some(cur) = queue.pop_front() {
                for &nb in &nbrs[cur * 6..cur * 6 + 6] {
                    if nb < 0 {
                        continue;
                    }
                    let nb = nb as usize;
                    if seen[nb] || castle_or_keep[nb] == 0 {
                        continue;
                    }
                    seen[nb] = true;
                    network[nb] = 1;
                    queue.push_back(nb);
                    if occupied[nb] == 0 && recruit_rej[nb] == 0 {
                        recruit_row[nb] = 1;
                    }
                }
            }
        }
    }

    SideView { disc, visible, zoc, enemy, ally, occupied, inert, recruit_row, network, leader_on_keep }
}

/// `observe_slices` over numpy arrays. Returns (disc[H], visible[N],
/// zoc[H], enemy[H], ally[H], occupied[H], inert[H], recruit_row[H],
/// network[H], leader_on_keep) as u8 arrays and a bool.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn observe_side<'py>(
    py: Python<'py>,
    hx: PyReadonlyArray1<'py, i64>,
    hy: PyReadonlyArray1<'py, i64>,
    nbrs: PyReadonlyArray1<'py, i64>,
    castle_or_keep: PyReadonlyArray1<'py, u8>,
    keep: PyReadonlyArray1<'py, u8>,
    recruit_rej: PyReadonlyArray1<'py, u8>,
    ux: PyReadonlyArray1<'py, i64>,
    uy: PyReadonlyArray1<'py, i64>,
    uhex: PyReadonlyArray1<'py, i64>,
    uside: PyReadonlyArray1<'py, i64>,
    uradius: PyReadonlyArray1<'py, i64>,
    uscenery: PyReadonlyArray1<'py, u8>,
    upetrified: PyReadonlyArray1<'py, u8>,
    uleader: PyReadonlyArray1<'py, u8>,
    uhider: PyReadonlyArray1<'py, u8>,
    uzoc: PyReadonlyArray1<'py, u8>,
    side: i64,
    fog_on: bool,
) -> PyResult<(
    Bound<'py, PyArray1<u8>>,
    Bound<'py, PyArray1<u8>>,
    Bound<'py, PyArray1<u8>>,
    Bound<'py, PyArray1<u8>>,
    Bound<'py, PyArray1<u8>>,
    Bound<'py, PyArray1<u8>>,
    Bound<'py, PyArray1<u8>>,
    Bound<'py, PyArray1<u8>>,
    Bound<'py, PyArray1<u8>>,
    bool,
)> {
    let hx = hx.as_slice()?;
    let hy = hy.as_slice()?;
    let nbrs = nbrs.as_slice()?;
    let castle_or_keep = castle_or_keep.as_slice()?;
    let keep = keep.as_slice()?;
    let recruit_rej = recruit_rej.as_slice()?;
    let facts = UnitFacts {
        ux: ux.as_slice()?,
        uy: uy.as_slice()?,
        uhex: uhex.as_slice()?,
        uside: uside.as_slice()?,
        uradius: uradius.as_slice()?,
        uscenery: uscenery.as_slice()?,
        upetrified: upetrified.as_slice()?,
        uleader: uleader.as_slice()?,
        uhider: uhider.as_slice()?,
        uzoc: uzoc.as_slice()?,
    };
    let h = hx.len();
    let n = facts.ux.len();
    if hy.len() != h
        || nbrs.len() != h * 6
        || [castle_or_keep, keep, recruit_rej].iter().any(|a| a.len() != h)
        || [facts.uy, facts.uhex, facts.uside, facts.uradius].iter().any(|a| a.len() != n)
        || [facts.uscenery, facts.upetrified, facts.uleader, facts.uhider, facts.uzoc]
            .iter()
            .any(|a| a.len() != n)
        || facts.uhex.iter().any(|&i| i >= h as i64)
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "inconsistent array lengths",
        ));
    }
    let v = observe_slices(hx, hy, nbrs, castle_or_keep, keep, recruit_rej, &facts, side, fog_on);
    Ok((
        v.disc.into_pyarray(py),
        v.visible.into_pyarray(py),
        v.zoc.into_pyarray(py),
        v.enemy.into_pyarray(py),
        v.ally.into_pyarray(py),
        v.occupied.into_pyarray(py),
        v.inert.into_pyarray(py),
        v.recruit_row.into_pyarray(py),
        v.network.into_pyarray(py),
        v.leader_on_keep,
    ))
}
