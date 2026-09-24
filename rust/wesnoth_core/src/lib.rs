//! wesnoth_core — Rust hot-path kernels (docs/rust_port_plan.md).
//!
//! Phase 1: single-turn reachability Dijkstra, a line-for-line port
//! of `tools/pathfind_sim.py::unit_reach`'s array loop. BIT-EXACT
//! contract: identical float composition (f64, same op order),
//! identical heap semantics (total order on (cost, seq) — seq makes
//! keys unique, so ANY correct min-heap pops the same sequence),
//! identical relax condition (strict <). Certified by
//! tests/test_rust_reach.py differential tests; the Python
//! implementation remains the permanent diff oracle.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

mod encode;
mod observe;
mod combat;
mod core;
mod core_step;
mod core_move;
mod core_attack;
mod core_observe;
mod core_fog;
mod core_encode;
mod core_sim;

/// Movement cost >= this is Wesnoth's UNREACHABLE sentinel
/// (movetype.hpp: UNREACHABLE = 99).
const UNREACHABLE: i64 = 99;

/// Wesnoth's tie-break scale (pathfind.cpp:820): subcosts divide by
/// 10000 so they can never outweigh a full movement point. Must
/// compose EXACTLY like Python's `subcost * (1.0 / 10000.0)`.
const SUBCOST_SCALE: f64 = 1.0 / 10000.0;

/// Min-heap entry ordered by (cost, seq) — strict total order
/// because seq is unique per push. BinaryHeap is a max-heap, so the
/// Ord impl is reversed. cost is never NaN (finite sums of finite
/// terms), so partial_cmp cannot fail.
struct Entry {
    cost: f64,
    seq: u64,
    idx: usize,
}

impl PartialEq for Entry {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost && self.seq == other.seq
    }
}
impl Eq for Entry {}
impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Entry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reversed: BinaryHeap pops the LARGEST, we want the smallest.
        other
            .cost
            .partial_cmp(&self.cost)
            .expect("reach costs are never NaN")
            .then_with(|| other.seq.cmp(&self.seq))
    }
}

/// Shared Dijkstra core: fills mp/cost/prev in MAP space. Exact
/// contract as documented on `unit_reach_arrays`.
#[allow(clippy::too_many_arguments)]
fn dijkstra_reach(
    nbrs: &[i64],
    mcost: &[i64],
    dsub: &[i64],
    zoc: &[u8],
    enemy: &[u8],
    ally: &[u8],
    s_idx: usize,
    budget: i64,
    skirmisher: bool,
    mp: &mut [i64],
    cost: &mut [f64],
    prev: &mut [i64],
) {
    mp.fill(-1);
    cost.fill(f64::INFINITY);
    prev.fill(-1);
    mp[s_idx] = 0;
    cost[s_idx] = 0.0;

    let mut seq: u64 = 0;
    let mut heap: BinaryHeap<Entry> = BinaryHeap::new();
    heap.push(Entry { cost: 0.0, seq: 0, idx: s_idx });

    while let Some(Entry { cost: c, idx: i, .. }) = heap.pop() {
        if c > cost[i] {
            continue;
        }
        let spent = mp[i];
        if i != s_idx && !skirmisher && zoc[i] != 0 {
            continue;
        }
        if spent >= budget {
            continue;
        }
        let remaining = budget - spent;
        for &ni in &nbrs[i * 6..i * 6 + 6] {
            if ni < 0 {
                continue;
            }
            let ni = ni as usize;
            if enemy[ni] != 0 {
                continue;
            }
            let terrain_cost = mcost[ni];
            if terrain_cost >= UNREACHABLE || terrain_cost > remaining {
                continue;
            }
            let mp_charge = if !skirmisher && zoc[ni] != 0 {
                remaining // pathfind.cpp:806
            } else {
                terrain_cost
            };
            let mut subcost = dsub[ni];
            if ally[ni] != 0 {
                subcost += 1; // pathfind.cpp:785
            }
            let ncost = c + mp_charge as f64 + subcost as f64 * SUBCOST_SCALE;
            if ncost < cost[ni] {
                cost[ni] = ncost;
                mp[ni] = spent + mp_charge;
                prev[ni] = i as i64;
                seq += 1;
                heap.push(Entry { cost: ncost, seq, idx: ni });
            }
        }
    }
}

/// Single-turn reachability over pre-built per-map arrays.
///
/// Inputs (H = hex count):
///   nbrs   [H*6] i64  — neighbor hex index per (hex, direction),
///                       -1 = off-map; column order IS the Python
///                       hex_neighbors order (heap tie-break parity
///                       depends on it).
///   mcost  [H]  i64   — movement cost per hex for this unit type.
///   dsub   [H]  i64   — defense-pct subcost per hex.
///   zoc/enemy/ally [H] u8 — context flags for the acting side.
///   s_idx, budget, skirmisher — the moving unit.
///
/// Returns (mp [H] i64 (-1 = unreached), cost [H] f64 (inf =
/// unreached), prev [H] i64 (-1 = none)).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn unit_reach_arrays<'py>(
    py: Python<'py>,
    nbrs: PyReadonlyArray1<'py, i64>,
    mcost: PyReadonlyArray1<'py, i64>,
    dsub: PyReadonlyArray1<'py, i64>,
    zoc: PyReadonlyArray1<'py, u8>,
    enemy: PyReadonlyArray1<'py, u8>,
    ally: PyReadonlyArray1<'py, u8>,
    s_idx: usize,
    budget: i64,
    skirmisher: bool,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<i64>>,
)> {
    let nbrs = nbrs.as_slice()?;
    let mcost = mcost.as_slice()?;
    let dsub = dsub.as_slice()?;
    let zoc = zoc.as_slice()?;
    let enemy = enemy.as_slice()?;
    let ally = ally.as_slice()?;
    let h = mcost.len();
    if nbrs.len() != h * 6
        || dsub.len() != h
        || zoc.len() != h
        || enemy.len() != h
        || ally.len() != h
        || s_idx >= h
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "inconsistent array lengths",
        ));
    }

    let mut mp = vec![0i64; h];
    let mut cost = vec![0f64; h];
    let mut prev = vec![0i64; h];
    dijkstra_reach(
        nbrs, mcost, dsub, zoc, enemy, ally, s_idx, budget, skirmisher,
        &mut mp, &mut cost, &mut prev,
    );

    Ok((
        mp.into_pyarray(py),
        cost.into_pyarray(py),
        prev.into_pyarray(py),
    ))
}

/// Landable rows in map space for the units of `unit_hexidx` (-1 =
/// not an acting unit): the hexes a unit can end a move on this turn,
/// reached by the Dijkstra, not its own hex, not visibly occupied
/// (pathfind_sim.UnitReach.landable). An all-zero row for a unit that
/// cannot move. Move rejection is applied at mask time
/// (`rows_from_landable`), so a row is a fact of the terrain and the
/// reach context alone; the relevant hex set is the union of the rows
/// (visibility.relevant_hex_positions, part a).
#[allow(clippy::too_many_arguments)]
pub(crate) fn landable_rows(
    nbrs: &[i64],
    type_mcost: &[i64],
    type_dsub: &[i64],
    unit_hexidx: &[i64],
    unit_type: &[i64],
    unit_budget: &[i64],
    unit_skirm: &[u8],
    unit_can_move: &[u8],
    zoc: &[u8],
    enemy: &[u8],
    ally: &[u8],
    occupied: &[u8],
) -> PyResult<Vec<u8>> {
    let h = zoc.len();
    let un = unit_hexidx.len();
    let t = if h == 0 { 0 } else { type_mcost.len() / h };
    if nbrs.len() != h * 6
        || type_mcost.len() != t * h
        || type_dsub.len() != type_mcost.len()
        || [enemy, ally, occupied].iter().any(|a| a.len() != h)
        || [unit_type, unit_budget].iter().any(|a| a.len() != un)
        || [unit_skirm, unit_can_move].iter().any(|a| a.len() != un)
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "inconsistent array lengths",
        ));
    }
    let mut rows = vec![0u8; un * h];
    let mut mp = vec![0i64; h];
    let mut cost = vec![0f64; h];
    let mut prev = vec![0i64; h];
    for u in 0..un {
        let s = unit_hexidx[u];
        if s < 0 || unit_can_move[u] == 0 {
            continue;
        }
        let s = s as usize;
        if s >= h {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "unit hex index out of range",
            ));
        }
        let ty = unit_type[u] as usize;
        if ty >= t {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "unit type index out of range",
            ));
        }
        dijkstra_reach(
            nbrs,
            &type_mcost[ty * h..(ty + 1) * h],
            &type_dsub[ty * h..(ty + 1) * h],
            zoc,
            enemy,
            ally,
            s,
            unit_budget[u],
            unit_skirm[u] != 0,
            &mut mp,
            &mut cost,
            &mut prev,
        );
        for i in 0..h {
            if mp[i] >= 0 && i != s && occupied[i] == 0 {
                rows[u * h + i] = 1;
            }
        }
    }
    Ok(rows)
}

/// The move and attack rows in token space from landable rows, the
/// semantics of `_build_legality_masks`: a move row is the landable
/// row minus the move-rejected hexes; an enemy token is attackable
/// when a map neighbour of its hex is the unit's own hex or a landable
/// hex (rejection not applied). Units with `unit_hexidx` < 0 get
/// empty rows. `tok_of_hex[i]` maps a map hex to its token slot or -1
/// (the full board or the relevant subset).
///
/// Every non-negative `tok_of_hex` entry must be < `ht`. The caller
/// builds the two from the same hex-position list, so the invariant
/// holds by construction today; it is checked anyway because breaking
/// it does NOT fail loudly. `row[u * ht + tok]` with `tok >= ht` still
/// lands inside the buffer for every unit but the last, so the write
/// silently sets a bit in the NEXT unit's row — a legality row that
/// looks like a model error, not like a kernel bug. Only the last
/// unit's overflow leaves the buffer and panics (CLAUDE.md: failures
/// are visible). One O(H) pass, negligible beside the Dijkstra.
#[allow(clippy::too_many_arguments)]
fn rows_from_landable(
    landable: &[u8],
    nbrs: &[i64],
    tok_of_hex: &[i64],
    unit_hexidx: &[i64],
    unit_can_move: &[u8],
    unit_can_attack: &[u8],
    move_rej: &[u8],
    enemy_hexids: &[i64],
    ht: usize,
) -> PyResult<(Vec<u8>, Vec<u8>)> {
    let h = tok_of_hex.len();
    let un = unit_hexidx.len();
    if landable.len() != un * h
        || nbrs.len() != h * 6
        || move_rej.len() != h
        || [unit_can_move, unit_can_attack].iter().any(|a| a.len() != un)
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "inconsistent array lengths",
        ));
    }
    if tok_of_hex.iter().any(|&t| t >= 0 && t as usize >= ht) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "token index out of range for the row width",
        ));
    }
    let mut move_rows = vec![0u8; un * ht];
    let mut attack_rows = vec![0u8; un * ht];
    for u in 0..un {
        let s = unit_hexidx[u];
        if s < 0 {
            continue;
        }
        let s = s as usize;
        if s >= h {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "unit hex index out of range",
            ));
        }
        let row = &landable[u * h..(u + 1) * h];
        if unit_can_move[u] != 0 {
            for i in 0..h {
                if row[i] != 0 && move_rej[i] == 0 {
                    let tok = tok_of_hex[i];
                    if tok >= 0 {
                        move_rows[u * ht + tok as usize] = 1;
                    }
                }
            }
        }
        if unit_can_attack[u] != 0 {
            let is_attack_pos = |i: usize| -> bool { i == s || row[i] != 0 };
            for &e in enemy_hexids {
                if e < 0 || e as usize >= h {
                    continue;
                }
                let e = e as usize;
                let tok = tok_of_hex[e];
                if tok < 0 {
                    continue;
                }
                for &n in &nbrs[e * 6..e * 6 + 6] {
                    if n >= 0 && is_attack_pos(n as usize) {
                        attack_rows[u * ht + tok as usize] = 1;
                        break;
                    }
                }
            }
        }
    }
    Ok((move_rows, attack_rows))
}

/// `landable_rows` for Python: [UN*H] u8 in map space.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn reach_rows<'py>(
    py: Python<'py>,
    nbrs: PyReadonlyArray1<'py, i64>,
    type_mcost: PyReadonlyArray1<'py, i64>,
    type_dsub: PyReadonlyArray1<'py, i64>,
    unit_hexidx: PyReadonlyArray1<'py, i64>,
    unit_type: PyReadonlyArray1<'py, i64>,
    unit_budget: PyReadonlyArray1<'py, i64>,
    unit_skirm: PyReadonlyArray1<'py, u8>,
    unit_can_move: PyReadonlyArray1<'py, u8>,
    zoc: PyReadonlyArray1<'py, u8>,
    enemy: PyReadonlyArray1<'py, u8>,
    ally: PyReadonlyArray1<'py, u8>,
    occupied: PyReadonlyArray1<'py, u8>,
) -> PyResult<Bound<'py, PyArray1<u8>>> {
    let rows = landable_rows(
        nbrs.as_slice()?,
        type_mcost.as_slice()?,
        type_dsub.as_slice()?,
        unit_hexidx.as_slice()?,
        unit_type.as_slice()?,
        unit_budget.as_slice()?,
        unit_skirm.as_slice()?,
        unit_can_move.as_slice()?,
        zoc.as_slice()?,
        enemy.as_slice()?,
        ally.as_slice()?,
        occupied.as_slice()?,
    )?;
    Ok(rows.into_pyarray(py))
}

/// `rows_from_landable` for Python: (move_rows, attack_rows) as
/// [UN*HT] u8 in token space.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn rows_from_reach<'py>(
    py: Python<'py>,
    landable: PyReadonlyArray1<'py, u8>,
    nbrs: PyReadonlyArray1<'py, i64>,
    tok_of_hex: PyReadonlyArray1<'py, i64>,
    unit_hexidx: PyReadonlyArray1<'py, i64>,
    unit_can_move: PyReadonlyArray1<'py, u8>,
    unit_can_attack: PyReadonlyArray1<'py, u8>,
    move_rej: PyReadonlyArray1<'py, u8>,
    enemy_hexids: PyReadonlyArray1<'py, i64>,
    ht: usize,
) -> PyResult<(Bound<'py, PyArray1<u8>>, Bound<'py, PyArray1<u8>>)> {
    let (m, a) = rows_from_landable(
        landable.as_slice()?,
        nbrs.as_slice()?,
        tok_of_hex.as_slice()?,
        unit_hexidx.as_slice()?,
        unit_can_move.as_slice()?,
        unit_can_attack.as_slice()?,
        move_rej.as_slice()?,
        enemy_hexids.as_slice()?,
        ht,
    )?;
    Ok((m.into_pyarray(py), a.into_pyarray(py)))
}

/// Phase 2: per-STATE move/attack row enumeration in one call, all
/// acting units at once (the marshaling-amortizing boundary the
/// phase-1 measurement demanded): `landable_rows` then
/// `rows_from_landable`. Map space = gs.map.hexes ordering (H hexes);
/// token space = the encoder's hex-token ordering (HT slots).
///
/// Per unit slot u (of UN): unit_hexidx[u] (map hex, -1 = not
/// acting), unit_type[u] (row into the [T*H] per-type terrain
/// stacks), unit_budget / unit_skirm / unit_can_move /
/// unit_can_attack. Semantics mirror _build_legality_masks exactly
/// (tests/test_rust_enumerate.py).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn enumerate_moves<'py>(
    py: Python<'py>,
    nbrs: PyReadonlyArray1<'py, i64>,
    tok_of_hex: PyReadonlyArray1<'py, i64>,
    type_mcost: PyReadonlyArray1<'py, i64>,
    type_dsub: PyReadonlyArray1<'py, i64>,
    unit_hexidx: PyReadonlyArray1<'py, i64>,
    unit_type: PyReadonlyArray1<'py, i64>,
    unit_budget: PyReadonlyArray1<'py, i64>,
    unit_skirm: PyReadonlyArray1<'py, u8>,
    unit_can_move: PyReadonlyArray1<'py, u8>,
    unit_can_attack: PyReadonlyArray1<'py, u8>,
    zoc: PyReadonlyArray1<'py, u8>,
    enemy: PyReadonlyArray1<'py, u8>,
    ally: PyReadonlyArray1<'py, u8>,
    occupied: PyReadonlyArray1<'py, u8>,
    move_rej: PyReadonlyArray1<'py, u8>,
    enemy_hexids: PyReadonlyArray1<'py, i64>,
    ht: usize,
) -> PyResult<(Bound<'py, PyArray1<u8>>, Bound<'py, PyArray1<u8>>)> {
    let nbrs = nbrs.as_slice()?;
    let tok_of_hex = tok_of_hex.as_slice()?;
    let unit_hexidx = unit_hexidx.as_slice()?;
    let unit_can_move = unit_can_move.as_slice()?;
    let unit_can_attack = unit_can_attack.as_slice()?;
    let zoc = zoc.as_slice()?;
    if zoc.len() != tok_of_hex.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "inconsistent array lengths",
        ));
    }
    let rows = landable_rows(
        nbrs,
        type_mcost.as_slice()?,
        type_dsub.as_slice()?,
        unit_hexidx,
        unit_type.as_slice()?,
        unit_budget.as_slice()?,
        unit_skirm.as_slice()?,
        unit_can_move,
        zoc,
        enemy.as_slice()?,
        ally.as_slice()?,
        occupied.as_slice()?,
    )?;
    let (m, a) = rows_from_landable(
        &rows,
        nbrs,
        tok_of_hex,
        unit_hexidx,
        unit_can_move,
        unit_can_attack,
        move_rej.as_slice()?,
        enemy_hexids.as_slice()?,
        ht,
    )?;
    Ok((m.into_pyarray(py), a.into_pyarray(py)))
}

#[pymodule]
fn wesnoth_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(unit_reach_arrays, m)?)?;
    m.add_function(wrap_pyfunction!(enumerate_moves, m)?)?;
    m.add_function(wrap_pyfunction!(reach_rows, m)?)?;
    m.add_function(wrap_pyfunction!(rows_from_reach, m)?)?;
    m.add_function(wrap_pyfunction!(encode::encode_raw_streams, m)?)?;
    m.add_function(wrap_pyfunction!(observe::observe_side, m)?)?;
    m.add_function(wrap_pyfunction!(combat::resolve_attack, m)?)?;
    m.add_function(wrap_pyfunction!(combat::random_int, m)?)?;
    m.add_class::<core::GameCore>()?;
    // 9: rows_from_landable rejects a token index >= the row width
    // instead of writing it into the next unit's row.
    // 10: nightstalk's cover reads the illuminated time of day; the
    // map's cover flags are named hides_ambush / hides_concealment /
    // hides_submerge.
    // 12: vision follows the engine (docs/wesnoth_rules.md "Vision and
    // fog"): observe_side takes the side's seen hexes instead of
    // drawing a disc, and GameCore tracks each side's cleared hexes.
    // 13: apply_init_side hides the side's revealed hiders again after
    // turn 1; refresh_uncovered is gone.
    m.add("__phase__", 13)?;
    Ok(())
}
