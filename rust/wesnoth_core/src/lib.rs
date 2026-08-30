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

/// Phase 2: per-STATE move/attack row enumeration — one call per
/// decision, all acting units at once (the marshaling-amortizing
/// boundary the phase-1 measurement demanded).
///
/// Map space = gs.map.hexes ordering (H hexes); token space = the
/// encoder's hex-token ordering (HT slots). `tok_of_hex[i]` maps a
/// map hex to its token slot or -1.
///
/// Per unit slot u (of UN):
///   unit_hexidx[u] — map hex of the unit, -1 = slot not eligible;
///   unit_type[u]   — row into the [T*H] per-type terrain stacks;
///   unit_budget / unit_skirm / unit_can_move / unit_can_attack.
///
/// Semantics mirror _build_legality_masks exactly:
///   move row:   landable (reached, not start, not visibly
///               occupied) AND not move-rejected, in token space —
///               only when can_move;
///   attack row: enemy token e is attackable iff some map-neighbor
///               of e is the unit's own hex, or (can_move) a
///               landable hex; move-rejection does NOT filter
///               attack positions (matches Python).
/// Dijkstra runs only when can_move (Python calls it always but
/// consumes landable only under can_move — identical outputs).
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
    let type_mcost = type_mcost.as_slice()?;
    let type_dsub = type_dsub.as_slice()?;
    let unit_hexidx = unit_hexidx.as_slice()?;
    let unit_type = unit_type.as_slice()?;
    let unit_budget = unit_budget.as_slice()?;
    let unit_skirm = unit_skirm.as_slice()?;
    let unit_can_move = unit_can_move.as_slice()?;
    let unit_can_attack = unit_can_attack.as_slice()?;
    let zoc = zoc.as_slice()?;
    let enemy = enemy.as_slice()?;
    let ally = ally.as_slice()?;
    let occupied = occupied.as_slice()?;
    let move_rej = move_rej.as_slice()?;
    let enemy_hexids = enemy_hexids.as_slice()?;

    let h = tok_of_hex.len();
    let un = unit_hexidx.len();
    let t = if h == 0 { 0 } else { type_mcost.len() / h };
    if nbrs.len() != h * 6
        || type_mcost.len() != t * h
        || type_dsub.len() != type_mcost.len()
        || [zoc, enemy, ally, occupied, move_rej]
            .iter()
            .any(|a| a.len() != h)
        || [unit_type, unit_budget].iter().any(|a| a.len() != un)
        || [unit_skirm, unit_can_move, unit_can_attack]
            .iter()
            .any(|a| a.len() != un)
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "inconsistent array lengths",
        ));
    }

    let mut move_rows = vec![0u8; un * ht];
    let mut attack_rows = vec![0u8; un * ht];
    let mut mp = vec![0i64; h];
    let mut cost = vec![0f64; h];
    let mut prev = vec![0i64; h];

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
        let can_move = unit_can_move[u] != 0;
        let can_attack = unit_can_attack[u] != 0;
        let ty = unit_type[u] as usize;
        if can_move {
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
                if mp[i] >= 0 && i != s && occupied[i] == 0 && move_rej[i] == 0
                {
                    let tok = tok_of_hex[i];
                    if tok >= 0 {
                        move_rows[u * ht + tok as usize] = 1;
                    }
                }
            }
        }
        if can_attack {
            // attack position = own hex, or (can_move) landable hex
            // (occupied excluded, move-rejection NOT applied).
            let is_attack_pos = |i: usize| -> bool {
                i == s
                    || (can_move
                        && mp[i] >= 0
                        && i != s
                        && occupied[i] == 0)
            };
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

    Ok((move_rows.into_pyarray(py), attack_rows.into_pyarray(py)))
}

#[pymodule]
fn wesnoth_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(unit_reach_arrays, m)?)?;
    m.add_function(wrap_pyfunction!(enumerate_moves, m)?)?;
    m.add("__phase__", 2)?;
    Ok(())
}
