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

    let mut mp = vec![-1i64; h];
    let mut cost = vec![f64::INFINITY; h];
    let mut prev = vec![-1i64; h];
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
            // EXACT Python composition order:
            // c + mp_charge + subcost * (1/10000).
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

    Ok((
        mp.into_pyarray(py),
        cost.into_pyarray(py),
        prev.into_pyarray(py),
    ))
}

#[pymodule]
fn wesnoth_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(unit_reach_arrays, m)?)?;
    m.add("__phase__", 1)?;
    Ok(())
}
