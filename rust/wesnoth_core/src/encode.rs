//! Phase 2b (docs/rust_port_plan.md): `encoder.encode_raw`'s per-encode
//! stream loops. Python keeps the slot orderings, the vocab lookups,
//! the fog / ownership predicates (dict and set lookups on Python
//! objects) and RawEncoded's Python-object fields; it hands the
//! gathered facts over as flat arrays and this kernel composes every
//! numpy array of RawEncoded in one call.
//!
//! BIT-EXACT contract with encoder.py: each feature is the f64
//! expression Python evaluates (Python float arithmetic, same operand
//! order), cast once to f32 -- the cast numpy applies when a list of
//! Python floats lands in a float32 array. Integer inputs arrive as
//! f64 (exact below 2^53), so an int or float attribute on the Python
//! side yields the same value here.

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use numpy::PyUntypedArrayMethods;
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;

/// encoder.py UNIT_NUMERIC_FEATS: the per-unit numeric feature table
/// (max_hp, hp ratio, max_moves, moves ratio, max_exp, exp ratio,
/// cost, is_leader, has_attacked); the alignment one-hot follows.
const UNIT_NUMERIC_FEATS: usize = 9;
/// Columns of `unit_ints` (encoder._unit_rows): type id, side code,
/// x, y, alignment, is_leader, has_attacked.
const UNIT_INT_COLS: usize = 7;
/// Columns of `unit_stats` (encoder._unit_rows): max_hp, current_hp,
/// max_moves, current_moves, max_exp, current_exp, cost.
const UNIT_STAT_COLS: usize = 7;
/// Columns of `recruit_stats` (encoder._recruit_stats_for): max_hp,
/// max_moves, max_exp, cost, alignment.
const RECRUIT_STAT_COLS: usize = 5;
/// Columns of `village_entries` (encoder._village_entries): hex slot,
/// owner code (1 ours, 2 another side's, 0 neutral), owner visible.
const VILLAGE_ENTRY_COLS: usize = 3;
/// encoder.py NUM_HEX_MODIFIERS / NUM_HEX_DYNAMIC_FLAGS / GLOBAL_FEAT_DIM.
pub(crate) const NUM_HEX_MODIFIERS: usize = 3;
pub(crate) const NUM_HEX_DYNAMIC_FLAGS: usize = 3;
pub(crate) const GLOBAL_FEAT_DIM: usize = 6;
/// Order of the `norms` argument (encoder.py's module values).
pub(crate) const NUM_NORMS: usize = 8;

/// Normalization divisors, in `norms` order.
struct Norms {
    hp: f64,
    moves: f64,
    exp: f64,
    cost: f64,
    gold: f64,
    income: f64,
    villages: f64,
    turn: f64,
}

impl Norms {
    fn from_array(a: [f64; NUM_NORMS]) -> Self {
        Norms {
            hp: a[0],
            moves: a[1],
            exp: a[2],
            cost: a[3],
            gold: a[4],
            income: a[5],
            villages: a[6],
            turn: a[7],
        }
    }
}

/// encoder.py position clamp: `0 if v < 0 else (LIMIT if v > LIMIT else v)`.
fn clamp_pos(v: i64, limit: i64) -> i64 {
    if v < 0 {
        0
    } else if v > limit {
        limit
    } else {
        v
    }
}

/// Python's `max(x, 1)`: the second argument wins only when strictly
/// greater (so a NaN `x` stays NaN, unlike f64::max).
fn py_max_1(x: f64) -> f64 {
    if 1.0 > x {
        1.0
    } else {
        x
    }
}

/// `alignment_onehot[idx] = 1.0` on a Python list of `n` zeros
/// (encoder._unit_features / _recruit_features_for): negative indices
/// count from the end, anything else raises IndexError.
fn onehot_index(idx: i64, n: usize) -> PyResult<usize> {
    let n_i = n as i64;
    let i = if idx < 0 { idx + n_i } else { idx };
    if i < 0 || i >= n_i {
        return Err(PyIndexError::new_err("alignment index out of range"));
    }
    Ok(i as usize)
}

/// encoder._unit_features: the numeric block then the alignment
/// one-hot, written into `out` (length UNIT_NUMERIC_FEATS + n_align).
fn unit_feature_row(
    stats: &[f64],
    alignment: i64,
    is_leader: i64,
    has_attacked: i64,
    norms: &Norms,
    n_align: usize,
    out: &mut [f32],
) -> PyResult<()> {
    let max_hp = py_max_1(stats[0]);
    let max_mv = py_max_1(stats[2]);
    let max_xp = py_max_1(stats[4]);
    out[0] = (stats[0] / norms.hp) as f32;
    out[1] = (stats[1] / max_hp) as f32;
    out[2] = (stats[2] / norms.moves) as f32;
    out[3] = (stats[3] / max_mv) as f32;
    out[4] = (stats[4] / norms.exp) as f32;
    out[5] = (stats[5] / max_xp) as f32;
    out[6] = (stats[6] / norms.cost) as f32;
    out[7] = if is_leader != 0 { 1.0 } else { 0.0 };
    out[8] = if has_attacked != 0 { 1.0 } else { 0.0 };
    write_onehot(alignment, n_align, &mut out[UNIT_NUMERIC_FEATS..])
}

/// encoder._recruit_features_for: a full-HP / 0-MP / 0-XP / non-leader
/// phantom of the type (stats = max_hp, max_moves, max_exp, cost,
/// alignment).
fn recruit_feature_row(
    stats: &[f64],
    norms: &Norms,
    n_align: usize,
    out: &mut [f32],
) -> PyResult<()> {
    out[0] = (stats[0] / norms.hp) as f32;
    out[1] = 1.0;
    out[2] = (stats[1] / norms.moves) as f32;
    out[3] = 0.0;
    out[4] = (stats[2] / norms.exp) as f32;
    out[5] = 0.0;
    out[6] = (stats[3] / norms.cost) as f32;
    out[7] = 0.0;
    out[8] = 0.0;
    write_onehot(stats[4] as i64, n_align, &mut out[UNIT_NUMERIC_FEATS..])
}

fn write_onehot(alignment: i64, n_align: usize, out: &mut [f32]) -> PyResult<()> {
    out.fill(0.0);
    out[onehot_index(alignment, n_align)?] = 1.0;
    Ok(())
}

/// encoder.py global features: turn, side in [-1, 1], gold, income,
/// our villages, their villages (`globals` order).
fn global_feature_row(g: &[f64; GLOBAL_FEAT_DIM], norms: &Norms) -> Vec<f32> {
    vec![
        (g[0] / norms.turn) as f32,
        ((g[1] - 1.5) * 2.0) as f32, // 1 -> -1, 2 -> +1
        (g[2] / norms.gold) as f32,
        (g[3] / norms.income) as f32,
        (g[4] / norms.villages) as f32,
        (g[5] / norms.villages) as f32,
    ]
}

fn to_array2<'py>(
    py: Python<'py>,
    rows: usize,
    cols: usize,
    data: Vec<f32>,
) -> Bound<'py, PyArray2<f32>> {
    Array2::from_shape_vec((rows, cols), data)
        .expect("row-major buffer sized rows*cols")
        .into_pyarray(py)
}

fn bad_len(what: &str) -> PyErr {
    PyValueError::new_err(format!("inconsistent array length: {what}"))
}

type HexArrays<'py> = (Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>);
type StreamArrays<'py> = (
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray2<f32>>,
);

/// The arrays of `encode_raw_streams` before they become numpy
/// (the core's encoder builds them from its own state, core_encode.rs).
pub(crate) struct Composed {
    pub u: usize,
    pub r: usize,
    pub feat_dim: usize,
    pub modifier_flags: Vec<f32>,   // [H*3]
    pub dynamic_flags: Vec<f32>,    // [H*3]
    pub is_ours: Vec<f32>,
    pub type_ids: Vec<i64>,
    pub side_ids: Vec<i64>,
    pub xs: Vec<i64>,
    pub ys: Vec<i64>,
    pub feats: Vec<f32>,            // [U*feat_dim]
    pub r_ids: Vec<i64>,
    pub lx: i64,
    pub ly: i64,
    pub r_feats: Vec<f32>,          // [R*feat_dim]
    pub global_feats: Vec<f32>,
}

/// Every RawEncoded array from the gathered facts, over slices.
///
/// Inputs (H hexes, U units, R recruits, C village candidates):
///   static_modifier_flags [H*3] f32 -- the cached static hex bits;
///                         copied, column 0 set per owner visibility.
///   village_entries [C*3] i64 -- (slot, owner code, owner visible).
///   rejected_slots  [K]   i64 -- hexes a recruit bounced on this turn.
///   unit_ints  [U*7] i64, unit_stats [U*7] f64 -- see the column
///                         constants; positions are clamped here.
///   recruit_type_ids [R] i64, recruit_stats [R*5] f64.
///   leader_x/y -- the mover's leader position (recruit phantoms sit
///                         there), clamped here.
///   globals [6] f64 -- turn, current side, gold, income, our
///                         villages, their villages.
///   norms [8] f64 -- HP, MOVES, EXP, COST, GOLD, INCOME, VILLAGES,
///                         TURN divisors.
///   map_limit -- MAX_MAP_SIZE - 1; num_alignments -- one-hot width.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compose_streams(
    static_modifier_flags: &[f32],
    h: usize,
    village_entries: &[i64],
    rejected_slots: &[i64],
    unit_ints: &[i64],
    unit_stats: &[f64],
    recruit_type_ids: &[i64],
    recruit_stats: &[f64],
    leader_x: i64,
    leader_y: i64,
    globals: [f64; GLOBAL_FEAT_DIM],
    norms: [f64; NUM_NORMS],
    map_limit: i64,
    num_alignments: usize,
) -> PyResult<Composed> {
    let norms = Norms::from_array(norms);
    let feat_dim = UNIT_NUMERIC_FEATS + num_alignments;

    // ---- hexes ----
    if static_modifier_flags.len() != h * NUM_HEX_MODIFIERS {
        return Err(bad_len("static_modifier_flags"));
    }
    if village_entries.len() % VILLAGE_ENTRY_COLS != 0 {
        return Err(bad_len("village_entries"));
    }
    let mut modifier_flags = static_modifier_flags.to_vec();
    let mut dynamic_flags = vec![0f32; h * NUM_HEX_DYNAMIC_FLAGS];
    let slot_of = |v: i64| -> PyResult<usize> {
        if v < 0 || v as usize >= h {
            return Err(PyIndexError::new_err("hex slot out of range"));
        }
        Ok(v as usize)
    };
    // encoder._python_hex_arrays: the village bit shows when the
    // owner is visible; ours / theirs are exclusive, theirs also
    // fog-gated.
    for e in village_entries.chunks_exact(VILLAGE_ENTRY_COLS) {
        let slot = slot_of(e[0])?;
        let code = e[1];
        let visible = e[2] != 0;
        if visible {
            modifier_flags[slot * NUM_HEX_MODIFIERS] = 1.0;
        }
        if code == 1 {
            dynamic_flags[slot * NUM_HEX_DYNAMIC_FLAGS + 1] = 1.0;
        } else if code == 2 && visible {
            dynamic_flags[slot * NUM_HEX_DYNAMIC_FLAGS + 2] = 1.0;
        }
    }
    for &r in rejected_slots {
        dynamic_flags[slot_of(r)? * NUM_HEX_DYNAMIC_FLAGS] = 1.0;
    }

    // ---- units ----
    if unit_ints.len() % UNIT_INT_COLS != 0 {
        return Err(bad_len("unit_ints"));
    }
    let u = unit_ints.len() / UNIT_INT_COLS;
    if unit_stats.len() != u * UNIT_STAT_COLS {
        return Err(bad_len("unit_stats"));
    }
    let mut is_ours = vec![0f32; u];
    let mut type_ids = vec![0i64; u];
    let mut side_ids = vec![0i64; u];
    let mut xs = vec![0i64; u];
    let mut ys = vec![0i64; u];
    let mut feats = vec![0f32; u * feat_dim];
    for i in 0..u {
        let ints = &unit_ints[i * UNIT_INT_COLS..(i + 1) * UNIT_INT_COLS];
        let stats = &unit_stats[i * UNIT_STAT_COLS..(i + 1) * UNIT_STAT_COLS];
        type_ids[i] = ints[0];
        side_ids[i] = ints[1];
        is_ours[i] = if ints[1] == 0 { 1.0 } else { 0.0 }; // side code 0 = ours
        xs[i] = clamp_pos(ints[2], map_limit);
        ys[i] = clamp_pos(ints[3], map_limit);
        unit_feature_row(
            stats,
            ints[4],
            ints[5],
            ints[6],
            &norms,
            num_alignments,
            &mut feats[i * feat_dim..(i + 1) * feat_dim],
        )?;
    }

    // ---- recruits: the mover's own list only, phantoms at its
    // leader's position ----
    let r = recruit_type_ids.len();
    if recruit_stats.len() != r * RECRUIT_STAT_COLS {
        return Err(bad_len("recruit_stats"));
    }
    let lx = clamp_pos(leader_x, map_limit);
    let ly = clamp_pos(leader_y, map_limit);
    let mut r_feats = vec![0f32; r * feat_dim];
    for i in 0..r {
        recruit_feature_row(
            &recruit_stats[i * RECRUIT_STAT_COLS..(i + 1) * RECRUIT_STAT_COLS],
            &norms,
            num_alignments,
            &mut r_feats[i * feat_dim..(i + 1) * feat_dim],
        )?;
    }
    let global_feats = global_feature_row(&globals, &norms);
    Ok(Composed {
        u, r, feat_dim, modifier_flags, dynamic_flags, is_ours, type_ids, side_ids, xs, ys, feats,
        r_ids: recruit_type_ids.to_vec(), lx, ly, r_feats, global_feats,
    })
}

/// Every numpy array of RawEncoded from the facts Python gathered
/// (`compose_streams` over numpy inputs; the argument doc is there).
///
/// Returns ((hex_modifier_flags, hex_dynamic_flags),
///          (unit_is_ours, unit_type_ids, unit_side_ids, unit_xs,
///           unit_ys, unit_feats),
///          (recruit_is_ours, recruit_type_ids, recruit_side_ids,
///           recruit_xs, recruit_ys, recruit_feats),
///          global_feats).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn encode_raw_streams<'py>(
    py: Python<'py>,
    static_modifier_flags: PyReadonlyArray2<'py, f32>,
    village_entries: PyReadonlyArray1<'py, i64>,
    rejected_slots: PyReadonlyArray1<'py, i64>,
    unit_ints: PyReadonlyArray1<'py, i64>,
    unit_stats: PyReadonlyArray1<'py, f64>,
    recruit_type_ids: PyReadonlyArray1<'py, i64>,
    recruit_stats: PyReadonlyArray1<'py, f64>,
    leader_x: i64,
    leader_y: i64,
    globals: [f64; GLOBAL_FEAT_DIM],
    norms: [f64; NUM_NORMS],
    map_limit: i64,
    num_alignments: usize,
) -> PyResult<(
    HexArrays<'py>,
    StreamArrays<'py>,
    StreamArrays<'py>,
    Bound<'py, PyArray1<f32>>,
)> {
    let shape = static_modifier_flags.shape();
    if shape.len() != 2 || shape[1] != NUM_HEX_MODIFIERS {
        return Err(bad_len("static_modifier_flags"));
    }
    let h = shape[0];
    let c = compose_streams(
        static_modifier_flags.as_slice()?, h, village_entries.as_slice()?, rejected_slots.as_slice()?,
        unit_ints.as_slice()?, unit_stats.as_slice()?, recruit_type_ids.as_slice()?,
        recruit_stats.as_slice()?, leader_x, leader_y, globals, norms, map_limit, num_alignments,
    )?;
    let hex_arrays = (
        to_array2(py, h, NUM_HEX_MODIFIERS, c.modifier_flags),
        to_array2(py, h, NUM_HEX_DYNAMIC_FLAGS, c.dynamic_flags),
    );
    let unit_arrays = (
        c.is_ours.into_pyarray(py),
        c.type_ids.into_pyarray(py),
        c.side_ids.into_pyarray(py),
        c.xs.into_pyarray(py),
        c.ys.into_pyarray(py),
        to_array2(py, c.u, c.feat_dim, c.feats),
    );
    let recruit_arrays = (
        vec![1f32; c.r].into_pyarray(py),
        c.r_ids.into_pyarray(py),
        vec![0i64; c.r].into_pyarray(py),
        vec![c.lx; c.r].into_pyarray(py),
        vec![c.ly; c.r].into_pyarray(py),
        to_array2(py, c.r, c.feat_dim, c.r_feats),
    );
    let global_feats = c.global_feats.into_pyarray(py);
    Ok((hex_arrays, unit_arrays, recruit_arrays, global_feats))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn norms() -> Norms {
        Norms::from_array([80.0, 10.0, 150.0, 80.0, 500.0, 50.0, 30.0, 60.0])
    }

    #[test]
    fn clamp_matches_encoder_expression() {
        assert_eq!(clamp_pos(-3, 127), 0);
        assert_eq!(clamp_pos(0, 127), 0);
        assert_eq!(clamp_pos(127, 127), 127);
        assert_eq!(clamp_pos(128, 127), 127);
        assert_eq!(clamp_pos(45, 127), 45);
    }

    #[test]
    fn py_max_keeps_first_argument_unless_one_is_greater() {
        assert_eq!(py_max_1(0.0), 1.0);
        assert_eq!(py_max_1(1.0), 1.0);
        assert_eq!(py_max_1(33.0), 33.0);
        assert!(py_max_1(f64::NAN).is_nan());
    }

    #[test]
    fn onehot_index_wraps_like_a_python_list() {
        assert_eq!(onehot_index(2, 4).unwrap(), 2);
        assert_eq!(onehot_index(-1, 4).unwrap(), 3);
        assert!(onehot_index(4, 4).is_err());
        assert!(onehot_index(-5, 4).is_err());
    }

    #[test]
    fn unit_row_is_the_f64_expression_cast_once() {
        // Elvish Fighter-like: 33/80 hp, 17/33 current, 5 mp with 2 left,
        // 40 max xp with 7, cost 14, leader, attacked, lawful (1).
        let stats = [33.0, 17.0, 5.0, 2.0, 40.0, 7.0, 14.0];
        let mut out = [0f32; 13];
        unit_feature_row(&stats, 1, 1, 1, &norms(), 4, &mut out).unwrap();
        let expect: [f32; 13] = [
            (33.0f64 / 80.0) as f32,
            (17.0f64 / 33.0) as f32,
            (5.0f64 / 10.0) as f32,
            (2.0f64 / 5.0) as f32,
            (40.0f64 / 150.0) as f32,
            (7.0f64 / 40.0) as f32,
            (14.0f64 / 80.0) as f32,
            1.0,
            1.0,
            0.0,
            1.0,
            0.0,
            0.0,
        ];
        assert_eq!(out.map(f32::to_bits), expect.map(f32::to_bits));
        // A zero max is divided as 1 (max(0, 1)), not as 0.
        let zero_max = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        unit_feature_row(&zero_max, 0, 0, 0, &norms(), 4, &mut out).unwrap();
        assert_eq!(out[1], 0.0);
        assert_eq!(out[3], 0.0);
        assert_eq!(out[5], 0.0);
        assert_eq!(&out[9..], &[1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn recruit_row_is_a_fresh_phantom() {
        let stats = [38.0, 5.0, 43.0, 17.0, 2.0]; // chaotic
        let mut out = [0f32; 13];
        recruit_feature_row(&stats, &norms(), 4, &mut out).unwrap();
        assert_eq!(out[0].to_bits(), ((38.0f64 / 80.0) as f32).to_bits());
        assert_eq!(out[1], 1.0);
        assert_eq!(out[2].to_bits(), ((5.0f64 / 10.0) as f32).to_bits());
        assert_eq!(out[3], 0.0);
        assert_eq!(out[4].to_bits(), ((43.0f64 / 150.0) as f32).to_bits());
        assert_eq!(&out[5..9], &[0.0, (17.0f64 / 80.0) as f32, 0.0, 0.0]);
        assert_eq!(&out[9..], &[0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn global_row_maps_side_to_sign() {
        let g = global_feature_row(&[7.0, 1.0, 100.0, 2.0, 3.0, 4.0], &norms());
        assert_eq!(g[1], -1.0);
        let g2 = global_feature_row(&[7.0, 2.0, 100.0, 2.0, 3.0, 4.0], &norms());
        assert_eq!(g2[1], 1.0);
        assert_eq!(g[0].to_bits(), ((7.0f64 / 60.0) as f32).to_bits());
        assert_eq!(g[5].to_bits(), ((4.0f64 / 30.0) as f32).to_bits());
    }
}
