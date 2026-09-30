//! Phase 4: `encoder.encode_raw` over the core (docs/rust_port_plan.md):
//! the token basis (the full board in row-major order, or the relevant
//! subset in that order), the static hex facts in slot order, the
//! village entries as the mover sees them, the visible units in slot
//! order with their facts, the global values, then `compose_streams`
//! (encode.rs) for every RawEncoded array. Python keeps the vocab
//! (unit-type ids per registered type, faction ids), the recruit
//! options' ids and stats, and the Position objects. Byte-identity
//! with `encode_raw` on the Python state is the certificate
//! (tests/test_game_core.py).
//!
//! Under `observation_parity` the rows widen with the parity columns
//! (core_parity.rs) and a sighting stream is added; `relevant_set_version`
//! 2 widens the relevant set. That encoding is built here only.

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::core::GameCore;
use crate::core_observe::{observation_dict, CoreObservation};
use crate::core_parity::{
    recruit_extra, widen, ParityNorms, GLOBAL_EXTRA, HAS_ATTACKED_COL, HEX_EXTRA, NUM_PARITY_NORMS, SIGHT_FEAT_DIM,
    UNIT_EXTRA,
};
use crate::encode::{compose_streams, GLOBAL_FEAT_DIM, NUM_HEX_DYNAMIC_FLAGS, NUM_HEX_MODIFIERS, NUM_NORMS};

/// encoder.py position clamp: `0 if v < 0 else (LIMIT if v > LIMIT else v)`.
fn clamp_pos(v: i64, limit: i64) -> i64 {
    if v < 0 { 0 } else if v > limit { limit } else { v }
}

/// The other player's side (`wesnoth_ai.classes.opponent_of`). The
/// sides a scenario declares beyond the players (statues, a neutral
/// AI) are nobody's opponent, and a side that is not a player's has
/// none.
fn opponent_of(side: i64) -> PyResult<i64> {
    match side {
        1 => Ok(2),
        2 => Ok(1),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "side {side} is not a player side (1, 2)"))),
    }
}

/// The parity divisors, when the flag is on; the flag needs the terrain
/// set view and the divisors.
fn parity_norms_of(observation_parity: bool, terrain_multi_hot: bool,
                   parity_norms: Option<[f64; NUM_PARITY_NORMS]>) -> PyResult<Option<ParityNorms>> {
    if !observation_parity {
        return Ok(None);
    }
    if !terrain_multi_hot {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "observation_parity reads each hex's terrain as its set: pass terrain_multi_hot=True"));
    }
    match parity_norms {
        Some(a) => Ok(Some(ParityNorms::from_array(a))),
        None => Err(pyo3::exceptions::PyValueError::new_err("observation_parity needs parity_norms")),
    }
}

#[pymethods]
impl GameCore {
    /// Every array of `encoder.encode_raw` for `side`, as a dict:
    /// full_slots (the full-board slot of each token), hex_xs, hex_ys,
    /// hex_terrain_ids, hex_modifier_flags, hex_dynamic_flags;
    /// unit_ids, unit_raw_xs, unit_raw_ys (the visible units in slot
    /// order), unit_is_ours, unit_type_ids, unit_side_ids, unit_xs,
    /// unit_ys, unit_feats; recruit_is_ours, recruit_type_ids,
    /// recruit_side_ids, recruit_xs, recruit_ys, recruit_feats;
    /// global_feats, material, and the observation (`observe`) with
    /// tok_of_hex set to this basis.
    ///
    /// `type_vocab[t]` is the vocab id of registered type t (already
    /// clamped to the overflow bucket); the recruit ids and stats are
    /// the side's own recruit list in order (`encoder._recruit_rows`).
    /// With `terrain_multi_hot` the hex terrain stream carries each hex's
    /// terrain set as a bitmask (its one class's bit when the set is
    /// unresolved), as `encoder.encode_raw` does under the flag.
    ///
    /// With `observation_parity` (which needs `terrain_multi_hot` and
    /// `parity_norms`: damage, strikes, lawful bonus, leadership, village
    /// gold and village support divisors) the unit and recruit rows take
    /// the parity columns, the recruit rows their type's base values (the
    /// passed stats are not read), the hex rows the fog overlay and the
    /// hex's time of day, the global row the economy; the terrain set
    /// has its mushroom-grove and reef classes and every village hex its
    /// village bit; the recruit-rejected column is written 0; and
    /// sight_type_ids, sight_xs, sight_ys and sight_feats hold the
    /// sighting stream. `relevant_set_version` 2 (with `relevant_set`)
    /// widens the relevant set (`relevant_positions`).
    #[pyo3(signature = (side, relevant_set, type_vocab, recruit_type_ids, recruit_stats,
                        fog_hides_enemy_villages, norms, map_limit, num_alignments, terrain_multi_hot=false,
                        observation_parity=false, parity_norms=None, relevant_set_version=1))]
    #[allow(clippy::too_many_arguments)]
    fn encode_streams<'py>(&self, py: Python<'py>, side: i64, relevant_set: bool, type_vocab: Vec<i64>,
                           recruit_type_ids: Vec<i64>, recruit_stats: Vec<f64>, fog_hides_enemy_villages: bool,
                           norms: [f64; NUM_NORMS], map_limit: i64, num_alignments: usize,
                           terrain_multi_hot: bool, observation_parity: bool,
                           parity_norms: Option<[f64; NUM_PARITY_NORMS]>, relevant_set_version: i64)
        -> PyResult<Bound<'py, PyDict>> {
        let them_side = opponent_of(side)?;
        let parity = parity_norms_of(observation_parity, terrain_multi_hot, parity_norms)?;
        let obs = self.observe_relevant(side, relevant_set, relevant_set_version)?;
        let m = &self.map;
        let h = m.h;
        let fog_on = self.global.fog_on;
        let overflow = type_vocab.iter().copied().max().unwrap_or(0);

        // The token basis: map hex per token, full-board slot per token.
        let slots: Vec<usize> = if relevant_set {
            m.hex_of_slot.iter().copied().filter(|&mh| obs.relevant[mh] != 0).collect()
        } else {
            m.hex_of_slot.clone()
        };
        let ht = slots.len();
        let mut tok_of_hex = vec![-1i64; h];
        for (t, &mh) in slots.iter().enumerate() {
            tok_of_hex[mh] = t as i64;
        }
        let full_slots: Vec<i64> = slots.iter().map(|&mh| m.full_slot[mh]).collect();

        // Static hex facts in slot order (encoder._build_static_hex_arrays).
        // Under the parity flag every village hex carries its village bit
        // and the terrain set its mushroom-grove and reef classes.
        let mut static_flags = vec![0f32; ht * NUM_HEX_MODIFIERS];
        let mut hex_xs = vec![0i64; ht];
        let mut hex_ys = vec![0i64; ht];
        let mut hex_tids = vec![0i64; ht];
        for (t, &mh) in slots.iter().enumerate() {
            static_flags[t * NUM_HEX_MODIFIERS + 1] = if m.keep[mh] != 0 { 1.0 } else { 0.0 };
            static_flags[t * NUM_HEX_MODIFIERS + 2] = if m.castle_mod[mh] != 0 { 1.0 } else { 0.0 };
            hex_xs[t] = clamp_pos(m.hx[mh], map_limit);
            hex_ys[t] = clamp_pos(m.hy[mh], map_limit);
            hex_tids[t] = if parity.is_some() && m.terrain_mask_parity[mh] != 0 {
                m.terrain_mask_parity[mh]
            } else if terrain_multi_hot && m.terrain_mask[mh] != 0 {
                m.terrain_mask[mh]
            } else if terrain_multi_hot {
                1 << m.terrain_type_id[mh]
            } else {
                m.terrain_type_id[mh]
            };
            if parity.is_some() && (m.village_terrain[mh] != 0 || m.village_mod[mh] != 0) {
                static_flags[t * NUM_HEX_MODIFIERS] = 1.0;
            }
        }

        // Village entries (encoder._village_entries): the hexes with the
        // village modifier and the owned hexes; owner code and the fog
        // gate on the owner.
        let mut entries: Vec<i64> = Vec::new();
        for (t, &mh) in slots.iter().enumerate() {
            let owner = self.village_owner[mh];
            if m.village_mod[mh] == 0 && owner == 0 {
                continue;
            }
            let ours = owner == side;
            let visible = ours || !fog_on || obs.view.seen[mh] != 0;
            let code = if ours { 1 } else if owner != 0 { 2 } else { 0 };
            entries.extend([t as i64, code, visible as i64]);
        }
        // Under the parity flag the recruit-rejected column is written 0:
        // a replay records only where a recruit landed, so corpus
        // reconstruction never sets it and its weights would never train.
        // The legality mask still reads the rejections (principle 6).
        let rejected: Vec<i64> = if parity.is_some() {
            Vec::new()
        } else {
            (0..h)
                .filter(|&mh| self.recruit_rejected[mh] != 0 && tok_of_hex[mh] >= 0)
                .map(|mh| tok_of_hex[mh])
                .collect()
        };

        // The visible units in slot order (visibility.visible_units_in_slot_order).
        let n = self.units.len();
        let mut vis: Vec<usize> = (0..n).filter(|&i| obs.view.visible[i] != 0).collect();
        vis.sort_by(|&a, &b| {
            let (ua, ub) = (&self.units[a], &self.units[b]);
            (ua.y, ua.x, &ua.id).cmp(&(ub.y, ub.x, &ub.id))
        });
        let mut unit_ints: Vec<i64> = Vec::with_capacity(vis.len() * 7);
        let mut unit_stats: Vec<f64> = Vec::with_capacity(vis.len() * 7);
        let mut unit_ids: Vec<String> = Vec::with_capacity(vis.len());
        let mut raw_xs: Vec<i64> = Vec::with_capacity(vis.len());
        let mut raw_ys: Vec<i64> = Vec::with_capacity(vis.len());
        let mut material = 0.0f64;
        for &i in &vis {
            let u = &self.units[i];
            let vocab = if u.type_idx >= 0 && (u.type_idx as usize) < type_vocab.len() {
                type_vocab[u.type_idx as usize]
            } else {
                overflow
            };
            let side_code = if self.is_scenery(i) { 2 } else if u.side == side { 0 } else { 1 };
            unit_ints.extend([vocab, side_code, u.x, u.y, u.alignment, u.is_leader as i64, u.has_attacked as i64]);
            unit_stats.extend([u.max_hp as f64, u.current_hp as f64, u.max_moves as f64, u.current_moves as f64,
                               u.max_exp as f64, u.current_exp as f64, u.cost as f64]);
            unit_ids.push(u.id.clone());
            raw_xs.push(u.x);
            raw_ys.push(u.y);
            // material.material_of_units over the same list, the same order
            if (u.side == 1 || u.side == 2) && u.max_hp > 0 {
                let v = u.cost as f64 * u.current_hp as f64 / u.max_hp as f64;
                material += if u.side == side { v } else { -v };
            }
        }
        // The mover's leader (the first in unit order) sites the recruit phantoms.
        let (mut lx, mut ly) = (0i64, 0i64);
        if let Some(l) = self.units.iter().find(|u| u.is_leader && u.side == side) {
            lx = l.x;
            ly = l.y;
        }
        // Global values: turn, side, gold, income, our and their
        // villages (the other player's count, or under the fog gate the
        // enemy villages among the hexes it sees). The other player is
        // found by its side number: a replayed game lists every side
        // its scenario declares, statues and tentacles included.
        let ns = self.sides.len();
        let us = side - 1;
        let them = them_side - 1;
        let side_ok = |k: i64| k >= 0 && (k as usize) < ns;
        let our_gold = if side_ok(us) { self.sides[us as usize].current_gold } else { 0 };
        let our_income = if side_ok(us) { self.sides[us as usize].base_income } else { 0 };
        let our_villages = if side_ok(us) { self.sides[us as usize].nb_villages } else { 0 };
        let mut their_villages = if side_ok(them) { self.sides[them as usize].nb_villages } else { 0 };
        if fog_hides_enemy_villages && fog_on {
            their_villages = (0..h)
                .filter(|&mh| self.village_owner[mh] != 0 && self.village_owner[mh] != side && obs.view.seen[mh] != 0)
                .count() as i64;
        }
        // Slots 6 and 7: this turn's board-level lawful bonus and the
        // next turn's. `lawful_bonus_at(-1, ..)` is the board cycle --
        // a negative hex skips the time-area and lit-terrain branches --
        // and it honours tod_start_offset, so a random-start scenario
        // reads the phase it actually drew. Mirrors encoder.py's
        // `_lawful_bonus_for_turn`.
        let turn = self.global.turn_number;
        let globals: [f64; GLOBAL_FEAT_DIM] = [
            turn as f64, side as f64, our_gold as f64, our_income as f64,
            our_villages as f64, their_villages as f64,
            self.lawful_bonus_at(-1, turn) as f64,
            self.lawful_bonus_at(-1, turn + 1) as f64,
        ];
        // Under the parity flag the recruit rows take their type's base
        // values from the core's unit table, the side's recruit list order.
        let parity_recruits: Vec<f64> = if parity.is_some() {
            self.parity_recruit_names(side, recruit_type_ids.len())?
                .iter().flat_map(|name| self.recruit_base_stats(name)).collect()
        } else {
            Vec::new()
        };
        let recruit_stats: &[f64] = if parity.is_some() { &parity_recruits } else { &recruit_stats };
        let c = compose_streams(&static_flags, ht, &entries, &rejected, &unit_ints, &unit_stats,
                                &recruit_type_ids, recruit_stats, lx, ly, globals, norms, map_limit,
                                num_alignments)?;
        let feat_dim = c.feat_dim;
        let to2 = |rows: usize, cols: usize, data: Vec<f32>| -> Bound<'py, PyArray2<f32>> {
            Array2::from_shape_vec((rows, cols), data).expect("row-major buffer sized rows*cols").into_pyarray(py)
        };
        let d = PyDict::new(py);
        d.set_item("full_slots", full_slots.into_pyarray(py))?;
        d.set_item("hex_xs", hex_xs.into_pyarray(py))?;
        d.set_item("hex_ys", hex_ys.into_pyarray(py))?;
        d.set_item("hex_terrain_ids", hex_tids.into_pyarray(py))?;
        d.set_item("hex_modifier_flags", to2(ht, NUM_HEX_MODIFIERS, c.modifier_flags))?;
        d.set_item("unit_ids", unit_ids)?;
        d.set_item("unit_raw_xs", raw_xs)?;
        d.set_item("unit_raw_ys", raw_ys)?;
        d.set_item("unit_is_ours", c.is_ours.into_pyarray(py))?;
        d.set_item("unit_type_ids", c.type_ids.into_pyarray(py))?;
        d.set_item("unit_side_ids", c.side_ids.into_pyarray(py))?;
        d.set_item("unit_xs", c.xs.into_pyarray(py))?;
        d.set_item("unit_ys", c.ys.into_pyarray(py))?;
        d.set_item("recruit_is_ours", vec![1f32; c.r].into_pyarray(py))?;
        d.set_item("recruit_type_ids", c.r_ids.into_pyarray(py))?;
        d.set_item("recruit_side_ids", vec![0i64; c.r].into_pyarray(py))?;
        d.set_item("recruit_xs", vec![c.lx; c.r].into_pyarray(py))?;
        d.set_item("recruit_ys", vec![c.ly; c.r].into_pyarray(py))?;
        d.set_item("material", material)?;
        match &parity {
            None => {
                d.set_item("hex_dynamic_flags", to2(ht, NUM_HEX_DYNAMIC_FLAGS, c.dynamic_flags))?;
                d.set_item("unit_feats", to2(c.u, feat_dim, c.feats))?;
                d.set_item("recruit_feats", to2(c.r, feat_dim, c.r_feats))?;
                d.set_item("global_feats", c.global_feats.into_pyarray(py))?;
            }
            Some(pn) => {
                let hex_extra = self.hex_extra(&slots, &obs.view.seen, pn);
                d.set_item("hex_dynamic_flags", to2(ht, NUM_HEX_DYNAMIC_FLAGS + HEX_EXTRA,
                    widen(&c.dynamic_flags, NUM_HEX_DYNAMIC_FLAGS, &hex_extra, HEX_EXTRA)))?;
                let mut unit_extra = vec![0f32; vis.len() * UNIT_EXTRA];
                for (k, &i) in vis.iter().enumerate() {
                    self.unit_extra(i, pn, &mut unit_extra[k * UNIT_EXTRA..(k + 1) * UNIT_EXTRA]);
                }
                d.set_item("unit_feats", to2(c.u, feat_dim + UNIT_EXTRA,
                    widen(&c.feats, feat_dim, &unit_extra, UNIT_EXTRA)))?;
                d.set_item("recruit_feats", to2(c.r, feat_dim + UNIT_EXTRA,
                    self.parity_recruit_feats(side, &c.r_feats, feat_dim, pn)?))?;
                let mut global = c.global_feats;
                global.extend(self.global_extra(side, them_side, norms[4], norms[5], pn));
                debug_assert_eq!(global.len(), GLOBAL_FEAT_DIM + GLOBAL_EXTRA);
                d.set_item("global_feats", global.into_pyarray(py))?;
                let (sight_ids, sight_xs, sight_ys, sight_feats) =
                    self.sighting_stream(side, &obs.view.visible, &type_vocab, map_limit, norms[0]);
                let s = sight_ids.len();
                d.set_item("sight_type_ids", sight_ids.into_pyarray(py))?;
                d.set_item("sight_xs", sight_xs.into_pyarray(py))?;
                d.set_item("sight_ys", sight_ys.into_pyarray(py))?;
                d.set_item("sight_feats", to2(s, SIGHT_FEAT_DIM, sight_feats))?;
            }
        }
        let od = observation_dict(py, self, obs, side)?;
        od.set_item("tok_of_hex", tok_of_hex.into_pyarray(py))?;
        d.set_item("observation", od)?;
        Ok(d)
    }
}

#[pymethods]
impl GameCore {
    /// The relevant hex set of `side` as (x, y) in token order (row-major):
    /// version 1 obs8's, version 2 the parity recipe's (`widen_relevant`).
    /// The label builder's target slots under version 2
    /// (`replay_dataset._action_indices`).
    fn relevant_positions(&self, side: i64, version: i64) -> PyResult<Vec<(i64, i64)>> {
        let obs = self.observe_relevant(side, true, version)?;
        let m = &self.map;
        Ok(m.hex_of_slot.iter().filter(|&&mh| obs.relevant[mh] != 0).map(|&mh| (m.hx[mh], m.hy[mh])).collect())
    }
}

impl GameCore {
    /// `observe_core` with the relevant set of `version` (1, obs8's; 2,
    /// the parity recipe's, which needs `relevant_set`).
    fn observe_relevant(&self, side: i64, relevant_set: bool, version: i64) -> PyResult<CoreObservation> {
        match (version, relevant_set) {
            (1, _) | (2, true) => {}
            (2, false) => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "relevant_set_version 2 widens the relevant set: pass relevant_set=True"));
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "relevant_set_version {version}: 1 or 2")));
            }
        }
        let mut obs = self.observe_core(side, relevant_set)?;
        if version == 2 {
            self.widen_relevant(side, &mut obs.relevant, &obs.view.seen);
        }
        Ok(obs)
    }

    /// The side's recruit list, which the passed recruit ids follow.
    fn parity_recruit_names(&self, side: i64, n_ids: usize) -> PyResult<Vec<String>> {
        let names = if side >= 1 && side as usize <= self.sides.len() {
            self.sides[side as usize - 1].recruits.clone()
        } else {
            Vec::new()
        };
        if names.len() != n_ids {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "{n_ids} recruit ids for the {} recruits of side {side}", names.len())));
        }
        Ok(names)
    }

    /// The recruit rows under the parity flag: the composed 13 columns
    /// with has_attacked set (a recruit has no attack left on the turn it
    /// arrives, docs/wesnoth_rules.md "Recruit place_recruit zeroes MP
    /// and attacks"), then the type's parity columns.
    fn parity_recruit_feats(&self, side: i64, base: &[f32], feat_dim: usize, pn: &ParityNorms)
        -> PyResult<Vec<f32>> {
        let names = self.parity_recruit_names(side, if feat_dim == 0 { 0 } else { base.len() / feat_dim })?;
        let mut base = base.to_vec();
        let mut extra = vec![0f32; names.len() * UNIT_EXTRA];
        for (k, name) in names.iter().enumerate() {
            base[k * feat_dim + HAS_ATTACKED_COL] = 1.0;
            recruit_extra(&self.db.get(name), pn, &mut extra[k * UNIT_EXTRA..(k + 1) * UNIT_EXTRA]);
        }
        Ok(widen(&base, feat_dim, &extra, UNIT_EXTRA))
    }
}
