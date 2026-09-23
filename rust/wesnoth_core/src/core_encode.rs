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

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::core::GameCore;
use crate::core_observe::observation_dict;
use crate::encode::{compose_streams, GLOBAL_FEAT_DIM, NUM_HEX_DYNAMIC_FLAGS, NUM_HEX_MODIFIERS, NUM_NORMS};

/// encoder.py position clamp: `0 if v < 0 else (LIMIT if v > LIMIT else v)`.
fn clamp_pos(v: i64, limit: i64) -> i64 {
    if v < 0 { 0 } else if v > limit { limit } else { v }
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
    #[pyo3(signature = (side, relevant_set, type_vocab, recruit_type_ids, recruit_stats,
                        fog_hides_enemy_villages, norms, map_limit, num_alignments))]
    #[allow(clippy::too_many_arguments)]
    fn encode_streams<'py>(&self, py: Python<'py>, side: i64, relevant_set: bool, type_vocab: Vec<i64>,
                           recruit_type_ids: Vec<i64>, recruit_stats: Vec<f64>, fog_hides_enemy_villages: bool,
                           norms: [f64; NUM_NORMS], map_limit: i64, num_alignments: usize)
        -> PyResult<Bound<'py, PyDict>> {
        let obs = self.observe_core(side, relevant_set)?;
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
        let mut static_flags = vec![0f32; ht * NUM_HEX_MODIFIERS];
        let mut hex_xs = vec![0i64; ht];
        let mut hex_ys = vec![0i64; ht];
        let mut hex_tids = vec![0i64; ht];
        for (t, &mh) in slots.iter().enumerate() {
            static_flags[t * NUM_HEX_MODIFIERS + 1] = if m.keep[mh] != 0 { 1.0 } else { 0.0 };
            static_flags[t * NUM_HEX_MODIFIERS + 2] = if m.castle_mod[mh] != 0 { 1.0 } else { 0.0 };
            hex_xs[t] = clamp_pos(m.hx[mh], map_limit);
            hex_ys[t] = clamp_pos(m.hy[mh], map_limit);
            hex_tids[t] = m.terrain_type_id[mh];
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
            let visible = ours || !fog_on || obs.view.disc[mh] != 0;
            let code = if ours { 1 } else if owner != 0 { 2 } else { 0 };
            entries.extend([t as i64, code, visible as i64]);
        }
        let rejected: Vec<i64> = (0..h)
            .filter(|&mh| self.recruit_rejected[mh] != 0 && tok_of_hex[mh] >= 0)
            .map(|mh| tok_of_hex[mh])
            .collect();

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
        // villages (the other side's count, or under the fog gate the
        // enemy villages inside the vision disc).
        let ns = self.sides.len();
        let us = side - 1;
        let them = if ns == 2 { 1 - us } else { us };
        let side_ok = |k: i64| k >= 0 && (k as usize) < ns;
        let our_gold = if side_ok(us) { self.sides[us as usize].current_gold } else { 0 };
        let our_income = if side_ok(us) { self.sides[us as usize].base_income } else { 0 };
        let our_villages = if side_ok(us) { self.sides[us as usize].nb_villages } else { 0 };
        let mut their_villages = if side_ok(them) { self.sides[them as usize].nb_villages } else { 0 };
        if fog_hides_enemy_villages && fog_on {
            their_villages = (0..h)
                .filter(|&mh| self.village_owner[mh] != 0 && self.village_owner[mh] != side && obs.view.disc[mh] != 0)
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
        let c = compose_streams(&static_flags, ht, &entries, &rejected, &unit_ints, &unit_stats,
                                &recruit_type_ids, &recruit_stats, lx, ly, globals, norms, map_limit,
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
        d.set_item("hex_dynamic_flags", to2(ht, NUM_HEX_DYNAMIC_FLAGS, c.dynamic_flags))?;
        d.set_item("unit_ids", unit_ids)?;
        d.set_item("unit_raw_xs", raw_xs)?;
        d.set_item("unit_raw_ys", raw_ys)?;
        d.set_item("unit_is_ours", c.is_ours.into_pyarray(py))?;
        d.set_item("unit_type_ids", c.type_ids.into_pyarray(py))?;
        d.set_item("unit_side_ids", c.side_ids.into_pyarray(py))?;
        d.set_item("unit_xs", c.xs.into_pyarray(py))?;
        d.set_item("unit_ys", c.ys.into_pyarray(py))?;
        d.set_item("unit_feats", to2(c.u, feat_dim, c.feats))?;
        d.set_item("recruit_is_ours", vec![1f32; c.r].into_pyarray(py))?;
        d.set_item("recruit_type_ids", c.r_ids.into_pyarray(py))?;
        d.set_item("recruit_side_ids", vec![0i64; c.r].into_pyarray(py))?;
        d.set_item("recruit_xs", vec![c.lx; c.r].into_pyarray(py))?;
        d.set_item("recruit_ys", vec![c.ly; c.r].into_pyarray(py))?;
        d.set_item("recruit_feats", to2(c.r, feat_dim, c.r_feats))?;
        d.set_item("global_feats", c.global_feats.into_pyarray(py))?;
        d.set_item("material", material)?;
        let od = observation_dict(py, self, obs, side)?;
        od.set_item("tok_of_hex", tok_of_hex.into_pyarray(py))?;
        d.set_item("observation", od)?;
        Ok(d)
    }
}
