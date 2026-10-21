//! Phase 32: the decision's legality masks, packed as the inference
//! server reads them (`server_priors.PackedMasks`), computed beside the
//! encoding from the same observation: what
//! `action_sampler._build_legality_masks` builds and
//! `server_priors.pack_masks` packs.
//! Equality with the Python builder on played positions is the
//! certificate (tests/test_core_masks.py); the Python builder stays the
//! oracle and serves the unpacked masks.
//!
//! Per unit slot (the encoding's visible units, in slot order): its move
//! and attack rows in token space (`rows_from_landable` over the acting
//! units' landable rows, the attackable enemies being the token hexes a
//! seen, attackable, non-allied unit occupies); type ATTACK legal when
//! its attack row has a hex, MOVE when its move row has one, the slot
//! legal when either does. Per recruit slot: the recruit row (the
//! leader's castle network, in token space) when the leader stands on
//! its keep, the row has a hex and the side's gold covers the cost. The
//! end_turn slot is always legal. A slot acts when it is legal and the
//! side's own (`unit_is_ours`, every recruit slot); `n_attacks` counts
//! every unit slot's weapons up to MAX_ATTACKS.

use crate::core::GameCore;
use crate::core_observe::CoreObservation;
use crate::rows_from_landable;
use pyo3::prelude::*;

/// `model.UnitActionType`: the unit actor's action types and their columns.
pub(crate) const UNIT_ACTION_TYPES: usize = 2;
pub(crate) const TYPE_ATTACK: usize = 0;
pub(crate) const TYPE_MOVE: usize = 1;
/// `model.MAX_ATTACKS`: the weapon slots of a unit actor.
pub(crate) const MAX_ATTACKS: usize = 4;

/// One decision's packed masks over A = units + recruits + 1 slots and
/// HT token hexes, rows of ceil(HT / 8) bytes, first hex in the high bit
/// (numpy's `packbits` order).
pub(crate) struct LeafMasks {
    pub actor_mask: Vec<u8>,
    pub type_valid: Vec<u8>,
    pub attack_bits: Vec<u8>,
    pub move_bits: Vec<u8>,
    pub union_bits: Vec<u8>,
    pub n_attacks: Vec<i8>,
    pub bytes_per_row: usize,
}

/// Set token `t`'s bit in a packed row.
fn set_bit(row: &mut [u8], t: usize) {
    row[t / 8] |= 0x80 >> (t % 8);
}

impl GameCore {
    /// The packed masks of `side`'s decision over the encoding's token
    /// basis: `slots` the map hex of each token, `tok_of_hex` its inverse
    /// (-1 off the basis), `vis` the core index of each unit slot,
    /// `unit_is_ours` the slots' ownership, `recruit_costs` the gold of
    /// each of the side's recruit options in its recruit-list order.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn leaf_masks(&self, side: i64, obs: &CoreObservation, slots: &[usize], tok_of_hex: &[i64],
                             vis: &[usize], unit_is_ours: &[f32], recruit_costs: &[i64]) -> PyResult<LeafMasks> {
        let h = self.map.h;
        let ht = slots.len();
        let nu = vis.len();
        let nr = recruit_costs.len();
        let a = nu + nr + 1;
        let hb = ht.div_ceil(8);
        let mut out = LeafMasks {
            actor_mask: vec![0u8; a],
            type_valid: vec![0u8; a * UNIT_ACTION_TYPES],
            attack_bits: vec![0u8; a * hb],
            move_bits: vec![0u8; a * hb],
            union_bits: vec![0u8; a * hb],
            n_attacks: vec![0i8; a],
            bytes_per_row: hb,
        };
        out.actor_mask[a - 1] = 1;
        for (s, &i) in vis.iter().enumerate() {
            out.n_attacks[s] = self.units[i].attacks.len().min(MAX_ATTACKS) as i8;
        }
        if ht == 0 {
            return Ok(out);
        }
        let v = &obs.view;
        // The token hexes a seen, attackable enemy occupies.
        let enemy_hexids: Vec<i64> = slots.iter()
            .filter(|&&mh| v.occupied[mh] != 0 && v.inert[mh] == 0 && v.ally[mh] == 0)
            .map(|&mh| mh as i64)
            .collect();
        // The acting units' landable rows, by unit slot.
        let mut unit_hexidx = vec![-1i64; nu];
        let mut can_move = vec![0u8; nu];
        let mut can_attack = vec![0u8; nu];
        let mut landable = vec![0u8; nu * h];
        if obs.relevant.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "leaf_masks needs the observation with the acting units' reach"));
        }
        for (s, &i) in vis.iter().enumerate() {
            if obs.acting[i] == 0 {
                continue;
            }
            unit_hexidx[s] = obs.unit_hex[i];
            can_move[s] = obs.can_move[i];
            can_attack[s] = obs.can_attack[i];
            landable[s * h..(s + 1) * h].copy_from_slice(&obs.landable[i * h..(i + 1) * h]);
        }
        let (move_rows, attack_rows) = rows_from_landable(&landable, &self.map.nbrs, tok_of_hex, &unit_hexidx,
                                                          &can_move, &can_attack, &enemy_hexids, ht)?;
        for s in 0..nu {
            let row = s * hb..(s + 1) * hb;
            let (mut any_attack, mut any_move) = (false, false);
            for t in 0..ht {
                let at = attack_rows[s * ht + t] != 0;
                let mv = move_rows[s * ht + t] != 0;
                if at {
                    set_bit(&mut out.attack_bits[row.clone()], t);
                    any_attack = true;
                }
                if mv {
                    set_bit(&mut out.move_bits[row.clone()], t);
                    any_move = true;
                }
                if at || mv {
                    set_bit(&mut out.union_bits[row.clone()], t);
                }
            }
            out.type_valid[s * UNIT_ACTION_TYPES + TYPE_ATTACK] = any_attack as u8;
            out.type_valid[s * UNIT_ACTION_TYPES + TYPE_MOVE] = any_move as u8;
            out.actor_mask[s] = ((any_attack || any_move) && unit_is_ours[s] != 0.0) as u8;
        }
        // The recruit slots: the castle network in token space.
        if nr > 0 && v.leader_on_keep {
            let recruit_tokens: Vec<usize> = (0..h)
                .filter(|&mh| v.recruit_row[mh] != 0 && tok_of_hex[mh] >= 0)
                .map(|mh| tok_of_hex[mh] as usize)
                .collect();
            if !recruit_tokens.is_empty() {
                let us = side - 1;
                let gold = if us >= 0 && (us as usize) < self.sides.len() { self.sides[us as usize].current_gold } else { 0 };
                for (r, &cost) in recruit_costs.iter().enumerate() {
                    if cost > gold {
                        continue;
                    }
                    let slot = nu + r;
                    for &t in &recruit_tokens {
                        set_bit(&mut out.union_bits[slot * hb..(slot + 1) * hb], t);
                    }
                    out.actor_mask[slot] = 1;
                }
            }
        }
        Ok(out)
    }
}
