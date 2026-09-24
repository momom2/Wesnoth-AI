//! Phase 4: what the simulator (tools/wesnoth_sim.py) asks the core
//! between commands: a recruit rejection,
//! the progress fingerprint of the no-progress tracker, the sides
//! with a leader, the structural invariants, the advancement salt.

use pyo3::prelude::*;

use crate::core::{GameCore, Hasher};

#[pymethods]
impl GameCore {
    /// A recruit attempt bounced on (x, y) this turn.
    fn add_recruit_rejected(&mut self, x: i64, y: i64) {
        if let Some(&i) = self.map.pos_index.get(&(x, y)) {
            self.recruit_rejected[i] = 1;
        }
    }

    /// `WesnothSim.step`'s progress fingerprint: (unit count, hit
    /// points in play, a hash of the village owners).
    fn progress_fingerprint(&self) -> (i64, i64, i64) {
        let mut hs = Hasher::default();
        for i in 0..self.map.h {
            if self.village_owner[i] != 0 {
                hs.add_i(i as i64);
                hs.add_i(self.village_owner[i]);
            }
        }
        (self.units.len() as i64, self.units.iter().map(|u| u.current_hp).sum(), hs.value())
    }

    /// The sides that still have a leader, sorted.
    fn leader_sides(&self) -> Vec<i64> {
        let mut out: Vec<i64> = self.units.iter().filter(|u| u.is_leader).map(|u| u.side).collect();
        out.sort();
        out.dedup();
        out
    }

    /// `WesnothSim._assert_invariants`: hit points and movement within
    /// range, one unit per hex, at most one leader per side; the first
    /// violation as text, or None.
    fn invariant_violation(&self) -> Option<String> {
        let mut seen = vec![-1i64; self.map.h];
        let mut leaders: std::collections::HashMap<i64, Vec<String>> = std::collections::HashMap::new();
        for (i, u) in self.units.iter().enumerate() {
            if u.current_hp < 0 || u.current_hp > u.max_hp {
                return Some(format!("unit {} ({:?}) HP out of range: current_hp={}, max_hp={}",
                                    u.id, u.name, u.current_hp, u.max_hp));
            }
            if u.current_moves < 0 || u.current_moves > u.max_moves {
                return Some(format!("unit {} ({:?}) MP out of range: current_moves={}, max_moves={}",
                                    u.id, u.name, u.current_moves, u.max_moves));
            }
            if u.hex >= 0 {
                let h = u.hex as usize;
                if seen[h] >= 0 {
                    return Some(format!("hex ({}, {}) occupied by both {:?} and {:?}",
                                        u.x, u.y, self.units[seen[h] as usize].id, u.id));
                }
                seen[h] = i as i64;
            }
            if u.is_leader {
                leaders.entry(u.side).or_default().push(u.id.clone());
            }
        }
        let mut sides: Vec<&i64> = leaders.keys().collect();
        sides.sort();
        for side in sides {
            let l = &leaders[side];
            if l.len() > 1 {
                return Some(format!("side {} has {} leaders ({:?}); at most one allowed", side, l.len(), l));
            }
        }
        None
    }

    /// The recorder consumed the last attack's advancement events.
    fn clear_last_advance_events(&mut self) {
        self.last_advance_events.clear();
    }

    /// The advancement channel's salt (`_advance_salt`, a string).
    fn set_advance_salt(&mut self, salt: String) {
        self.global.advance_salt = salt;
    }

    /// The id of the first unit standing on (x, y), of `side` when
    /// given (0 = any side), or None.
    #[pyo3(signature = (x, y, side=0))]
    fn unit_id_at(&self, x: i64, y: i64, side: i64) -> Option<String> {
        self.units.iter().find(|u| u.x == x && u.y == y && (side == 0 || u.side == side)).map(|u| u.id.clone())
    }
}
