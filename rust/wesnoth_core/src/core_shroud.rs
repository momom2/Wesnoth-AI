//! Delayed shroud updates (docs/wesnoth_rules.md "Delayed shroud
//! updates"). A side whose player turned "delay shroud updates" on (the
//! `[auto_shroud] active=no` command) clears no fog when its units move
//! or are recruited during its turn: each such action waits on the undo
//! stack with the hexes the unit occupied and its vision, and the fog is
//! cleared from all of them at the next commit (an attack, a recruit that
//! drew random numbers, a move that was ambushed or blocked,
//! `[update_shroud]`, `[auto_shroud] active=yes`, the end of the turn).
//! An advancement on the side's own turn clears nothing. In a game with
//! the Plan Unit Advance modification, its handlers make two more actions
//! final: the first move of each side turn (the `moveto` handler calls
//! `wesnoth.allow_undo(false)`, data/modifications/pick_advance/main.lua)
//! and each Plan Advancement menu event (a WML command, which runs with
//! undo disabled). `wesnoth_ai.delayed_shroud` is the oracle.

use pyo3::prelude::*;

use crate::core::GameCore;
use crate::effects::warn_once;

/// A move or recruit whose fog clearing waits for the next commit
/// (`shroud_clearing_action`, src/actions/shroud_clearing_action.hpp at
/// 1.18.4): the hexes the unit occupied during the action, and its
/// vision points and slowed status then (`clearer_info`,
/// src/actions/vision.cpp:100-106).
#[derive(Clone, Debug, PartialEq)]
pub struct PendingVision {
    pub side: i64,
    pub route: Vec<(i64, i64)>,
    pub unit_id: String,
    pub vision: i64,
    pub slowed: bool,
}

/// A pending entry as it crosses to Python: (side, route, unit id,
/// vision points, slowed).
type PendingRow = (i64, Vec<(i64, i64)>, String, i64, bool);
/// The shroud state as it crosses to Python: the delaying sides, the
/// pending vision, whether the Plan Unit Advance modification is on, and
/// whether the current side turn has had its first move yet.
type ShroudState = (Vec<i64>, Vec<PendingRow>, bool, bool);

impl GameCore {
    /// Whether `side`'s fog clearing waits for a commit: fog is on, it is
    /// the side's turn and the side delays its shroud updates
    /// (`current_uses_fog_`, src/actions/move.cpp:371; the recruit,
    /// create.cpp:697; an advancement, vision.cpp:467-469).
    pub fn vision_delayed(&self, side: i64) -> bool {
        self.global.fog_on && side == self.global.current_side && self.shroud_delayed.contains(&side)
    }

    /// Unit `i`'s action over the map hexes `route` waits for the commit
    /// (`undo_list::add_move` / `add_recruit`).
    pub fn defer_vision(&mut self, i: usize, route: &[usize]) {
        let u = &self.units[i];
        let route = route.iter().map(|&j| (self.map.hx[j], self.map.hy[j])).collect();
        let entry = PendingVision {
            side: u.side,
            route,
            unit_id: u.id.clone(),
            vision: u.max_moves,
            slowed: u.has_status("slowed"),
        };
        self.pending_vision.push(entry);
    }

    /// `undo_list::apply_shroud_changes` (src/actions/undo.cpp:431-470):
    /// the fog cleared from every hex of every pending route with the
    /// vision recorded, when the current side still delays its updates.
    /// Returns whether a hex was cleared.
    fn apply_pending_vision(&mut self) -> bool {
        let side = self.global.current_side;
        if !self.global.fog_on || !self.shroud_delayed.contains(&side) || self.pending_vision.is_empty() {
            return false;
        }
        let mut cleared = false;
        let pending = std::mem::take(&mut self.pending_vision);
        for p in &pending {
            let Some(&i) = self.unit_index.get(&p.unit_id) else {
                warn_once(format!("delayed shroud: unit {} left the board before its vision was \
                                   committed; its pending vision is dropped", p.unit_id));
                continue;
            };
            let before = self.seen_by(p.side);
            let mut v = before.clone();
            for xy in &p.route {
                if let Some(&hex) = self.map.pos_index.get(xy) {
                    self.mark_vision_as(i, p.slowed, p.vision, hex, &mut v);
                }
            }
            if v != before {
                cleared = true;
                self.set_cleared(p.side, v);
            }
        }
        self.pending_vision = pending;
        cleared
    }

    /// `undo_list::clear` (undo.cpp:201-215): an action that cannot be
    /// undone commits the pending vision and empties the stack.
    pub fn clear_undo_stack(&mut self) {
        self.apply_pending_vision();
        self.pending_vision.clear();
    }

    /// After a move: in a game with the Plan Unit Advance modification the
    /// first move of each side turn is made final by the modification's
    /// `moveto` handler (move.cpp:1059 fires it before :1070-1079 read
    /// undo_blocked), which commits the stack, this move's entry included.
    pub fn after_move(&mut self) {
        if self.pa_fresh_turn {
            self.pa_fresh_turn = false;
            self.clear_undo_stack();
        }
    }

    /// `undo_list::commit_vision` (undo.cpp:222-236): the pending vision
    /// committed; the stack empties when something was cleared.
    fn commit_vision(&mut self) -> bool {
        let cleared = self.apply_pending_vision();
        if cleared {
            self.pending_vision.clear();
        }
        cleared
    }
}

#[pymethods]
impl GameCore {
    /// `_apply_command(["auto_shroud", active])`, the synced command
    /// (src/synced_commands.cpp:367-381): turning the updates back on
    /// commits the pending vision first.
    fn apply_auto_shroud(&mut self, active: bool) {
        let side = self.global.current_side;
        let delayed = self.shroud_delayed.contains(&side);
        if active && delayed {
            self.commit_vision();
        }
        if active {
            self.shroud_delayed.retain(|&s| s != side);
        } else if !delayed {
            self.shroud_delayed.push(side);
            self.shroud_delayed.sort_unstable();
        }
        self.note_sightings();
    }

    /// `_apply_command(["update_shroud"])`, the synced command
    /// (synced_commands.cpp:383-398): the pending vision committed.
    fn apply_update_shroud(&mut self) {
        self.commit_vision();
        self.note_sightings();
    }

    /// `_apply_command(["menu_item", id])`: a menu item's event, whose WML
    /// command runs with undo disabled, so the `[fire_event]` synced
    /// command clears the stack (synced_commands.cpp:337-345).
    fn apply_menu_item(&mut self, _id: &str) {
        self.clear_undo_stack();
        self.note_sightings();
    }

    /// The shroud state (`ShroudState`).
    fn shroud_state_export(&self) -> ShroudState {
        let rows = self.pending_vision.iter()
            .map(|p| (p.side, p.route.clone(), p.unit_id.clone(), p.vision, p.slowed))
            .collect();
        (self.shroud_delayed.clone(), rows, self.plan_unit_advance, self.pa_fresh_turn)
    }

    /// Replace the shroud state (a core built from a view).
    fn set_shroud_state(&mut self, delayed: Vec<i64>, pending: Vec<PendingRow>, plan_unit_advance: bool,
                        pa_fresh_turn: bool) {
        self.plan_unit_advance = plan_unit_advance;
        self.pa_fresh_turn = pa_fresh_turn;
        let mut sides = delayed;
        sides.sort_unstable();
        sides.dedup();
        self.shroud_delayed = sides;
        self.pending_vision = pending.into_iter()
            .map(|(side, route, unit_id, vision, slowed)| PendingVision { side, route, unit_id, vision, slowed })
            .collect();
    }
}
