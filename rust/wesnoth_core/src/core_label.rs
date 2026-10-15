//! Phase 35: a recorded command's label in the encoding's own basis,
//! built beside the encoding from the same orderings
//! (`encode_streams(label=...)`): what `replay_dataset._action_indices`
//! builds on a Python view and `encode_worker.label_in_raw_basis`
//! re-indexes into the encoding's hex tokens.
//!
//! The actor slot is the side's unit standing on the command's hex among
//! the visible units in slot order (sorted by y, x, id), or for a recruit
//! the first of the side's recruit options of that type after them; the
//! end_turn slot follows the recruits. The target is the command's hex in
//! the token basis (the full board, or the relevant set); a hex off the
//! map leaves the command without a label, a hex on the map without a
//! token keeps the label with no target slot. A move whose order the
//! engine cut short names the hex its player clicked when the unit could
//! end a move there in the observable state (`move_label_hex`: the
//! unit's single-turn reach, not the start, no visible unit on it), else
//! the path's end. These are training labels, not engine rules; the
//! Python label builder stays the oracle (tests/test_core_label.py).

use crate::core::GameCore;
use crate::observe::SideView;
use pyo3::prelude::*;

/// A command as Python passes it (`replay_dataset.core_label_args`):
/// (kind, x0, y0, x1, y1, clicked, recruit type). A move gives its path's
/// start and end, and the clicked hex when the label rule asks the reach
/// about it; an attack its attacker's and defender's hexes; a recruit its
/// target hex in (x1, y1).
pub(crate) type LabelArgs = (String, i64, i64, i64, i64, Option<(i64, i64)>, Option<String>);

/// A command the label builder reads.
pub(crate) enum Recorded {
    EndTurn,
    Move { start: (i64, i64), stop: (i64, i64), clicked: Option<(i64, i64)> },
    Attack { from: (i64, i64), to: (i64, i64) },
    Recruit { unit_type: String, to: (i64, i64) },
}

impl Recorded {
    pub(crate) fn from_args(args: LabelArgs) -> PyResult<Recorded> {
        let (kind, x0, y0, x1, y1, clicked, unit_type) = args;
        match kind.as_str() {
            "end_turn" => Ok(Recorded::EndTurn),
            "move" => Ok(Recorded::Move { start: (x0, y0), stop: (x1, y1), clicked }),
            "attack" => Ok(Recorded::Attack { from: (x0, y0), to: (x1, y1) }),
            "recruit" => match unit_type {
                Some(unit_type) => Ok(Recorded::Recruit { unit_type, to: (x1, y1) }),
                None => Err(pyo3::exceptions::PyValueError::new_err("a recruit's label needs its unit type")),
            },
            other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "{other:?} is not a command a label names (end_turn, move, attack, recruit)"))),
        }
    }
}

/// A command's label: the actor slot, the target's token (None for a
/// target hex without one), and for a move whether the clicked hex is
/// the one named.
pub(crate) struct CoreLabel {
    pub actor: i64,
    pub target_token: Option<i64>,
    pub clicked: bool,
}

impl CoreLabel {
    /// (actor slot, target token or None, the clicked hex named).
    pub(crate) fn into_tuple(self) -> (i64, Option<i64>, bool) {
        (self.actor, self.target_token, self.clicked)
    }
}

impl GameCore {
    /// The label of `rec` for `side` over the encoding's basis: `view`
    /// the side's view, `vis` the core index of each unit slot,
    /// `tok_of_hex` each map hex's token (-1 off the basis). None when
    /// the command names no unit, recruit option or map hex of the
    /// side's.
    pub(crate) fn label_of(&self, side: i64, rec: &Recorded, view: &SideView, vis: &[usize], tok_of_hex: &[i64])
        -> PyResult<Option<CoreLabel>> {
        let recruits: &[String] = if side >= 1 && side as usize <= self.sides.len() {
            &self.sides[side as usize - 1].recruits
        } else {
            &[]
        };
        // Some(token or None) for a map hex, None off the map.
        let token = |xy: (i64, i64)| -> Option<Option<i64>> {
            self.map.pos_index.get(&xy).map(|&mh| (tok_of_hex[mh] >= 0).then_some(tok_of_hex[mh]))
        };
        let actor_at = |xy: (i64, i64)| -> Option<usize> {
            vis.iter().position(|&i| {
                let u = &self.units[i];
                (u.x, u.y) == xy && u.side == side
            })
        };
        let label = |actor: usize, target: (i64, i64), clicked: bool| {
            token(target).map(|target_token| CoreLabel { actor: actor as i64, target_token, clicked })
        };
        Ok(match rec {
            Recorded::EndTurn => Some(CoreLabel { actor: (vis.len() + recruits.len()) as i64, target_token: None,
                                                  clicked: false }),
            Recorded::Move { start, stop, clicked } => {
                let Some(a) = actor_at(*start) else { return Ok(None) };
                match clicked {
                    Some(c) if self.can_end_move_on(vis[a], *c, view)? => label(a, *c, true),
                    _ => label(a, *stop, false),
                }
            }
            Recorded::Attack { from, to } => actor_at(*from).and_then(|a| label(a, *to, false)),
            Recorded::Recruit { unit_type, to } => recruits.iter().position(|r| r == unit_type)
                .and_then(|j| label(vis.len() + j, *to, false)),
        })
    }

    /// Whether unit `i` can end a move order on `hex` in its side's
    /// observable state `view` (`pathfind_sim.unit_reach`'s landable set,
    /// the reach the core computes for the unit standing on its hex with
    /// unit i's movement left): reached, not its own hex, no visible unit
    /// on it.
    fn can_end_move_on(&self, i: usize, hex: (i64, i64), view: &SideView) -> PyResult<bool> {
        let u = &self.units[i];
        let Some((mp, _cost, _prev)) = self.reach_of(u.x, u.y, u.current_moves)? else {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "no unit on the map at {:?} in the core", (u.x, u.y))));
        };
        Ok(match self.map.pos_index.get(&hex) {
            Some(&h) => mp[h] >= 0 && hex != (u.x, u.y) && view.occupied[h] == 0,
            None => false,
        })
    }
}
