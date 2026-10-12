//! The attacks one side's units could make on another side's units on
//! their next turn, each with the target's exact expected loss: the
//! exposure term of the look-ahead's material evaluator
//! (wesnoth_ai/lookahead_exposure.py,
//! docs/exposure_evaluator_prereg_20261012.md).
//!
//! An attacker is a unit of the attacking side on the map, not petrified,
//! with an attack. Its reach is the move planner's (`unit_reach`) at its
//! full movement, `max_moves`, under the attacking side's view: the zone
//! of control of the units that side sees, their hexes closed to it. A
//! target is a unit of the target side that the attacking side sees, not
//! petrified (the interface offers no attack on one, docs/wesnoth_rules.md
//! "Petrified units are untargetable for attack"). An attack hex is a hex
//! next to the target that the attacker reaches and may stand on: its own
//! hex, or one that no unit the attacking side sees holds.
//!
//! On each attack hex, with the attacker moved there on a scratch copy of
//! the core (its terrain defense, time of day, illumination, leadership
//! and backstab are read from there): the attacker's weapon as the attack
//! dialog preselects it (`dialog_weapon`), the defender's counter as the
//! engine chooses it (`counter_weapon`), and the fight's exact outcomes
//! (`outcome_distribution`, advancements taken uniformly). The target's
//! loss in an outcome is its hit points when it dies, else the hit points
//! it lost; an outcome that heals it (drain, feeding) or advances it loses
//! nothing. Per (attacker, target) the hex with the largest expected loss
//! is kept, the first in the target's neighbour order among equals.
//!
//! Two hexes from which the fight's whole input (`FightInputs`, the
//! attacker's first weapon against no counter: every input a hex changes
//! is in it, and none that a weapon changes depends on the hex) is the
//! same give the same fight; it is computed once per (attacker, target).

use std::collections::HashMap;

use pyo3::prelude::*;

use crate::combat::{fight_stats, FightInputs};
use crate::core::GameCore;
use crate::dijkstra_reach;
use crate::outcomes::{better_combat, engine_marginals, strike_dp, Marginals, OutcomeKey};

/// One attack of `GameCore.threats`: attacker id, target id, the attack
/// hex (x, y), the attacker's weapon, the defender's counter (-1: none),
/// the target's expected hit-point loss and its death probability.
type ThreatRow = (String, String, i64, i64, i64, i64, f64, f64);

/// A fight valued from one attack hex: (weapon, counter, whether a
/// heuristic chose them, expected loss, death probability), None when the
/// DP could not enumerate it.
type Valued = Option<(i64, i64, bool, f64, f64)>;

/// (the target's expected hit-point loss, its death probability) over an
/// attack's outcomes, the target starting at `hp` (module doc).
fn target_loss(dist: &[(OutcomeKey, f64)], hp: i64) -> (f64, f64) {
    let (mut loss, mut death) = (0.0, 0.0);
    for ((_, target), p) in dist {
        if target.hp <= 0 {
            loss += p * hp as f64;
            death += p;
        } else if target.hp < hp {
            loss += p * (hp - target.hp) as f64;
        }
    }
    (loss, death)
}

impl GameCore {
    /// The attack dialog's preselected weapon for unit `a` attacking `d`
    /// (`mouse_handler::fill_weapon_choices`, src/mouse_events.cpp:1272-1296
    /// at the 1.18.4 tag): the weapons in order, each with the counter the
    /// engine answers it with, a later one taking the place of the best so
    /// far when `better_attack` prefers it at harm weight `harm_weight`
    /// (the dialog's 0.5; `better_combat` from the attacker's side,
    /// src/actions/attack.cpp:451-460). Returns (weapon, counter, whether a
    /// heuristic chose): when a weapon's strike DP overflows, the weapon
    /// with the most damage x strikes, the lowest index among equals, as
    /// the counter's fallback does; a counter chosen by its own fallback
    /// counts too.
    pub(crate) fn dialog_weapon(&self, a: usize, d: usize, harm_weight: f64) -> PyResult<(i64, i64, bool)> {
        let n = self.weapons_of(a).len() as i64;
        let mut fights = Vec::with_capacity(n as usize);
        let mut fallback = false;
        for w in 0..n {
            let counter = self.counter_weapon(a, d, w)?;
            fallback |= counter.fallback;
            fights.push((w, counter.weapon, fight_stats(&self.fight_inputs(a, d, w, counter.weapon)?)));
        }
        if n == 1 {
            return Ok((0, fights[0].1, fallback));
        }
        let mut best: Option<(i64, i64, Marginals, Marginals)> = None;
        for (w, counter, (au, du, a_st, d_st)) in &fights {
            let Some(states) = strike_dp(a_st, d_st.as_ref(), au, du, true) else {
                let (w, counter) = most_damage(&fights);
                return Ok((w, counter, true));
            };
            let (a_m, d_m) = engine_marginals(&states, a_st, d_st.as_ref(), au, du);
            let better = match &best {
                None => true,
                Some((_, _, best_a, best_d)) => better_combat(&a_m, &d_m, best_a, best_d, harm_weight),
            };
            if better {
                best = Some((*w, *counter, a_m, d_m));
            }
        }
        let (w, counter, _, _) = best.expect("a unit with weapons has a best one");
        Ok((w, counter, fallback))
    }

    /// Unit `a` attacking `d` from where it stands (module doc): the
    /// dialog's weapon, the engine's counter, the target's expected loss
    /// and death probability.
    fn value_fight(&self, a: usize, d: usize, harm_weight: f64) -> PyResult<Valued> {
        let (weapon, counter, fallback) = self.dialog_weapon(a, d, harm_weight)?;
        let Some(dist) = self.outcome_distribution_with(a, d, weapon, counter, true)? else { return Ok(None) };
        let (loss, death) = target_loss(&dist, self.units[d].current_hp);
        Ok(Some((weapon, counter, fallback, loss, death)))
    }

    /// Unit `i` stood on hex `h` (a scratch copy's position change).
    fn stand_on(&mut self, i: usize, h: usize) {
        let u = &mut self.units[i];
        u.hex = h as i64;
        u.x = self.map.hx[h];
        u.y = self.map.hy[h];
    }

    /// The hexes unit `i` reaches with `budget` movement points under the
    /// view's zone of control and blocking (`unit_reach`): movement
    /// points spent per hex, -1 where unreached.
    fn reach_mp(&self, i: usize, zoc: &[u8], enemy: &[u8], ally: &[u8], budget: i64) -> PyResult<Vec<i64>> {
        let u = &self.units[i];
        let c = if u.has_status("slowed") { u.class_slowed_id } else { u.class_id };
        let classes = self.classes.read().unwrap();
        if c < 0 || c as usize >= classes.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!("unit {} has no movement class", u.id)));
        }
        let class = &classes[c as usize];
        let h = self.map.h;
        let (mut mp, mut cost, mut prev) = (vec![-1i64; h], vec![f64::INFINITY; h], vec![-1i64; h]);
        dijkstra_reach(&self.map.nbrs, &class.mcost, &class.dsub, zoc, enemy, ally, u.hex as usize, budget,
                       u.has_ability("skirmisher"), &mut mp, &mut cost, &mut prev);
        Ok(mp)
    }
}

/// The DP-overflow fallback of `dialog_weapon`: (weapon, counter) of the
/// most damage x strikes, the lowest index among equals.
fn most_damage(fights: &[(i64, i64, crate::combat::Fight)]) -> (i64, i64) {
    let mut best = (fights[0].0, fights[0].1);
    let mut best_score = -1;
    for (w, counter, (_, _, a_st, _)) in fights {
        let score = a_st.damage * a_st.n_attacks;
        if score > best_score {
            (best, best_score) = ((*w, *counter), score);
        }
    }
    best
}

#[pymethods]
impl GameCore {
    /// The attacks `attacker_side`'s units could make on `target_side`'s
    /// units on their next turn (module doc): (one row per (attacker,
    /// target) that has an attack hex, in unit order, as `ThreatRow`;
    /// the weapon choices a heuristic made; the attack hexes whose fight
    /// the DP could not enumerate, left out of their row).
    fn threats(&self, attacker_side: i64, target_side: i64, harm_weight: f64)
        -> PyResult<(Vec<ThreatRow>, i64, i64)> {
        let (view, _) = self.side_view(attacker_side);
        let targets: Vec<usize> = (0..self.units.len())
            .filter(|&t| {
                let u = &self.units[t];
                u.side == target_side && u.hex >= 0 && view.visible[t] != 0 && !u.has_status("petrified")
            })
            .collect();
        let (mut rows, mut fallbacks, mut unvalued) = (Vec::new(), 0i64, 0i64);
        if targets.is_empty() {
            return Ok((rows, fallbacks, unvalued));
        }
        let mut scratch = self.clone();
        for i in 0..self.units.len() {
            let u = &self.units[i];
            if u.side != attacker_side || u.hex < 0 || u.attacks.is_empty() || u.has_status("petrified") {
                continue;
            }
            let start = u.hex as usize;
            let mp = self.reach_mp(i, &view.zoc, &view.enemy, &view.ally, u.max_moves)?;
            for &t in &targets {
                let target = &self.units[t];
                let th = target.hex as usize;
                let mut fights: HashMap<FightInputs, Valued> = HashMap::new();
                let mut best: Option<ThreatRow> = None;
                for &nh in &self.map.nbrs[th * 6..th * 6 + 6] {
                    if nh < 0 {
                        continue;
                    }
                    let h = nh as usize;
                    if mp[h] < 0 || (h != start && view.occupied[h] != 0) {
                        continue;
                    }
                    scratch.stand_on(i, h);
                    let key = scratch.fight_inputs(i, t, 0, -1)?;
                    let valued = match fights.get(&key) {
                        Some(v) => *v,
                        None => {
                            let v = scratch.value_fight(i, t, harm_weight)?;
                            fights.insert(key, v);
                            v
                        }
                    };
                    let Some((weapon, counter, fallback, loss, death)) = valued else {
                        unvalued += 1;
                        continue;
                    };
                    fallbacks += fallback as i64;
                    if best.as_ref().map_or(true, |b| loss > b.6) {
                        best = Some((u.id.clone(), target.id.clone(), self.map.hx[h], self.map.hy[h], weapon,
                                     counter, loss, death));
                    }
                }
                rows.extend(best);
            }
            scratch.stand_on(i, start);
        }
        Ok((rows, fallbacks, unvalued))
    }
}
