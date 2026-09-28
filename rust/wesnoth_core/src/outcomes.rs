//! Exact fight outcomes on the core, a transcription of
//! `tools/combat_outcomes.py`: the strike-by-strike probability DP over
//! the fight's schedule, the defender's weapon choice (the engine's
//! `battle_context::choose_defender_weapon` with `better_combat`, and the
//! level-up and debuff predictions of attack_prediction.cpp, 1.18.4) and
//! the outcome distribution of an attack with its advancement branches
//! (MCTS chance nodes, the neutral AI, the swap detector).
//!
//! The fight's parameters come from the attack command's own setup
//! (`fight_inputs`, `combat::fight_stats`), a survivor's record from its
//! write-back (`write_back`) and an advanced unit from the live advance
//! (`advance_to`, `amla`), so an outcome here is the state the command
//! produces. Every probability is computed with the Python's float
//! operations in its order, and accumulated in first-reached order (a
//! Python dict's), so tables, choices and distributions are identical
//! to the last bit (tests/test_rust_outcomes.py).

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use std::collections::HashMap;
use std::hash::Hash;

use crate::combat::{fight_stats, Fight, Stats, Unit, DRAIN_CONSTANT, DRAIN_PERCENT};
use crate::core::{GameCore, UnitRec};
use crate::core_attack::{attacker_weapon, write_back};
use crate::core_units::amla;

/// Probabilities below this are dropped before the distribution is
/// renormalized (the engine snaps at 1e-9, round_prob_if_close_to_sure;
/// the consumers renormalize anyway).
const PROB_EPSILON: f64 = 1e-12;
/// The DP's work caps: live cells after a strike, and strike events
/// (berserk multiplies the schedule). Past them the caller samples, as
/// the engine switches to Monte-Carlo past its own complexity bound.
const MAX_DP_STATES: usize = 4096;
const MAX_SCHEDULE: usize = 512;
const COMBAT_EXPERIENCE: i64 = 1;
const KILL_EXPERIENCE: i64 = 8;
/// `game_config::poison_amount`, as a float for `better_combat`.
const POISON_AMOUNT: f64 = 8.0;

/// Values accumulated per key in first-reached order: a Python dict's
/// `d[k] = d.get(k, 0.0) + p`.
struct Ordered<K> {
    items: Vec<(K, f64)>,
    index: HashMap<K, usize>,
}

impl<K: Clone + Eq + Hash> Ordered<K> {
    fn new() -> Self {
        Ordered { items: Vec::new(), index: HashMap::new() }
    }

    fn add(&mut self, key: K, p: f64) {
        match self.index.get(&key) {
            Some(&i) => self.items[i].1 += p,
            None => {
                self.index.insert(key.clone(), self.items.len());
                self.items.push((key, p));
            }
        }
    }
}

/// CPython's `sum()` of floats since 3.12: the first term, then
/// Neumaier's compensated summation, the compensation added at the end.
fn py_sum(values: &[f64]) -> f64 {
    let Some((&first, rest)) = values.split_first() else { return 0.0 };
    let mut s = first;
    let mut c = 0.0;
    for &x in rest {
        let t = s + x;
        if s.abs() >= x.abs() {
            c += (s - t) + x;
        } else {
            c += (x - t) + s;
        }
        s = t;
    }
    if c != 0.0 && c.is_finite() {
        s += c;
    }
    s
}

/// Python's `max(a, b)` and `min(a, b)`: the first argument unless the
/// second is strictly larger (smaller).
fn py_max(a: f64, b: f64) -> f64 {
    if b > a { b } else { a }
}

fn py_min(a: f64, b: f64) -> f64 {
    if b < a { b } else { a }
}

/// `game_config::kill_xp`: experience for killing a unit of `level`.
fn kill_xp(level: i64) -> i64 {
    if level != 0 { KILL_EXPERIENCE * level } else { KILL_EXPERIENCE / 2 }
}

/// One cell of the strike DP: both combatants' hit points and statuses,
/// and whether each was hit at least once.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Cell {
    a_hp: i64,
    d_hp: i64,
    a_slowed: bool,
    d_slowed: bool,
    a_poisoned: bool,
    d_poisoned: bool,
    a_petrified: bool,
    d_petrified: bool,
    a_touched: bool,
    d_touched: bool,
}

impl Cell {
    /// A dead unit's statuses cleared: it is gone, and they must not
    /// split outcomes. The touched flags stay (a death was a hit).
    fn canonical(mut self) -> Self {
        if self.a_hp <= 0 {
            (self.a_slowed, self.a_poisoned, self.a_petrified) = (false, false, false);
        }
        if self.d_hp <= 0 {
            (self.d_slowed, self.d_poisoned, self.d_petrified) = (false, false, false);
        }
        self.a_hp = self.a_hp.max(0);
        self.d_hp = self.d_hp.max(0);
        self
    }

    fn over(&self) -> bool {
        self.a_hp <= 0 || self.d_hp <= 0 || self.a_petrified || self.d_petrified
    }
}

/// The fight's strike events in order (true: the attacker strikes), as
/// `resolve_fight`'s loop takes them: alternating while either side has
/// strikes, the defender first only with a firststrike the attacker
/// lacks, berserk refilling both sides `rounds - 1` times. None past
/// MAX_SCHEDULE.
fn schedule(a: &Stats, d: Option<&Stats>) -> Option<Vec<bool>> {
    let defender_first_of = || d.is_some_and(|ds| ds.firststrike && !a.firststrike);
    let mut defender_first = defender_first_of();
    let mut rounds_left = a.rounds.max(d.map_or(1, |ds| ds.rounds)) - 1;
    let mut a_n = a.n_attacks;
    let mut d_n = d.map_or(0, |ds| ds.n_attacks);
    let mut out = Vec::new();
    loop {
        if out.len() > MAX_SCHEDULE {
            return None;
        }
        if !defender_first && a_n > 0 {
            out.push(true);
            a_n -= 1;
        }
        defender_first = false;
        if d_n > 0 {
            out.push(false);
            d_n -= 1;
        }
        if rounds_left > 0 && a_n == 0 && d_n == 0 {
            a_n = a.n_attacks;
            d_n = d.map_or(0, |ds| ds.n_attacks);
            rounds_left -= 1;
            defender_first = defender_first_of();
            continue;
        }
        if a_n <= 0 && d_n <= 0 {
            return Some(out);
        }
    }
}

/// The probability DP over the schedule: the final cells in first-reached
/// order, or None past the caps. A cell is absorbing once a unit died or
/// was petrified. A hit follows `combat.perform_hit_body`: damage (halved
/// for a slowed striker), the drain heal, and on a survivor poison, slow
/// and petrification. `track_touched` marks each side hit at least once
/// (the engine's debuff odds count hits, not damage).
fn strike_dp(a: &Stats, d: Option<&Stats>, au: &Unit, du: &Unit, track_touched: bool)
    -> Option<Vec<(Cell, f64)>> {
    let sched = schedule(a, d)?;
    let start = Cell {
        a_hp: au.hp,
        d_hp: du.hp,
        a_slowed: au.slowed(),
        d_slowed: du.slowed(),
        a_poisoned: au.poisoned(),
        d_poisoned: du.poisoned(),
        a_petrified: au.petrified(),
        d_petrified: du.petrified(),
        a_touched: false,
        d_touched: false,
    };
    let mut states = vec![(start, 1.0)];
    for attacker_strikes in sched {
        let mut next = Ordered::new();
        let mut live_mass = 0.0;
        for &(c, p) in &states {
            if c.over() {
                next.add(c, p);
                continue;
            }
            live_mass += p;
            let (st, striker, target) = if attacker_strikes {
                (a, au, du)
            } else {
                (d.expect("a defender strike has the defender's stats"), du, au)
            };
            let cth = st.cth.clamp(0, 100) as f64 / 100.0;
            if cth < 1.0 {
                next.add(c, p * (1.0 - cth));
            }
            if cth <= 0.0 {
                continue;
            }
            let mut hit = c;
            if track_touched {
                if attacker_strikes { hit.d_touched = true } else { hit.a_touched = true }
            }
            let (st_hp, st_slowed, tg_hp) =
                if attacker_strikes { (c.a_hp, c.a_slowed, c.d_hp) } else { (c.d_hp, c.d_slowed, c.a_hp) };
            let dmg = if st_slowed { st.slow_damage } else { st.damage };
            if dmg <= 0 {
                // A hit without damage applies no status either.
                next.add(hit, p * cth);
                continue;
            }
            let tg_hp_new = (tg_hp - dmg).max(0);
            let damage_done = tg_hp - tg_hp_new;
            let mut st_hp_new = st_hp;
            if st.drains && damage_done > 0 && !target.undrainable() {
                let mut heal = damage_done * DRAIN_PERCENT / 100 + DRAIN_CONSTANT;
                if heal != 0 {
                    heal = heal.min(striker.max_hp - st_hp);
                    heal = heal.max(1 - st_hp);
                    st_hp_new = st_hp + heal;
                }
            }
            if tg_hp_new > 0 {
                let (poisoned, slowed, petrified) = if attacker_strikes {
                    (&mut hit.d_poisoned, &mut hit.d_slowed, &mut hit.d_petrified)
                } else {
                    (&mut hit.a_poisoned, &mut hit.a_slowed, &mut hit.a_petrified)
                };
                if st.poisons && !target.unpoisonable() {
                    *poisoned = true;
                }
                if st.slows {
                    *slowed = true;
                }
                // A surviving target of a petrifying hit turns to stone and
                // the fight ends (the cell is absorbing from here).
                if st.petrifies {
                    *petrified = true;
                }
            }
            if attacker_strikes {
                (hit.a_hp, hit.d_hp) = (st_hp_new, tg_hp_new);
            } else {
                (hit.a_hp, hit.d_hp) = (tg_hp_new, st_hp_new);
            }
            next.add(if tg_hp_new <= 0 { hit.canonical() } else { hit }, p * cth);
        }
        states = next.items;
        if states.len() > MAX_DP_STATES {
            return None;
        }
        if live_mass <= PROB_EPSILON {
            break;
        }
    }
    Some(states)
}

/// What the engine's combatant simulation gives `better_combat`: the
/// death probability, the average hit points after the level-up it
/// predicts, and the probability of ending poisoned.
#[derive(Clone, Copy)]
struct Marginals {
    death: f64,
    avg_hp: f64,
    poisoned: f64,
}

/// `calculate_probability_of_debuff` (attack_prediction.cpp): the
/// probability of carrying a debuff after the fight, where levelling up
/// on a kill cures it.
fn probability_of_debuff(initial: f64, enemy_gives: bool, prob_touched: f64, prob_stay_alive: f64,
                         kill_heals: bool, prob_kill: f64) -> f64 {
    let prob_touched = py_max(prob_touched, 0.0);
    let prob_stay_alive = py_max(prob_stay_alive, 0.0);
    let prob_kill = py_min(py_max(prob_kill, 0.0), 1.0);
    let already_not_touched = initial * (1.0 - prob_touched);
    let already_touched = initial * prob_touched;
    let healthy_touched = (1.0 - initial) * prob_touched;
    let survive_if_not_hit = 1.0;
    let survive_if_hit =
        if prob_touched > 0.0 { (prob_stay_alive - (1.0 - prob_touched)) / prob_touched } else { 1.0 };
    let kill_if_survive = if prob_stay_alive > 0.0 { prob_kill / prob_stay_alive } else { 0.0 };
    let mut debuff = 0.0;
    debuff += if kill_heals {
        already_not_touched * (1.0 - survive_if_not_hit * kill_if_survive)
    } else {
        already_not_touched
    };
    debuff += if kill_heals { already_touched * (1.0 - survive_if_hit * kill_if_survive) } else { already_touched };
    if enemy_gives {
        debuff += if kill_heals {
            healthy_touched * (1.0 - survive_if_hit * kill_if_survive)
        } else {
            healthy_touched
        };
    }
    debuff
}

/// Whether `do_fight` (attack_prediction.cpp:2211-2244) hands the fight to
/// `one_strike_fight`: no slow, effective drain, petrify or berserk on
/// either side, neither combatant slowed already, one strike each at most.
fn one_strike_fight(a: &Stats, d: Option<&Stats>, au: &Unit, du: &Unit) -> bool {
    if au.slowed() || du.slowed() {
        return false;
    }
    let simple = |st: &Stats, opp: &Unit| {
        let drains = st.drains && !opp.undrainable();
        !(st.slows || drains || st.petrifies || st.rounds != 1 || st.n_attacks > 1)
    };
    simple(a, du) && d.is_none_or(|ds| simple(ds, au))
}

/// `combatant::average_hp()` after the level-up `combatant::fight`
/// predicts: the fight's experience alone reaching the cap scores every
/// surviving outcome at full health; only a kill's reaching it scores the
/// kills at full health, exactly on the matrix path, and on the one-strike
/// path as if the unit's health and the kill were independent
/// (attack_prediction.cpp:2038-2048 and 1733-1752). `avg_hp` sums p * hp
/// over the survivals, `avg_hp_on_kill` over those where the opponent
/// died; `death` is this unit's death probability, `kill` the opponent's.
fn levelup_average_hp(u: &Unit, opp: &Unit, avg_hp: f64, avg_hp_on_kill: f64, death: f64, kill: f64,
                      one_strike: bool) -> f64 {
    if u.experience + COMBAT_EXPERIENCE * opp.level >= u.max_experience {
        return (1.0 - death) * u.max_hp as f64;
    }
    if u.experience + kill_xp(opp.level) < u.max_experience {
        return avg_hp;
    }
    if !one_strike {
        return avg_hp - avg_hp_on_kill + kill * u.max_hp as f64;
    }
    let survive = 1.0 - death;
    let scale = if survive > f64::MIN_POSITIVE { 1.0 - kill / survive } else { 0.0 };
    scale * avg_hp + kill * u.max_hp as f64
}

/// Both combatants' marginals from a touched-tracked DP, as
/// `combatant::fight` computes them; the touched probability is exact
/// where the engine approximates it strike by strike.
fn engine_marginals(states: &[(Cell, f64)], a: &Stats, d: Option<&Stats>, au: &Unit, du: &Unit)
    -> (Marginals, Marginals) {
    let (mut a_death, mut d_death, mut a_avg, mut d_avg) = (0.0, 0.0, 0.0, 0.0);
    let (mut a_touch, mut d_touch, mut a_avg_on_kill, mut d_avg_on_kill) = (0.0, 0.0, 0.0, 0.0);
    for &(c, p) in states {
        if c.a_hp <= 0 {
            a_death += p;
        } else {
            a_avg += p * c.a_hp as f64;
            if c.d_hp <= 0 {
                a_avg_on_kill += p * c.a_hp as f64;
            }
        }
        if c.d_hp <= 0 {
            d_death += p;
        } else {
            d_avg += p * c.d_hp as f64;
            if c.a_hp <= 0 {
                d_avg_on_kill += p * c.d_hp as f64;
            }
        }
        if c.a_touched {
            a_touch += p;
        }
        if c.d_touched {
            d_touch += p;
        }
    }
    let one_strike = one_strike_fight(a, d, au, du);
    let a_avg = levelup_average_hp(au, du, a_avg, a_avg_on_kill, a_death, d_death, one_strike);
    let d_avg = levelup_average_hp(du, au, d_avg, d_avg_on_kill, d_death, a_death, one_strike);
    let initial = |u: &Unit| if u.poisoned() { 1.0 } else { 0.0 };
    let mut a_poisoned = probability_of_debuff(
        initial(au), d.is_some_and(|ds| ds.poisons) && !au.unpoisonable(), a_touch, 1.0 - a_death,
        au.experience + kill_xp(du.level) >= au.max_experience, d_death,
    );
    let mut d_poisoned = probability_of_debuff(
        initial(du), a.poisons && !du.unpoisonable(), d_touch, 1.0 - d_death,
        du.experience + kill_xp(au.level) >= du.max_experience, a_death,
    );
    // The fight's experience alone levelling a unit up cures it
    // (`combatant::fight` applies this after the formula).
    if au.experience + COMBAT_EXPERIENCE * du.level >= au.max_experience {
        a_poisoned = 0.0;
    }
    if du.experience + COMBAT_EXPERIENCE * au.level >= du.max_experience {
        d_poisoned = 0.0;
    }
    (
        Marginals { death: a_death, avg_hp: a_avg, poisoned: a_poisoned },
        Marginals { death: d_death, avg_hp: d_avg, poisoned: d_poisoned },
    )
}

/// `battle_context::better_combat` (attack.cpp): is fight A better for
/// "us" than fight B? The kill balance, then the hit points kept net of
/// the poison that survives the fight, then the damage done.
fn better_combat(us_a: &Marginals, them_a: &Marginals, us_b: &Marginals, them_b: &Marginals,
                 harm_weight: f64) -> bool {
    let a = them_a.death - us_a.death * harm_weight;
    let b = them_b.death - us_b.death * harm_weight;
    if a - b < -0.01 {
        return false;
    }
    if a - b > 0.01 {
        return true;
    }
    let poison = |m: &Marginals| if m.poisoned > 0.0 { (m.poisoned - m.death) * POISON_AMOUNT } else { 0.0 };
    let a = (us_a.avg_hp - poison(us_a)) * harm_weight - (them_a.avg_hp - poison(them_a));
    let b = (us_b.avg_hp - poison(us_b)) * harm_weight - (them_b.avg_hp - poison(them_b));
    if a - b < -0.01 {
        return false;
    }
    if a - b > 0.01 {
        return true;
    }
    them_a.avg_hp < them_b.avg_hp
}

/// The defender's counter-attack weapon (-1: none), the strike tables
/// simulated to choose it (per candidate weapon, in weapon order; empty
/// when one weapon or none could answer), and whether the DP overflowed
/// into the heuristic fallback.
struct CounterChoice {
    weapon: i64,
    tables: Vec<(usize, Vec<(Cell, f64)>)>,
    fallback: bool,
}

/// One combatant's state in an outcome: type, hit points, slowed,
/// poisoned, petrified. A dead unit is ("", 0, false, false, false).
#[derive(Clone, PartialEq, Eq, Hash)]
struct Side {
    name: String,
    hp: i64,
    slowed: bool,
    poisoned: bool,
    petrified: bool,
}

impl Side {
    fn dead() -> Self {
        Side { name: String::new(), hp: 0, slowed: false, poisoned: false, petrified: false }
    }

    fn of(u: &UnitRec) -> Self {
        Side {
            name: u.name.clone(),
            hp: u.current_hp,
            slowed: u.has_status("slowed"),
            poisoned: u.has_status("poisoned"),
            petrified: u.has_status("petrified"),
        }
    }
}

/// An attack's outcome: the attacker's side, then the defender's.
type OutcomeKey = (Side, Side);

/// The two sides' fight statistics as dicts, the defender's None when
/// it does not answer.
type StatsPair<'py> = (Bound<'py, PyDict>, Option<Bound<'py, PyDict>>);

impl GameCore {
    /// `choose_defender_weapon` (attack.cpp, 1.18.4) for unit `a`
    /// attacking unit `d` with weapon `a_weapon`: among the defender's
    /// weapons of the attack's range, the one whose predicted fight is
    /// best for the defender (`better_combat`, harm weight 1). No weapon
    /// when either side has no attack or the defender is petrified.
    ///
    /// The engine first keeps the candidates whose simple rating (blows x
    /// damage x chance to hit x defense_weight) reaches the best weight's
    /// minimum; its loop assigns that weight before comparing against it,
    /// so the minimum stays 0 and every candidate passes (defense_weight
    /// has no other effect; the pinned scrape does not carry it). When the
    /// DP overflows (huge berserk or swarm fights, which the engine hands
    /// to Monte-Carlo) the choice falls back to the most damage x strikes,
    /// ties to the lowest index: a known divergence the caller counts.
    fn counter_weapon(&self, a: usize, d: usize, a_weapon: i64) -> PyResult<CounterChoice> {
        let none = CounterChoice { weapon: -1, tables: Vec::new(), fallback: false };
        if self.units[a].attacks.is_empty() || self.units[d].attacks.is_empty()
            || self.units[d].has_status("petrified") {
            return Ok(none);
        }
        let aw = self.weapons_of(a);
        let a_ranged = aw[attacker_weapon(aw.len(), a_weapon)?].ranged;
        let candidates: Vec<usize> =
            self.weapons_of(d).iter().enumerate().filter(|(_, w)| w.ranged == a_ranged).map(|(i, _)| i).collect();
        match candidates.len() {
            0 => return Ok(none),
            1 => return Ok(CounterChoice { weapon: candidates[0] as i64, ..none }),
            _ => {}
        }
        let mut fights = Vec::with_capacity(candidates.len());
        for &i in &candidates {
            fights.push((i, fight_stats(&self.fight_inputs(a, d, a_weapon, i as i64)?)));
        }
        let mut sims: Vec<(usize, Marginals, Marginals)> = Vec::with_capacity(fights.len());
        let mut tables = Vec::with_capacity(fights.len());
        for (i, (au, du, a_st, d_st)) in &fights {
            let Some(states) = strike_dp(a_st, d_st.as_ref(), au, du, true) else {
                return Ok(CounterChoice { weapon: fallback_weapon(&fights), tables: Vec::new(), fallback: true });
            };
            let (a_m, d_m) = engine_marginals(&states, a_st, d_st.as_ref(), au, du);
            sims.push((*i, a_m, d_m));
            tables.push((*i, states));
        }
        let mut best = 0;
        for k in 1..sims.len() {
            let (_, a_m, d_m) = &sims[k];
            let (_, best_a, best_d) = &sims[best];
            if better_combat(d_m, a_m, best_d, best_a, 1.0) {
                best = k;
            }
        }
        Ok(CounterChoice { weapon: sims[best].0 as i64, tables, fallback: false })
    }

    /// Every outcome of unit `a` attacking unit `d` with weapon `a_weapon`
    /// (the defender answering with `counter_weapon`), with its
    /// probability, dust under PROB_EPSILON dropped and the rest
    /// renormalized. None past the DP's caps, and, unless
    /// `uniform_advancement`, when a unit could reach its experience cap
    /// (the caller samples). With it, a survivor past its cap becomes each
    /// unit its advancement can make, the choice uniform among the types
    /// it is offered.
    fn outcome_distribution(&self, a: usize, d: usize, a_weapon: i64, uniform_advancement: bool)
        -> PyResult<Option<Vec<(OutcomeKey, f64)>>> {
        let counter = self.counter_weapon(a, d, a_weapon)?;
        let (au, du, a_st, d_st) = fight_stats(&self.fight_inputs(a, d, a_weapon, counter.weapon)?);
        let may_advance = |u: &Unit, opp: &Unit| {
            u.experience + kill_xp(opp.level).max(COMBAT_EXPERIENCE * opp.level) >= u.max_experience
        };
        if !uniform_advancement && (may_advance(&au, &du) || may_advance(&du, &au)) {
            return Ok(None);
        }
        let Some(states) = strike_dp(&a_st, d_st.as_ref(), &au, &du, false) else { return Ok(None) };
        let mut probs: Ordered<OutcomeKey> = Ordered::new();
        for (c, p) in states {
            if p < PROB_EPSILON {
                continue;
            }
            let (a_hp, d_hp) = (c.a_hp.max(0), c.d_hp.max(0));
            let a_branches = self.side_outcomes(a, d, a_hp, (c.a_slowed, c.a_poisoned, c.a_petrified), d_hp <= 0,
                                                du.level, uniform_advancement);
            let d_branches = self.side_outcomes(d, a, d_hp, (c.d_slowed, c.d_poisoned, c.d_petrified), a_hp <= 0,
                                                au.level, uniform_advancement);
            for (a_side, pa) in &a_branches {
                for (d_side, pd) in &d_branches {
                    probs.add((a_side.clone(), d_side.clone()), p * pa * pd);
                }
            }
        }
        let total = py_sum(&probs.items.iter().map(|(_, v)| *v).collect::<Vec<_>>());
        if probs.items.is_empty() || total <= 0.0 {
            return Ok(None);
        }
        Ok(Some(probs.items.into_iter().map(|(k, v)| (k, v / total)).collect()))
    }

    /// Unit `i`'s states in one fight outcome, with their probabilities:
    /// dead, or alive at `hp` with `statuses` (slowed, poisoned,
    /// petrified), a fed kill's +1 included; with `uniform_advancement`, a
    /// survivor whose experience reaches its cap expands into its
    /// advancements.
    #[allow(clippy::too_many_arguments)]
    fn side_outcomes(&self, i: usize, opp: usize, hp: i64, statuses: (bool, bool, bool), opp_died: bool,
                     opp_level: i64, uniform_advancement: bool) -> Vec<(Side, f64)> {
        if hp <= 0 {
            return vec![(Side::dead(), 1.0)];
        }
        let u = &self.units[i];
        let fed = opp_died && u.has_ability("feeding") && !self.unplagueable(opp);
        let (slowed, poisoned, petrified) = statuses;
        let base = Side { name: u.name.clone(), hp: hp + fed as i64, slowed, poisoned, petrified };
        let xp = u.current_exp + if opp_died { kill_xp(opp_level) } else { COMBAT_EXPERIENCE * opp_level };
        if !uniform_advancement || xp < u.max_exp {
            return vec![(base, 1.0)];
        }
        let mut post = u.clone();
        write_back(&mut post, hp, xp, slowed, poisoned, petrified, fed);
        self.advancement_outcomes(&post)
    }

    /// Every unit `u` becomes by advancing from its experience, with its
    /// probability under the uniform choice among the types it is offered,
    /// chains followed link by link and equal outcomes merged in
    /// first-reached order (`replay_dataset.enumerate_advancement_outcomes`).
    fn advancement_outcomes(&self, u: &UnitRec) -> Vec<(Side, f64)> {
        if u.current_exp < u.max_exp {
            return vec![(Side::of(u), 1.0)];
        }
        let targets = self.advance_targets(u);
        if targets.is_empty() {
            let mut v = u.clone();
            amla(&mut v);
            return self.advancement_outcomes(&v);
        }
        let p_each = 1.0 / targets.len() as f64;
        let mut out = Ordered::new();
        for t in &targets {
            let mut v = u.clone();
            self.advance_to(&mut v, t);
            for (side, p) in self.advancement_outcomes(&v) {
                out.add(side, p_each * p);
            }
        }
        out.items
    }
}

/// The DP-overflow fallback: the candidate with the most damage x
/// strikes, ties to the lowest index.
fn fallback_weapon(fights: &[(usize, Fight)]) -> i64 {
    let (mut best, mut best_score) = (-1, -1);
    for (i, (_, _, _, d_st)) in fights {
        let st = d_st.as_ref().expect("a candidate answers");
        let score = st.damage * st.n_attacks;
        if score > best_score {
            (best, best_score) = (*i as i64, score);
        }
    }
    best
}

fn cell_tuple<'py>(py: Python<'py>, c: &Cell) -> PyResult<Bound<'py, PyTuple>> {
    (c.a_hp, c.d_hp, c.a_slowed, c.d_slowed, c.a_poisoned, c.d_poisoned, c.a_petrified, c.d_petrified,
     c.a_touched, c.d_touched).into_pyobject(py)
}

/// `combat_outcomes.OutcomeKey`: (a_hp, d_hp, a_slowed, d_slowed,
/// a_poisoned, d_poisoned, a_petrified, d_petrified, a_type, d_type).
fn outcome_tuple<'py>(py: Python<'py>, (a, d): &OutcomeKey) -> PyResult<Bound<'py, PyTuple>> {
    (a.hp, d.hp, a.slowed, d.slowed, a.poisoned, d.poisoned, a.petrified, d.petrified, a.name.as_str(),
     d.name.as_str()).into_pyobject(py)
}

fn stats_dict<'py>(py: Python<'py>, st: &Stats) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("cth", st.cth)?;
    out.set_item("damage", st.damage)?;
    out.set_item("slow_damage", st.slow_damage)?;
    out.set_item("n_attacks", st.n_attacks)?;
    out.set_item("rounds", st.rounds)?;
    out.set_item("firststrike", st.firststrike)?;
    out.set_item("drains", st.drains)?;
    out.set_item("poisons", st.poisons)?;
    out.set_item("slows", st.slows)?;
    out.set_item("petrifies", st.petrifies)?;
    Ok(out)
}

#[pymethods]
impl GameCore {
    /// `combat_outcomes.counter_weapon_choice` for the units on (ax, ay)
    /// and (dx, dy): (the defender's weapon or -1, {weapon: {DP cell
    /// (a_hp, d_hp, a_slowed, d_slowed, a_poisoned, d_poisoned,
    /// a_petrified, d_petrified, a_touched, d_touched): p}}, whether the
    /// heuristic fallback chose). None when a hex holds no unit.
    fn counter_weapon_choice<'py>(&self, py: Python<'py>, ax: i64, ay: i64, dx: i64, dy: i64, a_weapon: i64)
        -> PyResult<Option<(i64, Bound<'py, PyDict>, bool)>> {
        let (Some(a), Some(d)) = (self.unit_at(ax, ay), self.unit_at(dx, dy)) else { return Ok(None) };
        let choice = self.counter_weapon(a, d, a_weapon)?;
        let tables = PyDict::new(py);
        for (w, states) in &choice.tables {
            let t = PyDict::new(py);
            for (c, p) in states {
                t.set_item(cell_tuple(py, c)?, *p)?;
            }
            tables.set_item(*w, t)?;
        }
        Ok(Some((choice.weapon, tables, choice.fallback)))
    }

    /// `combat_outcomes.enumerate_attack_outcomes` for the units on
    /// (ax, ay) and (dx, dy): ({outcome key: p}, attacker id, defender id),
    /// or None (an empty hex, the DP's caps, or a possible advancement
    /// without `uniform_advancement`).
    #[allow(clippy::too_many_arguments)]
    fn attack_outcomes<'py>(&self, py: Python<'py>, ax: i64, ay: i64, dx: i64, dy: i64, a_weapon: i64,
                            uniform_advancement: bool) -> PyResult<Option<(Bound<'py, PyDict>, String, String)>> {
        let (Some(a), Some(d)) = (self.unit_at(ax, ay), self.unit_at(dx, dy)) else { return Ok(None) };
        let Some(dist) = self.outcome_distribution(a, d, a_weapon, uniform_advancement)? else { return Ok(None) };
        let probs = PyDict::new(py);
        for (key, p) in &dist {
            probs.set_item(outcome_tuple(py, key)?, *p)?;
        }
        Ok(Some((probs, self.units[a].id.clone(), self.units[d].id.clone())))
    }

    /// Both sides' fight statistics (chance to hit, damage, slowed
    /// damage, strikes, rounds and the specials the fight reads) for the
    /// units on (ax, ay) and (dx, dy) with those weapons (`d_weapon` -1:
    /// no answer): (attacker's, defender's or None), or None when a hex
    /// holds no unit.
    #[allow(clippy::too_many_arguments)]
    fn fight_stats<'py>(&self, py: Python<'py>, ax: i64, ay: i64, dx: i64, dy: i64, a_weapon: i64, d_weapon: i64)
        -> PyResult<Option<StatsPair<'py>>> {
        let (Some(a), Some(d)) = (self.unit_at(ax, ay), self.unit_at(dx, dy)) else { return Ok(None) };
        let (_, _, a_st, d_st) = fight_stats(&self.fight_inputs(a, d, a_weapon, d_weapon)?);
        let d_dict = match d_st {
            Some(st) => Some(stats_dict(py, &st)?),
            None => None,
        };
        Ok(Some((stats_dict(py, &a_st)?, d_dict)))
    }
}
