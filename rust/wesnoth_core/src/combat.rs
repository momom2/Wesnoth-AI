//! Phase 3a: one attack resolved in Rust (docs/rust_port_plan.md).
//! A transcription of `wesnoth_ai/combat.py`: std::mt19937 with the
//! Knuth seeding (Wesnoth's `mt_rng`), `_compute_battle_stats`,
//! `resolve_attack` and `_perform_hit`, over the two units' snapshots
//! as flat integers. The Python module stays the diff oracle
//! (tests/test_rust_combat.py: fuzzed fights and the `[mp_checkup]`
//! corpus); every rule below cites the Python it mirrors, which cites
//! the engine.
//!
//! Per unit, `ints` (UNIT_INTS entries): hp, max_hp, level,
//! experience, max_experience, alignment (0 lawful, 1 neutral,
//! 2 chaotic, 3 liminal), defense_pct, opp_resist (the OPPONENT's
//! resistance to this unit's weapon type, 100 when the unit has no
//! weapon), weapon damage, weapon number, accuracy, parry.
//! `flags` (UNIT_FLAGS entries): slowed, poisoned, petrified,
//! invulnerable, fearless, undrainable, unpoisonable, steadfast, and
//! the weapon specials magical, marksman, deflect, backstab, charge,
//! swarm, drains, plague, poison, slow, petrifies, firststrike,
//! berserk.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

pub const UNIT_INTS: usize = 12;
pub const UNIT_FLAGS: usize = 21;
const COMBAT_EXPERIENCE: i64 = 1;
const KILL_EXPERIENCE: i64 = 8;
const MAX_LIMINAL_BONUS: i64 = 25;

// ---- ints ----
const HP: usize = 0;
const MAX_HP: usize = 1;
const LEVEL: usize = 2;
const EXPERIENCE: usize = 3;
const MAX_EXPERIENCE: usize = 4;
const ALIGNMENT: usize = 5;
const DEFENSE_PCT: usize = 6;
const OPP_RESIST: usize = 7;
const W_DAMAGE: usize = 8;
const W_NUMBER: usize = 9;
const W_ACCURACY: usize = 10;
const W_PARRY: usize = 11;
// ---- flags ----
const F_SLOWED: usize = 0;
const F_POISONED: usize = 1;
const F_PETRIFIED: usize = 2;
const F_INVULNERABLE: usize = 3;
const F_FEARLESS: usize = 4;
const F_UNDRAINABLE: usize = 5;
const F_UNPOISONABLE: usize = 6;
const F_STEADFAST: usize = 7;
const F_MAGICAL: usize = 8;
const F_MARKSMAN: usize = 9;
const F_DEFLECT: usize = 10;
const F_BACKSTAB: usize = 11;
const F_CHARGE: usize = 12;
const F_SWARM: usize = 13;
const F_DRAINS: usize = 14;
const F_PLAGUE: usize = 15;
const F_POISON: usize = 16;
const F_SLOW: usize = 17;
const F_PETRIFIES: usize = 18;
const F_FIRSTSTRIKE: usize = 19;
const F_BERSERK: usize = 20;

/// std::mt19937 (`combat.MTRng`): Knuth init, the standard twist and
/// tempering; `next_u32` is `mt_rng::get_next_random()`.
pub struct Mt19937 {
    mt: Vec<u32>,
    idx: usize,
    pub calls: u64,
}

impl Mt19937 {
    const N: usize = 624;
    const M: usize = 397;

    pub fn new(seed: u32, call_count: u64) -> Self {
        let mut mt = vec![0u32; Self::N];
        mt[0] = seed;
        for i in 1..Self::N {
            let prev = mt[i - 1];
            mt[i] = 1812433253u32
                .wrapping_mul(prev ^ (prev >> 30))
                .wrapping_add(i as u32);
        }
        let mut rng = Mt19937 { mt, idx: Self::N, calls: 0 };
        for _ in 0..call_count {
            rng.next_raw();
        }
        rng.calls = call_count;
        rng
    }

    fn twist(&mut self) {
        for i in 0..Self::N {
            let y = (self.mt[i] & 0x8000_0000) | (self.mt[(i + 1) % Self::N] & 0x7FFF_FFFF);
            let mut v = self.mt[(i + Self::M) % Self::N] ^ (y >> 1);
            if y & 1 != 0 {
                v ^= 0x9908_B0DF;
            }
            self.mt[i] = v;
        }
        self.idx = 0;
    }

    fn next_raw(&mut self) -> u32 {
        if self.idx >= Self::N {
            self.twist();
        }
        let mut y = self.mt[self.idx];
        self.idx += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9D2C_5680;
        y ^= (y << 15) & 0xEFC6_0000;
        y ^= y >> 18;
        y
    }

    pub fn next_u32(&mut self) -> u32 {
        self.calls += 1;
        self.next_raw()
    }
}

/// Where a fight's strike draws come from: the engine's generator, or a
/// script (`ScriptedRng`).
pub(crate) trait StrikeRng {
    fn draw(&mut self) -> u32;
    fn calls(&self) -> u64;
}

impl StrikeRng for Mt19937 {
    fn draw(&mut self) -> u32 {
        self.next_u32()
    }
    fn calls(&self) -> u64 {
        self.calls
    }
}

/// Draws forced by a prefix, true a hit (0) and false a miss (99), and a
/// hit past it (`swap_detector._EnumRNG`): a strike at chance 0 still
/// misses and one at 100 still hits.
pub(crate) struct ScriptedRng {
    prefix: Vec<bool>,
    calls: u64,
}

impl ScriptedRng {
    pub fn new(prefix: Vec<bool>) -> Self {
        ScriptedRng { prefix, calls: 0 }
    }
}

impl StrikeRng for ScriptedRng {
    fn draw(&mut self) -> u32 {
        let i = self.calls as usize;
        self.calls += 1;
        match self.prefix.get(i) {
            Some(false) => 99,
            _ => 0,
        }
    }
    fn calls(&self) -> u64 {
        self.calls
    }
}

/// `combat.round_damage`: the engine's rounding of base * bonus / divisor.
fn round_damage(base_damage: i64, bonus: i64, divisor: i64) -> i64 {
    if base_damage == 0 {
        return 0;
    }
    let rounding = if bonus < divisor || divisor == 1 { divisor / 2 } else { divisor / 2 - 1 };
    ((base_damage * bonus + rounding) / divisor).max(1)
}

/// `combat.combat_modifier`: percent points from alignment and time of day.
fn combat_modifier(alignment: i64, lawful_bonus: i64, fearless: bool) -> i64 {
    let bonus = match alignment {
        0 => lawful_bonus,
        1 => 0,
        2 => -lawful_bonus,
        3 => MAX_LIMINAL_BONUS - lawful_bonus.abs(),
        _ => 0,
    };
    if fearless {
        bonus.max(0)
    } else {
        bonus
    }
}

/// `combat.swarm_blows`.
fn swarm_blows(swarm_min: i64, swarm_max: i64, hp: i64, max_hp: i64) -> i64 {
    if hp >= max_hp {
        return swarm_max;
    }
    if swarm_max < swarm_min {
        swarm_min - (swarm_min - swarm_max) * hp / max_hp
    } else {
        swarm_min + (swarm_max - swarm_min) * hp / max_hp
    }
}

#[derive(Clone)]
pub(crate) struct Unit {
    pub hp: i64,
    pub max_hp: i64,
    pub level: i64,
    pub experience: i64,
    pub max_experience: i64,
    alignment: i64,
    defense_pct: i64,
    opp_resist: i64,
    w_damage: i64,
    w_number: i64,
    w_accuracy: i64,
    w_parry: i64,
    flags: [bool; UNIT_FLAGS],
}

impl Unit {
    pub fn slowed(&self) -> bool {
        self.flags[F_SLOWED]
    }
    pub fn poisoned(&self) -> bool {
        self.flags[F_POISONED]
    }
    pub fn petrified(&self) -> bool {
        self.flags[F_PETRIFIED]
    }
    pub fn undrainable(&self) -> bool {
        self.flags[F_UNDRAINABLE]
    }
    pub fn unpoisonable(&self) -> bool {
        self.flags[F_UNPOISONABLE]
    }

    fn from_arrays(ints: &[i64], flags: &[u8]) -> Self {
        let mut f = [false; UNIT_FLAGS];
        for (i, v) in flags.iter().enumerate() {
            f[i] = *v != 0;
        }
        Unit {
            hp: ints[HP],
            max_hp: ints[MAX_HP],
            level: ints[LEVEL],
            experience: ints[EXPERIENCE],
            max_experience: ints[MAX_EXPERIENCE],
            alignment: ints[ALIGNMENT],
            defense_pct: ints[DEFENSE_PCT],
            opp_resist: ints[OPP_RESIST],
            w_damage: ints[W_DAMAGE],
            w_number: ints[W_NUMBER],
            w_accuracy: ints[W_ACCURACY],
            w_parry: ints[W_PARRY],
            flags: f,
        }
    }
}

pub(crate) struct Stats {
    pub cth: i64,
    pub damage: i64,
    pub slow_damage: i64,
    pub n_attacks: i64,
    orig_attacks: i64,
    pub rounds: i64,
    pub firststrike: bool,
    pub drains: bool,
    plague: bool,
    pub poisons: bool,
    pub slows: bool,
    pub petrifies: bool,
}

/// The drain special's share of the damage done and its constant
/// (`combat._compute_battle_stats`: 50 and 0 for every drain weapon).
pub(crate) const DRAIN_PERCENT: i64 = 50;
pub(crate) const DRAIN_CONSTANT: i64 = 0;

/// What one fight is computed from: the two combatants' snapshots (the
/// module doc's layout) and what their surroundings give them.
pub(crate) struct FightInputs {
    pub a_ints: [i64; UNIT_INTS],
    pub a_flags: [u8; UNIT_FLAGS],
    pub d_ints: [i64; UNIT_INTS],
    pub d_flags: [u8; UNIT_FLAGS],
    pub d_has_weapon: bool,
    pub a_lawful_bonus: i64,
    pub d_lawful_bonus: i64,
    pub a_leadership_bonus: i64,
    pub d_leadership_bonus: i64,
    pub a_backstab_active: bool,
    pub d_backstab_active: bool,
}

/// The two combatants and their stats fixed at the fight's start (the
/// defender's None when it does not strike back).
pub(crate) type Fight = (Unit, Unit, Stats, Option<Stats>);

pub(crate) fn fight_stats(f: &FightInputs) -> Fight {
    let attacker = Unit::from_arrays(&f.a_ints, &f.a_flags);
    let defender = Unit::from_arrays(&f.d_ints, &f.d_flags);
    let a_stats = battle_stats(
        &attacker, &defender, f.d_has_weapon, f.a_lawful_bonus, f.a_leadership_bonus, true,
        f.a_backstab_active,
    );
    let d_stats = if f.d_has_weapon {
        Some(battle_stats(
            &defender, &attacker, true, f.d_lawful_bonus, f.d_leadership_bonus, false,
            f.d_backstab_active,
        ))
    } else {
        None
    };
    (attacker, defender, a_stats, d_stats)
}

/// `combat._compute_battle_stats` for one side. `has_weapon` says
/// whether this side strikes at all (the defender without a counter
/// weapon has no stats); `opp_has_weapon` whether the opponent's
/// weapon exists (parry, deflect, charge from it apply only then).
#[allow(clippy::too_many_arguments)]
fn battle_stats(
    me: &Unit,
    opp: &Unit,
    opp_has_weapon: bool,
    lawful_bonus: i64,
    leadership_bonus: i64,
    is_attacker: bool,
    backstab_active: bool,
) -> Stats {
    let f = &me.flags;
    let of = &opp.flags;
    let mut cth = if f[F_MAGICAL] {
        70
    } else {
        let mut c = opp.defense_pct + me.w_accuracy;
        if opp_has_weapon {
            c -= opp.w_parry;
        }
        if f[F_MARKSMAN] && is_attacker {
            c = c.max(60);
        }
        c
    };
    if is_attacker && opp_has_weapon && of[F_DEFLECT] {
        cth -= 10;
    }
    cth = cth.clamp(0, 100);
    if of[F_INVULNERABLE] {
        cth = 0;
    }
    let mut damage_multiplier = 100 + combat_modifier(me.alignment, lawful_bonus, f[F_FEARLESS]);
    damage_multiplier += leadership_bonus;
    let mut resist = me.opp_resist;
    if of[F_STEADFAST] && is_attacker {
        let base_bonus = 100 - resist;
        if 0 < base_bonus && base_bonus < 50 {
            let new_bonus = (base_bonus * 2).min(50);
            resist = 100 - new_bonus;
        }
    }
    let resist_mult = resist.max(0);
    let mut base_damage = me.w_damage;
    if backstab_active && f[F_BACKSTAB] && is_attacker {
        base_damage *= 2;
    }
    let charge_doubled =
        (f[F_CHARGE] && is_attacker) || (opp_has_weapon && of[F_CHARGE] && !is_attacker);
    if charge_doubled {
        base_damage *= 2;
    }
    damage_multiplier *= resist_mult;
    let damage = round_damage(base_damage, damage_multiplier, 10000);
    let slow_damage = round_damage(base_damage, damage_multiplier, 20000);
    let (swarm_min, swarm_max) = if f[F_SWARM] { (0, me.w_number) } else { (me.w_number, me.w_number) };
    let n_attacks = swarm_blows(swarm_min, swarm_max, me.hp, me.max_hp);
    Stats {
        cth,
        damage,
        slow_damage,
        n_attacks,
        orig_attacks: n_attacks,
        rounds: if f[F_BERSERK] { 30 } else { 1 },
        firststrike: f[F_FIRSTSTRIKE],
        drains: f[F_DRAINS],
        plague: f[F_PLAGUE],
        poisons: f[F_POISON],
        slows: f[F_SLOW],
        petrifies: f[F_PETRIFIES],
    }
}

/// `combat._perform_hit` and `_perform_hit_body`: one strike; false
/// ends the fight. Records (chance, hits, damage, dies) per strike.
fn perform_hit(
    striker: &mut Unit,
    target: &mut Unit,
    striker_stats: &mut Stats,
    target_stats: Option<&mut Stats>,
    rng: &mut dyn StrikeRng,
    record: &mut Vec<i64>,
) -> bool {
    striker_stats.n_attacks -= 1;
    let r = (rng.draw() % 100) as i64;
    let hits = r < striker_stats.cth;
    let dmg = if hits {
        if striker.flags[F_SLOWED] { striker_stats.slow_damage } else { striker_stats.damage }
    } else {
        0
    };
    record.extend_from_slice(&[striker_stats.cth, hits as i64, dmg, 0]);
    let cont = perform_hit_body(striker, target, striker_stats, target_stats, hits, dmg);
    let n = record.len();
    record[n - 1] = (target.hp <= 0) as i64;
    cont
}

fn perform_hit_body(
    striker: &mut Unit,
    target: &mut Unit,
    striker_stats: &mut Stats,
    target_stats: Option<&mut Stats>,
    hits: bool,
    dmg: i64,
) -> bool {
    if !hits || dmg <= 0 {
        return true;
    }
    let target_hp_pre = target.hp;
    target.hp -= dmg;
    if target.hp < 0 {
        target.hp = 0;
    }
    let damage_done = target_hp_pre - target.hp;
    if striker_stats.drains && damage_done > 0 && !target.flags[F_UNDRAINABLE] {
        let mut heal = damage_done * DRAIN_PERCENT / 100 + DRAIN_CONSTANT;
        if heal != 0 {
            heal = heal.min(striker.max_hp - striker.hp);
            heal = heal.max(1 - striker.hp);
            striker.hp += heal;
        }
    }
    if target.hp > 0 {
        if striker_stats.poisons && !target.flags[F_POISONED] && !target.flags[F_UNPOISONABLE] {
            target.flags[F_POISONED] = true;
        }
        if striker_stats.slows && !target.flags[F_SLOWED] {
            target.flags[F_SLOWED] = true;
        }
        if striker_stats.petrifies {
            target.flags[F_PETRIFIED] = true;
            striker_stats.n_attacks = 0;
            if let Some(ts) = target_stats {
                ts.n_attacks = -1;
            }
            return false;
        }
    }
    target.hp > 0
}

/// `combat.resolve_attack`. Returns
/// (a_hp, d_hp, a_xp, d_xp, a_slowed, a_poisoned, a_petrified,
///  d_slowed, d_poisoned, d_petrified, plague_spawned,
///  plague_spawned_attacker_died, rng_calls_used, strikes[n*4])
/// with strikes as (chance, hits, damage, dies) per strike in the
/// engine's checkup order.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn resolve_attack<'py>(
    py: Python<'py>,
    a_ints: PyReadonlyArray1<'py, i64>,
    a_flags: PyReadonlyArray1<'py, u8>,
    d_ints: PyReadonlyArray1<'py, i64>,
    d_flags: PyReadonlyArray1<'py, u8>,
    d_has_weapon: bool,
    a_lawful_bonus: i64,
    d_lawful_bonus: i64,
    a_leadership_bonus: i64,
    d_leadership_bonus: i64,
    a_backstab_active: bool,
    d_backstab_active: bool,
    seed: u32,
    call_count: u64,
) -> PyResult<(Vec<i64>, Bound<'py, PyArray1<i64>>)> {
    let (a_ints, a_flags, d_ints, d_flags) =
        (a_ints.as_slice()?, a_flags.as_slice()?, d_ints.as_slice()?, d_flags.as_slice()?);
    crate::check_lengths(
        &[
            ("a_ints", a_ints.len(), UNIT_INTS),
            ("d_ints", d_ints.len(), UNIT_INTS),
            ("a_flags", a_flags.len(), UNIT_FLAGS),
            ("d_flags", d_flags.len(), UNIT_FLAGS),
        ],
        || String::from("a combatant's snapshot, the module doc's layout"),
    )?;
    let inputs = FightInputs {
        a_ints: a_ints.try_into().expect("checked length"),
        a_flags: a_flags.try_into().expect("checked length"),
        d_ints: d_ints.try_into().expect("checked length"),
        d_flags: d_flags.try_into().expect("checked length"),
        d_has_weapon,
        a_lawful_bonus,
        d_lawful_bonus,
        a_leadership_bonus,
        d_leadership_bonus,
        a_backstab_active,
        d_backstab_active,
    };
    let (out, record) = resolve_fight(&inputs, seed, call_count);
    Ok((out, record.into_pyarray(py)))
}

/// The fight with the engine's generator seeded as the replay's
/// [random_seed] seeds it, `call_count` draws in.
pub(crate) fn resolve_fight(inputs: &FightInputs, seed: u32, call_count: u64) -> (Vec<i64>, Vec<i64>) {
    resolve_fight_with(inputs, &mut Mt19937::new(seed, call_count))
}

/// The fight: the 13 outputs of `resolve_attack` and the strike record,
/// four integers per strike (chance, hit, damage, dies). The core's
/// attack kernel (core_attack.rs) calls this directly.
pub(crate) fn resolve_fight_with(inputs: &FightInputs, rng: &mut dyn StrikeRng) -> (Vec<i64>, Vec<i64>) {
    let (mut attacker, mut defender, mut a_stats, mut d_stats) = fight_stats(inputs);
    let starting_calls = rng.calls();
    let defender_first_of = |a: &Stats, d: &Option<Stats>| -> bool {
        d.as_ref().is_some_and(|ds| ds.firststrike && !a.firststrike)
    };
    let mut defender_first = defender_first_of(&a_stats, &d_stats);
    let mut rounds_left = a_stats.rounds.max(d_stats.as_ref().map_or(1, |d| d.rounds)) - 1;
    let mut plague_spawned = false;
    let mut record: Vec<i64> = Vec::new();
    loop {
        if !defender_first && a_stats.n_attacks > 0
            && !perform_hit(
                &mut attacker, &mut defender, &mut a_stats, d_stats.as_mut(), rng,
                &mut record,
            )
        {
            break;
        }
        defender_first = false;
        if let Some(ds) = d_stats.as_mut() {
            if ds.n_attacks > 0
                && !perform_hit(
                    &mut defender, &mut attacker, ds, Some(&mut a_stats), rng, &mut record,
                )
            {
                break;
            }
        }
        let d_done = d_stats.as_ref().is_none_or(|d| d.n_attacks == 0);
        if rounds_left > 0 && a_stats.n_attacks == 0 && d_done {
            a_stats.n_attacks = a_stats.orig_attacks;
            if let Some(ds) = d_stats.as_mut() {
                ds.n_attacks = ds.orig_attacks;
            }
            rounds_left -= 1;
            defender_first = defender_first_of(&a_stats, &d_stats);
            continue;
        }
        let d_spent = d_stats.as_ref().is_none_or(|d| d.n_attacks <= 0);
        if a_stats.n_attacks <= 0 && d_spent {
            break;
        }
    }
    let mut a_xp_gain = COMBAT_EXPERIENCE * defender.level;
    let mut d_xp_gain = COMBAT_EXPERIENCE * attacker.level;
    if defender.hp <= 0 {
        defender.hp = 0;
        a_xp_gain = if defender.level != 0 { KILL_EXPERIENCE * defender.level } else { KILL_EXPERIENCE / 2 };
        if a_stats.plague {
            plague_spawned = true;
        }
    }
    let mut plague_spawned_attacker_died = false;
    if attacker.hp <= 0 {
        attacker.hp = 0;
        d_xp_gain = if attacker.level != 0 { KILL_EXPERIENCE * attacker.level } else { KILL_EXPERIENCE / 2 };
        if d_stats.as_ref().is_some_and(|d| d.plague) {
            plague_spawned_attacker_died = true;
        }
    }
    if attacker.hp > 0 {
        attacker.experience += a_xp_gain;
    }
    if defender.hp > 0 {
        defender.experience += d_xp_gain;
    }
    let out = vec![
        attacker.hp,
        defender.hp,
        attacker.experience,
        defender.experience,
        attacker.flags[F_SLOWED] as i64,
        attacker.flags[F_POISONED] as i64,
        attacker.flags[F_PETRIFIED] as i64,
        defender.flags[F_SLOWED] as i64,
        defender.flags[F_POISONED] as i64,
        defender.flags[F_PETRIFIED] as i64,
        plague_spawned as i64,
        plague_spawned_attacker_died as i64,
        (rng.calls() - starting_calls) as i64,
    ];
    (out, record)
}

/// `MTRng.get_random_int(low, high)` after `call_count` draws: the
/// uniform draw the advancement choice and every other synced random
/// takes (`replay_dataset._draw_uniform_advance`).
#[pyfunction]
pub fn random_int(seed: u32, call_count: u64, low: i64, high: i64) -> i64 {
    let mut rng = Mt19937::new(seed, call_count);
    let span = high - low + 1;
    low + (rng.next_u32() as i64) % span
}
