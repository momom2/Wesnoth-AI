//! Unit construction: a unit of a type (`replay_dataset._build_unit`), a
//! recruit with the trait roll its seed gives (`_build_recruit_unit`),
//! the Walking Corpse a plague kill raises (`_build_plague_corpse`), and
//! the traits themselves (`tools/traits.py`: the roll, then the effects
//! applied in roll order). The Python builders are the oracle
//! (tests/test_rust_units.py) until the port's certification.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use sha2::{Digest, Sha256};
use std::sync::Arc;

use crate::combat::Mt19937;
use crate::core::{sorted, unit_dict, AttackRec, DefTable, UnitRec};
use crate::db::{unit_db, UnitDb, UnitType, DAMAGE_TYPES};

/// The terrain keys of `Unit.defenses` (`_build_unit`'s list order).
const DEFENSE_KEYS: [&str; 14] = [
    "castle", "cave", "deep_water", "flat", "forest", "frozen", "fungus", "hills", "mountains", "reef",
    "sand", "shallow_water", "swamp_water", "village",
];

/// `utils::div100rounded` as `apply_modifier` uses it: round half away
/// from zero.
pub fn div100rounded(raw: i64) -> i64 {
    if raw < 0 { -((-raw + 50) / 100) } else { (raw + 50) / 100 }
}

/// `unit_type::experience_needed(true)`: the scaled cap, at least 1.
pub fn scaled_max_exp(base: i64, modifier: i64) -> i64 {
    ((base * modifier + 50).div_euclid(100)).max(1)
}

/// Python's `int(s, 16) & 0xFFFFFFFF`, 42 when `s` is not hex
/// (`combat.seed_int_of`): surrounding whitespace, a sign, an `0x`
/// prefix and single underscores between digits are accepted.
pub fn seed_int_of(s: &str) -> u32 {
    let t = s.trim();
    let (neg, t) = match t.as_bytes().first() {
        Some(b'-') => (true, &t[1..]),
        Some(b'+') => (false, &t[1..]),
        _ => (false, t),
    };
    let t = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")).map(|r| r.strip_prefix('_').unwrap_or(r))
        .unwrap_or(t);
    if t.is_empty() || t.starts_with('_') || t.ends_with('_') || t.contains("__") {
        return 42;
    }
    let mut v: u32 = 0;
    for c in t.chars() {
        if c == '_' {
            continue;
        }
        match c.to_digit(16) {
            Some(d) => v = v.wrapping_mul(16).wrapping_add(d),
            None => return 42,
        }
    }
    if neg { v.wrapping_neg() } else { v }
}

/// The first `n` hex digits of the SHA-256 of `text` as an integer.
pub fn sha256_prefix(text: &str, n_hex: usize) -> u64 {
    let digest = Sha256::digest(text.as_bytes());
    let mut v: u64 = 0;
    for k in 0..n_hex {
        let byte = digest[k / 2];
        let nib = if k % 2 == 0 { byte >> 4 } else { byte & 0x0F };
        v = (v << 4) | nib as u64;
    }
    v
}

/// The hex string of the first `n` digits of the SHA-256 of `text`.
pub fn sha256_hex(text: &str, n_hex: usize) -> String {
    format!("{:0width$x}", sha256_prefix(text, n_hex), width = n_hex)
}

// ---------------------------------------------------------------------
// Traits (tools/traits.py)
// ---------------------------------------------------------------------

/// `traits.TraitEffect`.
#[derive(Default)]
pub struct TraitEffect {
    pub hp_delta: i64,
    pub hp_per_level: i64,
    pub hp_pct: i64,
    pub melee_dmg_delta: i64,
    pub ranged_dmg_delta: i64,
    pub movement_delta: i64,
    pub xp_pct: i64,
    pub statuses: &'static [&'static str],
    pub defense_overrides: &'static [(&'static str, i64)],
}

const UNLIVING: &[&str] = &["unpoisonable", "undrainable", "unplagueable"];

/// `traits.TRAITS`.
pub fn trait_effect(id: &str) -> Option<TraitEffect> {
    let d = TraitEffect::default();
    Some(match id {
        "strong" => TraitEffect { hp_delta: 1, melee_dmg_delta: 1, ..d },
        "dextrous" => TraitEffect { ranged_dmg_delta: 1, ..d },
        "quick" => TraitEffect { movement_delta: 1, hp_pct: -5, ..d },
        "intelligent" => TraitEffect { xp_pct: -20, ..d },
        "resilient" => TraitEffect { hp_delta: 4, hp_per_level: 1, ..d },
        "healthy" => TraitEffect { hp_delta: 1, hp_per_level: 1, ..d },
        "fearless" | "loyal" => d,
        "feral" => TraitEffect { defense_overrides: &[("village", -50)], ..d },
        "weak" => TraitEffect { hp_delta: -1, melee_dmg_delta: -1, ..d },
        "slow" => TraitEffect { movement_delta: -1, hp_pct: 5, ..d },
        "dim" => TraitEffect { xp_pct: 20, ..d },
        "aged" => TraitEffect { movement_delta: -1, hp_delta: -8, ..d },
        "undead" | "mechanical" | "elemental" => TraitEffect { statuses: UNLIVING, ..d },
        _ => return None,
    })
}

const DEFAULT_POOL: &[&str] = &["strong", "quick", "intelligent", "resilient"];

/// `traits.RACE_POOLS` (a race it lacks draws from the default pool).
fn race_pool(race: &str) -> Vec<String> {
    let pool: &[&str] = match race {
        "elf" => &["strong", "quick", "intelligent", "resilient", "dextrous"],
        "dwarf" => &["strong", "quick", "intelligent", "resilient", "healthy"],
        "goblin" => &["weak", "slow", "dim", "fearless"],
        "undead" | "mechanical" | "elemental" | "ship" | "fake" => &[],
        _ => DEFAULT_POOL,
    };
    pool.iter().map(|s| s.to_string()).collect()
}

/// `traits.UNIT_MUSTHAVE`, then `traits.RACE_MUSTHAVE`.
fn fallback_musthave(unit_type: &str, race: &str) -> Vec<String> {
    let must: &[&str] = match (unit_type, race) {
        ("Necrophage", _) => &["fearless", "undead"],
        (_, "undead") => &["undead"],
        (_, "mechanical") => &["mechanical"],
        (_, "elemental") => &["elemental"],
        _ => &[],
    };
    must.iter().map(|s| s.to_string()).collect()
}

/// `traits.roll_traits`: the musthaves, then (not for a leader) as many
/// draws from the pool as the type's trait count leaves, each over the
/// pool's traits not yet taken (duplicates weight the draw). The draws
/// come from the recruit's seed (one draw spent first on the gender of
/// a type with two), or from a hash of `seed_token` for a record without
/// one. A leader draws nothing and gets quick when its type has 4 moves.
#[allow(clippy::too_many_arguments)]
pub fn roll_traits(t: &UnitType, seed_hex: &str, seed_token: &str, is_leader: bool) -> Vec<String> {
    let (must, pool, target_total) = match &t.traits {
        Some(info) => (info.musthave.clone(), info.pool.clone(), info.num_traits),
        None => (fallback_musthave(&t.name, &t.race), race_pool(&t.race), 2),
    };
    let mut out = must;
    if is_leader {
        if t.moves == 4 && !out.iter().any(|x| x == "quick") {
            out.push("quick".to_string());
        }
        return out;
    }
    let n_random = (target_total - out.len() as i64).max(0);
    if n_random == 0 || pool.is_empty() {
        return out;
    }
    let candidates = |out: &Vec<String>| -> Vec<String> {
        pool.iter().filter(|p| !out.contains(p)).cloned().collect()
    };
    if !seed_hex.is_empty() {
        let mut rng = Mt19937::new(seed_int_of(seed_hex), 0);
        if t.n_genders > 1 {
            rng.next_u32();
        }
        for _ in 0..n_random {
            let c = candidates(&out);
            if c.is_empty() {
                break;
            }
            let idx = rng.next_u32() as usize % c.len();
            out.push(c[idx].clone());
        }
    } else {
        let mut h = sha256_prefix(seed_token, 16);
        for _ in 0..n_random {
            let c = candidates(&out);
            if c.is_empty() {
                break;
            }
            out.push(c[(h % c.len() as u64) as usize].clone());
            h /= (c.len() as u64 + 1).max(1);
        }
    }
    out
}

/// `traits.apply_traits_to_unit`: each known trait's effects in order on
/// the running maxima (percentages through div100rounded), its statuses
/// and defense overrides; the current hit points follow a unit at full
/// health, the current moves are capped at the new maximum.
pub fn apply_traits(u: &mut UnitRec, trait_ids: &[String], level: i64, table: &mut DefTable) {
    let (old_max_hp, old_hp) = (u.max_hp, u.current_hp);
    let mut max_hp = u.max_hp;
    let mut max_moves = u.max_moves;
    let mut max_xp = u.max_exp;
    for tid in trait_ids {
        let eff = match trait_effect(tid) {
            Some(e) => e,
            None => continue,
        };
        if !u.traits.contains(tid) {
            u.traits.push(tid.clone());
        }
        max_hp += eff.hp_delta;
        max_hp += eff.hp_per_level * level;
        if eff.hp_pct != 0 {
            max_hp += div100rounded(max_hp * eff.hp_pct);
        }
        max_moves += eff.movement_delta;
        if eff.xp_pct != 0 {
            max_xp += div100rounded(max_xp * eff.xp_pct);
            max_xp = max_xp.max(1);
        }
        if eff.melee_dmg_delta != 0 || eff.ranged_dmg_delta != 0 {
            for a in u.attacks.iter_mut() {
                let bump = if a.ranged { eff.ranged_dmg_delta } else { eff.melee_dmg_delta };
                if bump != 0 {
                    a.damage = (a.damage + bump).max(1);
                }
            }
        }
        for s in eff.statuses {
            u.add_status(s);
        }
        for (terrain, cth) in eff.defense_overrides {
            match table.iter_mut().find(|(k, _)| k == terrain) {
                Some(e) => e.1 = *cth,
                None => table.push((terrain.to_string(), *cth)),
            }
        }
    }
    u.traits.sort();
    u.traits.dedup();
    u.current_hp = if old_hp == old_max_hp { max_hp } else { old_hp };
    u.max_hp = max_hp;
    u.current_moves = u.current_moves.min(max_moves);
    u.max_moves = max_moves;
    u.max_exp = max_xp;
}

// ---------------------------------------------------------------------
// Builders
// ---------------------------------------------------------------------

/// `_attacks_from_stats`: the type's attacks with no specials of the
/// unit's own; a type without one gets a 1x1 blade.
pub fn attacks_from_type(t: &UnitType) -> Vec<AttackRec> {
    let mut out: Vec<AttackRec> = t.attacks.iter().map(|a| AttackRec {
        type_id: DAMAGE_TYPES.iter().position(|d| *d == a.type_name).unwrap_or(0) as i64,
        strikes: a.number,
        damage: a.damage,
        ranged: a.ranged,
        specials: Vec::new(),
    }).collect();
    if out.is_empty() {
        out.push(AttackRec { type_id: 0, strikes: 1, damage: 1, ranged: false, specials: Vec::new() });
    }
    out
}

pub fn resistances_of(t: &UnitType) -> Vec<f64> {
    t.resist.iter().map(|&r| r as f64 / 100.0).collect()
}

pub fn defenses_of(t: &UnitType) -> Vec<f64> {
    DEFENSE_KEYS.iter().map(|k| t.defense_of(k).unwrap_or(50) as f64 / 100.0).collect()
}

/// What `_build_unit` reads from its record dict.
#[derive(Clone, Debug, Default)]
pub struct UnitSpec {
    pub uid: i64,
    pub unit_type: String,
    pub side: i64,
    pub x: i64,
    pub y: i64,
    pub is_leader: bool,
    pub max_hp: Option<i64>,
    pub max_moves: Option<i64>,
    pub max_exp: Option<i64>,
    pub cost: Option<i64>,
    pub hp: Option<i64>,
    pub petrified: bool,
}

/// A record not placed on any core yet: no type index, hex or class.
fn blank(id: String, name: String) -> UnitRec {
    UnitRec {
        id, name, type_idx: -1, name_id: 0, side: 0, is_leader: false, x: 0, y: 0, hex: -1, max_hp: 0,
        max_moves: 0, max_exp: 0, cost: 0, alignment: 1, levelup_names: Vec::new(), current_hp: 0,
        current_moves: 0, current_exp: 0, has_attacked: false, attacks: Vec::new(), resistances: Vec::new(),
        defenses: Vec::new(), movement_costs: vec![1; 14], abilities: Vec::new(), traits: Vec::new(),
        statuses: Vec::new(), class_id: -1, class_slowed_id: -1, def_table: None, pickadvance: None,
        feeding_count: None, trait_order: None, object_effects: Vec::new(), wml_role: None, ai_guardian: false,
    }
}

/// `_build_unit`: a unit of the record's type, the record's own numbers
/// where it has them; a petrified statue has no moves and no attacks.
/// With `leader_traits`, a leader gets its musthaves and the 4-move
/// quick, refreshed to full moves and hit points after
/// (`quick_4mp_leaders`, data/multiplayer/eras.lua:18-19).
pub fn build_unit(db: &UnitDb, spec: &UnitSpec, leader_traits: bool, game_id: &str, exp_modifier: i64) -> UnitRec {
    let t = db.get(&spec.unit_type);
    let max_hp = spec.max_hp.unwrap_or(t.hitpoints);
    let max_moves = spec.max_moves.unwrap_or(t.moves);
    let petrified = spec.petrified;
    let mut u = blank(format!("u{}", spec.uid), spec.unit_type.clone());
    u.side = spec.side;
    u.is_leader = spec.is_leader;
    u.x = spec.x;
    u.y = spec.y;
    u.max_hp = max_hp;
    u.max_moves = if petrified { 0 } else { max_moves };
    u.max_exp = spec.max_exp.unwrap_or_else(|| scaled_max_exp(t.experience, exp_modifier));
    u.cost = spec.cost.unwrap_or(t.cost);
    u.alignment = t.alignment;
    u.levelup_names = t.advances_to.clone();
    u.current_hp = spec.hp.unwrap_or(max_hp);
    u.current_moves = if petrified { 0 } else { max_moves };
    u.has_attacked = petrified;
    u.attacks = if petrified { Vec::new() } else { attacks_from_type(&t) };
    u.resistances = resistances_of(&t);
    u.defenses = defenses_of(&t);
    u.abilities = if petrified { Vec::new() } else { sorted(t.abilities.clone()) };
    if petrified {
        u.statuses = vec!["petrified".to_string()];
    }
    let mut table: DefTable = t.defense.clone();
    if leader_traits && spec.is_leader {
        let token = format!("{}:leader{}:{}", game_id, spec.side, spec.unit_type);
        let ids = roll_traits(&t, "", &token, true);
        if !ids.is_empty() {
            apply_traits(&mut u, &ids, t.level, &mut table);
            u.current_moves = u.max_moves;
            u.current_hp = u.max_hp;
        }
    }
    u.def_table = Some(Arc::new(table));
    u
}

/// `_build_recruit_unit`: a fresh unit of the type with the traits its
/// seed rolls, in roll order (kept for advancement).
pub fn build_recruit_unit(db: &UnitDb, unit_type: &str, side: i64, x: i64, y: i64, next_uid: i64, game_id: &str,
                          seed_hex: &str, exp_modifier: i64) -> UnitRec {
    let spec = UnitSpec { uid: next_uid, unit_type: unit_type.to_string(), side, x, y, ..Default::default() };
    let mut u = build_unit(db, &spec, false, game_id, exp_modifier);
    let t = db.get(unit_type);
    let token = format!("{}:u{}:{}", game_id, next_uid, unit_type);
    let ids = roll_traits(&t, seed_hex, &token, false);
    let mut table: DefTable = match &u.def_table {
        Some(tab) if !tab.is_empty() => tab.as_ref().clone(),
        _ => t.defense.clone(),
    };
    apply_traits(&mut u, &ids, t.level, &mut table);
    u.def_table = Some(Arc::new(table));
    u.trait_order = Some(ids);
    u
}

/// `_build_plague_corpse`: the Walking Corpse of the victim's undead
/// variation (its type's, else its race's), a variation the database
/// lacks falling back to the plain corpse; tagged with the variation,
/// unable to act this turn.
pub fn build_plague_corpse(db: &UnitDb, dead_name: &str, side: i64, x: i64, y: i64, next_uid: i64, game_id: &str,
                           exp_modifier: i64) -> UnitRec {
    let dead = db.get(dead_name);
    let mut variation = dead.undead_variation.trim().to_string();
    if variation.is_empty() {
        let race = dead.race.trim();
        if !race.is_empty() {
            variation = db.race_undead_variation(race).trim().to_string();
        }
    }
    let base = "Walking Corpse";
    let mut spawn = if variation.is_empty() { base.to_string() } else { format!("{base}:{variation}") };
    if !variation.is_empty() && !db.contains(&spawn) {
        spawn = base.to_string();
    }
    let mut u = build_recruit_unit(db, &spawn, side, x, y, next_uid, game_id, "", exp_modifier);
    if !variation.is_empty() {
        u.add_status(&format!("variation:{variation}"));
    }
    u.current_moves = 0;
    u.has_attacked = true;
    u
}

/// The number of a `u<digits>` id (`u.id[1:].isdigit()`), else None.
pub fn numeric_uid(id: &str) -> Option<i64> {
    let rest = id.strip_prefix('u')?;
    if rest.is_empty() || !rest.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    rest.parse().ok()
}

// ---------------------------------------------------------------------
// Python entry points (the differential tests' handles)
// ---------------------------------------------------------------------

fn spec_of(d: &Bound<'_, PyDict>) -> PyResult<UnitSpec> {
    fn opt<'py, T: FromPyObject<'py>>(d: &Bound<'py, PyDict>, k: &str) -> PyResult<Option<T>> {
        match d.get_item(k)? {
            Some(v) if !v.is_none() => Ok(Some(v.extract()?)),
            _ => Ok(None),
        }
    }
    Ok(UnitSpec {
        uid: opt(d, "uid")?.unwrap_or(0),
        unit_type: opt(d, "type")?.unwrap_or_default(),
        side: opt(d, "side")?.unwrap_or(0),
        x: opt(d, "x")?.unwrap_or(0),
        y: opt(d, "y")?.unwrap_or(0),
        is_leader: opt(d, "is_leader")?.unwrap_or(false),
        max_hp: opt(d, "max_hp")?,
        max_moves: opt(d, "max_moves")?,
        max_exp: opt(d, "max_exp")?,
        cost: opt(d, "cost")?,
        hp: opt(d, "hp")?,
        petrified: opt(d, "petrified")?.unwrap_or(false),
    })
}

/// `_build_unit(record, apply_leader_traits, game_id, exp_modifier)` as
/// a unit field dict (`game_core.unit_fields` form).
#[pyfunction]
#[pyo3(signature = (record, apply_leader_traits=false, game_id="", exp_modifier=100))]
pub fn build_unit_fields<'py>(py: Python<'py>, record: &Bound<'py, PyDict>, apply_leader_traits: bool,
                              game_id: &str, exp_modifier: i64) -> PyResult<Bound<'py, PyDict>> {
    let db = unit_db()?;
    unit_dict(py, &build_unit(&db, &spec_of(record)?, apply_leader_traits, game_id, exp_modifier))
}

#[pyfunction]
#[pyo3(signature = (unit_type, side, x, y, next_uid, game_id="", trait_seed_hex="", exp_modifier=100))]
#[allow(clippy::too_many_arguments)]
pub fn build_recruit_fields<'py>(py: Python<'py>, unit_type: &str, side: i64, x: i64, y: i64, next_uid: i64,
                                 game_id: &str, trait_seed_hex: &str, exp_modifier: i64)
    -> PyResult<Bound<'py, PyDict>> {
    let db = unit_db()?;
    unit_dict(py, &build_recruit_unit(&db, unit_type, side, x, y, next_uid, game_id, trait_seed_hex, exp_modifier))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn build_corpse_fields<'py>(py: Python<'py>, dead_name: &str, side: i64, x: i64, y: i64, next_uid: i64,
                                game_id: &str, exp_modifier: i64) -> PyResult<Bound<'py, PyDict>> {
    let db = unit_db()?;
    unit_dict(py, &build_plague_corpse(&db, dead_name, side, x, y, next_uid, game_id, exp_modifier))
}

/// `traits.roll_traits` for a type (its trait info and gender count).
#[pyfunction]
#[pyo3(signature = (unit_type, seed_hex="", seed_token="", is_leader=false))]
pub fn roll_type_traits(unit_type: &str, seed_hex: &str, seed_token: &str, is_leader: bool) -> PyResult<Vec<String>> {
    let db = unit_db()?;
    Ok(roll_traits(&db.get(unit_type), seed_hex, seed_token, is_leader))
}

#[pyfunction]
pub fn seed_int(seed_hex: &str) -> u32 {
    seed_int_of(seed_hex)
}
