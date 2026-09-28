//! The `[effect]` applier (`scenario_events._apply_effect_to_unit`): one
//! effect of an [object] or a custom trait on one unit. The apply_to
//! forms the ladder and mini scenarios use are modelled; display-only
//! ones do nothing; any other is reported once (`drain_warnings`) and
//! dropped. The Python applier is the oracle (tests/test_rust_units.py).

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashSet;
use std::sync::Mutex;

use crate::core::{sorted, AttackRec, UnitRec};
use crate::db::DAMAGE_TYPES;
use crate::units::div100rounded;
use crate::wml::{clean, Wml};

/// `_COSMETIC_APPLY_TO`: display only.
const COSMETIC_APPLY_TO: [&str; 10] = [
    "ellipse", "image_mod", "overlay", "profile", "new_animation", "halo", "portrait", "small_profile",
    "description", "usage",
];

static WARNINGS: Mutex<Vec<String>> = Mutex::new(Vec::new());
static WARNED: Mutex<Option<HashSet<String>>> = Mutex::new(None);

/// Record a warning once per distinct text; Python logs them
/// (`drain_warnings`).
pub fn warn_once(text: String) {
    let mut seen = WARNED.lock().unwrap();
    let set = seen.get_or_insert_with(HashSet::new);
    if set.insert(text.clone()) {
        WARNINGS.lock().unwrap().push(text);
    }
}

/// The warnings recorded since the last call.
#[pyfunction]
pub fn drain_warnings() -> Vec<String> {
    std::mem::take(&mut *WARNINGS.lock().unwrap())
}

fn warn_unmodelled_apply_to(apply_to: &str) {
    warn_once(format!(
        "[effect] apply_to={apply_to:?} is not modelled; the effect is dropped. Scenario behaviour will \
         differ from Wesnoth."));
}

/// Python's `int(s)` for a decimal string: surrounding whitespace, a
/// sign, single underscores between digits.
pub fn py_int(s: &str) -> Option<i64> {
    let t = s.trim();
    let (neg, digits) = match t.as_bytes().first() {
        Some(b'-') => (true, &t[1..]),
        Some(b'+') => (false, &t[1..]),
        _ => (false, t),
    };
    if digits.is_empty() || digits.starts_with('_') || digits.ends_with('_') || digits.contains("__") {
        return None;
    }
    let mut v: i64 = 0;
    for c in digits.chars() {
        if c == '_' {
            continue;
        }
        let d = c.to_digit(10)? as i64;
        v = v.checked_mul(10)?.checked_add(d)?;
    }
    Some(if neg { -v } else { v })
}

/// `_parse_increase(raw, base)`: a plain integer, or a percentage of
/// `base` through div100rounded; 0 when malformed.
pub fn parse_increase(raw: &str, base: i64) -> i64 {
    let s = clean(raw);
    if s.is_empty() {
        return 0;
    }
    if let Some(pct) = s.strip_suffix('%') {
        return match py_int(pct) {
            Some(p) => div100rounded(base * p),
            None => 0,
        };
    }
    py_int(&s).unwrap_or(0)
}

/// `_effect_member_ids`: each child's `id=`, its tag when it has none.
pub fn member_ids(container: Option<&Wml>) -> Vec<String> {
    let mut out = Vec::new();
    if let Some(c) = container {
        for ch in &c.children {
            let raw = clean(ch.attr("id").unwrap_or(""));
            out.push(if raw.is_empty() { ch.tag.clone() } else { raw });
        }
    }
    sorted(out)
}

fn truthy(s: &str) -> bool {
    matches!(s.trim().to_lowercase().as_str(), "yes" | "true" | "1")
}

/// `(eff.attrs.get(key, "") or default).strip().strip('"')`.
fn clean_or(eff: &Wml, key: &str, default: &str) -> String {
    match eff.attr(key) {
        Some(v) if !v.is_empty() => clean(v),
        _ => clean(default),
    }
}

/// Apply one `[effect]` to `u` in place.
pub fn apply_effect(u: &mut UnitRec, eff: &Wml) {
    let apply_to = eff.clean("apply_to");
    match apply_to.as_str() {
        "attack" => {
            let range = eff.clean("range");
            let new_specials = member_ids(eff.first("set_specials").map(|n| n.as_ref()));
            let inc_attacks = eff.attr("increase_attacks").unwrap_or("").to_string();
            let inc_damage = eff.attr("increase_damage").unwrap_or("").to_string();
            for a in u.attacks.iter_mut() {
                let matches = match range.as_str() {
                    "ranged" => a.ranged,
                    "melee" => !a.ranged,
                    _ => true,
                };
                if !matches {
                    continue;
                }
                a.damage = (a.damage + parse_increase(&inc_damage, a.damage)).max(0);
                a.strikes = (a.strikes + parse_increase(&inc_attacks, a.strikes)).max(0);
                let mut specials = a.specials.clone();
                specials.extend(new_specials.iter().cloned());
                a.specials = sorted(specials);
            }
        }
        "new_attack" => {
            let range = clean_or(eff, "range", "melee");
            let wtype = clean_or(eff, "type", "blade");
            let damage = py_int(&clean_or(eff, "damage", "0")).unwrap_or(0);
            let number = py_int(&clean_or(eff, "number", "1")).unwrap_or(1);
            let container = eff.first("specials").or_else(|| eff.first("set_specials"));
            let type_id = DAMAGE_TYPES.iter().position(|d| *d == wtype.to_lowercase()).unwrap_or(0) as i64;
            u.attacks.push(AttackRec {
                type_id, strikes: number, damage, ranged: range == "ranged",
                specials: member_ids(container.map(|n| n.as_ref())),
            });
        }
        "remove_attacks" => u.attacks.clear(),
        "hitpoints" => {
            let inc = eff.attr("increase_total").unwrap_or("");
            if !inc.is_empty() {
                let delta = parse_increase(inc, u.max_hp);
                let new_max = (u.max_hp + delta).max(1);
                u.current_hp = (u.current_hp + delta).min(new_max).max(1);
                u.max_hp = new_max;
            }
            let set = eff.attr("set").unwrap_or("");
            if !set.is_empty() {
                if let Some(v) = py_int(set) {
                    u.current_hp = v.min(u.max_hp).max(1);
                }
            }
            if truthy(eff.attr("heal_full").unwrap_or("")) {
                u.current_hp = u.max_hp;
            }
        }
        "movement" => {
            let set = eff.attr("set").unwrap_or("");
            let inc = eff.attr("increase").unwrap_or("");
            if !set.is_empty() {
                if let Some(v) = py_int(set) {
                    u.max_moves = v.max(0);
                    u.current_moves = u.current_moves.min(u.max_moves);
                }
            } else if !inc.is_empty() {
                u.max_moves = (u.max_moves + parse_increase(inc, u.max_moves)).max(0);
                u.current_moves = u.current_moves.min(u.max_moves);
            }
        }
        "status" => {
            let add = eff.clean("add");
            let rem = eff.clean("remove");
            if !add.is_empty() {
                u.add_status(&add);
            }
            if !rem.is_empty() {
                u.drop_status(&rem);
            }
        }
        "new_ability" | "remove_ability" => {
            let ids = member_ids(eff.first("abilities").map(|n| n.as_ref()));
            if !ids.is_empty() {
                if apply_to == "new_ability" {
                    let mut have = u.abilities.clone();
                    have.extend(ids);
                    u.abilities = sorted(have);
                } else {
                    u.abilities.retain(|a| !ids.contains(a));
                }
            }
        }
        "movement_costs" => {
            // Only a unit that moves needs its costs: the pool's carriers
            // are the minis' neutral Tentacles, which never move.
            if u.side == 1 || u.side == 2 {
                warn_unmodelled_apply_to(&apply_to);
            }
        }
        other => {
            if !other.is_empty() && !COSMETIC_APPLY_TO.contains(&other) {
                warn_unmodelled_apply_to(other);
            }
        }
    }
}

/// `_apply_effect_to_unit` on a unit field dict (the differential
/// tests' handle): the fields after the effect.
#[pyfunction]
pub fn apply_effect_fields<'py>(py: Python<'py>, fields: &Bound<'py, PyDict>, effect: Wml)
    -> PyResult<Bound<'py, PyDict>> {
    let mut u = crate::core::unit_from_dict(fields)?;
    apply_effect(&mut u, &effect);
    crate::core::unit_dict(py, &u)
}
