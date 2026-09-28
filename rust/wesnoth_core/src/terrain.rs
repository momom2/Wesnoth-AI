//! The terrain resolver (`wesnoth_ai.rules.terrain_resolver`): movement
//! cost and defense of a terrain code through the engine's alias graph
//! (movetype.cpp:276-369 calc_value; terrain.cpp:208-244 and 334-377 for
//! the composite codes), healing, the light bonus, the hide-ability
//! cover, the terrain classes a code belongs to, and the one-class view
//! the hex set keeps (`replay_dataset._parse_hex_code`). The Python
//! module is the oracle (tests/test_rust_terrain.py).

use pyo3::prelude::*;

use crate::db::{terrain_db, TerrainDb};

pub const UNREACHABLE_COST: i64 = 99;
const MARKER_PLUS: &str = "+";
const MARKER_MINUS: &str = "-";
const MARKER_BASE: &str = "_bas";

/// `classes.Terrain` values.
pub mod terrain_class {
    pub const CASTLE: i64 = 0;
    pub const CAVE: i64 = 1;
    pub const DEEPWATER: i64 = 2;
    pub const FLAT: i64 = 3;
    pub const FOREST: i64 = 4;
    pub const FROZEN: i64 = 5;
    pub const HILLS: i64 = 6;
    pub const IMPASSABLE: i64 = 7;
    pub const MOUNTAINS: i64 = 8;
    pub const SAND: i64 = 9;
    pub const SHALLOWWATER: i64 = 10;
    pub const SWAMP: i64 = 11;
    pub const UNWALKABLE: i64 = 12;
    pub const VILLAGE: i64 = 13;
}

/// `classes.TerrainModifiers` bits of a hex (1 << value); a code never
/// yields VILLAGE (bit 0), only the live bridge's states carry it.
pub const MOD_KEEP: u8 = 1 << 1;
pub const MOD_CASTLE: u8 = 1 << 2;

/// "1 Gg^Fp" -> "Gg^Fp": the text after the last space of the trimmed
/// cell (`terrain_resolver.split_start_position`).
pub fn strip_start_position(cell: &str) -> &str {
    let t = cell.trim();
    match t.rfind(' ') {
        Some(k) => &t[k + 1..],
        None => t,
    }
}

/// `merge_alias_lists`: `second` spliced in at `first`'s base marker,
/// followed by the marker that restores the state before it.
fn merge_alias_lists(first: &[String], second: &[String]) -> Vec<String> {
    let mut result: Vec<String> = first.to_vec();
    let mut revert = !result.is_empty() && result[0] == MARKER_MINUS;
    let mut i = 0;
    while i < result.len() {
        let tok = result[i].as_str();
        if tok == MARKER_PLUS {
            revert = false;
            i += 1;
            continue;
        }
        if tok == MARKER_MINUS {
            revert = true;
            i += 1;
            continue;
        }
        if tok == MARKER_BASE {
            result.remove(i);
            result.insert(i, if revert { MARKER_MINUS.into() } else { MARKER_PLUS.into() });
            for (j, s) in second.iter().enumerate() {
                result.insert(i + j, s.clone());
            }
            break;
        }
        i += 1;
    }
    result
}

#[derive(Clone, Copy)]
enum Kind {
    Mvt,
    Def,
}

fn kind_list(e: &crate::db::TerrainEntry, kind: Kind) -> &Vec<String> {
    match kind {
        Kind::Mvt => &e.mvt_type,
        Kind::Def => &e.def_type,
    }
}

/// `_get_underlying`: the alias list of `code` for movement or defense;
/// a composite merges the overlay's list with the base's.
fn get_underlying(db: &TerrainDb, code: &str, kind: Kind) -> Vec<String> {
    if let Some(k) = code.find('^') {
        let (base, overlay) = (&code[..k], &code[k..]);
        return match (db.entries.get(base), db.entries.get(overlay)) {
            (Some(b), Some(o)) => merge_alias_lists(kind_list(o, kind), kind_list(b, kind)),
            _ => vec![code.to_string()],
        };
    }
    match db.entries.get(code) {
        Some(e) => kind_list(e, kind).clone(),
        None => vec![code.to_string()],
    }
}

fn is_indivisible(code: &str, underlying: &[String]) -> bool {
    underlying.len() == 1 && underlying[0] == code
}

/// A table lookup by terrain id (`costs.get(terrain_id, default)`).
fn lookup(table: &[(String, i64)], id: &str) -> Option<i64> {
    table.iter().find(|(k, _)| k == id).map(|(_, v)| *v)
}

fn terminal_cost(db: &TerrainDb, code: &str, costs: &[(String, i64)], default: i64) -> i64 {
    match db.entries.get(code) {
        None => default,
        Some(e) => lookup(costs, &e.id).unwrap_or(default).abs(),
    }
}

/// `_collect_neg_caps`: the largest |v| over the negative entries the
/// alias walk meets (a defense floor, movetype.cpp config_to_min).
fn collect_neg_caps(db: &TerrainDb, code: &str, costs: &[(String, i64)], kind: Kind, recurse: u32) -> i64 {
    if recurse > 100 {
        return 0;
    }
    let underlying = get_underlying(db, code, kind);
    if is_indivisible(code, &underlying) {
        return match db.entries.get(code) {
            None => 0,
            Some(e) => {
                let v = lookup(costs, &e.id).unwrap_or(0);
                if v < 0 { -v } else { 0 }
            }
        };
    }
    let mut best = 0;
    for tok in &underlying {
        if tok == MARKER_PLUS || tok == MARKER_MINUS {
            continue;
        }
        best = best.max(collect_neg_caps(db, tok, costs, kind, recurse + 1));
    }
    best
}

#[allow(clippy::too_many_arguments)]
fn calc_value(db: &TerrainDb, code: &str, costs: &[(String, i64)], kind: Kind, default: i64,
              max_value: i64, min_value: i64, high_is_good: bool, recurse: u32) -> i64 {
    if recurse > 100 {
        return default;
    }
    let underlying = get_underlying(db, code, kind);
    if is_indivisible(code, &underlying) {
        return terminal_cost(db, code, costs, default);
    }
    let mut prefer_high = high_is_good;
    let mut result = default;
    if !underlying.is_empty() && underlying[0] == MARKER_MINUS {
        result = if result == max_value { min_value } else { max_value };
    }
    for tok in &underlying {
        if tok == MARKER_PLUS {
            prefer_high = high_is_good;
            continue;
        }
        if tok == MARKER_MINUS {
            prefer_high = !high_is_good;
            continue;
        }
        let num = calc_value(db, tok, costs, kind, default, max_value, min_value, high_is_good, recurse + 1);
        if (prefer_high && num > result) || (!prefer_high && num < result) {
            result = num;
        }
    }
    result.clamp(min_value, max_value)
}

/// `terrain_resolver.mvt_cost`.
pub fn mvt_cost(db: &TerrainDb, code: &str, costs: &[(String, i64)]) -> i64 {
    calc_value(db, code, costs, Kind::Mvt, UNREACHABLE_COST, UNREACHABLE_COST, 1, false, 0)
}

/// `terrain_resolver.def_pct`: the best chance to be hit over the
/// aliases, raised to any negative-entry floor.
pub fn def_pct(db: &TerrainDb, code: &str, defenses: &[(String, i64)]) -> i64 {
    let base = calc_value(db, code, defenses, Kind::Def, 100, 100, 0, false, 0);
    let cap = collect_neg_caps(db, code, defenses, Kind::Def, 0);
    if cap > base { cap } else { base }
}

fn split_base_overlay(code: &str) -> (&str, Option<&str>) {
    match code.find('^') {
        Some(k) => (&code[..k], Some(&code[k + 1..])),
        None => (code, None),
    }
}

/// `terrain_resolver.terrain_heals`: max(base, overlay).
pub fn terrain_heals(db: &TerrainDb, code: &str) -> i64 {
    let (base, overlay) = split_base_overlay(code);
    let b = db.entries.get(base).map_or(0, |e| e.heals);
    let o = overlay.map_or(0, |ov| db.entries.get(&format!("^{ov}")).map_or(0, |e| e.heals));
    b.max(o)
}

/// `game_core._light_params`: (light, max_light, min_light, any).
pub fn light_params(db: &TerrainDb, code: &str) -> (i64, i64, i64, bool) {
    let (base, overlay) = split_base_overlay(code);
    let (mut light, mut max_l, mut min_l) = match db.entries.get(base) {
        Some(e) => (e.light, e.max_light, e.min_light),
        None => (0, 0, 0),
    };
    if let Some(ov) = overlay {
        if let Some(o) = db.entries.get(&format!("^{ov}")) {
            light += o.light;
            max_l = max_l.max(o.max_light);
            min_l = min_l.min(o.min_light);
        }
    }
    (light, max_l, min_l, !(light == 0 && max_l == 0 && min_l == 0))
}

/// `terrain_resolver.hides_cover` for ambush, concealment and submerge.
pub fn hides_cover(code: &str, ability: &str) -> bool {
    let (base, overlay) = split_base_overlay(strip_start_position(code));
    let overlay = overlay.unwrap_or("");
    match ability {
        "ambush" => overlay.starts_with('F') || overlay == "Qhhf" || overlay == "Qhuf",
        "concealment" => overlay.starts_with('V'),
        "submerge" => base.starts_with("Wo"),
        _ => false,
    }
}

/// `ALIAS_TO_TERRAIN_NAME` as class values.
fn alias_class(alias: &str) -> Option<i64> {
    use terrain_class::*;
    Some(match alias {
        "Gt" | "Rt" => FLAT,
        "Ht" => HILLS,
        "Mt" => MOUNTAINS,
        "Ft" => FOREST,
        "Wst" | "Wrt" => SHALLOWWATER,
        "Wdt" => DEEPWATER,
        "St" => SWAMP,
        "Dt" => SAND,
        "At" => FROZEN,
        "Ut" | "Tt" => CAVE,
        "Xt" => IMPASSABLE,
        "Qt" => UNWALKABLE,
        "Vt" => VILLAGE,
        "Ct" => CASTLE,
        _ => return None,
    })
}

/// `terrain_resolver.terrain_mask`: the classes of the code's movement
/// and defense aliases as a bitmask over `classes.Terrain`. A code
/// terrain.cfg defines outright is found before any merge.
pub fn terrain_mask(db: &TerrainDb, code: &str) -> i64 {
    let code = strip_start_position(code);
    if code.is_empty() {
        return 0;
    }
    let lists: [Vec<String>; 2] = match db.entries.get(code) {
        Some(e) => [e.mvt_type.clone(), e.def_type.clone()],
        None => [get_underlying(db, code, Kind::Mvt), get_underlying(db, code, Kind::Def)],
    };
    let mut mask = 0i64;
    for lst in &lists {
        for a in lst {
            if a == MARKER_PLUS || a == MARKER_MINUS || a == MARKER_BASE {
                continue;
            }
            let class = alias_class(a).or_else(|| {
                db.entries.get(a.as_str()).and_then(|e| {
                    (e.id == "off_map" || e.id == "off_map2").then_some(terrain_class::IMPASSABLE)
                })
            });
            if let Some(c) = class {
                mask |= 1 << c;
            }
        }
    }
    mask
}

/// `replay_dataset._TERRAIN_BASE`: the base code's one class.
fn base_class(base: &str) -> i64 {
    use terrain_class::*;
    match base {
        "Aa" => FROZEN,
        "Gg" | "Gs" | "Gd" | "Rr" | "Re" => FLAT,
        "Hh" | "Ha" => HILLS,
        "Mm" | "Ms" | "Md" => MOUNTAINS,
        "Ww" | "Wwf" | "Wwt" | "Wwr" | "Wwg" => SHALLOWWATER,
        "Wo" => DEEPWATER,
        "Ss" => SWAMP,
        "Ds" => SAND,
        "Ql" | "Uu" | "Uue" => CAVE,
        "Xu" | "_off" => IMPASSABLE,
        "Ch" | "Cha" | "Chr" | "Chs" | "Chw" => CASTLE,
        _ => FLAT,
    }
}

/// `replay_dataset._parse_hex_code`: (terrain classes bitmask, modifier
/// bits) of a cell, and the class the hex set's one-class view gives it
/// (`encoder._first_terrain_id`, whose "first member" of a two-class
/// set follows CPython's set layout: slot = value & 7, the base inserted
/// first).
pub fn parse_hex_code(code: &str) -> (i64, u8, i64) {
    use terrain_class::*;
    let (base, overlay) = split_base_overlay(code);
    let overlay = overlay.unwrap_or("");
    let first = base_class(base);
    let mut types = 1i64 << first;
    let mut mods = 0u8;
    if overlay.contains('V') {
        types |= 1 << VILLAGE;
    }
    let forest = overlay.contains('F');
    if forest {
        types |= 1 << FOREST;
    }
    if overlay.contains('K') || base.contains('K') {
        mods |= MOD_KEEP;
        types |= 1 << CASTLE;
    }
    if base.contains('C') || overlay.contains('C') {
        mods |= MOD_CASTLE;
        types |= 1 << CASTLE;
    }
    let one = if types & (1 << VILLAGE) != 0 {
        VILLAGE
    } else if types & (1 << CASTLE) != 0 {
        CASTLE
    } else if forest && first != FOREST && (first & 7) > (FOREST & 7) {
        FOREST
    } else {
        first
    };
    (types, mods, one)
}

// ---------------------------------------------------------------------
// Python entry points: the differential tests' handles
// ---------------------------------------------------------------------

#[pyfunction]
pub fn terrain_mvt_cost(code: &str, costs: Vec<(String, i64)>) -> PyResult<i64> {
    let db = terrain_db()?;
    Ok(mvt_cost(&db, code, &costs))
}

#[pyfunction]
pub fn terrain_def_pct(code: &str, defenses: Vec<(String, i64)>) -> PyResult<i64> {
    let db = terrain_db()?;
    Ok(def_pct(&db, code, &defenses))
}

/// (heals, light, max_light, min_light, has_light, mask, ambush,
/// concealment, submerge, types, modifiers, one class) of one code.
#[pyfunction]
#[allow(clippy::type_complexity)]
pub fn terrain_facts(code: &str) -> PyResult<(i64, i64, i64, i64, bool, i64, bool, bool, bool, i64, u8, i64)> {
    let db = terrain_db()?;
    let c = strip_start_position(code);
    let (light, max_l, min_l, any) = light_params(&db, c);
    let (types, mods, one) = parse_hex_code(c);
    Ok((terrain_heals(&db, c), light, max_l, min_l, any, terrain_mask(&db, c), hides_cover(code, "ambush"),
        hides_cover(code, "concealment"), hides_cover(code, "submerge"), types, mods, one))
}
