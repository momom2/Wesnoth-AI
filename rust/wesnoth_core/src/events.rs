//! The scenario-event interpreter (`tools/scenario_events.py`): a
//! scenario's [event]s fired by name at the engine's moments, each action
//! tag applied to the core, WML variables, stored locations, terrain
//! changes and time areas included. Python reads and preprocesses the
//! scenario's WML and hands the parsed tree over (`setup_scenario`, or
//! `load_events` for a state whose events Python set up); everything
//! from there runs here. The Python interpreter is the oracle
//! (tests/test_rust_events.py) until the port's certification.

use pyo3::prelude::*;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::{Arc, Mutex, RwLock};

use crate::core::{DefTable, GameCore};
use crate::effects::{apply_effect, py_int};
use crate::terrain;
use crate::units::{apply_traits, build_unit, trait_effect, UnitSpec};
use crate::wml::{clean, Wml};

/// One [event] of a scenario (`scenario_events.ScenarioEvent`).
#[derive(Clone, Debug)]
pub struct EventDef {
    pub names: Vec<String>,
    pub first_time_only: bool,
    pub actions: Vec<Arc<Wml>>,
    pub scenario_id: String,
}

/// The engine's `event_handlers::standardize_name`: trimmed, every
/// internal space an underscore.
pub fn standard_event_name(name: &str) -> String {
    name.trim().replace(' ', "_")
}

/// The names an [event] answers to: its comma-separated `name=`.
pub fn event_names(raw: &str) -> Vec<String> {
    raw.split(',').filter(|p| !p.trim().is_empty()).map(standard_event_name).collect()
}

/// `_IGNORED_ACTIONS` and `_SUBSTITUTED_ACTIONS`: tags that do nothing
/// here, each for a recorded reason (tools/scenario_events.py).
const NO_OP_ACTIONS: [&str; 15] = [
    "message", "note", "objectives", "objective", "item", "label", "music", "sound", "scroll",
    "screen_fade", "delay", "variable", "case", "endlevel", "end_turn",
];

/// `_DEFAULT_LAWFUL_BY_TOD_ID` (data/core/macros/schedules.cfg).
fn default_lawful(id: &str) -> i64 {
    match id {
        "morning" | "midday" | "afternoon" => 25,
        "first_watch" | "second_watch" | "underground" | "deep_underground" => -25,
        _ => 0,
    }
}

static UNMODELLED_COUNTS: Mutex<BTreeMap<String, i64>> = Mutex::new(BTreeMap::new());
static UNMODELLED_SEEN: Mutex<Option<HashSet<(String, String)>>> = Mutex::new(None);

/// Python's `int(s)` raising ValueError as Python does.
fn int_strict(s: &str) -> PyResult<i64> {
    py_int(s).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!("invalid literal for int() with base 10: {s:?}"))
    })
}

/// Python's `float(s)` as the variable conditions read it.
fn py_float(s: &str) -> Option<f64> {
    let t = s.trim();
    if t.is_empty() {
        return None;
    }
    t.replace('_', "").parse::<f64>().ok()
}

/// `int(float(s))`, None where Python raises.
fn int_of_float(s: &str) -> Option<i64> {
    py_float(s).filter(|f| f.is_finite()).map(|f| f.trunc() as i64)
}

/// `wml_state.wml_int`: quotes and a percent sign tolerated, the
/// leading integer of a concatenated value salvaged.
fn wml_int(value: Option<&str>, default: i64) -> i64 {
    let v = match value {
        Some(v) => v,
        None => return default,
    };
    let mut text = v.trim().trim_matches('"').trim().to_string();
    if let Some(t) = text.strip_suffix('%') {
        text = t.trim().to_string();
    }
    if text.is_empty() {
        return default;
    }
    if let Some(i) = py_int(&text) {
        return i;
    }
    let b = text.as_bytes();
    let mut end = 0;
    if end < b.len() && b[end] == b'-' {
        end += 1;
    }
    let digits_start = end;
    while end < b.len() && b[end].is_ascii_digit() {
        end += 1;
    }
    if end > digits_start {
        text[..end].parse().unwrap_or(default)
    } else {
        default
    }
}

/// `_parse_int_csv`: "1,2,3" or "1..5".
fn parse_int_csv(s: &str) -> Vec<i64> {
    let mut out = Vec::new();
    for part in s.split(',') {
        let part = part.trim();
        if let Some((a, b)) = part.split_once("..") {
            if let (Some(a), Some(b)) = (py_int(a), py_int(b)) {
                out.extend(a.min(b)..=a.max(b));
            }
        } else if let Some(v) = py_int(part) {
            out.push(v);
        }
    }
    out
}

/// `_parse_int_or_range`: "5", "1-4" or "1..4".
fn parse_int_or_range(part: &str) -> Vec<i64> {
    let part = part.trim();
    if part.is_empty() {
        return Vec::new();
    }
    for sep in ["..", "-"] {
        if let Some((a, b)) = part.split_once(sep) {
            return match (py_int(a), py_int(b)) {
                (Some(a), Some(b)) => (a.min(b)..=a.max(b)).collect(),
                _ => Vec::new(),
            };
        }
    }
    py_int(part).map(|v| vec![v]).unwrap_or_default()
}

/// `_resolve_xy_attr`: the WML hexes (1-indexed) of x= / y=, paired
/// positionally when both are equal-length lists of single integers,
/// else the product of their ranges (a missing axis is the whole map).
fn resolve_xy(x_attr: &str, y_attr: &str, map_w: i64, map_h: i64) -> BTreeSet<(i64, i64)> {
    let parts = |s: &str| -> Vec<String> {
        s.split(',').map(|p| p.trim().to_string()).filter(|p| !p.is_empty()).collect()
    };
    let (xp, yp) = (parts(x_attr), parts(y_attr));
    let mut out = BTreeSet::new();
    if !xp.is_empty() && !yp.is_empty() && xp.len() == yp.len()
        && xp.iter().chain(yp.iter()).all(|p| !p.contains('-') && !p.contains("..")) {
        for (a, b) in xp.iter().zip(yp.iter()) {
            if let (Some(a), Some(b)) = (py_int(a), py_int(b)) {
                out.insert((a, b));
            }
        }
        return out;
    }
    let mut xs: Vec<i64> = xp.iter().flat_map(|p| parse_int_or_range(p)).collect();
    if xs.is_empty() {
        xs = (1..=map_w).collect();
    }
    let mut ys: Vec<i64> = yp.iter().flat_map(|p| parse_int_or_range(p)).collect();
    if ys.is_empty() {
        ys = (1..=map_h).collect();
    }
    for &x in &xs {
        for &y in &ys {
            out.insert((x, y));
        }
    }
    out
}

/// `*`-only glob, whole string.
fn glob_match(pattern: &str, text: &str) -> bool {
    let (p, t): (Vec<char>, Vec<char>) = (pattern.chars().collect(), text.chars().collect());
    let (mut i, mut j) = (0usize, 0usize);
    let (mut star, mut mark) = (usize::MAX, 0usize);
    while j < t.len() {
        if i < p.len() && p[i] == '*' {
            star = i;
            mark = j;
            i += 1;
        } else if i < p.len() && p[i] == t[j] {
            i += 1;
            j += 1;
        } else if star != usize::MAX {
            i = star + 1;
            mark += 1;
            j = mark;
        } else {
            return false;
        }
    }
    while i < p.len() && p[i] == '*' {
        i += 1;
    }
    i == p.len()
}

/// `_terrain_filter_match`: comma-separated alternatives, exact or `*` globs.
fn terrain_filter_match(code: &str, pattern: &str) -> bool {
    if pattern.is_empty() {
        return true;
    }
    pattern.split(',').map(str::trim).filter(|a| !a.is_empty())
        .any(|alt| alt == code || (alt.contains('*') && glob_match(alt, code)))
}

/// `_FACTION_LUA_RE`: `wml.variables["p"..tostring(i).."_faction"] =
/// side.faction`, whitespace allowed between the tokens.
fn publishes_factions(code: &str) -> bool {
    const TOKENS: [&str; 11] = ["[", "\"p\"", "..", "tostring(", "i", ")", "..", "\"_faction\"", "]", "=",
                                "side.faction"];
    let mut start = 0;
    while let Some(k) = code[start..].find("wml.variables") {
        let mut rest = &code[start + k + "wml.variables".len()..];
        let mut ok = true;
        for tok in TOKENS {
            rest = rest.trim_start();
            match rest.strip_prefix(tok) {
                Some(r) => rest = r,
                None => {
                    ok = false;
                    break;
                }
            }
        }
        if ok {
            return true;
        }
        start += k + 1;
    }
    false
}

/// `_effect_member_ids`-style trait ids of a [modifications] block.
fn trait_ids_of(mods: &Wml) -> Vec<String> {
    mods.children.iter().filter(|c| c.tag == "trait")
        .map(|c| clean(c.attr("id").unwrap_or("")).to_lowercase())
        .filter(|t| !t.is_empty()).collect()
}

/// `own_modification_effects`: the effects of a unit's own [object]s and
/// of its custom (unnamed) traits.
fn own_modification_effects(mods: Option<&Wml>) -> Vec<Arc<Wml>> {
    let mut out = Vec::new();
    if let Some(m) = mods {
        for node in &m.children {
            let tid = clean(node.attr("id").unwrap_or("")).to_lowercase();
            if node.tag == "object" || (node.tag == "trait" && trait_effect(&tid).is_none()) {
                out.extend(node.all("effect").cloned());
            }
        }
    }
    out
}

impl GameCore {
    /// `_report_unmodelled_value`: raise UnmodelledWML in strict mode,
    /// else warn.
    fn report_unmodelled_value(&self, what: &str) -> PyResult<()> {
        let text = if self.firing_scenario.is_empty() {
            what.to_string()
        } else {
            format!("{what} (scenario {})", self.firing_scenario)
        };
        if self.strict_wml {
            return Err(unmodelled_error(&text));
        }
        crate::effects::warn_once(text);
        Ok(())
    }

    /// `_report_unmodelled`: an action tag with no handler.
    fn report_unmodelled(&self, tag: &str) -> PyResult<()> {
        *UNMODELLED_COUNTS.lock().unwrap().entry(tag.to_string()).or_insert(0) += 1;
        if self.strict_wml {
            let mut text = format!("[{tag}] has no handler and no recorded reason to ignore it");
            if !self.firing_scenario.is_empty() {
                text += &format!(" (scenario {})", self.firing_scenario);
            }
            return Err(unmodelled_error(&text));
        }
        let key = (tag.to_string(), self.firing_scenario.clone());
        let mut seen = UNMODELLED_SEEN.lock().unwrap();
        if seen.get_or_insert_with(HashSet::new).insert(key) {
            crate::effects::warn_once(format!(
                "unmodelled event action [{tag}]{}: doing nothing",
                if self.firing_scenario.is_empty() { String::new() } else { format!(" in {}", self.firing_scenario) }));
        }
        Ok(())
    }

    /// `_subst_wml_vars`: each `$name` (letters, digits, `_`, `.`, `[`,
    /// `]`) replaced by the variable's value, "" when unset.
    fn subst(&self, raw: &str) -> String {
        if !raw.contains('$') {
            return raw.to_string();
        }
        let chars: Vec<char> = raw.chars().collect();
        let mut out = String::with_capacity(raw.len());
        let mut i = 0;
        while i < chars.len() {
            let c = chars[i];
            if c == '$' && i + 1 < chars.len() && (chars[i + 1].is_ascii_alphabetic() || chars[i + 1] == '_') {
                let mut j = i + 1;
                while j < chars.len()
                    && (chars[j].is_ascii_alphanumeric() || matches!(chars[j], '_' | '.' | '[' | ']')) {
                    j += 1;
                }
                let name: String = chars[i + 1..j].iter().collect();
                out += self.wml_vars.get(&name).map(|s| s.as_str()).unwrap_or("");
                i = j;
            } else {
                out.push(c);
                i += 1;
            }
        }
        out
    }

    fn map_bounds(&self) -> (i64, i64) {
        (self.size_x, self.size_y)
    }

    /// `_eval_location_clause`: x= / y= hexes, kept by terrain=.
    fn location_clause(&self, clause: &Wml) -> BTreeSet<(i64, i64)> {
        let (w, h) = self.map_bounds();
        let mut hexes = resolve_xy(clause.attr("x").unwrap_or(""), clause.attr("y").unwrap_or(""), w, h);
        let pattern = clause.attr("terrain").unwrap_or("").trim().to_string();
        if !pattern.is_empty() {
            hexes.retain(|&(wx, wy)| {
                let code = self.map.pos_index.get(&(wx - 1, wy - 1)).map(|&i| self.map.codes[i].as_str()).unwrap_or("");
                terrain_filter_match(code, &pattern)
            });
        }
        hexes
    }

    /// The hexes of x= / y= inside the map, 0-indexed.
    fn map_hexes(&self, x_attr: &str, y_attr: &str) -> BTreeSet<(i64, i64)> {
        let (w, h) = self.map_bounds();
        resolve_xy(x_attr, y_attr, w, h).into_iter()
            .filter(|&(wx, wy)| wx > 0 && wx <= w && wy > 0 && wy <= h)
            .map(|(wx, wy)| (wx - 1, wy - 1)).collect()
    }

    // ---- actions -----------------------------------------------------

    fn apply_action(&mut self, action: &Wml) -> PyResult<()> {
        match action.tag.as_str() {
            "terrain" => self.terrain_action(action),
            "modify_side" => self.modify_side_action(action),
            "gold" => self.gold_action(action),
            "store_locations" => self.store_locations_action(action),
            "clear_variable" => {
                let name = action.attr("name").unwrap_or("").trim().to_string();
                self.scenario_vars.remove(&name);
                Ok(())
            }
            "time_area" => self.time_area_action(action),
            "object" => self.object_action(action),
            "set_variable" => self.set_variable_action(action),
            "fire_event" => {
                let name = action.clean("name");
                if name.is_empty() || self.events.is_empty() {
                    return Ok(());
                }
                self.fire(&name).map(|_| ())
            }
            "switch" => self.switch_action(action),
            "lua" => {
                if publishes_factions(action.attr("code").unwrap_or("")) {
                    for (i, s) in self.sides.clone().iter().enumerate() {
                        self.wml_vars.insert(format!("p{}_faction", i + 1), s.faction.clone());
                    }
                }
                Ok(())
            }
            "unit" => self.unit_action(action),
            "capture_village" => self.capture_village_action(action),
            "modify_unit" => self.modify_unit_action(action),
            "store_unit" => self.store_unit_action(action),
            "if" => {
                let branch = if self.eval_condition(action) { "then" } else { "else" };
                for blk in action.all(branch).cloned().collect::<Vec<_>>() {
                    for sub in blk.children.clone() {
                        self.apply_action(&sub)?;
                    }
                }
                Ok(())
            }
            t if NO_OP_ACTIONS.contains(&t) => Ok(()),
            t => self.report_unmodelled(t),
        }
    }

    /// `_terrain_action`: each (x, y) pair gets the new code; the core
    /// resolves the hex's facts and every movement class again there.
    fn terrain_action(&mut self, action: &Wml) -> PyResult<()> {
        let xs = parse_int_csv(action.attr("x").unwrap_or(""));
        let ys = parse_int_csv(action.attr("y").unwrap_or(""));
        let code = action.attr("terrain").unwrap_or("").trim().to_string();
        if xs.is_empty() || ys.is_empty() || code.is_empty() {
            return Ok(());
        }
        let mut changed = Vec::new();
        for (&wx, &wy) in xs.iter().zip(ys.iter()) {
            self.terrain_log.push((wx, wy, code.clone()));
            match self.map.pos_index.get(&(wx - 1, wy - 1)) {
                Some(&h) => changed.push(h),
                None => crate::effects::warn_once(format!(
                    "[terrain] at ({wx},{wy}) is off the map; the core keeps its hex set")),
            }
        }
        self.set_terrain(&changed, &code);
        Ok(())
    }

    /// New terrain on `hexes`: the map's per-hex facts and the movement
    /// classes there, on a copy this core owns.
    pub fn set_terrain(&mut self, hexes: &[usize], code: &str) {
        if hexes.is_empty() {
            return;
        }
        let tdb = self.tdb.clone();
        let stripped = terrain::strip_start_position(code).to_string();
        let (types, mods, one) = terrain::parse_hex_code(code);
        let (light, max_l, min_l, any) = terrain::light_params(&tdb, &stripped);
        let heal = terrain::terrain_heals(&tdb, &stripped);
        let mask = terrain::terrain_mask(&tdb, code);
        {
            let map = Arc::make_mut(&mut self.map);
            for &h in hexes {
                map.codes[h] = stripped.clone();
                map.keep[h] = (mods & terrain::MOD_KEEP != 0) as u8;
                map.castle_or_keep[h] = (mods & (terrain::MOD_KEEP | terrain::MOD_CASTLE) != 0) as u8;
                map.castle_mod[h] = (mods & terrain::MOD_CASTLE != 0) as u8;
                map.village_terrain[h] = (types & (1 << terrain::terrain_class::VILLAGE) != 0) as u8;
                map.village_mod[h] = 0;
                map.terrain_type_id[h] = one;
                map.terrain_mask[h] = mask;
                map.heal[h] = heal;
                map.light_mod[h] = light;
                map.light_max[h] = max_l;
                map.light_min[h] = min_l;
                map.has_light[h] = any as u8;
                map.hides_ambush[h] = terrain::hides_cover(code, "ambush") as u8;
                map.hides_concealment[h] = terrain::hides_cover(code, "concealment") as u8;
                map.hides_submerge[h] = terrain::hides_cover(code, "submerge") as u8;
            }
        }
        self.reresolve_classes(hexes);
        self.map_version += 1;
    }

    /// Every registered movement class resolved again on `hexes`, into
    /// registries this core owns (its forks' classes keep the old map).
    fn reresolve_classes(&mut self, hexes: &[usize]) {
        let index: Vec<(crate::core::ClassKey, usize)> =
            self.class_index.read().unwrap().iter().map(|(k, &v)| (k.clone(), v)).collect();
        let mut classes = self.classes.read().unwrap().clone();
        for ((type_name, slowed, table), id) in &index {
            let t = self.db.get(type_name);
            let fresh = self.compute_class_at(&t, *slowed, table, hexes);
            for (k, &h) in hexes.iter().enumerate() {
                classes[*id].mcost[h] = fresh.0[k];
                classes[*id].dsub[h] = fresh.1[k];
                classes[*id].defense_pct[h] = fresh.2[k];
            }
        }
        self.classes = Arc::new(RwLock::new(classes));
        self.class_index = Arc::new(RwLock::new(index.into_iter().collect::<HashMap<_, _>>()));
    }

    /// `_modify_side_action`: gold, income, recruit list.
    fn modify_side_action(&mut self, action: &Wml) -> PyResult<()> {
        let side = int_strict(action.attr("side").filter(|s| !s.is_empty()).unwrap_or("0"))?;
        if side < 1 || side as usize > self.sides.len() {
            return Ok(());
        }
        let k = side as usize - 1;
        let gold = match action.attr("gold").filter(|s| !s.is_empty()) {
            Some(v) => int_strict(v)?,
            None => self.sides[k].current_gold,
        };
        let income = match action.attr("income").filter(|s| !s.is_empty()) {
            Some(v) => int_strict(v)?,
            None => self.sides[k].base_income,
        };
        let recruit = action.attr("recruit").unwrap_or("");
        let s = &mut self.sides[k];
        if !recruit.is_empty() {
            s.recruits = recruit.split(',').map(str::trim).filter(|r| !r.is_empty()).map(String::from).collect();
        }
        s.current_gold = gold;
        s.base_income = income;
        Ok(())
    }

    /// `_gold_action`.
    fn gold_action(&mut self, action: &Wml) -> PyResult<()> {
        let side = int_strict(action.attr("side").filter(|s| !s.is_empty()).unwrap_or("0"))?;
        let amount = int_strict(action.attr("amount").filter(|s| !s.is_empty()).unwrap_or("0"))?;
        if side >= 1 && side as usize <= self.sides.len() {
            self.sides[side as usize - 1].current_gold += amount;
        }
        Ok(())
    }

    /// `_store_locations_action`: x= / y= / terrain= clauses and their
    /// [or]s, kept inside the map, under the variable.
    fn store_locations_action(&mut self, action: &Wml) -> PyResult<()> {
        let var = action.attr("variable").unwrap_or("").trim().to_string();
        if var.is_empty() {
            return Ok(());
        }
        let (w, h) = self.map_bounds();
        let mut hexes = self.location_clause(action);
        for sub in action.all("or") {
            hexes.extend(self.location_clause(sub));
        }
        let py: BTreeSet<(i64, i64)> = hexes.into_iter()
            .filter(|&(wx, wy)| wx > 0 && wx <= w && wy > 0 && wy <= h)
            .map(|(wx, wy)| (wx - 1, wy - 1)).collect();
        self.scenario_vars.insert(var, py);
        Ok(())
    }

    /// `_time_area_action`: the area's cycle phased to turn 1 on its hexes.
    pub fn time_area_action(&mut self, action: &Wml) -> PyResult<()> {
        let mut cycle: Vec<i64> = Vec::new();
        for ch in action.all("time") {
            let lb = match ch.attr("lawful_bonus") {
                None | Some("") => default_lawful(ch.attr("id").unwrap_or("").trim()),
                Some(raw) => py_int(raw).unwrap_or(0),
            };
            cycle.push(lb);
        }
        if cycle.is_empty() {
            return Ok(());
        }
        let placed = self.global.turn_number.max(1);
        let n = cycle.len() as i64;
        let shift = (wml_int(action.attr("current_time"), 0) - (placed - 1)).rem_euclid(n) as usize;
        cycle.rotate_left(shift);
        let find_in = action.attr("find_in").unwrap_or("").trim().to_string();
        let hexes: BTreeSet<(i64, i64)> = if !find_in.is_empty() {
            self.scenario_vars.get(&find_in).cloned().unwrap_or_default()
        } else {
            self.map_hexes(action.attr("x").unwrap_or(""), action.attr("y").unwrap_or(""))
        };
        let idx: Vec<usize> = hexes.iter().filter_map(|p| self.map.pos_index.get(p).copied()).collect();
        if idx.is_empty() {
            return Ok(());
        }
        let map = Arc::make_mut(&mut self.map);
        let c = match map.cycles.iter().position(|c| *c == cycle) {
            Some(c) => c,
            None => {
                map.cycles.push(cycle);
                map.cycles.len() - 1
            }
        };
        for h in idx {
            map.area_cycle[h] = c as i64;
        }
        self.map_version += 1;
        Ok(())
    }

    /// `_object_action`: the [filter]'s units get the [effect]s in order,
    /// kept for re-application on advancement.
    fn object_action(&mut self, action: &Wml) -> PyResult<()> {
        let filt = match action.first("filter") {
            Some(f) => f.clone(),
            None => return Ok(()),
        };
        let hexes = self.map_hexes(filt.attr("x").unwrap_or(""), filt.attr("y").unwrap_or(""));
        let type_filter = filt.clean("type");
        let side_filter = py_int(&filt.clean("side")).unwrap_or(0);
        let effects: Vec<Arc<Wml>> = action.all("effect").cloned().collect();
        for u in self.units.iter_mut() {
            if !hexes.is_empty() && !hexes.contains(&(u.x, u.y)) {
                continue;
            }
            if !type_filter.is_empty() && u.name != type_filter {
                continue;
            }
            if side_filter != 0 && u.side != side_filter {
                continue;
            }
            for eff in &effects {
                apply_effect(u, eff);
            }
            u.object_effects.extend(effects.iter().cloned());
        }
        Ok(())
    }

    /// `_set_variable_action`: value= or literal=, then add= and sub=.
    fn set_variable_action(&mut self, action: &Wml) -> PyResult<()> {
        let name = action.clean("name");
        if name.is_empty() {
            return Ok(());
        }
        if let Some(v) = action.attr("value") {
            let val = clean(&self.subst(v));
            self.wml_vars.insert(name.clone(), val);
        } else if let Some(v) = action.attr("literal") {
            self.wml_vars.insert(name.clone(), clean(v));
        }
        for (key, sign) in [("add", 1i64), ("sub", -1i64)] {
            if let Some(v) = action.attr(key) {
                let cur = int_of_float(self.wml_vars.get(&name).map(|s| s.as_str()).unwrap_or("0")).unwrap_or(0);
                let inc = int_of_float(&clean(&self.subst(v))).unwrap_or(0);
                self.wml_vars.insert(name.clone(), (cur + sign * inc).to_string());
            }
        }
        Ok(())
    }

    /// `_switch_action`: the first [case] whose value list holds the
    /// variable's value, else the first [else].
    fn switch_action(&mut self, action: &Wml) -> PyResult<()> {
        let var = action.clean("variable");
        if var.is_empty() {
            return Ok(());
        }
        let cur = self.wml_vars.get(&var).cloned().unwrap_or_default();
        let mut target: Option<Arc<Wml>> = None;
        let mut else_case: Option<Arc<Wml>> = None;
        for child in &action.children {
            if child.tag == "case" {
                if child.attr("value").unwrap_or("").split(',').any(|v| v.trim() == cur) {
                    target = Some(child.clone());
                    break;
                }
            } else if child.tag == "else" && else_case.is_none() {
                else_case = Some(child.clone());
            }
        }
        if let Some(t) = target.or(else_case) {
            for sub in t.children.clone() {
                self.apply_action(&sub)?;
            }
        }
        Ok(())
    }

    /// `_eval_variable_cond`.
    fn eval_variable(&self, node: &Wml) -> bool {
        let cur = self.wml_vars.get(&node.clean("name")).cloned().unwrap_or_default();
        for op in ["equals", "not_equals", "numerical_equals", "greater_than", "less_than",
                   "greater_than_equal_to", "less_than_equal_to"] {
            let rhs = match node.attr(op) {
                Some(v) => clean(&self.subst(v)),
                None => continue,
            };
            let (a, b) = (py_float(&cur), py_float(&rhs));
            return match (op, a, b) {
                ("equals", Some(a), Some(b)) => a == b,
                ("equals", _, _) => cur == rhs,
                ("not_equals", Some(a), Some(b)) => a != b,
                ("not_equals", _, _) => cur != rhs,
                (_, None, _) | (_, _, None) => false,
                ("numerical_equals", Some(a), Some(b)) => a == b,
                ("greater_than", Some(a), Some(b)) => a > b,
                ("less_than", Some(a), Some(b)) => a < b,
                ("greater_than_equal_to", Some(a), Some(b)) => a >= b,
                (_, Some(a), Some(b)) => a <= b,
            };
        }
        false
    }

    /// `_eval_condition`: [variable] children and nested [and]/[or]/[not].
    fn eval_condition(&self, node: &Wml) -> bool {
        let mut ok = true;
        for child in &node.children {
            match child.tag.as_str() {
                "variable" => ok = ok && self.eval_variable(child),
                "and" => ok = ok && self.eval_condition(child),
                "or" => ok = ok || self.eval_condition(child),
                "not" => ok = ok && !self.eval_condition(child),
                _ => {}
            }
        }
        ok
    }

    /// `_heals_ability`: heals_4 or heals_8 from the block's value.
    fn heals_ability(&self, node: &Wml) -> PyResult<Option<&'static str>> {
        let raw = node.attr("value");
        let value = match raw.map(clean) {
            None => None,
            Some(v) if v.is_empty() => None,
            Some(v) => Some(v),
        };
        let v = match value {
            None => {
                self.report_unmodelled_value("[heals] with no value= (the engine heals 0)")?;
                return Ok(None);
            }
            Some(v) => v,
        };
        let value = match py_int(&v) {
            Some(n) => n,
            None => {
                self.report_unmodelled_value(&format!("[heals] value={:?} is not an integer", raw.unwrap_or("")))?;
                return Ok(None);
            }
        };
        Ok(match value {
            4 => Some("heals_4"),
            8 => Some("heals_8"),
            _ => {
                self.report_unmodelled_value(&format!("[heals] value={value}: the sim models 4 and 8"))?;
                if value > 8 { Some("heals_8") } else if value > 0 { Some("heals_4") } else { None }
            }
        })
    }

    /// `_unit_action`: a unit placed by an event, with its type's
    /// musthave traits and its [modifications]' traits in document order,
    /// its own [object]s and custom traits (kept for advancement), the
    /// petrified status, extra abilities, role and guardian flag.
    fn unit_action(&mut self, action: &Wml) -> PyResult<()> {
        let side = match py_int(&action.clean_or_default("side", "0")) {
            Some(s) if s > 0 => s,
            _ => return Ok(()),
        };
        let mut utype = action.clean("type");
        if utype.is_empty() {
            return Ok(());
        }
        let variation = action.clean("variation");
        let (wx, wy) = match (py_int(&action.clean_or_default("x", "0")), py_int(&action.clean_or_default("y", "0"))) {
            (Some(x), Some(y)) if x > 0 && y > 0 => (x, y),
            _ => return Ok(()),
        };
        if !variation.is_empty() {
            let composite = format!("{utype}:{variation}");
            if self.db.contains(&composite) {
                utype = composite;
            }
        }
        let uid = self.units.iter()
            .filter_map(|u| py_int(u.id.trim_start_matches('u'))).max().unwrap_or(0).max(0) + 1;
        let spec = UnitSpec { uid, unit_type: utype.clone(), side, x: wx - 1, y: wy - 1, ..Default::default() };
        let mut u = build_unit(&self.db, &spec, false, "", self.global.experience_modifier);
        let t = self.db.get(&utype);
        let mods = action.first("modifications").cloned();
        let mut trait_ids: Vec<String> = Vec::new();
        let must = t.traits.as_ref().map(|i| i.musthave.clone()).unwrap_or_default();
        let explicit = mods.as_ref().map(|m| trait_ids_of(m)).unwrap_or_default();
        for tid in must.into_iter().chain(explicit) {
            if !trait_ids.contains(&tid) {
                trait_ids.push(tid);
            }
        }
        if !trait_ids.is_empty() {
            let mut table: DefTable = match &u.def_table {
                Some(tab) if !tab.is_empty() => tab.as_ref().clone(),
                _ => t.defense.clone(),
            };
            apply_traits(&mut u, &trait_ids, t.level, &mut table);
            u.def_table = Some(Arc::new(table));
            u.trait_order = Some(trait_ids);
        }
        u.current_hp = u.max_hp;
        u.current_moves = u.max_moves;
        let own = own_modification_effects(mods.as_deref());
        for eff in &own {
            apply_effect(&mut u, eff);
        }
        u.object_effects = own;
        if let Some(st) = action.first("status") {
            if matches!(st.attr("petrified").unwrap_or("").trim().to_lowercase().as_str(), "yes" | "true" | "1") {
                u.add_status("petrified");
                u.current_moves = 0;
                u.has_attacked = true;
                u.attacks.clear();
            }
        }
        if let Some(ab) = action.first("abilities").cloned() {
            let mut abilities = u.abilities.clone();
            for child in &ab.children {
                if child.tag == "heals" {
                    if let Some(h) = self.heals_ability(child)? {
                        abilities.push(h.to_string());
                    }
                } else {
                    let id = clean(child.attr("id").unwrap_or(""));
                    abilities.push(if id.is_empty() { child.tag.clone() } else { id });
                }
            }
            u.abilities = crate::core::sorted(abilities);
        }
        let role = action.clean("role");
        if !role.is_empty() {
            u.wml_role = Some(role);
        }
        if action.clean("ai_special") == "guardian" {
            u.ai_guardian = true;
        }
        self.insert_unit(u)?;
        self.global.next_uid_counter += 1;
        Ok(())
    }

    /// `_capture_village_action`: the matched villages to the side (none
    /// for side 0); attributes and children it does not read reported.
    fn capture_village_action(&mut self, action: &Wml) -> PyResult<()> {
        let mut unread: Vec<String> = action.attrs.iter().map(|(k, _)| k.clone())
            .filter(|k| !matches!(k.as_str(), "side" | "x" | "y" | "terrain"))
            .collect::<BTreeSet<_>>().into_iter().collect();
        unread.extend(action.children.iter().map(|c| format!("[{}]", c.tag)));
        if !unread.is_empty() {
            self.report_unmodelled_value(&format!(
                "[capture_village] {}: the sim reads side, x, y and terrain", unread.join(", ")))?;
        }
        let side_raw = self.subst(action.attr("side").unwrap_or("")).trim().to_string();
        let side = if side_raw.is_empty() {
            0
        } else {
            match py_int(&side_raw) {
                Some(s) => s,
                None => return Ok(()),
            }
        };
        for (wx, wy) in self.location_clause(action) {
            if let Some(&h) = self.map.pos_index.get(&(wx - 1, wy - 1)) {
                if self.map.village_terrain[h] != 0 {
                    self.capture_village(h, side);
                }
            }
        }
        Ok(())
    }

    /// `_units_matching_filter`: x/y hexes, id=, side=, role=; the core's
    /// unit order (the Python walks its unit set).
    fn units_matching(&self, flt: Option<&Wml>) -> Vec<usize> {
        let flt = match flt {
            Some(f) => f,
            None => return (0..self.units.len()).collect(),
        };
        let (w, h) = self.map_bounds();
        let hexes: Option<BTreeSet<(i64, i64)>> =
            if !flt.attr("x").unwrap_or("").is_empty() || !flt.attr("y").unwrap_or("").is_empty() {
                Some(resolve_xy(flt.attr("x").unwrap_or(""), flt.attr("y").unwrap_or(""), w, h).into_iter()
                    .map(|(x, y)| (x - 1, y - 1)).collect())
            } else {
                None
            };
        let want_id = flt.clean("id");
        let want_side = flt.attr("side").unwrap_or("").trim().to_string();
        let want_role = flt.clean("role");
        (0..self.units.len()).filter(|&i| {
            let u = &self.units[i];
            hexes.as_ref().map_or(true, |hs| hs.contains(&(u.x, u.y)))
                && (want_id.is_empty() || u.id == want_id)
                && (want_side.is_empty() || u.side.to_string() == want_side)
                && (want_role.is_empty() || u.wml_role.as_deref() == Some(want_role.as_str()))
        }).collect()
    }

    /// `_modify_unit_action`: moves=, hitpoints=, experience= (current
    /// values, never below 0) on the [filter]'s units.
    fn modify_unit_action(&mut self, action: &Wml) -> PyResult<()> {
        let flt = match action.first("filter") {
            Some(f) => f.clone(),
            None => return Ok(()),
        };
        let mut changes: Vec<(&str, i64)> = Vec::new();
        for attr in ["moves", "hitpoints", "experience"] {
            if let Some(v) = action.attr(attr) {
                if let Some(n) = int_of_float(&clean(&self.subst(v))) {
                    changes.push((attr, n.max(0)));
                }
            }
        }
        if changes.is_empty() {
            return Ok(());
        }
        for i in self.units_matching(Some(&flt)) {
            let u = &mut self.units[i];
            for &(attr, v) in &changes {
                match attr {
                    "moves" => u.current_moves = v,
                    "hitpoints" => u.current_hp = v,
                    _ => u.current_exp = v,
                }
            }
        }
        Ok(())
    }

    /// `_store_unit_action`: the first matched unit's numbers under the
    /// variable (kill= is not modelled: the unit stays).
    fn store_unit_action(&mut self, action: &Wml) -> PyResult<()> {
        let var = action.clean("variable");
        if var.is_empty() {
            return Ok(());
        }
        let matched = self.units_matching(action.first("filter").map(|f| f.as_ref()));
        if matched.len() > 1 {
            crate::effects::warn_once(format!(
                "[store_unit] {var}: {} units match; the core stores its first, the Python its set's first",
                matched.len()));
        }
        let u = match matched.first() {
            Some(&i) => self.units[i].clone(),
            None => return Ok(()),
        };
        let v = &mut self.wml_vars;
        v.insert(format!("{var}.moves"), u.current_moves.to_string());
        v.insert(format!("{var}.max_moves"), u.max_moves.to_string());
        v.insert(format!("{var}.hitpoints"), u.current_hp.to_string());
        v.insert(format!("{var}.max_hitpoints"), u.max_hp.to_string());
        v.insert(format!("{var}.experience"), u.current_exp.to_string());
        v.insert(format!("{var}.side"), u.side.to_string());
        v.insert(format!("{var}.id"), u.id.clone());
        v.insert(format!("{var}.x"), (u.x + 1).to_string());
        v.insert(format!("{var}.y"), (u.y + 1).to_string());
        v.insert(format!("{var}.length"), matched.len().to_string());
        Ok(())
    }

    // ---- firing ------------------------------------------------------

    /// `fire_event`: every event answering to `trigger`, in WML order,
    /// latched after its actions when first_time_only. Returns how many.
    pub fn fire(&mut self, trigger: &str) -> PyResult<usize> {
        let trig = standard_event_name(trigger);
        let events = self.events.clone();
        let mut n = 0;
        for (k, ev) in events.iter().enumerate() {
            if !ev.names.contains(&trig) || (ev.first_time_only && self.fired[k]) {
                continue;
            }
            let outer = std::mem::replace(&mut self.firing_scenario, ev.scenario_id.clone());
            let mut result = Ok(());
            for action in &ev.actions {
                result = self.apply_action(action);
                if result.is_err() {
                    break;
                }
            }
            self.firing_scenario = outer;
            result?;
            self.fired[k] = true;
            n += 1;
        }
        Ok(n)
    }

    /// `fire_events`: each trigger in order.
    pub fn fire_all(&mut self, triggers: &[String]) -> PyResult<()> {
        if self.events.is_empty() {
            return Ok(());
        }
        for t in triggers {
            self.fire(t)?;
        }
        Ok(())
    }

    /// `collect_events`: the container's [event]s in order.
    fn collect_events(container: &Wml, scenario_id: &str) -> Vec<EventDef> {
        container.all("event").map(|ev| {
            let first = ev.attr("first_time_only").unwrap_or("yes").trim().to_lowercase();
            EventDef {
                names: event_names(&ev.clean("name")),
                first_time_only: matches!(first.as_str(), "yes" | "true" | "1"),
                actions: ev.children.iter().filter(|c| c.tag != "filter").cloned().collect(),
                scenario_id: scenario_id.to_string(),
            }
        }).collect()
    }

    /// `apply_side_unit_modifications`: the [side]-placed units' own
    /// [object]s and custom traits, matched by side, position and type.
    fn side_unit_modifications(&mut self, container: &Wml) {
        let mut effects: HashMap<(i64, i64, i64, String), Vec<Arc<Wml>>> = HashMap::new();
        for side_node in container.all("side") {
            let side = match py_int(side_node.attr("side").unwrap_or("0")) {
                Some(s) => s,
                None => continue,
            };
            for unit_node in side_node.all("unit") {
                let (x, y) = match (py_int(unit_node.attr("x").unwrap_or("0")), py_int(unit_node.attr("y").unwrap_or("0"))) {
                    (Some(x), Some(y)) => (x - 1, y - 1),
                    _ => continue,
                };
                let own = own_modification_effects(unit_node.first("modifications").map(|m| m.as_ref()));
                if !own.is_empty() {
                    effects.insert((side, x, y, unit_node.clean("type")), own);
                }
            }
        }
        if effects.is_empty() {
            return;
        }
        for u in self.units.iter_mut() {
            if let Some(own) = effects.get(&(u.side, u.x, u.y, u.name.clone())) {
                for eff in own {
                    apply_effect(u, eff);
                }
                u.object_effects.extend(own.iter().cloned());
            }
        }
    }
}

fn unmodelled_error(text: &str) -> PyErr {
    Python::with_gil(|py| {
        match py.import("wesnoth_ai.rules.scenario_cfg").and_then(|m| m.getattr("UnmodelledWML")) {
            Ok(cls) => match cls.call1((text,)) {
                Ok(inst) => PyErr::from_value(inst),
                Err(e) => e,
            },
            Err(e) => e,
        }
    })
}

impl Wml {
    /// `(attrs.get(key, default)).strip().strip('"')` with the default
    /// taken when the key is absent.
    pub fn clean_or_default(&self, key: &str, default: &str) -> String {
        clean(self.attr(key).unwrap_or(default))
    }
}

#[pymethods]
impl GameCore {
    /// `_setup_scenario_events` on the core: the scenario's top-level
    /// time areas, its [side] units' own modifications, its events, the
    /// `pN_faction` variables, then the prestart and start events. `root`
    /// is the scenario's parsed WML (`load_scenario_wml`), or None for a
    /// scenario without one (no events).
    #[pyo3(signature = (root, scenario_id, strict=false))]
    fn setup_scenario(&mut self, root: Option<Wml>, scenario_id: &str, strict: bool) -> PyResult<()> {
        self.strict_wml = strict;
        self.events = Arc::new(Vec::new());
        self.fired = Vec::new();
        let root = match root {
            Some(r) => r,
            None => return Ok(()),
        };
        let container = match root.first("multiplayer").or_else(|| root.first("scenario")) {
            Some(c) => c.clone(),
            None => return Ok(()),
        };
        self.firing_scenario = scenario_id.to_string();
        for ta in container.all("time_area") {
            self.time_area_action(ta)?;
        }
        self.side_unit_modifications(&container);
        self.firing_scenario = String::new();
        let events = Self::collect_events(&container, scenario_id);
        self.fired = vec![false; events.len()];
        self.events = Arc::new(events);
        self.wml_vars = self.sides.iter().enumerate()
            .map(|(i, s)| (format!("p{}_faction", i + 1), s.faction.clone())).collect();
        if !self.events.is_empty() {
            self.fire("prestart")?;
            self.fire("start")?;
        }
        Ok(())
    }

    /// Events a Python setup already built: (raw name, first_time_only,
    /// actions, scenario id, fired) each, with the WML and location
    /// variables they left.
    #[pyo3(signature = (events, wml_vars, scenario_vars, strict=false))]
    #[allow(clippy::type_complexity)]
    fn load_events(&mut self, events: Vec<(String, bool, Vec<Wml>, String, bool)>,
                   wml_vars: Vec<(String, String)>, scenario_vars: Vec<(String, Vec<(i64, i64)>)>,
                   strict: bool) {
        self.strict_wml = strict;
        self.fired = events.iter().map(|e| e.4).collect();
        self.events = Arc::new(events.into_iter().map(|(name, first, actions, sid, _)| EventDef {
            names: event_names(&name),
            first_time_only: first,
            actions: actions.into_iter().map(Arc::new).collect(),
            scenario_id: sid,
        }).collect());
        self.wml_vars = wml_vars.into_iter().collect();
        self.scenario_vars = scenario_vars.into_iter().map(|(k, v)| (k, v.into_iter().collect())).collect();
    }

    /// Fire the events on `names`, in order (the differential tests'
    /// handle on `fire_events`).
    fn fire_events(&mut self, names: Vec<String>) -> PyResult<()> {
        self.fire_all(&names)
    }

    /// (fired latches, WML variables, stored locations).
    #[allow(clippy::type_complexity)]
    fn events_export(&self) -> (Vec<bool>, Vec<(String, String)>, Vec<(String, Vec<(i64, i64)>)>) {
        (self.fired.clone(),
         self.wml_vars.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
         self.scenario_vars.iter().map(|(k, v)| (k.clone(), v.iter().copied().collect())).collect())
    }

    fn n_events(&self) -> usize {
        self.events.len()
    }

    /// The terrain writes of this game's events, in order: (WML x, WML y,
    /// code), off-map ones included.
    fn terrain_log(&self) -> Vec<(i64, i64, String)> {
        self.terrain_log.clone()
    }

    /// Bumped by every terrain change and time area.
    #[getter]
    fn map_version(&self) -> i64 {
        self.map_version
    }

    /// The time areas: (x, y, cycle phased to turn 1) per hex.
    fn time_areas_export(&self) -> Vec<(i64, i64, Vec<i64>)> {
        let m = &self.map;
        (0..m.h).filter(|&i| m.area_cycle[i] >= 0)
            .map(|i| (m.hx[i], m.hy[i], m.cycles[m.area_cycle[i] as usize].clone())).collect()
    }
}

/// Tag -> times an event action had no handler, since the last reset.
#[pyfunction]
pub fn unmodelled_action_counts() -> BTreeMap<String, i64> {
    UNMODELLED_COUNTS.lock().unwrap().clone()
}

#[pyfunction]
pub fn reset_unmodelled_actions() {
    UNMODELLED_COUNTS.lock().unwrap().clear();
    *UNMODELLED_SEEN.lock().unwrap() = None;
}
