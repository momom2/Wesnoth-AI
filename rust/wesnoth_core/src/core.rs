//! Phase 4: the game state owned by Rust (docs/rust_port_plan.md).
//!
//! `GameCore` holds what `wesnoth_ai.classes.GameState` holds and the
//! per-fork stash the simulator keeps on `global_info`: the units (with
//! the per-unit facts Python kept as underscore attributes: the defense
//! table, the pick-advance list, the feeding count, the trait order, the
//! persistent [object] effects, the WML role and the guardian flag), the
//! sides, the turn scalars, the village owners, the uncovered hiders,
//! the turn's recruit rejections and each side's cleared hexes. What
//! never changes within a game is shared across forks behind `Arc`: the
//! map (`MapStatic`: the hex set's geometry and one-class view from
//! `wesnoth_ai.game_core`, every terrain fact resolved here from the
//! hexes' terrain codes), the unit types (the process's unit database,
//! db.rs, registered per core on first use) and the movement classes
//! (per unit type, slowed status and defense table: movement cost,
//! defense subcost and defense percentage per hex, computed here from
//! the terrain codes by terrain.rs). `fork` is a clone: the dynamic part
//! copies, the static part is a reference count.
//!
//! Python constructs units (recruits, advancements, plague corpses:
//! `tools/replay_dataset.py` and `tools/traits.py`) and runs scenario
//! events on a Python view; everything else about a command applies
//! here (core_step.rs). Map space throughout: hex index = position in
//! `gs.map.hexes` (the observation kernels' order).

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

pub use crate::db::{TypeAttack as BaseAttack, UnitType as TypeRec};
use crate::db::{terrain_db, unit_db, TerrainDb, UnitDb};
use crate::terrain;
use crate::wml::Wml;

#[derive(Clone, Debug)]
pub struct AttackRec {
    pub type_id: i64,
    pub strikes: i64,
    pub damage: i64,
    pub ranged: bool,
    pub specials: Vec<String>,
}

/// A defense table: chance to be hit per terrain id, in its dict order.
pub type DefTable = Vec<(String, i64)>;

#[derive(Clone, Debug)]
pub struct UnitRec {
    pub id: String,
    pub name: String,
    pub type_idx: i64,               // into the core's type table
    pub name_id: i64,
    pub side: i64,
    pub is_leader: bool,
    pub x: i64,
    pub y: i64,
    pub hex: i64,                    // map index or -1
    pub max_hp: i64,
    pub max_moves: i64,
    pub max_exp: i64,
    pub cost: i64,
    pub alignment: i64,
    pub levelup_names: Vec<String>,
    pub current_hp: i64,
    pub current_moves: i64,
    pub current_exp: i64,
    pub has_attacked: bool,
    pub attacks: Vec<AttackRec>,
    pub resistances: Vec<f64>,
    pub defenses: Vec<f64>,
    pub movement_costs: Vec<i64>,
    pub abilities: Vec<String>,      // sorted
    pub traits: Vec<String>,         // sorted
    pub statuses: Vec<String>,       // sorted
    pub class_id: i64,               // movement class
    pub class_slowed_id: i64,
    // The per-unit facts Python kept as underscore attributes.
    pub def_table: Option<Arc<DefTable>>,        // `_defense_table`
    pub pickadvance: Option<Vec<String>>,        // `_pickadvance`
    pub feeding_count: Option<i64>,              // `_feeding_count`
    pub trait_order: Option<Vec<String>>,        // `_trait_order`
    pub object_effects: Vec<Arc<Wml>>,           // `_object_effects`
    pub wml_role: Option<String>,                // `_wml_role`
    pub ai_guardian: bool,                       // `_ai_guardian`
}

impl UnitRec {
    pub fn has_status(&self, s: &str) -> bool {
        self.statuses.iter().any(|x| x == s)
    }
    pub fn has_ability(&self, s: &str) -> bool {
        self.abilities.iter().any(|x| x == s)
    }
    pub fn has_trait(&self, s: &str) -> bool {
        self.traits.iter().any(|x| x == s)
    }
    pub fn add_status(&mut self, s: &str) {
        if !self.has_status(s) {
            self.statuses.push(s.to_string());
            self.statuses.sort();
        }
    }
    pub fn drop_status(&mut self, s: &str) {
        self.statuses.retain(|x| x != s);
    }
}

#[derive(Clone, Debug)]
pub struct SideRec {
    pub player: String,
    pub recruits: Vec<String>,
    pub current_gold: i64,
    pub base_income: i64,
    pub nb_villages: i64,
    pub faction: String,
}

#[derive(Clone, Debug, Default)]
pub struct GlobalRec {
    pub current_side: i64,
    pub turn_number: i64,
    pub time_of_day: String,
    pub village_gold: i64,
    pub village_upkeep: i64,
    pub base_income: i64,
    pub fog_on: bool,
    pub did_first_init_side: bool,
    pub tod_start_offset: i64,
    pub experience_modifier: i64,
    pub next_uid_counter: i64,
    pub rng_request_counter: i64,
    pub advance_uniform: bool,
    pub advance_salt: String,
    pub advance_counter: i64,
}

/// Per (unit type, slowed, defense table) and per map: the pathfinder's
/// arrays and the defense percentage on every hex.
#[derive(Clone, Debug)]
pub struct ClassRec {
    pub mcost: Vec<i64>,
    pub dsub: Vec<i64>,
    pub defense_pct: Vec<i64>,
}

/// A movement class's identity: the type (its movement costs), the
/// slowed status and the defense table's content, sorted.
pub type ClassKey = (String, bool, DefTable);

/// The map's static facts in map space.
#[derive(Debug)]
pub struct MapStatic {
    pub h: usize,
    pub hx: Vec<i64>,
    pub hy: Vec<i64>,
    pub nbrs: Vec<i64>,              // [H*6]
    pub pos_index: HashMap<(i64, i64), usize>,
    pub codes: Vec<String>,          // the hex's terrain code, start label dropped; "" = none
    pub castle_or_keep: Vec<u8>,
    pub keep: Vec<u8>,
    pub village_terrain: Vec<u8>,    // Terrain.VILLAGE in the hex's types
    pub village_mod: Vec<u8>,        // TerrainModifiers.VILLAGE (capturable)
    pub terrain_type_id: Vec<i64>,   // the encoder's one terrain id per hex
    pub heal: Vec<i64>,              // terrain_resolver.terrain_heals
    pub light_mod: Vec<i64>,
    pub light_max: Vec<i64>,
    pub light_min: Vec<i64>,
    pub has_light: Vec<u8>,
    pub area_cycle: Vec<i64>,        // index into cycles, -1 = default cycle
    pub cycles: Vec<Vec<i64>>,       // lawful bonus per turn phase
    // Hide-ability cover, from the engine's own [hides] terrain globs
    // (terrain_resolver.hides_cover: *^F*, *^V*, Wo*^*), NOT from the
    // hex's defense class. Read only by core_move::hide_cover_active.
    pub hides_ambush: Vec<u8>,
    pub hides_concealment: Vec<u8>,
    pub hides_submerge: Vec<u8>,
    pub full_slot: Vec<i64>,         // the full-board token slot of each hex
    pub castle_mod: Vec<u8>,         // TerrainModifiers.CASTLE (the encoder's static bit)
    pub hex_of_slot: Vec<usize>,     // the map hex of each full-board slot
}

pub const DEFAULT_CYCLE: [i64; 6] = [0, 25, 25, 0, -25, -25];
pub const TOD_NAMES: [&str; 6] = ["dawn", "morning", "afternoon", "dusk", "first_watch", "second_watch"];

#[pyclass]
#[derive(Clone)]
pub struct GameCore {
    pub map: Arc<MapStatic>,
    pub db: Arc<UnitDb>,
    pub tdb: Arc<TerrainDb>,
    pub types: Arc<RwLock<Vec<Arc<TypeRec>>>>,
    pub type_index: Arc<RwLock<HashMap<String, usize>>>,
    pub classes: Arc<RwLock<Vec<ClassRec>>>,
    pub class_index: Arc<RwLock<HashMap<ClassKey, usize>>>,
    pub game_id: String,
    pub size_x: i64,
    pub size_y: i64,
    pub units: Vec<UnitRec>,
    pub unit_index: HashMap<String, usize>,
    pub sides: Vec<SideRec>,
    pub global: GlobalRec,
    pub village_owner: Vec<i64>,     // [H] side or 0
    pub uncovered: Vec<String>,      // sorted
    pub fog_cleared: Vec<Vec<u8>>,   // per side (index side - 1): [H] cleared hexes, empty = untracked
    pub recruit_rejected: Vec<u8>,   // [H]
    pub advance_choices: Vec<i64>,
    pub pickadvance_game: Vec<(i64, String, Vec<String>)>,
    pub last_advance_events: Vec<(i64, i64)>,
    pub last_move_walk: Option<(i64, i64, i64, i64, String)>,   // ordered, landed, reason
    pub last_checkup_strikes: Vec<i64>,                          // (chance, hits, damage, dies) per strike
    pub game_over: bool,
    pub winner: i64,                 // -1 none
}

fn get<'py, T: FromPyObject<'py>>(d: &Bound<'py, PyDict>, key: &str) -> PyResult<T> {
    match d.get_item(key)? {
        Some(v) => v.extract(),
        None => Err(pyo3::exceptions::PyKeyError::new_err(key.to_string())),
    }
}

fn get_or<'py, T: FromPyObject<'py>>(d: &Bound<'py, PyDict>, key: &str, default: T) -> PyResult<T> {
    match d.get_item(key)? {
        Some(v) if !v.is_none() => v.extract(),
        _ => Ok(default),
    }
}

fn get_opt<'py, T: FromPyObject<'py>>(d: &Bound<'py, PyDict>, key: &str) -> PyResult<Option<T>> {
    match d.get_item(key)? {
        Some(v) if !v.is_none() => Ok(Some(v.extract()?)),
        _ => Ok(None),
    }
}

pub(crate) fn sorted(mut v: Vec<String>) -> Vec<String> {
    v.sort();
    v.dedup();
    v
}

/// A 64-bit mix (splitmix64 finalizer) over a running state.
#[derive(Default)]
pub struct Hasher(u64);

impl Hasher {
    pub fn add(&mut self, v: u64) {
        let mut z = self.0 ^ v.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        self.0 = z ^ (z >> 31);
    }
    pub fn add_i(&mut self, v: i64) {
        self.add(v as u64);
    }
    pub fn add_str(&mut self, s: &str) {
        self.add(s.len() as u64);
        for b in s.as_bytes() {
            self.add(*b as u64);
        }
    }
    pub fn value(&self) -> i64 {
        self.0 as i64
    }
}

impl GameCore {
    /// The index of a unit type in this core's table, registered from
    /// the unit database on first use (`_stats_for`: the fallback under
    /// its own name for a type the scrape lacks).
    pub fn type_idx(&self, name: &str) -> usize {
        if let Some(&i) = self.type_index.read().unwrap().get(name) {
            return i;
        }
        let mut index = self.type_index.write().unwrap();
        if let Some(&i) = index.get(name) {
            return i;
        }
        let mut types = self.types.write().unwrap();
        types.push(self.db.get(name));
        index.insert(name.to_string(), types.len() - 1);
        types.len() - 1
    }

    pub fn type_rec(&self, idx: i64) -> Arc<TypeRec> {
        self.types.read().unwrap()[idx as usize].clone()
    }

    /// The defense table a unit's class reads: its own, or its type's
    /// when it has none (`getattr(u, "_defense_table", None) or
    /// _stats_for(u.name)["defense"]`).
    pub fn class_table(&self, u: &UnitRec) -> DefTable {
        match &u.def_table {
            Some(t) if !t.is_empty() => t.as_ref().clone(),
            _ => self.type_rec(u.type_idx).defense.clone(),
        }
    }

    /// The movement class of (type, slowed, defense table), computed
    /// once per content and shared by every fork (`_class_id`: the
    /// pathfinder's `_terrain_arrays_for` and `_terrain_def_pct`).
    pub fn class_for(&self, type_idx: i64, slowed: bool, table: &DefTable) -> i64 {
        let t = self.type_rec(type_idx);
        let mut content = table.clone();
        content.sort();
        let key: ClassKey = (t.name.clone(), slowed, content);
        if let Some(&c) = self.class_index.read().unwrap().get(&key) {
            return c as i64;
        }
        let rec = self.compute_class(&t, slowed, table);
        let mut index = self.class_index.write().unwrap();
        if let Some(&c) = index.get(&key) {
            return c as i64;
        }
        let mut classes = self.classes.write().unwrap();
        classes.push(rec);
        index.insert(key, classes.len() - 1);
        (classes.len() - 1) as i64
    }

    /// One class's arrays over the map: `_move_cost_at_hex` (the type's
    /// movement costs, doubled below UNREACHABLE when slowed), the
    /// defense subcost `defense_pct_at` and the combat defense
    /// `_terrain_def_pct`; a hex with no terrain code reads the flat key.
    fn compute_class(&self, t: &TypeRec, slowed: bool, table: &DefTable) -> ClassRec {
        let costs: DefTable = t.movement_costs.iter()
            .map(|(k, v)| (k.clone(), if slowed && *v < terrain::UNREACHABLE_COST { 2 * v } else { *v }))
            .collect();
        let h = self.map.h;
        let mut rec = ClassRec { mcost: vec![0; h], dsub: vec![0; h], defense_pct: vec![0; h] };
        let mut memo: HashMap<&str, (i64, i64)> = HashMap::new();
        let flat = |tab: &DefTable| tab.iter().find(|(k, _)| k == "flat").map(|(_, v)| *v);
        for i in 0..h {
            let code = self.map.codes[i].as_str();
            if code.is_empty() {
                rec.mcost[i] = match flat(&costs) { Some(v) if v != 0 => v, _ => 1 };
                rec.dsub[i] = match flat(table) { Some(v) if v != 0 => v, _ => 50 };
                rec.defense_pct[i] = flat(table).unwrap_or(50);
                continue;
            }
            let (m, d) = *memo.entry(code).or_insert_with(|| {
                (terrain::mvt_cost(&self.tdb, code, &costs), terrain::def_pct(&self.tdb, code, table))
            });
            rec.mcost[i] = m;
            rec.dsub[i] = d;
            rec.defense_pct[i] = d;
        }
        rec
    }

    /// Set the unit's type index, hex and movement classes from its
    /// name, position and defense table.
    pub fn place_unit_facts(&self, u: &mut UnitRec) {
        u.type_idx = self.type_idx(&u.name) as i64;
        u.hex = self.map.pos_index.get(&(u.x, u.y)).map(|&i| i as i64).unwrap_or(-1);
        let table = self.class_table(u);
        u.class_id = self.class_for(u.type_idx, false, &table);
        u.class_slowed_id = self.class_for(u.type_idx, true, &table);
    }

    /// Add a unit record (its type, hex and classes set here).
    pub fn insert_unit(&mut self, mut rec: UnitRec) -> PyResult<usize> {
        if self.unit_index.contains_key(&rec.id) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!("duplicate unit id {}", rec.id)));
        }
        self.place_unit_facts(&mut rec);
        let id = rec.id.clone();
        self.units.push(rec);
        let idx = self.units.len() - 1;
        self.unit_index.insert(id, idx);
        Ok(idx)
    }
}

/// The terrain facts of a map resolved from its codes (heal, light,
/// hide cover) into `MapStatic`'s arrays.
fn resolve_terrain_facts(tdb: &TerrainDb, codes: &[String]) -> [Vec<i64>; 8] {
    let h = codes.len();
    let mut out: [Vec<i64>; 8] = Default::default();
    for v in out.iter_mut() {
        *v = vec![0; h];
    }
    for (i, code) in codes.iter().enumerate() {
        if code.is_empty() {
            continue;
        }
        let (light, max_l, min_l, any) = terrain::light_params(tdb, code);
        out[0][i] = terrain::terrain_heals(tdb, code);
        out[1][i] = light;
        out[2][i] = max_l;
        out[3][i] = min_l;
        out[4][i] = any as i64;
        out[5][i] = terrain::hides_cover(code, "ambush") as i64;
        out[6][i] = terrain::hides_cover(code, "concealment") as i64;
        out[7][i] = terrain::hides_cover(code, "submerge") as i64;
    }
    out
}

fn as_u8(v: &[i64]) -> Vec<u8> {
    v.iter().map(|&x| x as u8).collect()
}

fn table_of(v: Option<Vec<(String, i64)>>) -> Option<Arc<DefTable>> {
    v.map(Arc::new)
}

#[pymethods]
impl GameCore {
    /// The core of one game over a map: `map` carries the geometry, the
    /// one-class view and the terrain codes (wesnoth_ai.game_core.map_static).
    /// Units, sides and globals are added by the setters below. The unit
    /// and terrain databases must be loaded (`load_databases`).
    #[new]
    fn new(map: &Bound<'_, PyDict>, game_id: String, size_x: i64, size_y: i64) -> PyResult<Self> {
        let db = unit_db()?;
        let tdb = terrain_db()?;
        let hx: Vec<i64> = get(map, "hx")?;
        let hy: Vec<i64> = get(map, "hy")?;
        let h = hx.len();
        let mut pos_index = HashMap::with_capacity(h);
        for i in 0..h {
            pos_index.insert((hx[i], hy[i]), i);
        }
        let cycles: Vec<Vec<i64>> = get_or(map, "cycles", Vec::new())?;
        let full_slot: Vec<i64> = get(map, "full_slot")?;
        let mut hex_of_slot = vec![usize::MAX; h];
        for (m, &t) in full_slot.iter().enumerate() {
            if t < 0 || t as usize >= h || hex_of_slot[t as usize] != usize::MAX {
                return Err(pyo3::exceptions::PyValueError::new_err("full_slot is not a permutation"));
            }
            hex_of_slot[t as usize] = m;
        }
        let raw_codes: Vec<String> = get(map, "codes")?;
        let codes: Vec<String> = raw_codes.iter().map(|c| terrain::strip_start_position(c).to_string()).collect();
        let [heal, light_mod, light_max, light_min, has_light, ambush, concealment, submerge] =
            resolve_terrain_facts(&tdb, &codes);
        let map_static = MapStatic {
            h,
            hx,
            hy,
            nbrs: get(map, "nbrs")?,
            pos_index,
            codes,
            castle_or_keep: get(map, "castle_or_keep")?,
            keep: get(map, "keep")?,
            village_terrain: get(map, "village_terrain")?,
            village_mod: get(map, "village_mod")?,
            terrain_type_id: get(map, "terrain_type_id")?,
            heal,
            light_mod,
            light_max,
            light_min,
            has_light: as_u8(&has_light),
            area_cycle: get(map, "area_cycle")?,
            cycles,
            hides_ambush: as_u8(&ambush),
            hides_concealment: as_u8(&concealment),
            hides_submerge: as_u8(&submerge),
            full_slot,
            castle_mod: get(map, "castle_mod")?,
            hex_of_slot,
        };
        for (name, v) in [
            ("nbrs", map_static.nbrs.len() / 6), ("codes", map_static.codes.len()),
            ("castle_or_keep", map_static.castle_or_keep.len()), ("keep", map_static.keep.len()),
            ("village_terrain", map_static.village_terrain.len()), ("village_mod", map_static.village_mod.len()),
            ("terrain_type_id", map_static.terrain_type_id.len()), ("area_cycle", map_static.area_cycle.len()),
            ("full_slot", map_static.full_slot.len()), ("castle_mod", map_static.castle_mod.len()),
        ] {
            if v != h {
                return Err(pyo3::exceptions::PyValueError::new_err(format!("map array {name}: {v} != {h}")));
            }
        }
        Ok(GameCore {
            map: Arc::new(map_static),
            db,
            tdb,
            types: Arc::new(RwLock::new(Vec::new())),
            type_index: Arc::new(RwLock::new(HashMap::new())),
            classes: Arc::new(RwLock::new(Vec::new())),
            class_index: Arc::new(RwLock::new(HashMap::new())),
            game_id,
            size_x,
            size_y,
            units: Vec::new(),
            unit_index: HashMap::new(),
            sides: Vec::new(),
            global: GlobalRec::default(),
            village_owner: vec![0; h],
            uncovered: Vec::new(),
            fog_cleared: Vec::new(),
            recruit_rejected: vec![0; h],
            advance_choices: Vec::new(),
            pickadvance_game: Vec::new(),
            last_advance_events: Vec::new(),
            last_move_walk: None,
            last_checkup_strikes: Vec::new(),
            game_over: false,
            winner: -1,
        })
    }

    #[getter]
    fn h(&self) -> usize {
        self.map.h
    }

    #[getter]
    fn current_side(&self) -> i64 {
        self.global.current_side
    }

    #[getter]
    fn turn_number(&self) -> i64 {
        self.global.turn_number
    }

    /// The index of a unit type, registered on first use; shared by
    /// every fork of this core.
    fn type_index_of(&self, name: &str) -> usize {
        self.type_idx(name)
    }

    fn n_classes(&self) -> usize {
        self.classes.read().unwrap().len()
    }

    /// One movement class's (mcost, dsub, defense_pct) arrays: the
    /// differential tests' handle on `class_for`.
    fn class_arrays(&self, id: usize) -> (Vec<i64>, Vec<i64>, Vec<i64>) {
        let c = &self.classes.read().unwrap()[id];
        (c.mcost.clone(), c.dsub.clone(), c.defense_pct.clone())
    }

    /// The resolved terrain facts per hex: (heal, light_mod, light_max,
    /// light_min, has_light, ambush, concealment, submerge).
    #[allow(clippy::type_complexity)]
    fn terrain_arrays(&self) -> (Vec<i64>, Vec<i64>, Vec<i64>, Vec<i64>, Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
        let m = &self.map;
        (m.heal.clone(), m.light_mod.clone(), m.light_max.clone(), m.light_min.clone(), m.has_light.clone(),
         m.hides_ambush.clone(), m.hides_concealment.clone(), m.hides_submerge.clone())
    }

    /// Add a unit from its field dict (wesnoth_ai.game_core.unit_fields).
    fn add_unit(&mut self, u: &Bound<'_, PyDict>) -> PyResult<usize> {
        let rec = unit_from_dict(u)?;
        self.insert_unit(rec)
    }

    pub fn remove_unit(&mut self, id: &str) -> PyResult<()> {
        let idx = match self.unit_index.get(id) {
            Some(&i) => i,
            None => return Err(pyo3::exceptions::PyKeyError::new_err(id.to_string())),
        };
        self.units.swap_remove(idx);
        self.unit_index.remove(id);
        if idx < self.units.len() {
            let moved = self.units[idx].id.clone();
            self.unit_index.insert(moved, idx);
        }
        Ok(())
    }

    fn n_units(&self) -> usize {
        self.units.len()
    }

    fn clear_units(&mut self) {
        self.units.clear();
        self.unit_index.clear();
    }

    fn unit_ids(&self) -> Vec<String> {
        self.units.iter().map(|u| u.id.clone()).collect()
    }

    /// One unit's fields as a dict (the adapter rebuilds the dataclass).
    fn unit_export<'py>(&self, py: Python<'py>, id: &str) -> PyResult<Bound<'py, PyDict>> {
        let idx = *self.unit_index.get(id)
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(id.to_string()))?;
        unit_dict(py, &self.units[idx])
    }

    fn units_export<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let out = PyList::empty(py);
        for u in &self.units {
            out.append(unit_dict(py, u)?)?;
        }
        Ok(out)
    }

    /// Change a unit's dynamic fields (the keys present in `changes`).
    fn update_unit(&mut self, id: &str, changes: &Bound<'_, PyDict>) -> PyResult<()> {
        let idx = *self.unit_index.get(id)
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(id.to_string()))?;
        let u = &mut self.units[idx];
        if let Some(v) = changes.get_item("x")? { u.x = v.extract()?; }
        if let Some(v) = changes.get_item("y")? { u.y = v.extract()?; }
        if changes.contains("x")? || changes.contains("y")? {
            u.hex = self.map.pos_index.get(&(u.x, u.y)).map(|&i| i as i64).unwrap_or(-1);
        }
        if let Some(v) = changes.get_item("max_hp")? { u.max_hp = v.extract()?; }
        if let Some(v) = changes.get_item("max_moves")? { u.max_moves = v.extract()?; }
        if let Some(v) = changes.get_item("max_exp")? { u.max_exp = v.extract()?; }
        if let Some(v) = changes.get_item("current_hp")? { u.current_hp = v.extract()?; }
        if let Some(v) = changes.get_item("current_moves")? { u.current_moves = v.extract()?; }
        if let Some(v) = changes.get_item("current_exp")? { u.current_exp = v.extract()?; }
        if let Some(v) = changes.get_item("has_attacked")? { u.has_attacked = v.extract()?; }
        if let Some(v) = changes.get_item("statuses")? { u.statuses = sorted(v.extract()?); }
        if let Some(v) = changes.get_item("traits")? { u.traits = sorted(v.extract()?); }
        if let Some(v) = changes.get_item("abilities")? { u.abilities = sorted(v.extract()?); }
        if let Some(v) = changes.get_item("pickadvance")? { u.pickadvance = v.extract()?; }
        if let Some(v) = changes.get_item("feeding_count")? { u.feeding_count = v.extract()?; }
        Ok(())
    }

    /// `team::spend_gold`: bare subtraction, no clamp (a recruit's cost).
    pub fn spend_gold(&mut self, side: i64, amount: i64) {
        if side >= 1 && (side as usize) <= self.sides.len() {
            self.sides[side as usize - 1].current_gold -= amount;
        }
    }

    fn set_sides(&mut self, sides: Vec<(String, Vec<String>, i64, i64, i64, String)>) {
        self.sides = sides.into_iter().map(|(p, r, g, b, v, f)| SideRec {
            player: p, recruits: r, current_gold: g, base_income: b, nb_villages: v, faction: f,
        }).collect();
    }

    fn sides_export(&self) -> Vec<(String, Vec<String>, i64, i64, i64, String)> {
        self.sides.iter().map(|s| (s.player.clone(), s.recruits.clone(), s.current_gold,
                                   s.base_income, s.nb_villages, s.faction.clone())).collect()
    }

    fn set_globals(&mut self, g: &Bound<'_, PyDict>) -> PyResult<()> {
        self.global = GlobalRec {
            current_side: get(g, "current_side")?,
            turn_number: get(g, "turn_number")?,
            time_of_day: get(g, "time_of_day")?,
            village_gold: get(g, "village_gold")?,
            village_upkeep: get(g, "village_upkeep")?,
            base_income: get(g, "base_income")?,
            fog_on: get_or(g, "fog_on", true)?,
            did_first_init_side: get_or(g, "did_first_init_side", false)?,
            tod_start_offset: get_or(g, "tod_start_offset", 0)?,
            experience_modifier: get_or(g, "experience_modifier", 100)?,
            next_uid_counter: get_or(g, "next_uid_counter", 1)?,
            rng_request_counter: get_or(g, "rng_request_counter", 0)?,
            advance_uniform: get_or(g, "advance_uniform", false)?,
            advance_salt: get_or(g, "advance_salt", String::new())?,
            advance_counter: get_or(g, "advance_counter", 0)?,
        };
        self.game_over = get_or(g, "game_over", false)?;
        self.winner = get_or(g, "winner", -1)?;
        Ok(())
    }

    /// One integer global by name: next_uid_counter, advance_counter,
    /// rng_request_counter (what the Python builders advance),
    /// advance_uniform (0/1), current_side.
    fn set_global_int(&mut self, name: &str, value: i64) -> PyResult<()> {
        match name {
            "next_uid_counter" => self.global.next_uid_counter = value,
            "advance_counter" => self.global.advance_counter = value,
            "rng_request_counter" => self.global.rng_request_counter = value,
            "advance_uniform" => self.global.advance_uniform = value != 0,
            "current_side" => self.global.current_side = value,
            _ => return Err(pyo3::exceptions::PyKeyError::new_err(name.to_string())),
        }
        Ok(())
    }

    fn globals_export<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        let g = &self.global;
        d.set_item("current_side", g.current_side)?;
        d.set_item("turn_number", g.turn_number)?;
        d.set_item("time_of_day", &g.time_of_day)?;
        d.set_item("village_gold", g.village_gold)?;
        d.set_item("village_upkeep", g.village_upkeep)?;
        d.set_item("base_income", g.base_income)?;
        d.set_item("fog_on", g.fog_on)?;
        d.set_item("did_first_init_side", g.did_first_init_side)?;
        d.set_item("tod_start_offset", g.tod_start_offset)?;
        d.set_item("experience_modifier", g.experience_modifier)?;
        d.set_item("next_uid_counter", g.next_uid_counter)?;
        d.set_item("rng_request_counter", g.rng_request_counter)?;
        d.set_item("advance_uniform", g.advance_uniform)?;
        d.set_item("advance_salt", &g.advance_salt)?;
        d.set_item("advance_counter", g.advance_counter)?;
        d.set_item("game_over", self.game_over)?;
        d.set_item("winner", self.winner)?;
        Ok(d)
    }

    fn set_village_owner(&mut self, entries: Vec<(i64, i64, i64)>) -> PyResult<()> {
        self.village_owner = vec![0; self.map.h];
        for (x, y, side) in entries {
            if let Some(&i) = self.map.pos_index.get(&(x, y)) {
                self.village_owner[i] = side;
            } else {
                return Err(pyo3::exceptions::PyValueError::new_err(format!("village owner off map: ({x},{y})")));
            }
        }
        Ok(())
    }

    fn village_owner_export(&self) -> Vec<(i64, i64, i64)> {
        (0..self.map.h).filter(|&i| self.village_owner[i] != 0)
            .map(|i| (self.map.hx[i], self.map.hy[i], self.village_owner[i])).collect()
    }

    fn set_uncovered(&mut self, ids: Vec<String>) {
        self.uncovered = sorted(ids);
    }

    fn uncovered_export(&self) -> Vec<String> {
        self.uncovered.clone()
    }

    /// The hexes a recruit bounced on this turn (`_recruit_rejected_hexes`).
    fn set_recruit_rejected(&mut self, hexes: Vec<(i64, i64)>) {
        self.recruit_rejected = vec![0; self.map.h];
        for (x, y) in hexes {
            if let Some(&i) = self.map.pos_index.get(&(x, y)) { self.recruit_rejected[i] = 1; }
        }
    }

    fn recruit_rejected_hexes(&self) -> Vec<(i64, i64)> {
        (0..self.map.h).filter(|&i| self.recruit_rejected[i] != 0)
            .map(|i| (self.map.hx[i], self.map.hy[i])).collect()
    }

    fn set_advance_state(&mut self, choices: Vec<i64>, pickadvance: Vec<(i64, String, Vec<String>)>,
                         last_events: Vec<(i64, i64)>) {
        self.advance_choices = choices;
        self.pickadvance_game = pickadvance;
        self.last_advance_events = last_events;
    }

    fn advance_state_export(&self) -> (Vec<i64>, Vec<(i64, String, Vec<String>)>, Vec<(i64, i64)>) {
        (self.advance_choices.clone(), self.pickadvance_game.clone(), self.last_advance_events.clone())
    }

    #[pyo3(signature = (walk=None))]
    fn set_last_move_walk(&mut self, walk: Option<(i64, i64, i64, i64, String)>) {
        self.last_move_walk = walk;
    }

    fn last_move_walk_export(&self) -> Option<(i64, i64, i64, i64, String)> {
        self.last_move_walk.clone()
    }

    fn set_last_checkup_strikes(&mut self, strikes: Vec<i64>) {
        self.last_checkup_strikes = strikes;
    }

    fn last_checkup_strikes_export(&self) -> Vec<i64> {
        self.last_checkup_strikes.clone()
    }

    /// A copy: the dynamic state is cloned, the map, the unit types and
    /// the movement classes are shared.
    fn fork(&self) -> GameCore {
        self.clone()
    }

    /// `classes.state_key`'s content over the same fields: the units
    /// (by id, sorted), the sides, the village owners, the uncovered
    /// set, the recruit rejections, the cleared hexes, the turn scalars. Equal
    /// states hash equal; a changed field changes it
    /// (tests/test_game_core.py).
    fn state_key(&self) -> i64 {
        let mut hs = Hasher::default();
        let mut order: Vec<usize> = (0..self.units.len()).collect();
        order.sort_by(|&a, &b| self.units[a].id.cmp(&self.units[b].id));
        for i in order {
            let u = &self.units[i];
            hs.add_str(&u.id);
            hs.add_i(u.side);
            hs.add_i(u.x);
            hs.add_i(u.y);
            hs.add_i(u.current_hp);
            hs.add_i(u.current_moves);
            hs.add_i(u.current_exp);
            hs.add(u.has_attacked as u64);
            for s in &u.statuses {
                hs.add_str(s);
            }
            hs.add_str(&u.name);
            hs.add(u.is_leader as u64);
        }
        for s in &self.sides {
            hs.add_str(&s.faction);
            hs.add_i(s.current_gold);
            hs.add_i(s.base_income);
            hs.add_i(s.nb_villages);
            for r in &s.recruits {
                hs.add_str(r);
            }
        }
        for i in 0..self.map.h {
            if self.village_owner[i] != 0 {
                hs.add_i(i as i64);
                hs.add_i(self.village_owner[i]);
            }
        }
        for u in &self.uncovered {
            hs.add_str(u);
        }
        for i in 0..self.map.h {
            if self.recruit_rejected[i] != 0 {
                hs.add_i(i as i64);
            }
        }
        for (k, v) in self.fog_cleared.iter().enumerate() {
            if v.is_empty() {
                continue;
            }
            hs.add_i(k as i64 + 1);
            for (j, &c) in v.iter().enumerate() {
                if c != 0 {
                    hs.add_i(j as i64);
                }
            }
        }
        let g = &self.global;
        hs.add_i(g.current_side);
        hs.add_i(g.turn_number);
        hs.add_str(&g.time_of_day);
        hs.add_i(g.village_gold);
        hs.add_i(g.village_upkeep);
        hs.add_i(g.base_income);
        hs.add_i(g.rng_request_counter);
        hs.value()
    }
}

/// A unit record from its field dict (`game_core.unit_fields`, or
/// `unit_dict`'s own output): the type index, hex and classes are left
/// for the core that takes it.
pub(crate) fn unit_from_dict(u: &Bound<'_, PyDict>) -> PyResult<UnitRec> {
    let attacks_in: Vec<(i64, i64, i64, bool, Vec<String>)> = get(u, "attacks")?;
    let effects: Vec<Wml> = get_or(u, "object_effects", Vec::new())?;
    Ok(UnitRec {
        type_idx: -1,
        id: get(u, "id")?,
        name: get(u, "name")?,
        name_id: get(u, "name_id")?,
        side: get(u, "side")?,
        is_leader: get(u, "is_leader")?,
        x: get(u, "x")?,
        y: get(u, "y")?,
        hex: -1,
        max_hp: get(u, "max_hp")?,
        max_moves: get(u, "max_moves")?,
        max_exp: get(u, "max_exp")?,
        cost: get(u, "cost")?,
        alignment: get(u, "alignment")?,
        levelup_names: get(u, "levelup_names")?,
        current_hp: get(u, "current_hp")?,
        current_moves: get(u, "current_moves")?,
        current_exp: get(u, "current_exp")?,
        has_attacked: get(u, "has_attacked")?,
        attacks: attacks_in.into_iter().map(|(t, n, d, r, sp)| AttackRec {
            type_id: t, strikes: n, damage: d, ranged: r, specials: sorted(sp),
        }).collect(),
        resistances: get(u, "resistances")?,
        defenses: get(u, "defenses")?,
        movement_costs: get(u, "movement_costs")?,
        abilities: sorted(get(u, "abilities")?),
        traits: sorted(get(u, "traits")?),
        statuses: sorted(get(u, "statuses")?),
        class_id: -1,
        class_slowed_id: -1,
        def_table: table_of(get_opt(u, "defense_table")?),
        pickadvance: get_opt(u, "pickadvance")?,
        feeding_count: get_opt(u, "feeding_count")?,
        trait_order: get_opt(u, "trait_order")?,
        object_effects: effects.into_iter().map(Arc::new).collect(),
        wml_role: get_opt(u, "wml_role")?,
        ai_guardian: get_or(u, "ai_guardian", false)?,
    })
}

pub(crate) fn unit_dict<'py>(py: Python<'py>, u: &UnitRec) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("id", &u.id)?;
    d.set_item("name", &u.name)?;
    d.set_item("name_id", u.name_id)?;
    d.set_item("side", u.side)?;
    d.set_item("is_leader", u.is_leader)?;
    d.set_item("x", u.x)?;
    d.set_item("y", u.y)?;
    d.set_item("max_hp", u.max_hp)?;
    d.set_item("max_moves", u.max_moves)?;
    d.set_item("max_exp", u.max_exp)?;
    d.set_item("cost", u.cost)?;
    d.set_item("alignment", u.alignment)?;
    d.set_item("levelup_names", u.levelup_names.clone())?;
    d.set_item("current_hp", u.current_hp)?;
    d.set_item("current_moves", u.current_moves)?;
    d.set_item("current_exp", u.current_exp)?;
    d.set_item("has_attacked", u.has_attacked)?;
    let attacks: Vec<(i64, i64, i64, bool, Vec<String>)> = u.attacks.iter()
        .map(|a| (a.type_id, a.strikes, a.damage, a.ranged, a.specials.clone())).collect();
    d.set_item("attacks", attacks)?;
    d.set_item("resistances", u.resistances.clone())?;
    d.set_item("defenses", u.defenses.clone())?;
    d.set_item("movement_costs", u.movement_costs.clone())?;
    d.set_item("abilities", u.abilities.clone())?;
    d.set_item("traits", u.traits.clone())?;
    d.set_item("statuses", u.statuses.clone())?;
    d.set_item("class_id", u.class_id)?;
    d.set_item("class_slowed_id", u.class_slowed_id)?;
    d.set_item("defense_table", u.def_table.as_ref().map(|t| t.as_ref().clone()))?;
    d.set_item("pickadvance", u.pickadvance.clone())?;
    d.set_item("feeding_count", u.feeding_count)?;
    d.set_item("trait_order", u.trait_order.clone())?;
    let effects = PyList::empty(py);
    for e in &u.object_effects {
        effects.append(e.to_py(py)?)?;
    }
    d.set_item("object_effects", effects)?;
    d.set_item("wml_role", u.wml_role.clone())?;
    d.set_item("ai_guardian", u.ai_guardian)?;
    Ok(d)
}
