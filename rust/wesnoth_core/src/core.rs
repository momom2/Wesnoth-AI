//! Phase 4: the game state owned by Rust (docs/rust_port_plan.md).
//!
//! `GameCore` holds what `wesnoth_ai.classes.GameState` holds and the
//! per-fork stash the simulator keeps on `global_info`: the units, the
//! sides, the turn scalars, the village owners, the uncovered hiders
//! and the per-turn rejection sets. What never changes within a game
//! is shared across forks behind `Arc`: the map (`MapStatic`, built
//! once by `wesnoth_ai.game_core` from the hex set, the terrain codes
//! and the time areas), the unit-type table and the movement classes
//! (per unit type and slowed status: movement cost, defense subcost
//! and defense percentage per hex, resolved by Python's terrain
//! resolver once per map). `fork` is a clone: the dynamic part copies,
//! the static part is a reference count.
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

#[derive(Clone, Debug)]
pub struct AttackRec {
    pub type_id: i64,
    pub strikes: i64,
    pub damage: i64,
    pub ranged: bool,
    pub specials: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct UnitRec {
    pub id: String,
    pub name: String,
    pub type_idx: i64,               // into TypeTable, -1 unknown
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
    pub class_id: i64,               // movement class, -1 unknown
    pub class_slowed_id: i64,
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

/// A base attack of a unit type (`unit_stats.json`): what the combat
/// snapshot takes from the type rather than from the unit.
#[derive(Clone, Debug)]
pub struct BaseAttack {
    pub type_name: String,
    pub ranged: bool,
    pub specials: Vec<String>,
    pub accuracy: i64,
    pub parry: i64,
}

#[derive(Clone, Debug)]
pub struct TypeRec {
    pub name: String,
    pub level: i64,
    pub alignment: i64,
    pub resist: [i64; 6],            // combat.DAMAGE_TYPES order
    pub abilities: Vec<String>,
    pub attacks: Vec<BaseAttack>,
    pub cost: i64,
    pub race: String,
    pub undead_variation: String,
    pub advances_to: Vec<String>,
}

/// Per (unit type, slowed, defense table) and per map: the pathfinder's
/// arrays and the defense percentage on every hex.
#[derive(Clone, Debug)]
pub struct ClassRec {
    pub mcost: Vec<i64>,
    pub dsub: Vec<i64>,
    pub defense_pct: Vec<i64>,
}

/// The map's static facts in map space.
#[derive(Debug)]
pub struct MapStatic {
    pub h: usize,
    pub hx: Vec<i64>,
    pub hy: Vec<i64>,
    pub nbrs: Vec<i64>,              // [H*6]
    pub pos_index: HashMap<(i64, i64), usize>,
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
    pub is_forest: Vec<u8>,          // defense keys of the hex (hide cover)
    pub is_village_key: Vec<u8>,
    pub is_deep_water: Vec<u8>,
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
    pub types: Arc<RwLock<Vec<TypeRec>>>,
    pub type_index: Arc<RwLock<HashMap<String, usize>>>,
    pub classes: Arc<RwLock<Vec<ClassRec>>>,
    pub game_id: String,
    pub size_x: i64,
    pub size_y: i64,
    pub units: Vec<UnitRec>,
    pub unit_index: HashMap<String, usize>,
    pub sides: Vec<SideRec>,
    pub global: GlobalRec,
    pub village_owner: Vec<i64>,     // [H] side or 0
    pub uncovered: Vec<String>,      // sorted
    pub recruit_rejected: Vec<u8>,   // [H]
    pub move_rejected: Vec<u8>,      // [H]
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

fn sorted(mut v: Vec<String>) -> Vec<String> {
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

#[pymethods]
impl GameCore {
    /// The core of one game over a map: `map` carries the static arrays
    /// (wesnoth_ai.game_core.map_static). Units, sides and globals are
    /// added by the setters below.
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(map: &Bound<'_, PyDict>, game_id: String, size_x: i64, size_y: i64) -> PyResult<Self> {
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
        let map_static = MapStatic {
            h,
            hx,
            hy,
            nbrs: get(map, "nbrs")?,
            pos_index,
            castle_or_keep: get(map, "castle_or_keep")?,
            keep: get(map, "keep")?,
            village_terrain: get(map, "village_terrain")?,
            village_mod: get(map, "village_mod")?,
            terrain_type_id: get(map, "terrain_type_id")?,
            heal: get(map, "heal")?,
            light_mod: get(map, "light_mod")?,
            light_max: get(map, "light_max")?,
            light_min: get(map, "light_min")?,
            has_light: get(map, "has_light")?,
            area_cycle: get(map, "area_cycle")?,
            cycles,
            is_forest: get(map, "is_forest")?,
            is_village_key: get(map, "is_village_key")?,
            is_deep_water: get(map, "is_deep_water")?,
            full_slot,
            castle_mod: get(map, "castle_mod")?,
            hex_of_slot,
        };
        for (name, v) in [
            ("nbrs", map_static.nbrs.len() / 6), ("castle_or_keep", map_static.castle_or_keep.len()),
            ("keep", map_static.keep.len()), ("village_terrain", map_static.village_terrain.len()),
            ("village_mod", map_static.village_mod.len()), ("terrain_type_id", map_static.terrain_type_id.len()),
            ("heal", map_static.heal.len()), ("light_mod", map_static.light_mod.len()),
            ("area_cycle", map_static.area_cycle.len()), ("is_forest", map_static.is_forest.len()),
            ("full_slot", map_static.full_slot.len()), ("castle_mod", map_static.castle_mod.len()),
        ] {
            if v != h {
                return Err(pyo3::exceptions::PyValueError::new_err(format!("map array {name}: {v} != {h}")));
            }
        }
        Ok(GameCore {
            map: Arc::new(map_static),
            types: Arc::new(RwLock::new(Vec::new())),
            type_index: Arc::new(RwLock::new(HashMap::new())),
            classes: Arc::new(RwLock::new(Vec::new())),
            game_id,
            size_x,
            size_y,
            units: Vec::new(),
            unit_index: HashMap::new(),
            sides: Vec::new(),
            global: GlobalRec::default(),
            village_owner: vec![0; h],
            uncovered: Vec::new(),
            recruit_rejected: vec![0; h],
            move_rejected: vec![0; h],
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

    /// A unit type (`tools.replay_dataset._stats_for`), once per name;
    /// returns its index. Shared by every fork of this core.
    fn register_type(&mut self, t: &Bound<'_, PyDict>) -> PyResult<usize> {
        let name: String = get(t, "name")?;
        if let Some(&i) = self.type_index.read().unwrap().get(&name) {
            return Ok(i);
        }
        let resist_v: Vec<i64> = get(t, "resist")?;
        if resist_v.len() != 6 {
            return Err(pyo3::exceptions::PyValueError::new_err("resist needs 6 entries"));
        }
        let mut resist = [100i64; 6];
        resist.copy_from_slice(&resist_v);
        let attacks_in: Vec<(String, bool, Vec<String>, i64, i64)> = get(t, "attacks")?;
        let rec = TypeRec {
            name: name.clone(),
            level: get(t, "level")?,
            alignment: get(t, "alignment")?,
            resist,
            abilities: sorted(get(t, "abilities")?),
            attacks: attacks_in.into_iter().map(|(ty, r, sp, acc, par)| BaseAttack {
                type_name: ty, ranged: r, specials: sp, accuracy: acc, parry: par,
            }).collect(),
            cost: get(t, "cost")?,
            race: get_or(t, "race", String::new())?,
            undead_variation: get_or(t, "undead_variation", String::new())?,
            advances_to: get_or(t, "advances_to", Vec::new())?,
        };
        let mut types = self.types.write().unwrap();
        types.push(rec);
        let idx = types.len() - 1;
        self.type_index.write().unwrap().insert(name, idx);
        Ok(idx)
    }

    fn type_index_of(&self, name: &str) -> i64 {
        self.type_index.read().unwrap().get(name).map(|&i| i as i64).unwrap_or(-1)
    }

    /// A movement class: the pathfinder's cost and subcost per hex and
    /// the defense percentage per hex for one (type, slowed, defense
    /// table). Returns its id; shared by every fork.
    fn register_class(&mut self, mcost: Vec<i64>, dsub: Vec<i64>, defense_pct: Vec<i64>) -> PyResult<usize> {
        let h = self.map.h;
        if mcost.len() != h || dsub.len() != h || defense_pct.len() != h {
            return Err(pyo3::exceptions::PyValueError::new_err("class arrays must have H entries"));
        }
        let mut classes = self.classes.write().unwrap();
        classes.push(ClassRec { mcost, dsub, defense_pct });
        Ok(classes.len() - 1)
    }

    fn n_classes(&self) -> usize {
        self.classes.read().unwrap().len()
    }

    /// Add a unit from its field dict (wesnoth_ai.game_core.unit_fields).
    fn add_unit(&mut self, u: &Bound<'_, PyDict>) -> PyResult<usize> {
        let id: String = get(u, "id")?;
        if self.unit_index.contains_key(&id) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!("duplicate unit id {id}")));
        }
        let x: i64 = get(u, "x")?;
        let y: i64 = get(u, "y")?;
        let attacks_in: Vec<(i64, i64, i64, bool, Vec<String>)> = get(u, "attacks")?;
        let name: String = get(u, "name")?;
        let rec = UnitRec {
            type_idx: self.type_index_of(&name),
            id: id.clone(),
            name,
            name_id: get(u, "name_id")?,
            side: get(u, "side")?,
            is_leader: get(u, "is_leader")?,
            x,
            y,
            hex: self.map.pos_index.get(&(x, y)).map(|&i| i as i64).unwrap_or(-1),
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
            class_id: get_or(u, "class_id", -1)?,
            class_slowed_id: get_or(u, "class_slowed_id", -1)?,
        };
        self.units.push(rec);
        let idx = self.units.len() - 1;
        self.unit_index.insert(id, idx);
        Ok(idx)
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
        if let Some(v) = changes.get_item("class_id")? { u.class_id = v.extract()?; }
        if let Some(v) = changes.get_item("class_slowed_id")? { u.class_slowed_id = v.extract()?; }
        Ok(())
    }

    /// `team::spend_gold`: bare subtraction, no clamp (a recruit's cost).
    fn spend_gold(&mut self, side: i64, amount: i64) {
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

    fn set_rejected(&mut self, recruit: Vec<(i64, i64)>, moves: Vec<(i64, i64)>) {
        self.recruit_rejected = vec![0; self.map.h];
        self.move_rejected = vec![0; self.map.h];
        for (x, y) in recruit {
            if let Some(&i) = self.map.pos_index.get(&(x, y)) { self.recruit_rejected[i] = 1; }
        }
        for (x, y) in moves {
            if let Some(&i) = self.map.pos_index.get(&(x, y)) { self.move_rejected[i] = 1; }
        }
    }

    fn rejected_export(&self) -> (Vec<(i64, i64)>, Vec<(i64, i64)>) {
        let pick = |v: &Vec<u8>| (0..self.map.h).filter(|&i| v[i] != 0)
            .map(|i| (self.map.hx[i], self.map.hy[i])).collect::<Vec<_>>();
        (pick(&self.recruit_rejected), pick(&self.move_rejected))
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
    /// and rejected sets, the turn scalars. Equal states hash equal;
    /// a changed field changes it (tests/test_game_core.py).
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

fn unit_dict<'py>(py: Python<'py>, u: &UnitRec) -> PyResult<Bound<'py, PyDict>> {
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
    Ok(d)
}
