//! The unit-type and terrain databases, loaded once per process from the
//! dicts Python parses out of the committed `unit_stats.json` and
//! `terrain_db.json` (`wesnoth_ai.game_core.load_databases`). Every core
//! built afterwards reads the same tables; a type the scrape lacks gets
//! `_FALLBACK_STATS` under its own name, as `replay_dataset._stats_for`
//! gives it.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// `combat.DAMAGE_TYPES`: the resistance order.
pub const DAMAGE_TYPES: [&str; 6] = ["blade", "pierce", "impact", "fire", "cold", "arcane"];

/// A type's attack as the scrape lists it.
#[derive(Clone, Debug)]
pub struct TypeAttack {
    pub name: String,
    pub type_name: String,
    pub ranged: bool,
    pub damage: i64,
    pub number: i64,
    pub accuracy: i64,
    pub parry: i64,
    pub specials: Vec<String>,
}

/// A type's trait rules (`num_traits`, `musthave`, `pool`; the pool keeps
/// its duplicates, which weight the draw).
#[derive(Clone, Debug)]
pub struct TraitInfo {
    pub num_traits: i64,
    pub musthave: Vec<String>,
    pub pool: Vec<String>,
}

/// One unit type: what the builders, combat and the economy read.
#[derive(Clone, Debug)]
pub struct UnitType {
    pub name: String,
    pub known: bool,
    pub race: String,
    pub alignment: i64,
    pub level: i64,
    pub cost: i64,
    pub hitpoints: i64,
    pub experience: i64,
    pub moves: i64,
    pub undead_variation: String,
    pub advances_to: Vec<String>,
    /// Chance to be hit per terrain id, in the scrape's order.
    pub defense: Vec<(String, i64)>,
    /// Movement cost per terrain id; empty for a type with no movetype.
    pub movement_costs: Vec<(String, i64)>,
    pub resist: [i64; 6],
    pub attacks: Vec<TypeAttack>,
    pub abilities: Vec<String>,
    pub traits: Option<TraitInfo>,
    pub n_genders: i64,
}

/// A race's defaults (`races` in the scrape).
#[derive(Clone, Debug, Default)]
pub struct Race {
    pub undead_variation: String,
}

#[derive(Debug, Default)]
pub struct UnitDb {
    pub types: HashMap<String, Arc<UnitType>>,
    pub races: HashMap<String, Race>,
}

/// One terrain code of `terrain_db.json`.
#[derive(Clone, Debug)]
pub struct TerrainEntry {
    pub id: String,
    pub mvt_type: Vec<String>,
    pub def_type: Vec<String>,
    pub heals: i64,
    pub light: i64,
    pub max_light: i64,
    pub min_light: i64,
}

#[derive(Debug, Default)]
pub struct TerrainDb {
    pub entries: HashMap<String, TerrainEntry>,
}

static UNIT_DB: RwLock<Option<Arc<UnitDb>>> = RwLock::new(None);
static TERRAIN_DB: RwLock<Option<Arc<TerrainDb>>> = RwLock::new(None);

pub fn unit_db() -> PyResult<Arc<UnitDb>> {
    UNIT_DB.read().unwrap().clone().ok_or_else(|| {
        pyo3::exceptions::PyRuntimeError::new_err("wesnoth_core.load_databases was not called")
    })
}

pub fn terrain_db() -> PyResult<Arc<TerrainDb>> {
    TERRAIN_DB.read().unwrap().clone().ok_or_else(|| {
        pyo3::exceptions::PyRuntimeError::new_err("wesnoth_core.load_databases was not called")
    })
}

/// `classes.Alignment` of an alignment name; unknown names are neutral
/// (`replay_dataset._alignment_from_str`).
pub fn alignment_of(s: &str) -> i64 {
    match s.to_lowercase().as_str() {
        "lawful" => 0,
        "chaotic" => 2,
        "liminal" => 3,
        _ => 1,
    }
}

impl UnitType {
    /// `_FALLBACK_STATS` under `name`: 33 hp, 5 moves, 50 experience,
    /// cost 14, neutral level 1, one 5x2 blade attack, 50% defense on
    /// the keys the fallback lists, no resistances, no movetype.
    pub fn fallback(name: &str) -> UnitType {
        let mut defense: Vec<(String, i64)> = DAMAGE_TYPES.iter().map(|d| (d.to_string(), 50)).collect();
        for t in ["flat", "forest", "hills"] {
            defense.push((t.to_string(), 50));
        }
        UnitType {
            name: name.to_string(),
            known: false,
            race: String::new(),
            alignment: 1,
            level: 1,
            cost: 14,
            hitpoints: 33,
            experience: 50,
            moves: 5,
            undead_variation: String::new(),
            advances_to: Vec::new(),
            defense,
            movement_costs: Vec::new(),
            resist: [100; 6],
            attacks: vec![TypeAttack {
                name: "blade".into(), type_name: "blade".into(), ranged: false, damage: 5, number: 2,
                accuracy: 0, parry: 0, specials: Vec::new(),
            }],
            abilities: Vec::new(),
            traits: None,
            n_genders: 1,
        }
    }

    pub fn defense_of(&self, key: &str) -> Option<i64> {
        self.defense.iter().find(|(k, _)| k == key).map(|(_, v)| *v)
    }
}

impl UnitDb {
    /// The type of `name`, or the fallback under that name.
    pub fn get(&self, name: &str) -> Arc<UnitType> {
        match self.types.get(name) {
            Some(t) => t.clone(),
            None => Arc::new(UnitType::fallback(name)),
        }
    }

    pub fn contains(&self, name: &str) -> bool {
        self.types.contains_key(name)
    }

    /// The race's undead variation (`_RACE_DB[race]["undead_variation"]`).
    pub fn race_undead_variation(&self, race: &str) -> String {
        self.races.get(race).map(|r| r.undead_variation.clone()).unwrap_or_default()
    }
}

fn opt<'py, T: FromPyObject<'py>>(d: &Bound<'py, PyDict>, key: &str) -> PyResult<Option<T>> {
    match d.get_item(key)? {
        Some(v) if !v.is_none() => Ok(Some(v.extract()?)),
        _ => Ok(None),
    }
}

fn int_of(v: &Bound<'_, PyAny>) -> PyResult<i64> {
    // The scrape stores numbers as ints; `int(...)` of a float or a
    // numeric string is what the Python readers take.
    if let Ok(i) = v.extract::<i64>() {
        return Ok(i);
    }
    if let Ok(f) = v.extract::<f64>() {
        return Ok(f as i64);
    }
    let s: String = v.str()?.extract()?;
    s.trim().parse::<i64>().map_err(|_| pyo3::exceptions::PyValueError::new_err(format!("not an integer: {s:?}")))
}

fn int_or(d: &Bound<'_, PyDict>, key: &str, default: i64) -> PyResult<i64> {
    match d.get_item(key)? {
        Some(v) if !v.is_none() => int_of(&v),
        _ => Ok(default),
    }
}

fn str_or(d: &Bound<'_, PyDict>, key: &str) -> PyResult<String> {
    match d.get_item(key)? {
        Some(v) if !v.is_none() => Ok(v.str()?.extract()?),
        _ => Ok(String::new()),
    }
}

fn int_table(d: Option<Bound<'_, PyDict>>) -> PyResult<Vec<(String, i64)>> {
    let mut out = Vec::new();
    if let Some(d) = d {
        for (k, v) in d.iter() {
            out.push((k.extract::<String>()?, int_of(&v)?));
        }
    }
    Ok(out)
}

fn parse_type(name: &str, u: &Bound<'_, PyDict>, movetypes: &Bound<'_, PyDict>) -> PyResult<UnitType> {
    let res: Option<Bound<'_, PyDict>> = opt(u, "resistance")?;
    let mut resist = [100i64; 6];
    if let Some(r) = &res {
        for (k, dt) in DAMAGE_TYPES.iter().enumerate() {
            if let Some(v) = r.get_item(*dt)? {
                resist[k] = int_of(&v)?;
            }
        }
    }
    let mut attacks = Vec::new();
    if let Some(list) = opt::<Bound<'_, PyList>>(u, "attacks")? {
        for a in list.iter() {
            let a = a.downcast::<PyDict>()?;
            let specials: Vec<String> = opt(a, "specials")?.unwrap_or_default();
            attacks.push(TypeAttack {
                name: str_or(a, "name")?,
                type_name: opt::<String>(a, "type")?.unwrap_or_else(|| "blade".into()),
                ranged: opt::<String>(a, "range")?.as_deref() == Some("ranged"),
                damage: int_or(a, "damage", 1)?,
                number: int_or(a, "number", 1)?,
                accuracy: int_or(a, "accuracy", 0)?,
                parry: int_or(a, "parry", 0)?,
                specials,
            });
        }
    }
    // `_movetype_costs`: the type's merged table, else its movetype's.
    let mut movement_costs = int_table(opt(u, "movement_costs")?)?;
    if movement_costs.is_empty() {
        if let Some(mt) = opt::<String>(u, "movement_type")? {
            if let Some(m) = movetypes.get_item(&mt)? {
                let m = m.downcast::<PyDict>()?;
                movement_costs = int_table(opt(m, "movement_costs")?)?;
            }
        }
    }
    let traits = match opt::<Bound<'_, PyDict>>(u, "traits")? {
        Some(t) => Some(TraitInfo {
            num_traits: int_or(&t, "num_traits", 2)?,
            musthave: opt(&t, "musthave")?.unwrap_or_default(),
            pool: opt(&t, "pool")?.unwrap_or_default(),
        }),
        None => None,
    };
    Ok(UnitType {
        name: name.to_string(),
        known: true,
        race: str_or(u, "race")?,
        alignment: alignment_of(&opt::<String>(u, "alignment")?.unwrap_or_else(|| "neutral".into())),
        level: int_or(u, "level", 1)?,
        cost: int_or(u, "cost", 14)?,
        hitpoints: int_or(u, "hitpoints", 33)?,
        experience: int_or(u, "experience", 50)?,
        moves: int_or(u, "moves", 5)?,
        undead_variation: str_or(u, "undead_variation")?,
        advances_to: opt(u, "advances_to")?.unwrap_or_default(),
        defense: int_table(opt(u, "defense")?)?,
        movement_costs,
        resist,
        attacks,
        abilities: opt(u, "abilities")?.unwrap_or_default(),
        traits,
        n_genders: int_or(u, "n_genders", 1)?,
    })
}

fn parse_terrain(e: &Bound<'_, PyDict>) -> PyResult<TerrainEntry> {
    Ok(TerrainEntry {
        id: str_or(e, "id")?,
        mvt_type: opt(e, "mvt_type")?.unwrap_or_default(),
        def_type: opt(e, "def_type")?.unwrap_or_default(),
        heals: int_or(e, "heals", 0)?,
        light: int_or(e, "light", 0)?,
        max_light: int_or(e, "max_light", 0)?,
        min_light: int_or(e, "min_light", 0)?,
    })
}

/// Load the unit database (`unit_stats.json`: units, races,
/// movement_types) and the terrain database (`terrain_db.json`) for every
/// core this process builds.
#[pyfunction]
pub fn load_databases(unit_stats: &Bound<'_, PyDict>, terrain: &Bound<'_, PyDict>) -> PyResult<()> {
    let units: Bound<'_, PyDict> = opt(unit_stats, "units")?.unwrap_or_else(|| PyDict::new(unit_stats.py()));
    let movetypes: Bound<'_, PyDict> =
        opt(unit_stats, "movement_types")?.unwrap_or_else(|| PyDict::new(unit_stats.py()));
    let mut db = UnitDb::default();
    for (k, v) in units.iter() {
        let name: String = k.extract()?;
        let t = parse_type(&name, v.downcast::<PyDict>()?, &movetypes)?;
        db.types.insert(name, Arc::new(t));
    }
    if let Some(races) = opt::<Bound<'_, PyDict>>(unit_stats, "races")? {
        for (k, v) in races.iter() {
            let r = v.downcast::<PyDict>()?;
            db.races.insert(k.extract()?, Race { undead_variation: str_or(r, "undead_variation")? });
        }
    }
    let mut tdb = TerrainDb::default();
    for (k, v) in terrain.iter() {
        tdb.entries.insert(k.extract()?, parse_terrain(v.downcast::<PyDict>()?)?);
    }
    *UNIT_DB.write().unwrap() = Some(Arc::new(db));
    *TERRAIN_DB.write().unwrap() = Some(Arc::new(tdb));
    Ok(())
}

#[pyfunction]
pub fn databases_loaded() -> bool {
    UNIT_DB.read().unwrap().is_some() && TERRAIN_DB.read().unwrap().is_some()
}
