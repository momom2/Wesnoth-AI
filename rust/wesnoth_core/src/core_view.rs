//! Phase 32: the Python view's units, built by the core. `CoreState.to_state`
//! materializes every unit of the state as a `classes.Unit` after each
//! command the simulator applies; it built them in Python from
//! `units_export`'s dicts (`game_core.unit_from_fields`), about half of a
//! view's cost. Here the core makes the same objects directly: each a
//! dataclass instance whose `__dict__` holds the fields in the order the
//! dataclass's `__init__` sets them, then the underscore attributes in the
//! order `unit_from_fields` sets them, with the same Python types (the
//! enum members, sets for abilities, traits, statuses and weapon
//! specials, lists for the rest, `Position` and `Attack` instances, the
//! `[object]` effects as `wml_node` makes them). Equality with
//! `unit_from_fields` on played positions is the certificate
//! (tests/test_core_view.py); `unit_from_fields` stays for the single
//! units the builders return.

use pyo3::exceptions::PyValueError;
use pyo3::intern;
use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};

use pyo3::types::{PyDict, PyFrozenSet, PyList, PySet, PyTuple};

use crate::core::{GameCore, UnitRec};

/// The Python types a view's unit is made of: `object.__new__`, the
/// `Unit`, `Position` and `Attack` classes, the `Alignment` and
/// `DamageType` members by value, and `game_core.wml_node`.
struct ViewTypes<'py> {
    new: Bound<'py, PyAny>,
    unit: Bound<'py, PyAny>,
    position: Bound<'py, PyAny>,
    attack: Bound<'py, PyAny>,
    alignments: Bound<'py, PyDict>,
    damage_types: Bound<'py, PyDict>,
    wml_node: Bound<'py, PyAny>,
}

impl<'py> ViewTypes<'py> {
    fn from_tuple(t: &Bound<'py, PyTuple>) -> PyResult<Self> {
        if t.len() != 7 {
            return Err(PyValueError::new_err(format!(
                "units_view takes (object.__new__, Unit, Position, Attack, alignments, damage types, wml_node), got {} items",
                t.len())));
        }
        Ok(ViewTypes {
            new: t.get_item(0)?, unit: t.get_item(1)?, position: t.get_item(2)?, attack: t.get_item(3)?,
            alignments: t.get_item(4)?.downcast_into::<PyDict>()?,
            damage_types: t.get_item(5)?.downcast_into::<PyDict>()?, wml_node: t.get_item(6)?,
        })
    }

    /// An instance of `cls` holding `d` as its `__dict__`.
    fn instance(&self, cls: &Bound<'py, PyAny>, d: Bound<'py, PyDict>) -> PyResult<Bound<'py, PyAny>> {
        let obj = self.new.call1((cls,))?;
        obj.setattr(intern!(obj.py(), "__dict__"), d)?;
        Ok(obj)
    }
}

/// The enum member of value `v`, or the ValueError the enum's call raises.
fn member<'py>(members: &Bound<'py, PyDict>, v: i64, what: &str) -> PyResult<Bound<'py, PyAny>> {
    members.get_item(v)?.ok_or_else(|| PyValueError::new_err(format!("{v} is not a valid {what}")))
}

fn unit_object<'py>(py: Python<'py>, t: &ViewTypes<'py>, u: &UnitRec) -> PyResult<Bound<'py, PyAny>> {
    let pos = PyDict::new(py);
    pos.set_item(intern!(py, "x"), u.x)?;
    pos.set_item(intern!(py, "y"), u.y)?;
    let attacks = PyList::empty(py);
    for a in &u.attacks {
        let ad = PyDict::new(py);
        ad.set_item(intern!(py, "type_id"), member(&t.damage_types, a.type_id, "DamageType")?)?;
        ad.set_item(intern!(py, "number_strikes"), a.strikes)?;
        ad.set_item(intern!(py, "damage_per_strike"), a.damage)?;
        ad.set_item(intern!(py, "is_ranged"), a.ranged)?;
        ad.set_item(intern!(py, "weapon_specials"), PySet::new(py, &a.specials)?)?;
        attacks.append(t.instance(&t.attack, ad)?)?;
    }
    let d = PyDict::new(py);
    d.set_item(intern!(py, "id"), &u.id)?;
    d.set_item(intern!(py, "name"), &u.name)?;
    d.set_item(intern!(py, "name_id"), u.name_id)?;
    d.set_item(intern!(py, "side"), u.side)?;
    d.set_item(intern!(py, "is_leader"), u.is_leader)?;
    d.set_item(intern!(py, "position"), t.instance(&t.position, pos)?)?;
    d.set_item(intern!(py, "max_hp"), u.max_hp)?;
    d.set_item(intern!(py, "max_moves"), u.max_moves)?;
    d.set_item(intern!(py, "max_exp"), u.max_exp)?;
    d.set_item(intern!(py, "cost"), u.cost)?;
    d.set_item(intern!(py, "alignment"), member(&t.alignments, u.alignment, "Alignment")?)?;
    d.set_item(intern!(py, "levelup_names"), PyList::new(py, &u.levelup_names)?)?;
    d.set_item(intern!(py, "current_hp"), u.current_hp)?;
    d.set_item(intern!(py, "current_moves"), u.current_moves)?;
    d.set_item(intern!(py, "current_exp"), u.current_exp)?;
    d.set_item(intern!(py, "has_attacked"), u.has_attacked)?;
    d.set_item(intern!(py, "attacks"), attacks)?;
    d.set_item(intern!(py, "resistances"), PyList::new(py, &u.resistances)?)?;
    d.set_item(intern!(py, "defenses"), PyList::new(py, &u.defenses)?)?;
    d.set_item(intern!(py, "movement_costs"), PyList::new(py, &u.movement_costs)?)?;
    d.set_item(intern!(py, "abilities"), PySet::new(py, &u.abilities)?)?;
    d.set_item(intern!(py, "traits"), PySet::new(py, &u.traits)?)?;
    d.set_item(intern!(py, "statuses"), PySet::new(py, &u.statuses)?)?;
    if let Some(table) = &u.def_table {
        let td = PyDict::new(py);
        for (k, v) in table.iter() {
            td.set_item(k, *v)?;
        }
        d.set_item(intern!(py, "_defense_table"), td)?;
    }
    if let Some(p) = &u.pickadvance {
        d.set_item(intern!(py, "_pickadvance"), PyList::new(py, p)?)?;
    }
    if let Some(o) = &u.trait_order {
        d.set_item(intern!(py, "_trait_order"), PyList::new(py, o)?)?;
    }
    if let Some(f) = u.feeding_count {
        d.set_item(intern!(py, "_feeding_count"), f)?;
    }
    if let Some(r) = &u.wml_role {
        d.set_item(intern!(py, "_wml_role"), r)?;
    }
    if !u.object_effects.is_empty() {
        let effects = PyList::empty(py);
        for e in &u.object_effects {
            effects.append(t.wml_node.call1((e.to_py(py)?,))?)?;
        }
        d.set_item(intern!(py, "_object_effects"), effects)?;
    }
    if u.ai_guardian {
        d.set_item(intern!(py, "_ai_guardian"), true)?;
    }
    t.instance(&t.unit, d)
}

#[pymethods]
impl GameCore {
    /// Every unit as the view's `classes.Unit`, in the core's unit order:
    /// what `[unit_from_fields(d) for d in units_export()]` builds. `types`
    /// is (object.__new__, Unit, Position, Attack, {value: Alignment},
    /// {value: DamageType}, wml_node).
    fn units_view<'py>(&self, py: Python<'py>, types: &Bound<'py, PyTuple>) -> PyResult<Bound<'py, PyList>> {
        let t = ViewTypes::from_tuple(types)?;
        let out = PyList::empty(py);
        for u in &self.units {
            out.append(unit_object(py, &t, u)?)?;
        }
        Ok(out)
    }
}

/// Phase 34: what the simulator's live view (`WesnothSim.gs`) rebuilds
/// after each command, kept from one refresh to the next: each unit's
/// Python object with the record it was built from, and the side's
/// cleared fog as the view holds it. A refresh rebuilds the units whose
/// record changed and the fog when it changed; the rest are the same
/// objects (a view's units are read-only, `game_core.bind_view`). Correct
/// whatever changed the core, since every record is compared whole.
#[pyclass(module = "wesnoth_core")]
pub struct ViewCache {
    units: HashMap<String, (UnitRec, Py<PyAny>)>,
    fog: Option<(Vec<Vec<u8>>, Py<PyDict>)>,
    rebuilt: u64,
    reused: u64,
}

#[pymethods]
impl ViewCache {
    #[new]
    fn new() -> Self {
        ViewCache { units: HashMap::new(), fog: None, rebuilt: 0, reused: 0 }
    }

    /// `core.units_view(types)`, each unit's object reused while its
    /// record is unchanged.
    fn units<'py>(&mut self, py: Python<'py>, core: PyRef<'_, GameCore>, types: &Bound<'py, PyTuple>)
                  -> PyResult<Bound<'py, PyList>> {
        let t = ViewTypes::from_tuple(types)?;
        let out = PyList::empty(py);
        let mut present: HashSet<&str> = HashSet::with_capacity(core.units.len());
        for u in &core.units {
            let hit = match self.units.get(&u.id) {
                Some((rec, obj)) if rec == u => Some(obj.clone_ref(py)),
                _ => None,
            };
            let obj = match hit {
                Some(obj) => {
                    self.reused += 1;
                    obj
                }
                None => {
                    let obj = unit_object(py, &t, u)?.unbind();
                    self.units.insert(u.id.clone(), (u.clone(), obj.clone_ref(py)));
                    self.rebuilt += 1;
                    obj
                }
            };
            out.append(obj)?;
            present.insert(u.id.as_str());
        }
        self.units.retain(|id, _| present.contains(id.as_str()));
        Ok(out)
    }

    /// The view's `_fog_cleared`: {side: frozenset of (x, y)} for every
    /// tracked side, as `CoreState.to_state` builds it from
    /// `fog_cleared_export`; a new dict of the same frozensets while the
    /// record is unchanged.
    fn fog_cleared<'py>(&mut self, py: Python<'py>, core: PyRef<'_, GameCore>) -> PyResult<Bound<'py, PyDict>> {
        if let Some((rec, d)) = &self.fog {
            if *rec == core.fog_cleared {
                return d.bind(py).copy();
            }
        }
        let d = PyDict::new(py);
        for (k, v) in core.fog_cleared.iter().enumerate() {
            if v.is_empty() {
                continue;
            }
            let hexes: Vec<Bound<'py, PyTuple>> = (0..core.map.h).filter(|&j| v[j] != 0)
                .map(|j| PyTuple::new(py, [core.map.hx[j], core.map.hy[j]]))
                .collect::<PyResult<_>>()?;
            d.set_item(k as i64 + 1, PyFrozenSet::new(py, &hexes)?)?;
        }
        self.fog = Some((core.fog_cleared.clone(), d.clone().unbind()));
        d.copy()
    }

    /// (units rebuilt, units reused) over the cache's life.
    #[getter]
    fn counts(&self) -> (u64, u64) {
        (self.rebuilt, self.reused)
    }
}
