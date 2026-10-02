//! A parsed WML tag as the core keeps it: the tag, its attributes in
//! document order and its children. Python parses and preprocesses a
//! scenario's WML (`tools.replay_extract.WMLNode`) and hands the nodes
//! over as nested `(tag, [(key, value)], [children])` tuples
//! (`wesnoth_ai.game_core.wml_tuple`); the core reads them when it
//! applies an `[effect]` or runs an event.

use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use std::sync::Arc;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Wml {
    pub tag: String,
    pub attrs: Vec<(String, String)>,
    pub children: Vec<Arc<Wml>>,
}

impl Wml {
    /// The raw value of `key` (`node.attrs.get(key)`).
    pub fn attr(&self, key: &str) -> Option<&str> {
        self.attrs.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
    }

    /// `(node.attrs.get(key, "") or "").strip().strip('"')`.
    pub fn clean(&self, key: &str) -> String {
        clean(self.attr(key).unwrap_or(""))
    }

    pub fn has(&self, key: &str) -> bool {
        self.attr(key).is_some()
    }

    pub fn first(&self, tag: &str) -> Option<&Arc<Wml>> {
        self.children.iter().find(|c| c.tag == tag)
    }

    pub fn all<'a>(&'a self, tag: &'a str) -> impl Iterator<Item = &'a Arc<Wml>> + 'a {
        self.children.iter().filter(move |c| c.tag == tag)
    }
}

/// `s.strip().strip('"')`.
pub fn clean(s: &str) -> String {
    s.trim().trim_matches('"').to_string()
}

impl<'py> FromPyObject<'py> for Wml {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let t = ob.downcast::<PyTuple>()?;
        let tag: String = t.get_item(0)?.extract()?;
        let attrs: Vec<(String, String)> = t.get_item(1)?.extract()?;
        let kids = t.get_item(2)?;
        let kids = kids.downcast::<PyList>()?;
        let mut children = Vec::with_capacity(kids.len());
        for k in kids.iter() {
            children.push(Arc::new(k.extract::<Wml>()?));
        }
        Ok(Wml { tag, attrs, children })
    }
}

impl Wml {
    /// The `(tag, [(key, value)], [children])` tuple form, for Python.
    pub fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let attrs = PyList::empty(py);
        for (k, v) in &self.attrs {
            attrs.append((k, v))?;
        }
        let kids = PyList::empty(py);
        for c in &self.children {
            kids.append(c.to_py(py)?)?;
        }
        let tag = pyo3::types::PyString::new(py, &self.tag);
        PyTuple::new(py, [tag.into_any(), attrs.into_any(), kids.into_any()])
    }
}
