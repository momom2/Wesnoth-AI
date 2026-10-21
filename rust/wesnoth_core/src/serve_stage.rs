//! Phase 34: the inference server's host work per batch
//! (tools/serve_stage.py drives it). The server read each request into
//! Python objects (`byte_wire.decode_request`: a RawEncoded and a
//! PackedMasks per leaf), concatenated their arrays into the pinned
//! staging buffers with numpy, built the packed trunk's index arrays, the
//! masks and the memory states, and wrote one reply per request from the
//! priors' output buffer: 23.5 ms of host time per 40-leaf batch on the
//! laptop (tools/bench_serve_host.py). Here the requests are parsed in
//! place (`ServeRequests`, views into the received bytes) and every
//! buffer is written straight from them: the embedding stage in
//! `GameStateEncoder.stage_raws`'s layout, the index arrays of
//! `packed_trunk.build_packed_layout(source="streams")` and
//! `memory.stream_row_index`, the masks of `server_priors.write_masks`
//! with the compaction capacity of `_staged_capacity`, the memory states
//! of `InferenceServer._memory_batch`, and the replies in
//! `byte_wire.encode_replies`'s bytes. Each is the Python builder's
//! output byte for byte (tests/test_serve_stage.py).
//!
//! `MemoryTable` keeps the books of the memory states the server holds
//! between decisions (tools/serve_stage.py `ServerMemory` holds the
//! states on the device): which device rows hold each game-side's last
//! two states, keyed by the connection and the id its client sends, and
//! the version of each.

use std::collections::{HashMap, HashSet};

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadwriteArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

/// `byte_wire.FIELD_WORDS`: per field, its dtype code, ndim and three dims.
const FIELD_WORDS: usize = 5;
const MAX_NDIM: usize = 3;
const ALIGN: usize = 8;
/// Item sizes of `byte_wire.DTYPES` by code: bool, int8, uint8, int16,
/// int32, int64, float16, float32, float64.
const ITEMSIZE: [usize; 9] = [1, 1, 1, 2, 4, 8, 2, 4, 8];
const CODE_INT8: i64 = 1;
const CODE_INT32: i64 = 4;
const CODE_FLOAT32: i64 = 7;
const CODE_FLOAT64: i64 = 8;
/// `server_priors` slot kinds.
const SLOT_UNIT: u8 = 1;
const SLOT_RECRUIT: u8 = 2;
const SLOT_END: u8 = 3;
/// `model_output.UnitActionType`.
const TYPE_ATTACK: usize = 0;
const TYPE_MOVE: usize = 1;

fn err<T>(msg: impl Into<String>) -> PyResult<T> {
    Err(PyValueError::new_err(msg.into()))
}

fn word(data: &[u8], i: usize) -> i64 {
    i64::from_le_bytes(data[8 * i..8 * i + 8].try_into().unwrap())
}

#[derive(Clone, Copy, Default)]
struct FieldRef {
    off: usize,
    nbytes: usize,
    code: i64, // -1: absent
    ndim: usize,
    dims: [usize; MAX_NDIM],
}

impl FieldRef {
    fn present(&self) -> bool {
        self.code >= 0
    }
    fn rows(&self) -> usize {
        if !self.present() {
            0
        } else if self.ndim == 0 {
            1
        } else {
            self.dims[0]
        }
    }
}

struct Leaf {
    msg: usize,
    scalars: Vec<i64>,
    fields: Vec<FieldRef>,
}

/// The roles `ServeRequests` reads by name: the indices of the wire's
/// fields and scalars.
struct Roles {
    hex: usize,
    unit: usize,
    recruit: usize,
    sight: Option<usize>,
    actor_mask: usize,
    type_valid: usize,
    attack_valid: usize,
    move_valid: usize,
    union_valid: usize,
    n_attacks: usize,
    memory_state: usize,
    s_memory_k: usize,
    s_end_turn_bias: usize,
    s_n_units: usize,
    s_n_recruits: usize,
    s_n_hexes: usize,
}

fn index_of(names: &[String], name: &str) -> PyResult<usize> {
    match names.iter().position(|n| n == name) {
        Some(i) => Ok(i),
        None => err(format!("ServeRequests: no wire entry named {name:?}")),
    }
}

impl Roles {
    fn new(fields: &[String], scalars: &[String]) -> PyResult<Self> {
        Ok(Roles {
            hex: index_of(fields, "hex_xs")?,
            unit: index_of(fields, "unit_type_ids")?,
            recruit: index_of(fields, "recruit_type_ids")?,
            sight: fields.iter().position(|n| n == "sight_type_ids"),
            actor_mask: index_of(fields, "actor_mask")?,
            type_valid: index_of(fields, "type_valid")?,
            attack_valid: index_of(fields, "attack_valid")?,
            move_valid: index_of(fields, "move_valid")?,
            union_valid: index_of(fields, "union_valid")?,
            n_attacks: index_of(fields, "n_attacks")?,
            memory_state: index_of(fields, "memory_state")?,
            s_memory_k: index_of(scalars, "memory_k")?,
            s_end_turn_bias: index_of(scalars, "end_turn_bias")?,
            s_n_units: index_of(scalars, "n_units")?,
            s_n_recruits: index_of(scalars, "n_recruits")?,
            s_n_hexes: index_of(scalars, "n_hexes")?,
        })
    }
}

/// A batch's request messages, parsed in place: per leaf its scalars and,
/// per field, where its array lies in its message.
#[pyclass(module = "wesnoth_core")]
pub struct ServeRequests {
    messages: Vec<Py<PyBytes>>,
    leaves: Vec<Leaf>,
    rids: Vec<i64>,
    per_request: Vec<usize>,
    roles: Roles,
    /// Per leaf: its slot count for the trunk (the state's rows when the
    /// state travels, the player's k otherwise); None for a leaf without a memory.
    slot_counts: Vec<Option<usize>>,
}

/// The parse of one message from byte `start` (after the caller's head
/// words): its leaves, appended to `out`.
fn parse_bundle(data: &[u8], start: usize, n_scalars: usize, codes: &[i64], optional: &[bool],
                msg: usize, out: &mut Vec<Leaf>) -> Result<usize, String> {
    let n_fields = codes.len();
    if data.len() < start + 8 {
        return Err(format!("a message of {} bytes has no leaf count", data.len()));
    }
    let n = word(data, start / 8);
    if n < 0 {
        return Err(format!("a leaf count of {n}"));
    }
    let n = n as usize;
    let width = n_scalars + FIELD_WORDS * n_fields;
    let mut off = start + 8 * (1 + n * width);
    if data.len() < off {
        return Err(format!("a message of {} bytes for {n} leaf headers", data.len()));
    }
    for i in 0..n {
        let row = start / 8 + 1 + i * width;
        let scalars: Vec<i64> = (0..n_scalars).map(|j| word(data, row + j)).collect();
        let mut fields = Vec::with_capacity(n_fields);
        for f in 0..n_fields {
            let j = row + n_scalars + FIELD_WORDS * f;
            let code = word(data, j);
            if code < 0 {
                if !optional[f] {
                    return Err(format!("leaf {i}: field {f} is absent"));
                }
                fields.push(FieldRef { code: -1, ..Default::default() });
                continue;
            }
            if code != codes[f] {
                return Err(format!("leaf {i}: field {f} has dtype code {code}, expected {}", codes[f]));
            }
            let ndim = word(data, j + 1);
            if !(0..=MAX_NDIM as i64).contains(&ndim) {
                return Err(format!("leaf {i}: field {f} has {ndim} dimensions"));
            }
            let mut dims = [0usize; MAX_NDIM];
            let mut nbytes = ITEMSIZE[code as usize];
            for (k, d) in dims.iter_mut().enumerate().take(ndim as usize) {
                let v = word(data, j + 2 + k);
                if v < 0 {
                    return Err(format!("leaf {i}: field {f} has dimension {v}"));
                }
                *d = v as usize;
                nbytes *= *d;
            }
            if off + nbytes > data.len() {
                return Err(format!("leaf {i}: field {f} runs past the message's {} bytes", data.len()));
            }
            fields.push(FieldRef { off, nbytes, code, ndim: ndim as usize, dims });
            off += nbytes + (ALIGN - nbytes % ALIGN) % ALIGN;
        }
        out.push(Leaf { msg, scalars, fields });
    }
    if off != data.len() {
        return Err(format!("a message of {} bytes whose leaves end at byte {off}", data.len()));
    }
    Ok(n)
}

impl ServeRequests {
    fn data<'a>(&'a self, py: Python<'a>, leaf: &Leaf) -> &'a [u8] {
        self.messages[leaf.msg].bind(py).as_bytes()
    }

    fn field<'a>(&'a self, py: Python<'a>, leaf: &Leaf, f: usize) -> &'a [u8] {
        let r = &leaf.fields[f];
        &self.data(py, leaf)[r.off..r.off + r.nbytes]
    }

    fn size(&self, leaf: &Leaf) -> (usize, usize, usize) {
        (leaf.fields[self.roles.unit].rows(), leaf.fields[self.roles.recruit].rows(),
         leaf.fields[self.roles.hex].rows())
    }

    fn sightings(&self, leaf: &Leaf) -> usize {
        self.roles.sight.map_or(0, |s| leaf.fields[s].rows())
    }
}

fn put_i64s(buf: &mut [u8], at: &mut usize, values: impl IntoIterator<Item = i64>) {
    for v in values {
        buf[*at..*at + 8].copy_from_slice(&v.to_le_bytes());
        *at += 8;
    }
}

fn region<'a>(buf: &'a mut [u8], off: usize, len: usize, what: &str) -> PyResult<&'a mut [u8]> {
    if off + len > buf.len() {
        return err(format!("{what}: {len} bytes at {off} past a buffer of {}", buf.len()));
    }
    Ok(&mut buf[off..off + len])
}

#[pymethods]
impl ServeRequests {
    /// `messages`: the batch's request messages, each `head_words` int64
    /// words (kind, request id, leaves) then a `byte_wire.encode_bundle`
    /// bundle of the request's leaves. `fields` and `scalars`: the wire's
    /// field and scalar names in order; `codes`: each field's dtype code;
    /// `optional`: which fields a leaf may leave absent.
    #[new]
    fn new(py: Python<'_>, messages: Vec<Py<PyBytes>>, head_words: usize, fields: Vec<String>,
           scalars: Vec<String>, codes: Vec<i64>, optional: Vec<bool>) -> PyResult<Self> {
        if codes.len() != fields.len() || optional.len() != fields.len() {
            return err("ServeRequests: one dtype code and one optional flag per field");
        }
        if head_words < 3 {
            return err("ServeRequests: a request head is (kind, request id, leaves)");
        }
        let roles = Roles::new(&fields, &scalars)?;
        let mut leaves = Vec::new();
        let mut rids = Vec::with_capacity(messages.len());
        let mut per_request = Vec::with_capacity(messages.len());
        for (m, msg) in messages.iter().enumerate() {
            let data = msg.bind(py).as_bytes();
            if data.len() < 8 * head_words {
                return err(format!("request {m}: a message of {} bytes", data.len()));
            }
            let rid = word(data, 1);
            let declared = word(data, 2);
            let n = parse_bundle(data, 8 * head_words, scalars.len(), &codes, &optional, m, &mut leaves)
                .or_else(|e| err(format!("request {m} (id {rid}): {e}")))?;
            if declared != n as i64 {
                return err(format!("request {m} (id {rid}) declares {declared} leaves and carries {n}"));
            }
            rids.push(rid);
            per_request.push(n);
        }
        let mut out = ServeRequests { messages, leaves, rids, per_request, roles, slot_counts: Vec::new() };
        out.slot_counts = out.leaves.iter().map(|l| {
            let state = &l.fields[out.roles.memory_state];
            if state.present() {
                Some(state.rows())
            } else if l.scalars[out.roles.s_memory_k] >= 0 {
                Some(l.scalars[out.roles.s_memory_k] as usize)
            } else {
                None
            }
        }).collect();
        for (b, l) in out.leaves.iter().enumerate() {
            let (u, r, h) = out.size(l);
            let s = &l.scalars;
            if (s[out.roles.s_n_units], s[out.roles.s_n_recruits], s[out.roles.s_n_hexes])
                != (u as i64, r as i64, h as i64) {
                return err(format!("leaf {b}: masks for ({}, {}, {}) units, recruits and hexes, streams of \
                                    ({u}, {r}, {h})", s[out.roles.s_n_units], s[out.roles.s_n_recruits],
                                   s[out.roles.s_n_hexes]));
            }
            if l.fields[out.roles.actor_mask].rows() != u + r + 1 {
                return err(format!("leaf {b}: an actor mask of {} slots for {} actors",
                                   l.fields[out.roles.actor_mask].rows(), u + r + 1));
            }
        }
        Ok(out)
    }

    fn __len__(&self) -> usize {
        self.leaves.len()
    }

    /// Each request's id, in order.
    fn rids(&self) -> Vec<i64> {
        self.rids.clone()
    }

    /// Each request's leaves, in order.
    fn per_request(&self) -> Vec<usize> {
        self.per_request.clone()
    }

    /// Scalar `j` of every leaf, int64 [B].
    fn scalars<'py>(&self, py: Python<'py>, j: usize) -> PyResult<Bound<'py, PyArray1<i64>>> {
        if self.leaves.first().is_some_and(|l| j >= l.scalars.len()) {
            return err(format!("no scalar {j}"));
        }
        Ok(self.leaves.iter().map(|l| l.scalars[j]).collect::<Vec<_>>().into_pyarray(py))
    }

    /// Each leaf's (units, recruits, hexes).
    fn sizes(&self) -> Vec<(usize, usize, usize)> {
        self.leaves.iter().map(|l| self.size(l)).collect()
    }

    /// Each leaf's sighting count; None when no leaf carries the sighting
    /// stream. A batch where some leaves carry it and others not is refused.
    fn sighting_counts(&self) -> PyResult<Option<Vec<usize>>> {
        let Some(s) = self.roles.sight else { return Ok(None) };
        let present = self.leaves.iter().filter(|l| l.fields[s].present()).count();
        if present == 0 {
            return Ok(None);
        }
        if present != self.leaves.len() {
            return err(format!("{present} of {} leaves carry the sighting stream", self.leaves.len()));
        }
        Ok(Some(self.leaves.iter().map(|l| self.sightings(l)).collect()))
    }

    /// Each leaf's memory slots for the trunk (`InferenceServer.
    /// _memory_batch`'s counts: a state's rows, or the player's k when no
    /// state travels); None for a batch without memory. `slots`: the
    /// model's memory slots (0: none). A leaf without a memory for a model
    /// with one, or the reverse, is refused, as `_checked_memories` does.
    fn memory_counts(&self, slots: usize) -> PyResult<Option<Vec<usize>>> {
        let given = self.slot_counts.iter().filter(|c| c.is_some()).count();
        if slots == 0 {
            if given > 0 {
                return err("memory sent to a model without a memory");
            }
            return Ok(None);
        }
        if given != self.leaves.len() {
            return err(format!("a model with a memory ({slots} slots) needs every leaf's memory"));
        }
        let counts: Vec<usize> = self.slot_counts.iter().map(|c| c.unwrap()).collect();
        if let Some(k) = counts.iter().find(|&&k| k > slots) {
            return err(format!("a memory of {k} slots for a model of {slots}"));
        }
        Ok(Some(counts))
    }

    /// Whether field `f` of every leaf is present (true), of none (false);
    /// a mix is refused.
    fn all_present(&self, f: usize) -> PyResult<bool> {
        let present = self.leaves.iter().filter(|l| l.fields[f].present()).count();
        if present != 0 && present != self.leaves.len() {
            return err(format!("{present} of {} leaves carry field {f}", self.leaves.len()));
        }
        Ok(present != 0)
    }

    /// Whether any leaf carries field `f`.
    fn any_present(&self, f: usize) -> bool {
        self.leaves.iter().any(|l| l.fields[f].present())
    }

    /// Each entry (source, byte offset, bytes, exact) of `plan` into
    /// `buf`: a field (source >= 0) concatenated over the leaves, or a
    /// scalar (source -1 - j, scalar j) as one int64 per leaf; `exact`
    /// requires the entry's bytes to be filled, otherwise up to them.
    fn fill_concat(&self, py: Python<'_>, mut buf: PyReadwriteArray1<'_, u8>,
                   plan: Vec<(i64, usize, usize, bool)>) -> PyResult<()> {
        let buf = buf.as_slice_mut().or_else(|_| err("fill_concat: a non-contiguous buffer"))?;
        for (src, off, nbytes, exact) in plan {
            let dst = region(buf, off, nbytes, "fill_concat")?;
            let mut at = 0usize;
            if src >= 0 {
                let f = src as usize;
                for leaf in &self.leaves {
                    if f >= leaf.fields.len() {
                        return err(format!("fill_concat: no field {f}"));
                    }
                    let bytes = self.field(py, leaf, f);
                    if at + bytes.len() > nbytes {
                        return err(format!("fill_concat: field {f} holds more than its {nbytes} bytes"));
                    }
                    dst[at..at + bytes.len()].copy_from_slice(bytes);
                    at += bytes.len();
                }
            } else {
                let j = (-1 - src) as usize;
                if 8 * self.leaves.len() > nbytes {
                    return err(format!("fill_concat: scalar {j} for {} leaves in {nbytes} bytes", self.leaves.len()));
                }
                put_i64s(dst, &mut at, self.leaves.iter().map(|l| l.scalars[j]));
            }
            if exact && at != nbytes {
                return err(format!("fill_concat: source {src} filled {at} of its {nbytes} bytes \
                                    (a field width differs from the layout's)"));
            }
        }
        Ok(())
    }

    /// The masks of `server_priors.write_masks` into the mask layout's
    /// fields at byte offsets `offsets` (actor_bias, actor_mask,
    /// slot_kind, n_attacks, type_valid, attack_bits, move_bits,
    /// union_bits), whose rows are `a_stride` actor slots of `t` types,
    /// `hb_stride` packed hex bytes and `h_stride` hexes; `zero` (byte
    /// offset, bytes) is cleared first, and with `dirty` (rows and actor
    /// slots the previous batch wrote, `graphed_serve._clear_masks`)
    /// those of every field. Returns the compaction capacity
    /// (`_staged_capacity`) at `w` weapon slots.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (buf, offsets, a_stride, t, hb_stride, h_stride, w, zero, dirty=None))]
    fn fill_masks(&self, py: Python<'_>, mut buf: PyReadwriteArray1<'_, u8>, offsets: [i64; 8],
                  a_stride: usize, t: usize, hb_stride: usize, h_stride: usize, w: i64,
                  zero: (usize, usize), dirty: Option<(usize, usize)>) -> PyResult<i64> {
        let buf = buf.as_slice_mut().or_else(|_| err("fill_masks: a non-contiguous buffer"))?;
        region(buf, zero.0, zero.1, "fill_masks zero")?.fill(0);
        if let Some((rows, slots)) = dirty {
            // Per field the bytes of one actor slot.
            let widths = [4, 1, 1, 1, t, hb_stride, hb_stride, hb_stride];
            for (f, &off) in offsets.iter().enumerate() {
                if off < 0 {
                    continue;
                }
                let slot = widths[f];
                for b in 0..rows {
                    let row = off as usize + b * a_stride * slot;
                    region(buf, row, slots * slot, "fill_masks dirty")?.fill(0);
                }
            }
        }
        let r = &self.roles;
        let need = |i: usize| -> PyResult<usize> {
            if offsets[i] < 0 { err(format!("fill_masks: mask field {i} has no offset")) } else { Ok(offsets[i] as usize) }
        };
        let (o_bias, o_mask, o_slot, o_natt, o_tv) = (need(0)?, need(1)?, need(2)?, need(3)?, need(4)?);
        let (o_atk, o_mv, o_un) = (need(5)?, need(6)?, need(7)?);
        let mut capacity = 0i64;
        for (b, leaf) in self.leaves.iter().enumerate() {
            let (u, rc, h) = self.size(leaf);
            let a_n = u + rc + 1;
            if a_n > a_stride || h > h_stride {
                return err(format!("fill_masks: leaf {b} of {a_n} actors and {h} hexes for strides {a_stride}, {h_stride}"));
            }
            if b * a_stride + a_n > buf.len() {
                return err("fill_masks: the buffer is too small");
            }
            let tv_ref = &leaf.fields[r.type_valid];
            if tv_ref.ndim != 2 || tv_ref.dims[0] != a_n || tv_ref.dims[1] != t {
                return err(format!("fill_masks: leaf {b}'s type mask is {:?} for {a_n} actors of {t} types",
                                   &tv_ref.dims[..tv_ref.ndim]));
            }
            let row0 = b * a_stride;
            let bias = f64::from_bits(leaf.scalars[r.s_end_turn_bias] as u64) as f32;
            region(buf, o_bias + 4 * (row0 + u + rc), 4, "actor_bias")?.copy_from_slice(&bias.to_le_bytes());
            let mask = self.field(py, leaf, r.actor_mask);
            region(buf, o_mask + row0, a_n, "actor_mask")?.copy_from_slice(mask);
            let slots = region(buf, o_slot + row0, a_n, "slot_kind")?;
            slots[..u].fill(SLOT_UNIT);
            slots[u..u + rc].fill(SLOT_RECRUIT);
            slots[u + rc] = SLOT_END;
            let natt = self.field(py, leaf, r.n_attacks);
            if natt.len() != a_n {
                return err(format!("fill_masks: leaf {b}'s weapon counts hold {} slots for {a_n}", natt.len()));
            }
            region(buf, o_natt + row0, a_n, "n_attacks")?.copy_from_slice(natt);
            let tv = self.field(py, leaf, r.type_valid);
            region(buf, o_tv + row0 * t, a_n * t, "type_valid")?.copy_from_slice(tv);
            let mut rows = [&[][..]; 3];
            for (k, (f, o)) in [(r.attack_valid, o_atk), (r.move_valid, o_mv), (r.union_valid, o_un)].into_iter().enumerate() {
                let fr = &leaf.fields[f];
                let hb = if fr.ndim == 2 { fr.dims[1] } else { 0 };
                if fr.ndim != 2 || fr.dims[0] != a_n || hb > hb_stride {
                    return err(format!("fill_masks: leaf {b}'s bit mask {f} is {:?} for {a_n} actors, {hb_stride} bytes",
                                       &fr.dims[..fr.ndim]));
                }
                let bits = self.field(py, leaf, f);
                for a in 0..a_n {
                    region(buf, o + (row0 + a) * hb_stride, hb, "bits")?
                        .copy_from_slice(&bits[a * hb..(a + 1) * hb]);
                }
                rows[k] = bits;
            }
            // `_staged_capacity` over this leaf's rows.
            let hb = |k: usize| leaf.fields[[r.attack_valid, r.move_valid, r.union_valid][k]].dims[1];
            let popcount = |k: usize, a: usize| -> i64 {
                let n = hb(k);
                rows[k][a * n..(a + 1) * n].iter().map(|x| x.count_ones() as i64).sum()
            };
            for a in 0..a_n {
                if mask[a] == 0 {
                    continue;
                }
                if a < u {
                    let n_att = (natt[a] as i8 as i64).min(w);
                    if tv[a * t + TYPE_ATTACK] != 0 {
                        capacity += popcount(0, a) * n_att;
                    }
                    if tv[a * t + TYPE_MOVE] != 0 {
                        capacity += popcount(1, a);
                    }
                } else if a < u + rc {
                    capacity += popcount(2, a);
                } else {
                    capacity += 1;
                }
            }
        }
        Ok(capacity)
    }

    /// The widths the packed index takes: (tokens, A_max, H_max, U_max,
    /// R_max, S_max, K_max, memory rows, longest segment) for memory slot
    /// counts `memory` (None: no memory stream).
    #[pyo3(signature = (memory=None))]
    fn index_dims(&self, memory: Option<Vec<usize>>) -> PyResult<(usize, usize, usize, usize, usize, usize, usize, usize, usize)> {
        let ks = self.check_memory(&memory)?;
        let (mut total, mut hm, mut um, mut rm, mut sm, mut km, mut longest) = (0, 0, 0, 0, 0, 0, 0);
        for (b, leaf) in self.leaves.iter().enumerate() {
            let (u, r, h) = self.size(leaf);
            let s = self.sightings(leaf);
            let k = ks[b];
            let len = h + u + r + s + k + 2;
            total += len;
            longest = longest.max(len);
            hm = hm.max(h);
            um = um.max(u);
            rm = rm.max(r);
            sm = sm.max(s);
            km = km.max(k);
        }
        Ok((total, um + rm + 1, hm, um, rm, sm, km, ks.iter().sum(), longest))
    }

    /// `build_packed_layout(sizes, ..., source="streams", sightings,
    /// memory)` into `buf` from byte `off`, as int64 arrays in the order
    /// `PackedLayout.to_device` concatenates them (src, kind, actor, hex,
    /// unit, glob, memory, cu_seqlens), then the actor kinds [B, A_max]
    /// and `memory.stream_row_index(memory, K_max)`. `kinds`: TokenKind's
    /// HEX, UNIT, RECRUIT, SIGHTING, MEMORY, GLOBAL, END_TURN;
    /// `actor_kinds`: ActorKind's UNIT, RECRUIT, END_TURN. Returns the
    /// bytes written.
    #[pyo3(signature = (buf, off, kinds, actor_kinds, memory=None))]
    fn fill_packed_index(&self, mut buf: PyReadwriteArray1<'_, u8>, off: usize, kinds: [i64; 7],
                         actor_kinds: [i64; 3], memory: Option<Vec<usize>>) -> PyResult<usize> {
        let buf = buf.as_slice_mut().or_else(|_| err("fill_packed_index: a non-contiguous buffer"))?;
        let ks = self.check_memory(&memory)?;
        let n = self.leaves.len();
        let (total, a_max, h_max, u_max, _r_max, _s_max, k_max, mem_rows, _) = self.index_dims(memory)?;
        let need = 8 * (2 * total + n * (2 * a_max + h_max + u_max + 1 + k_max) + n + 1 + mem_rows);
        let dst = region(buf, off, need, "fill_packed_index")?;
        // Per leaf its block counts (hex, unit, recruit, sighting, memory, global, end_turn).
        let counts: Vec<[usize; 7]> = self.leaves.iter().enumerate().map(|(b, l)| {
            let (u, r, h) = self.size(l);
            [h, u, r, self.sightings(l), ks[b], 1, 1]
        }).collect();
        let mut block_totals = [0usize; 5];
        for c in &counts {
            for j in 0..5 {
                block_totals[j] += c[j];
            }
        }
        let mut block_base = [0usize; 5];
        for j in 1..5 {
            block_base[j] = block_base[j - 1] + block_totals[j - 1];
        }
        let globals = block_base[4] + block_totals[4];
        let mut cu = Vec::with_capacity(n + 1);
        cu.push(0i64);
        for c in &counts {
            cu.push(cu.last().unwrap() + c.iter().sum::<usize>() as i64);
        }
        let mut at = 0usize;
        // src: each segment's blocks from their stream rows.
        let mut seen = [0usize; 5];
        for (b, c) in counts.iter().enumerate() {
            for j in 0..5 {
                let start = (block_base[j] + seen[j]) as i64;
                put_i64s(dst, &mut at, (0..c[j] as i64).map(|i| start + i));
                seen[j] += c[j];
            }
            put_i64s(dst, &mut at, [(globals + b) as i64, (globals + n) as i64]);
        }
        for c in &counts {
            for j in 0..7 {
                put_i64s(dst, &mut at, std::iter::repeat(kinds[j]).take(c[j]));
            }
        }
        let ends: Vec<i64> = (0..n).map(|b| cu[b + 1] - 1).collect();
        for (b, c) in counts.iter().enumerate() {
            let (seg0, h, u, r) = (cu[b], c[0] as i64, c[1] as i64, c[2] as i64);
            put_i64s(dst, &mut at, (0..a_max as i64).map(|a| if a < u + r { seg0 + h + a } else { ends[b] }));
        }
        for (b, c) in counts.iter().enumerate() {
            let (seg0, h) = (cu[b], c[0] as i64);
            put_i64s(dst, &mut at, (0..h_max as i64).map(|x| if x < h { seg0 + x } else { ends[b] }));
        }
        for (b, c) in counts.iter().enumerate() {
            let (seg0, h, u) = (cu[b], c[0] as i64, c[1] as i64);
            put_i64s(dst, &mut at, (0..u_max as i64).map(|x| if x < u { seg0 + h + x } else { ends[b] }));
        }
        put_i64s(dst, &mut at, ends.iter().map(|e| e - 1));
        for (b, c) in counts.iter().enumerate() {
            let before = cu[b] + (c[0] + c[1] + c[2] + c[3]) as i64;
            let k = c[4] as i64;
            put_i64s(dst, &mut at, (0..k_max as i64).map(|x| if x < k { before + x } else { ends[b] }));
        }
        put_i64s(dst, &mut at, cu.iter().copied());
        for c in &counts {
            let (u, r) = (c[1] as i64, c[2] as i64);
            put_i64s(dst, &mut at, (0..a_max as i64).map(|a| {
                if a < u { actor_kinds[0] } else if a < u + r { actor_kinds[1] } else { actor_kinds[2] }
            }));
        }
        for (b, c) in counts.iter().enumerate() {
            put_i64s(dst, &mut at, (0..c[4] as i64).map(|x| (b * k_max) as i64 + x));
        }
        debug_assert_eq!(at, need);
        Ok(need)
    }

    /// The graphed path's index arrays (wesnoth_ai/graphed_serve.py
    /// `static_index` and `static_sources`) for a bucket of `b_cap`
    /// segments, `a_cap` actor and `h_cap` hex slots, `t_cap` token rows
    /// and `k_cap` memory slots, into `buf` at the byte offsets `offs` of
    /// its fields (cu_seqlens int32 [b_cap + 1]; src, kind [t_cap]; actor
    /// [b_cap * a_cap]; hex [b_cap * h_cap]; glob [b_cap]; memory [b_cap *
    /// k_cap], int64), the actor kinds [b_cap, a_cap] into `actor_kind`.
    /// `bases`: the token table's first hex, unit, recruit, sighting,
    /// memory and global row, then its end_turn row. The rows b < B are the
    /// real layout's (padded slots point at the row's end_turn token), then
    /// b_cap - B pad segments of one token each; token rows past the
    /// segments read the end_turn row. Returns the real segments' rows.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (buf, offs, b_cap, a_cap, h_cap, t_cap, k_cap, bases, kinds, actor_kinds, actor_kind,
                        memory=None))]
    fn fill_static_index(&self, mut buf: PyReadwriteArray1<'_, u8>, offs: [usize; 7], b_cap: usize, a_cap: usize,
                         h_cap: usize, t_cap: usize, k_cap: usize, bases: [i64; 7], kinds: [i64; 7],
                         actor_kinds: [i64; 3], mut actor_kind: PyReadwriteArray1<'_, i64>,
                         memory: Option<Vec<usize>>) -> PyResult<usize> {
        let buf = buf.as_slice_mut().or_else(|_| err("fill_static_index: a non-contiguous buffer"))?;
        let ak = actor_kind.as_slice_mut().or_else(|_| err("fill_static_index: non-contiguous actor kinds"))?;
        let n = self.leaves.len();
        if n > b_cap || ak.len() != b_cap * a_cap {
            return err(format!("fill_static_index: {n} segments for {b_cap}, {} actor kinds for {b_cap}x{a_cap}",
                               ak.len()));
        }
        let ks = self.check_memory(&memory)?;
        if let Some(k) = ks.iter().find(|&&k| k > k_cap) {
            return err(format!("static_index: a memory of {k} slots for k_cap {k_cap}"));
        }
        // Per segment its block counts (hex, unit, recruit, sighting, memory, global, end_turn);
        // a pad segment holds one token.
        let counts: Vec<[usize; 7]> = self.leaves.iter().enumerate().map(|(b, l)| {
            let (u, r, h) = self.size(l);
            [h, u, r, self.sightings(l), ks[b], 1, 1]
        }).collect();
        let mut cu = Vec::with_capacity(b_cap + 1);
        cu.push(0i64);
        for b in 0..b_cap {
            let len = if b < n { counts[b].iter().sum::<usize>() } else { 1 };
            cu.push(cu[b] + len as i64);
        }
        let rows = cu[b_cap];
        if rows as usize > t_cap {
            return err(format!("static_index: {rows} rows for t_cap {t_cap}"));
        }
        let total = cu[n] as usize;
        let zero = [0usize; 7];
        let at = |b: usize| -> &[usize; 7] { if b < n { &counts[b] } else { &zero } };
        let end = |b: usize| cu[b + 1] - 1;
        let mut actor = Vec::with_capacity(b_cap * a_cap);
        let mut hexes = Vec::with_capacity(b_cap * h_cap);
        let mut glob = Vec::with_capacity(b_cap);
        let mut mem = Vec::with_capacity(b_cap * k_cap);
        for b in 0..b_cap {
            let c = at(b);
            let (seg0, h, u, r, s, k) = (cu[b], c[0] as i64, c[1] as i64, c[2] as i64, c[3] as i64, c[4] as i64);
            actor.extend((0..a_cap as i64).map(|a| if a < u + r { seg0 + h + a } else { end(b) }));
            hexes.extend((0..h_cap as i64).map(|x| if x < h { seg0 + x } else { end(b) }));
            mem.extend((0..k_cap as i64).map(|x| if x < k { seg0 + h + u + r + s + x } else { end(b) }));
            glob.push(if b < n { end(b) - 1 } else { end(b) });
            for a in 0..a_cap as i64 {
                ak[b * a_cap + a as usize] =
                    if a < u { actor_kinds[0] } else if a < u + r { actor_kinds[1] } else { actor_kinds[2] };
            }
        }
        // Every slot addresses a row of its own segment, below the rows; a
        // wrong index would be a device-side assert inside a replay, which
        // poisons the whole CUDA context.
        for (name, arr) in [("actor", &actor), ("hex", &hexes), ("glob", &glob), ("memory", &mem)] {
            if arr.iter().any(|&x| x < 0 || x >= rows) {
                return err(format!("static_index: {name} slot outside the {rows} rows"));
            }
        }
        // The sources and kinds of the real rows, then the end_turn row.
        let mut src = Vec::with_capacity(t_cap);
        let mut kind = Vec::with_capacity(t_cap);
        let mut seen = [0i64; 4];
        for (b, c) in counts.iter().enumerate() {
            for j in 0..7 {
                let start = match j {
                    0..=3 => bases[j] + seen[j],
                    4 => bases[4] + (b * k_cap) as i64,
                    5 => bases[5] + b as i64,
                    _ => bases[6],
                };
                src.extend((0..c[j] as i64).map(|i| start + i));
                kind.extend(std::iter::repeat(kinds[j]).take(c[j]));
                if j < 4 {
                    seen[j] += c[j] as i64;
                }
            }
        }
        src.resize(t_cap, bases[6]);
        kind.resize(t_cap, kinds[6]);
        let [o_cu, o_src, o_kind, o_actor, o_hex, o_glob, o_mem] = offs;
        let cu32 = region(buf, o_cu, 4 * (b_cap + 1), "cu_seqlens")?;
        for (i, v) in cu.iter().enumerate() {
            cu32[4 * i..4 * i + 4].copy_from_slice(&(*v as i32).to_le_bytes());
        }
        for (o, arr, what) in [(o_src, &src, "src"), (o_kind, &kind, "kind"), (o_actor, &actor, "actor"),
                               (o_hex, &hexes, "hex"), (o_glob, &glob, "glob"), (o_mem, &mem, "memory")] {
            let dst = region(buf, o, 8 * arr.len(), what)?;
            let mut i = 0usize;
            put_i64s(dst, &mut i, arr.iter().copied());
        }
        Ok(total)
    }

    /// The memory states that travel with the leaves into `buf` from byte
    /// `off`: float32 [B, k_max, d], zeros past each leaf's state, as
    /// `InferenceServer._memory_batch` pads them. Returns the leaves whose
    /// state is absent (a game-side's first decision: the learned initial
    /// memory).
    fn fill_memory_states(&self, py: Python<'_>, mut buf: PyReadwriteArray1<'_, u8>, off: usize,
                          k_max: usize, d: usize, slots: usize) -> PyResult<Vec<usize>> {
        let buf = buf.as_slice_mut().or_else(|_| err("fill_memory_states: a non-contiguous buffer"))?;
        let n = self.leaves.len();
        let row = 4 * k_max * d;
        let dst = region(buf, off, n * row, "fill_memory_states")?;
        let mut initial = Vec::new();
        for (b, leaf) in self.leaves.iter().enumerate() {
            let f = &leaf.fields[self.roles.memory_state];
            let leaf_row = &mut dst[b * row..(b + 1) * row];
            if !f.present() {
                initial.push(b);
                leaf_row.fill(0);
                continue;
            }
            if f.code != CODE_FLOAT32 || f.ndim != 2 || f.dims[1] != d || f.dims[0] > slots || f.dims[0] > k_max {
                return err(format!("a memory state is [k <= {slots}, {d}] float32, got {:?}", &f.dims[..f.ndim]));
            }
            let bytes = self.field(py, leaf, self.roles.memory_state);
            leaf_row[..bytes.len()].copy_from_slice(bytes);
            leaf_row[bytes.len()..].fill(0);
        }
        Ok(initial)
    }

    /// One reply message per request, in `byte_wire.encode_replies`'s
    /// bytes: the request's id and status 0, then per leaf its units,
    /// recruits and served memory version (`versions`, -1 per leaf when
    /// None), the head outputs (`extras`: value, value_logits, cliffness,
    /// aux_score, moves_left, each (byte offset in `out`, item code, tail
    /// dims) or None), its actor kinds as int8 (`actor_kind`, rows of
    /// a_max slots, leaf b's in row b),
    /// its compact legal actions (`compact`: byte offsets in `out` of the
    /// int32 ends [B] and of actor, kind, target, weapon and prior
    /// [capacity]) and its memory state (`memory`: offset of float32 [B,
    /// k_max, d] in `out`, k_max, d, or None; each leaf's first rows by its
    /// slot count).
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (out, compact, capacity, extras, actor_kind, a_max, memory=None, versions=None, status=0))]
    fn replies<'py>(&self, py: Python<'py>, out: PyReadonlyArray1<'_, u8>, compact: [usize; 6], capacity: usize,
                    extras: Vec<Option<(usize, i64, Vec<usize>)>>, actor_kind: PyReadonlyArray1<'_, i64>,
                    a_max: usize, memory: Option<(usize, usize, usize)>, versions: Option<Vec<i64>>,
                    status: i64) -> PyResult<Vec<Bound<'py, PyBytes>>> {
        let out = out.as_slice().or_else(|_| err("replies: a non-contiguous output buffer"))?;
        let kinds = actor_kind.as_slice().or_else(|_| err("replies: non-contiguous actor kinds"))?;
        let n = self.leaves.len();
        if extras.len() != 5 {
            return err("replies: five head outputs (value, value_logits, cliffness, aux_score, moves_left)");
        }
        if kinds.len() < n * a_max {
            return err(format!("replies: {} actor kinds for {n} leaves of {a_max} slots", kinds.len()));
        }
        if let Some(v) = &versions {
            if v.len() != n {
                return err("replies: one memory version per leaf");
            }
        }
        let read_i32 = |off: usize, i: usize| -> i64 {
            i32::from_le_bytes(out[off + 4 * i..off + 4 * i + 4].try_into().unwrap()) as i64
        };
        let [o_ends, o_actor, o_kind, o_target, o_weapon, o_prior] = compact;
        if o_ends + 4 * n > out.len() || o_prior + 8 * capacity > out.len() {
            return err("replies: the compact actions lie past the output buffer");
        }
        let counts = self.memory_counts_unchecked();
        let present = extras.iter().filter(|e| e.is_some()).count();
        let mut replies = Vec::with_capacity(self.per_request.len());
        let mut b0 = 0usize;
        let mut lo = 0i64;
        for (r, &leaves) in self.per_request.iter().enumerate() {
            let mut words: Vec<i64> = vec![self.rids[r], status, leaves as i64];
            let mut chunks: Vec<Chunk> = Vec::with_capacity(leaves * (present + 7));
            for b in b0..b0 + leaves {
                let (u, rc, _h) = self.size(&self.leaves[b]);
                let a_n = u + rc + 1;
                let hi = read_i32(o_ends, b);
                if hi < lo || hi as usize > capacity {
                    return err(format!("batched_priors: {hi} legal entries exceed the mask-derived capacity \
                                        {capacity}; a legality mask admits fewer entries than its factors"));
                }
                let (lo_u, m) = (lo as usize, (hi - lo) as usize);
                words.extend([u as i64, rc as i64, versions.as_ref().map_or(-1, |v| v[b])]);
                for e in &extras {
                    match e {
                        None => header(&mut words, -1, &[]),
                        Some((off, code, tail)) => {
                            let mut dims = vec![1usize];
                            dims.extend(tail);
                            header(&mut words, *code, &dims);
                            let row = ITEMSIZE[*code as usize] * tail.iter().product::<usize>();
                            chunks.push(Chunk::Out(off + b * row, row));
                        }
                    }
                }
                header(&mut words, CODE_INT8, &[1, a_n]);
                chunks.push(Chunk::Kinds(b * a_max, a_n));
                for (off, code, size) in [(o_actor, CODE_INT32, 4), (o_kind, CODE_INT8, 1), (o_target, CODE_INT32, 4),
                                          (o_weapon, CODE_INT8, 1), (o_prior, CODE_FLOAT64, 8)] {
                    header(&mut words, code, &[m]);
                    chunks.push(Chunk::Out(off + size * lo_u, size * m));
                }
                match memory {
                    None => header(&mut words, -1, &[]),
                    Some((off, k_max, d)) => {
                        let k = counts[b].unwrap_or(0);
                        header(&mut words, CODE_FLOAT32, &[k, d]);
                        chunks.push(Chunk::Out(off + 4 * b * k_max * d, 4 * k * d));
                    }
                }
                lo = hi;
            }
            b0 += leaves;
            let padded = |len: usize| len + (ALIGN - len % ALIGN) % ALIGN;
            if chunks.iter().any(|c| matches!(*c, Chunk::Out(at, len) if at + len > out.len())) {
                return err("replies: an output array lies past the output buffer");
            }
            let total = 8 * words.len() + chunks.iter().map(|c| padded(c.len())).sum::<usize>();
            // Written in place into the bytes object: one allocation, one pass.
            let msg = PyBytes::new_with(py, total, |msg: &mut [u8]| {
                let mut at = 0usize;
                for w in &words {
                    msg[at..at + 8].copy_from_slice(&w.to_le_bytes());
                    at += 8;
                }
                for c in &chunks {
                    let len = c.len();
                    match *c {
                        Chunk::Out(src, _) => msg[at..at + len].copy_from_slice(&out[src..src + len]),
                        Chunk::Kinds(src, _) => {
                            for (dst, &k) in msg[at..at + len].iter_mut().zip(&kinds[src..src + len]) {
                                *dst = k as i8 as u8;
                            }
                        }
                    }
                    msg[at + len..at + padded(len)].fill(0);
                    at += padded(len);
                }
                Ok(())
            })?;
            replies.push(msg);
        }
        Ok(replies)
    }
}

/// An array of a reply: bytes of the output buffer (start, length), or
/// a leaf's actor kinds as int8 (start in the kinds, count).
enum Chunk {
    Out(usize, usize),
    Kinds(usize, usize),
}

impl Chunk {
    fn len(&self) -> usize {
        match *self {
            Chunk::Out(_, n) | Chunk::Kinds(_, n) => n,
        }
    }
}

/// A field's header words: its dtype code (-1: absent), ndim and dims.
fn header(words: &mut Vec<i64>, code: i64, dims: &[usize]) {
    words.push(code);
    words.push(if code < 0 { 0 } else { dims.len() as i64 });
    for k in 0..MAX_NDIM {
        words.push(*dims.get(k).unwrap_or(&0) as i64);
    }
}

impl ServeRequests {
    fn memory_counts_unchecked(&self) -> &[Option<usize>] {
        &self.slot_counts
    }

    /// The slot counts `memory` gives, one per leaf, zeros for None.
    fn check_memory(&self, memory: &Option<Vec<usize>>) -> PyResult<Vec<usize>> {
        match memory {
            None => Ok(vec![0; self.leaves.len()]),
            Some(m) if m.len() == self.leaves.len() => Ok(m.clone()),
            Some(m) => err(format!("{} memory counts for {} leaves", m.len(), self.leaves.len())),
        }
    }
}

// ---------------------------------------------------------------------
// The memory states the server keeps between decisions
// ---------------------------------------------------------------------

#[derive(Clone)]
struct Entry {
    rows: [i64; 2],
    versions: [i64; 2], // -1: the row holds no state
    k: i64,
}

/// The books of the device rows that hold each game-side's memory: per
/// (connection, key) two rows, the last two states written with their
/// versions. A decision reads version v (0: the learned initial memory,
/// row 0) and writes version v + 1 into the row it did not read, so the
/// state before the last decision survives: a recruit the engine refuses
/// is decided again from it (`RawPolicyPlayer.drop_last_pending`). A read
/// of a version the books do not hold is refused, never served from
/// another state.
#[pyclass(module = "wesnoth_core")]
pub struct MemoryTable {
    entries: HashMap<(i64, i64), Entry>,
    by_conn: HashMap<i64, HashSet<i64>>,
    free: Vec<i64>,
    rows: i64,
    max_keys: usize,
    pending: Vec<((i64, i64), usize, i64)>,
}

#[pymethods]
impl MemoryTable {
    /// `max_keys`: the most game-sides one connection may hold at once (a
    /// connection plays one game: two sides, more after re-decided first
    /// decisions); past it a new key is refused.
    #[new]
    fn new(max_keys: usize) -> Self {
        MemoryTable { entries: HashMap::new(), by_conn: HashMap::new(), free: Vec::new(), rows: 1,
                      max_keys, pending: Vec::new() }
    }

    /// The device rows the states need (row 0 is the initial memory).
    #[getter]
    fn rows(&self) -> i64 {
        self.rows
    }

    /// The game-sides held.
    fn __len__(&self) -> usize {
        self.entries.len()
    }

    /// Per leaf (its connection, key, version read, slot count): the row
    /// it reads, the row it writes, and why it is refused (None when it is
    /// not). A refused leaf reads and writes row -1. The writes are
    /// pending until `commit`.
    fn plan(&mut self, cids: Vec<i64>, keys: Vec<i64>, versions: Vec<i64>, ks: Vec<i64>)
            -> PyResult<(Vec<i64>, Vec<i64>, Vec<Option<String>>)> {
        let n = cids.len();
        if keys.len() != n || versions.len() != n || ks.len() != n {
            return err("MemoryTable.plan: one connection, key, version and slot count per leaf");
        }
        self.pending.clear();
        let mut reads = vec![-1i64; n];
        let mut writes = vec![-1i64; n];
        let mut why: Vec<Option<String>> = vec![None; n];
        let mut seen: HashSet<(i64, i64)> = HashSet::new();
        for b in 0..n {
            let id = (cids[b], keys[b]);
            if keys[b] < 0 {
                why[b] = Some(format!("no memory key (key {})", keys[b]));
                continue;
            }
            if !seen.insert(id) {
                why[b] = Some(format!("memory key {} twice in one batch", keys[b]));
                continue;
            }
            let v = versions[b];
            if v < 0 {
                why[b] = Some(format!("memory key {}: version {v}", keys[b]));
                continue;
            }
            let read_slot = match self.entries.get(&id) {
                Some(e) if e.k != ks[b] => {
                    why[b] = Some(format!("memory key {}: {} slots asked of a state of {}", keys[b], ks[b], e.k));
                    continue;
                }
                Some(e) => {
                    if v == 0 { None } else {
                        match (0..2).find(|&i| e.versions[i] == v) {
                            Some(i) => Some(i),
                            None => {
                                why[b] = Some(format!(
                                    "memory key {} version {v} is not held (the server holds versions {:?}): \
                                     refused, never served from another state", keys[b], e.versions));
                                continue;
                            }
                        }
                    }
                }
                None if v == 0 => None,
                None => {
                    why[b] = Some(format!(
                        "memory key {} version {v} is not held by this server (a restarted server, or a \
                         state released): refused, never served from another state", keys[b]));
                    continue;
                }
            };
            if !self.entries.contains_key(&id) {
                let held = self.by_conn.get(&cids[b]).map_or(0, |s| s.len());
                if held >= self.max_keys {
                    why[b] = Some(format!("connection {} already holds {held} memory states", cids[b]));
                    continue;
                }
                let rows = [self.take_row(), self.take_row()];
                self.entries.insert(id, Entry { rows, versions: [-1, -1], k: ks[b] });
                self.by_conn.entry(cids[b]).or_default().insert(keys[b]);
            }
            let e = &self.entries[&id];
            let write_slot = match read_slot { Some(i) => 1 - i, None => 0 };
            reads[b] = match read_slot { Some(i) => e.rows[i], None => 0 };
            writes[b] = e.rows[write_slot];
            self.pending.push((id, write_slot, v + 1));
        }
        Ok((reads, writes, why))
    }

    /// The pending writes of the last `plan` are on the device: their
    /// versions are held.
    fn commit(&mut self) {
        for (id, slot, version) in self.pending.drain(..) {
            if let Some(e) = self.entries.get_mut(&id) {
                e.versions[slot] = version;
            }
        }
    }

    /// The pending writes of the last `plan` did not complete (its batch
    /// failed or is planned again): the rows they were to write hold no
    /// state from now on, whatever the device wrote into them. What they
    /// held before is dead: a decision that reads version v never needs
    /// the state before v again.
    fn abort(&mut self) {
        for (id, slot, _) in self.pending.drain(..) {
            if let Some(e) = self.entries.get_mut(&id) {
                e.versions[slot] = -1;
            }
        }
    }

    /// Every state connection `cid` holds goes; returns how many.
    fn release_connection(&mut self, cid: i64) -> usize {
        let Some(keys) = self.by_conn.remove(&cid) else { return 0 };
        let n = keys.len();
        for key in keys {
            if let Some(e) = self.entries.remove(&(cid, key)) {
                self.free.extend(e.rows);
            }
        }
        n
    }
}

impl MemoryTable {
    fn take_row(&mut self) -> i64 {
        self.free.pop().unwrap_or_else(|| {
            self.rows += 1;
            self.rows - 1
        })
    }
}
