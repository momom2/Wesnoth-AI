"""Columnar sequence records: a game-side's positions as a few flat arrays
(docs/trainer_speed_20261013.md, round 2).

The pickled records (`GameSequence`, tools/preencode_sequences.py and
tools/own_targets.py) hold every position as a dozen Python objects and
every hex as a `Position`: unpickling a game builds tens of thousands of
objects on threads that hold the interpreter lock, and the collector walks
them all. A columnar record holds what the sequence trainer's step reads,
numbers only: per game-side, each field of every position concatenated into
one array, with offsets where its length varies by position. Reading a
game-side is one zlib inflate (which releases the interpreter lock) and one
`np.frombuffer` per array; a position is a slice of each.

    <game>.seq.cols   magic, header length (u64), JSON header, one zlib
                      blob per side

The header names each side's blob (offset, compressed and raw sizes) and its
arrays (name, dtype, shape, offset in the inflated blob). Every array keeps
the dtype the pickled record holds, so a position read here equals the
pickled one field for field (`ColumnarSide.position`, which
tools/columnize_sequences.py checks for every position it writes). What the
step does not read is left out: the hexes' and units' `Position` objects,
the units' ids, the recruits' type names, the label's source and target hex
and recruited type.
"""
from __future__ import annotations

import json
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

COLUMNS_SUFFIX = ".seq.cols"
COLUMNS_FORMAT = 1
_MAGIC = b"WSEQCOLS"
_ALIGN = 64                       # bytes; every array of an inflated blob starts on this boundary
ZLIB_LEVEL = 1
# A label's action type, by its code in `label_kind`.
LABEL_KINDS = ("attack", "move", "recruit", "end_turn", "timeout")
# An index the label does not have (target, weapon, type).
NO_INDEX = -1


class ColumnsError(ValueError):
    """A record that cannot be written as columns, or a columnar file that
    is not one."""


def columns_path(directory: Path, file: str) -> Path:
    return Path(directory) / (file + COLUMNS_SUFFIX)


# ---- what a position holds, column by column -------------------------------

@dataclass(frozen=True)
class Group:
    """Columns of one length per position: `count(position)` entries each
    (None: one entry per position, the group of rows). `columns`: (name,
    dtype, getter); a getter returns the position's entries (an array whose
    first axis has `count` entries, or one value for the rows)."""
    name: str
    count: Optional[Callable]
    columns: Tuple[Tuple[str, str, Callable], ...]


def _optional_index(v) -> int:
    if v is None:
        return NO_INDEX
    if int(v) < 0:
        raise ColumnsError(f"a label index {v} below 0 cannot be told from a missing one")
    return int(v)


def _label_kind(ai) -> int:
    try:
        return LABEL_KINDS.index(ai.action_type)
    except ValueError:
        raise ColumnsError(f"a label of action type {ai.action_type!r}, not one of {LABEL_KINDS}") from None


def _belief_mask(p) -> np.ndarray:
    if p.no_visible_unit.shape[0] != p.raw.hex_xs.shape[0]:
        raise ColumnsError(f"belief targets for {p.no_visible_unit.shape[0]} hex tokens, the encoding has "
                           f"{p.raw.hex_xs.shape[0]}")
    return p.no_visible_unit


CORPUS_GROUPS = (
    Group("rows", None, (
        ("global_feats", "<f4", lambda p: p.raw.global_feats),
        ("our_faction_id", "<i8", lambda p: p.raw.our_faction_id),
        ("their_faction_id", "<i8", lambda p: p.raw.their_faction_id),
        ("their_faction_probs", "<f4", lambda p: p.raw.their_faction_probs),
        ("material", "<f8", lambda p: p.raw.material),
        ("hex_subset", "|b1", lambda p: p.raw.hex_subset),
        ("turn", "<i8", lambda p: p.turn),
        ("label_kind", "|u1", lambda p: _label_kind(p.label)),
        ("label_actor", "<i8", lambda p: p.label.actor_idx),
        ("label_target", "<i8", lambda p: _optional_index(p.label.target_idx)),
        ("label_weapon", "<i8", lambda p: _optional_index(p.label.weapon_idx)),
        ("label_type", "<i8", lambda p: _optional_index(p.label.type_idx)),
        ("label_off_subset", "|b1", lambda p: p.label.target_off_subset),
    )),
    Group("hex", lambda p: p.raw.hex_xs.shape[0], (
        ("hex_xs", "<i8", lambda p: p.raw.hex_xs),
        ("hex_ys", "<i8", lambda p: p.raw.hex_ys),
        ("hex_terrain_ids", "<i8", lambda p: p.raw.hex_terrain_ids),
        ("hex_modifier_flags", "<f4", lambda p: p.raw.hex_modifier_flags),
        ("hex_dynamic_flags", "<f4", lambda p: p.raw.hex_dynamic_flags),
        ("no_visible_unit", "|b1", _belief_mask),
    )),
    Group("unit", lambda p: p.raw.unit_xs.shape[0], (
        ("unit_is_ours", "<f4", lambda p: p.raw.unit_is_ours),
        ("unit_type_ids", "<i8", lambda p: p.raw.unit_type_ids),
        ("unit_side_ids", "<i8", lambda p: p.raw.unit_side_ids),
        ("unit_xs", "<i8", lambda p: p.raw.unit_xs),
        ("unit_ys", "<i8", lambda p: p.raw.unit_ys),
        ("unit_feats", "<f4", lambda p: p.raw.unit_feats),
    )),
    Group("recruit", lambda p: p.raw.recruit_type_ids.shape[0], (
        ("recruit_is_ours", "<f4", lambda p: p.raw.recruit_is_ours),
        ("recruit_type_ids", "<i8", lambda p: p.raw.recruit_type_ids),
        ("recruit_side_ids", "<i8", lambda p: p.raw.recruit_side_ids),
        ("recruit_xs", "<i8", lambda p: p.raw.recruit_xs),
        ("recruit_ys", "<i8", lambda p: p.raw.recruit_ys),
        ("recruit_feats", "<f4", lambda p: p.raw.recruit_feats),
    )),
    Group("sighting", lambda p: p.raw.sight_type_ids.shape[0], (
        ("sight_type_ids", "<i8", lambda p: p.raw.sight_type_ids),
        ("sight_xs", "<i8", lambda p: p.raw.sight_xs),
        ("sight_ys", "<i8", lambda p: p.raw.sight_ys),
        ("sight_feats", "<f4", lambda p: p.raw.sight_feats),
    )),
    Group("hidden", lambda p: p.hidden_tokens.shape[0], (
        ("hidden_tokens", "<i8", lambda p: p.hidden_tokens),
    )),
)


def _type_bias(p) -> np.ndarray:
    m = p.masks
    return np.zeros(m.type_valid.shape, np.float32) if m.type_bias is None else m.type_bias


OWN_GROUPS = (
    Group("own_rows", None, (
        ("mask_end_turn_bias", "<f8", lambda p: p.masks.end_turn_bias),
        ("mask_units", "<i8", lambda p: p.masks.n_units),
        ("mask_recruits", "<i8", lambda p: p.masks.n_recruits),
        ("mask_hexes", "<i8", lambda p: p.masks.n_hexes),
        ("mask_bits_width", "<i8", lambda p: p.masks.attack_valid.shape[1]),
        ("mask_type_bias_on", "|b1", lambda p: p.masks.type_bias is not None),
        ("mask_attack_bias_on", "|b1", lambda p: p.masks.attack_bias is not None),
        ("target_end_prior", "<f8", lambda p: p.own_target.end_prior),
    )),
    Group("actor", lambda p: p.masks.actor_mask.shape[0], (
        ("actor_mask", "|u1", lambda p: p.masks.actor_mask),
        ("n_attacks", "|i1", lambda p: p.masks.n_attacks),
        ("type_valid", "|u1", lambda p: p.masks.type_valid),
        ("type_bias", "<f4", _type_bias),
    )),
    Group("bits", lambda p: p.masks.attack_valid.size, (
        ("attack_valid", "|u1", lambda p: p.masks.attack_valid.reshape(-1)),
        ("move_valid", "|u1", lambda p: p.masks.move_valid.reshape(-1)),
        ("union_valid", "|u1", lambda p: p.masks.union_valid.reshape(-1)),
    )),
    Group("attack_bias", lambda p: 0 if p.masks.attack_bias is None else p.masks.attack_bias.size, (
        ("attack_bias", "<f4", lambda p: (np.zeros(0, np.float32) if p.masks.attack_bias is None
                                          else p.masks.attack_bias.reshape(-1))),
    )),
    Group("target", lambda p: p.own_target.prior.shape[0], (
        ("target_actor", "<i4", lambda p: p.own_target.actor),
        ("target_kind", "|i1", lambda p: p.own_target.kind),
        ("target_target", "<i4", lambda p: p.own_target.target),
        ("target_weapon", "|i1", lambda p: p.own_target.weapon),
        ("target_prior", "<f8", lambda p: p.own_target.prior),
    )),
)


def _strict(value, dtype: str, where: str) -> np.ndarray:
    """`value` as an array of `dtype` without any change of value: an
    array of another dtype is refused (a cast could round), a Python
    scalar is converted and checked."""
    want = np.dtype(dtype)
    if isinstance(value, np.ndarray):
        if value.dtype != want:
            raise ColumnsError(f"{where}: dtype {value.dtype}, the columns hold {want}")
        return value
    arr = np.asarray(value, dtype=want)
    if isinstance(value, (bool, np.bool_)) or want.kind == "b":
        ok = bool(arr) == bool(value)
    else:
        ok = arr.item() == value
    if not ok:
        raise ColumnsError(f"{where}: {value!r} does not fit {want}")
    return arr


def _group_arrays(group: Group, positions: Sequence, where: str) -> Dict[str, np.ndarray]:
    """The group's columns over `positions` (and its offsets, when its
    length varies by position)."""
    out: Dict[str, np.ndarray] = {}
    if group.count is None:
        for name, dtype, get in group.columns:
            values = [_strict(get(p), dtype, f"{where} {name}") for p in positions]
            out[name] = np.stack(values) if values else np.zeros(0, dtype)
        return out
    counts = np.array([group.count(p) for p in positions], dtype=np.int64)
    offsets = np.zeros(len(positions) + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    out[f"{group.name}_offsets"] = offsets
    for name, dtype, get in group.columns:
        parts = [_strict(get(p), dtype, f"{where} {name}") for p in positions]
        for part, n in zip(parts, counts):
            if part.shape[:1] != (n,):
                raise ColumnsError(f"{where} {name}: {part.shape[0]} entries where the group has {n}")
        inner = {part.shape[1:] for part in parts}
        if len(inner) > 1:
            raise ColumnsError(f"{where} {name}: entries of shapes {sorted(inner)}")
        out[name] = np.concatenate(parts) if parts else np.zeros(0, dtype)
    return out


def side_arrays(positions: Sequence, where: str) -> Dict[str, np.ndarray]:
    """A game-side's columns: the corpus's groups, and the own groups when
    its positions carry the masks and targets their player decided under."""
    own = [p.masks is not None for p in positions]
    if any(own) and not all(own):
        raise ColumnsError(f"{where}: some positions carry own masks and some do not")
    groups = CORPUS_GROUPS + (OWN_GROUPS if positions and all(own) else ())
    arrays: Dict[str, np.ndarray] = {}
    for group in groups:
        arrays.update(_group_arrays(group, positions, where))
    return arrays


# ---- the file ------------------------------------------------------------------

def _pack(arrays: Dict[str, np.ndarray]) -> Tuple[bytes, List]:
    """The arrays in one buffer, each on an `_ALIGN` boundary, and their
    table (name, dtype, shape, offset)."""
    table, chunks, offset = [], [], 0
    for name, arr in arrays.items():
        arr = np.ascontiguousarray(arr)
        pad = -offset % _ALIGN
        chunks.append(b"\0" * pad)
        offset += pad
        table.append([name, arr.dtype.str, list(arr.shape), offset])
        chunks.append(arr.tobytes())
        offset += arr.nbytes
    return b"".join(chunks), table


def write_columns(path: Path, seq) -> None:
    """A `GameSequence` as a columnar file, written atomically."""
    header = {"format": COLUMNS_FORMAT, "file": seq.file, "winner": int(seq.winner),
              "n_commands": int(seq.n_commands), "counts": {k: int(v) for k, v in seq.counts.items()},
              "sides": {}}
    blobs = []
    for side, positions in sorted(seq.sides.items()):
        raw, table = _pack(side_arrays(positions, f"{seq.file} side {side}"))
        blob = zlib.compress(raw, ZLIB_LEVEL)
        header["sides"][str(int(side))] = {"positions": len(positions), "nbytes": len(blob),
                                           "raw_nbytes": len(raw), "arrays": table}
        blobs.append(blob)
    # Offsets count from the end of the header, so the header does not depend on its own length.
    offset = 0
    for entry, blob in zip(header["sides"].values(), blobs):
        entry["offset"] = offset
        offset += len(blob)
    text = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(_MAGIC + struct.pack("<Q", len(text)) + text)
        for blob in blobs:
            f.write(blob)
    tmp.replace(path)


def read_header(f) -> Tuple[Dict, int]:
    """(header, the byte where the blobs start) of an open columnar file."""
    head = f.read(len(_MAGIC) + 8)
    if head[:len(_MAGIC)] != _MAGIC:
        raise ColumnsError(f"{getattr(f, 'name', 'the file')} is not a columnar sequence record")
    (n,) = struct.unpack("<Q", head[len(_MAGIC):])
    header = json.loads(f.read(n).decode("utf-8"))
    if header.get("format") != COLUMNS_FORMAT:
        raise ColumnsError(f"{getattr(f, 'name', 'the file')}: columns format {header.get('format')}, "
                           f"this reader reads {COLUMNS_FORMAT}")
    return header, len(head) + n


def read_side(path: Path, side: int) -> "ColumnarSide":
    """One side of a columnar file: its blob read and inflated, its arrays
    read-only views of the inflated bytes."""
    with open(path, "rb") as f:
        header, start = read_header(f)
        entry = header["sides"].get(str(int(side)))
        if entry is None:
            raise ColumnsError(f"{path} has no side {side}")
        f.seek(start + entry["offset"])
        blob = f.read(entry["nbytes"])
    return _side_of(path, header, int(side), entry, blob)


def read_sides(path: Path) -> Dict[int, "ColumnarSide"]:
    """Every side of a columnar file."""
    with open(path, "rb") as f:
        header, start = read_header(f)
        blobs = {}
        for side, entry in header["sides"].items():
            f.seek(start + entry["offset"])
            blobs[int(side)] = (entry, f.read(entry["nbytes"]))
    return {side: _side_of(path, header, side, entry, blob) for side, (entry, blob) in blobs.items()}


def _side_of(path: Path, header: Dict, side: int, entry: Dict, blob: bytes) -> "ColumnarSide":
    raw = zlib.decompress(blob, bufsize=max(1, entry["raw_nbytes"]))
    if len(raw) != entry["raw_nbytes"]:
        raise ColumnsError(f"{path} side {side}: {len(raw)} bytes inflated, the header says {entry['raw_nbytes']}")
    arrays = {}
    for name, dtype, shape, offset in entry["arrays"]:
        dt = np.dtype(dtype)
        count = int(np.prod(shape)) if shape else 1
        arrays[name] = np.frombuffer(raw, dtype=dt, count=count, offset=offset).reshape(shape)
    return ColumnarSide(file=header["file"], side=int(side), n=int(entry["positions"]), arrays=arrays)


# ---- a side in memory ------------------------------------------------------------

@dataclass
class ColumnLabel:
    """A label read from columns: the fields of `replay_dataset.ActionIndices`
    the trainer reads."""
    action_type: str
    actor_idx: int
    target_idx: Optional[int]
    weapon_idx: Optional[int]
    type_idx: Optional[int]
    target_off_subset: bool


@dataclass
class ColumnarSide:
    """A game-side's columns: `arrays` by name, `<group>_offsets` [n + 1]
    for the groups whose length varies by position."""
    file: str
    side: int
    n: int
    arrays: Dict[str, np.ndarray] = field(repr=False)

    def __post_init__(self):
        # The offsets as Python lists: a time step reads a few hundred spans.
        self._offsets = {name[:-len("_offsets")]: arr.tolist() for name, arr in self.arrays.items()
                         if name.endswith("_offsets")}

    def __len__(self) -> int:
        return self.n

    @property
    def own(self) -> bool:
        return "actor_mask" in self.arrays

    def span(self, group: str, i: int) -> Tuple[int, int]:
        off = self._offsets[group]
        return off[i], off[i + 1]

    def _slice(self, group: str, i: int, *names: str) -> List[np.ndarray]:
        lo, hi = self.span(group, i)
        return [self.arrays[n][lo:hi] for n in names]

    def label(self, i: int) -> ColumnLabel:
        a = self.arrays

        def opt(name):
            v = int(a[name][i])
            return None if v == NO_INDEX else v
        return ColumnLabel(action_type=LABEL_KINDS[int(a["label_kind"][i])], actor_idx=int(a["label_actor"][i]),
                           target_idx=opt("label_target"), weapon_idx=opt("label_weapon"),
                           type_idx=opt("label_type"), target_off_subset=bool(a["label_off_subset"][i]))

    def position(self, i: int):
        """Position `i` as a `SequencePosition` of views (the fields this
        record keeps; the left-out ones are None)."""
        from wesnoth_ai.encoder import RawEncoded
        from wesnoth_ai.sequence_records import SequencePosition
        a = self.arrays
        hx, hy, ht, hm, hd, nvu = self._slice("hex", i, "hex_xs", "hex_ys", "hex_terrain_ids", "hex_modifier_flags",
                                              "hex_dynamic_flags", "no_visible_unit")
        uo, ut, us, ux, uy, uf = self._slice("unit", i, "unit_is_ours", "unit_type_ids", "unit_side_ids", "unit_xs",
                                             "unit_ys", "unit_feats")
        ro, rt, rs, rx, ry, rf = self._slice("recruit", i, "recruit_is_ours", "recruit_type_ids", "recruit_side_ids",
                                             "recruit_xs", "recruit_ys", "recruit_feats")
        st, sx, sy, sf = self._slice("sighting", i, "sight_type_ids", "sight_xs", "sight_ys", "sight_feats")
        (hidden,) = self._slice("hidden", i, "hidden_tokens")
        raw = RawEncoded(
            hex_positions=None, hex_xs=hx, hex_ys=hy, hex_terrain_ids=ht, hex_modifier_flags=hm,
            hex_dynamic_flags=hd, unit_positions=None, unit_ids=None, unit_is_ours=uo, unit_type_ids=ut,
            unit_side_ids=us, unit_xs=ux, unit_ys=uy, unit_feats=uf, recruit_types=None, recruit_is_ours=ro,
            recruit_type_ids=rt, recruit_side_ids=rs, recruit_xs=rx, recruit_ys=ry, recruit_feats=rf,
            global_feats=a["global_feats"][i], our_faction_id=int(a["our_faction_id"][i]),
            their_faction_id=int(a["their_faction_id"][i]), material=float(a["material"][i]), observation=None,
            hex_subset=bool(a["hex_subset"][i]), their_faction_probs=a["their_faction_probs"][i],
            sight_type_ids=st, sight_xs=sx, sight_ys=sy, sight_feats=sf)
        masks = target = None
        if self.own:
            masks, target = self._own(i), self._own_target(i)
        return SequencePosition(raw=raw, label=self.label(i), hidden_tokens=hidden, no_visible_unit=nvu,
                                turn=int(a["turn"][i]), masks=masks, own_target=target)

    def _own(self, i: int):
        from wesnoth_ai.server_priors import PackedMasks
        a = self.arrays
        mask, n_att, tv, tb = self._slice("actor", i, "actor_mask", "n_attacks", "type_valid", "type_bias")
        width = int(a["mask_bits_width"][i])
        att, mov, uni = (x.reshape(-1, width) for x in self._slice("bits", i, "attack_valid", "move_valid",
                                                                      "union_valid"))
        (ab,) = self._slice("attack_bias", i, "attack_bias")
        return PackedMasks(actor_mask=mask, type_valid=tv, attack_valid=att, move_valid=mov, union_valid=uni,
                           n_attacks=n_att, end_turn_bias=float(a["mask_end_turn_bias"][i]),
                           n_units=int(a["mask_units"][i]), n_recruits=int(a["mask_recruits"][i]),
                           n_hexes=int(a["mask_hexes"][i]),
                           type_bias=tb if a["mask_type_bias_on"][i] else None,
                           attack_bias=(ab.reshape(mask.shape[0], int(a["mask_hexes"][i]))
                                        if a["mask_attack_bias_on"][i] else None))

    def _own_target(self, i: int):
        from wesnoth_ai.sequence_records import OwnTarget
        actor, kind, target, weapon, prior = self._slice("target", i, "target_actor", "target_kind",
                                                         "target_target", "target_weapon", "target_prior")
        return OwnTarget(actor=actor, kind=kind, target=target, weapon=weapon, prior=prior,
                         end_prior=float(self.arrays["target_end_prior"][i]))


# ---- equality with the pickled record ----------------------------------------------

_RAW_ARRAYS = ("hex_xs", "hex_ys", "hex_terrain_ids", "hex_modifier_flags", "hex_dynamic_flags", "unit_is_ours",
               "unit_type_ids", "unit_side_ids", "unit_xs", "unit_ys", "unit_feats", "recruit_is_ours",
               "recruit_type_ids", "recruit_side_ids", "recruit_xs", "recruit_ys", "recruit_feats", "global_feats",
               "their_faction_probs", "sight_type_ids", "sight_xs", "sight_ys", "sight_feats")
_RAW_VALUES = ("our_faction_id", "their_faction_id", "material", "hex_subset")
_LABEL_VALUES = ("action_type", "actor_idx", "target_idx", "weapon_idx", "type_idx", "target_off_subset")
_MASK_ARRAYS = ("actor_mask", "type_valid", "attack_valid", "move_valid", "union_valid", "n_attacks", "type_bias",
                "attack_bias")
_MASK_VALUES = ("end_turn_bias", "n_units", "n_recruits", "n_hexes")
_TARGET_ARRAYS = ("actor", "kind", "target", "weapon", "prior")


def _same_array(x, y) -> bool:
    """The same dtype, shape (leading unit axes aside) and bytes; both None."""
    if x is None or y is None:
        return x is None and y is None
    x, y = np.asarray(x), np.asarray(y)
    squeeze = (lambda s: tuple(s[next((i for i, d in enumerate(s) if d != 1), len(s)):]))
    return x.dtype == y.dtype and squeeze(x.shape) == squeeze(y.shape) and x.tobytes() == y.tobytes()


def _same_value(x, y) -> bool:
    return (x is None and y is None) or (x is not None and y is not None and bool(x == y))


def position_difference(kept, original) -> Optional[str]:
    """The first field the columns keep in which `kept` (a position read
    from columns) and `original` (the pickled one) differ; None when they
    agree on every one, bytes and dtypes included."""
    checks = [(f"raw.{n}", _same_array, getattr(kept.raw, n), getattr(original.raw, n)) for n in _RAW_ARRAYS]
    checks += [(f"raw.{n}", _same_value, getattr(kept.raw, n), getattr(original.raw, n)) for n in _RAW_VALUES]
    checks += [(f"label.{n}", _same_value, getattr(kept.label, n), getattr(original.label, n))
               for n in _LABEL_VALUES]
    checks += [("hidden_tokens", _same_array, kept.hidden_tokens, original.hidden_tokens),
               ("no_visible_unit", _same_array, kept.no_visible_unit, original.no_visible_unit),
               ("turn", _same_value, kept.turn, original.turn)]
    if (kept.masks is None) != (original.masks is None) or (kept.own_target is None) != (original.own_target is None):
        return "masks or own_target"
    if original.masks is not None:
        checks += [(f"masks.{n}", _same_array, getattr(kept.masks, n), getattr(original.masks, n))
                   for n in _MASK_ARRAYS]
        checks += [(f"masks.{n}", _same_value, getattr(kept.masks, n), getattr(original.masks, n))
                   for n in _MASK_VALUES]
    if original.own_target is not None:
        checks += [(f"own_target.{n}", _same_array, getattr(kept.own_target, n), getattr(original.own_target, n))
                   for n in _TARGET_ARRAYS]
        checks.append(("own_target.end_prior", _same_value, kept.own_target.end_prior,
                       original.own_target.end_prior))
    for name, same, x, y in checks:
        if not same(x, y):
            return name
    return None
