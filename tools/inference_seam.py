"""Inference seam for the actor/learner split (plan §3.1b, stage B1).

The MCTS search touches the neural nets through a tiny, fixed surface:
`encoder.encode(gs) -> EncodedState`, `model(encoded) -> ModelOutput`,
and `model.forward_batch([encoded,...]) -> [ModelOutput,...]`. This
module provides drop-in duck-types for that surface so the forward
pass can be RELOCATED to another process without touching `mcts.py`:

  actor process                          server process (owns GPU)
  -------------                          -------------------------
  RemoteEncoder.encode(gs):              InferenceServer.infer(raw):
    raw = encode_raw(gs, vocab)            enc = encode_from_raw(raw)   # trained
    enc = build_light_encoded(raw)         out = model(enc)
    enc._raw = raw   ----- raw ----->       return out.cpu()
    return enc                       <----- ModelOutput (CPU) -----

Why the cut is at `RawEncoded` (not `EncodedState`):
  * `encode` is two phases. Phase 1 (`encode_raw`, a free function) is
    pure-Python/numpy, weight-free, and picklable -- it was explicitly
    built for cross-process transport. Phase 2 (`encode_from_raw`) runs
    the TRAINED embeddings/projections, so it must live with the
    parameters (the server).
  * The action sampler downstream of the leaf forward
    (`enumerate_legal_actions_with_priors`) only reads RawEncoded's
    Python fields (positions / ids / types), the small `*_is_ours`
    flag arrays, and the stream COUNTS (U/R/H) -- never the dense
    `d_model` token tensors (those are model-only). So the actor can
    build a "light" EncodedState that carries exactly those fields plus
    width-1 placeholder token tensors (whose only use is `.size(1)`).
  * Vocab discipline: the actor's `encode_raw` uses a FROZEN, shared
    vocab; unseen names fall to the overflow bucket rather than
    mutating the dict (encoder.py:encode_raw). Pre-seed + freeze the
    vocab before spawning actors so the server's embedding rows line up.

The server returns ModelOutput on CPU: the actor's sampler builds its
legality masks on its local (CPU) device and combines them with the
returned logits, so the logits must be CPU too (and CUDA tensors don't
pickle across a process boundary anyway).

In stage B1 the "transport" is an in-process `InferenceServer` -- this
is verified loss-less by `test_inference_seam` (identical ModelOutput
and identical legal-action priors vs the direct path). Stage B2 swaps
the transport for an IPC client to a real server process; the seam
above is unchanged.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Protocol

import torch

from classes import GameState
from encoder import EncodedState, RawEncoded, encode_raw
from model import ModelOutput


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def move_model_output(out: ModelOutput, device: torch.device) -> ModelOutput:
    """Return a copy of `out` with every tensor field moved to
    `device` (non-tensor fields, e.g. num_units, passed through).
    Used to bring server outputs back to the actor's CPU device and to
    make them picklable across a process boundary."""
    kw = {}
    for f in dataclasses.fields(out):
        v = getattr(out, f.name)
        kw[f.name] = v.to(device) if torch.is_tensor(v) else v
    return type(out)(**kw)


def batched_outputs_to_cpu(outs: List[ModelOutput]) -> List[ModelOutput]:
    """Move a whole batch of (device-resident) ModelOutputs to CPU
    with ONE device->host transfer per tensor FIELD: per field,
    flatten every sample's tensor to 1D, cat on-device (one kernel),
    one .cpu(), then split+reshape host-side (shapes/dtypes are
    per-sample and preserved; a field is same-dtype across samples).
    Semantically identical to per-sample move_model_output -- pinned
    by test_batched_outputs_to_cpu_matches_per_sample -- but 9
    transfers per batch instead of 9xB serialized syncs."""
    if not outs:
        return []
    cpu = torch.device("cpu")
    fields = dataclasses.fields(outs[0])
    tensor_names = [f.name for f in fields
                    if torch.is_tensor(getattr(outs[0], f.name))]
    per_field: Dict[str, List[torch.Tensor]] = {}
    for name in tensor_names:
        ts = [getattr(o, name) for o in outs]
        shapes = [t.shape for t in ts]
        flat = torch.cat([t.reshape(-1) for t in ts]).to(cpu)
        parts = flat.split([t.numel() for t in ts])
        per_field[name] = [p.reshape(s) for p, s in zip(parts, shapes)]
    rebuilt = []
    for i, o in enumerate(outs):
        kw = {}
        for f in fields:
            v = getattr(o, f.name)
            kw[f.name] = (per_field[f.name][i]
                          if f.name in per_field else v)
        rebuilt.append(type(o)(**kw))
    return rebuilt


def output_to_wire(out: ModelOutput) -> Dict:
    """ModelOutput -> plain-numpy dict for mp-queue transport.

    Plain numpy arrays pickle INLINE into the queue byte stream;
    torch tensors instead route through torch.multiprocessing's
    tensor-sharing machinery -- one staged shm file (+fd) per tensor
    per message under the 'file_system' strategy. At ~9 tensors per
    ModelOutput that constant cost is what capped the old per-leaf
    protocol at ~200 req/s with the GPU idle (and fed the
    2026-07-03 fd-leak incident under 'file_descriptor')."""
    w = {}
    for f in dataclasses.fields(out):
        v = getattr(out, f.name)
        if torch.is_tensor(v):
            w[f.name] = ("t", v.detach().cpu().numpy())
        else:
            w[f.name] = ("p", v)
    return w


def output_from_wire(w: Dict) -> ModelOutput:
    """Inverse of output_to_wire (actor side). torch.from_numpy is
    zero-copy; MCTS only reads these tensors."""
    kw = {}
    for name, (tag, v) in w.items():
        kw[name] = torch.from_numpy(v) if tag == "t" else v
    return ModelOutput(**kw)


def build_light_encoded(
    raw: RawEncoded, device: torch.device,
) -> EncodedState:
    """Build the EncodedState the ACTION SAMPLER needs from a
    RawEncoded, WITHOUT the trained dense embeddings (those live on the
    server). The `*_tokens` are width-1 placeholders -- the sampler
    only ever reads their `.size(1)` (the stream length) and `.device`,
    never their values. Everything the sampler actually consumes (the
    Python position/id/type lists, the `*_is_ours` flags, the
    visible-unit set, the pos->hex map) is reconstructed exactly from
    `raw`."""
    H = len(raw.hex_positions)
    U = len(raw.unit_positions)
    R = len(raw.recruit_types)

    def _ph(n: int) -> torch.Tensor:            # placeholder token tensor
        return torch.zeros((1, n, 1), device=device)

    pos_to_hex = {(p.x, p.y): j for j, p in enumerate(raw.hex_positions)}
    unit_is_ours = torch.from_numpy(raw.unit_is_ours).to(device).unsqueeze(0)
    recruit_is_ours = (torch.from_numpy(raw.recruit_is_ours)
                       .to(device).unsqueeze(0))
    return EncodedState(
        hex_tokens=_ph(H),
        hex_positions=raw.hex_positions,
        pos_to_hex=pos_to_hex,
        unit_tokens=_ph(U),
        unit_is_ours=unit_is_ours,
        unit_positions=raw.unit_positions,
        unit_ids=raw.unit_ids,
        recruit_tokens=_ph(R),
        recruit_is_ours=recruit_is_ours,
        recruit_types=raw.recruit_types,
        global_token=_ph(1),
        end_turn_token=_ph(1),
        recruit_is_ours_np=raw.recruit_is_ours,
        visible_unit_ids=frozenset(raw.unit_ids),
    )


# ---------------------------------------------------------------------
# Transport protocol + in-process server
# ---------------------------------------------------------------------

class InferenceTransport(Protocol):
    """What RemoteModel needs from whatever sits behind the seam."""
    def infer(self, raw: RawEncoded) -> ModelOutput: ...
    def infer_batch(self, raws: List[RawEncoded]) -> List[ModelOutput]: ...


class InferenceServer:
    """Owns the trained encoder + model and turns RawEncoded into
    ModelOutput. Runs `encode_from_raw` (phase 2, trained embeddings)
    then the model forward, returning the output on `output_device`
    (CPU by default -- see module docstring).

    This is the SAME object the B2 server process runs; in B1 it's
    called in-process by RemoteModel for the parity test."""

    def __init__(
        self, model, encoder, *,
        device: Optional[torch.device] = None,
        output_device: Optional[torch.device] = None,
    ):
        self._model = model
        self._encoder = encoder
        self._device = device or next(model.parameters()).device
        self._out_dev = output_device or torch.device("cpu")

    def infer(self, raw: RawEncoded) -> ModelOutput:
        with torch.no_grad():
            enc = self._encoder.encode_from_raw(raw, device=self._device)
            out = self._model(enc)
        return move_model_output(out, self._out_dev)

    def infer_batch(self, raws: List[RawEncoded]) -> List[ModelOutput]:
        if not raws:
            return []
        with torch.no_grad():
            encs = self._encoder.encode_from_raw_batch(
                raws, device=self._device)
            outs = self._model.forward_batch(encs)
            if (self._out_dev.type == "cpu"
                    and outs and torch.is_tensor(outs[0].actor_logits)
                    and outs[0].actor_logits.device.type != "cpu"):
                # Coalesced device->host: one flatten-cat + one .cpu()
                # per FIELD per batch (9 transfers) instead of one
                # sync per field per sample (9xB -- ~414 serialized
                # cudaMemcpy per 46-leaf batch = the 11.9ms/leaf
                # serve ceiling measured on the 4090, 2026-07-22).
                return batched_outputs_to_cpu(outs)
        return [move_model_output(o, self._out_dev) for o in outs]


# ---------------------------------------------------------------------
# Actor-side duck-types (passed to mcts_search in place of encoder/model)
# ---------------------------------------------------------------------

class RemoteEncoder:
    """Duck-types `encoder.encode`. Builds a RawEncoded against a
    FROZEN, shared vocab (no growth), then a light EncodedState carrying
    that RawEncoded on `._raw` for RemoteModel to ship. Holds no trained
    parameters."""

    def __init__(
        self, type_to_id: Dict[str, int], faction_to_id: Dict[str, int],
        *, device: Optional[torch.device] = None,
    ):
        self._type_to_id = type_to_id
        self._faction_to_id = faction_to_id
        self._device = device or torch.device("cpu")

    def encode(self, game_state: GameState) -> EncodedState:
        raw = encode_raw(
            game_state,
            type_to_id=self._type_to_id,
            faction_to_id=self._faction_to_id,
        )
        enc = build_light_encoded(raw, self._device)
        # Stash the wire payload for RemoteModel; EncodedState is a
        # plain dataclass (no __slots__), so this attribute sticks.
        enc._raw = raw
        return enc


class RemoteModel:
    """Duck-types `model(encoded)` and `model.forward_batch(list)` by
    forwarding the stashed RawEncoded(s) through a transport (an
    in-process InferenceServer in B1; an IPC client in B2)."""

    def __init__(self, transport: InferenceTransport):
        self._t = transport

    def __call__(self, encoded: EncodedState) -> ModelOutput:
        return self._t.infer(encoded._raw)

    def forward_batch(self, encoded_list: List[EncodedState]) -> List[ModelOutput]:
        return self._t.infer_batch([e._raw for e in encoded_list])
