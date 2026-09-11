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
import io
import time
from typing import Dict, List, Optional, Protocol, Tuple

import torch

from wesnoth_ai.classes import GameState
from wesnoth_ai.encoder import EncodedState, RawEncoded, encode_raw
from wesnoth_ai.model import ModelOutput


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
    visible-unit set, the pos->hex map, the hex-basis flag the
    sampler's relevant-set tripwires read) is reconstructed exactly
    from `raw`."""
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
        hex_subset=bool(raw.hex_subset),
        observation=getattr(raw, "observation", None),
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
        autocast_bf16: Optional[bool] = None,
        packed_embed: bool = False,
    ):
        self._model = model
        self._encoder = encoder
        # Packed embed (design note section 14): the batch's token
        # embeddings come from one pinned buffer and are ordered on the
        # device (encoder.encode_from_raw_embedded + model.forward_embedded)
        # instead of being built as padded streams and packed after.
        self._packed_embed = bool(packed_embed)
        self._device = device or next(model.parameters()).device
        self._out_dev = output_device or torch.device("cpu")
        # None: follow the model's `infer_autocast_bf16`. The actor
        # pool sets it explicitly so the generation path runs bf16
        # while the learner's own in-process probes keep the model's
        # setting (fp32 unless the policy was loaded with infer_bf16).
        self._autocast_bf16 = autocast_bf16

    def _use_bf16(self) -> bool:
        if self._device.type != "cuda":
            return False
        if self._autocast_bf16 is None:
            return bool(getattr(self._model, "infer_autocast_bf16", False))
        return bool(self._autocast_bf16)

    def infer(self, raw: RawEncoded) -> ModelOutput:
        with torch.no_grad():
            enc = self._encoder.encode_from_raw(raw, device=self._device)
            out = self._model(enc)
        return move_model_output(out, self._out_dev)

    def infer_batch(self, raws, stats: Optional[Dict[str, float]] = None) -> List[ModelOutput]:
        """`raws`: RawEncoded items, or (RawEncoded, PackedMasks) pairs
        for server-side priors (wesnoth_ai/server_priors.py). Mixed
        lists are refused: one batch, one protocol. `stats`, when
        given on a CUDA device, accumulates the priors protocol's
        device-stream milliseconds (encode, forward, priors) under
        "gpu_ms" and the host seconds of its stages under t_encode,
        t_forward (launches), t_priors (launches), t_finish (the one
        wait for the device) and t_reply (building the outputs)."""
        if not raws:
            return []
        paired = [isinstance(r, tuple) for r in raws]
        if any(paired):
            if not all(paired):
                raise ValueError("infer_batch: mixed raw and (raw, masks) items")
            return self._infer_with_priors([r for r, _ in raws],
                                           [m for _, m in raws], stats)
        with torch.no_grad():
            encs = self._encoder.encode_from_raw_batch(
                raws, device=self._device)
            outs = self._model.forward_batch(
                encs, autocast_bf16=self._use_bf16())
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

    def _infer_with_priors(self, raws, packs, stats=None) -> List[ModelOutput]:
        """Batched forward + masked softmaxes on the device; replies
        carry the compact legal actions, value and cliffness, with
        placeholder logits (the actor never reads them). The head
        outputs ride back in the priors' one device->host transfer.

        On CUDA with `stats`, a pair of stream events brackets encode
        + forward + priors: the elapsed milliseconds are the stream's
        wall time between the two records, i.e. device work plus any
        gap where the stream waited for the host to launch (and, with
        several serve threads on one stream, the others' interleaved
        work). The host seconds per stage are recorded on CUDA only:
        off the device, launching and waiting are not separable."""
        from wesnoth_ai.server_priors import start_priors
        model = self._model
        bf16 = self._use_bf16()
        timing = stats is not None and self._device.type == "cuda"
        if timing:
            ev_start = torch.cuda.Event(enable_timing=True)
            ev_end = torch.cuda.Event(enable_timing=True)
            ev_start.record()
        t0 = time.perf_counter()
        with torch.no_grad():
            if self._packed_embed:
                streams = self._encoder.encode_from_raw_embedded(raws, device=self._device)

                def forward():
                    return model.forward_embedded(streams)
            else:
                streams = self._encoder.encode_from_raw_padded(raws, device=self._device)

                def forward():
                    return model.forward_streams(*streams)
            t1 = time.perf_counter()
            if bf16:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    padded = forward().float32()
            else:
                padded = forward()
            t2 = time.perf_counter()
            names = ["value", "value_logits", "cliffness"]
            names += [n for n in ("aux_score", "moves_left") if getattr(padded, n) is not None]
            pending = start_priors(padded, packs, [getattr(padded, n) for n in names])
            if timing:
                ev_end.record()
            t3 = time.perf_counter()
            compact, host = pending.finish()
            t4 = time.perf_counter()
        if timing:
            stats["gpu_ms"] = stats.get("gpu_ms", 0.0) + ev_start.elapsed_time(ev_end)
        small = {n: torch.from_numpy(a) for n, a in zip(names, host)}
        aux = small.get("aux_score")
        ml = small.get("moves_left")
        outs = []
        for b, (U, R, H) in enumerate(padded.sizes):
            A = U + R + 1
            outs.append(ModelOutput(
                actor_logits=torch.zeros(1, A), actor_kind=padded.actor_kind[b:b + 1, :A],
                type_logits=torch.zeros(1, A, 0), target_logits=torch.zeros(1, A, 0),
                weapon_logits=torch.zeros(1, A, 0),
                value=small["value"][b:b + 1], value_logits=small["value_logits"][b:b + 1],
                cliffness=small["cliffness"][b:b + 1], num_units=U, num_recruits=R,
                aux_score=aux[b:b + 1] if aux is not None else None,
                moves_left=ml[b:b + 1] if ml is not None else None,
                legal_compact=compact[b]))
        if timing:
            t5 = time.perf_counter()
            for key, dt in (("t_encode", t1 - t0), ("t_forward", t2 - t1), ("t_priors", t3 - t2),
                            ("t_finish", t4 - t3), ("t_reply", t5 - t4)):
                stats[key] = stats.get(key, 0.0) + dt
        return outs


# ---------------------------------------------------------------------
# A copy of the inference pair in another process (the serve processes
# of tools/actor_pool.py)
# ---------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class InferenceBlueprint:
    """Constructor arguments that rebuild a learner's inference model
    and encoder elsewhere, read off the live modules so the copy's
    state_dict keys and shapes match the learner's. Weights travel
    separately (pack_inference_state / load_inference_state)."""
    model_kwargs: Dict
    encoder_kwargs: Dict


def inference_blueprint(model, encoder) -> InferenceBlueprint:
    layer = model.encoder.layers[0]
    return InferenceBlueprint(
        model_kwargs=dict(
            d_model=int(model.d_model), num_layers=len(model.encoder.layers),
            num_heads=int(layer.self_attn.num_heads),
            d_ff=int(layer.linear1.out_features), dropout=float(layer.dropout.p),
            max_attacks=int(model.max_attacks), aux_score=bool(model.has_aux_score),
            moves_left=bool(model.has_moves_left), gbc=bool(model.has_gbc)),
        encoder_kwargs=dict(d_model=int(encoder.d_model),
                            relevant_set_hexes=bool(encoder.relevant_set_hexes)))


def build_inference_pair(blueprint: InferenceBlueprint, device: torch.device) -> Tuple:
    """A fresh (model, encoder) at the blueprint's architecture, in eval
    mode on `device`, with random weights until load_inference_state."""
    from wesnoth_ai.encoder import GameStateEncoder
    from wesnoth_ai.model import WesnothModel
    model = WesnothModel(**blueprint.model_kwargs).to(device).eval()
    encoder = GameStateEncoder(**blueprint.encoder_kwargs).to(device).eval()
    return model, encoder


def pack_inference_state(model, encoder) -> bytes:
    """Both state_dicts as one torch.save byte string: a single
    message on a control queue, every dtype supported, no shared-memory
    or CUDA IPC handle whose lifetime the sender would have to manage
    (see ActorPool.sync_servers)."""
    buf = io.BytesIO()
    torch.save({"model": model.state_dict(), "encoder": encoder.state_dict()}, buf)
    return buf.getvalue()


def load_inference_state(blob: bytes, model, encoder, device: torch.device) -> None:
    state = torch.load(io.BytesIO(blob), map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    encoder.load_state_dict(state["encoder"])
    model.eval()
    encoder.eval()


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
        relevant_set: bool = False,
        server_priors: bool = False,
        fog_hides_enemy_villages: bool = False,
    ):
        self._type_to_id = type_to_id
        self._faction_to_id = faction_to_id
        self._fog_hides_enemy_villages = bool(fog_hides_enemy_villages)
        self._device = device or torch.device("cpu")
        # Server-side priors: the actor packs the legality masks at
        # encode time and RemoteModel ships them with the leaf.
        self._server_priors = bool(server_priors)
        # Action-space basis (project round-2 C3: hardcoded False
        # made pool actors encode full-board while an inherited
        # --relevant-set-hexes put the learner on the relevant-set
        # basis -- with no tripwire on this path).
        self._relevant_set = bool(relevant_set)

    def encode(self, game_state: GameState) -> EncodedState:
        raw = encode_raw(
            game_state,
            type_to_id=self._type_to_id,
            faction_to_id=self._faction_to_id,
            relevant_set=self._relevant_set,
            fog_hides_enemy_villages=self._fog_hides_enemy_villages,
        )
        enc = build_light_encoded(raw, self._device)
        # Stash the wire payload for RemoteModel; EncodedState is a
        # plain dataclass (no __slots__), so this attribute sticks.
        enc._raw = raw
        if self._server_priors:
            from wesnoth_ai.server_priors import pack_masks
            enc._masks = pack_masks(enc, game_state)
        return enc


class RemoteModel:
    """Duck-types `model(encoded)` and `model.forward_batch(list)` by
    forwarding the stashed RawEncoded(s) through a transport (an
    in-process InferenceServer in B1; an IPC client in B2)."""

    def __init__(self, transport: InferenceTransport):
        self._t = transport

    @staticmethod
    def _payload(encoded: EncodedState):
        masks = getattr(encoded, "_masks", None)
        return encoded._raw if masks is None else (encoded._raw, masks)

    def __call__(self, encoded: EncodedState) -> ModelOutput:
        payload = self._payload(encoded)
        if isinstance(payload, tuple):
            return self._t.infer_batch([payload])[0]
        return self._t.infer(payload)

    def forward_batch(self, encoded_list: List[EncodedState]) -> List[ModelOutput]:
        return self._t.infer_batch([self._payload(e) for e in encoded_list])
