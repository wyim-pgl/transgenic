"""B6 evidence-mode model (plan B6_WP2_WP4_PLAN_v4 section 9; protocol A46 rules 6-7).

One model, two modes. Per-base evidence tracks (K = 31) enter through the existing linear fusion of the
B5 encoder (``HyenaEncoder.evidence_proj``, added before positional injection and downsampling). Transcript
chains enter through a separate memory of 512 elements that every decoder layer reads with a gated
cross-attention block placed after ``encoder_attn_layer_norm``:

    v   = W_d * M[idx_d] + W_a * M[idx_a] + E_order + E_order_rev + E_slot + E_source + E_strand
          + w_count * log1p(units_bin)                      (w_count a scalar, as b6_track_spec.MODEL_DIMS)                      (M = downsampled encoder states, B x Lmax/6 x 1536)
    x  <- x + tanh(alpha_l) * m * ChainAttn_l(x, v)         (m = chain_mask.any(-1), alpha_l = 0 at init)

Exactness contracts (tested in ``tests/test_b6_model.py``):
  * no chain inputs -> the B5 forward is called unchanged;
  * all chains masked -> the update is ``torch.where(m, ..., x)``, so the layer output keeps x's bits exactly
    (a literal ``x + 0 * out`` would flip a -0.0 to +0.0);
  * zero tracks -> ``evidence_proj(0) == 0`` (B5/MVP fusion, reused);
  * generation computes v once from the encoder states and lets HF repeat it for beams; there is no chain K/V
    cache and ``_reorder_cache`` is the B5 one.

This file is new; the 92 files pinned by ``tests/test_mvp_final_source_integration.py`` are imported, never
edited. The LED decoder layer and B5 backbone are upgraded in place by class substitution
(``_adopt_subclass``) so that every B5 parameter keeps its name, its initialisation and its RNG consumption;
the chain modules are created and initialised afterwards.

Constants below are the model-side shape contract of plan section 7.2/7.7. The track and chain *content*
spec (channel order, permutations, spec hashes) belongs to ``transgenic.evidence.b6_track_spec`` (Writer B);
``tests/test_b6_integration_contract.py`` checks that the two agree once that module is present.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.led import modeling_led as _led

from transgenic.model import modeling_HyenaTransgenic as _b5
from transgenic.model.configuration_transgenic import HyenaTransgenicConfig

# ---- shape contract (plan 7.1, 7.2, 7.7, 9) -------------------------------------------------------------
B6_EVIDENCE_CHANNELS = 31
CHAIN_ELEMENTS = 512                    # element 0 is the dummy (mask False, index 0, features 0)
CHAIN_FEAT_COLUMNS = ("order_fwd", "order_rev", "slot", "source", "strand", "units_bin")
CHAIN_EMBED_ROWS = {"order_fwd": 32, "order_rev": 32, "slot": 64, "source": 3, "strand": 2}
UNITS_BIN_MAX = 64
DOWNSAMPLE_FACTOR = 6
CHAIN_D_MODEL = 512
CHAIN_HEADS = 8
CHAIN_DROPOUT = 0.1
B6_MODEL_VERSION = "b6-chain-gated-xattn-v1"
ALPHA_PARAM_SUFFIX = "chain_attn.alpha"


def b6_config(**kwargs) -> HyenaTransgenicConfig:
    """HyenaTransgenicConfig for the B6 architecture (evidence_channels = 31 plus the chain block)."""
    kwargs.setdefault("evidence_channels", B6_EVIDENCE_CHANNELS)
    kwargs.setdefault("b6_chain_d_model", CHAIN_D_MODEL)
    kwargs.setdefault("b6_chain_heads", CHAIN_HEADS)
    kwargs.setdefault("b6_chain_dropout", CHAIN_DROPOUT)
    kwargs.setdefault("b6_chain_attn_dropout", kwargs["b6_chain_dropout"])
    kwargs.setdefault("b6_chain_elements", CHAIN_ELEMENTS)
    kwargs.setdefault("b6_model_version", B6_MODEL_VERSION)
    config = HyenaTransgenicConfig(**kwargs)
    validate_b6_config(config)
    return config


def validate_b6_config(config) -> None:
    """Fail closed on a config that is not the B6 architecture."""
    if getattr(config, "b6_model_version", None) != B6_MODEL_VERSION:
        raise ValueError(f"not a B6 config: b6_model_version={getattr(config, 'b6_model_version', None)!r}; "
                         f"build it with b6_config()")
    if config.evidence_channels != B6_EVIDENCE_CHANNELS:
        raise ValueError(f"B6 needs evidence_channels={B6_EVIDENCE_CHANNELS}, got {config.evidence_channels}")
    d, h = int(config.b6_chain_d_model), int(config.b6_chain_heads)
    if d <= 0 or h <= 0 or d % h:
        raise ValueError(f"b6_chain_d_model {d} must be a positive multiple of b6_chain_heads {h}")
    if int(config.b6_chain_elements) != CHAIN_ELEMENTS:
        raise ValueError(f"b6_chain_elements must be {CHAIN_ELEMENTS}")
    for name in ("b6_chain_dropout", "b6_chain_attn_dropout"):
        if not 0.0 <= float(getattr(config, name)) < 1.0:
            raise ValueError(f"{name} must be in [0, 1)")


# ---- chain memory -------------------------------------------------------------------------------------------

def check_chain_shapes(chain_idx, chain_feat, chain_mask, batch: int) -> None:
    """Static (shape/dtype) checks only, so a compiled forward has no data-dependent branch.
    Value checks (index range, dummy column, feature ranges) are in ``b6_runtime.validate_b6_batch``."""
    n = CHAIN_ELEMENTS
    if not (isinstance(chain_idx, torch.Tensor) and chain_idx.dtype == torch.int64
            and tuple(chain_idx.shape) == (batch, n, 2)):
        raise ValueError(f"chain_idx must be int64[{batch},{n},2]")
    if not (isinstance(chain_feat, torch.Tensor) and chain_feat.dtype in (torch.int16, torch.int32, torch.int64)
            and tuple(chain_feat.shape) == (batch, n, len(CHAIN_FEAT_COLUMNS))):
        raise ValueError(f"chain_feat must be an integer tensor [{batch},{n},{len(CHAIN_FEAT_COLUMNS)}]")
    if not (isinstance(chain_mask, torch.Tensor) and chain_mask.dtype == torch.bool
            and tuple(chain_mask.shape) == (batch, n)):
        raise ValueError(f"chain_mask must be bool[{batch},{n}]")
    if not (chain_idx.device == chain_feat.device == chain_mask.device):
        raise ValueError("chain tensors must share one device")


class ChainMemory(nn.Module):
    """Builds the 512 chain elements from the downsampled encoder states (plan 9, "Gather")."""

    def __init__(self, decoder_dim: int, chain_dim: int):
        super().__init__()
        self.w_d = nn.Linear(decoder_dim, chain_dim)
        self.w_a = nn.Linear(decoder_dim, chain_dim)
        self.e_order = nn.Embedding(CHAIN_EMBED_ROWS["order_fwd"], chain_dim)
        self.e_order_rev = nn.Embedding(CHAIN_EMBED_ROWS["order_rev"], chain_dim)
        self.e_slot = nn.Embedding(CHAIN_EMBED_ROWS["slot"], chain_dim)
        self.e_source = nn.Embedding(CHAIN_EMBED_ROWS["source"], chain_dim)
        self.e_strand = nn.Embedding(CHAIN_EMBED_ROWS["strand"], chain_dim)
        self.w_count = nn.Parameter(torch.zeros(()))        # scalar (CHAIN_SPEC MODEL_DIMS: scalar * log1p)

    def forward(self, states: torch.Tensor, chain_idx: torch.Tensor, chain_feat: torch.Tensor) -> torch.Tensor:
        b, s, d = states.shape
        n = chain_idx.shape[1]
        flat = chain_idx.reshape(b, n * 2)
        gathered = torch.gather(states, 1, flat.unsqueeze(-1).expand(b, n * 2, d)).view(b, n, 2, d)
        feat = chain_feat.to(torch.int64)                                   # int16 on the wire (plan 7.7)
        v = self.w_d(gathered[:, :, 0]) + self.w_a(gathered[:, :, 1])
        v = (v + self.e_order(feat[..., 0]) + self.e_order_rev(feat[..., 1]) + self.e_slot(feat[..., 2])
             + self.e_source(feat[..., 3]) + self.e_strand(feat[..., 4]))
        units = torch.log1p(feat[..., 5].to(v.dtype)).unsqueeze(-1)
        return v + self.w_count.to(v.dtype) * units


class ChainCrossAttention(nn.Module):
    """Gated cross-attention from decoder states to the chain memory (8 x 64 heads)."""

    def __init__(self, decoder_dim: int, chain_dim: int, heads: int, dropout: float,
                 attn_dropout: Optional[float] = None):
        super().__init__()
        self.heads, self.head_dim = heads, chain_dim // heads
        self.scale = self.head_dim ** -0.5
        self.dropout = dropout                                   # on the projected output (A46 rule 6a)
        self.attn_dropout = dropout if attn_dropout is None else attn_dropout   # on the softmax weights
        self.q_proj = nn.Linear(decoder_dim, chain_dim)
        self.k_proj = nn.Linear(chain_dim, chain_dim)
        self.v_proj = nn.Linear(chain_dim, chain_dim)
        self.out_proj = nn.Linear(chain_dim, decoder_dim)
        self.alpha = nn.Parameter(torch.zeros(()))          # tanh(0) = 0: step-0 output equals the chain-less one

    def attend(self, x: torch.Tensor, memory: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
        """Attention output (before the gate). All-masked rows get exactly zero weights and a finite output."""
        b, t, _ = x.shape
        n = memory.shape[1]
        q = self.q_proj(x).view(b, t, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(memory).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        val = self.v_proj(memory).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        scores = scores.masked_fill(~key_mask[:, None, None, :], torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        weights = weights.masked_fill(~key_mask.any(-1)[:, None, None, None], 0.0)
        weights = F.dropout(weights, p=self.attn_dropout, training=self.training)
        ctx = torch.matmul(weights.to(val.dtype), val).transpose(1, 2).reshape(b, t, self.heads * self.head_dim)
        return F.dropout(self.out_proj(ctx), p=self.dropout, training=self.training)

    def forward(self, x, memory, key_mask, row_mask):
        out = self.attend(x, memory, key_mask)
        gate = torch.tanh(self.alpha).to(out.dtype)
        return torch.where(row_mask[:, None, None], x + gate * out, x)


# ---- decoder layer and loop ---------------------------------------------------------------------------------

class B6DecoderLayer(_led.LEDDecoderLayer):
    """LEDDecoderLayer with the chain block after ``encoder_attn_layer_norm``.

    Instances are made by ``_adopt_subclass`` on the LED layer the B5 backbone already built, followed by
    ``add_module('chain_attn', ...)``. Without chain tensors it is the parent forward, unchanged.
    """

    def forward(self, hidden_states, attention_mask=None, encoder_hidden_states=None, encoder_attention_mask=None,
                layer_head_mask=None, cross_attn_layer_head_mask=None, past_key_value=None,
                output_attentions=False, use_cache=True, chain_memory=None, chain_key_mask=None, chain_row_mask=None):
        if chain_memory is None:
            return super().forward(hidden_states, attention_mask=attention_mask,
                                   encoder_hidden_states=encoder_hidden_states,
                                   encoder_attention_mask=encoder_attention_mask, layer_head_mask=layer_head_mask,
                                   cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                                   past_key_value=past_key_value, output_attentions=output_attentions,
                                   use_cache=use_cache)
        # Body of LEDDecoderLayer.forward (transformers 4.44-4.57), with the chain block inserted.
        residual = hidden_states
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states, past_key_value=self_attn_past_key_value, attention_mask=attention_mask,
            layer_head_mask=layer_head_mask, output_attentions=output_attentions)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = self.self_attn_layer_norm(residual + hidden_states)

        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states, key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask, layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value, output_attentions=output_attentions)
            hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = self.encoder_attn_layer_norm(residual + hidden_states)
            present_key_value = present_key_value + cross_attn_present_key_value

        hidden_states = self.chain_attn(hidden_states, chain_memory, chain_key_mask, chain_row_mask)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = self.final_layer_norm(residual + hidden_states)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


def b6_decoder_forward(decoder, *, input_ids=None, attention_mask=None, encoder_hidden_states=None,
                       encoder_attention_mask=None, head_mask=None, cross_attn_head_mask=None, past_key_values=None,
                       inputs_embeds=None, use_cache=None, output_attentions=None, output_hidden_states=None,
                       return_dict=None, chain_memory=None, chain_key_mask=None):
    """``LEDDecoder.forward`` (transformers 4.44-4.57) with the chain tensors passed to every layer.

    Under gradient checkpointing the chain tensors are explicit positional arguments of the checkpointed call,
    so non-reentrant recomputation sees them as inputs. LayerDrop drops the whole layer, chain block included.
    """
    cfg = decoder.config
    output_attentions = output_attentions if output_attentions is not None else cfg.output_attentions
    output_hidden_states = output_hidden_states if output_hidden_states is not None else cfg.output_hidden_states
    use_cache = use_cache if use_cache is not None else cfg.use_cache
    return_dict = return_dict if return_dict is not None else cfg.use_return_dict
    if (input_ids is None) == (inputs_embeds is None):
        raise ValueError("specify exactly one of decoder input_ids and inputs_embeds")
    if input_ids is not None:
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
    else:
        input_shape = inputs_embeds.size()[:-1]
    past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
    if inputs_embeds is None:
        inputs_embeds = decoder.embed_tokens(input_ids)

    combined_attention_mask = None
    if input_shape[-1] > 1:
        combined_attention_mask = _led._create_4d_causal_attention_mask(
            input_shape, inputs_embeds.dtype, inputs_embeds.device, past_key_values_length=past_key_values_length)
    if attention_mask is not None and combined_attention_mask is not None:
        combined_attention_mask = combined_attention_mask + _led._prepare_4d_attention_mask_inverted(
            attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
    if encoder_hidden_states is not None and encoder_attention_mask is not None:
        encoder_attention_mask = _led._prepare_4d_attention_mask_inverted(
            encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

    positions = decoder.embed_positions(input_shape, past_key_values_length)
    hidden_states = decoder.layernorm_embedding(inputs_embeds + positions)
    hidden_states = F.dropout(hidden_states, p=decoder.dropout, training=decoder.training)

    checkpointing = bool(getattr(decoder, "gradient_checkpointing", False)) and decoder.training
    if checkpointing:
        use_cache = False
    for mask, name in ((head_mask, "head_mask"), (cross_attn_head_mask, "cross_attn_head_mask")):
        if mask is not None and mask.size()[0] != len(decoder.layers):
            raise ValueError(f"{name} must cover {len(decoder.layers)} layers")

    chain_row_mask = chain_key_mask.any(-1)
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    all_cross_attentions = () if output_attentions else None
    next_decoder_cache = () if use_cache else None
    for idx, layer in enumerate(decoder.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        # LayerDrop with LED's probability and its one torch.rand([]) draw per layer, applied as a select rather
        # than a Python branch on a tensor (a Dynamo graph break whose resume frames would compile late and
        # trip the post-warm-up recompile guard). A dropped layer, chain block included, leaves hidden_states
        # unchanged; its compute is discarded instead of skipped.
        drop = (torch.rand([], device=hidden_states.device) < decoder.layerdrop) if decoder.training else None
        layer_input = hidden_states
        past_key_value = past_key_values[idx] if past_key_values is not None else None
        layer_head = head_mask[idx] if head_mask is not None else None
        cross_head = cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
        if checkpointing:
            layer_outputs = decoder._gradient_checkpointing_func(
                layer.__call__, hidden_states, combined_attention_mask, encoder_hidden_states,
                encoder_attention_mask, layer_head, cross_head, None, output_attentions, use_cache,
                chain_memory, chain_key_mask, chain_row_mask)
        else:
            layer_outputs = layer(hidden_states, attention_mask=combined_attention_mask,
                                  encoder_hidden_states=encoder_hidden_states,
                                  encoder_attention_mask=encoder_attention_mask, layer_head_mask=layer_head,
                                  cross_attn_layer_head_mask=cross_head, past_key_value=past_key_value,
                                  output_attentions=output_attentions, use_cache=use_cache,
                                  chain_memory=chain_memory, chain_key_mask=chain_key_mask,
                                  chain_row_mask=chain_row_mask)
        hidden_states = layer_outputs[0] if drop is None else torch.where(drop, layer_input, layer_outputs[0])
        if use_cache:
            next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)
        if output_attentions:
            all_self_attns += (layer_outputs[1],)
            all_cross_attentions += (layer_outputs[2],)
    if output_hidden_states:
        all_hidden_states += (hidden_states,)
    next_cache = next_decoder_cache if use_cache else None
    if not return_dict:
        return tuple(v for v in (hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions)
                     if v is not None)
    return BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=hidden_states, past_key_values=next_cache, hidden_states=all_hidden_states,
        attentions=all_self_attns, cross_attentions=all_cross_attentions)


# ---- L_GSF ---------------------------------------------------------------------------------------------------

def b5_gsf_loss(lm_logits: torch.Tensor, labels: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """The B5 training loss, term for term (``modeling_HyenaTransgenic.py:985-1008``): token-weighted CE plus
    0.05 x the digit:structure ratio penalty. ``tests/test_b6_model.py`` pins it bitwise against the B5
    forward. B6 adds no auxiliary term (A50.2: loss is L_GSF only)."""
    weight = torch.ones(vocab_size, device=lm_logits.device)
    weight[2] = 5.0
    weight[17] = 4.0
    weight[21] = 3.0
    weight[272] = 3.0
    weight[22:272] = 2.0
    loss = nn.CrossEntropyLoss(weight=weight, ignore_index=-100)(lm_logits.view(-1, vocab_size), labels.view(-1))
    with torch.no_grad():
        pred_ids = lm_logits.argmax(dim=-1)
        valid_mask = labels.ne(-100)
        is_digit = (pred_ids >= 4) & (pred_ids <= 13)
        is_struct = ((pred_ids >= 22) & (pred_ids <= 271)) | (pred_ids == 21) | (pred_ids == 17) | (pred_ids == 2)
        digit_count = (is_digit & valid_mask).float().sum(dim=-1)
        struct_count = (is_struct & valid_mask).float().sum(dim=-1)
    ratio_penalty = F.relu(digit_count / (struct_count + 1.0) - 8.0).mean()
    return loss + 0.05 * ratio_penalty


# ---- backbone and LM model ---------------------------------------------------------------------------------

def _adopt_subclass(module: nn.Module, cls: type) -> nn.Module:
    """Turn an instance of ``cls``'s direct base into ``cls`` without re-initialising anything."""
    base = cls.__bases__[0]
    if type(module) is not base:
        raise TypeError(f"expected exactly {base.__name__}, got {type(module).__name__}")
    module.__class__ = cls
    return module


class B6TransgenicModel(_b5.transgenicModel):
    """B5 backbone plus the chain gather right after downsampling (plan 9)."""

    def downsample_states(self, encoder_states: torch.Tensor) -> torch.Tensor:
        """Positional injection + 6x downsampling, exactly as ``transgenicModel.forward`` (:746-749)."""
        injected = self.EncoderOutputPositionalEmbedding(encoder_states)
        return self.downsample(injected.permute(0, 2, 1)).permute(0, 2, 1)

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None,
                head_mask=None, decoder_head_mask=None, cross_attn_head_mask=None, encoder_outputs=None,
                global_attention_mask=None, past_key_values=None, inputs_embeds=None, decoder_inputs_embeds=None,
                use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None,
                evidence_tracks=None, chain_idx=None, chain_feat=None, chain_mask=None, chain_memory=None,
                chain_key_mask=None):
        gathered = chain_idx is not None or chain_feat is not None or chain_mask is not None
        precomputed = chain_memory is not None or chain_key_mask is not None
        if not gathered and not precomputed:
            return super().forward(input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids,
                                   decoder_attention_mask=decoder_attention_mask, head_mask=head_mask,
                                   decoder_head_mask=decoder_head_mask, cross_attn_head_mask=cross_attn_head_mask,
                                   encoder_outputs=encoder_outputs, global_attention_mask=global_attention_mask,
                                   past_key_values=past_key_values, inputs_embeds=inputs_embeds,
                                   decoder_inputs_embeds=decoder_inputs_embeds, use_cache=use_cache,
                                   output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                                   return_dict=return_dict, evidence_tracks=evidence_tracks)
        if gathered and precomputed:
            raise ValueError("pass either chain_idx/chain_feat/chain_mask or chain_memory/chain_key_mask, not both")
        if precomputed and (chain_memory is None or chain_key_mask is None):
            raise ValueError("chain_memory and chain_key_mask go together")
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states if output_hidden_states is not None
                                else self.config.output_hidden_states)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = _b5.shift_tokens_right(input_ids, self.config.pad_token_id,
                                                       self.config.decoder_start_token_id)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_ids, evidence_tracks=evidence_tracks)
            encoder_outputs.attention_mask = attention_mask
        else:
            if not isinstance(encoder_outputs, torch.Tensor):
                encoder_outputs = encoder_outputs.last_hidden_state
            encoder_outputs = _b5.HyenaModelOutput(last_hidden_state=encoder_outputs, attention_mask=attention_mask)
        downsampled = self.downsample_states(encoder_outputs.last_hidden_state)
        cross_mask = torch.ones(downsampled.shape[0:2], device=downsampled.device)
        if gathered:
            if chain_idx is None or chain_feat is None or chain_mask is None:
                raise ValueError("chain_idx, chain_feat and chain_mask go together")
            check_chain_shapes(chain_idx, chain_feat, chain_mask, downsampled.shape[0])
            chain_memory = self.chain_memory(downsampled, chain_idx, chain_feat)
            chain_key_mask = chain_mask
        elif tuple(chain_key_mask.shape) != (downsampled.shape[0], CHAIN_ELEMENTS) or chain_key_mask.dtype != torch.bool:
            raise ValueError(f"chain_key_mask must be bool[{downsampled.shape[0]},{CHAIN_ELEMENTS}]")
        if past_key_values is not None and isinstance(past_key_values, tuple):
            past_key_values = _b5.LegacyCacheWrapper(past_key_values)
        decoder_outputs = b6_decoder_forward(
            self.decoder, input_ids=decoder_input_ids, attention_mask=decoder_attention_mask,
            encoder_hidden_states=downsampled, encoder_attention_mask=cross_mask, head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask, past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds, use_cache=use_cache, output_attentions=output_attentions,
            output_hidden_states=output_hidden_states, return_dict=True, chain_memory=chain_memory,
            chain_key_mask=chain_key_mask)
        if decoder_outputs.past_key_values is not None and hasattr(decoder_outputs.past_key_values, "to_legacy_cache"):
            decoder_outputs.past_key_values = decoder_outputs.past_key_values.to_legacy_cache()
        if not return_dict:
            return decoder_outputs.to_tuple() + encoder_outputs.to_tuple()
        return _b5.LEDSeq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state, past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states, decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.encoder_hidden_states,
            encoder_attentions=encoder_outputs.attention_mask, encoder_global_attentions=None,
            segmentation_logits=encoder_outputs.segmentation_logits)


class B6TransgenicForConditionalGeneration(_b5.transgenicForConditionalGeneration):
    """B6 model: B5 LM model + tracks (K = 31) + chain memory and 12 gated chain blocks."""

    def __init__(self, config, unlink=False):
        validate_b6_config(config)
        super().__init__(config, unlink=unlink)          # B5 structure, post_init, initialize_weights
        backbone = _adopt_subclass(self.transgenic, B6TransgenicModel)
        chain_dim, heads = int(config.b6_chain_d_model), int(config.b6_chain_heads)
        backbone.add_module("chain_memory", ChainMemory(self.decoder_dim, chain_dim))
        for layer in backbone.decoder.layers:
            _adopt_subclass(layer, B6DecoderLayer)
            layer.add_module("chain_attn", ChainCrossAttention(
                self.decoder_dim, chain_dim, heads, float(config.b6_chain_dropout),
                float(getattr(config, "b6_chain_attn_dropout", config.b6_chain_dropout))))
        self._init_chain_modules()

    # -- initialisation --
    def _init_chain_modules(self) -> None:
        """New modules only, after the B5 initialisation: memory as HF _init_weights, chain attention Linears
        xavier-uniform like ``initialize_weights`` does for every decoder Linear, then every alpha = 0."""
        memory = self.transgenic.chain_memory
        memory.apply(self._init_weights)
        nn.init.normal_(memory.w_count, mean=0.0, std=self.config.init_std)
        for layer in self.transgenic.decoder.layers:
            for m in layer.chain_attn.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)
        self.zero_chain_alphas()

    def initialize_weights(self):
        """B5 ``initialize_weights`` (:886), then alpha = 0 (plan 9: alpha is set after initialize_weights)."""
        super().initialize_weights()
        self.zero_chain_alphas()

    def zero_chain_alphas(self) -> None:
        with torch.no_grad():
            for layer in self.transgenic.decoder.layers:
                if hasattr(layer, "chain_attn"):
                    layer.chain_attn.alpha.zero_()

    def chain_alphas(self):
        return [float(layer.chain_attn.alpha.detach().float()) for layer in self.transgenic.decoder.layers]

    # -- forward --
    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None,
                head_mask=None, decoder_head_mask=None, cross_attn_head_mask=None, encoder_outputs=None,
                global_attention_mask=None, past_key_values=None, inputs_embeds=None, decoder_inputs_embeds=None,
                labels=None, transcript_counts=None, use_cache=None, output_attentions=None,
                output_hidden_states=None, return_dict=None, evidence_tracks=None, chain_idx=None, chain_feat=None,
                chain_mask=None, chain_memory=None, chain_key_mask=None):
        chain_kwargs = {k: v for k, v in (("chain_idx", chain_idx), ("chain_feat", chain_feat),
                                          ("chain_mask", chain_mask), ("chain_memory", chain_memory),
                                          ("chain_key_mask", chain_key_mask)) if v is not None}
        if not chain_kwargs:
            return super().forward(input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids,
                                   decoder_attention_mask=decoder_attention_mask, head_mask=head_mask,
                                   decoder_head_mask=decoder_head_mask, cross_attn_head_mask=cross_attn_head_mask,
                                   encoder_outputs=encoder_outputs, global_attention_mask=global_attention_mask,
                                   past_key_values=past_key_values, inputs_embeds=inputs_embeds,
                                   decoder_inputs_embeds=decoder_inputs_embeds, labels=labels,
                                   transcript_counts=transcript_counts, use_cache=use_cache,
                                   output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                                   return_dict=return_dict, evidence_tracks=evidence_tracks)
        if labels is not None:
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = _b5.shift_tokens_right(labels, self.config.pad_token_id,
                                                           self.config.decoder_start_token_id)
        outputs = self.transgenic(input_ids, evidence_tracks=evidence_tracks, attention_mask=attention_mask,
                                  decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask,
                                  encoder_outputs=encoder_outputs, global_attention_mask=global_attention_mask,
                                  head_mask=head_mask, decoder_head_mask=decoder_head_mask,
                                  cross_attn_head_mask=cross_attn_head_mask, past_key_values=past_key_values,
                                  inputs_embeds=inputs_embeds, decoder_inputs_embeds=decoder_inputs_embeds,
                                  use_cache=use_cache, output_attentions=output_attentions,
                                  output_hidden_states=output_hidden_states, return_dict=True, **chain_kwargs)
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias
        loss = None
        if labels is not None:
            loss = b5_gsf_loss(lm_logits, labels, self.config.vocab_size)
            if transcript_counts is not None:
                raise ValueError("B6 trains on L_GSF only (A50.2); transcript_counts is not accepted")
        return _b5.LEDSeq2SeqLMOutput(
            loss=loss, logits=lm_logits, past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states, decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions, encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states, encoder_attentions=outputs.encoder_attentions,
            encoder_global_attentions=outputs.encoder_global_attentions,
            segmentation_logits=outputs.segmentation_logits)

    # -- generation --
    @torch.no_grad()
    def build_chain_memory(self, input_ids, evidence_tracks, chain_idx, chain_feat, chain_mask):
        """Encoder states and chain memory computed once (plan 9, "Generation")."""
        encoder_outputs = self.get_encoder()(input_ids, evidence_tracks=evidence_tracks)
        states = self.transgenic.downsample_states(encoder_outputs.last_hidden_state)
        check_chain_shapes(chain_idx, chain_feat, chain_mask, states.shape[0])
        return encoder_outputs, self.transgenic.chain_memory(states, chain_idx, chain_feat)

    @torch.no_grad()
    def generate(self, inputs=None, evidence_tracks=None, chain_idx=None, chain_feat=None, chain_mask=None, **kwargs):
        """HF ``generate``; with chains, the encoder runs once here and v is passed as a model kwarg, which HF
        repeats per beam exactly like ``encoder_outputs``. Without chains this is the B5 path."""
        if chain_idx is None and chain_feat is None and chain_mask is None:
            return super().generate(inputs, evidence_tracks=evidence_tracks, **kwargs)
        if chain_idx is None or chain_feat is None or chain_mask is None:
            raise ValueError("chain_idx, chain_feat and chain_mask go together")
        if "encoder_outputs" in kwargs or "chain_memory" in kwargs:
            raise ValueError("generate() with chains computes encoder_outputs and chain_memory itself")
        input_ids = inputs if inputs is not None else kwargs.pop("input_ids", None)
        if input_ids is None:
            raise ValueError("generate() with chains needs the encoder input_ids")
        encoder_outputs, memory = self.build_chain_memory(input_ids, evidence_tracks, chain_idx, chain_feat,
                                                          chain_mask)
        return super().generate(input_ids, encoder_outputs=encoder_outputs, chain_memory=memory,
                                chain_key_mask=chain_mask, **kwargs)

    def prepare_inputs_for_generation(self, decoder_input_ids, past_key_values=None, chain_memory=None,
                                      chain_key_mask=None, **kwargs):
        prepared = super().prepare_inputs_for_generation(decoder_input_ids, past_key_values=past_key_values,
                                                         **kwargs)
        if chain_memory is not None:
            prepared.update(chain_memory=chain_memory, chain_key_mask=chain_key_mask)
        return prepared


def alpha_parameter_names(model: nn.Module):
    """Names of the per-layer gates (excluded from weight decay, recorded in runtime identity)."""
    return [name for name, _ in model.named_parameters() if name.endswith(ALPHA_PARAM_SUFFIX)]
