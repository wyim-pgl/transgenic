"""
TransGenic model architecture: HyenaDNA encoder + Longformer decoder.

This module implements the full seq2seq model for predicting gene structure
annotations (GFF format) from raw DNA sequences. The architecture:

  1. HyenaDNA Encoder: Pretrained DNA foundation model that processes raw
     nucleotide sequences using Hyena operators (long-range convolutions).
     Outputs per-nucleotide embeddings of dimension d_model.

  2. Sinusoidal Positional Embedding: Injects position information into
     encoder outputs before downsampling.

  3. Conv1d Downsampling (6x compression): Two-stage strided convolution
     with learnable relative positional biases. Reduces sequence length
     by ~6x while expanding hidden dimension from d_model to 2*d_model.
     This makes the sequence short enough for the Longformer decoder's
     O(n*w) attention to be tractable.

  4. Longformer Decoder: Sliding-window attention decoder that generates
     GFF annotation tokens autoregressively. Cross-attends to the
     downsampled encoder outputs.

  5. LM Head: Linear projection from decoder hidden states to GFF vocabulary
     logits, with optional weight tying to decoder input embeddings.

Model hierarchy:
  transgenicForConditionalGeneration   (top-level, includes LM head)
    └─ transgenicModel                 (encoder-decoder backbone)
        ├─ HyenaEncoder                (pretrained DNA encoder)
        ├─ SinusoidalPositionalEmbedding
        ├─ HyenaDownsampleWithRelPosBias (2-stage Conv1d compression)
        └─ LEDForConditionalGeneration.decoder (Longformer decoder)
"""

import sys, math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, PreTrainedModel, AutoModel, GenerationMixin
from transformers import LEDForConditionalGeneration  # Longformer Encoder-Decoder
from transformers.modeling_outputs import ModelOutput

from dataclasses import dataclass
from .configuration_transgenic import HyenaTransgenicConfig


# ═══════════════════════════════════════════════════════════════════════════
# Legacy Cache Compatibility
# ═══════════════════════════════════════════════════════════════════════════

class LegacyCacheWrapper:
	"""Wrapper to make legacy tuple-based KV cache compatible with newer transformers API.

	Newer versions of HuggingFace transformers expect cache objects with methods
	like get_seq_length(), while older code returns plain tuples of tensors.
	This wrapper bridges the two interfaces.
	"""
	def __init__(self, past_key_values):
		"""Initialize the cache wrapper around a tuple-based KV cache.

		Args:
			past_key_values: Tuple of per-layer (key, value) tensor pairs,
				or None if no cache exists.
		"""
		self._past = past_key_values

	def get_seq_length(self, layer_idx=0):
		"""Return the cached sequence length from the first layer's key tensor."""
		if self._past is None or len(self._past) == 0:
			return 0
		# Shape of each layer's cache: (key, value) where key is (batch, heads, seq_len, head_dim)
		return self._past[0][0].shape[2]

	def __iter__(self):
		"""Iterate over per-layer cache entries, yielding (key, value) tensor pairs."""
		return iter(self._past)

	def __getitem__(self, idx):
		"""Return the cached (key, value) tensor pair for a given layer index."""
		return self._past[idx]

	def __len__(self):
		"""Return the number of cached layers, or 0 if no cache exists."""
		return len(self._past) if self._past else 0

	def to_legacy_cache(self):
		"""Convert back to the original tuple format."""
		return self._past


# ═══════════════════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════════════════

def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
	"""Shift decoder input IDs one position to the right for teacher forcing.

	In seq2seq training, the decoder input at position i should be the
	label at position i-1 (shifted right). Position 0 gets the
	decoder_start_token_id. Any -100 values (ignored positions in labels)
	are replaced with pad_token_id.

	Args:
		input_ids: Label token IDs, shape (batch_size, seq_len).
		pad_token_id: Token ID to use for replacing -100 values.
		decoder_start_token_id: Token ID to prepend at position 0.

	Returns:
		Shifted token IDs, same shape as input_ids.
	"""
	shifted_input_ids = input_ids.new_zeros(input_ids.shape)
	shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()  # Shift right by 1
	shifted_input_ids[:, 0] = decoder_start_token_id       # Prepend start token

	if pad_token_id is None:
		raise ValueError("config.pad_token_id has to be defined.")
	# Replace -100 (CrossEntropyLoss ignore_index) with pad_token_id
	shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

	return shifted_input_ids


def init_weights(m):
	"""Initialize model weights with appropriate strategies per layer type.

	Applied via model.apply(init_weights) to initialize the downsampling
	convolution layers and decoder linear layers.
	"""
	if isinstance(m, nn.Conv1d):
		# Kaiming initialization for ReLU-activated convolutions
		nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
		if m.bias is not None:
			nn.init.constant_(m.bias, 0)

	elif isinstance(m, nn.ConvTranspose1d):
		# Xavier initialization for transposed convolutions (if used)
		nn.init.xavier_normal_(m.weight)
		if m.bias is not None:
			nn.init.constant_(m.bias, 0)

	elif isinstance(m, nn.BatchNorm1d):
		# BatchNorm: weight=1 (scale), bias=0 (shift) — identity transform initially
		nn.init.constant_(m.weight, 1)
		nn.init.constant_(m.bias, 0)

	elif isinstance(m, nn.Linear):
		# Xavier initialization for linear layers (good for sigmoid/tanh activations)
		nn.init.xavier_normal_(m.weight)
		if m.bias is not None:
			nn.init.constant_(m.bias, 0)


# ═══════════════════════════════════════════════════════════════════════════
# Output Dataclasses
# ═══════════════════════════════════════════════════════════════════════════
# These mirror HuggingFace's LED output classes with an added
# segmentation_logits field for auxiliary segmentation task support.

@dataclass
class LEDSeq2SeqLMOutput(ModelOutput):
	"""Output type for TransGenic conditional generation (includes loss + logits).

	Extends the standard LED output with segmentation_logits for optional
	auxiliary per-position segmentation predictions.
	"""

	loss: Optional[torch.FloatTensor] = None                          # Cross-entropy language modeling loss
	logits: torch.FloatTensor = None                                  # (batch, seq_len, vocab_size) prediction scores
	past_key_values: Optional[List[torch.FloatTensor]] = None         # Cached KV states for generation
	decoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
	decoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
	cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
	encoder_last_hidden_state: Optional[torch.FloatTensor] = None
	encoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
	encoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
	encoder_global_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
	segmentation_logits: Optional[Tuple[torch.FloatTensor, ...]] = None  # Auxiliary segmentation output


@dataclass
class LEDSeq2SeqModelOutput(ModelOutput):
	"""Output type for TransGenic backbone model (no loss computation).

	Used internally by transgenicModel.forward() before the LM head
	computes logits and loss.
	"""

	last_hidden_state: torch.FloatTensor = None
	past_key_values: Optional[List[torch.FloatTensor]] = None
	decoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
	decoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
	cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
	encoder_last_hidden_state: Optional[torch.FloatTensor] = None
	encoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
	encoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
	encoder_global_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
	segmentation_logits: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class HyenaModelOutput(ModelOutput):
	"""Output type for the HyenaDNA encoder wrapper."""
	last_hidden_state: torch.FloatTensor = None     # Per-nucleotide embeddings (batch, seq_len, d_model)
	attention_mask: torch.FloatTensor = None          # Passed through for downstream use
	encoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
	segmentation_logits: Optional[Tuple[torch.FloatTensor, ...]] = None
	segmentation_loss: Optional[Tuple[torch.FloatTensor, ...]] = None


# ═══════════════════════════════════════════════════════════════════════════
# Base Model Class
# ═══════════════════════════════════════════════════════════════════════════

class TransgenicPreTrainedModel(PreTrainedModel):
	"""Base class for all TransGenic models. Provides weight initialization
	and gradient checkpointing support.
	"""
	config_class = HyenaTransgenicConfig   # Links to our custom config
	base_model_prefix = "led"               # HF internal prefix for state dict keys
	supports_gradient_checkpointing = True  # Enable activation checkpointing support

	def _init_weights(self, module):
		"""Initialize weights for Linear and Embedding layers using normal distribution."""
		std = self.config.init_std  # Default: 0.02
		if isinstance(module, nn.Linear):
			module.weight.data.normal_(mean=0.0, std=std)
			if module.bias is not None:
				module.bias.data.zero_()
		elif isinstance(module, nn.Embedding):
			module.weight.data.normal_(mean=0.0, std=std)
			if module.padding_idx is not None:
				module.weight.data[module.padding_idx].zero_()  # Padding embedding stays at zero

	@property
	def dummy_inputs(self):
		"""Provide dummy inputs for model tracing and shape inference."""
		pad_token = self.config.pad_token_id
		input_ids = torch.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]], device=self.device)
		dummy_inputs = {
			"attention_mask": input_ids.ne(pad_token),
			"input_ids": input_ids,
		}
		return dummy_inputs


# ═══════════════════════════════════════════════════════════════════════════
# Downsampling Module
# ═══════════════════════════════════════════════════════════════════════════

class HyenaDownsampleWithRelPosBias(nn.Module):
	"""Two-stage strided convolution with learnable relative positional biases.

	Compresses the encoder output sequence by ~6x while expanding the
	hidden dimension from d_model to 2*d_model:

	  Stage 1: Conv1d(d_model → 1.5*d_model, kernel=6, stride=3) → ~3x compression
	  Stage 2: Conv1d(1.5*d_model → 2*d_model, kernel=2, stride=2) → ~2x compression

	Total compression: stride 3 * stride 2 = 6x sequence length reduction.

	Relative Positional Bias:
	  Instead of using bias terms in the convolution (which are position-independent),
	  we convolve a tensor of ones with learnable bias parameters. This creates
	  position-dependent bias values relative to the convolution kernel window,
	  preserving positional information through the downsampling.

	Example: Input (batch=1, d_model=1152, seq_len=6000)
	  → Stage 1: (1, 1728, 2000)
	  → Stage 2: (1, 2304, 1000)
	"""
	def __init__(self, in_channels):
		"""
		Args:
			in_channels: Input channel dimension (d_model from encoder).
		"""
		super(HyenaDownsampleWithRelPosBias, self).__init__()

		# Stage 1: 3x downsampling (kernel_size=6, stride=3, padding=2)
		self.conv1 = nn.Conv1d(
			in_channels,
			in_channels + (in_channels // 2),  # d_model → 1.5 * d_model
			kernel_size=6,
			stride=3,
			padding=2,
			bias=False)  # No bias — using relative positional bias instead

		# Learnable relative positional bias for stage 1 convolution
		# Shape: (out_channels, 1, kernel_size) — broadcast across input channels
		self.rel_bias1 = nn.Parameter(torch.zeros(self.conv1.out_channels, 1, 6))
		self.norm1 = nn.LayerNorm(in_channels + (in_channels // 2))  # Post-conv normalization

		# Stage 2: 2x downsampling (kernel_size=2, stride=2)
		self.conv2 = nn.Conv1d(
			in_channels + (in_channels // 2),
			in_channels * 2,               # 1.5 * d_model → 2 * d_model
			kernel_size=2,
			stride=2,
			bias=False)  # No bias — using relative positional bias

		# Learnable relative positional bias for stage 2 convolution
		self.rel_bias2 = nn.Parameter(torch.zeros(self.conv2.out_channels, 1, 2))
		self.norm2 = nn.LayerNorm(in_channels * 2)  # Post-conv normalization

		self.relu = nn.ReLU(inplace=True)  # In-place activation to save memory

	def forward(self, x):
		"""
		Args:
			x: Encoder output, shape (batch, channels, seq_len) — channels-first format.

		Returns:
			Downsampled tensor, shape (batch, 2*channels, seq_len//6).
		"""
		# ── Stage 1: 3x downsampling ──
		# Convolve the input sequence (without bias)
		conv1_out = F.conv1d(x, self.conv1.weight, bias=None, stride=self.conv1.stride, padding=self.conv1.padding)
		# Convolve ones with the positional bias to get position-dependent bias values
		ones1 = torch.ones(x.size(0), 1, x.size(2), device=x.device)
		bias_out1 = F.conv1d(ones1, self.rel_bias1, bias=None,
							stride=self.conv1.stride, padding=self.conv1.padding)
		out1 = conv1_out + bias_out1  # Add relative positional bias

		# LayerNorm operates on the last dimension, so transpose: (batch, C, L) → (batch, L, C)
		out1 = out1.transpose(1, 2)
		out1 = self.norm1(out1)
		out1 = out1.transpose(1, 2)   # Back to (batch, C, L) for Conv1d
		out1 = self.relu(out1)

		# ── Stage 2: 2x downsampling ──
		conv2_out = F.conv1d(out1, self.conv2.weight, bias=None, stride=self.conv2.stride, padding=self.conv2.padding)
		ones2 = torch.ones(out1.size(0), 1, out1.size(2), device=out1.device)
		bias_out2 = F.conv1d(ones2, self.rel_bias2, bias=None, stride=self.conv2.stride, padding=self.conv2.padding)
		out2 = conv2_out + bias_out2  # Add relative positional bias

		# Normalize and activate
		out2 = out2.transpose(1, 2)
		out2 = self.norm2(out2)
		out2 = out2.transpose(1, 2)
		out2 = self.relu(out2)
		return out2


# ═══════════════════════════════════════════════════════════════════════════
# Encoder
# ═══════════════════════════════════════════════════════════════════════════

class HyenaEncoder(nn.Module):
	"""Wrapper around HyenaDNA pretrained encoder.

	Instantiates a HyenaDNA model from HuggingFace config and runs
	it as the TransGenic encoder. The encoder processes raw DNA
	nucleotide sequences and outputs per-position embeddings.

	Note: HyenaDNA uses Hyena operators (long-range implicit convolutions)
	instead of self-attention, giving O(n log n) complexity for sequence
	length n — enabling processing of sequences up to 1M nucleotides.
	"""
	def __init__(self, config):
		"""Initialize the HyenaDNA encoder from a TransGenic config.

		Loads the HyenaDNA architecture and overrides its max sequence
		length, hidden dimension, and layer count from the TransGenic config.

		Args:
			config: HyenaTransgenicConfig with encoder_model, max_encoder_seqlen,
				d_model, and encoder_n_layer fields.
		"""
		super().__init__()

		# Load HyenaDNA configuration and override with TransGenic settings
		HyenaConfig = AutoConfig.from_pretrained(config.encoder_model, trust_remote_code=True)
		HyenaConfig.max_seq_len = config.max_encoder_seqlen  # Override max sequence length
		HyenaConfig.d_model = config.d_model                  # Override hidden dimension
		HyenaConfig.n_layer = config.encoder_n_layer           # Override layer count
		# Instantiate from config (not from_pretrained — weights are initialized fresh
		# and loaded separately or fine-tuned from scratch)
		self.hyena = AutoModel.from_config(HyenaConfig, trust_remote_code=True)

	def forward(self, input_ids, segLabels=None, *args, **kwargs):
		"""
		Args:
			input_ids: DNA nucleotide token IDs, shape (batch, seq_len).
			segLabels: Optional segmentation labels (unused, for API compatibility).

		Returns:
			HyenaModelOutput with last_hidden_state (batch, seq_len, d_model).
		"""
		output = self.hyena(input_ids)  # Run HyenaDNA forward pass

		return HyenaModelOutput(
			last_hidden_state=output.last_hidden_state,  # Per-nucleotide embeddings
			encoder_hidden_states=output.hidden_states    # Intermediate layer outputs (if requested)
		)


# ═══════════════════════════════════════════════════════════════════════════
# Positional Embedding
# ═══════════════════════════════════════════════════════════════════════════

class SinusoidalPositionalEmbedding(torch.nn.Module):
	"""Fixed sinusoidal positional encoding (not learned).

	Adds position-dependent sinusoidal signals to the encoder output
	before downsampling. This injects absolute position information
	that would otherwise be lost during strided convolution.

	Uses the standard Vaswani et al. (2017) formulation:
	  PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
	  PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

	Registered as a buffer so it moves with the model to GPU but is
	not updated during training (no gradients).
	"""
	def __init__(self, max_len, d_model):
		"""
		Args:
			max_len: Maximum sequence length to pre-compute embeddings for.
			d_model: Hidden dimension of the model.
		"""
		super().__init__()
		self.d_model = d_model

		# Pre-compute the full positional encoding matrix
		position = torch.arange(max_len).unsqueeze(1)  # Shape: (max_len, 1)
		div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))  # Frequency terms

		pe = torch.zeros(max_len, d_model)
		pe[:, 0::2] = torch.sin(position * div_term)  # Even indices: sin
		pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices: cos

		# Register as buffer: moves with model but not a parameter (no gradients)
		self.register_buffer("pe", pe)

	def forward(self, x):
		"""Add positional encoding to input tensor.

		Args:
			x: Input tensor, shape (batch_size, seq_len, d_model).

		Returns:
			x + positional encoding, same shape.
		"""
		return x + self.pe[:x.shape[1], :].unsqueeze(0)  # Slice to actual seq_len, broadcast over batch


# ═══════════════════════════════════════════════════════════════════════════
# TransGenic Backbone Model
# ═══════════════════════════════════════════════════════════════════════════

class transgenicModel(TransgenicPreTrainedModel):
	"""Encoder-decoder backbone (without LM head).

	Data flow:
	  1. input_ids → HyenaEncoder → hidden_states (batch, seq_len, d_model)
	  2. hidden_states → SinusoidalPE → position-aware hidden_states
	  3. hidden_states → Conv1d Downsample → compressed (batch, seq_len//6, 2*d_model)
	  4. compressed → Longformer Decoder → decoder_output (batch, label_len, 2*d_model)

	The decoder uses sliding-window attention (Longformer) with cross-attention
	to the downsampled encoder output. The 2x dimension expansion in downsampling
	is critical: it gives the decoder enough capacity to represent the compressed
	information.
	"""
	_tied_weights_keys = ["decoder_embed_tokens.weight", "decoder.embed_tokens.weight"]

	def __init__(self, config):
		"""Initialize the encoder-decoder backbone.

		Constructs the HyenaDNA encoder, sinusoidal positional embedding,
		Conv1d downsampler (6x compression), and Longformer sliding-window
		decoder. The decoder operates at 2*d_model hidden dimension to
		match the expanded downsampled encoder output.

		Args:
			config: HyenaTransgenicConfig instance with model hyperparameters.
		"""
		super().__init__(config)

		padding_idx, vocab_size = config.pad_token_id, config.vocab_size
		# Decoder token embeddings: vocab_size → 2*d_model (matches downsampled encoder output dim)
		self.decoder_embed_tokens = nn.Embedding(vocab_size, config.d_model * 2, padding_idx)

		# HyenaDNA encoder (pretrained DNA foundation model)
		self.encoder = HyenaEncoder(config)

		# Sinusoidal positional embeddings for encoder output
		# Injects position information before downsampling destroys it
		self.EncoderOutputPositionalEmbedding = SinusoidalPositionalEmbedding(config.max_encoder_seqlen, config.d_model)

		# 6x downsampling: Conv1d compression with relative positional bias
		# Reduces seq_len by 6x while expanding d_model → 2*d_model
		self.downsample = HyenaDownsampleWithRelPosBias(config.d_model)

		# Longformer decoder: initialize from LED config with doubled d_model
		# We temporarily modify config.d_model to 2x for decoder initialization,
		# then restore it (to keep encoder config correct)
		config.d_model = config.d_model * 2
		self.decoder = LEDForConditionalGeneration(config).led.decoder  # Extract just the decoder
		self.decoder.embed_tokens = self.decoder_embed_tokens            # Share embeddings
		config.d_model = config.d_model // 2  # Restore original d_model

		# HuggingFace post-initialization: applies _init_weights to all modules
		self.post_init()

	def get_input_embeddings(self):
		return self.decoder_embed_tokens

	def set_input_embeddings(self, value):
		self.decoder_embed_tokens = value
		self.decoder.embed_tokens = self.decoder_embed_tokens  # Keep in sync

	def get_encoder(self):
		return self.encoder

	def get_decoder(self):
		return self.decoder

	def forward(
		self,
		input_ids: Optional[torch.LongTensor] = None,
		attention_mask: Optional[torch.Tensor] = None,
		decoder_input_ids: Optional[torch.LongTensor] = None,
		decoder_attention_mask: Optional[torch.LongTensor] = None,
		head_mask: Optional[torch.Tensor] = None,
		decoder_head_mask: Optional[torch.Tensor] = None,
		cross_attn_head_mask: Optional[torch.Tensor] = None,
		encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
		global_attention_mask: Optional[torch.FloatTensor] = None,
		past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
		inputs_embeds: Optional[torch.FloatTensor] = None,
		decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
		use_cache: Optional[bool] = None,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
	) -> Union[Tuple[torch.Tensor], LEDSeq2SeqModelOutput]:
		"""Full encoder-decoder forward pass.

		If encoder_outputs is None, runs the encoder on input_ids. Otherwise,
		uses the provided encoder outputs (for generation, where the encoder
		only needs to run once).

		Args:
			input_ids: DNA nucleotide token IDs for the encoder.
			attention_mask: Encoder attention mask.
			decoder_input_ids: GFF token IDs for the decoder (teacher forcing).
			encoder_outputs: Pre-computed encoder outputs (for generation).
			past_key_values: Cached KV states for incremental generation.
			... (other args follow HuggingFace convention)

		Returns:
			LEDSeq2SeqModelOutput with decoder hidden states and encoder outputs.
		"""
		# Resolve optional flags from config defaults
		output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
		output_hidden_states = (
			output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
		)
		use_cache = use_cache if use_cache is not None else self.config.use_cache
		return_dict = return_dict if return_dict is not None else self.config.use_return_dict

		# Auto-generate decoder_input_ids from labels via right-shift
		if decoder_input_ids is None and decoder_inputs_embeds is None:
			decoder_input_ids = shift_tokens_right(
				input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
			)

		# ── Encoder ──
		if encoder_outputs is None:
			# Fresh encoding: run HyenaDNA on DNA input
			encoder_outputs = self.encoder(input_ids)
			encoder_outputs.attention_mask = attention_mask
		else:
			# Pre-computed encoder outputs (during generation)
			encoder_outputs = HyenaModelOutput(
				last_hidden_state=encoder_outputs,
				attention_mask=attention_mask
				)

		# ── Positional Embedding + Downsampling ──
		# Inject sinusoidal positional encoding before compression
		injected = self.EncoderOutputPositionalEmbedding(encoder_outputs.last_hidden_state)

		# Compress: (batch, seq_len, d_model) → permute → conv1d → permute → (batch, seq_len//6, 2*d_model)
		downsampled = self.downsample(injected.permute(0, 2, 1)).permute(0, 2, 1)
		# Create new attention mask for the downsampled sequence (all ones — no padding after compression)
		attention_mask = torch.ones(downsampled.shape[0:2]).to(downsampled.device)

		# Wrap legacy tuple KV cache for newer transformers compatibility
		if past_key_values is not None and isinstance(past_key_values, tuple):
			past_key_values = LegacyCacheWrapper(past_key_values)

		# ── Decoder ──
		# Longformer decoder with cross-attention to downsampled encoder output
		decoder_outputs = self.decoder(
			input_ids=decoder_input_ids,
			attention_mask=decoder_attention_mask,
			encoder_hidden_states=downsampled,       # Cross-attention keys/values
			encoder_attention_mask=attention_mask,     # Mask for cross-attention
			global_attention_mask=global_attention_mask,
			head_mask=decoder_head_mask,
			cross_attn_head_mask=cross_attn_head_mask,
			past_key_values=past_key_values,
			inputs_embeds=decoder_inputs_embeds,
			use_cache=use_cache,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)

		# Convert cache back to legacy tuple format for compatibility
		if decoder_outputs.past_key_values is not None:
			if hasattr(decoder_outputs.past_key_values, 'to_legacy_cache'):
				decoder_outputs.past_key_values = decoder_outputs.past_key_values.to_legacy_cache()

		if not return_dict:
			return decoder_outputs + encoder_outputs

		return LEDSeq2SeqModelOutput(
			last_hidden_state=decoder_outputs.last_hidden_state,
			past_key_values=decoder_outputs.past_key_values,
			decoder_hidden_states=decoder_outputs.hidden_states,
			decoder_attentions=decoder_outputs.attentions,
			cross_attentions=decoder_outputs.cross_attentions,
			encoder_last_hidden_state=encoder_outputs.last_hidden_state,
			encoder_hidden_states=encoder_outputs.encoder_hidden_states,
			encoder_attentions=encoder_outputs.attention_mask,
			encoder_global_attentions=None,
			segmentation_logits=encoder_outputs.segmentation_logits,
		)


# ═══════════════════════════════════════════════════════════════════════════
# Full Model with LM Head
# ═══════════════════════════════════════════════════════════════════════════

class transgenicForConditionalGeneration(TransgenicPreTrainedModel, GenerationMixin):
	"""TransGenic model with language modeling head for conditional generation.

	This is the top-level model class used for training and inference.
	It wraps the transgenicModel backbone and adds:
	  - A linear LM head that projects decoder hidden states to vocabulary logits
	  - A final_logits_bias buffer (for compatibility with LED architecture)
	  - Cross-entropy loss computation when labels are provided
	  - HuggingFace GenerationMixin for beam search / greedy generation

	Weight tying: By default, the decoder input embeddings and the LM head
	share the same weight matrix (reduces parameters by ~vocab_size * d_model).
	Set unlink=True to use separate weights.
	"""
	base_model_prefix = "transgenic"
	_keys_to_ignore_on_load_missing = ["final_logits_bias"]
	_tied_weights_keys = ["transgenic.decoder_embed_tokens.weight", "lm_head.weight"]

	def __init__(self, config, unlink=False):
		"""
		Args:
			config: HyenaTransgenicConfig instance.
			unlink: If True, do not tie decoder embedding and LM head weights.
		"""
		if not unlink:
			_tied_weights_keys = []  # Clear tied keys to prevent weight sharing
		super().__init__(config)

		# Backbone encoder-decoder model
		self.transgenic = transgenicModel(config)
		# Logits bias: (1, vocab_size) — added to logits for fine-grained output adjustment
		self.register_buffer("final_logits_bias", torch.zeros((1, self.transgenic.decoder_embed_tokens.num_embeddings)))
		# LM head: projects from 2*d_model to vocabulary size
		self.lm_head = nn.Linear(config.d_model * 2, self.transgenic.decoder_embed_tokens.num_embeddings, bias=False)

		# Initialize weights and apply post-processing (HuggingFace convention)
		self.post_init()
		self.initialize_weights()  # Xavier init for decoder linear layers

	def get_encoder(self):
		return self.transgenic.get_encoder()

	def get_decoder(self):
		return self.transgenic.get_decoder()

	def resize_token_embeddings(self, new_num_tokens: int, pad_to_multiple_of: Optional[int] = None) -> nn.Embedding:
		"""Resize token embeddings and update logits bias to match."""
		new_embeddings = super().resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
		self._resize_final_logits_bias(new_embeddings.weight.shape[0])
		return new_embeddings

	def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
		"""Resize the final_logits_bias buffer to match new vocabulary size."""
		old_num_tokens = self.final_logits_bias.shape[-1]
		if new_num_tokens <= old_num_tokens:
			new_bias = self.final_logits_bias[:, :new_num_tokens]  # Truncate
		else:
			# Extend with zeros
			extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
			new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
		self.register_buffer("final_logits_bias", new_bias)

	def get_output_embeddings(self):
		return self.lm_head

	def set_output_embeddings(self, new_embeddings):
		self.lm_head = new_embeddings

	def initialize_weights(self):
		"""Initialize all decoder Linear layers with Xavier uniform distribution.

		This is applied on top of the HuggingFace default initialization to
		ensure the decoder starts with well-scaled weights for training stability.
		"""
		for m in self.transgenic.decoder.modules():
			if isinstance(m, nn.Linear):
				nn.init.xavier_uniform_(m.weight)
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)

	def forward(
		self,
		input_ids: Optional[torch.LongTensor] = None,
		attention_mask: Optional[torch.Tensor] = None,
		decoder_input_ids: Optional[torch.LongTensor] = None,
		decoder_attention_mask: Optional[torch.LongTensor] = None,
		head_mask: Optional[torch.Tensor] = None,
		decoder_head_mask: Optional[torch.Tensor] = None,
		cross_attn_head_mask: Optional[torch.Tensor] = None,
		encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
		global_attention_mask: Optional[torch.FloatTensor] = None,
		past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
		inputs_embeds: Optional[torch.FloatTensor] = None,
		decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
		labels: Optional[torch.LongTensor] = None,
		use_cache: Optional[bool] = None,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
	) -> Union[Tuple[torch.Tensor], LEDSeq2SeqLMOutput]:
		"""Full forward pass with optional loss computation.

		When labels are provided:
		  1. Automatically generates decoder_input_ids via right-shift of labels
		  2. Disables KV caching (not compatible with teacher forcing)
		  3. Computes cross-entropy loss between logits and labels

		When labels are None (generation mode):
		  Returns logits only, with optional KV cache for incremental decoding.

		Args:
			input_ids: DNA nucleotide token IDs, shape (batch, dna_seq_len).
			labels: Target GFF token IDs, shape (batch, gff_seq_len).
			... (other args follow HuggingFace convention)

		Returns:
			LEDSeq2SeqLMOutput with loss, logits, and all intermediate states.
		"""
		return_dict = return_dict if return_dict is not None else self.config.use_return_dict

		if labels is not None:
			if use_cache:
				print("The `use_cache` argument is changed to `False` since `labels` is provided.", file=sys.stderr)
			use_cache = False  # KV caching incompatible with teacher forcing
			if decoder_input_ids is None and decoder_inputs_embeds is None:
				# Generate decoder inputs by shifting labels right
				decoder_input_ids = shift_tokens_right(
					labels, self.config.pad_token_id, self.config.decoder_start_token_id
				)

		# Run the encoder-decoder backbone
		outputs = self.transgenic(
			input_ids,
			attention_mask=attention_mask,
			decoder_input_ids=decoder_input_ids,
			decoder_attention_mask=decoder_attention_mask,
			encoder_outputs=encoder_outputs,
			global_attention_mask=global_attention_mask,
			head_mask=head_mask,
			decoder_head_mask=decoder_head_mask,
			cross_attn_head_mask=cross_attn_head_mask,
			past_key_values=past_key_values,
			inputs_embeds=inputs_embeds,
			decoder_inputs_embeds=decoder_inputs_embeds,
			use_cache=use_cache,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)

		# Project decoder hidden states to vocabulary logits and add bias
		lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

		if not return_dict:
			output = (lm_logits,) + outputs[1:]
			return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

		# Compute cross-entropy loss when labels are provided
		masked_lm_loss = None
		if labels is not None:
			loss_fct = nn.CrossEntropyLoss()  # Ignores pad tokens (index 0) by default
			# Flatten logits and labels for loss computation
			masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

		return LEDSeq2SeqLMOutput(
			loss=masked_lm_loss,
			logits=lm_logits,
			past_key_values=outputs.past_key_values,
			decoder_hidden_states=outputs.decoder_hidden_states,
			decoder_attentions=outputs.decoder_attentions,
			cross_attentions=outputs.cross_attentions,
			encoder_last_hidden_state=outputs.encoder_last_hidden_state,
			encoder_hidden_states=outputs.encoder_hidden_states,
			encoder_attentions=outputs.encoder_attentions,
			encoder_global_attentions=outputs.encoder_global_attentions,
			segmentation_logits=outputs.segmentation_logits,
		)

	def prepare_inputs_for_generation(
		self,
		decoder_input_ids,
		past_key_values=None,
		attention_mask=None,
		global_attention_mask=None,
		head_mask=None,
		decoder_head_mask=None,
		cross_attn_head_mask=None,
		use_cache=None,
		encoder_outputs=None,
		**kwargs,
	):
		"""Prepare model inputs for autoregressive generation.

		During generation, after the first step, we only need the last
		decoder token (the rest is cached in past_key_values). The encoder
		outputs are reused from the first step.

		Returns:
			Dict of model inputs for the next generation step.
		"""
		# When using KV cache, only pass the last generated token
		if past_key_values is not None:
			decoder_input_ids = decoder_input_ids[:, -1:]

		return {
			"input_ids": None,  # Not needed — encoder_outputs already computed
			"encoder_outputs": encoder_outputs.last_hidden_state,
			"past_key_values": past_key_values,
			"decoder_input_ids": decoder_input_ids,
			"attention_mask": attention_mask,
			"global_attention_mask": global_attention_mask,
			"head_mask": head_mask,
			"decoder_head_mask": decoder_head_mask,
			"cross_attn_head_mask": cross_attn_head_mask,
			"use_cache": use_cache,
		}

	def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
		"""Create decoder input IDs from labels via right-shift (used by HF Trainer)."""
		return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

	@staticmethod
	def _reorder_cache(past_key_values, beam_idx):
		"""Reorder KV cache for beam search.

		When beam search selects/reorders beams, the cached key-value states
		must be reindexed to match the new beam ordering. Cross-attention
		caches (indices 2+) are not reordered since they depend on the
		(fixed) encoder output, not the varying decoder beams.
		"""
		reordered_past = ()
		for layer_past in past_key_values:
			# Reorder self-attention caches (first 2 tensors per layer)
			# Keep cross-attention caches unchanged (last tensors per layer)
			reordered_past += (
				tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past[:2])
				+ layer_past[2:],
			)
		return reordered_past
