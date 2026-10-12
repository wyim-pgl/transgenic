"""
TransGenic Model: HyenaDNA Encoder + Longformer Decoder for genomic annotation.

This is the primary model file for the TransGenic architecture, a sequence-to-sequence
model that reads raw DNA nucleotide sequences and generates structured text annotations
in GFF/GSF format. The architecture combines three major components:

    1. HyenaDNA Encoder: Processes full-length DNA sequences (up to ~49k nucleotides)
       using sub-quadratic Hyena operators, which replace standard self-attention with
       long convolutions and element-wise gating. This enables efficient processing of
       very long genomic sequences that would be intractable with quadratic attention.

    2. Conv1d Downsampling with Relative Positional Bias: A two-stage convolutional
       compression module that reduces encoder output length by 6x while doubling the
       hidden dimension (d_model -> 2*d_model). This bridges the gap between the long
       encoder sequences and the decoder's Longformer attention budget. A learnable
       relative positional bias is added at each stage to preserve positional information
       through the strided convolutions.

    3. Longformer Decoder: A Transformer decoder with sliding-window local attention
       (from the LED / Longformer-Encoder-Decoder architecture) that autoregressively
       generates GFF annotation tokens. Cross-attention to the compressed encoder output
       allows the decoder to attend to relevant genomic regions.

Architecture Diagram:
    DNA tokens -> [HyenaDNA Encoder] -> [Sinusoidal PE] -> [Conv1d Downsample 6x]
                                                                    |
    GFF tokens -> [Longformer Decoder] <--- cross-attention --- [compressed states]
                         |
                    [LM Head] -> GFF token logits

Note: The filename contains a typo ("Heyna" instead of "Hyena") but this is the
production model file used in published results. Do not rename without updating all
references.

Classes defined here:
    - LEDSeq2SeqLMOutput: Dataclass for language model outputs (from LED).
    - LEDSeq2SeqModelOutput: Dataclass for base model outputs (from LED).
    - HyenaModelOutput: Dataclass for HyenaDNA encoder outputs.
    - TransgenicPreTrainedModel: Abstract base with weight initialization logic.
    - HyenaDownsampleWithRelPosBias: Two-stage Conv1d compression module.
    - HyenaEncoder: Wrapper around HyenaDNA AutoModel.
    - SinusoidalPositionalEmbedding: Standard sinusoidal positional encoding.
    - transgenicModel: Full encoder-decoder model (no LM head).
    - transgenicForConditionalGeneration: Adds LM head for token prediction and generation.
"""

import sys, math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, PreTrainedModel, AutoModel, GenerationMixin
from transformers import LEDForConditionalGeneration
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass
from .configuration_transgenic import HyenaTransgenicConfig


# =============================================================================
# Utility Functions
# =============================================================================


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
	"""
	Shift input ids one token to the right to create decoder input from labels.

	This is the standard teacher-forcing preparation for seq2seq models: the decoder
	input at position i should be the label at position i-1. Position 0 is filled
	with the decoder_start_token_id (typically </s> = 2).

	Any positions with value -100 (the PyTorch CrossEntropyLoss ignore index, used to
	mask out tokens that should not contribute to the loss) are replaced with
	pad_token_id so they become valid token IDs for the decoder embedding lookup.

	Args:
		input_ids: Label token IDs of shape (batch_size, sequence_length).
		pad_token_id: The padding token ID to replace -100 ignore markers with.
		decoder_start_token_id: The token ID to place at position 0 (start of decoding).

	Returns:
		shifted_input_ids: Right-shifted tensor suitable for decoder input.
	"""
	# Allocate a zero tensor with the same shape and device as input_ids
	shifted_input_ids = input_ids.new_zeros(input_ids.shape)
	# Copy all tokens shifted one position to the right
	shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
	# Insert the decoder start token at the beginning of every sequence
	shifted_input_ids[:, 0] = decoder_start_token_id

	if pad_token_id is None:
		raise ValueError("config.pad_token_id has to be defined.")
	# Replace possible -100 values in labels by `pad_token_id` so that the
	# decoder embedding layer receives valid token indices
	shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

	return shifted_input_ids

def init_weights(m):
	"""
	Initialize weights for a single module using type-appropriate strategies.

	This function is designed to be passed to nn.Module.apply() for recursive
	weight initialization across all submodules. Each layer type gets a
	different initialization scheme suited to its activation function and role:

	- Conv1d: Kaiming (He) normal initialization, optimized for layers followed
	  by ReLU activations. Preserves variance in the forward pass.
	- ConvTranspose1d: Xavier (Glorot) normal initialization, suitable for
	  transposed convolutions used in upsampling paths.
	- BatchNorm1d: Weight set to 1.0 (identity scaling) and bias to 0.0,
	  so normalization initially passes through unchanged.
	- Linear: Xavier normal initialization, suitable for layers with
	  symmetric activations or as a general default.

	All biases are initialized to zero for all layer types.

	Args:
		m: A single nn.Module instance to initialize.
	"""
	if isinstance(m, nn.Conv1d):
		# Kaiming Initialization for Conv1d layers
		nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
		if m.bias is not None:
			nn.init.constant_(m.bias, 0)

	elif isinstance(m, nn.ConvTranspose1d):
		# Bilinear Initialization for ConvTranspose1d
		nn.init.xavier_normal_(m.weight)
		if m.bias is not None:
			nn.init.constant_(m.bias, 0)

	elif isinstance(m, nn.BatchNorm1d):
		# Constant Initialization for BatchNorm layers
		nn.init.constant_(m.weight, 1)
		nn.init.constant_(m.bias, 0)

	elif isinstance(m, nn.Linear):
		# Xavier Initialization for Linear layers
		nn.init.xavier_normal_(m.weight)
		if m.bias is not None:
			nn.init.constant_(m.bias, 0)

# =============================================================================
# Output Dataclasses
# =============================================================================
# These dataclasses extend the HuggingFace ModelOutput to carry all intermediate
# and final tensors through the model pipeline. They are adapted from the LED
# (Longformer Encoder-Decoder) implementation, with an additional
# `segmentation_logits` field for optional auxiliary segmentation tasks.


# Adapted from transformers.models.led.modeling_led.py
# Extended with `segmentation_logits` for optional genomic segmentation output.
@dataclass
class LEDSeq2SeqLMOutput(ModelOutput):
	"""
	Output dataclass for the TransGenic conditional generation model (with LM head).

	Extends the standard LED seq2seq LM output with a `segmentation_logits` field
	that can carry auxiliary per-position classification logits (e.g., for predicting
	gene boundaries or functional regions alongside the main GFF generation task).

	This is returned by `transgenicForConditionalGeneration.forward()`.

	Args:
		loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
			Language modeling loss.
		logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
			Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
		past_key_values (`List[torch.FloatTensor]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
			List of `torch.FloatTensor` of length `config.n_layers`, with each tensor of shape `(2, batch_size,
			num_heads, sequence_length, embed_size_per_head)`).

			Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
			used (see `past_key_values` input) to speed up sequential decoding.
		decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
			Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
			shape `(batch_size, sequence_length, hidden_size)`.

			Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
		decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
			Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
			sequence_length)`.

			Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
			self-attention heads.
		cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
			Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
			sequence_length)`.

			Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
			weighted average in the cross-attention heads.
		encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
			Sequence of hidden-states at the output of the last layer of the encoder of the model.
		encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
			Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
			shape `(batch_size, sequence_length, hidden_size)`.

			Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
		encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
			Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
			sequence_length)`.

			Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
			self-attention heads.
		encoder_global_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
			Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`,
			where `x` is the number of tokens with global attention mask.

			Global attentions weights after the attention softmax, used to compute the weighted average in the
			self-attention heads. Those are the attention weights from every token with global attention to every token
			in the sequence.
		segmentation_logits (`torch.FloatTensor`, *optional*):
			Auxiliary per-position segmentation logits from the encoder, if a segmentation
			head is attached. Used for multi-task training with genomic boundary detection.
	"""

	loss: Optional[torch.FloatTensor] = None
	logits: torch.FloatTensor = None
	past_key_values: Optional[List[torch.FloatTensor]] = None
	decoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
	decoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
	cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
	encoder_last_hidden_state: Optional[torch.FloatTensor] = None
	encoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
	encoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
	encoder_global_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
	segmentation_logits: Optional[Tuple[torch.FloatTensor, ...]] = None

# Adapted from transformers.models.led.modeling_led.py
# Extended with `segmentation_logits` for optional genomic segmentation output.
@dataclass
class LEDSeq2SeqModelOutput(ModelOutput):
	"""
	Output dataclass for the base TransGenic encoder-decoder model (without LM head).

	This carries decoder hidden states, cached key/values for generation, and all
	intermediate attention tensors. Extends the standard LED model output with
	`segmentation_logits` for optional auxiliary segmentation tasks.

	This is returned by `transgenicModel.forward()`.

	Args:
		last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
			Sequence of hidden-states at the output of the last layer of the decoder of the model.

			If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
			hidden_size)` is output.
		past_key_values (`List[torch.FloatTensor]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
			List of `torch.FloatTensor` of length `config.n_layers`, with each tensor of shape `(2, batch_size,
			num_heads, sequence_length, embed_size_per_head)`).

			Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
			used (see `past_key_values` input) to speed up sequential decoding.
		decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
			Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
			shape `(batch_size, sequence_length, hidden_size)`.

			Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
		decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
			Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
			sequence_length)`.

			Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
			self-attention heads.
		cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
			Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
			sequence_length)`.

			Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
			weighted average in the cross-attention heads.
		encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
			Sequence of hidden-states at the output of the last layer of the encoder of the model.
		encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
			Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
			shape `(batch_size, sequence_length, hidden_size)`.

			Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
		encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
			Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
			sequence_length)`.

			Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
			self-attention heads.
		encoder_global_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
			Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`,
			where `x` is the number of tokens with global attention mask.

			Global attentions weights after the attention softmax, used to compute the weighted average in the
			self-attention heads. Those are the attention weights from every token with global attention to every token
			in the sequence.
		segmentation_logits (`torch.FloatTensor`, *optional*):
			Auxiliary per-position segmentation logits from the encoder.
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
	"""
	Output dataclass for the HyenaDNA encoder component.

	Carries the encoder's hidden states along with the attention mask (preserved for
	downstream use) and optional auxiliary outputs. This is the interface between the
	HyenaDNA encoder and the downsampling / decoder stages.

	Attributes:
		last_hidden_state: Final hidden states from the HyenaDNA encoder.
			Shape: (batch_size, seq_len, d_model).
		attention_mask: The input attention mask, passed through so downstream
			modules can access it without separate bookkeeping.
		encoder_hidden_states: Intermediate hidden states from all encoder layers,
			if requested via output_hidden_states=True.
		segmentation_logits: Optional per-position segmentation logits from an
			auxiliary classification head (e.g., for gene boundary prediction).
		segmentation_loss: Optional pre-computed segmentation loss, if segmentation
			labels were provided during training.
	"""
	last_hidden_state: torch.FloatTensor = None
	attention_mask: torch.FloatTensor = None
	encoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
	segmentation_logits: Optional[Tuple[torch.FloatTensor, ...]] = None
	segmentation_loss: Optional[Tuple[torch.FloatTensor, ...]] = None

# =============================================================================
# Base Pre-Trained Model Class
# =============================================================================


class TransgenicPreTrainedModel(PreTrainedModel):
	"""
	Abstract base class for all TransGenic model variants.

	Inherits from HuggingFace's PreTrainedModel to get standard functionality like
	from_pretrained(), save_pretrained(), and automatic weight initialization.
	Subclasses (transgenicModel, transgenicForConditionalGeneration) inherit this
	and add their specific architectures.

	Configuration:
		config_class: Uses HyenaTransgenicConfig for all hyperparameters.
		base_model_prefix: Set to "led" for compatibility with LED-based decoder
			weight loading and parameter naming conventions.
		supports_gradient_checkpointing: Enabled to allow trading compute for
			memory during training of long sequences.
	"""
	config_class = HyenaTransgenicConfig
	base_model_prefix = "led"
	supports_gradient_checkpointing = True

	def _init_weights(self, module):
		"""
		Initialize weights for Linear and Embedding modules using normal distribution.

		Called automatically by HuggingFace's PreTrainedModel.init_weights() during
		model construction (via post_init). Uses the standard deviation from
		config.init_std (default 0.02).

		- Linear layers: Normal(0, init_std) for weights, zero for bias.
		- Embedding layers: Normal(0, init_std) for weights, zero for padding index row.

		Args:
			module: A single nn.Module to initialize.
		"""
		std = self.config.init_std
		if isinstance(module, nn.Linear):
			module.weight.data.normal_(mean=0.0, std=std)
			if module.bias is not None:
				module.bias.data.zero_()
		elif isinstance(module, nn.Embedding):
			module.weight.data.normal_(mean=0.0, std=std)
			if module.padding_idx is not None:
				module.weight.data[module.padding_idx].zero_()

	@property
	def dummy_inputs(self):
		"""
		Provide dummy inputs for tracing / ONNX export and shape inference.

		Returns a dict with 'input_ids' (two short sequences with one padded)
		and a corresponding 'attention_mask' that masks out the pad token.
		"""
		pad_token = self.config.pad_token_id
		input_ids = torch.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]], device=self.device)
		dummy_inputs = {
			"attention_mask": input_ids.ne(pad_token),
			"input_ids": input_ids,
		}
		return dummy_inputs

# =============================================================================
# Downsampling Module
# =============================================================================


class HyenaDownsampleWithRelPosBias(nn.Module):
	"""
	Two-stage Conv1d downsampling with learnable relative positional bias.

	This module bridges the HyenaDNA encoder and the Longformer decoder by compressing
	the encoder's long sequence output into a shorter, wider representation that fits
	within the decoder's attention budget.

	Compression stages:
		Stage 1: Conv1d(d_model, d_model*1.5, kernel=6, stride=3, padding=2)
			- Compresses sequence length by ~3x
			- Expands channels from d_model to d_model * 1.5
			- Example: d_model=768 -> 1152 channels, length / 3

		Stage 2: Conv1d(d_model*1.5, d_model*2, kernel=2, stride=2, padding=0)
			- Compresses sequence length by another 2x
			- Expands channels from d_model * 1.5 to d_model * 2
			- Example: 1152 -> 1536 channels, length / 2

		Total effect: 6x length compression, 2x channel expansion.
		For d_model=768: (batch, 768, L) -> (batch, 1536, L/6)

	Relative Positional Bias mechanism:
		Instead of using a standard bias term in the convolution (which would be a
		single scalar per output channel, applied identically to every position),
		this module uses a learnable kernel-sized bias parameter. The bias is convolved
		with a tensor of ones to produce position-dependent bias values. This means
		each position within the receptive field of the kernel contributes a different
		learned bias, effectively encoding relative position information within each
		convolution window. This is particularly important because strided convolutions
		can destroy fine-grained positional information that the decoder needs for
		accurate genomic coordinate prediction.

	Each stage is followed by LayerNorm (applied in the channel dimension after
	transposing) and ReLU activation.

	Args:
		in_channels: Input channel dimension (d_model from the encoder).
	"""
	def __init__(self, in_channels):
		super(HyenaDownsampleWithRelPosBias, self).__init__()

		# --- Stage 1: 3x length compression ---
		# First downsampling convolution without bias.
		# The bias is handled separately via the relative positional bias mechanism.
		# in_channels -> in_channels * 1.5 (e.g., 768 -> 1152)
		# kernel=6, stride=3, padding=2 gives approximately 3x downsampling
		self.conv1 = nn.Conv1d(
			in_channels,
			in_channels + (in_channels // 2),
			kernel_size=6,
			stride=3,
			padding=2,
			bias=False)

		# Learnable relative positional bias for conv1.
		# Shape: (out_channels, 1, kernel_size) -- one bias value per position in the
		# kernel window, per output channel. The middle dimension is 1 so that when
		# convolved with a (batch, 1, length) tensor of ones, the bias broadcasts
		# across the batch and produces position-dependent offsets.
		# Initialized to zero so the bias starts as a no-op.
		self.rel_bias1 = nn.Parameter(torch.zeros(self.conv1.out_channels, 1, 6))
		# LayerNorm over the expanded channel dimension after stage 1
		self.norm1 = nn.LayerNorm(in_channels + (in_channels // 2))

		# --- Stage 2: 2x length compression ---
		# Second downsampling convolution without bias.
		# in_channels * 1.5 -> in_channels * 2 (e.g., 1152 -> 1536)
		# kernel=2, stride=2 gives exactly 2x downsampling
		self.conv2 = nn.Conv1d(
			in_channels + (in_channels // 2),
			in_channels * 2,
			kernel_size=2,
			stride=2,
			bias=False)

		# Learnable relative positional bias for conv2
		# Same mechanism as rel_bias1 but with kernel_size=2
		self.rel_bias2 = nn.Parameter(torch.zeros(self.conv2.out_channels, 1, 2))
		# LayerNorm over the final expanded channel dimension
		self.norm2 = nn.LayerNorm(in_channels * 2)

		# Activation function shared by both stages
		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):
		"""
		Apply two-stage downsampling with relative positional bias.

		The relative positional bias is computed by convolving a tensor of ones with
		the learned bias kernel. Because the input to this auxiliary convolution is
		all ones, the output at each position is a weighted sum of the bias kernel
		values within the receptive field -- effectively a position-dependent bias
		that encodes where each output element sits relative to the convolution window.

		Args:
			x: Input tensor of shape (batch, channels, length) where channels = d_model.

		Returns:
			Downsampled tensor of shape (batch, channels*2, length//6).
		"""
		# --- Stage 1 ---
		# x shape: (batch, channels, length)
		# Convolve the input with learned weights (no bias term in the conv itself)
		conv1_out = F.conv1d(x, self.conv1.weight, bias=None,stride=self.conv1.stride, padding=self.conv1.padding)
		# Create a tensor of ones with same batch size and length as input, single channel.
		# Convolving ones with the bias kernel produces position-dependent bias values.
		ones1 = torch.ones(x.size(0), 1, x.size(2), device=x.device)
		bias_out1 = F.conv1d(ones1, self.rel_bias1, bias=None,
							stride=self.conv1.stride, padding=self.conv1.padding)
		# Add the position-dependent bias to the convolution output
		out1 = conv1_out + bias_out1

		# Apply LayerNorm (requires channels-last format, so transpose around it)
		out1 = out1.transpose(1, 2)  # (batch, length', channels) for LayerNorm
		out1 = self.norm1(out1)
		out1 = out1.transpose(1, 2)  # (batch, channels, length') back to Conv1d format
		out1 = self.relu(out1)

		# --- Stage 2 ---
		# Same pattern: convolve input and ones separately, then add
		conv2_out = F.conv1d(out1, self.conv2.weight, bias=None,stride=self.conv2.stride, padding=self.conv2.padding)
		ones2 = torch.ones(out1.size(0), 1, out1.size(2), device=out1.device)
		bias_out2 = F.conv1d(ones2, self.rel_bias2, bias=None, stride=self.conv2.stride, padding=self.conv2.padding)
		out2 = conv2_out + bias_out2

		# Apply normalization and activation.
		out2 = out2.transpose(1, 2)  # (batch, length'', channels) for LayerNorm
		out2 = self.norm2(out2)
		out2 = out2.transpose(1, 2)  # (batch, channels, length'') back to Conv1d format
		out2 = self.relu(out2)
		return out2

# =============================================================================
# Encoder: HyenaDNA Wrapper
# =============================================================================


class HyenaEncoder(nn.Module):
	"""
	Wrapper around the HyenaDNA model that serves as the DNA sequence encoder.

	HyenaDNA uses Hyena operators -- a combination of long convolutions and
	element-wise gating -- to achieve sub-quadratic complexity in sequence length.
	This allows processing DNA sequences up to ~49k nucleotides without the
	quadratic memory cost of standard self-attention.

	The wrapper loads the HyenaDNA architecture configuration from a pretrained
	checkpoint identifier (e.g., "LongSafari/hyenadna-large-1m-seqlen-hf") but
	instantiates the model from config only (via AutoModel.from_config), meaning
	the encoder weights are randomly initialized and must be trained or loaded
	separately. Key config values (max_seq_len, d_model, n_layer) are overridden
	from the TransGenic config to allow customization.

	Args:
		config: HyenaTransgenicConfig containing encoder_model (HF model ID),
			max_encoder_seqlen, d_model, and encoder_n_layer.
	"""
	def __init__(self, config):
		super().__init__()

		# Load the HyenaDNA architecture config from the pretrained model identifier.
		# trust_remote_code=True is required because HyenaDNA uses custom modeling code
		# hosted on the HuggingFace Hub, not built into the transformers library.
		HyenaConfig = AutoConfig.from_pretrained(config.encoder_model, trust_remote_code=True)
		# Override key parameters from the TransGenic config to customize the encoder
		HyenaConfig.max_seq_len = config.max_encoder_seqlen  # Max DNA input length
		HyenaConfig.d_model = config.d_model                 # Hidden dimension
		HyenaConfig.n_layer = config.encoder_n_layer         # Number of Hyena layers
		# Instantiate the model from config (random weights, not pretrained weights)
		self.hyena = AutoModel.from_config(HyenaConfig, trust_remote_code=True)

	def forward(self, input_ids, segLabels = None, *args, **kwargs):
		"""
		Encode DNA token IDs into continuous hidden representations.

		Args:
			input_ids: DNA nucleotide token IDs of shape (batch_size, seq_len).
				Typically encoded as: A=0, C=1, G=2, T=3, plus special tokens.
			segLabels: Optional segmentation labels (unused in base encoder,
				reserved for subclasses with auxiliary segmentation heads).
			*args, **kwargs: Additional arguments (absorbed but not used).

		Returns:
			HyenaModelOutput with last_hidden_state of shape
			(batch_size, seq_len, d_model) and optional encoder_hidden_states.
		"""
		# Run the HyenaDNA forward pass on the input token IDs
		output = self.hyena(input_ids)

		# Wrap the output in our custom dataclass for consistent interface
		return HyenaModelOutput(
			last_hidden_state=output.last_hidden_state,
			encoder_hidden_states=output.hidden_states
		)

# =============================================================================
# Positional Encoding
# =============================================================================


class SinusoidalPositionalEmbedding(torch.nn.Module):
	"""
	Standard fixed sinusoidal positional encoding (Vaswani et al., 2017).

	Adds position-dependent sine and cosine signals to the input embeddings so that
	the model can distinguish token positions. Even dimensions use sine and odd
	dimensions use cosine, with frequencies that decrease geometrically from high
	frequency (period = 2*pi) to low frequency (period = 2*pi * 10000).

	In the TransGenic pipeline, this is applied to the HyenaDNA encoder output
	BEFORE downsampling, injecting absolute position information into the encoder
	representations. This complements the relative positional bias in the
	downsampling module: the sinusoidal encoding provides global position awareness,
	while the downsampling bias preserves local relative positions through the
	strided convolutions.

	The positional encoding is registered as a buffer (not a parameter), so it is
	not updated during training and is automatically moved to the correct device.

	Args:
		max_len: Maximum sequence length to pre-compute encodings for.
		d_model: Dimension of the embeddings (must match encoder output dimension).
	"""
	def __init__(self, max_len, d_model):
		super().__init__()
		self.d_model = d_model

		# Create position indices: (max_len, 1) for broadcasting with div_term
		position = torch.arange(max_len).unsqueeze(1)  # Shape: (max_len, 1)
		# Compute the geometric frequency decay term: exp(-2i/d * ln(10000))
		# This creates frequencies that span from high (dimension 0) to low (dimension d_model-1)
		div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))  # Shape: (d_model/2)

		# Compute sinusoidal values and fill the positional encoding matrix
		pe = torch.zeros(max_len, d_model)
		pe[:, 0::2] = torch.sin(position * div_term)  # Apply sine to even indices (0, 2, 4, ...)
		pe[:, 1::2] = torch.cos(position * div_term)  # Apply cosine to odd indices (1, 3, 5, ...)

		# Register as a buffer so it's not updated during training but is saved with model state
		self.register_buffer("pe", pe)

	def forward(self, x):
		"""
		Add positional encoding to the input tensor.

		The encoding is sliced to match the actual sequence length (which may be
		shorter than max_len) and broadcast across the batch dimension.

		Args:
			x: Input tensor of shape (batch_size, seq_len, d_model).

		Returns:
			Tensor of same shape with positional encoding added element-wise.
		"""
		return x + self.pe[:x.shape[1], :].unsqueeze(0)  # Shape: (batch_size, seq_len, d_model)

# =============================================================================
# Full Encoder-Decoder Model (without LM head)
# =============================================================================


class transgenicModel(TransgenicPreTrainedModel):
	"""
	TransGenic encoder-decoder model: HyenaDNA encoder -> downsample -> Longformer decoder.

	This is the core seq2seq model that transforms DNA nucleotide sequences into
	continuous hidden representations suitable for language modeling. It does NOT
	include the final linear projection (LM head) that maps to vocabulary logits;
	that is added by `transgenicForConditionalGeneration`.

	Architecture flow:
		1. DNA input_ids -> HyenaDNA encoder -> hidden states (batch, seq_len, d_model)
		2. Add sinusoidal positional encoding to encoder output
		3. Conv1d downsample: (batch, seq_len, d_model) -> (batch, seq_len/6, d_model*2)
		4. Create all-ones attention mask for downsampled sequence (no padding after compression)
		5. Longformer decoder with cross-attention to compressed encoder states

	The decoder operates in a doubled hidden dimension (d_model*2) because the
	downsampling module expands channels by 2x. The decoder embedding layer is
	also created with d_model*2 dimensions to match.

	Weight tying: The decoder input embeddings (decoder_embed_tokens) are shared
	with the decoder's internal embed_tokens to ensure consistent representations.

	Args:
		config: HyenaTransgenicConfig with all model hyperparameters.
	"""
	# Keys for weight tying between decoder embedding and internal decoder embed_tokens
	_tied_weights_keys = ["decoder_embed_tokens.weight", "decoder.embed_tokens.weight"]

	def __init__(self, config):
		super().__init__(config)

		padding_idx, vocab_size = config.pad_token_id, config.vocab_size
		# Decoder token embeddings: vocab_size -> d_model*2 dimensional space
		# The *2 is because the downsampled encoder output has d_model*2 channels,
		# and the decoder must operate in the same dimensional space
		self.decoder_embed_tokens = nn.Embedding(vocab_size, config.d_model*2, padding_idx)

		# Encoder model: HyenaDNA-based, processes raw DNA nucleotide token IDs
		self.encoder = HyenaEncoder(config)

		# Sinusoidal positional embeddings added to encoder output before downsampling.
		# This provides absolute position information to the encoder representations
		# before they are compressed by the strided convolutions.
		self.EncoderOutputPositionalEmbedding = SinusoidalPositionalEmbedding(config.max_encoder_seqlen, config.d_model)

		# Two-stage Conv1d compression: 6x length reduction, 2x channel expansion
		self.downsample = HyenaDownsampleWithRelPosBias(config.d_model)

		# Decoder Model: Longformer (LED) decoder with sliding-window attention.
		# Temporarily double d_model in config so the LED decoder is constructed with
		# the correct hidden dimension (d_model*2), then restore the original value.
		config.d_model = config.d_model * 2
		# Extract just the decoder portion from a full LEDForConditionalGeneration model
		self.decoder = LEDForConditionalGeneration(config).led.decoder
		# Share embedding weights: point the decoder's internal embeddings to ours
		self.decoder.embed_tokens = self.decoder_embed_tokens
		# Restore original d_model in config to avoid side effects on other components
		config.d_model = config.d_model//2

		# Initialize weights and apply final processing (calls _init_weights recursively)
		self.post_init()

	def get_input_embeddings(self):
		"""Return the decoder token embedding layer (used by HF for weight tying)."""
		return self.decoder_embed_tokens

	def set_input_embeddings(self, value):
		"""Set decoder embeddings and sync with internal decoder embed_tokens."""
		self.decoder_embed_tokens = value
		self.decoder.embed_tokens = self.decoder_embed_tokens

	def get_encoder(self):
		"""Return the HyenaDNA encoder module (used by HF generation utilities)."""
		return self.encoder

	def get_decoder(self):
		"""Return the Longformer decoder module (used by HF generation utilities)."""
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
		"""
		Forward pass through the full encoder-decoder pipeline.

		The encoder is only run when encoder_outputs is None (i.e., the first forward
		pass). During autoregressive generation, encoder_outputs are cached and passed
		in directly to avoid re-encoding the DNA sequence at every decoding step.

		Args:
			input_ids: DNA nucleotide token IDs, shape (batch_size, dna_seq_len).
			attention_mask: Mask for encoder input (1 = real token, 0 = padding).
				After downsampling, this is replaced with an all-ones mask.
			decoder_input_ids: GFF token IDs for decoder input, shape (batch_size, gff_seq_len).
				If None, automatically created by shifting labels right.
			decoder_attention_mask: Mask for decoder input tokens.
			head_mask: Per-layer head mask for encoder attention (unused for Hyena).
			decoder_head_mask: Per-layer head mask for decoder self-attention.
			cross_attn_head_mask: Per-layer head mask for decoder cross-attention.
			encoder_outputs: Pre-computed encoder hidden states (for generation caching).
				If provided, the encoder is skipped and these are used directly.
			global_attention_mask: Longformer global attention mask for the decoder.
			past_key_values: Cached decoder key/value states for fast autoregressive generation.
			inputs_embeds: Pre-computed encoder input embeddings (alternative to input_ids).
			decoder_inputs_embeds: Pre-computed decoder input embeddings.
			use_cache: Whether to return past_key_values for generation caching.
			output_attentions: Whether to return attention weight tensors.
			output_hidden_states: Whether to return hidden states from all layers.
			return_dict: Whether to return a ModelOutput dataclass or a plain tuple.

		Returns:
			LEDSeq2SeqModelOutput (if return_dict=True) or tuple of tensors.
		"""
		# Resolve optional arguments: use config defaults if not explicitly provided
		output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
		output_hidden_states = (
			output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
		)
		use_cache = use_cache if use_cache is not None else self.config.use_cache
		return_dict = return_dict if return_dict is not None else self.config.use_return_dict

		# If no decoder input is provided, create it by shifting input_ids right.
		# This follows the BART/LED convention for teacher forcing: the decoder sees
		# the target sequence shifted by one position, with decoder_start_token_id prepended.
		# Reference: https://github.com/huggingface/transformers/blob/ac3cb660cad283163f7c73cad511124e845ca388/src/transformers/models/bart/modeling_bart.py#L1153
		if decoder_input_ids is None and decoder_inputs_embeds is None:
			decoder_input_ids = shift_tokens_right(
				input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
			)
		# --- Encoder ---
		# Run the HyenaDNA encoder if we don't already have cached encoder outputs
		if encoder_outputs is None:
			encoder_outputs = self.encoder(input_ids)
			encoder_outputs.attention_mask = attention_mask
		else:
			# During generation, encoder_outputs is passed as a raw tensor (last_hidden_state).
			# Wrap it in our dataclass for consistent downstream access.
			encoder_outputs = HyenaModelOutput(
				last_hidden_state=encoder_outputs,
				attention_mask=attention_mask
				)

		# --- Positional Encoding + Downsampling ---
		# Add sinusoidal positional encoding to encoder output
		injected = self.EncoderOutputPositionalEmbedding(encoder_outputs.last_hidden_state)

		# Downsample: permute to (batch, channels, length) for Conv1d, then back.
		# After downsampling: seq_len is reduced by 6x, channels are expanded by 2x.
		downsampled = self.downsample(injected.permute(0,2,1)).permute(0,2,1)
		# After downsampling, create an all-ones attention mask for the compressed sequence.
		# No padding exists in the compressed space because the convolution processes all
		# positions uniformly (padding was handled at the input level, and the strided
		# convolution produces valid outputs for all positions).
		attention_mask = torch.ones(downsampled.shape[0:2]).to(downsampled.device)

		# --- Decoder ---
		# Run the Longformer decoder with cross-attention to compressed encoder states.
		# decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
		decoder_outputs = self.decoder(
			input_ids=decoder_input_ids,
			attention_mask=decoder_attention_mask,
			encoder_hidden_states=downsampled,
			encoder_attention_mask=attention_mask,
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

		# --- Return ---
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
			segmentation_logits = encoder_outputs.segmentation_logits,
		)

class transgenicForConditionalGeneration(TransgenicPreTrainedModel, GenerationMixin):
	base_model_prefix = "transgenic"
	_keys_to_ignore_on_load_missing = ["final_logits_bias"]
	_tied_weights_keys = ["transgenic.decoder_embed_tokens.weight", "lm_head.weight"]

	def __init__(self, config, unlink=False):
		if not unlink:
			_tied_weights_keys = []
		super().__init__(config)
		self.transgenic = transgenicModel(config)
		self.register_buffer("final_logits_bias", torch.zeros((1, self.transgenic.decoder_embed_tokens.num_embeddings)))
		self.lm_head = nn.Linear(config.d_model, self.transgenic.decoder_embed_tokens.num_embeddings, bias=False)

		# Initialize weights and apply final processing
		self.post_init()
		self.initialize_weights()

	def get_encoder(self):
		return self.transgenic.get_encoder()

	def get_decoder(self):
		return self.transgenic.get_decoder()

	def resize_token_embeddings(self, new_num_tokens: int, pad_to_multiple_of: Optional[int] = None) -> nn.Embedding:
		new_embeddings = super().resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
		self._resize_final_logits_bias(new_embeddings.weight.shape[0])
		return new_embeddings

	def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
		old_num_tokens = self.final_logits_bias.shape[-1]
		if new_num_tokens <= old_num_tokens:
			new_bias = self.final_logits_bias[:, :new_num_tokens]
		else:
			extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
			new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
		self.register_buffer("final_logits_bias", new_bias)

	def get_output_embeddings(self):
		return self.lm_head

	def set_output_embeddings(self, new_embeddings):
		self.lm_head = new_embeddings
	
	def initialize_weights(self):
		for m in self.transgenic.decoder.modules():
			if isinstance(m, nn.Linear):
				nn.init.xavier_uniform_(m.weight)
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)

	#@add_start_docstrings_to_model_forward(LED_INPUTS_DOCSTRING)
	#@replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
	#@add_end_docstrings(LED_GENERATION_EXAMPLE)
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
		r"""
		labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
			Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
			config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
			(masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

		Returns:

		Conditional generation example:

		```python
		>>> from transformers import AutoTokenizer, LEDForConditionalGeneration

		>>> tokenizer = AutoTokenizer.from_pretrained("allenai/led-base-16384")
		>>> TXT = "My friends are <mask> but they eat too many carbs."

		>>> model = LEDForConditionalGeneration.from_pretrained("allenai/led-base-16384")
		>>> input_ids = tokenizer([TXT], return_tensors="pt")["input_ids"]

		>>> prediction = model.generate(input_ids)[0]
		>>> print(tokenizer.decode(prediction, skip_special_tokens=True))
		```"""
		return_dict = return_dict if return_dict is not None else self.config.use_return_dict

		if labels is not None:
			if use_cache:
				print("The `use_cache` argument is changed to `False` since `labels` is provided.", file=sys.stderr)
			use_cache = False
			if decoder_input_ids is None and decoder_inputs_embeds is None:
				decoder_input_ids = shift_tokens_right(
					labels, self.config.pad_token_id, self.config.decoder_start_token_id
				)
		
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
		lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

		if not return_dict:
			output = (lm_logits,) + outputs[1:]
			return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

		masked_lm_loss = None
		if labels is not None:
			loss_fct = nn.CrossEntropyLoss()
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
		# cut decoder_input_ids if past is used
		if past_key_values is not None:
			decoder_input_ids = decoder_input_ids[:, -1:]

		return {
			"input_ids": None,  # encoder_outputs is defined. input_ids not needed
			"encoder_outputs": encoder_outputs.last_hidden_state,
			"past_key_values": past_key_values,
			"decoder_input_ids": decoder_input_ids,
			"attention_mask": attention_mask,
			"global_attention_mask": global_attention_mask,
			"head_mask": head_mask,
			"decoder_head_mask": decoder_head_mask,
			"cross_attn_head_mask": cross_attn_head_mask,
			"use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
		}

	def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
		return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

	@staticmethod
	def _reorder_cache(past_key_values, beam_idx):
		reordered_past = ()
		for layer_past in past_key_values:
			# cached cross_attention states don't have to be reordered -> they are always the same
			reordered_past += (
				tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past[:2])
				+ layer_past[2:],
			)
		return reordered_past
