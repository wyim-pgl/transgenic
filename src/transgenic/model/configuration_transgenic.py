"""
TransGenic model configuration module.

Defines HyenaTransgenicConfig, the configuration class for the TransGenic
seq2seq architecture (HyenaDNA encoder + Longformer decoder). All model
hyperparameters — dimensions, layer counts, attention windows, dropout,
and special token IDs — are stored here and consumed by the model classes
in modeling_HyenaTransgenic.py.

The default values match the base architecture (~400M params):
  d_model=768, 12 encoder/decoder layers, 6 attention heads.
The "wide" architecture used in production training overrides these to:
  d_model=1152, 16 layers, 8 heads (~1.17B params, ~3x base).
"""

from typing import List, Union
from transformers.configuration_utils import PretrainedConfig  # HuggingFace base config class
from transformers.utils import logging

logger = logging.get_logger(__name__)


class HyenaTransgenicConfig(PretrainedConfig):
	r"""
	Configuration class for the TransGenic model (HyenaDNA encoder + Longformer decoder).

	Instantiating with defaults yields a config similar to:
	[jlomas/HyenaTransgenic-768L12A6-400M](https://huggingface.co/jlomas/HyenaTransgenic-768L12A6-400M)

	Args:
		vocab_size: Number of GFF decoder tokens (see GFFTokenizer). Default 288.
		max_encoder_position_embeddings: Max encoder positional embedding length.
		max_decoder_position_embeddings: Max decoder positional embedding length.
		encoder_layers / decoder_layers: Number of transformer layers in each stack.
		encoder_ffn_dim / decoder_ffn_dim: Feed-forward hidden dimension (typically 4x d_model).
		encoder_attention_heads / decoder_attention_heads: Number of multi-head attention heads.
		d_model: Hidden size shared by encoder output projection and decoder embeddings.
		dropout: General dropout probability applied in attention and FFN layers.
		attention_window: Longformer local attention window size per decoder layer.
		encoder_model: HuggingFace model ID for the pretrained HyenaDNA encoder backbone.
		max_encoder_seqlen: Maximum DNA input sequence length the encoder can handle.
		encoder_n_layer: Number of HyenaDNA layers (may differ from decoder layers).
		unlink: If True, decoder embedding and LM head weights are NOT tied.
	max_transcript_count: Maximum number of alternative transcripts for the
		transcript count regression head (default 15).
	"""

	model_type = "transgenicHyena"  # Identifies this config type in HuggingFace model registry

	# Maps HuggingFace standard attribute names to TransGenic-specific names
	# so that generic HF utilities (e.g. from_pretrained) work correctly.
	attribute_map = {
		"num_attention_heads": "encoder_attention_heads",   # HF expects num_attention_heads
		"hidden_size": "d_model",                           # HF expects hidden_size
		"attention_probs_dropout_prob": "attention_dropout", # HF expects attention_probs_dropout_prob
		"initializer_range": "init_std",                    # HF expects initializer_range
	}

	def __init__(
		self,
		vocab_size=288,                           # GFF token vocabulary size (special + digit + feature + isoform tokens)
		max_encoder_position_embeddings=16384,    # Max positional embeddings for encoder sinusoidal PE
		max_decoder_position_embeddings=2048,     # Max positional embeddings for Longformer decoder
		encoder_layers=12,                        # Number of HyenaDNA encoder transformer blocks
		encoder_ffn_dim=3072,                     # Encoder feed-forward inner dimension (4x d_model for base)
		encoder_attention_heads=6,                # Number of attention heads in each encoder layer
		decoder_layers=12,                        # Number of Longformer decoder transformer blocks
		decoder_ffn_dim=3072,                     # Decoder feed-forward inner dimension
		decoder_attention_heads=6,                # Number of attention heads in each decoder layer
		encoder_layerdrop=0.0,                    # Probability of dropping entire encoder layers (disabled)
		decoder_layerdrop=0.1,                    # Probability of dropping entire decoder layers (regularization)
		use_cache=True,                           # Cache past key/values for faster autoregressive generation
		is_encoder_decoder=True,                  # Signals HF that this is a seq2seq model
		activation_function="gelu",               # Non-linearity in FFN blocks
		d_model=768,                              # Backward-compat base dim (defaults encoder dim)
		encoder_d_model=None,                     # Explicit encoder hidden dim (defaults to d_model)
		decoder_d_model=None,                     # Explicit decoder hidden dim (defaults to 2*encoder_d_model)
		dropout=0.1,                              # Dropout probability for attention weights and FFN outputs
		attention_dropout=0.0,                    # Extra dropout on attention probabilities (disabled by default)
		activation_dropout=0.0,                   # Dropout after activation in FFN (disabled by default)
		init_std=0.02,                            # Standard deviation for weight initialization (normal dist)
		decoder_start_token_id=2,                 # Token ID that begins decoder generation (</s> = 2)
		classifier_dropout=0.0,                   # Dropout for classification head (unused in seq2seq mode)
		pad_token_id=1,                           # <pad> token ID for masking padded positions
		bos_token_id=0,                           # <s> beginning-of-sequence token ID
		eos_token_id=2,                           # </s> end-of-sequence token ID
		attention_window: Union[List[int], int] = [  # Longformer local attention window per decoder layer
			1024, 1024, 1024, 1024, 1024, 1024,     # Each value = one-sided window; total window = 2 * value
			1024, 1024, 1024, 1024, 1024, 1024,      # 12 values for 12 decoder layers (base config)
		],
		encoder_model="LongSafari/hyenadna-large-1m-seqlen-hf",  # Pretrained HyenaDNA checkpoint
		max_encoder_seqlen=49152,                 # Max DNA nucleotide sequence length for encoder input
		encoder_n_layer=12,                       # Number of HyenaDNA layers to instantiate
		load_pretrained_encoder=True,             # Load compatible pretrained encoder weights when available
		allow_partial_encoder_load=True,          # Allow shape-filtered partial load for widened/deeper configs
		unlink=False,                             # Whether to untie decoder embedding ↔ LM head weights
		max_transcript_count=15,                  # Max transcript count for regression head
		**kwargs,
	):
		# Store all hyperparameters as instance attributes
		self.vocab_size = vocab_size
		self.max_encoder_position_embeddings = max_encoder_position_embeddings
		self.max_decoder_position_embeddings = max_decoder_position_embeddings
		# Resolve asymmetric dims while preserving backward compatibility.
		resolved_encoder_d_model = d_model if encoder_d_model is None else encoder_d_model
		resolved_decoder_d_model = (resolved_encoder_d_model * 2) if decoder_d_model is None else decoder_d_model

		# Keep d_model as encoder width for compatibility with older code.
		self.d_model = resolved_encoder_d_model
		self.encoder_d_model = resolved_encoder_d_model
		self.decoder_d_model = resolved_decoder_d_model
		self.encoder_ffn_dim = encoder_ffn_dim
		self.encoder_layers = encoder_layers
		self.encoder_attention_heads = encoder_attention_heads
		self.decoder_ffn_dim = decoder_ffn_dim
		self.decoder_layers = decoder_layers
		self.decoder_attention_heads = decoder_attention_heads
		self.dropout = dropout
		self.attention_dropout = attention_dropout
		self.activation_dropout = activation_dropout
		self.activation_function = activation_function
		self.init_std = init_std
		self.encoder_layerdrop = encoder_layerdrop
		self.decoder_layerdrop = decoder_layerdrop
		self.classifier_dropout = classifier_dropout
		self.use_cache = use_cache
		self.num_hidden_layers = encoder_layers    # HF expects num_hidden_layers for layer iteration
		self.attention_window = attention_window
		self.encoder_model = encoder_model
		self.encoder_n_layer = encoder_n_layer
		self.max_encoder_seqlen = max_encoder_seqlen
		self.load_pretrained_encoder = load_pretrained_encoder
		self.allow_partial_encoder_load = allow_partial_encoder_load
		self.unlink = unlink
		self.max_transcript_count = max_transcript_count

		# Initialize the HuggingFace PretrainedConfig base class with special token IDs
		super().__init__(
			pad_token_id=pad_token_id,
			bos_token_id=bos_token_id,
			eos_token_id=eos_token_id,
			is_encoder_decoder=is_encoder_decoder,
			decoder_start_token_id=decoder_start_token_id,
			**kwargs,
		)
