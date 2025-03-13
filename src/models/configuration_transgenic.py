from typing import List, Union
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

# Modified from the LED configuration (transformers.models.led.configuration_led.py)
class NTTransgenicConfig(PretrainedConfig):
	r"""
	This is the configuration class to store the configuration of a [`transgenicModel`]. It is used to instantiate an Transgenic
	model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
	defaults will yield a similar configuration to that of the
	[jlomas/transgenic-agro-E9](https://huggingface.co/jlomas/transgenic-agro-E9) architecture.

	Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
	documentation from [`PretrainedConfig`] for more information.


	Args:
		vocab_size (`int`, *optional*, defaults to 272):
			Vocabulary size of the Transgenic model. Defines the number of different tokens that can be represented by the
			`decoder_inputs_ids` passed when calling [`transgenicModel`].
		d_model (`int`, *optional*, defaults to 1024):
			Dimensionality of the layers and the pooler layer.
		encoder_layers (`int`, *optional*, defaults to 12):
			Number of encoder layers.
		decoder_layers (`int`, *optional*, defaults to 12):
			Number of decoder layers.
		encoder_attention_heads (`int`, *optional*, defaults to 16):
			Number of attention heads for each attention layer in the Transformer encoder.
		decoder_attention_heads (`int`, *optional*, defaults to 16):
			Number of attention heads for each attention layer in the Transformer decoder.
		decoder_ffn_dim (`int`, *optional*, defaults to 4096):
			Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
		encoder_ffn_dim (`int`, *optional*, defaults to 4096):
			Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
		activation_function (`str` or `function`, *optional*, defaults to `"gelu"`):
			The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
			`"relu"`, `"silu"` and `"gelu_new"` are supported.
		dropout (`float`, *optional*, defaults to 0.1):
			The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
		attention_dropout (`float`, *optional*, defaults to 0.0):
			The dropout ratio for the attention probabilities.
		activation_dropout (`float`, *optional*, defaults to 0.0):
			The dropout ratio for activations inside the fully connected layer.
		classifier_dropout (`float`, *optional*, defaults to 0.0):
			The dropout ratio for classifier.
		max_encoder_position_embeddings (`int`, *optional*, defaults to 16384):
			The maximum sequence length that the encoder might ever be used with.
		max_decoder_position_embeddings (`int`, *optional*, defaults to 16384):
			The maximum sequence length that the decoder might ever be used with.
		init_std (`float`, *optional*, defaults to 0.02):
			The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
		encoder_layerdrop (`float`, *optional*, defaults to 0.0):
			The LayerDrop probability for the encoder. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556)
			for more details.
		decoder_layerdrop (`float`, *optional*, defaults to 0.0):
			The LayerDrop probability for the decoder. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556)
			for more details.
		use_cache (`bool`, *optional*, defaults to `True`):
			Whether or not the model should return the last key/values attentions (not used by all models)
	```"""

	model_type = "transgenic"
	attribute_map = {
		"num_attention_heads": "encoder_attention_heads",
		"hidden_size": "d_model",
		"attention_probs_dropout_prob": "attention_dropout",
		"initializer_range": "init_std",
	}

	def __init__(
		self,
		vocab_size=272,
		max_encoder_position_embeddings=16384,
		max_decoder_position_embeddings=2048,
		encoder_layers=6,
		encoder_ffn_dim=3072,
		encoder_attention_heads=12,
		decoder_layers=6,
		decoder_ffn_dim=3072,
		decoder_attention_heads=12,
		encoder_layerdrop=0.0,
		decoder_layerdrop=0.1,
		use_cache=True,
		is_encoder_decoder=True,
		activation_function="gelu",
		d_model=768,
		dropout=0.1,
		attention_dropout=0.0,
		activation_dropout=0.0,
		init_std=0.02,
		decoder_start_token_id=2,
		classifier_dropout=0.0,
		pad_token_id=1,
		bos_token_id=0,
		eos_token_id=2,
		attention_window: Union[List[int], int] = [
			1024,
			1024,
			1024,
			1024,
			1024,
			1024
		],
		encoder_model="InstaDeepAI/agro-nucleotide-transformer-1b",
		unlink=False,
		s_model="InstaDeepAI/segment_nt_multi_species",
		numSegClasses=14,
		do_segment = True,
		**kwargs,
	):
		self.vocab_size = vocab_size
		self.max_encoder_position_embeddings = max_encoder_position_embeddings
		self.max_decoder_position_embeddings = max_decoder_position_embeddings
		self.d_model = d_model
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
		self.num_hidden_layers = encoder_layers
		self.attention_window = attention_window
		self.encoder_model = encoder_model
		self.unlink = unlink
		self.s_model = s_model
		self.numSegClasses = numSegClasses
		self.do_segment = do_segment

		super().__init__(
			pad_token_id=pad_token_id,
			bos_token_id=bos_token_id,
			eos_token_id=eos_token_id,
			is_encoder_decoder=is_encoder_decoder,
			decoder_start_token_id=decoder_start_token_id,
			**kwargs,
		)

class HyenaTransgenicConfig(PretrainedConfig):
	r"""
	This is the configuration class to store the configuration of a [`transgenicModel`]. It is used to instantiate an Transgenic
	model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
	defaults will yield a similar configuration to that of the
	[jlomas/transgenic-agro-E9](https://huggingface.co/jlomas/transgenic-agro-E9) architecture.

	Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
	documentation from [`PretrainedConfig`] for more information.


	Args:
		vocab_size (`int`, *optional*, defaults to 272):
			Vocabulary size of the Transgenic model. Defines the number of different tokens that can be represented by the
			`decoder_inputs_ids` passed when calling [`transgenicModel`].
		d_model (`int`, *optional*, defaults to 1024):
			Dimensionality of the layers and the pooler layer.
		encoder_layers (`int`, *optional*, defaults to 12):
			Number of encoder layers.
		decoder_layers (`int`, *optional*, defaults to 12):
			Number of decoder layers.
		encoder_attention_heads (`int`, *optional*, defaults to 16):
			Number of attention heads for each attention layer in the Transformer encoder.
		decoder_attention_heads (`int`, *optional*, defaults to 16):
			Number of attention heads for each attention layer in the Transformer decoder.
		decoder_ffn_dim (`int`, *optional*, defaults to 4096):
			Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
		encoder_ffn_dim (`int`, *optional*, defaults to 4096):
			Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
		activation_function (`str` or `function`, *optional*, defaults to `"gelu"`):
			The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
			`"relu"`, `"silu"` and `"gelu_new"` are supported.
		dropout (`float`, *optional*, defaults to 0.1):
			The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
		attention_dropout (`float`, *optional*, defaults to 0.0):
			The dropout ratio for the attention probabilities.
		activation_dropout (`float`, *optional*, defaults to 0.0):
			The dropout ratio for activations inside the fully connected layer.
		classifier_dropout (`float`, *optional*, defaults to 0.0):
			The dropout ratio for classifier.
		max_encoder_position_embeddings (`int`, *optional*, defaults to 16384):
			The maximum sequence length that the encoder might ever be used with.
		max_decoder_position_embeddings (`int`, *optional*, defaults to 16384):
			The maximum sequence length that the decoder might ever be used with.
		init_std (`float`, *optional*, defaults to 0.02):
			The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
		encoder_layerdrop (`float`, *optional*, defaults to 0.0):
			The LayerDrop probability for the encoder. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556)
			for more details.
		decoder_layerdrop (`float`, *optional*, defaults to 0.0):
			The LayerDrop probability for the decoder. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556)
			for more details.
		use_cache (`bool`, *optional*, defaults to `True`):
			Whether or not the model should return the last key/values attentions (not used by all models)
	```"""

	model_type = "transgenicHyena"
	attribute_map = {
		"num_attention_heads": "encoder_attention_heads",
		"hidden_size": "d_model",
		"attention_probs_dropout_prob": "attention_dropout",
		"initializer_range": "init_std",
	}

	def __init__(
		self,
		vocab_size=272,
		max_encoder_position_embeddings=16384,
		max_decoder_position_embeddings=2048,
		encoder_layers=6,
		encoder_ffn_dim=3072,
		encoder_attention_heads=4,
		decoder_layers=9,
		decoder_ffn_dim=3072,
		decoder_attention_heads=4,
		encoder_layerdrop=0.0,
		decoder_layerdrop=0.1,
		use_cache=True,
		is_encoder_decoder=True,
		activation_function="gelu",
		d_model=256,
		dropout=0.1,
		attention_dropout=0.0,
		activation_dropout=0.0,
		init_std=0.02,
		decoder_start_token_id=2,
		classifier_dropout=0.0,
		pad_token_id=1,
		bos_token_id=0,
		eos_token_id=2,
		attention_window: Union[List[int], int] = [
			1024,
			1024,
			1024,
			1024,
			1024,
			1024
		],
		encoder_model="LongSafari/hyenadna-large-1m-seqlen-hf",
		max_encoder_seqlen=49152,
		encoder_n_layer=9,
		unlink=False,
		s_model="InstaDeepAI/segment_nt_multi_species",
		numSegClasses=9,
		do_segment = False,
		**kwargs,
	):
		self.vocab_size = vocab_size
		self.max_encoder_position_embeddings = max_encoder_position_embeddings
		self.max_decoder_position_embeddings = max_decoder_position_embeddings
		self.d_model = d_model
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
		self.num_hidden_layers = encoder_layers
		self.attention_window = attention_window
		self.encoder_model = encoder_model
		self.encoder_n_layer = encoder_n_layer
		self.max_encoder_seqlen = max_encoder_seqlen
		self.unlink = unlink
		self.s_model = s_model
		self.numSegClasses = numSegClasses
		self.do_segment = do_segment

		super().__init__(
			pad_token_id=pad_token_id,
			bos_token_id=bos_token_id,
			eos_token_id=eos_token_id,
			is_encoder_decoder=is_encoder_decoder,
			decoder_start_token_id=decoder_start_token_id,
			**kwargs,
		)

class NTTransgenicT5Config(PretrainedConfig):
	def __init__(
			self,
			classifier_dropout = 0.0,
			d_ff = 2048,
			d_kv = 64,
			d_model = 768,
			dense_act_fn = "relu",
			dropout_rate = 0.1,
			eos_token_id = 2,
			feed_forward_proj = "relu",
			initializer_factor = 1.0,
			is_encoder_decoder = True,
			is_gated_act = False,
			layer_norm_epsilon = 1e-06,
			model_type = "t5",
			num_decoder_layers = 6,
			num_heads = 8,
			num_layers = 6,
			pad_token_id = 1,
			decoder_start_token_id = 2,
			init_std=0.02,
			relative_attention_max_distance = 128,
			relative_attention_num_buckets = 32,
			transformers_version = "4.44.2",
			use_cache = True,
			vocab_size = 272,
			encoder_model="InstaDeepAI/agro-nucleotide-transformer-1b",
			s_model = "InstaDeepAI/segment_nt_multi_species",
			numSegClasses = 14,
			do_segment = False,
			**kwargs
	):
		self.classifier_dropout = classifier_dropout
		self.d_ff = d_ff
		self.d_kv = d_kv
		self.d_model = d_model
		self.dense_act_fn = dense_act_fn
		self.dropout_rate = dropout_rate
		self.eos_token_id = eos_token_id
		self.feed_forward_proj = feed_forward_proj
		self.initializer_factor = initializer_factor
		self.is_encoder_decoder = is_encoder_decoder
		self.is_gated_act = is_gated_act
		self.layer_norm_epsilon = layer_norm_epsilon
		self.model_type = model_type
		self.num_decoder_layers = num_decoder_layers
		self.num_heads = num_heads
		self.num_layers = num_layers
		self.pad_token_id = pad_token_id
		self.decoder_start_token_id = decoder_start_token_id
		self.init_std = init_std
		self.relative_attention_max_distance = relative_attention_max_distance
		self.relative_attention_num_buckets = relative_attention_num_buckets
		self.transformers_version = transformers_version
		self.use_cache = use_cache
		self.vocab_size = vocab_size
		self.encoder_model = encoder_model
		self.s_model = s_model
		self.numSegClasses = numSegClasses
		self.do_segment = do_segment

		act_info = self.feed_forward_proj.split("-")
		self.dense_act_fn = act_info[-1]
		self.is_gated_act = act_info[0] == "gated"

		if len(act_info) > 1 and act_info[0] != "gated" or len(act_info) > 2:
			raise ValueError(
				f"`feed_forward_proj`: {feed_forward_proj} is not a valid activation function of the dense layer. "
				"Please make sure `feed_forward_proj` is of the format `gated-{ACT_FN}` or `{ACT_FN}`, e.g. "
				"'gated-gelu' or 'relu'"
			)

		# for backwards compatibility
		if feed_forward_proj == "gated-gelu":
			self.dense_act_fn = "gelu_new"

		super().__init__(
			pad_token_id=pad_token_id,
			eos_token_id=eos_token_id,
			is_encoder_decoder=is_encoder_decoder,
			decoder_start_token_id=decoder_start_token_id,
			**kwargs,
		)

class HyenaTransgenicT5Config(PretrainedConfig):
	def __init__(
			self,
			classifier_dropout = 0.0,
			d_ff = 2048,
			d_kv = 64,
			d_model = 256,
			dense_act_fn = "relu",
			dropout_rate = 0.1,
			eos_token_id = 2,
			feed_forward_proj = "relu",
			initializer_factor = 1.0,
			is_encoder_decoder = True,
			is_gated_act = False,
			layer_norm_epsilon = 1e-06,
			model_type = "t5",
			num_decoder_layers = 6,
			num_heads = 8,
			num_layers = 6,
			pad_token_id = 1,
			decoder_start_token_id = 2,
			init_std=0.02,
			relative_attention_max_distance = 128,
			relative_attention_num_buckets = 32,
			transformers_version = "4.44.2",
			use_cache = True,
			vocab_size = 272,
			encoder_model="LongSafari/hyenadna-large-1m-seqlen-hf",
			max_encoder_seqlen=49152,
			encoder_n_layer=9,
			s_model = "InstaDeepAI/segment_nt_multi_species",
			numSegClasses = 14,
			do_segment = False,
			**kwargs
	):
		self.classifier_dropout = classifier_dropout
		self.d_ff = d_ff
		self.d_kv = d_kv
		self.d_model = d_model
		self.dense_act_fn = dense_act_fn
		self.dropout_rate = dropout_rate
		self.eos_token_id = eos_token_id
		self.feed_forward_proj = feed_forward_proj
		self.initializer_factor = initializer_factor
		self.is_encoder_decoder = is_encoder_decoder
		self.is_gated_act = is_gated_act
		self.layer_norm_epsilon = layer_norm_epsilon
		self.model_type = model_type
		self.num_decoder_layers = num_decoder_layers
		self.num_heads = num_heads
		self.num_layers = num_layers
		self.pad_token_id = pad_token_id
		self.decoder_start_token_id = decoder_start_token_id
		self.init_std = init_std
		self.relative_attention_max_distance = relative_attention_max_distance
		self.relative_attention_num_buckets = relative_attention_num_buckets
		self.transformers_version = transformers_version
		self.use_cache = use_cache
		self.vocab_size = vocab_size
		self.encoder_model = encoder_model
		self.max_encoder_seqlen=max_encoder_seqlen
		self.encoder_n_layer=encoder_n_layer
		self.s_model = s_model
		self.numSegClasses = numSegClasses
		self.do_segment = do_segment

		act_info = self.feed_forward_proj.split("-")
		self.dense_act_fn = act_info[-1]
		self.is_gated_act = act_info[0] == "gated"

		if len(act_info) > 1 and act_info[0] != "gated" or len(act_info) > 2:
			raise ValueError(
				f"`feed_forward_proj`: {feed_forward_proj} is not a valid activation function of the dense layer. "
				"Please make sure `feed_forward_proj` is of the format `gated-{ACT_FN}` or `{ACT_FN}`, e.g. "
				"'gated-gelu' or 'relu'"
			)

		# for backwards compatibility
		if feed_forward_proj == "gated-gelu":
			self.dense_act_fn = "gelu_new"

		super().__init__(
			pad_token_id=pad_token_id,
			eos_token_id=eos_token_id,
			is_encoder_decoder=is_encoder_decoder,
			decoder_start_token_id=decoder_start_token_id,
			**kwargs,
		)