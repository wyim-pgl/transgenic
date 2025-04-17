from typing import List, Union
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

class HyenaTransgenicConfig(PretrainedConfig):
	r"""
	This is the configuration class to store the configuration of a [`transgenicModel`]. It is used to instantiate an Transgenic
	model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
	defaults will yield a similar configuration to that of the
	[jlomas/HyenaTransgenic-768L12A6-400M](https://huggingface.co/jlomas/HyenaTransgenic-768L12A6-400M) architecture.
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
		encoder_layers=12,
		encoder_ffn_dim=3072,
		encoder_attention_heads=6,
		decoder_layers=12,
		decoder_ffn_dim=3072,
		decoder_attention_heads=6,
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
			1024,1024,1024,1024,1024,1024,
			1024,1024,1024,1024,1024,1024,
		],
		encoder_model="LongSafari/hyenadna-large-1m-seqlen-hf",
		max_encoder_seqlen=49152,
		encoder_n_layer=12,
		unlink=False,
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

		super().__init__(
			pad_token_id=pad_token_id,
			bos_token_id=bos_token_id,
			eos_token_id=eos_token_id,
			is_encoder_decoder=is_encoder_decoder,
			decoder_start_token_id=decoder_start_token_id,
			**kwargs,
		)