import sys, re, os, json
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM, AutoConfig, PreTrainedTokenizer, PreTrainedModel, AutoModel
from transformers import LEDForConditionalGeneration, EsmForMaskedLM
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass
from configuration_transgenic import TransgenicConfig

def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
	"""
	Shift input ids one token to the right.
	"""
	shifted_input_ids = input_ids.new_zeros(input_ids.shape)
	shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
	shifted_input_ids[:, 0] = decoder_start_token_id

	if pad_token_id is None:
		raise ValueError("config.pad_token_id has to be defined.")
	# replace possible -100 values in labels by `pad_token_id`
	shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

	return shifted_input_ids

#Copied from transformers.models.led.modeling_led.py
class LEDLearnedPositionalEmbedding(nn.Embedding):
	"""
	This module learns positional embeddings up to a fixed maximum size.
	"""

	def __init__(self, num_embeddings: int, embedding_dim: int):
		super().__init__(num_embeddings, embedding_dim)

	def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0):
		"""`input_ids_shape` is expected to be [bsz x seqlen]."""
		bsz, seq_len = input_ids_shape[:2]
		positions = torch.arange(
			past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
		)
		return super().forward(positions)

class FiLMLayer(nn.Module):
	def __init__(self, embedding_dim, num_classes):
		super(FiLMLayer, self).__init__()
		# Linear layers to generate gamma and beta
		self.gamma_layer = nn.Linear(num_classes, embedding_dim)
		self.beta_layer = nn.Linear(num_classes, embedding_dim)

	def forward(self, embedding, class_probs):
		# Compute gamma and beta from conditioning information
		gamma = self.gamma_layer(class_probs)
		beta = self.beta_layer(class_probs)
		
		# Modulate the embedding with gamma and beta
		modulated_embedding = gamma * embedding + beta
		
		return modulated_embedding

#Copied from transformers.models.led.modeling_led.py
@dataclass
class LEDSeq2SeqLMOutput(ModelOutput):
	"""
	Base class for sequence-to-sequence language models outputs.

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

#Copied from transformers.models.led.modeling_led.py
@dataclass
class LEDSeq2SeqModelOutput(ModelOutput):
	"""
	Base class for model encoder's outputs that also contains : pre-computed hidden states that can speed up sequential
	decoding.

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

class NumberMaskEmbedTokens(nn.Embedding):
	def __init__(self, num_embeddings, embedding_dim, padding_idx=0):
		super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)
		self.num_feature_embed = nn.Embedding(2, embedding_dim)

	def forward(self, input_ids):
		# Create a mask for numerical tokens and embed it
		num_mask = self.create_numerical_mask(input_ids)
		num_mask = self.num_feature_embed(num_mask)
		
		# Generate embeddings for the input tokens
		num_embeddings = super().forward(input_ids)

		# Add the numerical mask to the embeddings
		num_embeddings = num_embeddings + num_mask
		
		return num_embeddings

	def create_numerical_mask(self, input_ids):
		# Create a binary mask where 1 indicates a numerical token and 0 otherwise
		num_mask = torch.zeros_like(input_ids, dtype=torch.long)
		for i, sentence in enumerate(input_ids):
			for j, token in enumerate(sentence):
				if (token.item() >= 4) & (token.item() <= 13):
					num_mask[i,j] = 1
		return num_mask

class segmented_sequence_embeddings(EsmForMaskedLM):
	def __init__(self, e_model, s_model, numSegClasses, outputSize = 768):
		self.cache_dir = "./HFmodels"
		self.encoder_model = e_model
		self.segmentation_model = s_model
		self.numSegClasses = numSegClasses
		config = AutoConfig.from_pretrained(self.encoder_model, is_decoder=False, trust_remote_code=True)
		super().__init__(config)
		
		self.esm = AutoModelForMaskedLM.from_pretrained(self.encoder_model, cache_dir=self.cache_dir, trust_remote_code=True)
		SegmentNT = AutoModel.from_pretrained(self.segmentation_model, trust_remote_code=True)
		self.unet = SegmentNT.unet
		self.uFC = SegmentNT.fc
		self.uActivation = SegmentNT.activation_fn
		self.film = FiLMLayer(outputSize, self.numSegClasses)

		# TODO: Exlpore other options? (hidden states, BiLSTM, linear, attention, pooling, convolution)
		#TODO: Put into config - plants -> 1500, multispecies -> 1024, longformer -> 768
		self.hidden_mapping = nn.Linear(config.hidden_size, 768)
		self.hidden_mapping_layernorm = nn.LayerNorm(768, eps=1e-5)
		self.unet_mapping = nn.Linear(config.hidden_size, 1024)
		self.unet_mapping_layernorm = nn.LayerNorm(1024, eps=1e-5)
	
	def forward(self, input_ids, attention_mask=None, segLabels=None, **kwargs):
		batch_size = input_ids.shape[0]
		num_segs = input_ids.shape[1] // 1024
		input_ids = input_ids.reshape(batch_size, int(num_segs), 1024)
		attention_mask = attention_mask.reshape(batch_size, int(num_segs), 1024)
		for i in range(batch_size):
			#with torch.no_grad():
			embeddings = self.esm(
				input_ids[i, :, :],
				attention_mask=attention_mask[i,:,:],
				encoder_attention_mask=attention_mask[i,:,:],
				output_hidden_states=True
			)['hidden_states'][-1]
			
			if i == 0:
				batch_embeds = embeddings.reshape(1, num_segs*1024, -1)
				batch_mask = attention_mask[i,:,:].reshape(1, num_segs*1024)
			else:
				batch_embeds = torch.cat((batch_embeds, embeddings.reshape(1, num_segs*1024, -1)), dim=0)
				batch_mask = torch.cat((batch_mask, attention_mask[i,:,:].reshape(1, num_segs*1024)), dim=0)
		
		# Use the last hidden state from the nucleotide encoder as input to the decoder
		# Transform the encoder hidden states to match the decoder input size
		decoder_inputs_embeds = self.hidden_mapping(batch_embeds)
		decoder_inputs_embeds = self.hidden_mapping_layernorm(decoder_inputs_embeds)

		# Use the last hidden state from the nucleotide encoder as input to the segmentation model
		# Transform the encoder hidden states to match the decoder input size
		seg_inputs = self.unet_mapping(batch_embeds)
		seg_inputs = self.unet_mapping_layernorm(seg_inputs)
		
		
		# Invert the channels and sequence length channel
		#seg_inputs = seg_inputs[:,1:,:] # Remove CLS token
		seg_inputs = torch.transpose(seg_inputs, 2,1)

		# Pass through UNET
		x = self.uActivation(self.unet(seg_inputs))

		# Invert the channels and sequence length channel
		x = torch.transpose(x, 2,1)

		# Compute logits for the segmentation model
		seg_logits = self.uFC(x)

		# Final reshape to have logits per nucleotides, per feature
		seg_logits = torch.reshape(seg_logits, (seg_logits.shape[0], seg_logits.shape[1] * 6, self.numSegClasses, 2))

		# Compute segmentation loss if called for
		if segLabels != None:

			# Define weight of positive splice junction classes to counteract class imbalance
			pos_weight = torch.ones(self.numSegClasses)
			pos_weight[[4, 5]] = 7.0
			pos_weight = pos_weight.to(segLabels.device)

			# Define per-nucleotide weights to focus on basic genic features
			weight = torch.Tensor((5.0, 0.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
			weight = weight.to(segLabels.device)

			# Define the loss function with pos_weight
			segLossFn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, weight=weight)
			seg_loss = segLossFn(torch.squeeze(seg_logits[:,0:segLabels.shape[0]][...,0]), segLabels)
		else:
			seg_loss = None


		# Aggregate U-Net output and apply FiLM to the encoder hidden states
		# TODO: portion embeddings into a batch for the film layer
		#sl_aggregated = seg_logits[...,0].view(-1, 6, 1024, self.numSegClasses)  
		#sl_aggregated = sl_aggregated.mean(dim=1)
		#decoder_inputs_embeds = self.film(decoder_inputs_embeds, sl_aggregated)

		return ModelOutput(
			inputs_embeds=decoder_inputs_embeds, 
			attention_mask=batch_mask, 
			seg_logits=seg_logits, 
			seg_loss=seg_loss)

class TransgenicPreTrainedModel(PreTrainedModel):
	config_class = TransgenicConfig
	base_model_prefix = "led"
	supports_gradient_checkpointing = True

	def _init_weights(self, module):
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
		pad_token = self.config.pad_token_id
		input_ids = torch.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]], device=self.device)
		dummy_inputs = {
			"attention_mask": input_ids.ne(pad_token),
			"input_ids": input_ids,
		}
		return dummy_inputs

class transgenicModel(TransgenicPreTrainedModel):
	_tied_weights_keys = ["decoder_embed_tokens.weight", "decoder.embed_tokens.weight"]

	def __init__(self, config):
		super().__init__(config)

		padding_idx, vocab_size = config.pad_token_id, config.vocab_size
		#self.decoder_embed_tokens = NumberMaskEmbedTokens(vocab_size, config.d_model, padding_idx)
		self.decoder_embed_tokens = nn.Embedding(vocab_size, config.d_model, padding_idx)
		
		self.encoder = segmented_sequence_embeddings(config.encoder_model)
		self.decoder = LEDForConditionalGeneration(config).led.decoder
		self.decoder.embed_tokens = self.decoder_embed_tokens

		# Initialize weights and apply final processing
		self.post_init()

	def get_input_embeddings(self):
		return self.decoder_embed_tokens

	def set_input_embeddings(self, value):
		self.decoder_embed_tokens = value
		self.decoder.embed_tokens = self.decoder_embed_tokens

	def get_encoder(self):
		return self.encoder

	def get_decoder(self):
		return self.decoder

	#@add_start_docstrings_to_model_forward(LED_INPUTS_DOCSTRING)
	#@add_code_sample_docstrings(
	#	checkpoint=_CHECKPOINT_FOR_DOC,
	#	output_type=Seq2SeqModelOutput,
	#	config_class=_CONFIG_FOR_DOC,
	#)
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
		output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
		output_hidden_states = (
			output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
		)
		use_cache = use_cache if use_cache is not None else self.config.use_cache
		return_dict = return_dict if return_dict is not None else self.config.use_return_dict

		# Using this like Bart, as LED is derived from it. So far
		# No checkpoint on the hub exists that uses that in practice.
		# https://github.com/huggingface/transformers/blob/ac3cb660cad283163f7c73cad511124e845ca388/src/transformers/models/bart/modeling_bart.py#L1153
		if decoder_input_ids is None and decoder_inputs_embeds is None:
			decoder_input_ids = shift_tokens_right(
				input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
			)
		# Compute the embeddings with nucleotide transformer encoder
		if encoder_outputs is None:
			encoder_outputs = self.encoder(input_ids, 
										attention_mask=attention_mask, 
										return_dict=return_dict
										).to_tuple()
		else:
			encoder_outputs = (encoder_outputs["inputs_embeds"], encoder_outputs["attention_mask"])
		#if encoder_outputs is None:
		#	encoder_outputs = self.encoder(
		#		input_ids=input_ids,
		#		attention_mask=attention_mask,
		#		global_attention_mask=global_attention_mask,
		#		head_mask=head_mask,
		#		inputs_embeds=inputs_embeds,
		#		output_attentions=output_attentions,
		#		output_hidden_states=output_hidden_states,
		#		return_dict=return_dict,
		#	)

		# If the user passed a tuple for encoder_outputs, we wrap it in a LEDEncoderBaseModelOutput when return_dict=False
		#elif return_dict and not isinstance(encoder_outputs, LEDEncoderBaseModelOutput):
		#	encoder_outputs = LEDEncoderBaseModelOutput(
		#		last_hidden_state=encoder_outputs[0],
		#		hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
		#		attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
		#		global_attentions=encoder_outputs[3] if len(encoder_outputs) > 3 else None,
		#	)

		# decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
		decoder_outputs = self.decoder(
			input_ids=decoder_input_ids,
			attention_mask=decoder_attention_mask,
			encoder_hidden_states=encoder_outputs[0],
			encoder_attention_mask=encoder_outputs[1].long(), #attention_mask,
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

		if not return_dict:
			return decoder_outputs + encoder_outputs

		return LEDSeq2SeqModelOutput(
			last_hidden_state=decoder_outputs.last_hidden_state,
			past_key_values=decoder_outputs.past_key_values,
			decoder_hidden_states=decoder_outputs.hidden_states,
			decoder_attentions=decoder_outputs.attentions,
			cross_attentions=decoder_outputs.cross_attentions,
			encoder_last_hidden_state=encoder_outputs[0],
			encoder_hidden_states=None,
			encoder_attentions=encoder_outputs[1],
			encoder_global_attentions=None,
		)

class transgenicForConditionalGeneration(TransgenicPreTrainedModel):
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
		nn.init.xavier_uniform_(self.transgenic.encoder.hidden_mapping.weight)
		nn.init.constant_(self.transgenic.encoder.hidden_mapping.bias, 0)

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
			#loss_fct = HybridSequenceLoss()
			#masked_lm_loss = loss_fct(lm_logits, labels)

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
			"encoder_outputs": encoder_outputs,
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