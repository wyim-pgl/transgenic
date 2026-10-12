"""
HuggingFace integration utilities for TransGenic models.

This module provides functions for loading, configuring, and publishing TransGenic
models to the HuggingFace Hub. TransGenic is a sequence-to-sequence architecture
that pairs a HyenaDNA encoder (based on ESM protein language model layers) with a
Longformer-style decoder for genomic annotation tasks.

The module supports three model-loading variants:
  - Standard model:       HyenaDNA encoder + Longformer decoder (getModel)
  - Large-decoder model:  Same encoder with a wider/deeper decoder  (getLargeDecoderModel)
  - PEFT-adapted model:   Standard model with IA3 adapters injected into the
                          ESM encoder's self-attention and feed-forward layers (getPeftModel)

All loaders accept safetensors checkpoints and perform key remapping so that
weights saved from PEFT-wrapped or torch-compiled training runs can be loaded
into the vanilla architecture.

Typical usage:
    from transgenic.utils.huggingface_integration import getModel, registerModel

    config = HyenaTransgenicConfig(d_model=768, decoder_layers=12, ...)
    model  = getModel(config, safetensors_model="checkpoint.safetensors", mode="train")
    registerModel("username/model-name", model)
"""

import re, sys, os
from safetensors import safe_open
from peft import IA3Config, get_peft_model

# Prepend the project's src/ directory so that local model modules can be
# imported regardless of the caller's working directory.
sys.path.insert(0, f'{os.getcwd()}/src')
from models.modeling_HeynaTransgenic import transgenicForConditionalGeneration, transgenicModel
from models.configuration_transgenic import HyenaTransgenicConfig
from models.tokenization_transgenic import GFFTokenizer


def getModel(config, safetensors_model=None, device="cpu", mode="predict"):
	"""
	Load a standard TransGenic model (HyenaDNA encoder + Longformer decoder).

	The function instantiates a ``transgenicForConditionalGeneration`` model from
	the given configuration. When *mode* is ``"train"``, the ESM encoder backbone
	is frozen while the decoder and the hidden-mapping projection layers remain
	trainable. An optional safetensors checkpoint can be loaded with automatic
	key remapping to strip PEFT prefixes and IA3-specific weights.

	Args:
		config: A ``HyenaTransgenicConfig`` instance defining the model
			architecture. If falsy (e.g. ``None``), a default config is created.
		safetensors_model (str, optional): Path to a ``.safetensors`` checkpoint
			file. When provided the weights are loaded into the model after
			instantiation. Keys are remapped to remove ``"base_model.model."``
			and ``".base_layer"`` prefixes that originate from PEFT-wrapped
			training, and any IA3-specific weights are discarded so the
			checkpoint can be loaded into the non-PEFT model.
		device (str): Target device string (e.g. ``"cpu"``, ``"cuda:0"``).
			Used only to decide whether gradient checkpointing should be
			enabled (skipped on CPU to avoid unnecessary overhead).
		mode (str): Either ``"predict"`` or ``"train"``. In training mode the
			encoder is frozen and gradient checkpointing is enabled on GPU.

	Returns:
		transgenicForConditionalGeneration: The initialised (and optionally
		checkpoint-loaded) model instance.
	"""
	if not config:
		# Load the model and add to device
		config = HyenaTransgenicConfig()

	model = transgenicForConditionalGeneration(config)

	if mode == "train":
		# ---- Freeze / unfreeze strategy for fine-tuning ----
		# Freeze the entire ESM encoder backbone so its pre-trained weights
		# are not updated during training.
		for param in model.transgenic.encoder.esm.parameters():
			param.requires_grad = False

		# Add gradients back for the entire decoder and the hidden mapping layers
		# The decoder is the primary target of fine-tuning.
		for param in model.transgenic.decoder.parameters():
			param.requires_grad = True

		# The hidden_mapping linear layer projects the encoder's hidden states
		# to match the decoder's expected dimensionality, so it must also be
		# trainable.
		for param in model.transgenic.encoder.hidden_mapping.parameters():
			param.requires_grad = True

		# The layer-norm applied after the hidden mapping also needs gradients.
		for param in model.transgenic.encoder.hidden_mapping_layernorm.parameters():
			param.requires_grad = True

		#model.print_trainable_parameters()

		# Enable gradient checkpointing on GPU to reduce peak memory usage at the
		# cost of recomputing intermediate activations during the backward pass.
		# ``use_reentrant=False`` is the newer, safer mode that works correctly
		# with ``torch.autograd.Function`` and avoids subtle bugs with the
		# reentrant implementation.
		if f"{device}" != "cpu":
			try:
				model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
			except:
				print("\nNo gradient checkpointing available\n", file=sys.stderr)

	# Load checkpoint if provided
	if safetensors_model:
		# Read all tensors from the safetensors file into a flat dictionary.
		tensors = {}
		with safe_open(safetensors_model, framework="pt", device="cpu") as f:
			for k in f.keys():
				tensors[k] = f.get_tensor(k)

		# ---- Weight tying for the decoder embedding ----
		# The model ties the decoder input embedding weights (decoder_embed_tokens)
		# with the output projection head (lm_head). Ensure both keys point to
		# the same tensor so that ``load_state_dict`` receives consistent values.
		tensors["transgenic.decoder_embed_tokens.weight"] = tensors["lm_head.weight"]
		tensors["transgenic.decoder.embed_tokens.weight"] = tensors["transgenic.decoder_embed_tokens.weight"]

		# Remove the optional numerical feature embedding weight if present; it
		# is not part of the standard model and would cause a key mismatch.
		if "transgenic.decoder_embed_tokens.num_feature_embed.weight" in tensors.keys():
			del tensors["transgenic.decoder_embed_tokens.num_feature_embed.weight"]

		# ---- PEFT prefix cleanup ----
		# Checkpoints saved from PEFT-wrapped models contain keys prefixed with
		# "base_model.model." and may include ".base_layer" in layer names.
		# Strip these so the keys align with the unwrapped model's state_dict.
		newtensors = {k.replace("base_model.model.", "").replace(".base_layer", ""):tensors[k] for k in tensors}

		# ---- Remove IA3 adapter weights ----
		# IA3 parameters (identified by the "ia3" substring in their key names)
		# are specific to the PEFT adapter and should not be loaded into the
		# base model. Filter them out entirely.
		newnewtensors = {}
		for k in newtensors:
			if "ia3" not in k:
				newnewtensors[k] = newtensors[k]

		# Load the cleaned-up state dict into the model (strict by default,
		# so any remaining key mismatch will raise an error).
		model.load_state_dict(newnewtensors)

	return model


def getLargeDecoderModel(config, safetensors_model=None, device="cpu", mode="predict"):
	"""
	Load a TransGenic variant with a larger Longformer decoder.

	This variant uses a wider hidden dimension (d_model=1500), more decoder
	layers (12), more attention heads (15), and a larger feed-forward
	intermediate size (ffn_dim=6000) compared to the standard model. The
	larger decoder provides greater capacity for generation tasks that
	require richer output representations.

	Args:
		config: A ``HyenaTransgenicConfig`` instance. If falsy, a default
			config with the large-decoder hyperparameters is created:
			  - d_model = 1500
			  - attention_window = 1024
			  - decoder_ffn_dim = 6000
			  - encoder/decoder_attention_heads = 15
			  - decoder_layers = 12
		safetensors_model (str, optional): Path to a ``.safetensors``
			checkpoint. Unlike ``getModel``, no key remapping is performed
			here -- the checkpoint is expected to match the model's
			state_dict exactly.
		device (str): Target device string. Used to decide whether to
			enable gradient checkpointing (skipped on CPU).
		mode (str): Either ``"predict"`` or ``"train"``. In training mode
			all decoder parameters are made trainable and gradient
			checkpointing is enabled on GPU.

	Returns:
		transgenicForConditionalGeneration: The initialised model instance
		with the large-decoder configuration.
	"""
	if not config:
		# Load the model and add to device
		# Construct a config with the enlarged decoder dimensions.
		config = HyenaTransgenicConfig(
			d_model=1500,
			attention_window=1024,
			decoder_ffn_dim=6000,
			encoder_attention_heads=15,
			decoder_attention_heads=15,
			decoder_layers=12
			)

	model = transgenicForConditionalGeneration(config)

	if mode == "train":
		# Add gradients back for the entire decoder and the hidden mapping layers
		# Only the decoder is unfrozen here; the encoder remains fully frozen
		# (its parameters default to requires_grad=False from the model init).
		for param in model.transgenic.decoder.parameters():
			param.requires_grad = True

		# Enable gradient checkpointing on GPU for memory-efficient training.
		if f"{device}" != "cpu":
			try:
				model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
			except:
				print("\nNo gradient checkpointing available\n", file=sys.stderr)

	# Load checkpoint if provided
	if safetensors_model:
		# Read all tensors from the checkpoint file. No key remapping is
		# needed because the large-decoder checkpoint is assumed to be saved
		# directly from this model variant (not from a PEFT wrapper).
		tensors = {}
		with safe_open(safetensors_model, framework="pt", device="cpu") as f:
			for k in f.keys():
				tensors[k] = f.get_tensor(k)
		model.load_state_dict(tensors)

	return model


def getPeftModel(decoder_checkpoint, segment_checkpoint, config=None, unlink=False, safetensors_model=None, device="cpu", mode="predict"):
	"""
	Load a TransGenic model with PEFT IA3 adapters on the ESM encoder.

	IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations) is a
	parameter-efficient fine-tuning method that learns per-element rescaling
	vectors for selected weight matrices. This function injects IA3 adapters
	into every self-attention (Q, K, V, output-dense) and feed-forward (intermediate,
	output) linear layer of the ESM encoder, then loads two separate safetensors
	checkpoints -- one for the decoder weights and one for the segmentation
	(U-Net + fully-connected) head -- and merges them before calling
	``load_state_dict``.

	Args:
		decoder_checkpoint (str): Path to a ``.safetensors`` file containing
			the decoder (and IA3-adapted encoder) weights. This checkpoint
			is expected to use PEFT-style key names with the
			``"base_model.model."`` prefix.
		segment_checkpoint (str): Path to a ``.safetensors`` file containing
			the segmentation head weights (U-Net and fully-connected
			classifier). Keys are remapped from short names (``"unet.*"``,
			``"uFC.*"``) to their full paths under the transgenic encoder
			namespace.
		config: A ``HyenaTransgenicConfig`` instance. If falsy, a default
			config is created.
		unlink (bool): Currently unused; reserved for future use (e.g. to
			untie the decoder embedding and lm_head weights).
		safetensors_model (str, optional): Currently unused; the two
			separate checkpoint paths (decoder_checkpoint and
			segment_checkpoint) are used instead.
		device (str): Target device string. Used to decide whether to
			enable gradient checkpointing.
		mode (str): Either ``"predict"`` or ``"train"``. In training mode
			the decoder, hidden-mapping layers, and IA3 adapter weights
			are all trainable.

	Returns:
		PeftModelForSeq2SeqLM: The PEFT-wrapped model with IA3 adapters
		injected and both checkpoints loaded.
	"""
	if not config:
		# Load the model and add to device
		config = HyenaTransgenicConfig()

	model = transgenicForConditionalGeneration(config)

	# ---- Define IA3 target modules via regex patterns ----
	# Targets all self-attention components and dense linear layers for peft adaptors in the ESM encoder
	# These patterns match every attention sub-layer (query, key, value, output
	# projection) and the two feed-forward dense layers across all encoder
	# transformer blocks.
	target_modules = [
		r".*esm.encoder.layer.*.attention.self.query",
		r".*esm.encoder.layer.*.attention.self.key",
		r".*esm.encoder.layer.*.attention.self.value",
		r".*esm.encoder.layer.*.attention.output.dense",
		r".*esm.encoder.layer.*.intermediate.dense",
		r".*esm.encoder.layer.*.output.dense"]

	# Recompute activations for the dense layers
	# Feed-forward modules are a subset of target modules. IA3 treats these
	# differently: their learned rescaling vectors are applied to the *output*
	# of the layer (multiplicative gating after the activation), whereas
	# attention layers are rescaled on the *input* side.
	feedforward_modules = [
		r".*esm.encoder.layer.*.attention.output.dense",
		r".*esm.encoder.layer.*.intermediate.dense",
		r".*esm.encoder.layer.*.output.dense"]

	# ---- Resolve regex patterns to concrete module names ----
	# Walk through all named modules in the model and collect those whose
	# fully-qualified name matches one of the regex patterns above. The
	# resulting lists of concrete module paths are passed to IA3Config.
	# Find the target and feedforward modules
	peft_targets = []
	peft_feedforward = []
	for module in model.named_modules():
		for pattern in target_modules:
			if re.match(pattern, module[0]):
				peft_targets.append(module[0])
		for pattern in feedforward_modules:
			if re.match(pattern, module[0]):
				peft_feedforward.append(module[0])

	# ---- Create and apply the IA3 adapter ----
	# Load the IA3 adaptor
	# task_type="SEQ_2_SEQ_LM" tells PEFT that this is an encoder-decoder
	# language model, so it knows which layers to wrap and how to handle the
	# forward pass.  init_ia3_weights=True initialises the scaling vectors
	# to ones so the model's initial behaviour is unchanged.
	peft_config = IA3Config(task_type="SEQ_2_SEQ_LM", target_modules = peft_targets, feedforward_modules = peft_feedforward, init_ia3_weights = True)
	model = get_peft_model(model, peft_config)

	if mode == "train":
		# ---- Configure trainable parameters for fine-tuning ----
		# Add gradients back for the entire decoder and the hidden mapping layers
		# The IA3 adapter parameters (scaling vectors) are already trainable
		# by default after ``get_peft_model``. Here we additionally unfreeze
		# the full decoder and the encoder's hidden-mapping projection.
		for param in model.transgenic.decoder.parameters():
			param.requires_grad = True
		for param in model.transgenic.encoder.hidden_mapping.parameters():
			param.requires_grad = True
		for param in model.transgenic.encoder.hidden_mapping_layernorm.parameters():
			param.requires_grad = True

		# Print a summary showing the total vs. trainable parameter count
		# to verify the PEFT configuration is correct.
		model.print_trainable_parameters()

		# Enable gradient checkpointing on GPU. Non-reentrant mode is used
		# for compatibility with PEFT-wrapped layers.
		if f"{device}" != "cpu":
			try:
				model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
			except:
				print("\nNo gradient checkpointing available\n", file=sys.stderr)

	# ---- Load and merge two separate checkpoints ----
	# The PEFT model requires weights from two independently-trained stages:
	# (1) the decoder + IA3-adapted encoder, and (2) the segmentation head.

	# Load checkpoint if provided
	# Load decoder checkpoint
	# Stage 1: Decoder checkpoint -- contains the decoder, lm_head, encoder
	# backbone (with IA3 adapter weights), and hidden-mapping layers.
	decoder_tensors = {}
	with safe_open(decoder_checkpoint, framework="pt", device="cpu") as f:
		for k in f.keys():
			decoder_tensors[k] = f.get_tensor(k)

	# ---- Weight tying for the decoder embedding ----
	# Copy the decoder_embed_tokens weight to both the lm_head (output projection)
	# and the decoder's internal embed_tokens, ensuring weight tying is honoured.
	decoder_tensors["base_model.model.lm_head.weight"] = decoder_tensors["base_model.model.transgenic.decoder_embed_tokens.weight"]
	decoder_tensors["base_model.model.transgenic.decoder.embed_tokens.weight"] = decoder_tensors["base_model.model.transgenic.decoder_embed_tokens.weight"]


	# Load segmentation checkpoint
	# Stage 2: Segmentation checkpoint -- contains the U-Net and fully-connected
	# classifier head weights, saved with short key names (e.g. "unet.down1.conv...").
	segment_tensors = {}
	with safe_open(segment_checkpoint, framework="pt", device="cpu") as f:
		for k in f.keys():
			segment_tensors[k] = f.get_tensor(k)

	# Remap the short segmentation key names into the full PEFT-namespaced paths
	# expected by the model's state_dict (e.g. "unet.*" becomes
	# "base_model.model.transgenic.encoder.unet.*").
	segment_tensors = {k.replace("unet", "base_model.model.transgenic.encoder.unet").replace("uFC", "base_model.model.transgenic.encoder.uFC"):segment_tensors[k] for k in segment_tensors}

	# Filter out keys that belong to the ESM encoder, hidden-mapping layers,
	# lm_head, or FiLM conditioning layers. These are already provided by
	# the decoder checkpoint and should not be overwritten by the segmentation
	# checkpoint.
	newSegment_tensors = {}
	for k in segment_tensors:
		if ("esm" not in k) & ("hidden" not in k) & ("lm_head" not in k) & ("film" not in k):
			newSegment_tensors[k] = segment_tensors[k]

	# Merge dictionaries and load
	# Merge the two dictionaries. If there are overlapping keys the segmentation
	# checkpoint values take precedence (right-hand side of ``**``), but after
	# filtering above there should be no overlap.
	tensors = {**decoder_tensors, **newSegment_tensors}
	model.load_state_dict(tensors)

	return model


def registerModel(hub_name, model):
	"""
	Register TransGenic model classes with HuggingFace Auto* APIs and push to Hub.

	This function performs two tasks:
	  1. Registers the configuration, model, and tokenizer classes so that
	     ``AutoConfig.from_pretrained``, ``AutoModel.from_pretrained``, and
	     ``AutoTokenizer.from_pretrained`` can reconstruct them from the Hub
	     repository without users needing to import TransGenic-specific classes.
	  2. Pushes the model weights and tokenizer files to the specified
	     HuggingFace Hub repository.

	Args:
		hub_name (str): The HuggingFace Hub repository identifier in the form
			``"username/model-name"`` (e.g. ``"jlomas/HyenaTransgenic-768L12A6-400M"``).
		model (transgenicForConditionalGeneration): The model instance whose
			weights will be uploaded. Must already have the desired weights
			loaded.

	Returns:
		None
	"""
	# Register each class so that HuggingFace's Auto* registry knows how to
	# map from config type to the correct model/tokenizer implementation.
	HyenaTransgenicConfig.register_for_auto_class()
	transgenicModel.register_for_auto_class("AutoModel")
	transgenicForConditionalGeneration.register_for_auto_class("AutoModel")
	GFFTokenizer.register_for_auto_class("AutoTokenizer")

	# Instantiate a default tokenizer to push alongside the model.
	tokenizer = GFFTokenizer()

	# Push the model weights to the Hub. ``safe_serialization=False`` saves in
	# the legacy PyTorch bin format rather than safetensors (may be required
	# for compatibility with some downstream consumers).
	model.push_to_hub(hub_name, safe_serialization=False)

	# Push tokenizer files (vocab, config, etc.) to the same Hub repository.
	tokenizer.push_to_hub(hub_name)


# ---------------------------------------------------------------------------
# Example: Load a 768-dim, 12-layer checkpoint and publish to HuggingFace Hub
# ---------------------------------------------------------------------------
if __name__ == "__main__":
	# Path to the safetensors checkpoint produced by a training run with
	# the 768-dim / 12-layer / 6-head architecture (referred to as "768L12").
	generation_checkpoint = "checkpoints/Hyena_Gen9G_6144nt_768L12_E22.safetensors"

	# Number of transformer layers for both encoder and decoder.
	layers = 12

	# Per-layer attention window sizes for the Longformer decoder. Each layer
	# uses a 1024-token sliding window. Only 12 values are active (matching
	# ``layers``); the commented-out line shows how to extend for 18 layers.
	attentionWindow = [
			1024,1024,1024,1024,1024,1024,
			1024,1024,1024,1024,1024,1024,
			#1024,1024,1024,1024,1024,1024
		]

	# Build a config matching the training run's hyperparameters.
	# Segmentation is disabled (do_segment=False) as this is a generation-only
	# model.  Dropout is set to 0 for deterministic inference.
	config = HyenaTransgenicConfig(
	do_segment=False,
	numSegClasses=9,
	d_model=768,
	encoder_layers=layers,
	decoder_layers=layers,
	encoder_n_layer=layers,
	attention_window = attentionWindow,
	dropout=0,
	encoder_attention_heads=6,
	decoder_attention_heads=6
	)

	# Instantiate the model from the config (randomly initialised at this point).
	model = transgenicForConditionalGeneration(config)

	# ---- Load and clean up the generation checkpoint ----
	generation_tensors = {}
	with safe_open(generation_checkpoint, framework="pt", device="cpu") as f:
		for k in f.keys():
			# Skip segmentation-specific weights that may be present in the
			# checkpoint but are not needed for the generation-only model.
			if "segment" not in k:
				# Remove the "_orig_mod." prefix injected by ``torch.compile``
				# so that keys match the eager-mode model's state_dict.
				generation_tensors[k.replace("_orig_mod.", "")] = f.get_tensor(k)

	# ---- Weight tying for the decoder embedding ----
	# Ensure the decoder input embeddings and the output projection head share
	# the same tensor, matching the model's weight-tying scheme.
	generation_tensors["transgenic.decoder_embed_tokens.weight"] = generation_tensors["lm_head.weight"]
	generation_tensors["transgenic.decoder.embed_tokens.weight"] = generation_tensors["transgenic.decoder_embed_tokens.weight"]

	# ---- Duplicate frequency tensors for HyenaDNA positional layers ----
	# The HyenaDNA encoder stores learned frequency parameters at specific
	# positions in its layer hierarchy. The checkpoint may only contain one
	# copy, but the model architecture expects the same frequencies at two
	# sub-layer indices (3 and 5). Duplicate them here so that strict loading
	# succeeds without missing keys.
	freq_tensors = {}
	for k in generation_tensors.keys():
		if "freq" in k:
			# Take the first 9 dotted components of the key (identifying the
			# parent module) and append the sub-layer index + ".freq".
			freq_tensors[".".join(k.split(".")[0:9]) + ".3.freq"] = generation_tensors[k]
			freq_tensors[".".join(k.split(".")[0:9]) + ".5.freq"] = generation_tensors[k]

	# Merge the generation tensors with the duplicated frequency tensors and
	# load everything into the model. ``strict=True`` ensures no keys are
	# missing or unexpected.
	model.load_state_dict(generation_tensors | freq_tensors, strict=True)

	# Push the fully-loaded model (and tokenizer) to HuggingFace Hub.
	registerModel("jlomas/HyenaTransgenic-768L12A6-400M", model)
