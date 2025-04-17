import re, sys, os
from safetensors import safe_open
from peft import IA3Config, get_peft_model

sys.path.insert(0, f'{os.getcwd()}/src')
from models.modeling_HeynaTransgenic import transgenicForConditionalGeneration, transgenicModel
from models.configuration_transgenic import HyenaTransgenicConfig
from models.tokenization_transgenic import GFFTokenizer

def getModel(config, safetensors_model=None, device="cpu", mode="predict"):
	if not config:
		# Load the model and add to device
		config = HyenaTransgenicConfig()

	model = transgenicForConditionalGeneration(config)

	if mode == "train":
		# Add gradients back for the entire decoder and the hidden mapping layers
		for param in model.transgenic.encoder.esm.parameters():
			param.requires_grad = False
		for param in model.transgenic.decoder.parameters():
			param.requires_grad = True
		for param in model.transgenic.encoder.hidden_mapping.parameters():
			param.requires_grad = True
		for param in model.transgenic.encoder.hidden_mapping_layernorm.parameters():
			param.requires_grad = True
		#model.print_trainable_parameters()
		if f"{device}" != "cpu":
			try:
				model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
			except:
				print("\nNo gradient checkpointing available\n", file=sys.stderr)

	# Load checkpoint if provided
	if safetensors_model:
		tensors = {}
		with safe_open(safetensors_model, framework="pt", device="cpu") as f:
			for k in f.keys():
				tensors[k] = f.get_tensor(k)
		tensors["transgenic.decoder_embed_tokens.weight"] = tensors["lm_head.weight"]
		tensors["transgenic.decoder.embed_tokens.weight"] = tensors["transgenic.decoder_embed_tokens.weight"]
		if "transgenic.decoder_embed_tokens.num_feature_embed.weight" in tensors.keys():
			del tensors["transgenic.decoder_embed_tokens.num_feature_embed.weight"]
		
		newtensors = {k.replace("base_model.model.", "").replace(".base_layer", ""):tensors[k] for k in tensors}
		newnewtensors = {}
		for k in newtensors:
			if "ia3" not in k:
				newnewtensors[k] = newtensors[k]
		model.load_state_dict(newnewtensors)
	
	return model

def getLargeDecoderModel(config, safetensors_model=None, device="cpu", mode="predict"):
	if not config:
		# Load the model and add to device
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
		for param in model.transgenic.decoder.parameters():
			param.requires_grad = True
		if f"{device}" != "cpu":
			try:
				model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
			except:
				print("\nNo gradient checkpointing available\n", file=sys.stderr)

	# Load checkpoint if provided
	if safetensors_model:
		tensors = {}
		with safe_open(safetensors_model, framework="pt", device="cpu") as f:
			for k in f.keys():
				tensors[k] = f.get_tensor(k)
		model.load_state_dict(tensors)
	
	return model

def getPeftModel(decoder_checkpoint, segment_checkpoint, config=None, unlink=False, safetensors_model=None, device="cpu", mode="predict"):
	if not config:
		# Load the model and add to device
		config = HyenaTransgenicConfig()

	model = transgenicForConditionalGeneration(config)

	# Targets all self-attention components and dense linear layers for peft adaptors in the ESM encoder
	target_modules = [
		r".*esm.encoder.layer.*.attention.self.query",
		r".*esm.encoder.layer.*.attention.self.key",
		r".*esm.encoder.layer.*.attention.self.value",
		r".*esm.encoder.layer.*.attention.output.dense",
		r".*esm.encoder.layer.*.intermediate.dense",
		r".*esm.encoder.layer.*.output.dense"]

	# Recompute activations for the dense layers
	feedforward_modules = [
		r".*esm.encoder.layer.*.attention.output.dense",
		r".*esm.encoder.layer.*.intermediate.dense",
		r".*esm.encoder.layer.*.output.dense"]

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

	# Load the IA3 adaptor
	peft_config = IA3Config(task_type="SEQ_2_SEQ_LM", target_modules = peft_targets, feedforward_modules = peft_feedforward, init_ia3_weights = True)
	model = get_peft_model(model, peft_config)

	if mode == "train":
		# Add gradients back for the entire decoder and the hidden mapping layers
		for param in model.transgenic.decoder.parameters():
			param.requires_grad = True
		for param in model.transgenic.encoder.hidden_mapping.parameters():
			param.requires_grad = True
		for param in model.transgenic.encoder.hidden_mapping_layernorm.parameters():
			param.requires_grad = True
		model.print_trainable_parameters()
		if f"{device}" != "cpu":
			try:
				model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
			except:
				print("\nNo gradient checkpointing available\n", file=sys.stderr)

	# Load checkpoint if provided
	# Load decoder checkpoint
	decoder_tensors = {}
	with safe_open(decoder_checkpoint, framework="pt", device="cpu") as f:
		for k in f.keys():
			decoder_tensors[k] = f.get_tensor(k)
	decoder_tensors["base_model.model.lm_head.weight"] = decoder_tensors["base_model.model.transgenic.decoder_embed_tokens.weight"] 
	decoder_tensors["base_model.model.transgenic.decoder.embed_tokens.weight"] = decoder_tensors["base_model.model.transgenic.decoder_embed_tokens.weight"]
	

	# Load segmentation checkpoint
	segment_tensors = {}
	with safe_open(segment_checkpoint, framework="pt", device="cpu") as f:
		for k in f.keys():
			segment_tensors[k] = f.get_tensor(k)
	segment_tensors = {k.replace("unet", "base_model.model.transgenic.encoder.unet").replace("uFC", "base_model.model.transgenic.encoder.uFC"):segment_tensors[k] for k in segment_tensors}
	newSegment_tensors = {}
	for k in segment_tensors:
		if ("esm" not in k) & ("hidden" not in k) & ("lm_head" not in k) & ("film" not in k):
			newSegment_tensors[k] = segment_tensors[k]
	
	# Merge dictionaries and load
	tensors = {**decoder_tensors, **newSegment_tensors}
	model.load_state_dict(tensors)
	
	return model

def registerModel(hub_name, model):

	HyenaTransgenicConfig.register_for_auto_class()
	transgenicModel.register_for_auto_class("AutoModel")
	transgenicForConditionalGeneration.register_for_auto_class("AutoModel")
	GFFTokenizer.register_for_auto_class("AutoTokenizer")
	tokenizer = GFFTokenizer()

	model.push_to_hub(hub_name, safe_serialization=False)
	tokenizer.push_to_hub(hub_name)

if __name__ == "__main__":
	generation_checkpoint = "checkpoints/Hyena_Gen9G_6144nt_768L12_E22.safetensors"
	layers = 12
	attentionWindow = [
			1024,1024,1024,1024,1024,1024,
			1024,1024,1024,1024,1024,1024,
			#1024,1024,1024,1024,1024,1024
		]

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
	model = transgenicForConditionalGeneration(config)

	generation_tensors = {}
	with safe_open(generation_checkpoint, framework="pt", device="cpu") as f:
		for k in f.keys():
			if "segment" not in k:
				generation_tensors[k.replace("_orig_mod.", "")] = f.get_tensor(k)
	generation_tensors["transgenic.decoder_embed_tokens.weight"] = generation_tensors["lm_head.weight"]
	generation_tensors["transgenic.decoder.embed_tokens.weight"] = generation_tensors["transgenic.decoder_embed_tokens.weight"]

	freq_tensors = {}
	for k in generation_tensors.keys():
		if "freq" in k:
			freq_tensors[".".join(k.split(".")[0:9]) + ".3.freq"] = generation_tensors[k]
			freq_tensors[".".join(k.split(".")[0:9]) + ".5.freq"] = generation_tensors[k]

	model.load_state_dict(generation_tensors | freq_tensors, strict=True)

	registerModel("jlomas/HyenaTransgenic-768L12A6-400M", model)
