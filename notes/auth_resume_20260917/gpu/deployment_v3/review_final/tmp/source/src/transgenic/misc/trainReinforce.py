import wandb
from trl import PPOConfig, PPOTrainer
from torcheval.metrics.functional import bleu_score
from utils_transgenic import *
from modeling_transgenic import *
from safetensors.torch import save_model

def trainReinforce(
		train_ds,
		decoder_checkpoint,
		segment_checkpoint,
		tokenizer,
		batch_size = 1,
		checkpoint_path = "checkpoints_PPO",
		output_dir= "saved_models_PPO",
		log_wandb=True,
		wandb_config={}
):
	# Set device
	device = torch.device("cuda")
	
	# Ensure output directories exist
	if not os.path.exists(checkpoint_path):
		os.makedirs(checkpoint_path)
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	
	# Initialize wandb logging
	if log_wandb:
		wandb.init(project="transgenic", config=wandb_config)

	# Set up DataLoaders
	torch.manual_seed(567)
	torch.cuda.manual_seed_all(567)
	train_ds = makeDataLoader(train_ds, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=4)

	# Load models
	config = TransgenicConfig(do_segment=True, decoder_layerdrop=0)
	model = transgenicForConditionalGeneration(config)
	ref_model = transgenicForConditionalGeneration(config)
	#tokenizer.pad_token = tokenizer.eos_token

	# Load decoder checkpoint
	decoder_tensors = {}
	with safe_open(decoder_checkpoint, framework="pt", device="cpu") as f:
		for k in f.keys():
			decoder_tensors[k] = f.get_tensor(k)
	decoder_tensors["transgenic.decoder_embed_tokens.weight"] = decoder_tensors["lm_head.weight"]
	decoder_tensors["transgenic.decoder.embed_tokens.weight"] = decoder_tensors["transgenic.decoder_embed_tokens.weight"]
	newDecoder_tensors = {}
	for k in decoder_tensors:
		if "ia3" not in k:
			newDecoder_tensors[k] = decoder_tensors[k]

	# Load segmentation checkpoint
	segment_tensors = {}
	with safe_open(segment_checkpoint, framework="pt", device="cpu") as f:
		for k in f.keys():
			segment_tensors[k] = f.get_tensor(k)
	segment_tensors = {k.replace("unet", "transgenic.encoder.unet").replace("uFC", "transgenic.encoder.uFC"):segment_tensors[k] for k in segment_tensors}
	newSegment_tensors = {}
	for k in segment_tensors:
		if ("esm" not in k) & ("hidden" not in k) & ("lm_head" not in k):
			newSegment_tensors[k] = segment_tensors[k]

	# Merge dictionaries and load
	tensors = {**newDecoder_tensors, **newSegment_tensors}
	model.load_state_dict(tensors, strict=False)
	ref_model.load_state_dict(tensors, strict=False)
	ref_model.eval()
	model.train()
	
	model.to(device)
	ref_model.to(device)
	
	for param in model.transgenic.encoder.esm.parameters():
		param.requires_grad = False
	for param in model.transgenic.decoder.parameters():
		param.requires_grad = True
	for param in model.transgenic.encoder.hidden_mapping.parameters():
		param.requires_grad = True
	for param in model.transgenic.encoder.hidden_mapping_layernorm.parameters():
		param.requires_grad = True
	for param in model.transgenic.encoder.uFC.parameters():
		param.requires_grad = False
	for param in model.transgenic.encoder.unet.parameters():
		param.requires_grad = False
	for param in model.transgenic.encoder.unet_mapping.parameters():
		param.requires_grad = False
	for param in model.transgenic.encoder.unet_mapping_layernorm.parameters():
		param.requires_grad = False

	model = TransgenicWithValueHead(model)
	ref_model = TransgenicWithValueHead(ref_model)

	# Initialize PPO trainer
	ppo_config = {"mini_batch_size": 1,"batch_size": 1}
	config = PPOConfig(**ppo_config, log_with='wandb')
	ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer)

	# Set generation kwargs
	generation_kwargs = {
		#"num_return_sequences":1, 
		"max_length":2048,
		"top_k": 0.0,
		"top_p": 1.0,
		"do_sample": True,
		"min_length": -1,
		"early_stopping": False,
		"pad_token_id": 1,
		"bos_token_id":0
	}

	# Training loop
	for epoch in range(num_epochs):
		for step, batch in enumerate(tqdm(train_ds, miniters=10, disable=False)):
			
			# Skippin maize...
			if 'Zm0' in batch[4][0]: 
				continue
				
			# Generate output - TODO: generation is not working properly
			#response_tensor = ppo_trainer.generate([item for item in batch[0].to(model.device)], return_prompt=True, **generation_kwargs)
			response_tensor = model.generate(input_ids = batch[0].to(device), **generation_kwargs)
			pred = dt.batch_decode(response_tensor, skip_special_tokens=True)[0].replace("|</s>", "").replace("</s>", "").replace("<s>", "")
			true = dt.batch_decode(batch[3], skip_special_tokens=True)[0].replace("|</s>", "").replace("</s>", "").replace("<s>", "")

			# Calculate reward
			r = bleu_score(
				pred.replace("|", " ").replace(";", " ").replace(">", " "), 
				[true.replace("|", " ").replace(";", " ").replace(">", " ")]
			)

			# 6. train model with ppo
			stats = ppo_trainer.step([item for item in batch[0]], [item for item in response_tensor], [r])
			ppo_trainer.log_stats(stats, {"query":pred, "response":true}, r)

		save_model(model, f"{checkpoint_path}/model.safetensors")
	
	if log_wandb:
		wandb.finish()


#if __name__ == '__main__':
	
db="Generation_10G_static6144_addExtra200_addRCIsoOnly_clean.db"
dt = GFFTokenizer()
ds = isoformData(db, dt, mode="training", encoder_model="InstaDeepAI/agro-nucleotide-transformer-1b", global_attention=False, shuffle=False)
train_data, eval_data, test_data = torch.utils.data.random_split(ds, [339817, 45309,67964])

decoder_checkpoint = "checkpoints/transgenic_Gen10G_6144nt_E4.safetensors"
segment_checkpoint = "checkpoints_SegmentNT/model.safetensors"
num_epochs = 2
max_grad_norm = 1
checkpoint_path = "checkpoints_PPO"
output_dir = "saved_models_PPO"
notes = """
Training transgenic decoder with PPO, reward: bleu
"""

wandb_config = {
		"loss": "PPO",
		"architecture": "FCG",
		"dataset": db,
		"epochs": num_epochs,
		"max_grad_norm": max_grad_norm,
		"Checkpoints":checkpoint_path,
		"Outputs":output_dir,
		"Notes":notes,
		"decoder_checkpoint": decoder_checkpoint,
		"segment_checkpoint": segment_checkpoint
		}

trainReinforce(
	train_data,
	decoder_checkpoint,
	segment_checkpoint,
	dt,
	batch_size = 1,
	checkpoint_path = checkpoint_path,
	output_dir= output_dir,
	log_wandb=True,
	wandb_config=wandb_config
)