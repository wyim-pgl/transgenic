import os
from torch.utils.data import random_split
from accelerate import Accelerator
from utils_transgenic import *

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['HF_HOME'] = './HFmodels'
os.environ["NCCL_DEBUG"] = "INFO"


def predictTransgenicAccelerate(encoder_model:str, safetensors_model:str, dataset:isoformData, outfile="transgenic.out", batch_size=1, generation_mode="greedy"):
	# Set up accelerator
	accelerator = Accelerator()
	device = accelerator.device
	print(f"Running transgenic in prediction mode on {device}", file=sys.stderr)
	
	# Set up DataLoader
	dataset = makeDataLoader(dataset, shuffle=False, batch_size=batch_size, pin_memory=True, num_workers=4)
	
	# Load the model and add to device
	model = getPeftModel(encoder_model, config=None, unlink=False, safetensors_model=safetensors_model, device=device, mode="train")
	model.to(device)
	model.eval()
	
	# Prep objects for use with accelerator
	model, dataset = accelerator.prepare(
		model, dataset
	)

	# Prediction loop
	dt = GFFTokenizer()
	predictions = []
	for step, batch in enumerate(tqdm(dataset)):
		with torch.no_grad():
			if generation_mode == "greedy":
				outputs = model.generate(
					inputs=batch[0], 
					attention_mask=batch[1], 
					num_return_sequences=1, 
					max_length=2048
				)
			elif generation_mode == "beam":
				outputs = model.generate(
					inputs=batch[0], 
					attention_mask=batch[1], 
					num_return_sequences=1, 
					max_length=2048, 
					num_beams=4,
					do_sample=True #,length_penalty=2.0  # Length penalty to avoid short sequences
				)
			elif generation_mode == "contrastive":
				outputs = model.module.generate(
					inputs=batch[0], 
					attention_mask=batch[1], 
					max_length=2048,
					num_return_sequences=1,
					do_sample=True,
					penalty_alpha=0.6, 
					top_k=4
				)
		pred = dt.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)
		true = dt.batch_decode(batch[3].reshape(batch_size,batch[3].size()[-1]).detach().cpu().numpy(), skip_special_tokens=True)
		predictions.append([batch[3], outputs.detach().cpu(), true[0], pred[0], batch[4]])	
	
		with open(f"{device}_{outfile}.txt", 'w') as out:
			for prediction in predictions:
				out.write("\t".join(str(prediction))+"\n")

		with open(f"{device}_{outfile}.pkl", 'wb') as out:
			pickle.dump(predictions, out)

if __name__ == '__main__':

	torch.manual_seed(123)

	with open("Train-Flagship_Genomes_49k_extra200_addRCIsoOnly.pkl", 'rb') as f:
		train_data = pickle.load(f)
	with open("Eval-Flagship_Genomes_49k_extra200_addRCIsoOnly.pkl", 'rb') as f:
		eval_data = pickle.load(f)
	with open("Test-Flagship_Genomes_49k_extra200_addRCIsoOnly.pkl", 'rb') as f:
		test_data = pickle.load(f)

	#InstaDeepAI/nucleotide-transformer-v2-500m-multi-species
	#InstaDeepAI/agro-nucleotide-transformer-1b
	mode = sys.argv[1]
	encoder_model = sys.argv[2]
	unlink = bool(sys.argv[3])
	notes = sys.argv[4]
	print(f"Running in {mode} mode", file=sys.stderr)
	
	predictTransgenicAccelerate(
		"InstaDeepAI/agro-nucleotide-transformer-1b",
		"checkpoints_ESMpeftReal_local09/model.safetensors", 
		eval_data, 
		outfile= mode+"Search.out", 
		batch_size=1,
		generation_mode=mode
	)