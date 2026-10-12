import torch, sys
from tqdm import tqdm
from safetensors import safe_open
from utils_transgenic import segmentationDataset, makeDataLoader
from scipy.ndimage import gaussian_filter1d, binary_dilation, binary_erosion, label
from utils_transgenic import *
from modeling_HeynaTransgenic import transgenicForConditionalGeneration
from configuration_transgenic import TransgenicHyenaConfig

def predictDeNovoTransgenic(
		encoder_model:str, 
		generation_checkpoint:str, 
		segment_checkpoint:str, 
		database:segmentationDataset,  
		batch_size=1):

	# Set device
	device = torch.device("cuda")
	print(f"Running transgenic in de novo prediction mode on {device}", file=sys.stderr)
	
	# Set up DataLoader
	dt = GFFTokenizer()
	ds  = preprocessedSegmentationDatasetHyena(database)
	train_data, eval_data, test_data = torch.utils.data.random_split(ds, [534331, 71244,106867])
	dataset = makeDataLoader(eval_data, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=4, collate_fn=hyena_segment_collate_fn)

	# Load the model and add to device
	config = TransgenicHyenaConfig(do_segment=True, numSegClasses=9)
	model = transgenicForConditionalGeneration(config)

	segmentation_tensors = {}
	with safe_open(segment_checkpoint, framework="pt", device="cpu") as f:
		for k in f.keys():
			if "segment" in k:
				segmentation_tensors["transgenic.encoder." + k] = f.get_tensor(k)
	
	generation_tensors = {}
	with safe_open(generation_checkpoint, framework="pt", device="cpu") as f:
		for k in f.keys():
			if "segment" not in k:
				generation_tensors[k] = f.get_tensor(k)
	generation_tensors["transgenic.decoder_embed_tokens.weight"] = generation_tensors["lm_head.weight"]
	generation_tensors["transgenic.decoder.embed_tokens.weight"] = generation_tensors["transgenic.decoder_embed_tokens.weight"]

	freq_tensors = {}
	for k in generation_tensors.keys():
		if "freq" in k:
			freq_tensors[".".join(k.split(".")[0:9]) + ".3.freq"] = generation_tensors[k]
			freq_tensors[".".join(k.split(".")[0:9]) + ".5.freq"] = generation_tensors[k]

	model.load_state_dict(segmentation_tensors | generation_tensors | freq_tensors, strict=True)
	model.to(device)
	model.eval()

	# Prediction loop
	for batch in tqdm(dataset):
		if batch[2][:, :, 0].sum() == 0:	
			genic = False
		else: 
			genic = True
		
		#if genic:
		with torch.no_grad():
			encoder_outputs = model.transgenic.encoder(batch[0].to(device), attention_mask=batch[1].to(device), segLabels=batch[2][:, :, 0:9].to(device))
		
		predictions = torch.sigmoid(encoder_outputs.segmentation_logits).squeeze().cpu()
		gene_classes = classifyGeneSignal(predictions[:,0], threshold=0.65, sigma=100)
				
		
		labels = batch[2][:, :, 0:9].detach().cpu().squeeze().int()
		
		seq = ds.encoder_tokenizer.batch_decode(batch[0].detach().cpu().numpy(), skip_special_tokens=True)[0].replace(" ", "")
#import matplotlib.pyplot as plt
#plt.figure(figsize=(10, 4))
#plt.bar(range(6144), labels[:,0].numpy())
#plt.bar(range(6144), gene_classes[1].astype(int))
#plt.bar(range(6144), predictions[:,0].numpy())
#plt.ylim(0.5, 1)
#plt.savefig("plot.png")
#plt.close()
		print(batch[3])
		
		
if __name__ == "__main__":
	torch.manual_seed(123)
	encoder_model = "LongSafari/hyenadna-large-1m-seqlen-hf"
	generation_checkpoint = "checkpoints/Hyena_Gen9G_6144nt_SinusoidalDownsample_E15.safetensors"
	segment_checkpoint = "checkpoints/Hyena_Segment_FocalDice_E0-4.safetensors"
	database = "Segmentation_9Genomes_preprocessed_scodons.db"

	predictDeNovoTransgenic(
		encoder_model, 
		generation_checkpoint,
		segment_checkpoint, 
		database,
		batch_size=1)
