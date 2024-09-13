import torch, sys
from tqdm import tqdm
from safetensors import safe_open
from accelerate import Accelerator
from torchmetrics.classification import MultilabelPrecision, MultilabelRecall, MultilabelF1Score, MultilabelMatthewsCorrCoef, BinaryMatthewsCorrCoef
from utils_transgenic import segmentationDataset, makeDataLoader, segment_collate_fn
from modeling_transgenic import segmented_sequence_embeddings

def predictTransgenicAccelerate(
		encoder_model:str, 
		segmentation_model:str, 
		safetensors_model:str, 
		database:segmentationDataset, 
		outfile="SegmentNTPerformance.out", 
		batch_size=1,
		window_size=12288,
		step_size=11000):
	
	# apply heaeder to output file
	features = ["Gene", "Exon", "Intron", "SDonor", "SAcceptor", "UTR5", "UTR3"]
	precision = []
	recall = []
	f1 = []
	mcc = []
	for feature in features:
		precision.append(f"Precision_{feature}")
		recall.append(f"Recall_{feature}")
		f1.append(f"F1_{feature}")
		mcc.append(f"MCC_{feature}")
	with open(outfile, "w") as f:
		print(f"Organism\tChromosome\tStart\tEnd\t{'\t'.join(precision)}\t{'\t'.join(recall)}\t{'\t'.join(f1)}\t{'\t'.join(mcc)}\tMLMCC\tGenic", file=f)

	# Set up accelerator
	accelerator = Accelerator()
	device = accelerator.device
	print(f"Running transgenic in prediction mode on {device}", file=sys.stderr)
	
	# Set up DataLoader
	ds  = segmentationDataset(window_size, step_size, database)
	_, eval_data, _ = torch.utils.data.random_split(ds, [231003, 30800, 46202])
	dataset =  makeDataLoader(eval_data, shuffle=True, batch_size=1, pin_memory=True, num_workers=4, collate_fn=segment_collate_fn)

	
	# Load the model and add to device
	model = segmented_sequence_embeddings(encoder_model, segmentation_model, 14)
	
	if safetensors_model:
		tensors = {}
		with safe_open(safetensors_model, framework="pt", device="cpu") as f:
			for k in f.keys():
				tensors[k] = f.get_tensor(k)
		model.load_state_dict(tensors)
	model.eval()
	
	# Prep objects for use with accelerator
	model, dataset = accelerator.prepare(
		model, dataset
	)

	# Prediction loop
	for batch in tqdm(dataset):
		if batch[2][:,0].sum() == 0:	
			genic = False
		else: 
			genic = True

		outputs = model(batch[0], attention_mask=batch[1], segLabels=batch[2])
		#probabilities =  torch.nn.functional.softmax(outputs.seg_logits, dim=-1)[...,0].squeeze()
		
		predictions = (outputs.seg_logits[..., 0] > outputs.seg_logits[..., 1]).long().squeeze()
		predictions = predictions[:, (0,2,3,4,5,6,7)]
		labels = batch[2][:, (0,2,3,4,5,6,7)].squeeze().int()

		mlp = MultilabelPrecision(num_labels=7, average=None)(predictions, labels).tolist() # False positive rate
		mlr = MultilabelRecall(num_labels=7, average=None)(predictions, labels).tolist() # False negative rate
		mlf1 = MultilabelF1Score(num_labels=7, average=None)(predictions, labels).tolist()
		mlmcc = MultilabelMatthewsCorrCoef(num_labels=7)(predictions, labels).tolist()
		mcc = []
		for i in range(7):
			mcc.append(BinaryMatthewsCorrCoef()(predictions[:,i], labels[:,i])).item()

		with open(outfile, "a") as f:
			print(f"{batch[3]}\t{batch[4]}\t{batch[5]}\t{batch[6]}\t{"\t".join(mlp)}\t{"\t".join(mlr)}\t{"\t".join(mlf1)}\t{"\t".join(mcc)}\t{mlmcc}\t{genic}", file=f)

if __name__ == "__main__":
	torch.manual_seed(123)
	encoder_model = "InstaDeepAI/agro-nucleotide-transformer-1b"
	segmentation_model = "InstaDeepAI/segment_nt_multi_species"
	safetensors_model = "checkpoints_SegmentNT/model.safetensors"
	database = "Segmentation_7Genomes.db"

	predictTransgenicAccelerate(
		encoder_model, 
		segmentation_model, 
		safetensors_model, 
		database)