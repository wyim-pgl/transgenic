import torch, sys
from tqdm import tqdm
from safetensors import safe_open
from torchmetrics.classification import MultilabelPrecision, MultilabelRecall, MultilabelF1Score, MultilabelMatthewsCorrCoef, BinaryMatthewsCorrCoef

from ..datasets.datasets import preprocessedSegmentationDataset, makeDataLoader, segment_collate_fn
from ..models.modeling_NTTransgenic import segmented_sequence_embeddings

def predictTransgenicAccelerate(
		encoder_model:str, 
		segmentation_model:str, 
		safetensors_model:str, 
		database:preprocessedSegmentationDataset, 
		outfile="SegmentNTPerformance.out", 
		batch_size=1,
		window_size=12288,
		step_size=11000):
	
	# apply heaeder to output file
	features = ["Gene", "Start_Codon", "Exon", "Intron", "SDonor", "SAcceptor", "UTR5", "UTR3","Stop_Codon"]
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
	#accelerator = Accelerator()
	device = torch.device("cuda")
	print(f"Running transgenic in prediction mode on {device}", file=sys.stderr)
	
	# Set up DataLoader
	ds  = preprocessedSegmentationDataset(database)
	_, eval_data, test_data = torch.utils.data.random_split(ds, [534331, 71244,106867])
	dataset =  makeDataLoader(test_data, shuffle=True, batch_size=1, pin_memory=True, num_workers=1, collate_fn=segment_collate_fn)

	# Load the model and add to device
	model = segmented_sequence_embeddings(encoder_model, segmentation_model, 14, do_segment=True)

	if safetensors_model:
		tensors = {}
		with safe_open(safetensors_model, framework="pt", device="cpu") as f:
			for k in f.keys():
				if "film" not in k:
					tensors[k] = f.get_tensor(k)
		model.load_state_dict(tensors)
	model.eval()
	model.to(device)
	
	# Prep objects for use with accelerator
	#model, dataset = accelerator.prepare(
	#	model, dataset
	#)

	# Prediction loop
	for batch in tqdm(dataset):
		if batch[2][:,:,0].sum() == 0:	
			genic = False
		else: 
			genic = True
		with torch.no_grad():
			outputs = model(batch[0].to(device), attention_mask=batch[1].to(device), segLabels=batch[2].to(device))
		#probabilities =  torch.nn.functional.softmax(outputs.seg_logits, dim=-1)[...,0].squeeze()
				
		predictions = (outputs.seg_logits[..., 0] > outputs.seg_logits[..., 1]).detach().cpu().long().squeeze()
		predictions = predictions[:, (0,1,2,3,4,5,6,7,8)]
		labels = batch[2][:, :, (0,1,2,3,4,5,6,7,8)].detach().cpu().squeeze().int()
		
		mlp = MultilabelPrecision(num_labels=9, average=None)(predictions, labels).tolist() # False positive rate
		mlr = MultilabelRecall(num_labels=9, average=None)(predictions, labels).tolist() # False negative rate
		mlf1 = MultilabelF1Score(num_labels=9, average=None)(predictions, labels).tolist()
		mlmcc = MultilabelMatthewsCorrCoef(num_labels=9)(predictions, labels).tolist()
		mcc = []
		for i in range(7):
			mcc.append(BinaryMatthewsCorrCoef()(predictions[:,i], labels[:,i]).item())

		with open(outfile, "a") as f:
			print(f"{batch[3][0]}\t{batch[4][0]}\t{str(batch[5][0])}\t{str(batch[6][0])}\t{"\t".join([str(i) for i in mlp])}\t{"\t".join([str(i) for i in mlr])}\t{"\t".join([str(i) for i in mlf1])}\t{"\t".join([str(i) for i in mcc])}\t{str(mlmcc)}\t{str(int(genic))}", file=f)

if __name__ == "__main__":
	torch.manual_seed(123)
	encoder_model = "InstaDeepAI/agro-nucleotide-transformer-1b"
	segmentation_model = "InstaDeepAI/segment_nt_multi_species"
	safetensors_model = "checkpoints/AgroSegmentNT_Epoch6_6144nt_restart_codons.safetensors"
	database = "Segmentation_9Genomes_preprocessed_scodons.db"

	predictTransgenicAccelerate(
		encoder_model, 
		segmentation_model, 
		safetensors_model, 
		database)
