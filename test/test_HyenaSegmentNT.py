"""
Test segmentation model performance using HyenaDNA encoder.

Evaluates multilabel classification metrics (Precision, Recall, F1, MCC) for
genomic feature prediction: Gene, Start_Codon, Exon, Intron, Splice Donor/Acceptor,
5'-UTR, 3'-UTR, Stop_Codon.

Output: HyenaSegmentPerformance.out (TSV with per-sample metrics)
"""
import torch, sys
from tqdm import tqdm
from safetensors import safe_open
from torchmetrics.classification import MultilabelPrecision, MultilabelRecall, MultilabelF1Score, MultilabelMatthewsCorrCoef, BinaryMatthewsCorrCoef

from ..datasets.datasets import segmentationDataset, preprocessedSegmentationDatasetHyena, makeDataLoader, hyena_segment_collate_fn
from ..models.tokenization_transgenic import GFFTokenizer
from ..models.modeling_HeynaTransgenic import HyenaEncoder 
from ..models.configuration_transgenic import HyenaTransgenicConfig


def predictTransgenicAccelerate(
		encoder_model:str, 
		segmentation_model:str, 
		safetensors_model:str, 
		database:segmentationDataset, 
		outfile="HyenaSegmentPerformance.out", 
		batch_size=1,
		window_size=12288,
		step_size=11000):
	
	# apply heaeder to output file
	features = ["Gene", "Start_Codon", "Exon", "Intron", "SDonor", "SAcceptor", "UTR5", "UTR3", "Stop_Codon"]
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

	device = torch.device("cuda")
	print(f"Running transgenic in prediction mode on {device}", file=sys.stderr)
	
	# Set up DataLoader
	dt = GFFTokenizer()
	ds  = preprocessedSegmentationDatasetHyena("Segmentation_9Genomes_preprocessed_scodons.db")
	train_data, eval_data, test_data = torch.utils.data.random_split(ds, [534331, 71244,106867])
	dataset = makeDataLoader(test_data, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=4, collate_fn=hyena_segment_collate_fn)

	# Load the model and add to device
	segment_checkpoint = "checkpoints/Hyena_SegmentFocalDice_E13-21.safetensors"#"checkpoints/Hyena_Segment_FocalDice_E0-4.safetensors"
	config = HyenaTransgenicConfig(do_segment=True, numSegClasses=9)
	model = HyenaEncoder(config)

	tensors = {}
	with safe_open(segment_checkpoint, framework="pt", device="cpu") as f:
		for k in f.keys():
			tensors[k] = f.get_tensor(k)
	#tensors["transgenic.decoder_embed_tokens.weight"] = tensors["lm_head.weight"]
	#tensors["transgenic.decoder.embed_tokens.weight"] = tensors["transgenic.decoder_embed_tokens.weight"]
	tensors = {key.replace("transgenic.encoder.", ""):tensors[key] for key in tensors}
	new_tensors = {}
	for key in tensors:
		if "transgenic.decoder." not in key:
			new_tensors[key] = tensors[key]
	#del new_tensors["segmentation_head.positional_embedding.pe"]

	freq_tensors = {}
	for k in tensors.keys():
		if "freq" in k:
			freq_tensors[".".join(k.split(".")[0:7]) + ".3.freq"] = tensors[k]
			freq_tensors[".".join(k.split(".")[0:7]) + ".5.freq"] = tensors[k]

	model.load_state_dict(new_tensors | freq_tensors, strict=False)
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
			outputs = model(batch[0].to(device), attention_mask=batch[1].to(device), segLabels=batch[2][:, :, 0:9].to(device))
		#probabilities =  torch.nn.functional.softmax(outputs.seg_logits, dim=-1)[...,0].squeeze()
				
		predictions = torch.sigmoid(outputs.segmentation_logits).squeeze().cpu()
		labels = batch[2][:, :, 0:9].detach().cpu().squeeze().int()
		
		#seq = ds.encoder_tokenizer.batch_decode(batch[0].detach().cpu().numpy(), skip_special_tokens=True)[0].replace(" ", "")
		#import matplotlib.pyplot as plt
		#plt.figure(figsize=(10, 4))
		#plt.bar(range(6144), labels[:,3].numpy())
		#plt.bar(range(6144), predictions[:,3].numpy())
		#plt.ylim(0.5, 1)
		#plt.savefig("plot.png")
		#plt.close()
		
		mlp = MultilabelPrecision(num_labels=9, average=None)(predictions, labels).tolist() # False positive rate
		mlr = MultilabelRecall(num_labels=9, average=None)(predictions, labels).tolist() # False negative rate
		mlf1 = MultilabelF1Score(num_labels=9, average=None)(predictions, labels).tolist()
		mlmcc = MultilabelMatthewsCorrCoef(num_labels=9)(predictions, labels).tolist()
		mcc = []
		for i in range(9):
			mcc.append(BinaryMatthewsCorrCoef()(predictions[:,i], labels[:,i]).item())

		with open(outfile, "a") as f:
			print(f"{batch[3][0]}\t{batch[4][0]}\t{str(batch[5][0])}\t{str(batch[6][0])}\t{"\t".join([str(i) for i in mlp])}\t{"\t".join([str(i) for i in mlr])}\t{"\t".join([str(i) for i in mlf1])}\t{"\t".join([str(i) for i in mcc])}\t{str(mlmcc)}\t{str(int(genic))}", file=f)

if __name__ == "__main__":
	torch.manual_seed(123)
	encoder_model = "LongSafari/hyenadna-large-1m-seqlen-hf"
	segmentation_model = None
	safetensors_model = "checkpoints_HyenaSegment/model.safetensors"
	database = "Segmentation_9Genomes_preprocessed_scodons.db"

	predictTransgenicAccelerate(
		encoder_model, 
		segmentation_model, 
		safetensors_model, 
		database)
