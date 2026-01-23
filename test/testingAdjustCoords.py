"""
Test coordinate adjustment using combined segmentation and generation models.

Uses segmentation probabilities to refine GSF predictions (start/stop codons,
splice junctions) via PredictionProcessor post-processing.

Output: validation_prediction.gff3, validation_labels.gff3
"""
import torch, sys
from tqdm import tqdm
from safetensors import safe_open

from ..models.configuration_transgenic import HyenaTransgenicConfig
from ..models.modeling_HeynaTransgenic import transgenicForConditionalGeneration
from ..models.tokenization_transgenic import GFFTokenizer
from ..datasets.datasets import makeDataLoader, isoformDataHyena, hyena_collate_fn
from ..utils.postprocess import PredictionProcessor
from ..utils.gsf import gffString2GFF3

torch.manual_seed(123)
generation_checkpoint = "checkpoints/Hyena_Gen9G_6144nt_SinusoidalDownsample_E15.safetensors"
segment_checkpoint = "checkpoints/Hyena_Segment_FocalDice_E0-4.safetensors"

config = HyenaTransgenicConfig(do_segment=True, numSegClasses=9)
model = transgenicForConditionalGeneration(config)

segmentation_tensors = {}
with safe_open(segment_checkpoint, framework="pt", device="cpu") as f:
	for k in f.keys():
		if "segment" in k:
			segmentation_tensors["transgenic.encoder." + k] = f.get_tensor(k)
del segmentation_tensors["transgenic.encoder.segmentation_head.positional_embedding.pe"]

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

model.load_state_dict(segmentation_tensors | generation_tensors | freq_tensors, strict=False)

device = torch.device("cuda")
model.eval()
model.to(device)


db="Generation_10G_static6144_addExtra200_addRCIsoOnly_clean.db"
dt = GFFTokenizer()
ds = isoformDataHyena(db, dt, mode="training", encoder_model="LongSafari/hyenadna-large-1m-seqlen-hf", global_attention=False)
train_data, eval_data, test_data = torch.utils.data.random_split(ds, [339817, 45309,67964])
eval_data = makeDataLoader(eval_data, shuffle=True, batch_size=1, pin_memory=True, num_workers=4, collate_fn=hyena_collate_fn)

def beamSearch(batch, iter=1, maxiter=5):
	ii, am, lab = batch[0].to(device), batch[1].to(device), batch[2].to(device)
	with torch.no_grad():
		#outputs = model(input_ids=ii, attention_mask=am, global_attention_mask=gam, labels=lab, return_dict=True)
		outputs = model.generate(
					inputs=ii, 
					attention_mask=am, 
					num_return_sequences=1, 
					max_length=2048, 
					num_beams=4,
					do_sample=True,
					decoder_input_ids = None
				)
		pred = dt.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0].replace("|</s>", "").replace("</s>", "").replace("<s>", "")
		true = dt.batch_decode(batch[2].detach().cpu().numpy(), skip_special_tokens=True)[0].replace("|</s>", "").replace("</s>", "").replace("<s>", "")
		sequence = ds.encoder_tokenizer.batch_decode(batch[0].detach().cpu().numpy(), skip_special_tokens=True)[0].replace(" ", "")
		probabilities = torch.sigmoid(model.transgenic.encoder.segmentation_logits.detach().cpu()).squeeze()
		pp = PredictionProcessor(pred, sequence, probabilities)
		pp.postProcessPrediction(buff=200, usingSeqFeature=True)
		#extras = pp.analyzeUnused()
		return pp, true

def greedySearch(batch):
	ii, am, gam, lab = batch[0].to(device), batch[1].to(device), batch[2], batch[3].to(device)
	with torch.no_grad():
		#outputs = model(input_ids=ii, attention_mask=am, global_attention_mask=gam, labels=lab, return_dict=True)
		outputs = model.generate(
					inputs=ii, 
					attention_mask=am, 
					num_return_sequences=1, 
					max_length=2048
				)
		pred = dt.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0].replace("|</s>", "").replace("</s>", "").replace("<s>", "")
		true = dt.batch_decode(batch[3].detach().cpu().numpy(), skip_special_tokens=True)[0].replace("|</s>", "").replace("</s>", "").replace("<s>", "")
		sequence = ds.encoder_tokenizer.batch_decode(batch[0].detach().cpu().numpy(), skip_special_tokens=True)[0].replace(" ", "")
		probabilities = torch.nn.functional.softmax(model.transgenic.encoder.segmentation_logits.detach().cpu(), dim=-1)[...,0].squeeze()[0:len(sequence), (0,1,2,3,4,5,6,7,8)]
		pp = PredictionProcessor(pred, sequence, probabilities)
		pp.postProcessPrediction(buff=200)
		return pp, true

with open("validation_prediction.gff3", "w") as predfile:
	with open("validation_labels.gff3", "w") as labfile:
		for step, batch in enumerate(tqdm(eval_data)):
			#if 'TAIR10' not in batch[4][0]: #AT1G02590 'TAIR10'
			#	continue
			#if batch[0].shape[1] <= 1024:
			#	continue
			#try:
			#print(batch[4][0])
			pp, true = beamSearch(batch)
			gff_predictions = gffString2GFF3(pp.stitchGFF(), batch[5][0], batch[6][0], f"GM={batch[4][0]}")
			for line in gff_predictions:
				predfile.write(line + "\n")
			#except:
			#	print(f"Error in beamSearch for GeneModel:{batch[4]} ... trying greedy.\n", file=sys.stdout)
				#try:
				#	pp, true = greedySearch(batch)
				#	gff_predictions = gffString2GFF3(pp.stitchGFF(), batch[5][0], batch[6][0], f"GM={batch[4][0]}")
				#	for line in gff_predictions:
				#		predfile.write(line + "\n")
				#except:
				#	print(f"Error in greedySearch for GeneModel:{batch[4]} ... giving up.\n", file=sys.stdout)

			try:
				gff_labels = gffString2GFF3(true, batch[5][0], batch[6][0], f"GM={batch[4][0]}")
				for line in gff_labels:
					labfile.write(line + "\n")
			except:
				print(f"Error in reference for GeneModel:{batch[4]} ...\n", file=sys.stdout)


print("Done")