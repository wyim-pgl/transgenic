from utils_transgenic import *
from modeling_HeynaTransgenic import *


import matplotlib as plt
import numpy as np

torch.manual_seed(123)
#"checkpoints/Hyena_Gen9G_6144nt_SinusoidalDownsample_E15.safetensors"
generation_checkpoint = "checkpoints/Hyena_Gen9G_6144nt_wide_E12.safetensors"
#"checkpoints/Hyena_SegmentFocalDice_E13-21.safetensors" 
segment_checkpoint = "checkpoints/Hyena_SegmentFocalDice_Wide_E5.safetensors"

#config = TransgenicHyenaConfig(do_segment=True, numSegClasses=9)
config = TransgenicHyenaConfig(
	do_segment=True, 
	numSegClasses=9,
	d_model=512,
	encoder_layers=9,
	decoder_layers=9,
	encoder_n_layer=9,
	attention_window = [
			1024,1024,1024,1024,1024,1024,
			1024,1024,1024
		])
model = transgenicForConditionalGeneration(config)

segmentation_tensors = {}
with safe_open(segment_checkpoint, framework="pt", device="cpu") as f:
	for k in f.keys():
		if "segment" in k:
			segmentation_tensors["transgenic.encoder." + k] = f.get_tensor(k)
#del segmentation_tensors["transgenic.encoder.segmentation_head.positional_embedding.pe"]

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

device = torch.device("cuda")
model.eval()
model.to(device)


db="Generation_10G_static6144_addExtra200_addRCIsoOnly_clean.db"
dt = GFFTokenizer()
ds = isoformDataHyena(db, dt, mode="training", encoder_model="LongSafari/hyenadna-large-1m-seqlen-hf", global_attention=False)
train_data, eval_data, test_data = torch.utils.data.random_split(ds, [339817, 45309,67964])
test_data = makeDataLoader(test_data, shuffle=True, batch_size=1, pin_memory=True, num_workers=4, collate_fn=hyena_collate_fn)

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
		#pp = PredictionProcessor(pred, sequence, probabilities)
		#pp.postProcessPrediction(buff=200, usingSeqFeature=True)
		return pred, true, sequence, probabilities

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
		#pp = PredictionProcessor(pred, sequence, probabilities)
		#pp.postProcessPrediction(buff=200)
		return pred, true, sequence, probabilities

predfile = open("hyenaTest_prediction_noPost.gff3", "w")	# Raw generative predictions
procPredFile = open("hyenaTest_prediction_post.gff3", "w") # Post-processed generative predictions
fullLabFile =  open("hyenaTest_labels.gff3", "w")			# Target annotations of all tested annotations
successLabFile = open("hyenaTest_labels_success.gff3", "w")# Target annotations of parsable generative predictions
for step, batch in enumerate(tqdm(test_data)):
	# Generate predictions and segmentation
	try:
		pred, true, sequence, probabilities = beamSearch(batch)
	except:
		print(f"Error in beamSearch for GeneModel: {batch[3][0]}")
	
	#Write full lab file
	try:
		gff_labels = gffString2GFF3(true, batch[4][0], batch[5][0], f"GM={batch[3][0]}")
		for line in gff_labels:
			fullLabFile.write(line + "\n")
	except:
		print(f"Error in reference for GeneModel:{batch[3]} ...\n", file=sys.stdout)
	
	# Write raw prediction and success labfile if raw prediction is parsable
	try:
		gff_predictions = gffString2GFF3(pred, batch[4][0], batch[5][0], f"GM={batch[3][0]}")
		for line in gff_predictions:
			predfile.write(line + "\n")
		gff_labels = gffString2GFF3(true, batch[4][0], batch[5][0], f"GM={batch[3][0]}")
		for line in gff_labels:
			successLabFile.write(line + "\n")
		generation = True
	except:
		generation = False
		print(f"Error writing raw prediction for GeneModel:{batch[3]} ...\n", file=sys.stdout)
	
	# Process predictions and write
	if generation:
		try:
			pp = PredictionProcessor(pred, sequence, probabilities)
			pp.postProcessPrediction(utr_splice_buf=100, splice_buffer=50, start_buffer=20, stop_buffer=20, seqFeature=True)
			gff_predictions_post = gffString2GFF3(pp.stitchGFF(), batch[4][0], batch[5][0], f"GM={batch[3][0]}")
			for line in gff_predictions_post:
				procPredFile.write(line + "\n")
		except:
			print(f"Error in processor for GeneModel:{batch[3]}.\n", file=sys.stdout)

predfile.close()
procPredFile.close()
fullLabFile.close()
successLabFile.close()


print("Done")