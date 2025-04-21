from utils_transgenic import *
from modeling_HeynaTransgenic import transgenicForConditionalGeneration, HyenaEncoder
from configuration_transgenic import TransgenicHyenaConfig

torch.manual_seed(123)
decoder_checkpoint =  "checkpoints_Hyena/model.safetensors"
config = TransgenicHyenaConfig(do_segment=False)
decoder_model = transgenicForConditionalGeneration(config)

# Load decoder checkpoint
decoder_tensors = {}
with safe_open(decoder_checkpoint, framework="pt", device="cpu") as f:
	for k in f.keys():
		if "segmentation" not in k:
			decoder_tensors[k] = f.get_tensor(k)
decoder_tensors["transgenic.decoder_embed_tokens.weight"] = decoder_tensors["lm_head.weight"]
decoder_tensors["transgenic.decoder.embed_tokens.weight"] = decoder_tensors["transgenic.decoder_embed_tokens.weight"]

decoder_model.load_state_dict(decoder_tensors, strict=False)

device = torch.device("cuda")
decoder_model.eval()
decoder_model.to(device)

# Load the model and add to device
segment_checkpoint = "checkpoints_HyenaSegment/model.safetensors"
config = TransgenicHyenaConfig(do_segment=True, numSegClasses=9)
smodel = HyenaEncoder(config)

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
smodel.load_state_dict(new_tensors, strict=False)
smodel.to(device)
smodel.eval()

db="Generation_10G_static6144_addExtra200_addRCIsoOnly_clean.db"
dt = GFFTokenizer()
ds = isoformDataHyena(db, dt, mode="training", encoder_model="LongSafari/hyenadna-large-1m-seqlen-hf", global_attention=False)
train_data, eval_data, test_data = torch.utils.data.random_split(ds, [339817, 45309,67964])
eval_data = makeDataLoader(eval_data, shuffle=True, batch_size=1, pin_memory=True, num_workers=4, collate_fn=hyena_collate_fn)

def beamSearch(batch, iter=1, maxiter=5):
	ii, am, lab = batch[0].to(device), batch[1].to(device), batch[2].to(device)
	with torch.no_grad():
		#outputs = model(input_ids=ii, attention_mask=am, global_attention_mask=gam, labels=lab, return_dict=True)
		outputs = decoder_model.generate(
					inputs=ii, 
					attention_mask=am, 
					num_return_sequences=1, 
					max_length=2048, 
					num_beams=4,
					do_sample=True,
					decoder_input_ids = None
				)
		probabilities = smodel(ii, attention_mask=am)

		pred = dt.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0].replace("|</s>", "").replace("</s>", "").replace("<s>", "")
		true = dt.batch_decode(batch[2].detach().cpu().numpy(), skip_special_tokens=True)[0].replace("|</s>", "").replace("</s>", "").replace("<s>", "")
		sequence = ds.encoder_tokenizer.batch_decode(batch[0].detach().cpu().numpy(), skip_special_tokens=True)[0].replace(" ", "")
		probabilities = torch.nn.functional.softmax(probabilities.segmentation_logits.detach().cpu(), dim=-1).squeeze()[0:len(sequence), (0,1,2,3,4,5,6,7,8)]
		pp = PredictionProcessor(pred, sequence, probabilities)
		stitched = pp.postProcessPrediction(buff=200)
		#extras = pp.analyzeUnused()
		return stitched, true
		#return pred, true

def greedySearch(batch):
	ii, am, lab = batch[0].to(device), batch[1].to(device), batch[2].to(device)
	with torch.no_grad():
		#outputs = model(input_ids=ii, attention_mask=am, global_attention_mask=gam, labels=lab, return_dict=True)
		outputs = decoder_model.generate(
					inputs=ii, 
					attention_mask=am, 
					num_return_sequences=1, 
					max_length=2048
				)
		pred = dt.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0].replace("|</s>", "").replace("</s>", "").replace("<s>", "")
		true = dt.batch_decode(batch[3].detach().cpu().numpy(), skip_special_tokens=True)[0].replace("|</s>", "").replace("</s>", "").replace("<s>", "")
		#sequence = ds.encoder_tokenizer.batch_decode(batch[0].detach().cpu().numpy(), skip_special_tokens=True)[0].replace(" ", "")
		#probabilities = torch.nn.functional.softmax(model.transgenic.encoder.segmentation_logits.detach().cpu(), dim=-1)[...,0].squeeze()[0:len(sequence), (0,1,2,3,4,5,6,7,8)]
		#pp = PredictionProcessor(pred, sequence, probabilities)
		#pp.postProcessPrediction(buff=200)
		#return pp, true
		return pred, true

with open("validation_hyena_prediction.gff3", "w") as predfile:
	with open("validation_hyena_labels.gff3", "w") as labfile:
		for step, batch in enumerate(tqdm(eval_data)):
			#if 'TAIR10' not in batch[3][0]: #AT1G02590 'TAIR10'
			#	continue
			try:
				#print(batch[4][0])
				pp, true = beamSearch(batch)
				#gff_predictions = gffString2GFF3(pp.stitchGFF(), batch[5][0], batch[6][0], f"GM={batch[4][0]}")
				gff_predictions = gffString2GFF3(pp, batch[5][0], batch[6][0], f"GM={batch[3][0]}")
				for line in gff_predictions:
					predfile.write(line + "\n")
			except:
				print(f"Error in beamSearch for GeneModel:{batch[3]} ... trying greedy.\n", file=sys.stdout)
				try:
					pp, true = greedySearch(batch)
			#		gff_predictions = gffString2GFF3(pp.stitchGFF(), batch[5][0], batch[6][0], f"GM={batch[4][0]}")
					gff_predictions = gffString2GFF3(pp, batch[5][0], batch[6][0], f"GM={batch[3][0]}")
					for line in gff_predictions:
						predfile.write(line + "\n")
				except:
					print(f"Error in greedySearch for GeneModel:{batch[3]} ... giving up.\n", file=sys.stdout)

			try:
				gff_labels = gffString2GFF3(true, batch[5][0], batch[6][0], f"GM={batch[3][0]}")
				for line in gff_labels:
					labfile.write(line + "\n")
			except:
				print(f"Error in reference for GeneModel:{batch[4]} ...\n", file=sys.stdout)


print("Done")