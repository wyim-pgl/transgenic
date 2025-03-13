import copy
from utils_transgenic import *
from modeling_HeynaTransgenic import transgenicForConditionalGeneration
from configuration_transgenic import TransgenicHyenaConfig

torch.manual_seed(123)
decoder_checkpoint =  "checkpoints/Hyena_Gen9G_6144nt_E30.safetensors"

config = TransgenicHyenaConfig(do_segment=False)
model = transgenicForConditionalGeneration(config)

# Load decoder checkpoint
decoder_tensors = {}
with safe_open(decoder_checkpoint, framework="pt", device="cpu") as f:
	for k in f.keys():
		decoder_tensors[k] = f.get_tensor(k)
decoder_tensors["transgenic.decoder_embed_tokens.weight"] = decoder_tensors["lm_head.weight"]
decoder_tensors["transgenic.decoder.embed_tokens.weight"] = decoder_tensors["transgenic.decoder_embed_tokens.weight"]

model.load_state_dict(decoder_tensors, strict=False)

device = torch.device("cuda")
model.eval()
model.to(device)


db="Generation_10G_static6144_addExtra200_addRCIsoOnly_clean.db"
dt = GFFTokenizer()
ds = isoformDataHyena(db, dt, mode="training", encoder_model="LongSafari/hyenadna-large-1m-seqlen-hf", global_attention=False)
train_data, eval_data, test_data = torch.utils.data.random_split(ds, [339817, 45309,67964])
train_data = makeDataLoader(train_data, shuffle=True, batch_size=1, pin_memory=True, num_workers=4, collate_fn=hyena_collate_fn)
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
		#sequence = ds.encoder_tokenizer.batch_decode(batch[0].detach().cpu().numpy(), skip_special_tokens=True)[0].replace(" ", "")
		#probabilities = torch.nn.functional.softmax(model.transgenic.encoder.segmentation_logits.detach().cpu(), dim=-1)[...,0].squeeze()[0:len(sequence), (0,1,2,3,4,5,6,7,8)]
		#probabilities = None
		#pp = PredictionProcessor(pred, sequence, probabilities)
		#stitched = pp.postProcessPrediction(buff=200)
		#extras = pp.analyzeUnused()
		#return stitched, true
		return pred, true

def greedySearch(batch):
	ii, am, lab = batch[0].to(device), batch[1].to(device), batch[2].to(device)
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
		#sequence = ds.encoder_tokenizer.batch_decode(batch[0].detach().cpu().numpy(), skip_special_tokens=True)[0].replace(" ", "")
		#probabilities = torch.nn.functional.softmax(model.transgenic.encoder.segmentation_logits.detach().cpu(), dim=-1)[...,0].squeeze()[0:len(sequence), (0,1,2,3,4,5,6,7,8)]
		#pp = PredictionProcessor(pred, sequence, probabilities)
		#pp.postProcessPrediction(buff=200)
		#return pp, true
		return pred, true

def createDenoisingLabel(pred, true):
	# Parse GSF strings
	true_features = [feat.split("|") for feat in true.split(">")[0].split(";")]
	true_transcripts = [set(feat.split("|")) for feat in true.split(">")[1].split(";")]
	pred_features = [feat.split("|") for feat in pred.split(">")[0].split(";")]
	pred_transcripts = [set(feat.split("|")) for feat in pred.split(">")[1].split(";")]
	
	# Set up feature index dictionaries for lookup
	true_fid = {}
	for i, feature in enumerate(true_features):
		true_fid[feature[1]] = i

	pred_fid = {}
	for i, feature in enumerate(pred_features):
		pred_fid[feature[1]] = i
	
	# Identify structurally matched transcripts
	matches = []
	match_type = None
	for i,p in enumerate(pred_transcripts):
		for j, t in enumerate(true_transcripts):
			if p == t:
				matches.append([i,j])
				if match_type != "CDS":
					match_type = "Full"
			else:
				p = {x for x in p if "CDS" in x}
				t = {x for x in t if "CDS" in x}
				if p==t:
					matches.append([i,j])
					match_type = "CDS"
	
	if matches == []:
		return None

	# Create label (corrected prediction)
	label_features = copy.deepcopy(pred_features)
	mask_features = []
	for p,t in matches:
		for feat in true_transcripts[t]:
			if (match_type == "CDS") & ("CDS" not in feat):
				continue
			label_features[pred_fid[feat]][0] = true_features[true_fid[feat]][0] # Start coord
			label_features[pred_fid[feat]][2] = true_features[true_fid[feat]][2] # End coord
			label_features[pred_fid[feat]][4] = true_features[true_fid[feat]][4] # Phase
			mask_features.append(feat)
	mask_features = set(mask_features)
	
	# Re-tokenize
	label = ";".join(["|".join(i) for i in label_features]) + ">" + pred.split(">")[1]
	tokenized_label = dt.batch_encode_plus(
		[label],
		return_tensors="pt",
		padding=True,
		truncation=True,
		add_special_tokens=True,
		max_length=2048)
	
	# Create mask for features with ground truth labels
	tokenized_mask_features = torch.tensor(dt.encode("|".join(list(mask_features)))[1:-1])
	tokens_to_mask_inverse = torch.tensor(list(range(4,17)))
	
	feature_transcript_split =  torch.nonzero(tokenized_label["input_ids"] == 17, as_tuple=True)[1][0]
	splits_indices = torch.nonzero(tokenized_label["input_ids"] == 21, as_tuple=True)[1]
	splits_indices = splits_indices[splits_indices < feature_transcript_split]

	mlm_mask = copy.deepcopy(tokenized_label["attention_mask"]) == False
	prev_split = 0
	for split in splits_indices:
		id_section = tokenized_label["input_ids"][:, int(prev_split):int(split)]
		if torch.isin(id_section, tokenized_mask_features).sum() > 0:
			mlm_mask[:, prev_split:split] = torch.isin(id_section, tokens_to_mask_inverse)
		prev_split = split

	return (label, mlm_mask)
	
match_number = 0
for step, batch in enumerate(tqdm(eval_data)):
	#if 'TAIR10' not in batch[3][0]: #AT1G02590 'TAIR10'
	#	continue
	try:
		pp, true = beamSearch(batch)
		gff_predictions = gffString2GFF3(pp, batch[5][0], batch[6][0], f"GM={batch[3][0]}")
		
	except:
		print(f"Error in beamSearch for GeneModel:{batch[3]} ... trying greedy.\n", file=sys.stdout)
		try:
			pp, true = greedySearch(batch)
			gff_predictions = gffString2GFF3(pp, batch[5][0], batch[6][0], f"GM={batch[3][0]}")
		except:
			print(f"Error in greedySearch for GeneModel:{batch[3]} ... giving up.\n", file=sys.stdout)

	try:
		gff_labels = gffString2GFF3(true, batch[5][0], batch[6][0], f"GM={batch[3][0]}")
	except:
		print(f"Error in reference for GeneModel:{batch[4]} ...\n", file=sys.stdout)

	# Extract correct numbers for prediciton
	# Provide mask for generation for extractable ground truth
	# Skip non-extractable positions
	
	denoising_label = createDenoisingLabel(pp, true)
	if denoising_label:
		match_number += 1

	if step % 100 == 0:
		print(match_number/(step+1))

	

print("Done")