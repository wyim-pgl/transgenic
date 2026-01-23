"""
Test Nucleotide Transformer (NT) based generation with segmentation.

Uses AgroNT encoder with decoder and segmentation heads for full annotation
prediction with coordinate refinement.

Output: validation_prediction_noPost.gff3, validation_prediction_post.gff3,
        validation_labels.gff3, validation_labels_success.gff3
"""
import torch, sys
from tqdm import tqdm
from safetensors import safe_open

from ..models.configuration_transgenic import NTTransgenicConfig
from ..models.modeling_NTTransgenic import transgenicForConditionalGeneration
from ..models.tokenization_transgenic import GFFTokenizer
from ..datasets.datasets import makeDataLoader, isoformData, target_collate_fn
from ..utils.postprocess import PredictionProcessor
from ..utils.gsf import gffString2GFF3

torch.manual_seed(123)
# "checkpoints/transgenic_Gen10G_6144nt_E4.safetensors"
# "checkpoints_Gen10G/model.safetensors"
decoder_checkpoint =  "checkpoints/transgenic_Gen10G_6144nt_E4.safetensors"

# "AgroSegmentNT_Epoch1-5_6144nt_restart_codons.safetensors"
# "checkpoints_SegmentNT/model.safetensors"
segment_checkpoint = "checkpoints_SegmentNT/model.safetensors"

config = NTTransgenicConfig(do_segment=True, vocab_size=272)
model = transgenicForConditionalGeneration(config)

# Load decoder checkpoint
decoder_tensors = {}
with safe_open(decoder_checkpoint, framework="pt", device="cpu") as f:
	for k in f.keys():
		decoder_tensors[k] = f.get_tensor(k)
decoder_tensors["transgenic.decoder_embed_tokens.weight"] = decoder_tensors["lm_head.weight"]
decoder_tensors["transgenic.decoder.embed_tokens.weight"] = decoder_tensors["transgenic.decoder_embed_tokens.weight"]
newDecoder_tensors = {}
for k in decoder_tensors:
	if ("ia3" not in k) and ("unet" not in k) and ("UFC" not in k) and ("film" not in k):
		newDecoder_tensors[k] = decoder_tensors[k]

# Load segmentation checkpoint
segment_tensors = {}
with safe_open(segment_checkpoint, framework="pt", device="cpu") as f:
	for k in f.keys():
		segment_tensors[k] = f.get_tensor(k)
segment_tensors = {k.replace("unet", "transgenic.encoder.unet").replace("uFC", "transgenic.encoder.uFC"):segment_tensors[k] for k in segment_tensors}
newSegment_tensors = {}
for k in segment_tensors:
	if ("esm" not in k) & ("hidden" not in k) & ("lm_head" not in k) & ("film" not in k):
		newSegment_tensors[k] = segment_tensors[k]

# Merge dictionaries and load
tensors = {**newDecoder_tensors, **newSegment_tensors}
model.load_state_dict(tensors,strict=True)

device = torch.device("cuda")
model.eval()
model.to(device)


db="Generation_10G_static6144_addExtra200_addRCIsoOnly_clean.db"
dt = GFFTokenizer()
ds = isoformData(db, dt, mode="training", encoder_model="InstaDeepAI/agro-nucleotide-transformer-1b", global_attention=False)
train_data, eval_data, test_data = torch.utils.data.random_split(ds, [339817, 45309,67964])
test_data = makeDataLoader(test_data, shuffle=True, batch_size=1, pin_memory=True, num_workers=4, collate_fn=target_collate_fn)

def beamSearch(batch, iter=1, maxiter=5):
	ii, am, gam, lab = batch[0].to(device), batch[1].to(device), batch[2], batch[3].to(device)
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
		true = dt.batch_decode(batch[3].detach().cpu().numpy(), skip_special_tokens=True)[0].replace("|</s>", "").replace("</s>", "").replace("<s>", "")
		sequence = ds.encoder_tokenizer.batch_decode(batch[0].detach().cpu().numpy(), skip_special_tokens=True)[0].replace(" ", "")
		probabilities = torch.nn.functional.softmax(model.transgenic.encoder.segmentation_logits.detach().cpu(), dim=-1)[...,0].squeeze()[0:len(sequence), (0,1,2,3,4,5,6,7,8)]
		#pp = PredictionProcessor(pred, sequence, probabilities)
		#pp.postProcessPrediction(splice_buffer=150, start_buffer=150, stop_buffer=150, seqFeature=True)
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
		#pp.postProcessPrediction(splice_buffer=150, start_buffer=150, stop_buffer=150, seqFeature=True)
		return pred, true, sequence, probabilities


predfile = open("validation_prediction_noPost.gff3", "w")	# Raw generative predictions
procPredFile = open("validation_prediction_post.gff3", "w") # Post-processed generative predictions
fullLabFile =  open("validation_labels.gff3", "w")			# Target annotations of all tested annotations
successLabFile = open("validation_labels_success.gff3", "w")# Target annotations of parsable generative predictions

for step, batch in enumerate(tqdm(test_data)):
	# Generate predictions and segmentation
	pred, true, sequence, probabilities = beamSearch(batch)
	
	#Write full lab file
	try:
		gff_labels = gffString2GFF3(true, batch[5][0], batch[6][0], f"GM={batch[4][0]}")
		for line in gff_labels:
			fullLabFile.write(line + "\n")
	except:
		print(f"Error in reference for GeneModel:{batch[4]} ...\n", file=sys.stdout)
	
	# Write raw prediction and success labfile if raw prediction is parsable
	try:
		gff_predictions = gffString2GFF3(pred, batch[5][0], batch[6][0], f"GM={batch[4][0]}")
		for line in gff_predictions:
			predfile.write(line + "\n")
		gff_labels = gffString2GFF3(true, batch[5][0], batch[6][0], f"GM={batch[4][0]}")
		for line in gff_labels:
			successLabFile.write(line + "\n")
		generation = True
	except:
		generation = False
		print(f"Error writing raw prediction for GeneModel:{batch[4]} ...\n", file=sys.stdout)
	
	# Process predictions and write
	if generation:
		try:
			pp = PredictionProcessor(pred, sequence, probabilities)
			pp.postProcessPrediction(splice_buffer=150, start_buffer=150, stop_buffer=150, seqFeature=True)
			gff_predictions_post = gffString2GFF3(pp.stitchGFF(), batch[5][0], batch[6][0], f"GM={batch[4][0]}")
			for line in gff_predictions_post:
				procPredFile.write(line + "\n")
		except:
			print(f"Error in processor for GeneModel:{batch[4]}.\n", file=sys.stdout)

predfile.close()
procPredFile.close()
fullLabFile.close()
successLabFile.close()

print("Done")