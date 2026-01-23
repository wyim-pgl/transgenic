"""
Test T5-style decoder with HyenaDNA encoder.

Uses T5 decoder architecture without segmentation head for pure generation.

Output: validation_T5Hyena_prediction.gff3, validation_T5Hyena_labels.gff3
"""
import torch, sys
from tqdm import tqdm
from safetensors import safe_open

from ..models.configuration_transgenic import NTTransgenicT5Config
from ..models.modeling_HeynaTransgenic import transgenicForConditionalGenerationT5
from ..models.tokenization_transgenic import GFFTokenizer
from ..datasets.datasets import makeDataLoader, isoformDataHyena, hyena_collate_fn
from ..utils.postprocess import PredictionProcessor
from ..utils.gsf import gffString2GFF3


torch.manual_seed(123)
decoder_checkpoint =  "checkpoints/T5Hyena_Gen9G_6144nt_E0-14.safetensors"
#segment_checkpoint = "checkpoints/AgroSegmentNT_Epoch1-5_6144nt_restart_codons.safetensors" #model.safetensors"

config = NTTransgenicT5Config(do_segment=False)
model = transgenicForConditionalGenerationT5(config)


# Load decoder checkpoint
decoder_tensors = {}
with safe_open(decoder_checkpoint, framework="pt", device="cpu") as f:
	for k in f.keys():
		decoder_tensors[k] = f.get_tensor(k)
	decoder_tensors["transgenic.decoder.embed_tokens.weight"] = decoder_tensors["gff_head.weight"]
newDecoder_tensors = {}
for k in decoder_tensors:
	if ("ia3" not in k) and ("unet" not in k) and ("UFC" not in k):
		newDecoder_tensors[k] = decoder_tensors[k]

# Load segmentation checkpoint
#segment_tensors = {}
#with safe_open(segment_checkpoint, framework="pt", device="cpu") as f:
#	for k in f.keys():
#		segment_tensors[k] = f.get_tensor(k)
#segment_tensors = {k.replace("unet", "transgenic.encoder.unet").replace("uFC", "transgenic.encoder.uFC"):segment_tensors[k] for k in segment_tensors}
#newSegment_tensors = {}
#for k in segment_tensors:
#	if ("esm" not in k) & ("hidden" not in k) & ("lm_head" not in k) & ("film" not in k):
#		newSegment_tensors[k] = segment_tensors[k]

# Merge dictionaries and load
tensors = {**newDecoder_tensors} #, **newSegment_tensors}
model.load_state_dict(tensors,strict=False)

device = torch.device("cuda")
model.eval()
model.to(device)


db="Generation_10G_static6144_addExtra200_addRCIsoOnly_clean.db"
dt = GFFTokenizer()
ds = isoformDataHyena(db, dt, mode="training", encoder_model="LongSafari/hyenadna-large-1m-seqlen-hf", global_attention=False)
train_data, eval_data, test_data = torch.utils.data.random_split(ds, [339817, 45309,67964])
eval_data = makeDataLoader(eval_data, shuffle=True, batch_size=1, pin_memory=True, num_workers=1, collate_fn=hyena_collate_fn)

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
		#pp = PredictionProcessor(pred, sequence, probabilities)
		#pp.postProcessPrediction(buff=200)
		#extras = pp.analyzeUnused()
		#return pp, true
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
		true = dt.batch_decode(batch[2].detach().cpu().numpy(), skip_special_tokens=True)[0].replace("|</s>", "").replace("</s>", "").replace("<s>", "")
		#sequence = ds.encoder_tokenizer.batch_decode(batch[0].detach().cpu().numpy(), skip_special_tokens=True)[0].replace(" ", "")
		#probabilities = torch.nn.functional.softmax(model.transgenic.encoder.segmentation_logits.detach().cpu(), dim=-1)[...,0].squeeze()[0:len(sequence), (0,1,2,3,4,5,6,7,8)]
		#pp = PredictionProcessor(pred, sequence, probabilities)
		#pp.postProcessPrediction(buff=200)
		#return pp, true
		return pred, true

with open("validation_T5Hyena_prediction.gff3", "w") as predfile:
	with open("validation_T5Hyena_labels.gff3", "w") as labfile:
		for step, batch in enumerate(tqdm(eval_data)):
			#if 'TAIR10' not in batch[4][0]: #AT1G02590 'TAIR10'
			#	continue
			#if batch[0].shape[1] <= 1024:
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
					#gff_predictions = gffString2GFF3(pp.stitchGFF(), batch[5][0], batch[6][0], f"GM={batch[4][0]}")
					gff_predictions = gffString2GFF3(pp, batch[5][0], batch[6][0], f"GM={batch[3][0]}")
					for line in gff_predictions:
						predfile.write(line + "\n")
				except:
					print(f"Error in greedySearch for GeneModel:{batch[4]} ... giving up.\n", file=sys.stdout)

			try:
				gff_labels = gffString2GFF3(true, batch[5][0], batch[6][0], f"GM={batch[3][0]}")
				for line in gff_labels:
					labfile.write(line + "\n")
			except:
				print(f"Error in reference for GeneModel:{batch[4]} ...\n", file=sys.stdout)


print("Done")