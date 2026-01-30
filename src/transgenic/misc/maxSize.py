from utils_transgenic import *
from modeling_HeynaTransgenic import *


import matplotlib as plt
import numpy as np

torch.manual_seed(123)

config = TransgenicHyenaConfig(
	do_segment=True, 
	numSegClasses=9,
	d_model=512,
	encoder_layers=18,
	decoder_layers=18,
	encoder_n_layer=18,
	attention_window = [
			1024,1024,1024,1024,1024,1024,
			1024,1024,1024,1024,1024,1024,
			1024,1024,1024,1024,1024,1024
		])
model = transgenicForConditionalGeneration(config)
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
		pred = dt.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True, clean_up_tokenization_spaces=True)[0].replace("|</s>", "").replace("</s>", "").replace("<s>", "")
		true = dt.batch_decode(batch[2].detach().cpu().numpy(), skip_special_tokens=True, clean_up_tokenization_spaces=True)[0].replace("|</s>", "").replace("</s>", "").replace("<s>", "")
		sequence = ds.encoder_tokenizer.batch_decode(batch[0].detach().cpu().numpy(), skip_special_tokens=True, clean_up_tokenization_spaces=True)[0].replace(" ", "")
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
		pred = dt.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True, clean_up_tokenization_spaces=True)[0].replace("|</s>", "").replace("</s>", "").replace("<s>", "")
		true = dt.batch_decode(batch[3].detach().cpu().numpy(), skip_special_tokens=True, clean_up_tokenization_spaces=True)[0].replace("|</s>", "").replace("</s>", "").replace("<s>", "")
		sequence = ds.encoder_tokenizer.batch_decode(batch[0].detach().cpu().numpy(), skip_special_tokens=True, clean_up_tokenization_spaces=True)[0].replace(" ", "")
		probabilities = torch.nn.functional.softmax(model.transgenic.encoder.segmentation_logits.detach().cpu(), dim=-1)[...,0].squeeze()[0:len(sequence), (0,1,2,3,4,5,6,7,8)]
		#pp = PredictionProcessor(pred, sequence, probabilities)
		#pp.postProcessPrediction(buff=200)
		return pred, true, sequence, probabilities


for step, batch in enumerate(tqdm(test_data)):
	# Generate predictions and segmentation
	
	#pred, true, sequence, probabilities = beamSearch(batch)
	ii, am, lab = batch[0].to(device), batch[1].to(device), batch[2].to(device)
	if ii.shape[1] < 6144:
		print("Skipping sequence...too short")
		continue
	with torch.no_grad():
		outputs = model(input_ids=ii, attention_mask=am, labels=lab, return_dict=True)


print("Done")