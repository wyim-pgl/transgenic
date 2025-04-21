import torch
from utils_transgenic import *

db="Generation_10G_static6144_addExtra200_addRCIsoOnly_clean.db"
dt = GFFTokenizer()
ds = isoformData(db, dt, mode="training", encoder_model="InstaDeepAI/agro-nucleotide-transformer-1b", global_attention=False)
train_data, eval_data, test_data = torch.utils.data.random_split(ds, [339817, 45309,67964])
eval_data = makeDataLoader(eval_data, shuffle=True, batch_size=2, pin_memory=False, num_workers=1, collate_fn=target_collate_fn)

for batch in eval_data:
	print(batch[4])
