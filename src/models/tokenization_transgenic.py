import re, json, os
from transformers import PreTrainedTokenizer

class GFFTokenizer(PreTrainedTokenizer):
	model_input_names = ["input_ids", "attention_mask"]

	def __init__(self, vocab=None, **kwargs):
		if vocab is None:
			self.vocab = {
				"<s>": 0, "<pad>": 1,"</s>":2, "<unk>":3, '0': 4, '1': 5, '2': 6, 
				'3': 7,'4': 8, '5': 9, '6': 10, '7': 11, '8': 12, 
				'9': 13, 'A': 14, 'B': 15, 'C': 16, ">":17, ".": 18, 
				"+": 19, "-": 20, ";":21
			}
			for i in range(1, 151):
				self.vocab[f"CDS{i}"] = i + 21
			for i in range(1, 51):
				self.vocab[f"five_prime_UTR{i}"] = i + 171
				self.vocab[f"three_prime_UTR{i}"] = i + 221
		else:
			self.vocab = vocab

		self.ids_to_tokens = {id: token for token, id in self.vocab.items()}
		super().__init__(**kwargs)
		self.pad_token = "<pad>"
		self.unk_token = "<unk>"
		self.eos_token = "</s>"

	@property
	def vocab_size(self):
		return len(self.vocab)

	def get_vocab(self):
		return dict(self.vocab, **self.added_tokens_encoder)

	def _tokenize(self, text):
		tokens = ["<s>"]

		for features in text.split(">"):
			for feature in features.split(";"):
				for column in feature.split("|"):
					if re.search(r'^\d+$', column):
						tokens.extend([digit for digit in column])
					else:
						tokens.append(column)
				tokens.append(";")
			tokens.append(">")
		return tokens[:-2] + ["</s>"]

	def _convert_token_to_id(self, token):
		return self.vocab.get(token, self.vocab.get(self.unk_token))

	def _convert_id_to_token(self, index):
		return self.ids_to_tokens.get(index, self.unk_token)

	def convert_tokens_to_string(self, tokens):
		toks = []
		for i,token in enumerate(tokens):
			if token.isnumeric() and i != 0:
				if tokens[i-1].isnumeric():
					toks[-1] = toks[-1] + token
					continue
			toks.append(token)
			
		toks = '|'.join([self._convert_id_to_token(token) if isinstance(token, int) else token for token in toks])
		toks = re.sub(r'\|;\|>\|', '>', toks)
		toks = re.sub(r';>', '>', toks)
		toks = re.sub(r'>\|', '>', toks)
		toks = re.sub(r'\|;\|', ';', toks)
		return toks
	
	def save_vocabulary(self, save_directory, filename_prefix=None):
		if not os.path.isdir(save_directory):
			raise ValueError(f"Provided path ({save_directory}) is not a directory.")

		vocab_file = os.path.join(
			save_directory, (filename_prefix + "-" if filename_prefix else "") + "vocab.json"
		)

		with open(vocab_file, "w", encoding="utf-8") as f:
			json.dump(self.vocab, f, ensure_ascii=False, indent=2)

		return (vocab_file,)