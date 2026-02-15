"""
GFF Tokenizer for the TransGenic decoder.

Converts GFF (Gene Feature Format) annotation strings into token sequences
that the Longformer decoder can process. The GFF format encodes gene
structure as:

  features > transcripts

Where features are semicolon-separated entries like:
  start|CDS1|end|strand|phase;start|CDS2|end|strand|phase;...

And transcripts list which features belong to each mRNA isoform:
  CDS1|CDS2|five_prime_UTR1;CDS1|CDS3|three_prime_UTR1

The tokenizer maps each GFF element (digits, feature names, delimiters)
to unique integer IDs. During decoding, token IDs are converted back to
a GFF string that can be interpreted as genomic annotations.

Vocabulary structure (272 tokens):
  0-3:     Special tokens (<s>, <pad>, </s>, <unk>)
  4-13:    Individual digits (0-9) for encoding coordinates
  14-21:   Letters and delimiters (A, B, C, >, ., +, -, ;)
  22-171:  CDS feature tokens (CDS1 through CDS150)
  172-221: 5' UTR tokens (five_prime_UTR1 through five_prime_UTR50)
  222-271: 3' UTR tokens (three_prime_UTR1 through three_prime_UTR50)
"""

import re, json, os
from transformers import PreTrainedTokenizer  # HuggingFace tokenizer base class


class GFFTokenizer(PreTrainedTokenizer):
	"""Tokenizer that encodes/decodes GFF gene annotation strings."""

	model_input_names = ["input_ids", "attention_mask"]  # Required by HF pipeline API

	def __init__(self, vocab=None, **kwargs):
		"""
		Initialize the GFF vocabulary.

		Args:
			vocab: Optional custom vocabulary dict {token_str: token_id}.
			       If None, builds the default 272-token GFF vocabulary.
		"""
		if vocab is None:
			# Build the default vocabulary mapping
			self.vocab = {
				"<s>": 0,      # Beginning of sequence
				"<pad>": 1,    # Padding token (ignored during loss computation)
				"</s>": 2,     # End of sequence
				"<unk>": 3,    # Unknown token fallback
				'0': 4, '1': 5, '2': 6,    # Individual digit tokens for encoding
				'3': 7, '4': 8, '5': 9,    # genomic coordinates character by character
				'6': 10, '7': 11, '8': 12, # (e.g., position 1234 → tokens [5,6,7,8])
				'9': 13,
				'A': 14,       # Phase A (first reading frame position)
				'B': 15,       # Phase B (second reading frame position)
				'C': 16,       # Phase C (third reading frame position)
				">": 17,       # Delimiter between features section and transcripts section
				".": 18,       # Phase "." (not applicable, used for UTR features)
				"+": 19,       # Forward strand indicator
				"-": 20,       # Reverse strand indicator
				";": 21        # Delimiter between individual features or transcript entries
			}
			# Add CDS feature tokens: CDS1 through CDS150
			# Each exon/CDS in a gene gets a unique numbered token
			for i in range(1, 151):
				self.vocab[f"CDS{i}"] = i + 21         # IDs 22-171

			# Add UTR feature tokens: 50 each for 5' and 3' UTRs
			for i in range(1, 51):
				self.vocab[f"five_prime_UTR{i}"] = i + 171   # IDs 172-221
				self.vocab[f"three_prime_UTR{i}"] = i + 221  # IDs 222-271
		else:
			self.vocab = vocab  # Use caller-provided vocabulary

		# Build reverse lookup: token_id → token_string (for decoding)
		self.ids_to_tokens = {id: token for token, id in self.vocab.items()}

		# Initialize HuggingFace base tokenizer
		super().__init__(**kwargs)

		# Set special tokens so HF utilities can find them
		self.pad_token = "<pad>"
		self.unk_token = "<unk>"
		self.eos_token = "</s>"

	@property
	def vocab_size(self):
		"""Return total number of tokens in the vocabulary."""
		return len(self.vocab)

	def get_vocab(self):
		"""Return the complete vocabulary including any added tokens."""
		return dict(self.vocab, **self.added_tokens_encoder)

	def _tokenize(self, text):
		"""
		Tokenize a GFF annotation string into a list of token strings.

		GFF format: "start|name|end|strand|phase;...>transcript1|transcript2;..."
		Each field is split by '>' (features vs transcripts), ';' (entries),
		and '|' (columns within an entry). Numeric columns are split into
		individual digit tokens; named columns become single tokens.

		Args:
			text: Raw GFF string (e.g., "100|CDS1|200|+|A;300|CDS2|400|+|B>CDS1|CDS2")

		Returns:
			List of token strings starting with <s> and ending with </s>.
		"""
		tokens = ["<s>"]  # Always start with beginning-of-sequence token

		for features in text.split(">"):         # Split features section from transcripts
			for feature in features.split(";"):   # Split individual feature entries
				for column in feature.split("|"):  # Split columns within a feature
					if re.search(r'^\d+$', column):
						# Numeric column (coordinate): split into individual digits
						# e.g., "1234" → ["1", "2", "3", "4"]
						tokens.extend([digit for digit in column])
					else:
						# Non-numeric column (feature name, strand, phase): single token
						# e.g., "CDS1", "+", "A"
						tokens.append(column)
				tokens.append(";")  # Re-add semicolon delimiter after each feature
			tokens.append(">")      # Re-add section delimiter after features/transcripts

		# Remove trailing ";>" pair and add end-of-sequence token
		return tokens[:-2] + ["</s>"]

	def _convert_token_to_id(self, token):
		"""Look up the integer ID for a token string. Returns <unk> ID if not found."""
		return self.vocab.get(token, self.vocab.get(self.unk_token))

	def _convert_id_to_token(self, index):
		"""Look up the token string for an integer ID. Returns <unk> string if not found."""
		return self.ids_to_tokens.get(index, self.unk_token)

	def convert_tokens_to_string(self, tokens):
		"""
		Convert a sequence of tokens back into a GFF annotation string.

		Consecutive digit tokens are merged into multi-digit numbers.
		Pipe '|' delimiters are re-inserted between columns, and ';'/'>'
		delimiters are cleaned up to produce valid GFF format.

		Args:
			tokens: List of token strings (or integer IDs).

		Returns:
			Reconstructed GFF string.
		"""
		toks = []
		for i, token in enumerate(tokens):
			if token.isnumeric() and i != 0:
				if tokens[i - 1].isnumeric():
					# Merge consecutive digits into a single number string
					# e.g., ["1", "2", "3"] → "123"
					toks[-1] = toks[-1] + token
					continue
			toks.append(token)

		# Join all tokens with pipe delimiter, then clean up formatting
		toks = '|'.join([self._convert_id_to_token(token) if isinstance(token, int) else token for token in toks])
		toks = re.sub(r'\|;\|>\|', '>', toks)  # Remove extra pipes around ;>
		toks = re.sub(r';>', '>', toks)          # Clean ;> to just >
		toks = re.sub(r'>\|', '>', toks)         # Remove pipe after >
		toks = re.sub(r'\|;\|', ';', toks)       # Remove extra pipes around ;
		return toks

	def save_vocabulary(self, save_directory, filename_prefix=None):
		"""
		Save the vocabulary to a JSON file in the given directory.

		Args:
			save_directory: Path to output directory (must exist).
			filename_prefix: Optional prefix for the vocab filename.

		Returns:
			Tuple containing the path to the saved vocabulary file.
		"""
		if not os.path.isdir(save_directory):
			raise ValueError(f"Provided path ({save_directory}) is not a directory.")

		# Construct output filename with optional prefix
		vocab_file = os.path.join(
			save_directory, (filename_prefix + "-" if filename_prefix else "") + "vocab.json"
		)

		# Write vocabulary as JSON
		with open(vocab_file, "w", encoding="utf-8") as f:
			json.dump(self.vocab, f, ensure_ascii=False, indent=2)

		return (vocab_file,)
