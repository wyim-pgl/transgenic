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

Vocabulary structure (288 tokens in "v2", 272 tokens in legacy "v1"):
  0-3:     Special tokens (<s>, <pad>, </s>, <unk>)
  4-13:    Individual digits (0-9) for encoding coordinates
  14-21:   Letters and delimiters (A, B, C, >, ., +, -, ;)
  22-171:  CDS feature tokens (CDS1 through CDS150)
  172-221: 5' UTR tokens (five_prime_UTR1 through five_prime_UTR50)
  222-271: 3' UTR tokens (three_prime_UTR1 through three_prime_UTR50)
  272:     <iso> — dedicated isoform/transcript separator (replaces ; in transcript section) [v2 only]
  273-287: <tx1> through <tx15> — transcript count planning tokens [v2 only]

The published HyenaTransgenic checkpoints (jlomas/HyenaTransgenic-*) have
vocab_size 272 and ship the legacy "v1" tokenizer in their HF repos, so
AutoTokenizer.from_pretrained(<checkpoint>) pairs correctly. Pass
vocab_version="v1" when using this local tokenizer class with those
checkpoints (e.g., fine-tuning) so no out-of-range token ids can be emitted.
"""

import re, json, os
from transformers import PreTrainedTokenizer  # HuggingFace tokenizer base class


class GFFTokenizer(PreTrainedTokenizer):
	"""Tokenizer that encodes/decodes GFF gene annotation strings."""

	model_input_names = ["input_ids", "attention_mask"]  # Required by HF pipeline API

	def __init__(self, vocab=None, vocab_version="v2", **kwargs):
		"""
		Initialize the GFF vocabulary.

		Args:
			vocab: Optional custom vocabulary dict {token_str: token_id}.
			       If None, builds the default vocabulary for `vocab_version`.
			vocab_version: "v2" (default) builds the 288-token isoform-aware
			       vocabulary (<iso>, <tx1>-<tx15>); "v1" builds the legacy
			       272-token vocabulary matching the published
			       jlomas/HyenaTransgenic-* checkpoints (vocab_size 272).
		"""
		if vocab_version not in ("v1", "v2"):
			raise ValueError(f"Unknown vocab_version '{vocab_version}'; expected 'v1' or 'v2'.")
		self.vocab_version = vocab_version
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
				";": 21        # Delimiter between individual features (in feature section)
			}
			# Add CDS feature tokens: CDS1 through CDS150
			# Each exon/CDS in a gene gets a unique numbered token
			for i in range(1, 151):
				self.vocab[f"CDS{i}"] = i + 21         # IDs 22-171

			# Add UTR feature tokens: 50 each for 5' and 3' UTRs
			for i in range(1, 51):
				self.vocab[f"five_prime_UTR{i}"] = i + 171   # IDs 172-221
				self.vocab[f"three_prime_UTR{i}"] = i + 221  # IDs 222-271

			# Isoform-aware tokens (v2 only; excluded from the legacy 272-token
			# v1 vocabulary so that ids stay within the published checkpoints'
			# vocab_size of 272):
			if self.vocab_version == "v2":
				# <iso> is a dedicated transcript separator that replaces ';' in the
				# transcript section (after '>'), disambiguating transcript boundaries
				# from feature boundaries.
				self.vocab["<iso>"] = 272

				# <tx1> through <tx15> are transcript count planning tokens. Emitted
				# immediately before '>' to signal how many transcripts will follow.
				# This gives the decoder an explicit planning signal about isoform
				# complexity in the upcoming transcript section.
				for i in range(1, 16):
					self.vocab[f"<tx{i}>"] = 272 + i       # IDs 273-287
			if self.vocab_version == "v3":
				# v3 = v2 + window-level tokens (protocol A26): <gene> separates gene blocks, <empty> marks a
				# window without any complete gene. IDs 288-289; vocab_size 290.
				self.vocab["<iso>"] = 272
				for i in range(1, 16):
					self.vocab[f"<tx{i}>"] = 272 + i
				self.vocab["<gene>"] = 288
				self.vocab["<empty>"] = 289
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

		In the isoform-aware mode (vocab_version="v2", the default):
		  - A <tx:N> token is emitted before '>' to signal the transcript count
		  - ';' delimiters in the transcript section (after '>') are replaced
		    with the dedicated '<iso>' separator token

		With vocab_version="v1" the legacy algorithm shipped with the published
		checkpoints is used (';' separates transcripts, no <iso>/<txN> tokens).

		Args:
			text: Raw GFF string (e.g., "100|CDS1|200|+|A;300|CDS2|400|+|B>CDS1|CDS2")

		Returns:
			List of token strings starting with <s> and ending with </s>.
		"""
		tokens = ["<s>"]  # Always start with beginning-of-sequence token

		if self.vocab_version == "v1":
			# Legacy tokenization matching the tokenizer shipped with the
			# published jlomas/HyenaTransgenic-* checkpoints: ';' separates
			# both features and transcripts, no <iso>/<txN> tokens, so all
			# emitted ids stay within the 272-token checkpoint vocabulary.
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

		if getattr(self, "vocab_version", "v1") == "v3":
			if text.strip() == "<empty>":
				return ["<empty>", "</s>"]
			blocks = text.split("<gene>")
			out = []
			for j, block in enumerate(blocks):
				toks = self._tokenize_block_v2(block)
				out.extend(toks[:-1])            # drop the block's </s>
				if j < len(blocks) - 1:
					out.append("<gene>")
			out.append("</s>")
			return out
		return self._tokenize_block_v2(text)

	def _tokenize_block_v2(self, text):
		tokens = []
		# Split into features section and transcripts section
		parts = text.split(">")
		features_section = parts[0] if len(parts) > 0 else ""
		transcripts_section = parts[1] if len(parts) > 1 else ""

		# Count transcripts and emit <tx:N> planning token
		if transcripts_section:
			tx_count = min(transcripts_section.count(";") + 1, 15)
		else:
			tx_count = 1
		tx_token = f"<tx{tx_count}>"

		# Tokenize features section (using ';' as delimiter between features)
		for feature in features_section.split(";"):
			for column in feature.split("|"):
				if re.search(r'^\d+$', column):
					# Numeric column (coordinate): split into individual digits
					tokens.extend([digit for digit in column])
				else:
					# Non-numeric column (feature name, strand, phase): single token
					tokens.append(column)
			tokens.append(";")  # Re-add semicolon delimiter after each feature

		# Replace trailing ';' with <tx:N> and '>' tokens
		if tokens[-1] == ";":
			tokens[-1] = tx_token
		tokens.append(">")

		# Tokenize transcripts section (using '<iso>' as delimiter between transcripts)
		if transcripts_section:
			for j, transcript in enumerate(transcripts_section.split(";")):
				for column in transcript.split("|"):
					if re.search(r'^\d+$', column):
						tokens.extend([digit for digit in column])
					else:
						tokens.append(column)
				# Use <iso> between transcripts (not after the last one)
				if j < tx_count - 1:
					tokens.append("<iso>")

		tokens.append("</s>")
		return tokens

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

		Isoform-aware tokens are handled:
		  - <iso> is converted back to ';' (standard transcript separator)
		  - <tx:N> tokens are stripped (they are planning signals only)

		Args:
			tokens: List of token strings (or integer IDs).

		Returns:
			Reconstructed GFF string.
		"""
		toks = []
		for i, token in enumerate(tokens):
			# Convert integer IDs to strings if needed
			if isinstance(token, int):
				token = self._convert_id_to_token(token)
			# Strip transcript count planning tokens (they are for the decoder only)
			if re.match(r'^<tx\d+>$', token):
				continue
			# Convert <iso> back to ';' for standard GFF output
			if token == "<iso>":
				toks.append(";")
				continue
			if token.isnumeric() and i != 0:
				# Check previous non-skipped token for digit merging
				if toks and toks[-1].isnumeric():
					# Merge consecutive digits into a single number string
					toks[-1] = toks[-1] + token
					continue
			toks.append(token)

		# Join all tokens with pipe delimiter, then clean up formatting
		toks = '|'.join(toks)
		toks = re.sub(r'\|;\|>\|', '>', toks)  # Remove extra pipes around ;>
		toks = re.sub(r';>', '>', toks)          # Clean ;> to just >
		toks = re.sub(r'\|>', '>', toks)         # Remove pipe before > (from stripped <tx:N>)
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
