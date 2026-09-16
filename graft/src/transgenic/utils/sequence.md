# src/transgenic/utils/sequence.py

- loadGenome · function · L23-L55 — def loadGenome(genome)
- reverseComplement · function · L58-L82 — def reverseComplement(sequence)
- validateCDS · function · L85-L153 — def validateCDS(gff: str, seq: str, geneModel: str) -> Tuple[bool, str]
- segmentSequence · function · L156-L176 — def segmentSequence(seq, piece_size=6144)
- scanGlobalAttentionTokens · function · L179-L284 — def scanGlobalAttentionTokens(vocab: dict, tokenized_sequence: List[int], seqlen: int) -> List[int]
- getSpliceBranchSeqs · function · L287-L326 — def getSpliceBranchSeqs() -> List[str]
- mask_sequences · function · L329-L373 — def mask_sequences(seqs, num_masked_nucleotides=921, mask_token=3)
