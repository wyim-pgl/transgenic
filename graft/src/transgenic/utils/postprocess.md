# src/transgenic/utils/postprocess.py

- classifyGeneSignal · function · L56-L121 — def classifyGeneSignal( probs, threshold=0.65, sigma = 100, #rising_threshold = 0.1, #falling_threshold = -0.1, window_length=6144)
- checkPhase · function · L124-L152 — def checkPhase(gff)
- PredictionProcessor · class · L159-L1448 — class PredictionProcessor()
- __init__ · method · L227-L354 — def __init__(self, gff:str, sequence:str, probs)
- tidyPrediction · method · L356-L458 — def tidyPrediction(self)
- reverseComplement · method · L460-L474 — def reverseComplement(self)
- findLargestGeneRegion · method · L476-L521 — def findLargestGeneRegion(self, lst, threshold=0.4)
- findSeqFeatures · method · L523-L630 — def findSeqFeatures(self, junctionThresh=0.6, startThresh=0.6)
- checkStartCodons · method · L632-L729 — def checkStartCodons(self, buff=50, usingSeqFeature=False)
- checkStopCodons · method · L731-L817 — def checkStopCodons(self, buff=50, usingSeqFeature=False)
- containsStopCodon · method · L819-L844 — def containsStopCodon(self, seq, phase)
- checkProbableSpliceJunctions · method · L847-L983 — def checkProbableSpliceJunctions(self, pairs, buff=100, cds=True)
- checkCanonicalSpliceJunctions · method · L986-L1171 — def checkCanonicalSpliceJunctions(self, buff=50)
- selectDonorByPhase · method · L1173-L1253 — def selectDonorByPhase(self, donor:str, donor_end:int, acceptor:str, dsite:str, asite:str) -> bool
- checkSpliceJunctions · method · L1255-L1274 — def checkSpliceJunctions(self, pairs, buff=50, cds=True)
- stitchGFF · method · L1276-L1298 — def stitchGFF(self, originalStrand=True)
- analyzeUnused · method · L1300-L1389 — def analyzeUnused(self)
- postProcessPrediction · method · L1394-L1448 — def postProcessPrediction(self, utr_splice_buf = 100, splice_buffer=10, start_buffer=150, stop_buffer=150, seqFeature=True) -> str
