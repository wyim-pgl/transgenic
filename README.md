# TransGenic 
TransGenic is a transformer for DNA-to-annotation machine translation. Gene annotations specify the structure of a gene within a DNA sequence by providing the composition of each mRNA transcript based on the coordinate locations of sub-genic features, including coding sequences (CDS), introns, and unstranslated regions (UTR). TransGenic uses a HyenaDNA encoder with the Longformer decoder to predict a text-based annotation format from raw DNA sequence. 

# Gene sentence format (GSF)
TransGenic produces output in a format modified from the standard Gene Feature Format (GFF). Gene sentence format (GSF) contains identical information as GFF but reduces the redundancy and length of output annotations. This permits generative decoding within reasonable memory requirements for the decoder's attention mechansims.

Gene sentence format specifies gene model outputs in two parts, a feature list and a transcript list. The feature list specifies the coordinate locations of sub-genic features (CDS, 5'-UTR, and 3'-UTR) and the transcript list specifies the composition of spliced mRNA transcripts based on the components in the feature list.

Given a mock gene in GFF:
```
Chr1 source gene 35  286 . + . ID=...
Chr1 source mRNA 35  286 . + . ID=...
Chr1 source cds  35  56  . + 0 ID=...
Chr1 source cds  97  125 . + 1 ID=...
Chr1 source cds  245 286 . + 0 ID=...
Chr1 source mRNA 97  286 . + . ID=...
Chr1 source cds  97  125 . + 1 ID=...
Chr1 source cds  245 286 . + 0 ID=...
```	

The GSF annotation is:
```
35|CDS1|56|+|A;97|CDS2|125|+|B;245|CDS3|286|+|A>CDS1|CDS2|CDS3;CDS2|CDS3
```

# Using TransGenic
## Quick start
See the following notebook examples on Google Colab:
- [Colab: Inference on a single sequence](link)

Or run the examples locally from the ```examples/``` folder after setting up an environment as described below.
## Set-up
```
# Clone the repo
git clone git@github.com:JohnnyLomas/transgenic.git
cd transgenic

# Environment set-up
mamba create -y -n transgenic && mamba activate transgenic 
mamba env update -f environment.yml
```

## Pretrained Checkpoints on HuggingFace

All checkpoints were trained on 9 plant genomes covering diverse phyla, including dicot, monocot, and moss species. The highest performance on test set evaluation (92% base-level F1 in *Arabidopsis*) was acheived using the 400M parameter model.

[HyenaTransgenic-768L12A6-400M](https://huggingface.co/jlomas/HyenaTransgenic-768L12A6-400M)
- Hidden size = 768, 12 layers, 6 attention heads (~400M parameters)

[HyenaTransgenic-512L9A4-160M](https://huggingface.co/jlomas/HyenaTransgenic-512L9A4-160M)
- Hidden size = 512, 9 layers, 4 attention heads (~160M parameters)

## Inference use-cases
### De novo annotation of DNA sequences
- Create an inference dataset genome + [bed|gff3] -> duckdb (gff3 will only use gene features)
- Generation, post-processing, and gff3 output  

### Add alternative transcripts to primary annotation 
- Create an inference dataset genome + gff3 -> duckdb
- Prompted generation, post-processing, and gff3 output (disallow modifications to original boundaries in post-processing?)

## Pretraining and fine-tuning