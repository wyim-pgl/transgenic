# Transgenic 
Transgenic is a seq2seq model for gene structure prediction.

## Environment
```
mamba create -y -n transgenic
mamba activate transgenic 
mamba env update -f environment.yml
```
## Checkpoints on Hugging Face
Describe checkpoints
Example generation and post-processing from hugging face with DNA sequence

## Pretraining and fine-tuning

## Inference use-cases
### De novo annotation of DNA sequences
- Create an inference dataset genome + [bed|gff3] -> duckdb (gff3 will only use gene features)
- Generation, post-processing, and gff3 output  

### Add alternative transcripts to primary annotation 
- Create an inference dataset genome + gff3 -> duckdb
- Prompted generation, post-processing, and gff3 output (disallow modifications to original boundaries in post-processing?)

## TODO
Add information for pretraining/fine-tuning
- Dataset creation
- Update training scripts for user deployment
- Fine tune decoder and segmentation

Add support for inference use cases
- De novo annotation of sequence data specified in a bed file or gff
- Add alternative transcripts to primary annotation