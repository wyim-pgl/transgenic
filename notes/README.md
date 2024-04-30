# Transgenic

Initial development work for a generative sequence to isoform generative model. 
The idea is to use a pretrained transformer encoder to create embeddings for
DNA sequences and fine-tune a small (relatively speaking) pretrained decoder 
to output predicted isoforms. 

## Outline
- From genomic sequence of a gene model, generate all the isoforms. 
- Input data: "ATGATAACATACATAAATCGCGC..." Output data:  1,100,200,500;1,200,500
- Then stitch together all the isofroms.
- How to handle genes longer than 6kb?
	- Encode overlapping gene segments separately, concatenate and pool and/or FC layer, feed whole thing to decoder

## Hurdles and open questions
- Best way to convert embeddings from the encoder into a format that can be 
used by the decoder? (Fully connected layer, convolution, reshaping, etc)
- How resource intensive will fine-tuning be?
- Define a training, validation, and test set of genomes (how much data will I need?)
- Can I make the sequence length arbitrary?
- Do we want to handle complete splicing with UTRs?
- Fine tune existing model head? Train new custom head tailored to gff output?

## Environment
```
mamba create -y -n transgenic
mamba activate transgenic

# Cloud (nvidia/cuda) 
mamba install pytorch-nightly::pytorch pytorch-cuda=11.8 python-duckdb libaio  -c pytorch-nightly -c nvidia -c conda-forge

# Local (OSx/MPS)
#mamba install pytorch-nightly::pytorch torchvision torchaudio python-duckdb libaio -c pytorch-nightly -c conda-forge

pip install --upgrade git+https://github.com/huggingface/transformers.git peft deepspeed
git clone https://github.com/NVIDIA/cutlass.git
export CUTLASS_PATH=/path/to/cutlass

# mamba install -c conda-forge gxx_linux-64
#export CC=$(conda info --base)/envs/transgenic/bin/x86_64-conda-linux-gnu-gcc
#export CXX=$(conda info --base)/envs/transgenic/bin/x86_64-conda-linux-gnu-g++

module load gcc/9.2.0

```

## TODO List
- Write routine for complete mRNA sequence (minus UTR)
- Test isoformData.__getitem__() with all three routines 
- Try plant nucleotide encoder (does it fit in memory?)
- Explore options for modifying embeddings to decoder input size?
- Choose sequence shortening routine with 10-fold cross validation (Arabidopsis)
- Define a training, validation, and test set of genomes (how much data will I need?)
- Write backmapping function to move output isoform lists to genemodel coordinates, and then augment the original gff
- Save the model as transfomers 'PreTrainedModel'
- Define a global attention mask for DNA seq tokens representing splice junctions or other important elements

## Implementation Notes

Spin up a slurm node to play inside of:
```
srun -A $agJL -p $pgJL -c 1 --mem=4g --gres=gpu:1 --pty /bin/bash
ssh gpu-?
conda activate transgenic
```
Run a job with gpu..
```
sbatch -A $agJL -p $pgJL -c 16 --mem=64g --gres=gpu:1 --time=1-00:00:00 -J transgenic -o transgenic.out --wrap="time python testing.py"
```

**Parameters to keep track of**
- Method of embedding transformation for input to LEDEncoder
- Fine-tune entire decoder or use adaptors?
- Generation method (Greedy, Beam, Sampling)

The the nucleotide transformer can be directly fine-tuned for calssification and regression tasks. This works by replacing the final language model head with a classification or regression head.

The max length of an input sequence is 6000 or 1000 tokens (token = 6-mer)
Due to sequence length contraints, reading an entire gene region will require some tricky pre- and post- processing. The dynamic range of gene length (transcript prior to splicing) is massive... from 1kb to 1,000 kb (human: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4158763/). This poses a problem for the model because all genes must be handled by a fixed maximum size that is acheived by padding. I will likely need to slim down my gene regions to scale genes into a more similar range...
If gene length is defined as the aggregate length of UTRs and all exons, then the vast majority of human genes are under 25kb...(https://www.frontiersin.org/journals/genetics/articles/10.3389/fgene.2021.559998/full). Must remove UTR and some intronic sequence to provide reasonable input sequence lengths...
- Only use exon/intron junctions? (I think this is too redundant with exons + intronic overlap... reconsider if my sequences are still unwieldy)
- Only use exons?
- Only use exons plus some amount of intronic overlap?
- Only use introns plus some amount of exonic overlap (Intronic splice signals found in human genome: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0001246)
- Train all three models with subsets of the Arabidopsis/human genome and determine the best by 10-fold cross validation?

**Context-dependent control of alternative splicing by RNA-binding proteins**

Alternative splicing is dependent on cell type and genomic context. Cell-type influences splicing based on the expression level of various RNA Binding Proteins (RBPs). Could pass predicted RBP gene expression into the decoder's encoder to predict cell-type specific alternative slpicing? This may not impact the current use-case because I want to identify all possible splice variants of a gene model.

Deep intronic elements have been implicated as important for determining the "splice-code"

Constitutive splice sites - recognized efficiently and always spliced in the same way
Alternative splice sites - sites that compete with one or more sites for incorporation

Both exonic and intronic splice enhancers and silencers exist and are likely important for the code...How long to make junction overlaps then?
200-300 base-pairs from observed splice sites contain most identifiable features (Dichotomous splicing signals in exon flanks). But more distant relationships also are in play (Rbfox proteins regulate alternative mRNA splicing through evolutionarily conserved RNA bridges). These distant relations may be contained deep inside introns.

Initially, I wanted to pre-compute embeddings for an entire gff and use this dataset to train the decoder. However, the encoded set of 4,127 genes from AthChr4 was 33G (15 mins on 4 gpu). This is an intractable file size to load and store, so I will have to encode on-the-fly for both training and inference.

Flagship phytozome genomes to train/validate on:
- Chlamydomonas reinhardtii: An algal model
- Physcomitrella patens: A moss model
- Arabidopsis thaliana: A model for plant genetics and biology
- Glycine max Wm82.a2.v1: A soybean
- Sorghum bicolor: A biofuel crop and potential cellulosic feedstock

The 1B parameter agro encoder does not fit on a single gpu. I will cross validate and check performance with the smaller 500M paramter multispecies 

Won wants to test performance by blasting against IsoSeq data (longread)

Three sequence truncation methods to try:
1. Whole sequence
2. CDS + intron overlap
3. Intron + CDS overlap
Actually, I now agree with won that its needs to be trained on whole gene model sequences. This will prohibit significant complications arising from variations in exon sizes between isoforms and remove the need for a complete primary annotation as input for inference. I can simply make as many segments for encoding as I need and T5 can accept any sequence length that can fit in memory...(https://github.com/huggingface/transformers/issues/5204). Look at reformer and longformer for more memory efficient models?
T5Config().max_postion_embeddings

First training trial run on AthChr4 (10 epochs, 3 hours):
real	180m13.728s
user	495m54.844s
sys	234m5.651s
Epoch 0: train_ppl train_ppl=tensor(14.0664, device='cuda:2'), train_epoch_loss train_epoch_loss=tensor(2.6438, device='cuda:2'), eval_ppl eval_ppl=tensor(7.8520, device='cuda:2'), eval_epoch_loss eval_epoch_loss=tensor(2.0608, device='cuda:2')
Epoch 0: train_ppl train_ppl=tensor(13.7738, device='cuda:1'), train_epoch_loss train_epoch_loss=tensor(2.6228, device='cuda:1'), eval_ppl eval_ppl=tensor(7.3367, device='cuda:1'), eval_epoch_loss eval_epoch_loss=tensor(1.9929, device='cuda:1')
Epoch 0: train_ppl train_ppl=tensor(13.6777, device='cuda:3'), train_epoch_loss train_epoch_loss=tensor(2.6158, device='cuda:3'), eval_ppl eval_ppl=tensor(7.6131, device='cuda:3'), eval_epoch_loss eval_epoch_loss=tensor(2.0299, device='cuda:3')
Epoch 1: train_ppl train_ppl=tensor(9.2781, device='cuda:0'), train_epoch_loss train_epoch_loss=tensor(2.2277, device='cuda:0'), eval_ppl eval_ppl=tensor(7.4686, device='cuda:0'), eval_epoch_loss eval_epoch_loss=tensor(2.0107, device='cuda:0')
Epoch 1: train_ppl train_ppl=tensor(9.4857, device='cuda:2'), train_epoch_loss train_epoch_loss=tensor(2.2498, device='cuda:2'), eval_ppl eval_ppl=tensor(7.2109, device='cuda:2'), eval_epoch_loss eval_epoch_loss=tensor(1.9756, device='cuda:2')
Epoch 1: train_ppl train_ppl=tensor(9.3534, device='cuda:1'), train_epoch_loss train_epoch_loss=tensor(2.2357, device='cuda:1'), eval_ppl eval_ppl=tensor(6.7407, device='cuda:1'), eval_epoch_loss eval_epoch_loss=tensor(1.9082, device='cuda:1')
Epoch 1: train_ppl train_ppl=tensor(9.0926, device='cuda:3'), train_epoch_loss train_epoch_loss=tensor(2.2075, device='cuda:3'), eval_ppl eval_ppl=tensor(6.9917, device='cuda:3'), eval_epoch_loss eval_epoch_loss=tensor(1.9447, device='cuda:3')
Epoch 2: train_ppl train_ppl=tensor(8.5659, device='cuda:0'), train_epoch_loss train_epoch_loss=tensor(2.1478, device='cuda:0'), eval_ppl eval_ppl=tensor(7.0289, device='cuda:0'), eval_epoch_loss eval_epoch_loss=tensor(1.9500, device='cuda:0')
Epoch 2: train_ppl train_ppl=tensor(8.4159, device='cuda:1'), train_epoch_loss train_epoch_loss=tensor(2.1301, device='cuda:1'), eval_ppl eval_ppl=tensor(6.3798, device='cuda:1'), eval_epoch_loss eval_epoch_loss=tensor(1.8531, device='cuda:1')
Epoch 2: train_ppl train_ppl=tensor(8.3564, device='cuda:3'), train_epoch_loss train_epoch_loss=tensor(2.1230, device='cuda:3'), eval_ppl eval_ppl=tensor(6.5332, device='cuda:3'), eval_epoch_loss eval_epoch_loss=tensor(1.8769, device='cuda:3')
Epoch 3: train_ppl train_ppl=tensor(8.1174, device='cuda:1'), train_epoch_loss train_epoch_loss=tensor(2.0940, device='cuda:1'), eval_ppl eval_ppl=tensor(6.3535, device='cuda:1'), eval_epoch_loss eval_epoch_loss=tensor(1.8490, device='cuda:1')
Epoch 3: train_ppl train_ppl=tensor(7.8887, device='cuda:3'), train_epoch_loss train_epoch_loss=tensor(2.0654, device='cuda:3'), eval_ppl eval_ppl=tensor(6.5910, device='cuda:3'), eval_epoch_loss eval_epoch_loss=tensor(1.8857, device='cuda:3')
Epoch 4: train_ppl train_ppl=tensor(7.9356, device='cuda:2'), train_epoch_loss train_epoch_loss=tensor(2.0714, device='cuda:2'), eval_ppl eval_ppl=tensor(6.6076, device='cuda:2'), eval_epoch_loss eval_epoch_loss=tensor(1.8882, device='cuda:2')
Epoch 4: train_ppl train_ppl=tensor(7.7684, device='cuda:1'), train_epoch_loss train_epoch_loss=tensor(2.0501, device='cuda:1'), eval_ppl eval_ppl=tensor(6.2512, device='cuda:1'), eval_epoch_loss eval_epoch_loss=tensor(1.8328, device='cuda:1')
Epoch 4: train_ppl train_ppl=tensor(7.6789, device='cuda:3'), train_epoch_loss train_epoch_loss=tensor(2.0385, device='cuda:3'), eval_ppl eval_ppl=tensor(6.4440, device='cuda:3'), eval_epoch_loss eval_epoch_loss=tensor(1.8632, device='cuda:3')
Epoch 5: train_ppl train_ppl=tensor(7.7628, device='cuda:2'), train_epoch_loss train_epoch_loss=tensor(2.0493, device='cuda:2'), eval_ppl eval_ppl=tensor(6.4868, device='cuda:2'), eval_epoch_loss eval_epoch_loss=tensor(1.8698, device='cuda:2')
Epoch 5: train_ppl train_ppl=tensor(7.6891, device='cuda:1'), train_epoch_loss train_epoch_loss=tensor(2.0398, device='cuda:1'), eval_ppl eval_ppl=tensor(6.1653, device='cuda:1'), eval_epoch_loss eval_epoch_loss=tensor(1.8189, device='cuda:1')
Epoch 5: train_ppl train_ppl=tensor(7.5936, device='cuda:3'), train_epoch_loss train_epoch_loss=tensor(2.0273, device='cuda:3'), eval_ppl eval_ppl=tensor(6.3193, device='cuda:3'), eval_epoch_loss eval_epoch_loss=tensor(1.8436, device='cuda:3')
Epoch 6: train_ppl train_ppl=tensor(7.6219, device='cuda:2'), train_epoch_loss train_epoch_loss=tensor(2.0310, device='cuda:2'), eval_ppl eval_ppl=tensor(6.4555, device='cuda:2'), eval_epoch_loss eval_epoch_loss=tensor(1.8649, device='cuda:2')
Epoch 6: train_ppl train_ppl=tensor(7.5502, device='cuda:1'), train_epoch_loss train_epoch_loss=tensor(2.0216, device='cuda:1'), eval_ppl eval_ppl=tensor(6.1459, device='cuda:1'), eval_epoch_loss eval_epoch_loss=tensor(1.8158, device='cuda:1')
Epoch 6: train_ppl train_ppl=tensor(7.4048, device='cuda:3'), train_epoch_loss train_epoch_loss=tensor(2.0021, device='cuda:3'), eval_ppl eval_ppl=tensor(6.2588, device='cuda:3'), eval_epoch_loss eval_epoch_loss=tensor(1.8340, device='cuda:3')
Epoch 7: train_ppl train_ppl=tensor(7.4702, device='cuda:2'), train_epoch_loss train_epoch_loss=tensor(2.0109, device='cuda:2'), eval_ppl eval_ppl=tensor(6.3682, device='cuda:2'), eval_epoch_loss eval_epoch_loss=tensor(1.8513, device='cuda:2')
Epoch 7: train_ppl train_ppl=tensor(7.3349, device='cuda:1'), train_epoch_loss train_epoch_loss=tensor(1.9926, device='cuda:1'), eval_ppl eval_ppl=tensor(6.0614, device='cuda:1'), eval_epoch_loss eval_epoch_loss=tensor(1.8019, device='cuda:1')
Epoch 7: train_ppl train_ppl=tensor(7.2639, device='cuda:3'), train_epoch_loss train_epoch_loss=tensor(1.9829, device='cuda:3'), eval_ppl eval_ppl=tensor(6.1050, device='cuda:3'), eval_epoch_loss eval_epoch_loss=tensor(1.8091, device='cuda:3')
Epoch 8: train_ppl train_ppl=tensor(7.2958, device='cuda:2'), train_epoch_loss train_epoch_loss=tensor(1.9873, device='cuda:2'), eval_ppl eval_ppl=tensor(6.3903, device='cuda:2'), eval_epoch_loss eval_epoch_loss=tensor(1.8548, device='cuda:2')
Epoch 8: train_ppl train_ppl=tensor(7.2416, device='cuda:1'), train_epoch_loss train_epoch_loss=tensor(1.9798, device='cuda:1'), eval_ppl eval_ppl=tensor(6.0489, device='cuda:1'), eval_epoch_loss eval_epoch_loss=tensor(1.7999, device='cuda:1')
Epoch 8: train_ppl train_ppl=tensor(7.1637, device='cuda:3'), train_epoch_loss train_epoch_loss=tensor(1.9690, device='cuda:3'), eval_ppl eval_ppl=tensor(6.0824, device='cuda:3'), eval_epoch_loss eval_epoch_loss=tensor(1.8054, device='cuda:3')
Epoch 9: train_ppl train_ppl=tensor(7.1495, device='cuda:0'), train_epoch_loss train_epoch_loss=tensor(1.9670, device='cuda:0'), eval_ppl eval_ppl=tensor(6.5171, device='cuda:0'), eval_epoch_loss eval_epoch_loss=tensor(1.8744, device='cuda:0')
Epoch 9: train_ppl train_ppl=tensor(7.1654, device='cuda:1'), train_epoch_loss train_epoch_loss=tensor(1.9693, device='cuda:1'), eval_ppl eval_ppl=tensor(6.0290, device='cuda:1'), eval_epoch_loss eval_epoch_loss=tensor(1.7966, device='cuda:1')
Epoch 9: train_ppl train_ppl=tensor(7.0264, device='cuda:3'), train_epoch_loss train_epoch_loss=tensor(1.9497, device='cuda:3'), eval_ppl eval_ppl=tensor(6.0680, device='cuda:3'), eval_epoch_loss eval_epoch_loss=tensor(1.8030, device='cuda:3')

Strange gene model with missplaced UTR3?
    4 Chr4    phytozome9_0    gene    2895    10455   .       -       .       ID=AT4G00020;Name=AT4G00020
    5 Chr4    phytozome9_0    mRNA    2895    10364   .       -       .       ID=PAC:19647157;Name=AT4G00020.2;pacid=19647157;longest=1;Parent=AT4G00020
    6 Chr4    phytozome9_0    CDS     10211   10364   .       -       0       ID=PAC:19647157.CDS.1;Parent=PAC:19647157;pacid=19647157
   ...
   25 Chr4    phytozome9_0    CDS     4227    4438    .       -       0       ID=PAC:19647157.CDS.20;Parent=PAC:19647157;pacid=19647157
   26 Chr4    phytozome9_0    CDS     4127    4149    .       -       1       ID=PAC:19647157.CDS.21;Parent=PAC:19647157;pacid=19647157
   27 Chr4    phytozome9_0    CDS     2895    3022    .       -       2       ID=PAC:19647157.CDS.22;Parent=PAC:19647157;pacid=19647157
   28 Chr4    phytozome9_0    mRNA    3895    10455   .       -       .       ID=PAC:19647158;Name=AT4G00020.1;pacid=19647158;longest=0;Parent=AT4G00020
   29 Chr4    phytozome9_0    CDS     10211   10364   .       -       0       ID=PAC:19647158.CDS.1;Parent=PAC:19647158;pacid=19647158
   30 Chr4    phytozome9_0    five_prime_UTR  10365   10455   .       -       .       ID=PAC:19647158.five_prime_UTR.1;Parent=PAC:19647158;pacid=19647158
   31 Chr4    phytozome9_0    CDS     9982    10125   .       -       2       ID=PAC:19647158.CDS.2;Parent=PAC:19647158;pacid=19647158
   32 Chr4    phytozome9_0    CDS     9686    9847    .       -       2       ID=PAC:19647158.CDS.3;Parent=PAC:19647158;pacid=19647158
   ...
   47 Chr4    phytozome9_0    CDS     4545    4749    .       -       1       ID=PAC:19647158.CDS.18;Parent=PAC:19647158;pacid=19647158
   48 Chr4    phytozome9_0    CDS     4265    4438    .       -       0       ID=PAC:19647158.CDS.19;Parent=PAC:19647158;pacid=19647158
   49 Chr4    phytozome9_0    three_prime_UTR 3895    4106    .       -       .       ID=PAC:19647158.three_prime_UTR.1;Parent=PAC:19647158;pacid=19647158
   50 Chr4    phytozome9_0    CDS     4107    4172    .       -       0       ID=PAC:19647158.CDS.20;Parent=PAC:19647158;pacid=19647158

Now, a whole new direction. Since I decided to work with entire gene sequences, I can return to the idea of direct GFF output. This way I will capture all gene features (UTRs, CDS of varying size, isoform structure). The memory requirements of this paradigm will be much larger and I'll need to look into model parallelism of fully sharded data parallel. Maybe I can also use the torch/slurm integration to distribute data to whole other compute nodes?

I'm going to switch from T5 to Longformer Encoder Decoder(LED) which is potentially more memory efficient on long sequences (upt to 4096 tokens or 24,576 bp).
There is also LongT5 available, maybe cross validate on those two models?
LED has a max label embedding of 1024... reduce redundancy in the gff, increase the vocab to reduce 4-digit numbers to one token, see if T5 long has a larger constraint...

Starting with fine tuning the existing LM head on LED... if the performance is poor I could consider making a custom head.

I think I'll need to make a custom head with custom vocabulary and tokenizer.
Yes, fully custom LM head with unique vocabulary optimized to reduce redundancy. 
This also required me to provide new embedding layers to the decoder which need to be changed from scratch.

It seems that the PEFT default configurations are desinged to work with specific model types. Thus, trying to use only the decoder from LED threw errors. It works with the full LED model only... so my architecture is encoder -> encoder -> decoder (A bit strange...)

Adding EOS to the end of the labels improved training and generation significantly!
Training on just AthChr4 for 10 epochs provided coherently structured text with incorrect numbers.

I doubled the max decoder length to allow for very long gene models. And will now try a full training run...
If the model is not accurate enough after this run, I will need to think about sharding the model and using the larger nucleotide encoders...

## Full Training Run

Download complete fasta and gff from Phytozome. Flagship phytozome genomes to train/validate on:
- Ptrichocarpa_533_v4.1
- Physcomitrella patens v3.3
- Arabidopsis thaliana TAIR10
- Glycine max Wm82.a6.v1
- Sorghum bicolor v5.1

Ensure proper sort order with ```agat_convert_sp_gxf2gxf.pl```:
```
singularity exec /data/gpfs/assoc/pgl/johnny/GA_helixer_fix.sif agat_convert_sp_gxf2gxf.pl -gff Athaliana_167_TAIR10.gene.gff3 -o Athaliana_167_TAIR10.gene.clean.gff3
singularity exec /data/gpfs/assoc/pgl/johnny/GA_helixer_fix.sif agat_convert_sp_gxf2gxf.pl -gff Gmax_880_Wm82.a6.v1.gene_exons.gff3 -o Gmax_880_Wm82.a6.v1.gene_exons.clean.gff3
singularity exec /data/gpfs/assoc/pgl/johnny/GA_helixer_fix.sif agat_convert_sp_gxf2gxf.pl -gff Ppatens_318_v3.3.gene_exons.gff3 -o Ppatens_318_v3.3.gene_exons.clean.gff3
singularity exec /data/gpfs/assoc/pgl/johnny/GA_helixer_fix.sif agat_convert_sp_gxf2gxf.pl -gff Ptrichocarpa_533_v4.1.gene_exons.gff3 -o Ptrichocarpa_533_v4.1.gene_exons.clean.gff3
singularity exec /data/gpfs/assoc/pgl/johnny/GA_helixer_fix.sif agat_convert_sp_gxf2gxf.pl -gff Sbicolor_730_v5.1.gene_exons.gff3 -o Sbicolor_730_v5.1.gene_exons.clean.gff3
```

Create database: ```Flagship_Genomes.db```:
```{python}
db = "Flagship_Genomes.db"
files = {
    "Athaliana_167_TAIR10.fa":" Athaliana_167_TAIR10.gene.clean.gff3",
    "Gmax_880_v6.0.fa":" Gmax_880_Wm82.a6.v1.gene_exons.clean.gff3",
    "Ppatens_318_v3.fa":" Ppatens_318_v3.3.gene_exons.clean.gff3",
    "Ptrichocarpa_533_v4.0.fa":" Ptrichocarpa_533_v4.1.gene_exons.clean.gff3",
    "Sbicolor_730_v5.0.fa":" Sbicolor_730_v5.1.gene_exons.clean.gff3"
}
for fasta, gff in files.items():
    genome2GeneList(fasta, gff, db=db)
    ds = isoformData(db, mode="training")
    length = len(ds)
    print(f"{length=}")
```

Run 10 epochs of training with train: 75%, validation: 10%, test: 15%
