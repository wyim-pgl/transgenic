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
mamba install pytorch pytorch-cuda=11.8 transformers peft deepspeed python-duckdb libaio libcurand cutlass matplotlib -c pytorch -c nvidia -c conda-forge
pip install ninja

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
- Improve the handling of predictions to make them easily integrable with the input database (add an output table to the db?)
- Improve error handling for writing predictions to GFF

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
    rice
    maize
    brachypodium
}
for fasta, gff in files.items():
    genome2GeneList(fasta, gff, db=db)
    ds = isoformData(db, mode="training")
    length = len(ds)
    print(f"{length=}")
```

Run 10 epochs of training with train: 75%, validation: 10%, test: 15%

Problem: token 20 has the highest probability everywhere regardless of token. All logits are the same regardless of token.
1. Check that the custom head was actually updated by the training...
2. Where do the outputs become uniform?

The last decoder hidden layer was uniform so the problem was likely deeper in the model.
I found that my new embedding and positional embedding layers didn't have gradients... restored their reguires_grad to True and re-training

Fixed a primary bug. The tokens were not being properly right shifted for the decoder. Now it is training properly!

Implementing upgrades for a final training run:
- Use 1b parameter plant encoder model
- Added peft adaptors to the encoder
- Added buffer sequence to each gene region so the model learns to find the start and end coords of the UTRs
- Increased the max sequence length to 49kb (batch size of 8 for segmented encoder)
- Added Bdistachyon and setaria viridis to the dataset to round out the representation form monocots

I finished a complete training run of the decoder/hidden_mapping with the 25kb 5 genomes dataset.
I acheived an average training set loss of 0.65 and an average validation set loss of 0.83. While training set loss improved over the final few epochs, the validation loss stopped improving around epoch 6 or 7. So far the model at least mimicks the output format...

A loss of 0.7 corresponds to an example where the large-scale gene model structure is generally correct (correct # of CDS/UTR, correct strand, correct number of alternative transcripts). However, the base-level coordinates of each feature are relatively close but don't match. I also noticed that the start coordinate is always larger than the end coordinate - that the model is learning something about the quantitatve nature of the numbers? I completed a complete test-set run with the 25kb_5genomes trained model and got an average loss of about 0.84.

For the next round of training, I am starting from the 25kb_5genomes checkpoint and adding PEFT adaptors to the encoder. Now working with 60M trainable parameters including encoder PEFT adaptors, hidden_mapping, and the entire LEDDecoder. The dataset is the updated set with maxLen ~ 49kb, two additional monocot genomes, and randomly buffered sequences (for UTR start/end learning).

Ideas:
- Try LongT5 decoder (about twice as large as LED)
- Train only to predict genic structure, then write grant to predict isoforms (problems from sparsity of alternatively spliced examples?)
- By not specifing a global_attention_mask I am using only local attention
    - Try adjusting the local attention window size (currently 1024)
    - Try providing a global_attention_mask using canonical splice junctions, start codons, stop codons, etc. (any heuristics other than dinucleotides?)
    - Train with global_attention_derived from gff, predict with global attention derived from RNAseq
    - Or, train a model to identify splice junctions and use it to build the global attention masks

likely splice site sequences (both strands ~ 25% of sixmer tokens):
"AGGT","AGG","AGGC","AGAT","ACG","ACCT","CCT","GCCT","ATCT","CGT"

branch point motif
YNYTRAY` (where Y = pyrimidine (CT), N = any nucleotide, and R = purine(AG)) located 20-50 nucleotides upstream of the 3' splice site.
motif represents 128 sequences with reverse complements

I implemented global attention by scanning for possible donor splice sites and branch sequences. I could not figure out a good heuristic for dertermining which strand the global tokens come from. Started another training run with 49kb/7genomes/200addextra dataset, peft in the ESM, global attention in the decoder. Started from the 25kb_5genomes checkpoint. I am not very hopeful that this will help because it did not significantly help in small 1-data-point testing on CPU.

Another strategy is to change the number tokenization to only 0-9. Maybe it will learn the quantitative relationships better than using two digit tokens? The new tokenizer performed signficantly better on a singly example. Will start a new local training run from scratch with this tokenizer... main thing to be worried about is the 2048 token output limit.

The gradients through the hidden mapping seem to be exploding, because they are getting clipped at 1. Potentially I could remove the hidden mapping entirely if I changed the d_model of LED to 1500 instead of 768... I would need to save memory in other places to do this, perhaps by changing the local self_attn window size to 512? (Add to config: d_model=1500, attention_window=512, decoder_ffn_dim=6000, encoder_attention_heads=15, decoder_attention_heads=15) This strategy did not reduce the loss on a single training example... The decoder was 4 times larger and the whole model took 8G as opposed to 6G. Changing the attention window to 1024 did not help either.

Other parameters to tweak:
- Batch size
- Learning rate
- Gradient clipping

PEFT was not active in the encoder because I still had with torch.no_grad in the forward call. I'm going to begin training from the 5th epoch of the most recent training run with PEFT enabled. Getting PEFT enabled cause OOM errors, I enbaled gradient checkpointing and now I have tons of space on each gpu (still just running DDP through accelerate because I had trouble with ZERO and gradient checkpointing). The iterations are a bit slower (by about 2x) but the gradients of the ia3 layers are visibly training in WandB.

With PEFT active, I ran a full epoch but ran out of memory in the epoch transition, implemented cache clearing and restarted. I also increased teh learning rate to 5e-4 from 1e-4 to see if the model will train faster. (now running on Won's compute...)
I trained for 4 epochs with validation loss decreases each time. Every epoch has OOMed at the end so I was restarting each at the first epoch. I realized that this meant the model was seeing the same batches every epoch. For the fifth epoch, I changed the RNG seed for the dataloaders to start switching up the batches. I also reduced the learning rate to 5e-5 from 1e-4. The validation loss began improving again after this fix, so I resumed training while changing the dataloader RNG seed each epoch.

One idea to augment the dataset would be to double the training examples by including the reverse complement. Thus, each training example would provide a sequence from both the + and - strand. Alternatively, I could add only the RC seqs of gene models which contain isoforms in order to improve performance on that front.

After epoch 5 the validation loss improved only slightly. I computed predictions in the test set and inspected them manually. It appears that the model tends to correctly predict the strand, number of exons, and phase (strange?). Yet it struggles on the coordinates and number and composition of the isoforms. To attempt to improve the coordinate situation, I have attempted to implement a hybrid loss function that uses CrossEntropy for the whole sequence and adds MSE for the numeric sequences (Coordinates). I did not change the RNG seed for epoch 6 so that I can monitor the training by comparing the same batches.
The hybrid loss function was buggy and in fixing it, I came across nuances that make it difficult to work with. I found that the CrossEntropyLoss function allows a weight tensor to apply different weights to certain tokens. So before fixing the hybrid loss function, I am going to try a CrossEntropyLoss with larger weights on numeric tokens. I am also augmenting the datasest with the reverse complement of gene models with alternative splicing, effectively doubling the representation of isoformic sequences in the dataset...I cancelled this run after finding a bug in the GFF reverse complementing function. Fixed.

I decided to see if my coordinates are close enough to allow me to complete the predictions using heuristics. (Start/stop codons, canonical/non-canonical splice junctions, presence of in-frame stop codons...)
On a single test prediction, I obtain the highest accuracy by first finding a high quality donor splice site "AGGT". Then look for the "AG" acceptor site nearest the predicted exon start that does not introduce an in-frame stop codon. I can also mandate that the donor site used preserves the predicted phase of the next cds. The corrections work well up to a point... some coordinate predictions are too far from the truth for me to find the correct splice site. The model seems to struggle most with internal exons.

Next training run:
1. Try scheduled sampling to reduce the distance between the training task and the generation task
2. Clean the dataset by removing transcripts that don't have start or stop codons, or that are not a multiple of three
3. Augment the dataset with reverseComplemented seqeunces for those that have isoforms
4. Use CrossEntropyLoss with higher weights on numeric tokens or explore the hybrid MSE strategy again
5. Freeze the encoder and train only the decoder to increase training speeds?

Seems like only plastidic and mitochondrial genes have strangeness with alternate start/stop codons. These non-canonical genes will be removed from the training set. 

Two epochs of training with increased cross entropy weights on numeric digits did not improve performance. During generation, many more predctions ran to the max decoder size repeating sequences UTRs and numbers. It seems to happen when the prediction for the first CDS is far from corrrect. The checkpoint before I implemented the numeric cross entropy weights has far fewer gibberish predictions, although the coordinate locations and isoform detection need significant improvement. Maybe I can play with the generation parameters and the old checkpoint to improve the predictions?

Checkpoint 5 results Greedy search (before adding RC isoforms and trying cross entropy weights):
#= Summary for dataset: greedy_predictions.pred.sobic.gff3
#     Query mRNAs :    2237 in    1771 loci  (1807 multi-exon transcripts)
#            (31 multi-transcript loci, ~1.3 transcripts per locus)
# Reference mRNAs :    2428 in    1791 loci  (2017 multi-exon)
# Super-loci w/ reference transcripts:     1723
#-----------------| Sensitivity | Precision  |
        Base level:    78.8     |    30.9    |
        Exon level:    17.7     |    20.8    |
      Intron level:    11.5     |    13.4    |
Intron chain level:     4.2     |     4.6    |
  Transcript level:    16.2     |    17.6    |
       Locus level:    21.9     |    22.1    |

     Matching intron chains:      84
       Matching transcripts:     393
              Matching loci:     393

          Missed exons:    1794/10351	( 17.3%)
           Novel exons:    1319/8437	( 15.6%)
        Missed introns:    1582/7691	( 20.6%)
         Novel introns:     489/6629	(  7.4%)
           Missed loci:      56/1791	(  3.1%)
            Novel loci:      47/1771	(  2.7%)

#= Summary for dataset: greedy_predictions.pred.seita.gff3
#     Query mRNAs :    1923 in    1904 loci  (1455 multi-exon transcripts)
#            (18 multi-transcript loci, ~1.0 transcripts per locus)
# Reference mRNAs :    2390 in    1907 loci  (2052 multi-exon)
# Super-loci w/ reference transcripts:     1833
#-----------------| Sensitivity | Precision  |
        Base level:    79.3     |    80.4    |
        Exon level:    19.0     |    21.9    |
      Intron level:    12.7     |    14.9    |
Intron chain level:     4.3     |     6.0    |
  Transcript level:    15.6     |    19.4    |
       Locus level:    19.6     |    19.6    |

     Matching intron chains:      88
       Matching transcripts:     373
              Matching loci:     373

          Missed exons:    1499/9783	( 15.3%)
           Novel exons:    1087/8388	( 13.0%)
        Missed introns:    1608/7604	( 21.1%)
         Novel introns:     530/6474	(  8.2%)
           Missed loci:      73/1907	(  3.8%)
            Novel loci:      70/1904	(  3.7%)

#= Summary for dataset: greedy_predictions.pred.artha.gff3
#     Query mRNAs :    1547 in    1523 loci  (1148 multi-exon transcripts)
#            (20 multi-transcript loci, ~1.0 transcripts per locus)
# Reference mRNAs :    1980 in    1525 loci  (1619 multi-exon)
# Super-loci w/ reference transcripts:     1488
#-----------------| Sensitivity | Precision  |
        Base level:    84.9     |    82.9    |
        Exon level:    23.3     |    26.9    |
      Intron level:    15.2     |    17.7    |
Intron chain level:     5.2     |     7.3    |
  Transcript level:    18.6     |    23.9    |
       Locus level:    24.2     |    24.2    |

     Matching intron chains:      84
       Matching transcripts:     369
              Matching loci:     369

          Missed exons:     787/8628	(  9.1%)
           Novel exons:     568/7247	(  7.8%)
        Missed introns:     915/6654	( 13.8%)
         Novel introns:     247/5710	(  4.3%)
           Missed loci:      36/1525	(  2.4%)
            Novel loci:      34/1523	(  2.2%)

#= Summary for dataset: greedy_predictions.pred.glyma.gff3
#     Query mRNAs :    2892 in    2628 loci  (2272 multi-exon transcripts)
#            (104 multi-transcript loci, ~1.1 transcripts per locus)
# Reference mRNAs :    4285 in    2658 loci  (3672 multi-exon)
# Super-loci w/ reference transcripts:     2591
#-----------------| Sensitivity | Precision  |
        Base level:    77.6     |    49.5    |
        Exon level:    17.0     |    20.4    |
      Intron level:    10.9     |    12.9    |
Intron chain level:     3.1     |     5.1    |
  Transcript level:    14.0     |    20.7    |
       Locus level:    22.5     |    22.7    |

     Matching intron chains:     115
       Matching transcripts:     598
              Matching loci:     598

          Missed exons:    3000/17689	( 17.0%)
           Novel exons:    2102/13648	( 15.4%)
        Missed introns:    2593/12882	( 20.1%)
         Novel introns:     613/10882	(  5.6%)
           Missed loci:      53/2658	(  2.0%)
            Novel loci:      37/2628	(  1.4%)

#= Summary for dataset: greedy_predictions.pred.potri.gff3
#     Query mRNAs :    2039 in    1894 loci  (1566 multi-exon transcripts)
#            (90 multi-transcript loci, ~1.1 transcripts per locus)
# Reference mRNAs :    2925 in    1904 loci  (2473 multi-exon)
# Super-loci w/ reference transcripts:     1855
#-----------------| Sensitivity | Precision  |
        Base level:    78.9     |    78.8    |
        Exon level:    17.6     |    21.3    |
      Intron level:    10.9     |    13.0    |
Intron chain level:     2.9     |     4.5    |
  Transcript level:    13.2     |    19.0    |
       Locus level:    20.3     |    20.4    |

     Matching intron chains:      71
       Matching transcripts:     387
              Matching loci:     387

          Missed exons:    1905/12048	( 15.8%)
           Novel exons:    1223/9318	( 13.1%)
        Missed introns:    1729/8800	( 19.6%)
         Novel introns:     364/7352	(  5.0%)
           Missed loci:      48/1904	(  2.5%)
            Novel loci:      39/1894	(  2.1%)

#= Summary for dataset: greedy_predictions.pred.bradi.gff3
#     Query mRNAs :    2326 in    1904 loci  (1844 multi-exon transcripts)
#            (41 multi-transcript loci, ~1.2 transcripts per locus)
# Reference mRNAs :    2944 in    1918 loci  (2574 multi-exon)
# Super-loci w/ reference transcripts:     1787
#-----------------| Sensitivity | Precision  |
        Base level:    75.7     |    40.2    |
        Exon level:    16.9     |    20.2    |
      Intron level:    11.1     |    13.4    |
Intron chain level:     3.3     |     4.6    |
  Transcript level:    11.8     |    14.9    |
       Locus level:    18.0     |    18.2    |

     Matching intron chains:      84
       Matching transcripts:     346
              Matching loci:     346

          Missed exons:    1752/10526	( 16.6%)
           Novel exons:    1241/8558	( 14.5%)
        Missed introns:    1839/8008	( 23.0%)
         Novel introns:     608/6610	(  9.2%)
           Missed loci:     120/1918	(  6.3%)
            Novel loci:     117/1904	(  6.1%)
  
#= Summary for dataset: greedy_predictions.pred.pp3c.gff3
#     Query mRNAs :    6704 in    1704 loci  (5643 multi-exon transcripts)
#            (836 multi-transcript loci, ~3.9 transcripts per locus)
# Reference mRNAs :    4204 in    1760 loci  (3674 multi-exon)
# Super-loci w/ reference transcripts:     1570
#-----------------| Sensitivity | Precision  |
        Base level:    74.4     |    27.5    |
        Exon level:     9.3     |    12.7    |
      Intron level:     6.1     |     8.3    |
Intron chain level:     0.8     |     0.5    |
  Transcript level:     5.2     |     3.3    |
       Locus level:    11.3     |    11.6    |

     Matching intron chains:      30
       Matching transcripts:     220
              Matching loci:     199

          Missed exons:    2321/13387	( 17.3%)
           Novel exons:    1156/9137	( 12.7%)
        Missed introns:    2605/9158	( 28.4%)
         Novel introns:     601/6788	(  8.9%)
           Missed loci:     169/1760	(  9.6%)
            Novel loci:     133/1704	(  7.8%)


Contrastive search
#= Summary for dataset: test_predictions.pred.sobic.gff3
#     Query mRNAs :    1592 in    1556 loci  (1213 multi-exon transcripts)
#            (27 multi-transcript loci, ~1.0 transcripts per locus)
# Reference mRNAs :    2115 in    1557 loci  (1759 multi-exon)
# Super-loci w/ reference transcripts:     1514
#-----------------| Sensitivity | Precision  |
        Base level:    78.6     |    81.2    |
        Exon level:    17.6     |    21.3    |
      Intron level:    11.6     |    13.8    |
Intron chain level:     3.9     |     5.7    |
  Transcript level:    15.9     |    21.2    |
       Locus level:    21.6     |    21.7    |

     Matching intron chains:      69
       Matching transcripts:     337
              Matching loci:     337

          Missed exons:    1538/8875	( 17.3%)
           Novel exons:    1019/7119	( 14.3%)
        Missed introns:    1455/6560	( 22.2%)
         Novel introns:     371/5543	(  6.7%)
           Missed loci:      42/1557	(  2.7%)
            Novel loci:      40/1556	(  2.6%)

#= Summary for dataset: test_predictions.pred.seita.gff3
#     Query mRNAs :    1673 in    1647 loci  (1243 multi-exon transcripts)
#            (15 multi-transcript loci, ~1.0 transcripts per locus)
# Reference mRNAs :    2089 in    1662 loci  (1791 multi-exon)
# Super-loci w/ reference transcripts:     1585
#-----------------| Sensitivity | Precision  |
        Base level:    78.9     |    34.2    |
        Exon level:    18.7     |    21.7    |
      Intron level:    12.3     |    14.6    |
Intron chain level:     3.6     |     5.2    |
  Transcript level:    15.2     |    19.0    |
       Locus level:    19.1     |    19.2    |

     Matching intron chains:      65
       Matching transcripts:     318
              Matching loci:     318

          Missed exons:    1244/8560	( 14.5%)
           Novel exons:     946/7275	( 13.0%)
        Missed introns:    1459/6666	( 21.9%)
         Novel introns:     400/5604	(  7.1%)
           Missed loci:      63/1662	(  3.8%)
            Novel loci:      61/1647	(  3.7%)

#= Summary for dataset: test_predictions.pred.artha.gff3
#     Query mRNAs :    1305 in    1293 loci  (944 multi-exon transcripts)
#            (11 multi-transcript loci, ~1.0 transcripts per locus)
# Reference mRNAs :    1675 in    1296 loci  (1357 multi-exon)
# Super-loci w/ reference transcripts:     1262
#-----------------| Sensitivity | Precision  |
        Base level:    83.9     |    86.8    |
        Exon level:    23.4     |    27.2    |
      Intron level:    15.2     |    18.1    |
Intron chain level:     5.6     |     8.1    |
  Transcript level:    19.3     |    24.8    |
       Locus level:    25.0     |    25.1    |

     Matching intron chains:      76
       Matching transcripts:     324
              Matching loci:     324

          Missed exons:     766/7217	( 10.6%)
           Novel exons:     484/5966	(  8.1%)
        Missed introns:     854/5556	( 15.4%)
         Novel introns:     194/4667	(  4.2%)
           Missed loci:      34/1296	(  2.6%)
            Novel loci:      31/1293	(  2.4%)

#= Summary for dataset: test_predictions.pred.glyma.gff3
#     Query mRNAs :    2429 in    2286 loci  (1868 multi-exon transcripts)
#            (79 multi-transcript loci, ~1.1 transcripts per locus)
# Reference mRNAs :    3732 in    2301 loci  (3194 multi-exon)
# Super-loci w/ reference transcripts:     2251
#-----------------| Sensitivity | Precision  |
        Base level:    77.7     |    41.3    |
        Exon level:    16.5     |    20.3    |
      Intron level:    10.5     |    12.8    |
Intron chain level:     2.9     |     5.0    |
  Transcript level:    13.9     |    21.4    |
       Locus level:    22.6     |    22.6    |

     Matching intron chains:      93
       Matching transcripts:     520
              Matching loci:     520

          Missed exons:    2601/15370	( 16.9%)
           Novel exons:    1781/11595	( 15.4%)
        Missed introns:    2440/11180	( 21.8%)
         Novel introns:     492/9226	(  5.3%)
           Missed loci:      40/2301	(  1.7%)
            Novel loci:      35/2286	(  1.5%)

#= Summary for dataset: test_predictions.pred.potri.gff3
#     Query mRNAs :    1691 in    1627 loci  (1275 multi-exon transcripts)
#            (46 multi-transcript loci, ~1.0 transcripts per locus)
# Reference mRNAs :    2492 in    1629 loci  (2097 multi-exon)
# Super-loci w/ reference transcripts:     1593
#-----------------| Sensitivity | Precision  |
        Base level:    78.9     |    80.6    |
        Exon level:    17.6     |    21.1    |
      Intron level:    11.1     |    13.3    |
Intron chain level:     2.8     |     4.5    |
  Transcript level:    13.4     |    19.8    |
       Locus level:    20.5     |    20.5    |

     Matching intron chains:      58
       Matching transcripts:     334
              Matching loci:     334

          Missed exons:    1585/10264	( 15.4%)
           Novel exons:    1086/7953	( 13.7%)
        Missed introns:    1472/7496	( 19.6%)
         Novel introns:     288/6293	(  4.6%)
           Missed loci:      36/1629	(  2.2%)
            Novel loci:      34/1627	(  2.1%)

#= Summary for dataset: test_predictions.pred.bradi.gff3
#     Query mRNAs :    1684 in    1638 loci  (1254 multi-exon transcripts)
#            (30 multi-transcript loci, ~1.0 transcripts per locus)
# Reference mRNAs :    2491 in    1643 loci  (2168 multi-exon)
# Super-loci w/ reference transcripts:     1541
#-----------------| Sensitivity | Precision  |
        Base level:    75.5     |    74.0    |
        Exon level:    16.9     |    20.6    |
      Intron level:    11.6     |    14.4    |
Intron chain level:     3.3     |     5.7    |
  Transcript level:    12.2     |    18.1    |
       Locus level:    18.6     |    18.6    |

     Matching intron chains:      72
       Matching transcripts:     305
              Matching loci:     305

          Missed exons:    1559/9062	( 17.2%)
           Novel exons:    1002/7233	( 13.9%)
        Missed introns:    1650/6924	( 23.8%)
         Novel introns:     475/5574	(  8.5%)
           Missed loci:      97/1643	(  5.9%)
            Novel loci:      96/1638	(  5.9%)

#= Summary for dataset: test_predictions.pred.pp3c.gff3
#     Query mRNAs :    3129 in    1529 loci  (2552 multi-exon transcripts)
#            (721 multi-transcript loci, ~2.0 transcripts per locus)
# Reference mRNAs :    3681 in    1539 loci  (3226 multi-exon)
# Super-loci w/ reference transcripts:     1405
#-----------------| Sensitivity | Precision  |
        Base level:    77.0     |    55.3    |
        Exon level:     9.6     |    13.3    |
      Intron level:     6.4     |     9.0    |
Intron chain level:     0.9     |     1.1    |
  Transcript level:     5.4     |     6.3    |
       Locus level:    11.6     |    11.6    |

     Matching intron chains:      29
       Matching transcripts:     198
              Matching loci:     178

          Missed exons:    1781/11784	( 15.1%)
           Novel exons:     978/7925	( 12.3%)
        Missed introns:    2460/8105	( 30.4%)
         Novel introns:     459/5807	(  7.9%)
           Missed loci:     128/1539	(  8.3%)
            Novel loci:     123/1529	(  8.0%)

Beam search:
#= Summary for dataset: beam_predictions.pred.sobic.gff3
#     Query mRNAs :    1965 in    1716 loci  (1528 multi-exon transcripts)
#            (167 multi-transcript loci, ~1.1 transcripts per locus)
# Reference mRNAs :    2330 in    1715 loci  (1931 multi-exon)
# Super-loci w/ reference transcripts:     1668
#-----------------| Sensitivity | Precision  |
        Base level:    79.8     |    80.8    |
        Exon level:    18.2     |    21.5    |
      Intron level:    11.3     |    13.2    |
Intron chain level:     3.6     |     4.5    |
  Transcript level:    16.0     |    19.0    |
       Locus level:    21.7     |    21.7    |

     Matching intron chains:      69
       Matching transcripts:     373
              Matching loci:     373

          Missed exons:    1592/9817	( 16.2%)
           Novel exons:    1134/7983	( 14.2%)
        Missed introns:    1523/7266	( 21.0%)
         Novel introns:     360/6233	(  5.8%)
           Missed loci:      47/1715	(  2.7%)
            Novel loci:      47/1716	(  2.7%)

#= Summary for dataset: beam_predictions.pred.seita.gff3
#     Query mRNAs :    1971 in    1813 loci  (1493 multi-exon transcripts)
#            (146 multi-transcript loci, ~1.1 transcripts per locus)
# Reference mRNAs :    2277 in    1814 loci  (1951 multi-exon)
# Super-loci w/ reference transcripts:     1750
#-----------------| Sensitivity | Precision  |
        Base level:    79.7     |    77.3    |
        Exon level:    18.6     |    21.5    |
      Intron level:    12.0     |    14.2    |
Intron chain level:     3.5     |     4.6    |
  Transcript level:    15.4     |    17.8    |
       Locus level:    19.3     |    19.4    |

     Matching intron chains:      69
       Matching transcripts:     351
              Matching loci:     351

          Missed exons:    1350/9294	( 14.5%)
           Novel exons:     967/7954	( 12.2%)
        Missed introns:    1578/7226	( 21.8%)
         Novel introns:     446/6096	(  7.3%)
           Missed loci:      63/1814	(  3.5%)
            Novel loci:      62/1813	(  3.4%)

#= Summary for dataset: beam_predictions.pred.artha.gff3
#     Query mRNAs :    1556 in    1450 loci  (1161 multi-exon transcripts)
#            (96 multi-transcript loci, ~1.1 transcripts per locus)
# Reference mRNAs :    1880 in    1452 loci  (1537 multi-exon)
# Super-loci w/ reference transcripts:     1417
#-----------------| Sensitivity | Precision  |
        Base level:    85.0     |    82.2    |
        Exon level:    23.8     |    27.3    |
      Intron level:    16.0     |    18.7    |
Intron chain level:     5.4     |     7.1    |
  Transcript level:    18.9     |    22.9    |
       Locus level:    24.5     |    24.6    |

     Matching intron chains:      83
       Matching transcripts:     356
              Matching loci:     356

          Missed exons:     775/8198	(  9.5%)
           Novel exons:     543/6945	(  7.8%)
        Missed introns:     892/6333	( 14.1%)
         Novel introns:     220/5412	(  4.1%)
           Missed loci:      34/1452	(  2.3%)
            Novel loci:      33/1450	(  2.3%)

#= Summary for dataset: beam_predictions.pred.glyma.gff3
#     Query mRNAs :    2965 in    2534 loci  (2325 multi-exon transcripts)
#            (359 multi-transcript loci, ~1.2 transcripts per locus)
# Reference mRNAs :    4101 in    2534 loci  (3514 multi-exon)
# Super-loci w/ reference transcripts:     2499
#-----------------| Sensitivity | Precision  |
        Base level:    78.5     |    77.7    |
        Exon level:    16.5     |    19.9    |
      Intron level:    10.2     |    12.1    |
Intron chain level:     2.9     |     4.3    |
  Transcript level:    14.0     |    19.3    |
       Locus level:    22.6     |    22.6    |

     Matching intron chains:     101
       Matching transcripts:     573
              Matching loci:     573

          Missed exons:    2718/16898	( 16.1%)
           Novel exons:    1907/13075	( 14.6%)
        Missed introns:    2531/12291	( 20.6%)
         Novel introns:     477/10356	(  4.6%)
           Missed loci:      35/2534	(  1.4%)
            Novel loci:      35/2534	(  1.4%)

#= Summary for dataset: beam_predictions.pred.potri.gff3
#     Query mRNAs :    2060 in    1793 loci  (1583 multi-exon transcripts)
#            (219 multi-transcript loci, ~1.1 transcripts per locus)
# Reference mRNAs :    2757 in    1804 loci  (2327 multi-exon)
# Super-loci w/ reference transcripts:     1756
#-----------------| Sensitivity | Precision  |
        Base level:    79.7     |    58.1    |
        Exon level:    17.4     |    20.8    |
      Intron level:    10.5     |    12.5    |
Intron chain level:     2.8     |     4.0    |
  Transcript level:    13.4     |    18.0    |
       Locus level:    20.5     |    20.5    |

     Matching intron chains:      64
       Matching transcripts:     370
              Matching loci:     370

          Missed exons:    1650/11375	( 14.5%)
           Novel exons:    1077/8911	( 12.1%)
        Missed introns:    1602/8313	( 19.3%)
         Novel introns:     302/6974	(  4.3%)
           Missed loci:      38/1804	(  2.1%)
            Novel loci:      37/1793	(  2.1%)

#= Summary for dataset: beam_predictions.pred.bradi.gff3
#     Query mRNAs :    2279 in    1816 loci  (1773 multi-exon transcripts)
#            (196 multi-transcript loci, ~1.3 transcripts per locus)
# Reference mRNAs :    2801 in    1818 loci  (2445 multi-exon)
# Super-loci w/ reference transcripts:     1701
#-----------------| Sensitivity | Precision  |
        Base level:    76.4     |    77.8    |
        Exon level:    16.7     |    20.1    |
      Intron level:    10.9     |    13.4    |
Intron chain level:     3.4     |     4.6    |
  Transcript level:    12.1     |    14.9    |
       Locus level:    18.7     |    18.7    |

     Matching intron chains:      82
       Matching transcripts:     340
              Matching loci:     340

          Missed exons:    1677/10000	( 16.8%)
           Novel exons:    1168/8062	( 14.5%)
        Missed introns:    1788/7602	( 23.5%)
         Novel introns:     447/6177	(  7.2%)
           Missed loci:     111/1818	(  6.1%)
            Novel loci:     114/1816	(  6.3%)

#= Summary for dataset: beam_predictions.pred.pp3c.gff3
#     Query mRNAs :    2978 in    1670 loci  (2281 multi-exon transcripts)
#            (910 multi-transcript loci, ~1.8 transcripts per locus)
# Reference mRNAs :    3997 in    1676 loci  (3498 multi-exon)
# Super-loci w/ reference transcripts:     1544
#-----------------| Sensitivity | Precision  |
        Base level:    78.4     |    36.5    |
        Exon level:    10.0     |    13.2    |
      Intron level:     6.4     |     8.6    |
Intron chain level:     0.8     |     1.2    |
  Transcript level:     5.5     |     7.3    |
       Locus level:    11.6     |    11.7    |

     Matching intron chains:      28
       Matching transcripts:     218
              Matching loci:     195

          Missed exons:    1751/12814	( 13.7%)
           Novel exons:    1141/9520	( 12.0%)
        Missed introns:    2476/8796	( 28.1%)
         Novel introns:     526/6536	(  8.0%)
           Missed loci:     126/1676	(  7.5%)
            Novel loci:     125/1670	(  7.5%)

Epoch8 predictoins:
BEAM
#= Summary for dataset: beam_predictions.pred.sobic.gff3
#     Query mRNAs :     476 in     457 loci  (353 multi-exon transcripts)
#            (14 multi-transcript loci, ~1.0 transcripts per locus)
# Reference mRNAs :     570 in     457 loci  (467 multi-exon)
# Super-loci w/ reference transcripts:      444
#-----------------| Sensitivity | Precision  |
        Base level:    85.8     |    84.7    |
        Exon level:    30.1     |    33.7    |
      Intron level:    22.5     |    24.9    |
Intron chain level:     6.4     |     8.5    |
  Transcript level:    19.8     |    23.7    |
       Locus level:    24.7     |    24.7    |

     Matching intron chains:      30
       Matching transcripts:     113
              Matching loci:     113

          Missed exons:     278/2536	( 11.0%)
           Novel exons:     235/2193	( 10.7%)
        Missed introns:     229/1911	( 12.0%)
         Novel introns:      63/1725	(  3.7%)
           Missed loci:      13/457	(  2.8%)
            Novel loci:      13/457	(  2.8%)

#= Summary for dataset: beam_predictions.pred.seita.gff3
#     Query mRNAs :     496 in     482 loci  (370 multi-exon transcripts)
#            (12 multi-transcript loci, ~1.0 transcripts per locus)
# Reference mRNAs :     586 in     484 loci  (505 multi-exon)
# Super-loci w/ reference transcripts:      463
#-----------------| Sensitivity | Precision  |
        Base level:    82.4     |    27.1    |
        Exon level:    28.5     |    32.9    |
      Intron level:    21.4     |    25.4    |
Intron chain level:     8.3     |    11.4    |
  Transcript level:    18.9     |    22.4    |
       Locus level:    22.9     |    23.0    |

     Matching intron chains:      42
       Matching transcripts:     111
              Matching loci:     111

          Missed exons:     301/2508	( 12.0%)
           Novel exons:     232/2151	( 10.8%)
        Missed introns:     332/1971	( 16.8%)
         Novel introns:      69/1660	(  4.2%)
           Missed loci:      19/484	(  3.9%)
            Novel loci:      19/482	(  3.9%)
#= Summary for dataset: beam_predictions.pred.artha.gff3
#     Query mRNAs :     395 in     390 loci  (284 multi-exon transcripts)
#            (5 multi-transcript loci, ~1.0 transcripts per locus)
# Reference mRNAs :     517 in     390 loci  (415 multi-exon)
# Super-loci w/ reference transcripts:      386
#-----------------| Sensitivity | Precision  |
        Base level:    88.6     |    89.1    |
        Exon level:    34.5     |    39.9    |
      Intron level:    25.7     |    30.0    |
Intron chain level:    11.3     |    16.5    |
  Transcript level:    24.0     |    31.4    |
       Locus level:    31.8     |    31.8    |

     Matching intron chains:      47
       Matching transcripts:     124
              Matching loci:     124

          Missed exons:     166/2150	(  7.7%)
           Novel exons:      99/1794	(  5.5%)
        Missed introns:     194/1640	( 11.8%)
         Novel introns:      22/1402	(  1.6%)
           Missed loci:       4/390	(  1.0%)
            Novel loci:       4/390	(  1.0%)
#= Summary for dataset: beam_predictions.pred.glyma.gff3
#     Query mRNAs :     716 in     672 loci  (536 multi-exon transcripts)
#            (28 multi-transcript loci, ~1.1 transcripts per locus)
# Reference mRNAs :    1018 in     674 loci  (856 multi-exon)
# Super-loci w/ reference transcripts:      668
#-----------------| Sensitivity | Precision  |
        Base level:    83.1     |    85.0    |
        Exon level:    28.0     |    33.0    |
      Intron level:    21.7     |    25.2    |
Intron chain level:     6.1     |     9.7    |
  Transcript level:    19.2     |    27.2    |
       Locus level:    28.9     |    29.0    |

     Matching intron chains:      52
       Matching transcripts:     195
              Matching loci:     195

          Missed exons:     555/4393	( 12.6%)
           Novel exons:     365/3526	( 10.4%)
        Missed introns:     456/3263	( 14.0%)
         Novel introns:      88/2811	(  3.1%)
           Missed loci:       6/674	(  0.9%)
            Novel loci:       4/672	(  0.6%)
#= Summary for dataset: beam_predictions.pred.potri.gff3
#     Query mRNAs :     479 in     460 loci  (355 multi-exon transcripts)
#            (17 multi-transcript loci, ~1.0 transcripts per locus)
# Reference mRNAs :     688 in     460 loci  (571 multi-exon)
# Super-loci w/ reference transcripts:      457
#-----------------| Sensitivity | Precision  |
        Base level:    86.1     |    85.9    |
        Exon level:    30.9     |    36.6    |
      Intron level:    24.3     |    28.2    |
Intron chain level:     6.5     |    10.4    |
  Transcript level:    18.6     |    26.7    |
       Locus level:    27.8     |    27.8    |

     Matching intron chains:      37
       Matching transcripts:     128
              Matching loci:     128

          Missed exons:     240/2787	(  8.6%)
           Novel exons:     162/2210	(  7.3%)
        Missed introns:     257/2010	( 12.8%)
         Novel introns:      52/1733	(  3.0%)
           Missed loci:       3/460	(  0.7%)
            Novel loci:       3/460	(  0.7%)
#= Summary for dataset: beam_predictions.pred.bradi.gff3
#     Query mRNAs :     546 in     504 loci  (414 multi-exon transcripts)
#            (14 multi-transcript loci, ~1.1 transcripts per locus)
# Reference mRNAs :     762 in     505 loci  (671 multi-exon)
# Super-loci w/ reference transcripts:      474
#-----------------| Sensitivity | Precision  |
        Base level:    79.6     |    80.6    |
        Exon level:    24.6     |    29.6    |
      Intron level:    19.0     |    23.1    |
Intron chain level:     5.2     |     8.5    |
  Transcript level:    13.6     |    19.0    |
       Locus level:    20.6     |    20.6    |

     Matching intron chains:      35
       Matching transcripts:     104
              Matching loci:     104

          Missed exons:     405/2727	( 14.9%)
           Novel exons:     265/2201	( 12.0%)
        Missed introns:     353/2055	( 17.2%)
         Novel introns:      84/1692	(  5.0%)
           Missed loci:      30/505	(  5.9%)
            Novel loci:      30/504	(  6.0%)
#= Summary for dataset: beam_predictions.pred.pp3c.gff3
#     Query mRNAs :     664 in     454 loci  (493 multi-exon transcripts)
#            (175 multi-transcript loci, ~1.5 transcripts per locus)
# Reference mRNAs :    1069 in     454 loci  (919 multi-exon)
# Super-loci w/ reference transcripts:      421
#-----------------| Sensitivity | Precision  |
        Base level:    82.2     |    80.6    |
        Exon level:    16.7     |    23.8    |
      Intron level:    13.1     |    18.3    |
Intron chain level:     2.0     |     3.7    |
  Transcript level:     8.0     |    12.8    |
       Locus level:    17.0     |    17.0    |

     Matching intron chains:      18
       Matching transcripts:      85
              Matching loci:      77

          Missed exons:     348/3506	(  9.9%)
           Novel exons:     188/2442	(  7.7%)
        Missed introns:     620/2418	( 25.6%)
         Novel introns:      82/1730	(  4.7%)
           Missed loci:      33/454	(  7.3%)
            Novel loci:      33/454	(  7.3%)
GREEDY:
#= Summary for dataset: greedy_predictions.pred.sobic.gff3
#     Query mRNAs :     523 in     509 loci  (402 multi-exon transcripts)
#            (9 multi-transcript loci, ~1.0 transcripts per locus)
# Reference mRNAs :     637 in     512 loci  (524 multi-exon)
# Super-loci w/ reference transcripts:      493
#-----------------| Sensitivity | Precision  |
        Base level:    85.3     |    84.4    |
        Exon level:    30.3     |    33.6    |
      Intron level:    23.0     |    25.4    |
Intron chain level:     7.4     |     9.7    |
  Transcript level:    19.9     |    24.3    |
       Locus level:    24.8     |    25.0    |

     Matching intron chains:      39
       Matching transcripts:     127
              Matching loci:     127

          Missed exons:     291/2837	( 10.3%)
           Novel exons:     236/2466	(  9.6%)
        Missed introns:     265/2143	( 12.4%)
         Novel introns:      77/1943	(  4.0%)
           Missed loci:      19/512	(  3.7%)
            Novel loci:      16/509	(  3.1%)

 Total union super-loci across all input datasets: 509
523 out of 523 consensus transcripts written in gffcmp.annotated.gtf (0 discarded as redundant)
#= Summary for dataset: greedy_predictions.pred.seita.gff3
#     Query mRNAs :     777 in     540 loci  (631 multi-exon transcripts)
#            (3 multi-transcript loci, ~1.4 transcripts per locus)
# Reference mRNAs :     661 in     545 loci  (565 multi-exon)
# Super-loci w/ reference transcripts:      522
#-----------------| Sensitivity | Precision  |
        Base level:    82.0     |    83.5    |
        Exon level:    27.8     |    32.9    |
      Intron level:    20.6     |    25.1    |
Intron chain level:     8.7     |     7.8    |
  Transcript level:    20.1     |    17.1    |
       Locus level:    24.4     |    24.6    |

     Matching intron chains:      49
       Matching transcripts:     133
              Matching loci:     133

          Missed exons:     421/2814	( 15.0%)
           Novel exons:     228/2355	(  9.7%)
        Missed introns:     431/2208	( 19.5%)
         Novel introns:      68/1815	(  3.7%)
           Missed loci:      23/545	(  4.2%)
            Novel loci:      18/540	(  3.3%)

 Total union super-loci across all input datasets: 540
777 out of 777 consensus transcripts written in gffcmp.annotated.gtf (0 discarded as redundant)
#= Summary for dataset: greedy_predictions.pred.artha.gff3
#     Query mRNAs :     446 in     439 loci  (326 multi-exon transcripts)
#            (7 multi-transcript loci, ~1.0 transcripts per locus)
# Reference mRNAs :     581 in     440 loci  (470 multi-exon)
# Super-loci w/ reference transcripts:      432
#-----------------| Sensitivity | Precision  |
        Base level:    87.8     |    89.3    |
        Exon level:    34.0     |    39.3    |
      Intron level:    25.3     |    29.5    |
Intron chain level:    10.4     |    15.0    |
  Transcript level:    22.7     |    29.6    |
       Locus level:    30.0     |    30.1    |

     Matching intron chains:      49
       Matching transcripts:     132
              Matching loci:     132

          Missed exons:     199/2474	(  8.0%)
           Novel exons:     107/2067	(  5.2%)
        Missed introns:     223/1895	( 11.8%)
         Novel introns:      30/1628	(  1.8%)
           Missed loci:       8/440	(  1.8%)
            Novel loci:       7/439	(  1.6%)

#= Summary for dataset: greedy_predictions.pred.glyma.gff3
#     Query mRNAs :     769 in     728 loci  (583 multi-exon transcripts)
#            (24 multi-transcript loci, ~1.1 transcripts per locus)
# Reference mRNAs :    1138 in     743 loci  (956 multi-exon)
# Super-loci w/ reference transcripts:      722
#-----------------| Sensitivity | Precision  |
        Base level:    79.8     |    86.2    |
        Exon level:    27.5     |    33.5    |
      Intron level:    21.1     |    25.4    |
Intron chain level:     5.8     |     9.4    |
  Transcript level:    18.5     |    27.3    |
       Locus level:    28.3     |    28.8    |

     Matching intron chains:      55
       Matching transcripts:     210
              Matching loci:     210

          Missed exons:     758/4848	( 15.6%)
           Novel exons:     381/3751	( 10.2%)
        Missed introns:     606/3605	( 16.8%)
         Novel introns:      79/2995	(  2.6%)
           Missed loci:      21/743	(  2.8%)
            Novel loci:       6/728	(  0.8%)
#= Summary for dataset: greedy_predictions.pred.potri.gff3
#     Query mRNAs :     535 in     516 loci  (401 multi-exon transcripts)
#            (19 multi-transcript loci, ~1.0 transcripts per locus)
# Reference mRNAs :     788 in     522 loci  (657 multi-exon)
# Super-loci w/ reference transcripts:      513
#-----------------| Sensitivity | Precision  |
        Base level:    85.0     |    87.2    |
        Exon level:    30.9     |    37.2    |
      Intron level:    24.5     |    28.8    |
Intron chain level:     6.5     |    10.7    |
  Transcript level:    17.8     |    26.2    |
       Locus level:    26.8     |    27.1    |

     Matching intron chains:      43
       Matching transcripts:     140
              Matching loci:     140

          Missed exons:     318/3184	( 10.0%)
           Novel exons:     171/2486	(  6.9%)
        Missed introns:     318/2297	( 13.8%)
         Novel introns:      58/1958	(  3.0%)
           Missed loci:       9/522	(  1.7%)
            Novel loci:       3/516	(  0.6%)

#= Summary for dataset: greedy_predictions.pred.bradi.gff3
#     Query mRNAs :     566 in     548 loci  (423 multi-exon transcripts)
#            (16 multi-transcript loci, ~1.0 transcripts per locus)
# Reference mRNAs :     838 in     555 loci  (737 multi-exon)
# Super-loci w/ reference transcripts:      517
#-----------------| Sensitivity | Precision  |
        Base level:    78.7     |    80.9    |
        Exon level:    24.3     |    29.2    |
      Intron level:    18.7     |    22.7    |
Intron chain level:     4.2     |     7.3    |
  Transcript level:    13.1     |    19.4    |
       Locus level:    19.8     |    20.1    |

     Matching intron chains:      31
       Matching transcripts:     110
              Matching loci:     110

          Missed exons:     443/2992	( 14.8%)
           Novel exons:     295/2407	( 12.3%)
        Missed introns:     394/2246	( 17.5%)
         Novel introns:     103/1848	(  5.6%)
           Missed loci:      36/555	(  6.5%)
            Novel loci:      31/548	(  5.7%)
#= Summary for dataset: greedy_predictions.pred.pp3c.gff3
#     Query mRNAs :    1013 in     498 loci  (689 multi-exon transcripts)
#            (201 multi-transcript loci, ~2.0 transcripts per locus)
# Reference mRNAs :    1222 in     512 loci  (1059 multi-exon)
# Super-loci w/ reference transcripts:      462
#-----------------| Sensitivity | Precision  |
        Base level:    78.5     |    37.1    |
        Exon level:    16.2     |    23.3    |
      Intron level:    13.2     |    18.6    |
Intron chain level:     1.8     |     2.8    |
  Transcript level:     7.4     |     9.0    |
       Locus level:    16.4     |    16.9    |

     Matching intron chains:      19
       Matching transcripts:      91
              Matching loci:      84

          Missed exons:     616/3954	( 15.6%)
           Novel exons:     241/2585	(  9.3%)
        Missed introns:     755/2709	( 27.9%)
         Novel introns:     116/1923	(  6.0%)
           Missed loci:      50/512	(  9.8%)
            Novel loci:      36/498	(  7.2%)

CONTRASTIVE:
#= Summary for dataset: contrast_predictions.pred.sobic.gff3
#     Query mRNAs :     728 in     552 loci  (589 multi-exon transcripts)
#            (80 multi-transcript loci, ~1.3 transcripts per locus)
# Reference mRNAs :     687 in     552 loci  (559 multi-exon)
# Super-loci w/ reference transcripts:      531
#-----------------| Sensitivity | Precision  |
        Base level:    84.0     |    77.8    |
        Exon level:    24.2     |    25.0    |
      Intron level:    17.5     |    18.5    |
Intron chain level:     6.1     |     5.8    |
  Transcript level:    19.7     |    18.5    |
       Locus level:    24.5     |    24.5    |

     Matching intron chains:      34
       Matching transcripts:     135
              Matching loci:     135

          Missed exons:     371/3037	( 12.2%)
           Novel exons:     366/2846	( 12.9%)
        Missed introns:     259/2291	( 11.3%)
         Novel introns:     121/2162	(  5.6%)
           Missed loci:      21/552	(  3.8%)
            Novel loci:      21/552	(  3.8%)
#= Summary for dataset: contrast_predictions.pred.seita.gff3
#     Query mRNAs :     654 in     573 loci  (499 multi-exon transcripts)
#            (58 multi-transcript loci, ~1.1 transcripts per locus)
# Reference mRNAs :     696 in     573 loci  (596 multi-exon)
# Super-loci w/ reference transcripts:      552
#-----------------| Sensitivity | Precision  |
        Base level:    81.3     |    76.5    |
        Exon level:    23.6     |    26.4    |
      Intron level:    16.9     |    19.5    |
Intron chain level:     6.9     |     8.2    |
  Transcript level:    18.2     |    19.4    |
       Locus level:    22.2     |    22.2    |

     Matching intron chains:      41
       Matching transcripts:     127
              Matching loci:     127

          Missed exons:     421/2963	( 14.2%)
           Novel exons:     320/2669	( 12.0%)
        Missed introns:     397/2325	( 17.1%)
         Novel introns:     142/2019	(  7.0%)
           Missed loci:      21/573	(  3.7%)
            Novel loci:      21/573	(  3.7%)
#= Summary for dataset: contrast_predictions.pred.artha.gff3
#     Query mRNAs :     542 in     455 loci  (416 multi-exon transcripts)
#            (58 multi-transcript loci, ~1.2 transcripts per locus)
# Reference mRNAs :     604 in     456 loci  (489 multi-exon)
# Super-loci w/ reference transcripts:      445
#-----------------| Sensitivity | Precision  |
        Base level:    86.7     |    86.5    |
        Exon level:    27.7     |    31.0    |
      Intron level:    19.5     |    21.9    |
Intron chain level:     8.0     |     9.4    |
  Transcript level:    20.9     |    23.2    |
       Locus level:    27.6     |    27.7    |

     Matching intron chains:      39
       Matching transcripts:     126
              Matching loci:     126

          Missed exons:     190/2564	(  7.4%)
           Novel exons:     122/2265	(  5.4%)
        Missed introns:     238/1966	( 12.1%)
         Novel introns:      46/1747	(  2.6%)
           Missed loci:      11/456	(  2.4%)
            Novel loci:      10/455	(  2.2%)
#= Summary for dataset: contrast_predictions.pred.glyma.gff3
#     Query mRNAs :    1152 in     778 loci  (939 multi-exon transcripts)
#            (169 multi-transcript loci, ~1.5 transcripts per locus)
# Reference mRNAs :    1196 in     780 loci  (1006 multi-exon)
# Super-loci w/ reference transcripts:      762
#-----------------| Sensitivity | Precision  |
        Base level:    81.3     |    73.4    |
        Exon level:    22.8     |    24.9    |
      Intron level:    17.4     |    18.8    |
Intron chain level:     4.9     |     5.2    |
  Transcript level:    17.1     |    17.7    |
       Locus level:    26.2     |    26.2    |

     Matching intron chains:      49
       Matching transcripts:     204
              Matching loci:     204

          Missed exons:     710/5086	( 14.0%)
           Novel exons:     575/4580	( 12.6%)
        Missed introns:     493/3778	( 13.0%)
         Novel introns:     206/3486	(  5.9%)
           Missed loci:      18/780	(  2.3%)
            Novel loci:      16/778	(  2.1%)
#= Summary for dataset: contrast_predictions.pred.potri.gff3
#     Query mRNAs :     734 in     546 loci  (586 multi-exon transcripts)
#            (103 multi-transcript loci, ~1.3 transcripts per locus)
# Reference mRNAs :     815 in     546 loci  (678 multi-exon)
# Super-loci w/ reference transcripts:      538
#-----------------| Sensitivity | Precision  |
        Base level:    85.0     |    81.2    |
        Exon level:    26.7     |    29.3    |
      Intron level:    18.7     |    20.5    |
Intron chain level:     4.7     |     5.5    |
  Transcript level:    16.4     |    18.3    |
       Locus level:    24.5     |    24.5    |

     Matching intron chains:      32
       Matching transcripts:     134
              Matching loci:     134

          Missed exons:     338/3320	( 10.2%)
           Novel exons:     274/2897	(  9.5%)
        Missed introns:     285/2403	( 11.9%)
         Novel introns:     104/2193	(  4.7%)
           Missed loci:       8/546	(  1.5%)
            Novel loci:       8/546	(  1.5%)

#= Summary for dataset: contrast_predictions.pred.bradi.gff3
#     Query mRNAs :     730 in     582 loci  (573 multi-exon transcripts)
#            (86 multi-transcript loci, ~1.3 transcripts per locus)
# Reference mRNAs :     872 in     584 loci  (765 multi-exon)
# Super-loci w/ reference transcripts:      547
#-----------------| Sensitivity | Precision  |
        Base level:    77.7     |    59.7    |
        Exon level:    20.5     |    23.1    |
      Intron level:    14.0     |    16.4    |
Intron chain level:     3.9     |     5.2    |
  Transcript level:    12.7     |    15.2    |
       Locus level:    19.0     |    19.1    |

     Matching intron chains:      30
       Matching transcripts:     111
              Matching loci:     111

          Missed exons:     483/3142	( 15.4%)
           Novel exons:     376/2736	( 13.7%)
        Missed introns:     375/2363	( 15.9%)
         Novel introns:     133/2028	(  6.6%)
           Missed loci:      35/584	(  6.0%)
            Novel loci:      35/582	(  6.0%)
#= Summary for dataset: contrast_predictions.pred.pp3c.gff3
#     Query mRNAs :     999 in     535 loci  (782 multi-exon transcripts)
#            (229 multi-transcript loci, ~1.9 transcripts per locus)
# Reference mRNAs :    1265 in     535 loci  (1099 multi-exon)
# Super-loci w/ reference transcripts:      487
#-----------------| Sensitivity | Precision  |
        Base level:    80.3     |    78.4    |
        Exon level:    12.1     |    15.0    |
      Intron level:     9.3     |    11.8    |
Intron chain level:     0.7     |     1.0    |
  Transcript level:     6.0     |     7.6    |
       Locus level:    13.1     |    13.1    |

     Matching intron chains:       8
       Matching transcripts:      76
              Matching loci:      70

          Missed exons:     520/4132	( 12.6%)
           Novel exons:     306/3313	(  9.2%)
        Missed introns:     660/2840	( 23.2%)
         Novel introns:     172/2244	(  7.7%)
           Missed loci:      48/535	(  9.0%)
            Novel loci:      48/535	(  9.0%)

Epoch9
Contrastive Search:
#= Summary for dataset: contrast_predictions.pred.artha.gff3
#     Query mRNAs :    2022 in    1770 loci  (1532 multi-exon transcripts)
#            (180 multi-transcript loci, ~1.1 transcripts per locus)
# Reference mRNAs :    2282 in    1775 loci  (1858 multi-exon)
# Super-loci w/ reference transcripts:     1740
#-----------------| Sensitivity | Precision  |
        Base level:    87.4     |    87.8    |
        Exon level:    29.3     |    32.4    |
      Intron level:    21.3     |    24.0    |
Intron chain level:     7.8     |     9.5    |
  Transcript level:    21.6     |    24.4    |
       Locus level:    27.8     |    27.9    |

     Matching intron chains:     145
       Matching transcripts:     494
              Matching loci:     494

          Missed exons:     755/10018	(  7.5%)
           Novel exons:     556/8863	(  6.3%)
        Missed introns:     873/7753	( 11.3%)
         Novel introns:     166/6885	(  2.4%)
           Missed loci:      33/1775	(  1.9%)
            Novel loci:      30/1770	(  1.7%)
#= Summary for dataset: contrast_predictions.pred.glyma.gff3
#     Query mRNAs :    3988 in    3038 loci  (3235 multi-exon transcripts)
#            (478 multi-transcript loci, ~1.3 transcripts per locus)
# Reference mRNAs :    4935 in    3050 loci  (4232 multi-exon)
# Super-loci w/ reference transcripts:     3000
#-----------------| Sensitivity | Precision  |
        Base level:    82.4     |    58.7    |
        Exon level:    24.6     |    28.1    |
      Intron level:    18.9     |    21.4    |
Intron chain level:     5.6     |     7.3    |
  Transcript level:    16.4     |    20.3    |
       Locus level:    26.5     |    26.5    |

     Matching intron chains:     237
       Matching transcripts:     808
              Matching loci:     808

          Missed exons:    2663/20273	( 13.1%)
           Novel exons:    2029/16940	( 12.0%)
        Missed introns:    2030/14718	( 13.8%)
         Novel introns:     531/13030	(  4.1%)
           Missed loci:      42/3050	(  1.4%)
            Novel loci:      38/3038	(  1.3%)
#= Summary for dataset: contrast_predictions.pred.sobic.gff3
#     Query mRNAs :    2400 in    2047 loci  (1855 multi-exon transcripts)
#            (235 multi-transcript loci, ~1.2 transcripts per locus)
# Reference mRNAs :    2778 in    2051 loci  (2295 multi-exon)
# Super-loci w/ reference transcripts:     1979
#-----------------| Sensitivity | Precision  |
        Base level:    82.9     |    77.5    |
        Exon level:    26.0     |    29.3    |
      Intron level:    19.3     |    21.7    |
Intron chain level:     7.0     |     8.6    |
  Transcript level:    19.3     |    22.3    |
       Locus level:    26.1     |    26.1    |

     Matching intron chains:     160
       Matching transcripts:     535
              Matching loci:     535

          Missed exons:    1526/11791	( 12.9%)
           Novel exons:    1217/10139	( 12.0%)
        Missed introns:    1348/8743	( 15.4%)
         Novel introns:     470/7771	(  6.0%)
           Missed loci:      69/2051	(  3.4%)
            Novel loci:      67/2047	(  3.3%)
#= Summary for dataset: contrast_predictions.pred.potri.gff3
#     Query mRNAs :    2695 in    2171 loci  (2125 multi-exon transcripts)
#            (313 multi-transcript loci, ~1.2 transcripts per locus)
# Reference mRNAs :    3332 in    2183 loci  (2811 multi-exon)
# Super-loci w/ reference transcripts:     2131
#-----------------| Sensitivity | Precision  |
        Base level:    82.8     |    44.4    |
        Exon level:    25.4     |    29.0    |
      Intron level:    19.2     |    21.9    |
Intron chain level:     5.3     |     7.0    |
  Transcript level:    16.1     |    20.0    |
       Locus level:    24.6     |    24.6    |

     Matching intron chains:     149
       Matching transcripts:     538
              Matching loci:     537

          Missed exons:    1595/13726	( 11.6%)
           Novel exons:    1151/11524	( 10.0%)
        Missed introns:    1444/10003	( 14.4%)
         Novel introns:     298/8791	(  3.4%)
           Missed loci:      43/2183	(  2.0%)
            Novel loci:      40/2171	(  1.8%)
#= Summary for dataset: contrast_predictions.pred.seita.gff3
#     Query mRNAs :    2519 in    2185 loci  (1918 multi-exon transcripts)
#            (192 multi-transcript loci, ~1.2 transcripts per locus)
# Reference mRNAs :    2734 in    2190 loci  (2345 multi-exon)
# Super-loci w/ reference transcripts:     2094
#-----------------| Sensitivity | Precision  |
        Base level:    81.9     |    79.6    |
        Exon level:    25.0     |    27.8    |
      Intron level:    18.5     |    21.1    |
Intron chain level:     6.8     |     8.3    |
  Transcript level:    18.3     |    19.9    |
       Locus level:    22.9     |    22.9    |

     Matching intron chains:     160
       Matching transcripts:     501
              Matching loci:     501

          Missed exons:    1506/11278	( 13.4%)
           Novel exons:    1131/10156	( 11.1%)
        Missed introns:    1432/8771	( 16.3%)
         Novel introns:     438/7699	(  5.7%)
           Missed loci:      93/2190	(  4.2%)
            Novel loci:      91/2185	(  4.2%)
#= Summary for dataset: contrast_predictions.pred.bradi.gff3
#     Query mRNAs :    2564 in    2167 loci  (1927 multi-exon transcripts)
#            (219 multi-transcript loci, ~1.2 transcripts per locus)
# Reference mRNAs :    3343 in    2177 loci  (2916 multi-exon)
# Super-loci w/ reference transcripts:     2028
#-----------------| Sensitivity | Precision  |
        Base level:    80.2     |    70.8    |
        Exon level:    23.3     |    27.6    |
      Intron level:    17.8     |    21.7    |
Intron chain level:     5.1     |     7.7    |
  Transcript level:    13.7     |    17.9    |
       Locus level:    21.1     |    21.2    |

     Matching intron chains:     149
       Matching transcripts:     459
              Matching loci:     459

          Missed exons:    1743/11848	( 14.7%)
           Novel exons:    1250/9911	( 12.6%)
        Missed introns:    1708/9001	( 19.0%)
         Novel introns:     496/7394	(  6.7%)
           Missed loci:     142/2177	(  6.5%)
            Novel loci:     136/2167	(  6.3%)
#= Summary for dataset: contrast_predictions.pred.pp3c.gff3
#     Query mRNAs :    4012 in    2016 loci  (3233 multi-exon transcripts)
#            (922 multi-transcript loci, ~2.0 transcripts per locus)
# Reference mRNAs :    4833 in    2021 loci  (4227 multi-exon)
# Super-loci w/ reference transcripts:     1828
#-----------------| Sensitivity | Precision  |
        Base level:    78.9     |    75.3    |
        Exon level:    13.9     |    16.7    |
      Intron level:    10.3     |    12.6    |
Intron chain level:     1.7     |     2.2    |
  Transcript level:     6.4     |     7.7    |
       Locus level:    14.1     |    14.1    |

     Matching intron chains:      70
       Matching transcripts:     310
              Matching loci:     285

          Missed exons:    1912/15351	( 12.5%)
           Novel exons:    1317/12741	( 10.3%)
        Missed introns:    2349/10526	( 22.3%)
         Novel introns:     761/8599	(  8.8%)
           Missed loci:     190/2021	(  9.4%)
            Novel loci:     187/2016	(  9.3%)

BEAM search
#= Summary for dataset: beam_predictions.pred.artha.gff3
#     Query mRNAs :    1517 in    1499 loci  (1098 multi-exon transcripts)
#            (16 multi-transcript loci, ~1.0 transcripts per locus)
# Reference mRNAs :    1935 in    1501 loci  (1565 multi-exon)
# Super-loci w/ reference transcripts:     1477
#-----------------| Sensitivity | Precision  |
        Base level:    89.0     |    90.1    |
        Exon level:    36.1     |    41.0    |
      Intron level:    28.0     |    32.1    |
Intron chain level:    11.1     |    15.8    |
  Transcript level:    25.1     |    32.0    |
       Locus level:    32.3     |    32.4    |

     Matching intron chains:     173
       Matching transcripts:     485
              Matching loci:     485

          Missed exons:     566/8352	(  6.8%)
           Novel exons:     354/7117	(  5.0%)
        Missed introns:     624/6423	(  9.7%)
         Novel introns:      65/5602	(  1.2%)
           Missed loci:      24/1501	(  1.6%)
            Novel loci:      22/1499	(  1.5%)
#= Summary for dataset: beam_predictions.pred.glyma.gff3
#     Query mRNAs :    2746 in    2597 loci  (2120 multi-exon transcripts)
#            (91 multi-transcript loci, ~1.1 transcripts per locus)
# Reference mRNAs :    4169 in    2598 loci  (3562 multi-exon)
# Super-loci w/ reference transcripts:     2570
#-----------------| Sensitivity | Precision  |
        Base level:    83.9     |    85.0    |
        Exon level:    30.0     |    35.8    |
      Intron level:    23.8     |    27.6    |
Intron chain level:     7.1     |    12.0    |
  Transcript level:    18.3     |    27.7    |
       Locus level:    29.3     |    29.3    |

     Matching intron chains:     254
       Matching transcripts:     762
              Matching loci:     762

          Missed exons:    1943/17005	( 11.4%)
           Novel exons:    1412/13302	( 10.6%)
        Missed introns:    1678/12299	( 13.6%)
         Novel introns:     286/10585	(  2.7%)
           Missed loci:      28/2598	(  1.1%)
            Novel loci:      27/2597	(  1.0%)
#= Summary for dataset: beam_predictions.pred.sobic.gff3
#     Query mRNAs :    1775 in    1742 loci  (1322 multi-exon transcripts)
#            (24 multi-transcript loci, ~1.0 transcripts per locus)
# Reference mRNAs :    2369 in    1742 loci  (1981 multi-exon)
# Super-loci w/ reference transcripts:     1698
#-----------------| Sensitivity | Precision  |
        Base level:    84.6     |    83.7    |
        Exon level:    31.0     |    35.7    |
      Intron level:    23.7     |    27.0    |
Intron chain level:     9.3     |    14.0    |
  Transcript level:    21.2     |    28.3    |
       Locus level:    28.8     |    28.8    |

     Matching intron chains:     185
       Matching transcripts:     502
              Matching loci:     502

          Missed exons:    1229/10067	( 12.2%)
           Novel exons:     955/8327	( 11.5%)
        Missed introns:    1185/7466	( 15.9%)
         Novel introns:     223/6551	(  3.4%)
           Missed loci:      43/1742	(  2.5%)
            Novel loci:      43/1742	(  2.5%)
#= Summary for dataset: beam_predictions.pred.potri.gff3
#     Query mRNAs :    1927 in    1854 loci  (1458 multi-exon transcripts)
#            (54 multi-transcript loci, ~1.0 transcripts per locus)
# Reference mRNAs :    2857 in    1855 loci  (2419 multi-exon)
# Super-loci w/ reference transcripts:     1830
#-----------------| Sensitivity | Precision  |
        Base level:    84.1     |    85.3    |
        Exon level:    30.1     |    35.2    |
      Intron level:    23.7     |    27.0    |
Intron chain level:     7.4     |    12.3    |
  Transcript level:    18.1     |    26.8    |
       Locus level:    27.8     |    27.8    |

     Matching intron chains:     180
       Matching transcripts:     516
              Matching loci:     516

          Missed exons:    1199/11748	( 10.2%)
           Novel exons:     906/9421	(  9.6%)
        Missed introns:    1105/8571	( 12.9%)
         Novel introns:     191/7505	(  2.5%)
           Missed loci:      25/1855	(  1.3%)
            Novel loci:      24/1854	(  1.3%)
#= Summary for dataset: beam_predictions.pred.seita.gff3
#     Query mRNAs :    1903 in    1870 loci  (1365 multi-exon transcripts)
#            (23 multi-transcript loci, ~1.0 transcripts per locus)
# Reference mRNAs :    2332 in    1870 loci  (1998 multi-exon)
# Super-loci w/ reference transcripts:     1805
#-----------------| Sensitivity | Precision  |
        Base level:    83.9     |    78.5    |
        Exon level:    29.7     |    33.9    |
      Intron level:    22.2     |    26.0    |
Intron chain level:     8.2     |    11.9    |
  Transcript level:    19.9     |    24.3    |
       Locus level:    24.8     |    24.8    |

     Matching intron chains:     163
       Matching transcripts:     463
              Matching loci:     463

          Missed exons:    1133/9560	( 11.9%)
           Novel exons:     813/8234	(  9.9%)
        Missed introns:    1222/7430	( 16.4%)
         Novel introns:     268/6351	(  4.2%)
           Missed loci:      64/1870	(  3.4%)
            Novel loci:      64/1870	(  3.4%)
#= Summary for dataset: beam_predictions.pred.bradi.gff3
#     Query mRNAs :    1942 in    1878 loci  (1390 multi-exon transcripts)
#            (29 multi-transcript loci, ~1.0 transcripts per locus)
# Reference mRNAs :    2864 in    1884 loci  (2493 multi-exon)
# Super-loci w/ reference transcripts:     1777
#-----------------| Sensitivity | Precision  |
        Base level:    82.3     |    54.7    |
        Exon level:    28.7     |    35.1    |
      Intron level:    22.9     |    28.4    |
Intron chain level:     6.2     |    11.2    |
  Transcript level:    15.3     |    22.6    |
       Locus level:    23.2     |    23.3    |

     Matching intron chains:     155
       Matching transcripts:     438
              Matching loci:     438

          Missed exons:    1296/10175	( 12.7%)
           Novel exons:     823/8128	( 10.1%)
        Missed introns:    1363/7717	( 17.7%)
         Novel introns:     239/6210	(  3.8%)
           Missed loci:     100/1884	(  5.3%)
            Novel loci:      99/1878	(  5.3%)
#= Summary for dataset: beam_predictions.pred.pp3c.gff3
#     Query mRNAs :    2574 in    1709 loci  (1905 multi-exon transcripts)
#            (660 multi-transcript loci, ~1.5 transcripts per locus)
# Reference mRNAs :    4112 in    1714 loci  (3586 multi-exon)
# Super-loci w/ reference transcripts:     1579
#-----------------| Sensitivity | Precision  |
        Base level:    81.9     |    82.2    |
        Exon level:    17.9     |    24.2    |
      Intron level:    14.0     |    18.9    |
Intron chain level:     2.3     |     4.4    |
  Transcript level:     7.9     |    12.6    |
       Locus level:    17.7     |    17.8    |

     Matching intron chains:      84
       Matching transcripts:     325
              Matching loci:     304

          Missed exons:    1468/13105	( 11.2%)
           Novel exons:     866/9322	(  9.3%)
        Missed introns:    2313/8986	( 25.7%)
         Novel introns:     405/6687	(  6.1%)
           Missed loci:     134/1714	(  7.8%)
            Novel loci:     129/1709	(  7.5%)

GREEDY search:
#= Summary for dataset: greedy_predictions.pred.artha.gff3
#     Query mRNAs :    1751 in    1738 loci  (1279 multi-exon transcripts)
#            (8 multi-transcript loci, ~1.0 transcripts per locus)
# Reference mRNAs :    2237 in    1740 loci  (1819 multi-exon)
# Super-loci w/ reference transcripts:     1711
#-----------------| Sensitivity | Precision  |
        Base level:    88.7     |    89.9    |
        Exon level:    35.8     |    40.6    |
      Intron level:    27.2     |    31.1    |
Intron chain level:    10.4     |    14.8    |
  Transcript level:    24.2     |    31.0    |
       Locus level:    31.1     |    31.2    |

     Matching intron chains:     189
       Matching transcripts:     542
              Matching loci:     542

          Missed exons:     687/9791	(  7.0%)
           Novel exons:     445/8374	(  5.3%)
        Missed introns:     727/7572	(  9.6%)
         Novel introns:      88/6631	(  1.3%)
           Missed loci:      29/1740	(  1.7%)
            Novel loci:      27/1738	(  1.6%)
#= Summary for dataset: greedy_predictions.pred.glyma.gff3
#     Query mRNAs :    3069 in    2961 loci  (2369 multi-exon transcripts)
#            (78 multi-transcript loci, ~1.0 transcripts per locus)
# Reference mRNAs :    4826 in    2993 loci  (4134 multi-exon)
# Super-loci w/ reference transcripts:     2937
#-----------------| Sensitivity | Precision  |
        Base level:    82.2     |    85.5    |
        Exon level:    29.5     |    36.0    |
      Intron level:    23.6     |    27.8    |
Intron chain level:     7.1     |    12.5    |
  Transcript level:    18.2     |    28.6    |
       Locus level:    29.4     |    29.7    |

     Matching intron chains:     295
       Matching transcripts:     879
              Matching loci:     879

          Missed exons:    2635/19796	( 13.3%)
           Novel exons:    1618/15205	( 10.6%)
        Missed introns:    2236/14351	( 15.6%)
         Novel introns:     359/12163	(  3.0%)
           Missed loci:      56/2993	(  1.9%)
            Novel loci:      24/2961	(  0.8%)
#= Summary for dataset: greedy_predictions.pred.sobic.gff3
#     Query mRNAs :    2057 in    1996 loci  (1534 multi-exon transcripts)
#            (19 multi-transcript loci, ~1.0 transcripts per locus)
# Reference mRNAs :    2724 in    2011 loci  (2253 multi-exon)
# Super-loci w/ reference transcripts:     1941
#-----------------| Sensitivity | Precision  |
        Base level:    83.5     |    56.6    |
        Exon level:    30.6     |    35.9    |
      Intron level:    23.5     |    27.2    |
Intron chain level:     9.5     |    14.0    |
  Transcript level:    21.8     |    28.9    |
       Locus level:    29.5     |    29.8    |

     Matching intron chains:     215
       Matching transcripts:     594
              Matching loci:     594

          Missed exons:    1498/11585	( 12.9%)
           Novel exons:     996/9434	( 10.6%)
        Missed introns:    1396/8596	( 16.2%)
         Novel introns:     273/7416	(  3.7%)
           Missed loci:      66/2011	(  3.3%)
            Novel loci:      54/1996	(  2.7%)
#= Summary for dataset: greedy_predictions.pred.potri.gff3
#     Query mRNAs :    2256 in    2129 loci  (1719 multi-exon transcripts)
#            (61 multi-transcript loci, ~1.1 transcripts per locus)
# Reference mRNAs :    3280 in    2143 loci  (2774 multi-exon)
# Super-loci w/ reference transcripts:     2100
#-----------------| Sensitivity | Precision  |
        Base level:    83.7     |    42.0    |
        Exon level:    30.3     |    35.7    |
      Intron level:    24.0     |    27.6    |
Intron chain level:     7.6     |    12.3    |
  Transcript level:    18.2     |    26.5    |
       Locus level:    27.9     |    27.9    |

     Matching intron chains:     212
       Matching transcripts:     597
              Matching loci:     597

          Missed exons:    1519/13559	( 11.2%)
           Novel exons:    1029/10792	(  9.5%)
        Missed introns:    1382/9894	( 14.0%)
         Novel introns:     229/8599	(  2.7%)
           Missed loci:      37/2143	(  1.7%)
            Novel loci:      29/2129	(  1.4%)
#= Summary for dataset: greedy_predictions.pred.seita.gff3
#     Query mRNAs :    2154 in    2138 loci  (1578 multi-exon transcripts)
#            (8 multi-transcript loci, ~1.0 transcripts per locus)
# Reference mRNAs :    2675 in    2145 loci  (2297 multi-exon)
# Super-loci w/ reference transcripts:     2067
#-----------------| Sensitivity | Precision  |
        Base level:    83.5     |    83.2    |
        Exon level:    30.7     |    35.1    |
      Intron level:    23.5     |    27.4    |
Intron chain level:     8.8     |    12.8    |
  Transcript level:    20.1     |    25.0    |
       Locus level:    25.1     |    25.2    |

     Matching intron chains:     202
       Matching transcripts:     538
              Matching loci:     538

          Missed exons:    1346/11042	( 12.2%)
           Novel exons:     932/9503	(  9.8%)
        Missed introns:    1391/8591	( 16.2%)
         Novel introns:     308/7352	(  4.2%)
           Missed loci:      76/2145	(  3.5%)
            Novel loci:      70/2138	(  3.3%)
#= Summary for dataset: greedy_predictions.pred.bradi.gff3
#     Query mRNAs :    2191 in    2133 loci  (1592 multi-exon transcripts)
#            (25 multi-transcript loci, ~1.0 transcripts per locus)
# Reference mRNAs :    3301 in    2148 loci  (2881 multi-exon)
# Super-loci w/ reference transcripts:     2025
#-----------------| Sensitivity | Precision  |
        Base level:    80.8     |    80.6    |
        Exon level:    27.5     |    33.6    |
      Intron level:    21.5     |    26.5    |
Intron chain level:     5.8     |    10.6    |
  Transcript level:    14.7     |    22.1    |
       Locus level:    22.6     |    22.7    |

     Matching intron chains:     168
       Matching transcripts:     485
              Matching loci:     485

          Missed exons:    1663/11689	( 14.2%)
           Novel exons:    1037/9356	( 11.1%)
        Missed introns:    1610/8879	( 18.1%)
         Novel introns:     335/7197	(  4.7%)
           Missed loci:     117/2148	(  5.4%)
            Novel loci:     107/2133	(  5.0%)
#= Summary for dataset: greedy_predictions.pred.pp3c.gff3
#     Query mRNAs :    3127 in    1919 loci  (2440 multi-exon transcripts)
#            (723 multi-transcript loci, ~1.6 transcripts per locus)
# Reference mRNAs :    4732 in    1983 loci  (4135 multi-exon)
# Super-loci w/ reference transcripts:     1769
#-----------------| Sensitivity | Precision  |
        Base level:    76.9     |    31.8    |
        Exon level:    17.8     |    24.9    |
      Intron level:    14.2     |    19.4    |
Intron chain level:     2.2     |     3.8    |
  Transcript level:     7.4     |    11.2    |
       Locus level:    16.4     |    17.0    |

     Matching intron chains:      92
       Matching transcripts:     350
              Matching loci:     326

          Missed exons:    2370/15067	( 15.7%)
           Novel exons:     941/10032	(  9.4%)
        Missed introns:    2752/10330	( 26.6%)
         Novel introns:     455/7559	(  6.0%)
           Missed loci:     203/1983	( 10.2%)
            Novel loci:     149/1919	(  7.8%)

# Training Stage 2

Performance is good at the base-level but we'd like to improve it at the exon level. This requires an improvement in the identification of splice junctions. Ideally, we also would like to improve the isoform prediction performance as it is the primary advancement.

Hiabao is interested in using base-wise probability scores to update the splice junctions instead of my nearest canonical junction heuristic. InstaDeep has created a U-net architecture for nucleotide-resolution segmentation which can provide the base-wise probabilities, but have not fine tuned it for plants. My goal is to incorporate the U-net into the model in order to simultaneously compute the base probabilities and the decoder output from the encoder output. The U-net logits can be used in post processing and may be fed back into the decoder to improve the generation task. The U-net also provides a potential method to scan the genome for genic regions which could make the model a stand-alone application for genome annotation.

I will add the U-net to the encoder model and fine-tune it on the 7 training genomes. Then I will continue training the full model with the combined input from the encoder and the U-net. Finally, a post-processing algorithm using the U-net derived probabilities will clean up the coordinate portion of the text output. The isoform part may be improved by customizing the CrossEntropy loss to focus on the second part of the output. Finally, I can implement a scanning algorithm which runs the encoder/U-net on genomic sequences until a complete genic segment is identified, then use the stored embeddings to run the decoder.

After the first epoch of training, the Unet reached a eval loss of ~0.42. The loss was not improving in the next epoch, so I decided to restart the trianing with a smaller learning rate and adjusted loss weights. The loss was still flat. I increased the batch size and the gradient clipping threshold and restarted. None of these tactics caused the loss to decline. I stopped fine-tuning for now and began to characterize the performance (recall, precision, f1, mcc) on the validation set.

## Isoform performance analysis

I need defensible performance metrics to determine the performance of isofrom prediction...

Analysis questions (break out labels with and without AS):
1. For each gene, is the correct number of transcripts identified?
2. For each transcript, how much of the correct features are included? (Break out CDS and UTR)
3. How many transcripts per-gene are perfectly correct (CDS/UTR level)
4. How many total transcripts are perfectly correct? (CDS level, UTR level)

