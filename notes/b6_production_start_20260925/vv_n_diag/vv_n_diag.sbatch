#!/bin/bash
#SBATCH --job-name=b6_vv_ndiag
#SBATCH --account=cpu-s1-pgl-0
#SBATCH --partition=cpu-s1-pgl-0
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=/data/gpfs/assoc/pgl/data/Transgenic/evidence/b6_prepare_20260925/logs/%x_%j.out
# DIAGNOSTIC ONLY (no training input): how large must the audit N be for Vvinifera to reach species exhaustiveness 0.999?
# 1) per source, count reads whose N=100 audit alignment has >= 100 secondary records (capped);
# 2) sample up to 20k capped reads per source, realign with the frozen audit command but -N 5000;
# 3) report the distribution of secondary counts -> projected per-read exhaustiveness at N in {100,200,500,1000,2000,5000}.
set -euo pipefail
R=/data/gpfs/assoc/pgl/data/Transgenic; A=$R/evidence/audit_align_n100/Vvinifera; BIN=/data/gpfs/assoc/pgl/bin/conda/conda_envs/RNASEQ_bch709/bin
OUT=$R/evidence/b6_prepare_20260925/vv_n_diag_v2; mkdir -p $OUT; cd $OUT  # v1 (6173012) counted without sorting (audit BAMs are coordinate-sorted)
REF=$R/training_input/genomes/Vvinifera_T2T_ref.fa
for dir in $(find $A -name '*.audit.bam' -printf '%h\n' | sort); do
  run=$(basename $dir); src=$(grep -m1 -oE '^source=[a-z]+' $dir/DONE | cut -d= -f2)
  bam=$(ls $dir/*.audit.bam); cmd=$(grep -m1 '^command' $dir/PROVENANCE.txt | sed 's/^command *//')
  $BIN/samtools view -F 2308 $bam | cut -f1 > /dev/null   # primary exists check (fast fail)
  $BIN/samtools view -f 256 $bam | cut -f1 | LC_ALL=C sort -S 24G --parallel=16 -T $OUT | uniq -c | awk -v s=$src -v r=$run '{n[$1>=100]++; if($1>=100) print $2 > ("capped_" r ".txt")} END{print s, r, "reads_with_secondaries", n[0]+n[1], "capped", n[1]+0}' >> capped_summary.txt
  total=$($BIN/samtools view -c -F 2308 $bam); echo "$src $run primary_reads $total" >> capped_summary.txt
  echo "$src|$run|$cmd" >> commands.txt
done
cat capped_summary.txt
# realign a sample of capped reads per source with -N 5000
for src in est pacbio ont; do
  grep "^$src|" commands.txt | while IFS='|' read s run cmd; do
    [ -s capped_$run.txt ] || continue
    reads=$(echo "$cmd" | awk '{print $NF}')
    shuf -n $((20000 / $(grep -c "^$src|" commands.txt) + 1)) --random-source=<(yes) capped_$run.txt | sort -u > sample_$run.txt
    $BIN/seqkit grep -f sample_$run.txt "$reads" > sample_$run.fa 2>/dev/null || /data/gpfs/assoc/pgl/bin/conda/conda_envs/sylvan/bin/python - "$reads" sample_$run.txt sample_$run.fa <<'PY'
import gzip,sys
want=set(open(sys.argv[2]).read().split()); out=open(sys.argv[3],'w'); keep=False
op=gzip.open if sys.argv[1].endswith('.gz') else open
with op(sys.argv[1],'rt') as f:
    for l in f:
        if l[0] in '>@': keep=l[1:].split()[0] in want
        if keep and l[0]!='+': out.write(l if l[0]!='@' else '>'+l[1:])
PY
    c2=$(echo "$cmd" | sed -e 's/-N 100/-N 5000/' -e "s#[^ ]*\$#$OUT/sample_$run.fa#" -e "s#^minimap2#$BIN/minimap2#")
    eval "$c2" 2>/dev/null | $BIN/samtools view -f 256 - | cut -f1 | LC_ALL=C sort | uniq -c | awk -v s=$s '{print s, $1}' >> realign_counts.txt
  done
done
/data/gpfs/assoc/pgl/bin/conda/conda_envs/sylvan/bin/python - <<'PY'
import collections
cnt=collections.defaultdict(list)
for l in open('realign_counts.txt'):
    s,n=l.split(); cnt[s].append(int(n))
for s,v in cnt.items():
    v.sort(); print(s,'sampled capped reads',len(v),'median',v[len(v)//2],'p90',v[int(.9*len(v))],'max',v[-1])
    for N in (100,200,500,1000,2000,5000):
        print(s,f'N={N} fraction_of_capped_still_capped', round(sum(1 for x in v if x>=N)/len(v),4))
PY
