#!/usr/bin/env bash
# Rewrite evidence/PROVENANCE_est.tsv from per-species COUNT/TOTAL/REPAIRED/md5 files.
cd /data/gpfs/assoc/pgl/data/Transgenic/evidence
out=PROVENANCE_est.tsv
printf "date\tsource\tspecies\ttaxid\tquery\tcount_at_download\trecords_on_disk\tmd5\tstatus\tnote\n" > $out.tmp
for pair in "Zmays 4577" "Athaliana 3702" "Gmax 3847" "Ptrichocarpa 3694" "Bdistachyon 15368" "Sitalica 4555" "Osativa 4530" "Sbicolor 4558" "Vvinifera 29760" "Ppatens 3218" "Slycopersicum 4081"; do
  set -- $pair; d=est/$1
  c=$(cat $d/COUNT 2>/dev/null); t=$(cat $d/TOTAL 2>/dev/null); m=$(cut -d' ' -f1 $d/est.fa.gz.md5 2>/dev/null)
  if [ -f $d/REPAIRED ]; then st="repaired"; note="est_fetch.sh + est_repair.sh (NCBI efetch retmax=10000 drops index+9999 per batch; truncated batches re-fetched); original kept as est.fa.gz.orig"
  elif [ -f $d/DONE ]; then st="downloaded_unrepaired"; note="est_fetch.sh only; TOTAL<COUNT expected until est_repair.sh runs"
  else st="pending"; note="est_fetch.sh"; fi
  [ -n "$t" ] && [ "$t" = "$c" ] && st="$st,complete" || st="$st,incomplete"
  printf "2026-09-01\tNCBI nuccore (former dbEST; nucest merged into nuccore)\t%s\t%s\ttxid%s[Organism] AND gbdiv_est[PROP]\t%s\t%s\t%s\t%s\t%s\n" "$1" "$2" "$2" "$c" "$t" "$m" "$st" "$note" >> $out.tmp
done
mv $out.tmp $out; cat $out | cut -f3,6,7,9
