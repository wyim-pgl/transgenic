# revision/scripts/fig3_infer.py

- species_of · function · L47-L51 — def species_of(gm)
- flip_rc · function · L114-L131 — def flip_rc(lines, st, fin): # gffString2GFF3 has already shifted features to absolute genome coordinates # (region_start + 1 + local offset), so the mirror must be taken over the # absolute window [st+1, fin]: mirroring with `fin - x + 1` alone treats the # coordinates as window-local and drops every -rc prediction near the # chromosome start, exactly region_start short (see # revision/results/fig3_gap_diagnosis/gap_diagnosis_summary.json).
- write_pred · function · L133-L147 — def write_pred(rn, out_row)
