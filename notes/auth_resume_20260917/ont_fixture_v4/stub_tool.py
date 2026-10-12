"""SYNTHETIC STUB ONLY. Test plumbing, never real-tool completion evidence."""
import gzip
from pathlib import Path
import shutil
import sys

name = Path(sys.argv[0]).name
args = sys.argv[1:]

def records(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as handle:
        text = handle.read()
    return [(x.splitlines()[0], ''.join(x.splitlines()[1:])) for x in text.split('>')[1:]]

if name == 'minimap2':
    rows = records(args[-1])
    sam = '-ax' in args
    if sam:
        print('@HD\tVN:1.6\tSO:unsorted\n@SQ\tSN:fixture_chr\tLN:60000\n@SQ\tSN:outside_chr\tLN:4000')
    for read, seq in rows:
        chrom, start, cigar, end = 'fixture_chr', 20000, '500M200N500M', 21200
        if read.startswith('below_threshold'):
            start, end = 30000, 31200
        elif read == 'unspliced':
            start, end, cigar = 40000, 41000, '1000M'
        elif read == 'outside':
            chrom, start, end, cigar = 'outside_chr', 1000, 2000, '1000M'
        if sam:
            print(f'{read}\t0\t{chrom}\t{start+1}\t60\t{cigar}\t*\t0\t0\t{seq}\t*\tts:A:+')
        else:
            print(f'{read}\t1000\t0\t1000\t+\t{chrom}\t{60000 if chrom == "fixture_chr" else 4000}\t{start}\t{end}\t1000\t1000\t60')
elif name == 'seqkit':
    names = set(Path(args[2]).read_text().splitlines())
    for read, seq in records(args[3]):
        if read in names:
            print('>'+read+'\n'+seq)
elif name == 'samtools':
    if args[0] == 'sort':
        shutil.copyfile(args[-1], args[2])
    elif args[0] == 'index':
        Path(args[1]+'.bai').write_text('STUB NOT BAM INDEX\n')
    elif args[0] == 'view':
        print(Path(args[-1]).read_text(), end='')
    else:
        raise ValueError(args)
else:
    raise ValueError(name)
