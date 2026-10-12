import gzip, re, sys
N=int(sys.argv[1]); comp=str.maketrans("ACGT","TGCA"); rc=lambda s:s.translate(comp)[::-1]
K=["AATGATACGGCGACCACCGA","CAAGCAGAAGACGGCATACG"]  # Illumina P5 / P7 cores
PR=re.compile("|".join(K+[rc(k) for k in K])); PA=re.compile("A{20,}|T{20,}")
print("file\tn\tillumina_any\tillumina_end150\tillumina_internal\tinternal_polyA20")
for path in sys.argv[2:]:
    n=a=e=i=p=0; name=None; seq=[]
    def done(s):
        global n,a,e,i,p
        n+=1; hs=[m.start() for m in PR.finditer(s)]; L=len(s)
        a+=bool(hs); e+=any(x<150 or x>L-170 for x in hs); i+=any(150<=x<=L-170 for x in hs)
        p+=bool(PA.search(s[150:-150])) if L>300 else 0
    with gzip.open(path,"rt") as fh:
        for line in fh:
            if line[0]==">":
                if name is not None:
                    done("".join(seq).upper())
                    if n>=N: break
                name,seq=line,[]
            else: seq.append(line.strip())
    print(f"{path}\t{n}\t{a/n:.4f}\t{e/n:.4f}\t{i/n:.4f}\t{p/n:.4f}",flush=True)
