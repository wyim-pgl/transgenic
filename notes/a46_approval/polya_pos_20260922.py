import gzip, re, sys, collections
N=int(sys.argv[1]); PA=re.compile("A{20,}"); PT=re.compile("T{20,}")
for path in sys.argv[2:]:
    n=0; name=None; seq=[]; bins=collections.Counter(); kinds=collections.Counter(); ex=[]
    def done(s):
        global n
        n+=1; L=len(s)
        ms=[("A",m) for m in PA.finditer(s)]+[("T",m) for m in PT.finditer(s)]
        kinds[len(ms)]+=1
        for k,m in ms:
            d3=L-m.end(); d5=m.start()
            bins[(k, "5p<300" if d5<300 else ("3p<300" if d3<300 else "mid"))]+=1
        if ms and len(ex)<3: ex.append((L,[(k,m.start(),m.end()) for k,m in ms], s[max(0,ms[0][1].start()-40):ms[0][1].end()+60]))
    with gzip.open(path,"rt") as fh:
        for line in fh:
            if line[0]==">":
                if name is not None:
                    done("".join(seq).upper())
                    if n>=N: break
                name,seq=line,[]
            else: seq.append(line.strip())
    print(path, n, "runs/read:", dict(sorted(kinds.items())), "pos:", dict(bins))
    for e in ex: print("  ", e)
