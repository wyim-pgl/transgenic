#!/usr/bin/env python3
"""Italicize species names left in plain runs of a pandoc-built .docx (in place).
Splits each affected run into plain / italic / plain runs; runs already italic are untouched."""
import re, sys, zipfile, shutil, os

GENERA = {"Arabidopsis": ["thaliana", "lyrata", "halleri"], "Vitis": ["vinifera"], "Glycine": ["max"],
          "Populus": ["trichocarpa"], "Sorghum": ["bicolor"], "Brachypodium": ["distachyon"],
          "Setaria": ["italica", "viridis"], "Oryza": ["sativa"], "Physcomitrium": ["patens"],
          "Physcomitrella": ["patens"], "Zea": ["mays"], "Brassica": ["rapa"], "Lactuca": ["sativa"],
          "Solanum": ["lycopersicum"], "Drosophila": ["melanogaster"]}
names = []
for g, sps in GENERA.items():
    for s in sps:
        names += [rf"{g} {s}", rf"{g[0]}\. {s}"]
names.append("Drosophila")
SPECIES = re.compile(r"(?<![A-Za-z])(" + "|".join(sorted(names, key=len, reverse=True)) + r")(?![A-Za-z])")
RUN = re.compile(r"<w:r(?: [^>]*)?>(.*?)</w:r>", re.S)
RPR = re.compile(r"<w:rPr>(.*?)</w:rPr>", re.S)
T = re.compile(r"<w:t(?: [^>]*)?>(.*?)</w:t>", re.S)

def is_italic(rpr):
    return rpr is not None and re.search(r"<w:i(?:/>| )", rpr) is not None

def rewrite(m):
    inner = m.group(1)
    rpr_m = RPR.search(inner); rpr = rpr_m.group(1) if rpr_m else None
    ts = T.findall(inner)
    rest = RPR.sub("", T.sub("", inner)).strip()
    if len(ts) != 1 or rest or is_italic(rpr) or not SPECIES.search(ts[0]):
        return m.group(0)
    plain_rpr = f"<w:rPr>{rpr}</w:rPr>" if rpr is not None else ""
    ital_rpr = f"<w:rPr><w:i/><w:iCs/>{rpr or ''}</w:rPr>"
    out = []
    for i, seg in enumerate(SPECIES.split(ts[0])):
        if not seg:
            continue
        pr = ital_rpr if i % 2 == 1 else plain_rpr
        out.append(f'<w:r>{pr}<w:t xml:space="preserve">{seg}</w:t></w:r>')
    rewrite.n += 1
    return "".join(out)
rewrite.n = 0

path = sys.argv[1]
tmp = path + ".tmp"
with zipfile.ZipFile(path) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
    for item in zin.infolist():
        data = zin.read(item.filename)
        if item.filename == "word/document.xml":
            xml = data.decode("utf-8")
            xml = RUN.sub(rewrite, xml)
            left = sum(1 for r in RUN.finditer(xml)
                       if not is_italic((RPR.search(r.group(1)) or [None, None])[1] if RPR.search(r.group(1)) else None)
                       and SPECIES.search(re.sub(r"<[^>]+>", "", r.group(1))))
            data = xml.encode("utf-8")
        zout.writestr(item, data)
shutil.move(tmp, path)
print(f"{os.path.basename(path)}: {rewrite.n} runs split; non-italic species runs remaining: {left}")
