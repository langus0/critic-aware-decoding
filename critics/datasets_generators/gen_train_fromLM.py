import csv
import sys
from collections import defaultdict
from tqdm import tqdm

from data import WebNLG

split = sys.argv[1]
INPUT = sys.argv[2]
OUTPUT = f"{split}-criticdata.csv"


def write_line(writer, line, clazz, txt=None):
    if txt is None:
        txt = line[0].strip().split(" ")
    else:
        txt = txt.strip().split(" ")
    for i in range(len(txt)):
        row = [" ".join(txt[:(i + 1)])]
        row.append(line[1])
        row.append(clazz)
        writer.writerow(row)


def linearize_triple(triple):
    X_SEP = " | "
    out = []
    t = triple
    return t.subj + X_SEP + t.pred + X_SEP + t.obj


data = WebNLG()
splits = [split]
data.load(splits, None)

cleanedData = []
data2ref = defaultdict(list)
data2data = {}
for split in splits:
    for dataEntry in tqdm(data.data[split]):
        for ref in dataEntry.refs:
            triplets1 = [linearize_triple(i) for i in dataEntry.data]
            triplets = " ▸ ".join(triplets1)
            triplets2 = "  ".join(triplets1)
            data2data[triplets] = triplets
            data2ref[triplets].append(ref)
            cleanedData.append((ref, triplets))

with open(OUTPUT, "w", encoding="UTF-8") as fw:
    writer = csv.writer(fw)
    writer.writerow(["ref", "data", "class"])
    for line in tqdm(cleanedData):
        # correct examples
        write_line(writer, line, 1)

    f = open(INPUT, encoding="UTF-8")
    csv_reader = csv.reader(f)
    neg_lines = []
    err = 0
    for line in tqdm(csv_reader):
        refs = data2ref[line[1]]
        if len(refs) == 0:
            err += 1
            continue
        if not any([r.startswith(line[0]) for r in refs]):
            neg_lines.append((line[0], data2data[line[1]], 0))
    f.close()
    for row in set(neg_lines):
        writer.writerow(row)
    print(err)
