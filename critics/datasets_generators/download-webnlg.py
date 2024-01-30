from decoding.data import WebNLG
import sys
import csv

def linearize_triple(triple):
    X_SEP = " | "
    out = []
    t = triple
    return t.subj + X_SEP + t.pred + X_SEP + t.obj

data = WebNLG()
splits = [sys.argv[1]]
data.load(splits,None)

cleanedData = []
for split in splits:
    for dataEntry in data.data[split]:
        for ref in dataEntry.refs:
                triplets = [linearize_triple(i) for i in dataEntry.data]
                cleanedData.append((ref,triplets))

with open(f"webnlg-{split}.tex", 'w', encoding="UTF-8") as f:
    writer = csv.writer(f)
    for txt,triplets in cleanedData:
        writer.writerow([txt, " ▸ ".join(triplets)])

