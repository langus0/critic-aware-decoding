import csv
import random
import sys

from sentence_splitter import split_text_into_sentences
from tqdm import tqdm

split = sys.argv[1]
OUTPUT = f"{split}-oldrandom.csv"

f = open(f'webnlg-{split}.tex', encoding="UTF-8")
csv_reader = csv.reader(f)
lines = [line for line in csv_reader]
f.close()


def get_random_sentence(triplets):
    while True:
        line = random.choice(lines)
        if line[1] == triplets:
            continue
        sentences = split_text_into_sentences(text=line[0], language='en')
        return random.choice(sentences)


def get_other_related_sentence(triplets, ref):
    other_refs = [line for line in lines if line[1] == triplets and line[0] != ref]
    if len(other_refs) == 0:
        return None
    new_ref = random.choice(other_refs)
    sentences = split_text_into_sentences(text=new_ref[0], language='en')
    return random.choice(sentences)


def has_many_trip(line):
    return line[1].find(";") != -1


def how_many_sent(line):
    sentences = split_text_into_sentences(text=line[0], language='en')
    return len(sentences)


def write_neg_line(writer, line, txt: str):
    txt_corr = line[0].strip().split(" ")
    txt = txt.strip().split(" ")
    for i in range(len(txt)):
        if txt[:(i + 1)] != txt_corr[:(i + 1)]:
            row = [" ".join(txt[:(i + 1)])]
            row.append(line[1])
            row.append(0)
            writer.writerow(row)


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


with open(OUTPUT, "w", encoding="UTF-8") as fw:
    writer = csv.writer(fw)
    writer.writerow(["ref", "data", "class"])
    for line in tqdm(lines):
        # correct examples
        write_line(writer, line, 1)

        # continue generation
        sent = get_other_related_sentence(line[1], line[0])
        if sent is None:
            sent = get_random_sentence(line[1])
        write_neg_line(writer, line, line[0] + " " + sent)

        # replace sentence with another
        sent = get_random_sentence(line[1])
        sentences = split_text_into_sentences(text=line[0], language='en')
        random_idx = random.randint(0, len(sentences) - 1)
        sentences[random_idx] = sent
        write_neg_line(writer, line, " ".join(sentences))

        # replace a random word
        words = line[0].split(" ")
        random_word = random.choice(sent.split(" "))
        random_idx = random.randint(1, len(words) - 1)
        words[random_idx] = random_word
        write_neg_line(writer, line, " ".join(words))
print(len(lines))
