import re
from collections import defaultdict
import math

orig = open('data/cleaning/Sorenson.en-US', 'r')
orig_t = open('data/cleaning/Sorenson.fa-IR', 'r')
comp = open('data/cleaning/OPUS.en-US', 'r')
comp_t = open('data/cleaning/OPUS.fa-IR', 'r')

orig_lines = orig.readlines()
orig_t_lines = orig_t.readlines()
comp_lines = comp.readlines()
comp_t_lines = comp_t.readlines()

orig.close()
orig_t.close()
comp.close()
comp_t.close()

def parse(lines):
    ans = defaultdict(lambda: 0)
    for sent in lines:
        sent = re.sub('\\b', ' ', sent.lower()).strip()
        sent = re.sub('[^\\w ]+', '', sent)
        sent = re.sub(' +', ' ', sent).split()
        sent = set(sent)
        for w in sent:
            ans[w] += 1
    return ans

orig_dict = parse(orig_lines)
comp_dict = parse(comp_lines)

len1 = len(orig_lines)
len2 = len(comp_lines)
for w, c in orig_dict.items():
    orig_dict[w] = math.log(c**0.8 / len1 / ((comp_dict[w] + 1) / len2))

outputs = []

for raw, rawt in zip(comp_lines, comp_t_lines):
    sent = re.sub('\\b', ' ', raw.lower()).strip()
    sent = re.sub('[^\\w ]+', '', sent)
    sent = re.sub(' +', ' ', sent).split()
    sent = set(sent)
    score = 0
    for w in sent:
        score += orig_dict[w]
    score/=len(sent)**2
    if score > -1:
        outputs.append((raw, rawt, score))

output = sorted(outputs, key=lambda x:-x[2])[:50000]

out1 = open('data/cleaning/comb.en-US', 'w')
out2 = open('data/cleaning/comb.fa-IR', 'w')

for l in orig_lines:
    out1.write(l)

for l in orig_t_lines:
    out2.write(l)

for l1, l2, _ in output:
    out1.write(l1)
    out2.write(l2)

out1.close()
out2.close()