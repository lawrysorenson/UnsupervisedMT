from collections import defaultdict
import os
import re
import random

memory = defaultdict(lambda: defaultdict(list))

path = 'data/dicts/'
for file in os.listdir(path):
    l1, l2 = file[:-4].split('-')
    file = path + file
    with open(file, 'r') as f:
        d1 = memory[l1 + '-' + l2]
        d2 = memory[l2 + '-' + l1]
        for line in f.readlines():
            line = line.strip()
            w1, w2 = line.split()
            d1[w1].append(w2)
            d2[w2].append(w1)

def dumb_translate(l1, l2, sent):
    # Seperate tokens by word boundary
    sent = re.sub('\\b', ' ', sent).strip()
    sent = re.sub(' +', ' ', sent).split()

    mem = memory[l1 + '-' + l2]

    ans = []
    for t in sent:
        if t in mem:
            possibs = mem[t]
            i = int(random.random() * len(possibs))
            ans.append(possibs[i])
        else:
            ans.append(t)
    
    #print(ans)

    return ans

#dumb_translate('fa', 'en', 'سلام, دوست خوبم!')
            