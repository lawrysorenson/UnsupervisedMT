import random

test_size = 2500 + 30000

corename = 'comb'
exts = ['.en-US', '.fa-IR']

files = ['data/cleaning/' + corename + ext for ext in exts]

fileshand = [open(f, 'r') for f in files]

lines = list(zip(*[f.readlines() for f in fileshand]))

for f in fileshand:
    f.close()

random.shuffle(lines)

test = lines[-test_size:]
del lines[-test_size:]

outfiles = ['data/split/' + corename + '-test' + ext for ext in exts]
outfileshand = [open(f, 'w') for f in outfiles]

for t in test:
    for l, f in zip(t, outfileshand):
        f.write(l)

for f in outfileshand:
    f.close()
    
outfiles = ['data/split/' + corename + '-train' + ext for ext in exts]
outfileshand = [open(f, 'w') for f in outfiles]

amount = len(lines) / len(exts)

for i, f in enumerate(outfileshand):
    for l in lines[int(i*amount): int((i+1)*amount)]:
        f.write(l[i])

for f in outfileshand:
    f.close()