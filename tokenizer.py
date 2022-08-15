from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece
from tokenizers.trainers import BpeTrainer, WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import BartTokenizer
import sys

corename = sys.argv[1]

tokenizer = Tokenizer(WordPiece())

trainer = WordPieceTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "[EN]", "[FA]"])

tokenizer.pre_tokenizer = Whitespace()

path = "data/split/"
files = [path + corename + '-train' + ext for ext in [".en-US", ".fa-IR"]]
tokenizer.train(files, trainer)

for f in files:
    nf = f[:-6] + '-token' + f[-6:]
    f = open(f, 'r')
    nf = open(nf, 'w')

    for l in f.readlines():
        l = l.lower().strip()
        nf.write(' '.join(tokenizer.encode(l).tokens) + '\n')

    f.close()
    nf.close()

tokenizer.post_processor = TemplateProcessing(
    single="$A [CLS]",
    special_tokens=[
        #("[CLS]", tokenizer.token_to_id("[CLS]")),
        ("[CLS]", tokenizer.token_to_id("[CLS]")),
    ],
)


tokenizer.save("data/tokenizers/" + corename + ".json")
