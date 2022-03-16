from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece
from tokenizers.trainers import BpeTrainer, WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import BartTokenizer

tokenizer = Tokenizer(WordPiece())

trainer = WordPieceTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "[EN]", "[FA]"])

tokenizer.pre_tokenizer = Whitespace()

path = "data/cleaning/"
files = [path + file for file in ["Sorenson.en-US", "Sorenson.fa-IR"]]
tokenizer.train(files, trainer)

tokenizer.post_processor = TemplateProcessing(
    single="$A [CLS]",
    special_tokens=[
        #("[CLS]", tokenizer.token_to_id("[CLS]")),
        ("[CLS]", tokenizer.token_to_id("[CLS]")),
    ],
)

tokenizer.save("data/tokenizer.json")