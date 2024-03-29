# -*- coding: utf-8 -*-
"""401R Final Project.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Ftr030cpKz87Lhc2HKo3KtB43o66NIi9
"""

from itertools import accumulate
import sys
from unittest.util import _MAX_LENGTH
import torch
from transformers import EncoderDecoderModel, BertTokenizer
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
#from torchvision import transforms, utils, datasets
from tqdm import tqdm
from torch.nn.parameter import Parameter
import pdb
import gc
import copy
from transformers import BartForConditionalGeneration, BartForSequenceClassification, BartConfig
import random

from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("Sorenson.json")

langs = ['[EN]', '[FA]']
l2ind = { s:i for i, s in enumerate(langs) }


configuration = BartConfig( vocab_size = tokenizer.get_vocab_size(),
                            max_position_embeddings = 512,
                            encoder_layers = 6,
                            encoder_ffn_dim = 2048,
                            encoder_attention_heads = 8,
                            decoder_layers = 6,
                            decoder_ffn_dim = 2048,
                            decoder_attention_heads = 8,
                            encoder_layerdrop = 0.0,
                            decoder_layerdrop = 0.0,
                            activation_function = 'swish',
                            d_model = 512,
                            dropout = 0.1,
                            attention_dropout = 0.0,
                            activation_dropout = 0.0,
                            init_std = 0.02,
                            classifier_dropout = 0.0,
                            scale_embedding = True,
                            pad_token_id = tokenizer.token_to_id("[PAD]"),
                            bos_token_id = 0,
                            eos_token_id = tokenizer.token_to_id("[CLS]"),
                            is_encoder_decoder = True,
                            decoder_start_token_id = tokenizer.token_to_id("[EN]"),
                            forced_eos_token_id = tokenizer.token_to_id("[CLS]"),
                            num_labels = len(langs) ) # for descriminator

model = BartForConditionalGeneration(configuration)

if torch.cuda.is_available():
  model.cuda()

def translate(sent, targ):
  label = torch.tensor(tokenizer.encode(sent.strip()).ids)
  if torch.cuda.is_available():
    label = label.cuda()
  generated = model.generate(label.unsqueeze(0), max_length=200, decoder_start_token_id=tokenizer.token_to_id(targ))
  trans = tokenizer.decode(generated[0].cpu().numpy(),skip_special_tokens=True)
  trans = trans.replace(' ##', '')
  return trans

model.load_state_dict(torch.load('demo-weights',map_location=torch.device('cpu')))
model.eval()

sents = ['وقتی دوباره پاکیزه می شویم، چه احساسی پیدا می کنیم؟', 'به خاطر شفقت منجی است که گناهان من می توانند بخشیده شوند.']

# Run test set
with torch.no_grad():
    for sent in sents:
        print(translate(sent, '[EN]'))

    print(translate('God loves us', '[FA]'))
    print(translate('We are God\'s children', '[FA]'))