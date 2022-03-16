# -*- coding: utf-8 -*-
"""401R Final Project.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Ftr030cpKz87Lhc2HKo3KtB43o66NIi9
"""

grad_accum = 4
batch_size = 8

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
from tokenizers.models import BPE, WordPiece
from tokenizers.trainers import BpeTrainer, WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import BartTokenizer

import dumb

from util import BartClassificationHead, BartEncoderLayer


#from google.colab import drive
#drive.mount('/content/gdrive')

path = "data/cleaning/"

files = [path + file for file in ["Sorenson.en-US", "Sorenson.fa-IR"]]

class TextDataset(Dataset):
  def __init__(self, files):
    self.data = []
    for file in files:
      lang = '['+file[-5:-3].upper()+']'
      with open(file, 'r') as f:
          self.data.extend([(l.strip(), lang) for l in f.readlines() if len(l) < 1000])

  def __getitem__(self, i):
    return self.data[i]
  
  def __len__(self):
    return len(self.data)

train_dataset = TextDataset(files)

train_dataset_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True)

tokenizer = Tokenizer.from_file("data/tokenizer.json")

langs = ['[EN]', '[FA]']
l2ind = { s:i for i, s in enumerate(langs) }

def noise(s):
  copy = list(s)
  l = len(copy)
  sel = int(random.random() * l)
  delete = random.sample(range(l), sel // 20)

  for d in sorted(delete, key=lambda x: -x):
    del copy[d]
  copy = ''.join(copy)
  
  answer = tokenizer.encode(copy).ids

  l = len(answer)
  sel = int(random.random() * l)
  delete = random.sample(range(l), sel // 20)

  for d in delete:
    answer[d] = tokenizer.token_to_id("[MASK]")

  return answer

def prep_descrim_batch(sents):
  ss = []
  ls = []
  for s, k in sents:
    source = noise(s)
    ss.append(source)
    ls.append(l2ind[k])
  
  pad_len = max([len(s) for s in ss])

  for s in ss:
      s.extend([tokenizer.token_to_id("[PAD]")] * (pad_len - len(s)))
  
  return ss, ls

def prep_auto_batch(sents):
  ss = []
  ts = []
  ls = []
  for s, k in zip(*sents):
    source = noise(s)
    label = tokenizer.encode(s).ids
    targ = [tokenizer.token_to_id(k)] + label[:-1]
    ss.append(source)
    ts.append(targ)
    ls.append(label)
  
  pad_len = max([len(s) for s in ss])

  for s in ss:
      s.extend([tokenizer.token_to_id("[PAD]")] * (pad_len - len(s)))

  pad_len = max([len(t) for t in ts])

  for t in ts:
      t.extend([tokenizer.token_to_id("[PAD]")] * (pad_len - len(t)))

  for l in ls:
      l.extend([tokenizer.token_to_id("[PAD]")] * (pad_len - len(l)))
  
  return ss, ts, ls

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

class Descriminator(nn.Module):
  def __init__(self, config):
    super(Descriminator, self).__init__()
    
    self.layers = nn.ModuleList([BartEncoderLayer(config) for _ in range(2)]) # TODO: EXPERIMENT WITH THIS NUMBER
    self.config = config
    self.layerdrop = 0 #config.decoder_layerdrop

    self.classification_head = BartClassificationHead(
      config.d_model,
      config.d_model,
      config.num_labels,
      config.classifier_dropout,
    )

  def forward(self, hidden_states):

    for encoder_layer in self.layers:

      # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
      dropout_probability = random.uniform(0, 1)
      if dropout_probability < self.layerdrop:  # skip the layer
          layer_outputs = (None, None)
      else:
          layer_outputs = encoder_layer(
              hidden_states,
              None, # attention mask
              layer_head_mask=None,
              output_attentions=None,
          )

          hidden_states = layer_outputs[0]

    #_, sentence_representation = self.gru(hidden_states)

    #sentence_representation = sentence_representation.view(hidden_states.shape[0], -1)

    #sentence_representation = hidden_states[:, -1, :] # Extract EOS representations for classification
    logits = self.classification_head(hidden_states).mean(dim=1)
    
    return logits

descrim = Descriminator(copy.deepcopy(configuration))

# sents = (['Hello, this is a test'], ['[EN]'])

# enc, dec, lab = prep_auto_batch(sents)

# enc = torch.tensor(enc)
# dec = torch.tensor(dec)
# lab = torch.tensor(lab)

# outputs = model.model.encoder(input_ids=enc)[0] #, decoder_input_ids=dec)[0]

# descrim(outputs)

# exit(0)

if torch.cuda.is_available():
  model.cuda()
  descrim.cuda()

objective = nn.CrossEntropyLoss(ignore_index = tokenizer.token_to_id("[PAD]"))
descrim_objective = nn.BCEWithLogitsLoss()

full_optimizer = optim.Adam(model.parameters(), lr=3e-4)
encoder_optimizer = optim.Adam(model.model.encoder.parameters(), lr=2e-4)
descrim_optimizer = optim.Adam(descrim.parameters(), lr=2e-4)


def get_descrim_batch():
    sel = random.sample(range(len(train_dataset)), batch_size)
    sel = [train_dataset[i] for i in sel]
    
    input_enc, labels = prep_descrim_batch(sel)
    input_enc = torch.tensor(input_enc)
    labels = torch.tensor(labels)
    labels = F.one_hot(labels, num_classes=len(langs)).float()

    return input_enc, labels

def train_descrim():
  descrim_steps = 10

  descrim_optimizer.zero_grad()
  batch = 0
  loss_desc = 0
  for _ in range(grad_accum * descrim_steps):
    batch += 1
    input_enc, labels = get_descrim_batch()

    if torch.cuda.is_available():
      input_enc = input_enc.cuda()
      labels = labels.cuda()

    with torch.no_grad():
      outputs = model.model.encoder(input_ids=input_enc)[0]

    logits = descrim(outputs)

    loss = descrim_objective(logits, labels)
    loss.backward()

    if batch % grad_accum == 0:
      loss_desc += loss.item()
      descrim_optimizer.step()
      descrim_optimizer.zero_grad()

  encoder_optimizer.zero_grad()
  loss_enc = 0
  batch = 0
  for _ in range(grad_accum * descrim_steps):
    batch += 1
    input_enc, labels = get_descrim_batch()

    if torch.cuda.is_available():
      input_enc = input_enc.cuda()
      labels = labels.cuda()

    labels = 1 - labels

    outputs = model.model.encoder(input_ids=input_enc)[0]

    logits = descrim(outputs)

    loss = descrim_objective(logits, labels)
    loss.backward()

    if batch % grad_accum == 0:
      loss_enc += loss.item()
      encoder_optimizer.step()
      encoder_optimizer.zero_grad()

  return loss_desc, loss_enc

def prep_cross_batch(sents, first):
  ss = []
  ts = []
  ls = []
  for s, sk in zip(*sents):
    # Randomly select a target language
    si = l2ind[sk]
    ti = int(random.random() * len(langs) - 1)
    if ti >= si:
      ti += 1
    tk = langs[ti]

    #TODO: not first
    trans = dumb.dumb_translate(sk[1:3].lower(), tk[1:3].lower(), s)
    source = tokenizer.encode(trans).ids

    label = tokenizer.encode(s).ids
    targ = [tokenizer.token_to_id(sk)] + label[:-1]
    ss.append(source)
    ts.append(targ)
    ls.append(label)
  
  pad_len = max([len(s) for s in ss])

  for s in ss:
      s.extend([tokenizer.token_to_id("[PAD]")] * (pad_len - len(s)))

  pad_len = max([len(t) for t in ts])

  for t in ts:
      t.extend([tokenizer.token_to_id("[PAD]")] * (pad_len - len(t)))

  for l in ls:
      l.extend([tokenizer.token_to_id("[PAD]")] * (pad_len - len(l)))
  
  return ss, ts, ls

full_optimizer.zero_grad()

dloss, eloss = 100, 100
for epoch in range(1000):
  batch = 0
  loop = tqdm(total=len(train_dataset_loader))
  for sent in train_dataset_loader:
    batch += 1
    #print('BATCH', batch)

    # Train auto encoder
    input_enc, input_dec, labels = prep_auto_batch(sent)

    input_enc = torch.tensor(input_enc)
    input_dec = torch.tensor(input_dec)
    labels = torch.tensor(labels)

    if torch.cuda.is_available():
      input_enc = input_enc.cuda()
      input_dec = input_dec.cuda()
      labels = labels.cuda()
    
    outputs = model(input_ids=input_enc, decoder_input_ids=input_dec)
    logits = outputs.logits # only compute loss once
    loss = objective(logits.view(-1, tokenizer.get_vocab_size()), labels.view(-1)) # ignore padding in loss function

    loss.backward()

    # Train cross encoder
    input_enc, input_dec, labels = prep_cross_batch(sent, True)

    input_enc = torch.tensor(input_enc)
    input_dec = torch.tensor(input_dec)
    labels = torch.tensor(labels)

    if torch.cuda.is_available():
      input_enc = input_enc.cuda()
      input_dec = input_dec.cuda()
      labels = labels.cuda()
    
    outputs = model(input_ids=input_enc, decoder_input_ids=input_dec)
    logits = outputs.logits # only compute loss once
    loss = objective(logits.view(-1, tokenizer.get_vocab_size()), labels.view(-1)) # ignore padding in loss function

    loss.backward()

    loop.update(1)

    if batch % grad_accum == 0:
      if batch % 200 == 0:
        generated = model.generate(labels[0].unsqueeze(0), max_length=200, decoder_start_token_id=tokenizer.token_to_id('[EN]'))
        print(sent[0][0])
        print(tokenizer.decode(generated[0].cpu().numpy(),skip_special_tokens=True))

        generated = model.generate(labels[0].unsqueeze(0), max_length=200, decoder_start_token_id=tokenizer.token_to_id('[FA]'))
        print(sent[0][0])
        print(tokenizer.decode(generated[0].cpu().numpy(),skip_special_tokens=True))
      full_optimizer.step()
      full_optimizer.zero_grad()

      if batch // grad_accum % 100 == 0:
        dloss, eloss = train_descrim()
      if batch % 24 == 0:
        loop.set_description('Epoch: {} Auto: {:.3f} Descriminator: {:.3f} Fooler: {:.3f}'.format(epoch, loss.item(), dloss, eloss))
    sys.stdout.flush()
  loop.close()

  torch.save(model.state_dict(), 'best')
  torch.save(descrim.state_dict(), 'best-descrim')