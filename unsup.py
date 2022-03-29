# -*- coding: utf-8 -*-
"""401R Final Project.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Ftr030cpKz87Lhc2HKo3KtB43o66NIi9
"""

grad_accum = 4
batch_size = 8
swapProb = 2
swapProbDecay = 0.002

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

job_id = 1
jobs = sorted([f for f in os.listdir('.') if 'slurm' in f])
if jobs:
  job_id = jobs[-1][6:-4]


#from google.colab import drive
#drive.mount('/content/gdrive')

path = "data/split/"
basename = 'comb'

files = [path + basename + file for file in ["-train.en-US", "-train.fa-IR"]]

class TextDataset(Dataset):
  def __init__(self, files):
    self.data = []
    for file in files:
      lang = '['+file[-5:-3].upper()+']'
      with open(file, 'r') as f:
          self.data.extend([(l.strip(), lang) for l in f.readlines() if len(l) < 300])

  def __getitem__(self, i):
    return self.data[i]
  
  def __len__(self):
    return len(self.data)

train_dataset = TextDataset(files)

train_dataset_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True)

with open(path + basename + '-test.en-US', 'r') as l1f:
  with open(path + basename + '-test.fa-IR', 'r') as l2f:
    anchor_dataset = list(zip(l1f.readlines(), l2f.readlines()))
    test_dataset = anchor_dataset[:2500]
    del anchor_dataset[:2500]

tokenizer = Tokenizer.from_file("data/tokenizers/Sorenson.json")

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

  # swap tokens
  sel = int(random.random() * l / swapProb)

  for _ in range(sel):
    i = int(random.random() * l - 1) # don't swap cls
    j = int(random.random() * 6 - 3) # swap dist of 2 tokens
    if j>=0:
      j+=1
    j = max(0, min(len(answer)-2, i+j))
    answer[i], answer[j] = answer[j], answer[i]

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
    
    self.layers = nn.ModuleList([BartEncoderLayer(config) for _ in range(6)]) # TODO: EXPERIMENT WITH THIS NUMBER
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
  descrim_steps = 1

  descrim_optimizer.zero_grad()
  batch = 0
  loss_desc = 0
  for _ in range(grad_accum * descrim_steps * 20):
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
      nn.utils.clip_grad_norm_(descrim.parameters(), 50.0)
      descrim_optimizer.step()
      descrim_optimizer.zero_grad()

  if loss_desc > 5:
    return loss_desc, 100

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
      nn.utils.clip_grad_norm_(model.model.encoder.parameters(), 50.0)
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

    label = tokenizer.encode(s).ids
    targ = [tokenizer.token_to_id(sk)] + label[:-1]

    if first:
      trans, countUNK = dumb.dumb_translate(sk[1:3].lower(), tk[1:3].lower(), s)
      source = tokenizer.encode(trans).ids

      if countUNK > 0.5 * len(source): # skip sentences with many unknowns
        continue
    else:
      with torch.no_grad():
        generated = last_model.generate(torch.tensor(label).cuda().unsqueeze(0), max_length=200, decoder_start_token_id=tokenizer.token_to_id(tk))
        trans = tokenizer.decode(generated[0].cpu().numpy(),skip_special_tokens=True)
        trans = trans.replace(' ##', '')
        #print(trans)
        source = tokenizer.encode(trans).ids

    ss.append(source)
    ts.append(targ)
    ls.append(label)

  if len(ss) == 0:
    return None
  
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

last_model = copy.deepcopy(model)

dloss, eloss = 100, 100
for epoch in range(90):
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

    if batch % 8 == 0:
      # Train cross encoder
      cross_batch = prep_cross_batch(sent, epoch * len(train_dataset_loader) + batch < 100000)

      if cross_batch:
        input_enc, input_dec, labels = cross_batch
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
      nn.utils.clip_grad_norm_(model.parameters(), 50.0)
      full_optimizer.step()
      full_optimizer.zero_grad()

      if loss.item() < 2 and batch // grad_accum % 100 == 0:
        dloss, eloss = train_descrim()
      if batch // grad_accum % 200 == 0: 
        swapProb += swapProbDecay
      if batch // grad_accum % 3 == 0:
        loop.set_description('Epoch: {} Auto: {:.3f} Descriminator: {:.3f} Fooler: {:.3f}'.format(epoch+1, loss.item(), dloss, eloss))
    sys.stdout.flush()
  loop.close()

  torch.save(model.state_dict(), 'data/output/weights-'+job_id)
  torch.save(descrim.state_dict(), 'data/output/descrim-weights-'+job_id)

  last_model = copy.deepcopy(model)
  last_model.eval()

model.eval()

def translate(sent, targ):
  label = torch.tensor(tokenizer.encode(sent.strip()).ids)
  if torch.cuda.is_available():
    label = label.cuda()
  generated = last_model.generate(label.unsqueeze(0), max_length=200, decoder_start_token_id=tokenizer.token_to_id(targ))
  trans = tokenizer.decode(generated[0].cpu().numpy(),skip_special_tokens=True)
  trans = trans.replace(' ##', '')
  return trans

# Run test set
with torch.no_grad():
  with open('data/output/' + basename + '-' + job_id + '.en-US', 'w') as out1:
    with open('data/output/' + basename + '-' + job_id + '.fa-IR', 'w') as out2:
      for l1, l2 in test_dataset:
        out2.write(translate(l1, '[FA]') + '\n')
        out2.flush()
        out1.write(translate(l2, '[EN]') + '\n')
        out1.flush()