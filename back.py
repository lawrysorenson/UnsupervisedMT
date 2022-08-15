# -*- coding: utf-8 -*-
"""401R Final Project.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Ftr030cpKz87Lhc2HKo3KtB43o66NIi9
"""
import sys

basename = sys.argv[1]
seed_model = sys.argv[2]
seed_base = sys.argv[3]

print('2-Step translation')

print(basename)
print(seed_model)
print(seed_base)

grad_accum = 4
batch_size = 8
swapProb = 2
swapProbDecay = 0.002

import sys
import torch
from transformers import EncoderDecoderModel, BertTokenizer
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
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
from nltk.translate.chrf_score import sentence_chrf

import dumb

from util import BartClassificationHead, BartEncoderLayer

#job_id = 1
#jobs = sorted([f for f in os.listdir('.') if 'slurm' in f])
#if jobs:
#  job_id = jobs[-1][6:-4]

job_id = os.environ.get('JOB_ID', 1)

print(job_id)

#from google.colab import drive
#drive.mount('/content/gdrive')

path = "data/split/"

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

  def shuffle(self):
    random.shuffle(self.data)

mono_en_dataset = TextDataset([files[0]])
mono_en_dataloader = DataLoader(mono_en_dataset, shuffle=False, batch_size=batch_size*16)
mono_fa_dataset = TextDataset([files[1]])
mono_fa_dataloader = DataLoader(mono_fa_dataset, shuffle=False, batch_size=batch_size*16)

class CrossDataset(Dataset):
  def __init__(self, data):
    self.data = data

  def __getitem__(self, i):
    return self.data[i]
  
  def __len__(self):
    return len(self.data)

  def shuffle(self):
    random.shuffle(self.data)

with open(path + basename + '-test.en-US', 'r') as l1f:
  with open(path + basename + '-test.fa-IR', 'r') as l2f:
    anchor_dataset = list(zip(l1f.readlines(), l2f.readlines()))
    test_dataset = anchor_dataset[:2500]
    del anchor_dataset[:2500]
    val_size = min(len(anchor_dataset) // 5, 2000)
    val_dataset = anchor_dataset[:val_size]
    del anchor_dataset[:val_size]
    print(len(anchor_dataset))

train_dataset = CrossDataset(anchor_dataset)
val_dataset = CrossDataset(val_dataset)
val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size*16)

tokenizer = Tokenizer.from_file("data/tokenizers/"+seed_base+".json")
#tokenizer = Tokenizer.from_file("data/tokenizers/"+basename+".json")

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

objective = nn.CrossEntropyLoss(ignore_index = tokenizer.token_to_id("[PAD]"))

full_optimizer = optim.Adam(model.parameters(), lr=3e-4)

def prep_anchor_batch(l1s, l2s):
  ss = []
  ts = []
  ls = []

  for l1, l2 in zip(l1s, l2s):
    source = noise(l1)
    label = tokenizer.encode(l2).ids
    targ = [tokenizer.token_to_id('[FA]')] + label[:-1]

    ss.append(source)
    ts.append(targ)
    ls.append(label)

    source = noise(l2)
    label = tokenizer.encode(l1).ids
    targ = [tokenizer.token_to_id('[EN]')] + label[:-1]

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

def translate(sent, targ):
  label = torch.tensor(tokenizer.encode(sent.strip()).ids)
  if torch.cuda.is_available():
    label = label.cuda()
  generated = model.generate(label.unsqueeze(0), min_length=1, max_length=200, decoder_start_token_id=tokenizer.token_to_id(targ))
  trans = tokenizer.decode(generated[0].cpu().numpy(),skip_special_tokens=True)
  trans = trans.replace(' ##', '')
  return trans

def translate_batch(sents, lang):
  label = [tokenizer.encode(sent.strip()).ids for sent in sents] # generate labels
  pad_len = max([len(s) for s in label])
  for s in label:
      s.extend([tokenizer.token_to_id("[PAD]")] * (pad_len - len(s)))

  label = torch.tensor(label)
  if torch.cuda.is_available():
    label = label.cuda()

  generated = model.generate(label, min_length=1, max_length=200, decoder_start_token_id=tokenizer.token_to_id(lang))
  return [tokenizer.decode(trans,skip_special_tokens=True).replace(' ##', '') for trans in generated.cpu().numpy()]

full_optimizer.zero_grad()

best_chrf = 0

early_stop = 0
train_dataset = CrossDataset(anchor_dataset)
back_steps = 1 # 30
for back_trans in range(back_steps):
  model = BartForConditionalGeneration(configuration)
  if torch.cuda.is_available():
    model.cuda()

  if back_trans==0:
   model.load_state_dict(torch.load('data/output/weights-'+seed_model))
   model.train()
  
  full_optimizer = optim.Adam(model.parameters(), lr=3e-4)

  best_sub_chrf = 0

  train_dataset_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, pin_memory=True)
  train_length = len(train_dataset_loader)
  #train_length = min(2, train_length)
  batch = 0
  full_optimizer.zero_grad()

  found_better = False
  early_sub_stop = 0
  for epoch in range(100):
    #random.shuffle(anchor_dataset)
    #break

    loop = tqdm(total=train_length)
    for stopper, (l1s, l2s) in enumerate(train_dataset_loader):
      if stopper >= train_length: # limit size of epoch
        break
      #break

      batch += 1

      input_enc, input_dec, labels = prep_anchor_batch(l1s, l2s)

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

      if batch % grad_accum == 0:
        if batch % 200 == 0:
          generated = model.generate(labels[1].unsqueeze(0), min_length=1, max_length=200, decoder_start_token_id=tokenizer.token_to_id('[FA]'))
          print(l1s[0])
          print(tokenizer.decode(generated[0].cpu().numpy(),skip_special_tokens=True))

          generated = model.generate(labels[0].unsqueeze(0), min_length=1, max_length=200, decoder_start_token_id=tokenizer.token_to_id('[EN]'))
          print(l2s[0])
          print(tokenizer.decode(generated[0].cpu().numpy(),skip_special_tokens=True))
        nn.utils.clip_grad_norm_(model.parameters(), 50.0)
        full_optimizer.step()
        full_optimizer.zero_grad()

        if batch // grad_accum % 200 == 0:
          swapProb += swapProbDecay
        if batch // grad_accum % 3 == 0:
          loop.set_description('Backtick: {} Epoch: {} Loss: {:.3f}'.format(back_trans, epoch+1, loss.item()))
      loop.update(1)
      sys.stdout.flush()
    loop.close()

    if epoch % 1 == 0:
      # Run validation set
      model.eval()
      with torch.no_grad():
        loop = tqdm(total=len(val_dataloader))
        loop.set_description('Backtick: {} Validation Epoch: {}'.format(back_trans, epoch+1))
        chrf = 0
        for l1, l2 in val_dataloader:
          for trans, ref in zip(translate_batch(l1, '[FA]'), l2):
            chrf += sentence_chrf(trans, ref)
          for trans, ref in zip(translate_batch(l2, '[EN]'), l1):
            chrf += sentence_chrf(trans, ref)
          loop.update(1)
        chrf /= len(val_dataset) * 2
        loop.close()
      model.train()

      if chrf > best_sub_chrf:
        print('Saving sub model...', chrf, '>', best_sub_chrf)
        best_sub_chrf = chrf
        torch.save(model.state_dict(), 'temp-weights-'+job_id)
        early_sub_stop = 0
      else:
        print('Skip save sub model...', chrf, '<', best_sub_chrf)
        early_sub_stop += 1
        if early_sub_stop >= 4:
          break

      if chrf > best_chrf:
        print('Saving model...', chrf, '>', best_chrf)
        best_chrf = chrf
        found_better = True
        torch.save(model.state_dict(), 'data/output/weights-'+job_id)
  
  if found_better:
    early_stop = 0
  else:
    early_stop += 1
    if early_stop >= 1:
      break

  # Refresh dataset with backtranslated data
  if back_trans+1 < back_steps:
    model.load_state_dict(torch.load('temp-weights-'+job_id))
    model.eval()

    with torch.no_grad():
      new_dataset = copy.deepcopy(anchor_dataset)

      loop = tqdm(total=(len(mono_en_dataloader) + len(mono_fa_dataloader)))
      loop.set_description('Back translating')
      for sents, lang in mono_en_dataloader: # From english
        trans = translate_batch(sents, '[FA]')
        for sent, tran in zip(sents, trans):
          new_dataset.append((sent, tran))
          #print((sent, tran))
        loop.update(1)

      for sents, lang in mono_fa_dataloader: # from persian
        trans = translate_batch(sents, '[EN]')
        for sent, tran in zip(sents, trans):
          new_dataset.append((tran, sent))
          #print((tran, sent))
        loop.update(1)
      loop.close()

      train_dataset = CrossDataset(new_dataset)

model.load_state_dict(torch.load('data/output/weights-'+job_id))
model.eval()

# Run test set
with torch.no_grad():
  with open('data/output/' + basename + '-' + job_id + '.en-US', 'w') as out1:
    with open('data/output/' + basename + '-' + job_id + '.fa-IR', 'w') as out2:
      test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size*16)
      loop = tqdm(total=len(test_dataloader))
      loop.set_description('Translating test set')
      for l1s, l2s in test_dataloader:
        for trans in translate_batch(l1s, '[FA]'):
          out2.write(trans + '\n')
        out2.flush()
        for trans in translate_batch(l2s, '[EN]'):
          out1.write(trans + '\n')
        out1.flush()
        loop.update(1)
      loop.close()

print(len(anchor_dataset))
