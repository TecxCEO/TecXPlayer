# Save this entire unified script as: geometry_constrained_transformer.py

import os
import csv
import json
import math
import string
#import torch
#import torch.nn as nn
#from torch.nn import functional as F

#=============================================================================
# 1. HARDCODED PARAMETERS & CONFIGURATION PROFILE
#=============================================================================
# Context Window Machine Variables
STEPS_PER_SEQ = 20 
# 20 stages per matrix sequence
TOKENS_PER_STEP = 44 
# Each step maps exactly 44 internal elements
BLOCK_SIZE = 880 
# Context window matrix size: 20 * 44 = 880
# Sub-Step Structural Segment Boundaries
INPUT_STAGE_TOKENS = 20 
# First 20 slots: 3-character & 2-character source elements
OUTPUT_STAGE_TOKENS = 20 
# Next 20 slots: Target stage transformations
CONTROL_TOKENS = 4 
# Final 4 slots: Meta commands (SOS, EOS, Task, Move)
# Training Hyperparameters
BATCH_SIZE = 18 
# Process exactly 18 parallel sequence streams per batch
PATIENCE_STEPS = 2000 
# Target patience window for post-optimization convergence
ACCURACY_GATE_MIN = 0.98 
# Minimum 98% accuracy required to pass performance gate
CSV_LOG_FILE = "phase1_training_metrics.csv"
VOCAB_LEDGER_FILE = "structural_vocabulary_ledger.json"
#=============================================================================
# 2. END-TO-END TIERED VOCABULARY ENGINE & REGISTRY
#=============================================================================
class AdvancedCustomVocabularyRegistry:
  """
  Implements a strict Tiered Tokenization Hierarchy compiling atomic characters,standard layout items, exhaustive scientific/engineering/computation symbols,grammatical structures, and specialized 95-number cube operations into a continuous indexline.
  """
  def __init__(self, ledger_path=VOCAB_LEDGER_FILE):
    self.ledger_path = ledger_path
    self.vocab_map = {}
    self.string_to_id = {}
    self.current_id = 0
    self._compile_exhaustive_hierarchy()
  def _add_token(self, token_str, tier_type, sub_category=""):
    if token_str not in self.string_to_id:
      token_id = self.current_id
      self.vocab_map[token_id] = {
        "token_id": token_id,
        "string_representation": token_str,
        "tier_type": tier_type,
        "sub_category": sub_category
      }
      self.string_to_id[token_str] = token_id
      self.current_id += 1
  def _compile_exhaustive_hierarchy(self):
    # TIER 5: Special Cube Operators (The Precise 95 Token Geometry Line)
    # ---------------------------------------------------------------------
    mosf={'b':'g','g':'b','o':'r','r':'o','w':'y','y':'w'}
    # 1. Three-Character Family Tokens (8 Families * 6 Flips = 48 Tokens)
    for fc in mosf.keys():
      for sc in mosf.keys():
        for tc in mosf.keys():
          if sc not in [fc, mosf[fc]] and tc not in [fc, mosf[fc], sc, mosf[sc]]:
            self._add_token(f"{fc}{sc}{tc}", "Tier_5_Cube", "3_Char_Family_Token")
            self._add_token(f"{fc}{tc}{sc}", "Tier_5_Cube", "3_Char_Family_Token")
            self._add_token(f"{sc}{fc}{tc}", "Tier_5_Cube", "3_Char_Family_Token")
            self._add_token(f"{sc}{tc}{fc}", "Tier_5_Cube", "3_Char_Family_Token")
            self._add_token(f"{tc}{fc}{sc}", "Tier_5_Cube", "3_Char_Family_Token")
            self._add_token(f"{tc}{sc}{fc}", "Tier_5_Cube", "3_Char_Family_Token")
    # 2. Two-Character Family Tokens (12 Families * 2 Flips = 24 Tokens)
    for fc in mosf.keys():
      for sc in mosf.keys():
        if sc not in [fc, mosf[fc]]:
          self._add_token(f"{fc}{sc}", "Tier_5_Cube", "2_Char_Family_Token")
          self._add_token(f"{sc}{fc}", "Tier_5_Cube", "2_Char_Family_Token")
    # 3. Possible Action Move Vectors (18 Tokens)
    move_paths=["<rgy>","<rgw>","<rgo>","<rby>","<rbw>","<rbo>","<grw>","<gry>","<grb>","<gow>","<goy>","<gob>","<yrg>","<yrb>","<yrw>","<yog>","<yob>","<yow>"]
    for m in range(18):
      self._add_token(move_paths[m], "Tier_5_Cube", "Action_Move_Token")
    # 4. Functional Meta Command Cues (5 Tokens)
    self._add_token("<SOS>", "Tier_5_Cube", "Control_SOS")
    self._add_token("<EOS>", "Tier_5_Cube", "Control_EOS_US")
    self._add_token("TASK_FWD", "Tier_5_Cube", "Control_TASK_FWD")
    self._add_token("TASK_REV", "Tier_5_Cube", "Control_TASK_REV")
    self._add_token("TASK_SOLV", "Tier_5_Cube", "Control_TASK_SOLV")
    # Write absolute compiled snapshot map ledger out to disk
    with open(self.ledger_path, "w", encoding="utf-8") as f:
      json.dump(self.vocab_map, f, indent=4, ensure_ascii=False)
  @property
  def total_vocab_size(self):
    return self.current_id
  def encode_sequence(self, string_list):
    return torch.tensor([self.string_to_id[s] for s in string_list], dtype=torch.long)
  def decode_sequence(self, tensor_or_list):
    if isinstance(tensor_or_list, torch.Tensor):
      tensor_or_list = tensor_or_list.tolist()
    return [self.vocab_map[idx]["string_representation"] for idx in tensor_or_list]
#
class EncodeDecode:
  def __init__(self, full_text = None):
    self.full_text = full_text
    acvr = AdvancedCustomVocabularyRegistry()
    token_str = acvr.vocab_map[token_id]["string_representation"]
    token_id = acvr.string_to_id[token_str]
    
    # here are all the unique characters that occur in this text
    # Define the components
    
    #lowercase = string.ascii_lowercase          # a-z (26)
    #uppercase = string.ascii_uppercase          # A-Z (26)
    #digits = string.digits                      # 0-9 (10)
    # special = """ !.,{"'}()[]:;?-\n"""
    # Combine them into one string
    #chars = lowercase + uppercase + digits + special
    #chars = sorted(list(set(chars)))
  
    # Create new stoi with special tokens
    self.stoi = {ch: i for i, ch in enumerate(chars, start=3)} # Shift everything by 3
    self.stoi['<PAD>'] = 0
    self.stoi['<SOS>'] = 1
    self.stoi['<EOS>'] = 2
    self.itos = {i: ch for ch, i in self.stoi.items()} # Reverse map
    print(f" stoi = {self.stoi}")
    print(f" stoi = {len(self.stoi)}")
    print(f" itos = {self.itos}")
    print(f" itos = {len(self.itos)}")
    
    self.encode = lambda s: [self.stoi[c] for c in s] # encoder: take a string, output a list of integers
    self.decode = lambda l: ''.join([self.itos[i] for i in l]) # decoder: take a list of integers, output a string
    
  def encoder(self, str_in):
    if self.encode(str_in):
      # Returns the value if found, otherwise returns None
      return self.encode(str_in)
    elif not self.encode(str_in) and not isinstance(str_in, (dict, list)):
      self.createTokens(str_in)
      return self.encode(str_in)
    elif isinstance(str_in, (dict, list)):
      for strin in str_in:
        if isinstance(strin, (dict, list)):
          if self.encode(strin):
            return self.encode(strin)
          else:
            self.encoder(strin)
  def return_stoi_size(self):
    return len(self.stoi)
  def createTokens(self, full_text, stoi, itos):
    self.stoi = stoi
    self.itos = itos
    print(f"stoi len before token cretion = {len(self.stoi)}")
    if isinstance(full_text, dict):
      print(f" full_text = {len(full_text)}")
      for key, value in full_text.items(): 
        str="<"+key+">"
        if not self.stoi.get(str) and isinstance(value, dict):
          t=len(self.stoi)
          self.stoi[str] = t
        if isinstance(value, dict) and len(value) in (16, 19, 20):
          # This works! It checks if len(value) is 16, 19, or 20.
          print(f"deep dive in dict of {key} of {len(value)} len value.\n")
          self.stoi, self.itos = self.createTokens(value, self.stoi, self.itos)
        elif not isinstance(value, (dict, list)):
          if not self.stoi.get(key) and key != "state":
            t=len(self.stoi) ######
            self.stoi[key] = t
          if  not self.stoi.get(value):
            t=len(self.stoi)
            self.stoi[value] = t
      print(f"itos len = {len(self.itos)}")
      self.itos = {i: ch for ch, i in self.stoi.items()}
      print(f"itos len at the end of createTokens in encoding decoding = {len(self.itos)}")
      print(f"stoi len at the end of createTokens in encoding decoding = {len(self.stoi)}")
      return self.stoi, self.itos
if __name__ == "__main__":
  acvr = AdvancedCustomVocabularyRegistry()
  print(f" vocab map = {acvr.vocab_map}\n#\n")
  print(f" string to id = {acvr.string_to_id}")
  # if __name__ == "__main__":
  # print(f"At start")
  # import json
  file= "data/dataset/cube3x3solvingdataset.json"
  with open(file, 'r') as f:
            data = json.load(f)
  edc = EncodeDecode(data['solution'])
  result=edc.createTokens(data["solution"])
  print(f" Result= {result}\n")
  print(f"stoi = {edc.stoi}\n\n")
  print(f"itos = {edc.itos}")
#
