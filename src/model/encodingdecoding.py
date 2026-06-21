# Save this entire unified script as: geometry_constrained_transformer.py

import os
import csv
import json
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

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
    self._add_token("SOS", "Tier_5_Cube", "Control_SOS")
    self._add_token("EOS_US", "Tier_5_Cube", "Control_EOS_US")
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
if __name__ == "__main__":
  acvr = AdvancedCustomVocabularyRegistry()
  print(f" vocab map = {acvr.vocab_map}\n#\n")
  print(f" string to id = {acvr.string_to_id}")
