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
    """
      # ---------------------------------------------------------------------
      # TIER 1: Foundational Atomic Characters
      # ---------------------------------------------------------------------
      for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        self._add_token(char,"Tier_1_Atomic", "Uppercase_Latin")
      for char in "abcdefghijklmnopqrstuvwxyz":
        self._add_token(char, "Tier_1_Atomic","Lowercase_Latin")
      for num in "0123456789":
        self._add_token(num, "Tier_1_Atomic", "Numeric_Digits")
      # ---------------------------------------------------------------------
      # TIER 2: Basic Text & Formatting Special Characters
      # ---------------------------------------------------------------------
      basic_special_chars = [
        " ", "!", '"', "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/", ":", ";", "<", "=", ">", "?", "@", "[", "\\", "]", "^", "_",., "`""{", "|", "}", "~", "\n", "\t"
        ]
      for char in basic_special_chars:
        self._add_token(char, "Tier_2_Basic_Text", "ASCII_Punctuation")
      # ---------------------------------------------------------------------
      # TIER 3: Complete Scientific, Engineering & Computation Glyphs
      # ---------------------------------------------------------------------
      # Sub-Block A: Calculus & General Mathematicsmath_symbols = ["±", "mp", "×", "÷", "·", "°", "∇", "Δ", "∂", "∫", "∬", "iiint", "oint","∑", "∏", "coprod", "√", "cbrt", "∞"]
      # Sub-Block B: Set Theory, Mathematical Logic & Proof Markerslogic_symbols = ["∀", "∃", "∄", "∈", "∉", "∋", "ℵ", "∅", "⊆", "⊇", "⊂", "⊃", "∪", "cap","∧", "∨", "¬", "⇒", "⇔", "∴", "∵", "■"]
      # Sub-Block C: Spaces, Set Enforcers & Matrix Fieldsmatrix_fields = ["ℝ", "ℂ", "ℤ", "ℕ", "ℚ", "d_model", "n_head", "d_ff"]# Sub-Block D: Geometric Properties & Equationsgeometry_symbols = ["≈", "≠", "≡", "≅", "∝", "∼", "≤", "≥", "≪", "≫", "∥", "⊥", "∠","measuredangle"]
      # Sub-Block E: Physics Constants & Electrical Engineeringphysics_units = ["ℏ", "h_bar", "μ_0", "ε_0", "k_B", "m_e", "m_p", "Ω", "mho", "λ", "Å", "μ", "σ", "ρ", "τ"]# Sub-Block F: Chemical Equations & Thermodynamicschem_thermo = ["ΔH", "ΔS", "pH", "⇌", "→", "←", "↑", "↓", "[CHEM]", "[PHYS]"]# Sub-Block G: Tensor Matrix Topology, AI/ML Formulations & Quantum Computeai_tensor_quantum = ["⊗", "⊕", "⊙", "×_n", ".T", ".†", "⟨", "⟩", "|", "ψ", "ϕ", "σ_x", "σ_y", "σ_z", "𝒪", "Null"]# Sub-Block H: Full Classical Greek Alphabet Matrix (Calculus & Physics Variables)greek_alphabet = ["α", "β", "γ", "δ", "ε", "ζ", "η", "θ", "ι", "κ", "λ", "μ", "ν", "ξ", "ο", "π", "ρ", "σ", "τ", "υ", "φ", "χ","ψ", "ω","Α", "Β", "Γ", "Δ", "Ε", "Ζ", "Η", "Θ", "Ι", "Κ", "Λ", "Μ", "Ν", "Ξ", "Ο", "Π", "Ρ", "Σ", "Τ", "Υ","Φ", "Χ", "Ψ", "Ω"]for s in math_symbols: self._add_token(s, "Tier_3_Scientific", "Calculus_Math")for s in logic_symbols: self._add_token(s, "Tier_3_Scientific", "Logic_Proof")for s in matrix_fields: self._add_token(s, "Tier_3_Scientific", "Matrix_Fields")for s in geometry_symbols: self._add_token(s, "Tier_3_Scientific", "Geometry_Relations")for s in physics_units: self._add_token(s, "Tier_3_Scientific", "Physics_Engineering")for s in chem_thermo: self._add_token(s, "Tier_3_Scientific","Chemistry_Thermodynamics")for s in ai_tensor_quantum: self._add_token(s, "Tier_3_Scientific","AI_ML_Quantum_Tensors")for s in greek_alphabet: self._add_token(s, "Tier_3_Scientific","Greek_Linguistic_Variables")# ---------------------------------------------------------------------# TIER 4: Grammatical Part-of-Speech & Context Word Tokens# ---------------------------------------------------------------------grammar_cues = ["[NOUN]", "[VERB]", "[ADJ]", "[ADV]", "[PRON]", "[PREP]", "[CONJ]","[PAST_TENSE]", "[PLURAL]"]technical_lexicon = ["derive", "integrate", "transform", "vectorize", "tensorize", "converge", "gradient","optimize", "parameter", "matrix", "qubit", "entangle", "manifold", "orthogonal"]for s in grammar_cues: self._add_token(s, "Tier_4_Grammar", "Syntactic_Markers")for s in technical_lexicon: self._add_token(s, "Tier_4_Grammar", "Technical_Root_Words")# ---------------------------------------------------------------------
    """
    # TIER 5: Special Cube Operators (The Precise 95 Token Geometry Line)
    # ---------------------------------------------------------------------
    # 1. Three-Character Family Tokens (8 Families * 6 Flips = 48 Tokens)
   
    three_char_families = ["F_ALPHA", "F_BETA", "F_GAMMA", "F_DELTA", "F_EPSILON","F_ZETA", "F_ETA", "F_THETA"]
   
   
    flip_variants_3ch = ["FLIP_0", "FLIP_1", "FLIP_2", "FLIP_3", "FLIP_4", "FLIP_5"]

   
    for family in three_char_families:

     
      for flip in flip_variants_3ch:

       
        # self._add_token(f"{family}_{flip}", "Tier_5_Cube", "3_Char_Family_Token")
        #####self._add_token(f"{bow}", "Tier_5_Cube", "3_Char_Family_Token")
        self._add_token(f"bow", "Tier_5_Cube", "3_Char_Family_Token")
    # 2. Two-Character Family Tokens (12 Families * 2 Flips = 24 Tokens)
    two_char_families = [f"CAT_{i:02d}" for i in range(1, 13)]
  
    
    flip_variants_2ch = ["FLIP_A", "FLIP_B"]
   
   
    for family in two_char_families:

     
      for flip in flip_variants_2ch:

       
        #self._add_token(f"{family}_{flip}", "Tier_5_Cube", "2_Char_Family_Token")
        ####self._add_token(f"{bo} ", "Tier_5_Cube", "2_Char_Family_Token")
        self._add_token(f"bo ", "Tier_5_Cube", "2_Char_Family_Token")
       
    # =====================================================================
    # 3. ANTI-FLIP DICTIONARY MANAGEMENT ENGINE
    # ====================================================================
    VALID_3_CHAR_IDS = list(range(0, 8))  # 0 to 7
    VALID_2_CHAR_IDS = list(range(8, 20)) # 8 to 20
    # Map every Token ID to a unique set of letters to track flips
    TOKEN_CHARACTER_MAP = {}
    # =====================================================================
    # 1. THREE-PAIR SELECTION RULE FOR 3-CHARACTER TOKENS 
    # =====================================================================
    pair1 = ('r','o')
    pair2 = ('g','b')
    pair3 = ('y','w')
    # Create 3 distinct pairs of 2 characters each using an indexing offset
    three_pairs = [
      ('r','o'),
      ('g','b'),
      ('y','w')
    ]
    
    for idx, token_id in enumerate(VALID_3_CHAR_IDS):
      char1 = random.choice(pair1)
      char2 = random.choice(pair2)
      char3 = random.choice(pair3)
      # Wrap into order-invariant frozen set tracker mapping for this ID
      TOKEN_CHARACTER_MAP[token_id] = frozenset([char1, char2, char3])
    # =====================================================================
    # 2. UPDATED: THREE-PAIR SELECTION RULE FOR 2-CHARACTER TOKENS
    # =====================================================================
    for idx, token_id in enumerate(VALID_2_CHAR_IDS):
      # Randomly select exactly 2 distinct pairs out of the 3 available pairs
      chosen_two_pairs = random.sample(three_pairs, 2)
    
      # Randomly select exactly ONE character from each of the two chosen pairs
      char1 = random.choice(chosen_two_pairs[0])
      char2 = random.choice(chosen_two_pairs[1])
    
      # Wrap into an order-invariant frozen set tracker mapping for this ID
      TOKEN_CHARACTER_MAP[token_id] = frozenset([char1, char2])
      
    ########$$$$$#####
       
    # 3. Possible Action Move Vectors (18 Tokens)
  
    #for m in range(1, 19):
    for m in range(18):
      ##self.move_paths=["rgy","rgw","rgo","rby","rbw","rbo","grw","gry","grb","gow","goy","gob","yrg","yrb","yrw","yog","yob","yow"]
      ####move_paths=["rgy","rgw","rgo","rby","rbw","rbo","grw","gry","grb","gow","goy","gob","yrg","yrb","yrw","yog","yob","yow"]
      move_paths=["<rgy>","<rgw>","<rgo>","<rby>","<rbw>","<rbo>","<grw>","<gry>","<grb>","<gow>","<goy>","<gob>","<yrg>","<yrb>","<yrw>","<yog>","<yob>","<yow>"]
      ####self.mosf={'b':'g','g':'b','o':'r','r':'o','w':'y','y':'w'}
      mosf={'b':'g','g':'b','o':'r','r':'o','w':'y','y':'w'}
      # self._add_token(f"MOVE_{m:02d}", "Tier_5_Cube", "Action_Move_Token")
      ##self._add_token(f"{move_paths[m]}", "Tier_5_Cube", "Action_Move_Token")
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
"""
#=============================================================================
# 3. ENFORCED GEOMETRIC MODEL ARCHITECTURE
#=============================================================================
    class StructuralPositionalEncoding(nn.Module):
      #" ""
      Enforces Perfect Serial Memory and Step-Zone Position Invariance.
      #" ""
      def __init__(self, d_model, block_size, tokens_per_step=44):
        super().__init__()
        cosine_lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
        for param_group in optimizer.param_groups:
          param_group['lr'] = cosine_lr
          logits, loss = model(X,Y)
          optimizer.zero_grad(set_to_none=True)
          loss.backward()
          torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
          optimizer.step()
          if current_step % 1000 == 0:
            in_acc, out_acc =compute_structural_zone_accuracy(model, data_streamer, vocab_size, device)
            print(f"Chunk{chunk_id} | Step {current_step}/{dataset_chunk.total_steps} | Loss: {loss.item():.4f} | Input Acc: {in_acc:.2%} | LR: {cosine_lr:.6f}")
            logger.log_step(chunk_id, 0, current_step, "Phase1",loss.item(), in_acc, out_acc, cosine_lr)
          if loss.item() <best_loss_this_chunk:
            best_loss_this_chunk = loss.item()
            torch.save({'chunk_id': chunk_id,'step':current_step,'model_state_dict': model.state_dict(),'loss': best_loss_this_chunk,},f"checkpoint_chunk_{chunk_id}_phase1.pt")
          # --- PHASE 2: CONVERGENCE LOOP WITHACCURACY PERFORMANCE GATE ---
          print(f"--> [PHASE 2] Launching Fine-TuningOptimization Epochs for Chunk {chunk_id}...")
for param_group inoptimizer.param_groups:
param_group['lr'] = base_lr * 0.1
no_improvement_counter =0
epoch_count = 0
while True:
  epoch_count += 1
  post_step = 0
  chunk_iterator =iter(data_streamer.load_next_chunk())
  print(f" [EPOCH PASSTHROUGH] Starting Pass #{epoch_count} for Structural Parameter Optimization...")
  while True:
    try:
      X, Y =data_streamer.get_batch(chunk_iterator, vocab_size, device=device)
      post_step += 1
      exceptStopIteration:
      break
      logits, loss = model(X,Y)
      optimizer.zero_grad(set_to_none=True)
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      optimizer.step()
      current_loss = loss.item()
      if current_loss <(best_loss_this_chunk - 1e-4):
        in_acc, out_acc = compute_structural_zone_accuracy(model,data_streamer, vocab_size, device)
        if in_acc >= ACCURACY_GATE_MIN and out_acc >=ACCURACY_GATE_MIN:
          print(f" [GATE PASSED] Epoch {epoch_count} Step {post_step} |Loss: {current_loss:.6f} | Acc: {in_acc:.2%}. Saving.")
          best_loss_this_chunk =current_loss
          no_improvement_counter = 0
          torch.save({'chunk_id': chunk_id,'epoch':epoch_count,'model_state_dict': model.state_dict(),'loss':best_loss_this_chunk,'input_accuracy': in_acc,},f"best_final_optimized_model_chunk_{chunk_id}.pt")
        else:
          no_improvement_counter +=1
      else:
        no_improvement_counter += 1
      """
if __name__ == "__main__":
  acvr = AdvancedCustomVocabularyRegistry()
  print(f" vocab map = {acvr.vocab_map}\n#\n")
  print(f" string to id = {acvr.string_to_id}")

