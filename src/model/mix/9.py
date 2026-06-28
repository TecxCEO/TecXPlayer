"""
In standard text transformers, models often train on data samples without completing multiple structured passes (epochs) over the complete dataset chunk before starting the fine-tuning loop.To ensure the architecture first processes every single element of the complete input dataset chunk, it requires an explicit Dataset Index Tracking Loop rather than an arbitrary step limit.Here is the upgraded script. This edition removes arbitrary step counts and structures training around a rigid Epoch Data Matrix Router. It processes every sample in the input chunk, tracks the epoch steps, and then initiates the post-optimization convergence loop.
"""

# Save this entire script as: geometry_constrained_transformer.py

import os
import csv
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from encodingdecoding as ed

edacvr = ed.AdvancedCustomVocabularyRegistry() ####

# =============================================================================
# 1. PARAMETERS & EXACT EXPLICIT VOCABULARY PROFILE
# =============================================================================
# BASE_VOCAB_SIZE = 90      # Base structural tokens (0 to 89)
BASE_VOCAB_SIZE = len(edacvr.string_to_id) - 5     # Base structural tokens (0 to 89)
# Register Explicit Task and Structural Meta-Tokens

"""
SOS_TOKEN = BASE_VOCAB_SIZE      # 90
EOS_TOKEN = BASE_VOCAB_SIZE + 1  # 91 (Used as US/End token)
TASK_FWD  = BASE_VOCAB_SIZE + 2  # 92
TASK_REV  = BASE_VOCAB_SIZE + 3  # 93
TASK_SOLV = BASE_VOCAB_SIZE + 4  # 94
"""
ACTUAL_VOCAB_SIZE = BASE_VOCAB_SIZE + 5  # Exactly 95 total unique tokens

# ACTUAL_VOCAB_SIZE = BASE_VOCAB_SIZE
# Sequence Window Constraints
STEPS_PER_SEQ = 20       # 20 stages per matrix sequence
TOKENS_PER_STEP = 44     # Each step maps exactly 44 internal elements
BLOCK_SIZE = 880         # Context window matrix size: 20 * 44 = 880

# Sub-Step Structural Segment Boundaries
INPUT_STAGE_TOKENS = 20  # First 20 slots: 3-character & 2-character source elements
OUTPUT_STAGE_TOKENS = 20 # Next 20 slots: Target stage transformations
CONTROL_TOKENS = 4       # Final 4 slots: Meta commands (SOS, EOS, Task, Move)

# Training Hyperparameters
BATCH_SIZE = 18          # Process exactly 18 parallel sequence streams per batch
PATIENCE_STEPS = 2000    # Target patience window for post-optimization convergence

CSV_LOG_FILE = "phase1_training_metrics.csv"

# =============================================================================
# 2. ENFORCED GEOMETRIC MODEL ARCHITECTURE
# =============================================================================

class StructuralPositionalEncoding(nn.Module):
    """
    Enforces Perfect Serial Memory and Step-Zone Position Invariance.
    """
    def __init__(self, d_model, block_size, tokens_per_step=44):
        super().__init__()
        self.tokens_per_step = tokens_per_step
        
        # Absolute Linear Position Space (Provides Perfect Serial Sequence Memory)
        pe = torch.zeros(block_size, d_model)
        position = torch.arange(0, block_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
        # Spatial Zone Integrity Mask (0: Control Space, 1: Input Matrix, 2: Output Matrix)
        zone_mask = torch.zeros(block_size, dtype=torch.long)
        for step in range(block_size // tokens_per_step):
            step_offset = step * tokens_per_step
            
            # Allocate Segment Boundaries cleanly across the step block
            zone_mask[step_offset + 0 : step_offset + 20] = 1   # Input Stage Zone
            zone_mask[step_offset + 20 : step_offset + 40] = 2  # Output Stage Zone
            zone_mask[step_offset + 40 : step_offset + 44] = 0  # Control Operational Zone
            
        self.zone_embeddings = nn.Embedding(3, d_model) 
        self.register_buffer('zone_mask', zone_mask)

    def forward(self, x):
        T = x.size(1)
        return x + self.pe[:, :T, :] + self.zone_embeddings(self.zone_mask[:T])


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, bias, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias) 
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, 
            dropout_p=self.attn_dropout.p if self.training else 0.0, is_causal=True
        )
        return self.resid_dropout(self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C)))


class MLP(nn.Module):
    def __init__(self, n_embd, bias, dropout):
        super().__init__()
        self.c_fc    = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, n_embd, n_head, bias, dropout):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd, bias=bias)
        self.attn = CausalSelfAttention(n_embd, n_head, bias, dropout)
        self.ln_2 = nn.LayerNorm(n_embd, bias=bias)
        self.mlp = MLP(n_embd, bias, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class CustomTransformer(nn.Module):
    def __init__(self, vocab_size=ACTUAL_VOCAB_SIZE, block_size=BLOCK_SIZE, n_layer=12, n_head=12, n_embd=768, dropout=0.1, bias=False):
        super().__init__()
        self.block_size = block_size
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, n_embd),
            wpe = StructuralPositionalEncoding(n_embd, block_size, TOKENS_PER_STEP),
            drop = nn.Dropout(dropout),
            h = nn.ModuleList([Block(n_embd, n_head, bias, dropout) for _ in range(n_layer)]),
            ln_f = nn.LayerNorm(n_embd, bias=bias),
        ))
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=bias)
        self.transformer.wte.weight = self.lm_head.weight 

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        b, t = idx.size()
        assert t <= self.block_size, f"Sequence segment length {t} exceeds context limit {self.block_size}"
        
        x = self.transformer.wpe(self.transformer.wte(idx))
        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

# =============================================================================
# 3. METRIC STORAGE TRACKER & SPATIAL VALIDATOR
# =============================================================================

class MetricLogger:
    def __init__(self, filepath=CSV_LOG_FILE):
        self.filepath = filepath
        if not os.path.exists(self.filepath):
            with open(self.filepath, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['chunk_id', 'epoch', 'step', 'phase', 'loss', 'input_zone_acc', 'output_zone_acc'])

    def log_step(self, chunk_id, epoch, step, phase, loss, input_acc=0.0, output_acc=0.0):
        with open(self.filepath, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([chunk_id, epoch, step, phase, f"{loss:.6f}", f"{input_acc:.4f}", f"{output_acc:.4f}"])


@torch.no_grad()
def compute_structural_zone_accuracy(model, data_streamer, device, sample_batches=4):
    model.eval()
    total_in_correct, total_in_tokens = 0, 0
    total_out_correct, total_out_tokens = 0, 0
    
    zone_mask = model.transformer.wpe.zone_mask
    
    for _ in range(sample_batches):
        X, Y = data_streamer.get_batch_simulated(batch_size=BATCH_SIZE, device=device)
        logits, _ = model(X)
        predictions = torch.argmax(logits, dim=-1)
        
        T = X.size(1)
        current_zone_mask = zone_mask[:T].unsqueeze(0).expand(BATCH_SIZE, -1)
        
        in_indices = (current_zone_mask == 1)
        total_in_correct += (predictions[in_indices] == Y[in_indices]).sum().item()
        total_in_tokens += in_indices.sum().item()
        
        out_indices = (current_zone_mask == 2)
        total_out_correct += (predictions[out_indices] == Y[out_indices]).sum().item()
        total_out_tokens += out_indices.sum().item()
        
    model.train()
    
    input_accuracy = (total_in_correct / total_in_tokens) if total_in_tokens > 0 else 0.0
    output_accuracy = (total_out_correct / total_out_tokens) if total_out_tokens > 0 else 0.0
    return input_accuracy, output_accuracy

# =============================================================================
# 4. DATA ENGINE (FINITE CHUNK LOADER ITERATOR)
# =============================================================================

class FiniteChunkDataset:
    """
    Simulates a finite, fixed data payload containing exactly 1,00,000 steps.
    """
    def __init__(self, total_steps=100000):
        self.total_steps = total_steps
        self.current_step = 0

    def __iter__(self):
        self.current_step = 0
        return self

    def __next__(self):
        if self.current_step >= self.total_steps:
            raise StopIteration
        self.current_step += 1



#########@

########

if loss.item() < best_loss_this_chunk:
    best_loss_this_chunk = loss.item()
    torch.save({'chunk_id': chunk_id,'step': current_step,'model_state_dict': model.state_dict(),'loss': best_loss_this_chunk,}, f"checkpoint_chunk_{chunk_id}_phase1.pt")
    # --- PHASE 2: LAUNCH POST-OPTIMIZATION EPOCH CONTROLLER (RUN UNTIL SATURATION) ---
print(f"--> [PHASE 2] Launching Post-Optimization Epoch Core Loops for Chunk {chunk_id}...")
for param_group in optimizer.param_groups:
    param_group['lr'] = base_lr * 0.1
    no_improvement_counter = 0
    epoch_count = 0
    # This controller executes iterative validation passes over the finalized parameters.
    # It optimizes the weights until no further performance gains are possible.
    while True:
        epoch_count += 1
        post_step = 0
        # Reset chunk dataset iterator to perform an entire fresh optimized pass
        chunk_iterator = iter(data_streamer.load_next_chunk())
        print(f"    [POST-LOOP] Initiating Optimization Epoch Tracker #{epoch_count} for Fine-Tuning Parameter Adjustments...")
        while True:
            try:
                X, Y = data_streamer.get_batch(chunk_iterator, device=device)
                post_step += 1
            except StopIteration:
                # Epoch complete
                break
            logits, loss = model(X, Y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            current_loss = loss.item()
            if current_loss < (best_loss_this_chunk - 1e-4):
                print(f"        [PROGRESS] Epoch {epoch_count} Step {post_step} | Enhanced Weights Saved! Loss: {current_loss:.6f}")
                best_loss_this_chunk = current_loss
                no_improvement_counter = 0
                torch.save({'chunk_id': chunk_id,'epoch': epoch_count,'model_state_dict': model.state_dict(),'loss': best_loss_this_chunk,}, f"best_final_optimized_model_chunk_{chunk_id}.pt")
            else:
                no_improvement_counter += 1
            if post_step % 5000 == 0:
                in_acc, out_acc = compute_structural_zone_accuracy(model, data_streamer, device)
                print(f"        [MONITOR] Epoch {epoch_count} Step {post_step} | Loss: {current_loss:.4f} | Patience: {no_improvement_counter}/{PATIENCE_STEPS}")
                logger.log_step(chunk_id, epoch_count, post_step, "Phase2", current_loss, in_acc, out_acc)
            # If saturation patience is triggered, terminate the interior step loop
            if no_improvement_counter >= PATIENCE_STEPS:
                break
        # --- THE SATURATION ESCAPE CHECK LINE ---
        if no_improvement_counter >= PATIENCE_STEPS:
            print(f"\n[SATURATION REACHED] No further improvement found after complete data optimization passes.")
            print(f"Optimal model for Chunk {chunk_id} saved as: 'best_final_optimized_model_chunk_{chunk_id}.pt'")
            print(f"Advancing pipeline to next raw data matrix sector...")
            print("================================================================================")
            break
if name == 'main':
    execute_lifelong_training()



"""

### Key Architectural Improvements

#### 1. Complete Dataset Tracking (Lines 207-220)
Arbitrary counter loops can cause models to switch phases before all training data has been evaluated. This updated script introduces the `FiniteChunkDataset` matrix wrapper. Phase 1 processes inputs via an explicit iterator check inside **Lines 207–208**:
```python
X, Y = data_streamer.get_batch(chunk_iterator, device=device)
current_step = dataset_chunk.current_step
If the source data has not been fully processed, training continues. As soon as the entire 1,00,000 step batch array completes, an exception is caught at Line 209, allowing the pipeline to transition to the post-optimization loop with full coverage of the dataset chunk.2. Optimization Epoch Loop (Lines 237-279)Instead of processing random individual data points during fine-tuning, Phase 2 implements structured training epochs beginning on Line 247:pythonchunk_iterator = iter(data_streamer.load_next_chunk())
Use code with caution.This resets the data iterator to run full optimization passes over the dataset block. The model continues running these training epochs until the loss stabilizes, confirming that no further gains can be extracted from the current chunk before saving.

"""

