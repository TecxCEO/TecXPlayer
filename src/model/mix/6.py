"""
To handle both tracking and structural validation, we need to upgrade the pipeline with two specific components: a CSV Loss Tracker to output analytical metrics, and a Strict Geometry Validation Guard to measure formatting accuracy across your processing zones.Here is the complete, unified script including both features. It automatically handles logging and checks validation rules across your token zones.
"""



# Save this entire script as: geometry_constrained_transformer.py

import os
import csv
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

# =============================================================================
# 1. PARAMETERS & GLOBAL CONFIGURATION PROFILE
# =============================================================================
VOCAB_SIZE = 95          # Exact unique token dictionary size
BLOCK_SIZE = 880         # Context Window: 20 steps * 44 tokens per step
STEPS_PER_SEQ = 20       # Sequences contain exactly 20 internal processing steps
TOKENS_PER_STEP = 44     # Every individual processing step contains 44 tokens

# Geometric Structural Constraints
THREE_CHAR_LIMIT = 8     # 8 slots reserved exclusively for 3-character tokens
TWO_CHAR_LIMIT = 12      # 12 slots reserved exclusively for 2-character tokens

# Training Hyperparameters
BATCH_SIZE = 18          # Explicitly set to 18 sequences per batch
PHASE1_STEPS = 100000    # Primary training run steps per data chunk
PATIENCE_STEPS = 2000    # Post-optimization saturation window limit

# File System Configurations
CSV_LOG_FILE = "phase1_training_metrics.csv"

# =============================================================================
# 2. ENFORCED GEOMETRIC MODEL ARCHITECTURE
# =============================================================================

class StructuralPositionalEncoding(nn.Module):
    """
    Enforces Perfect Serial Memory and Dual-Zone Segment Alignment.
    """
    def __init__(self, d_model, block_size, tokens_per_step=44):
        super().__init__()
        self.tokens_per_step = tokens_per_step
        
        # Absolute Positional Serial Coordinates (Sinusoidal Map)
        pe = torch.zeros(block_size, d_model)
        position = torch.arange(0, block_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
        # Spatial Multi-Zone Safety IDs (Categorical Enforcer)
        # Type 0: Meta/Control Space, Type 1: 3-Char Zone, Type 2: 2-Char Zone
        zone_mask = torch.zeros(block_size, dtype=torch.long)
        for step in range(block_size // tokens_per_step):
            step_offset = step * tokens_per_step
            
            # Input Stage Alignment (Tokens 0 to 23 of the step block)
            zone_mask[step_offset + 0 : step_offset + 8] = 1   # 8 slots for 3-char items
            zone_mask[step_offset + 8 : step_offset + 20] = 2  # 12 slots for 2-char items
            zone_mask[step_offset + 20 : step_offset + 24] = 0 # 4 Control/Meta tokens
            
            # Output Stage Alignment (Tokens 24 to 43 of the step block)
            zone_mask[step_offset + 24 : step_offset + 32] = 1 # 8 slots for 3-char target
            zone_mask[step_offset + 32 : step_offset + 44] = 2 # 12 slots for 2-char target
            
        self.zone_embeddings = nn.Embedding(3, d_model) # Resolves into Zone 0, 1, or 2 vectors
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
    def __init__(self, vocab_size=VOCAB_SIZE, block_size=BLOCK_SIZE, n_layer=12, n_head=12, n_embd=768, dropout=0.1, bias=False):
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
        assert t <= self.block_size, f"Sequence length {t} exceeds window boundary {self.block_size}"
        
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
# 3. AUTOMATED ANALYTICS LOGGER & STICK ZONE ACCURACY EVALUATOR
# =============================================================================

class MetricLogger:
    """
    Handles CSV logging for Phase 1 optimization tracking.
    """
    def __init__(self, filepath=CSV_LOG_FILE):
        self.filepath = filepath
        # Create file and initialize tracking headers if it doesn't exist yet
        if not os.path.exists(self.filepath):
            with open(self.filepath, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['chunk_id', 'step', 'phase', 'loss', 'zone_1_acc', 'zone_2_acc'])

    def log_step(self, chunk_id, step, phase, loss, zone_1_acc=0.0, zone_2_acc=0.0):
        with open(self.filepath, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([chunk_id, step, phase, f"{loss:.6f}", f"{zone_1_acc:.4f}", f"{zone_2_acc:.4f}"])


@torch.no_grad()
def compute_strict_zone_accuracy(model, data_streamer, device, sample_batches=4):
    """
    Validates structural integrity across Zone 1 (3-char) and Zone 2 (2-char).
    """
    model.eval()
    total_z1_correct, total_z1_tokens = 0, 0
    total_z2_correct, total_z2_tokens = 0, 0
    
    # Retrieve model's internal structural spatial reference layout
    zone_mask = model.transformer.wpe.zone_mask
    
    for _ in range(sample_batches):
        X, Y = data_streamer.get_batch(batch_size=BATCH_SIZE, block_size=BLOCK_SIZE, device=device)
        logits, _ = model(X) # Shape: [B, T, VOCAB_SIZE]
        predictions = torch.argmax(logits, dim=-1) # Shape: [B, T]
        
        # Build evaluation masks matches the context window length
        T = X.size(1)
        current_zone_mask = zone_mask[:T].unsqueeze(0).expand(BATCH_SIZE, -1)
        
        # Zone 1 Filter (3-Character Token Fields)
        z1_indices = (current_zone_mask == 1)
        total_z1_correct += (predictions[z1_indices] == Y[z1_indices]).sum().item()
        total_z1_tokens += z1_indices.sum().item()
        
        # Zone 2 Filter (2-Character Token Fields)
        z2_indices = (current_zone_mask == 2)
        total_z2_correct += (predictions[z2_indices] == Y[z2_indices]).sum().item()
        total_z2_tokens += z2_indices.sum().item()
        
    model.train()
    
    z1_accuracy = (total_z1_correct / total_z1_tokens) if total_z1_tokens > 0 else 0.0
    z2_accuracy = (total_z2_correct / total_z2_tokens) if total_z2_tokens > 0 else 0.0
    return z1_accuracy, z2_accuracy

# =============================================================================
# 4. STREAMING CHUNK DATA GENERATOR (SIMULATION PLATFORM)




############


##########


while True:post_step += 1X, Y = data_streamer.get_batch(batch_size=BATCH_SIZE, block_size=BATCH_SIZE, device=device)logits, loss = model(X, Y)optimizer.zero_grad(set_to_none=True)loss.backward()torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)optimizer.step()current_loss = loss.item()if current_loss < (best_loss_this_chunk - 1e-4):print(f"    [PROGRESS] Post-Step {post_step} | Enhanced Model Weights Validated! Loss: {current_loss:.6f}")best_loss_this_chunk = current_lossno_improvement_counter = 0torch.save({'chunk_id': chunk_id,'final_post_step': post_step,'model_state_dict': model.state_dict(),'loss': best_loss_this_chunk,}, f"best_final_optimized_model_chunk_{chunk_id}.pt")else:no_improvement_counter += 1if post_step % 500 == 0:z1_acc, z2_acc = compute_strict_zone_accuracy(model, data_streamer, device)print(f"    [MONITOR] Post-Step {post_step} | Loss: {current_loss:.4f} | Zone 1 Acc: {z1_acc:.2%} | Zone 2 Acc: {z2_acc:.2%}")logger.log_step(chunk_id, post_step, "Phase2", current_loss, z1_acc, z2_acc)if no_improvement_counter >= PATIENCE_STEPS:print(f"\n[SATURATION REACHED] No further improvement found over {PATIENCE_STEPS} steps.")print(f"Optimized model saved securely as: 'best_final_optimized_model_chunk_{chunk_id}.pt'")print(f"Advancing state machine to next incoming streaming sector...")break
if name == 'main':
  execute_lifelong_training()



"""

---

### Functional Highlights of the Upgraded Components

#### CSV Logger Component
* **Output Path**: `phase1_training_metrics.csv`
* **Mechanics**: Every 1,000 steps during primary training, the system records performance data into this table. It appends the metrics dynamically, preserving data across restarts or transitions between streaming data chunks.

#### Strict Zone Accuracy Evaluator
* **Tracking Target**: Evaluates formatting accuracy independently for **Zone 1** (3-character tokens) and **Zone 2** (2-character tokens).
* **Implementation Details**: The model isolates target indices using its internal `zone_mask`. This allows it to verify tokens against your structural constraints, helping confirm that the attention layer respects geometric zone boundaries.

Let me know if you would like to adjust the logging frequency or include additional runtime metrics like gradient norm averages.
"""
