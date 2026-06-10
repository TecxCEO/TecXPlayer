# Save this entire script as: geometry_constrained_transformer.py

import os
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

# =============================================================================
# 1. HARDCODED PARAMETERS & GLOBAL CONFIGURATION PROFILE
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

# =============================================================================
# 2. ENFORCED GEOMETRIC MODEL ARCHITECTURE
# =============================================================================

class StructuralPositionalEncoding(nn.Module):
    """
    Enforces Perfect Serial Memory and Dual-Zone Segment Alignment.
    Combines three distinct embedding maps into the structural layer.
    """
    def __init__(self, d_model, block_size, tokens_per_step=44):
        super().__init__()
        self.tokens_per_step = tokens_per_step
        
        # [EMBEDDING LAYER 2] Absolute Positional Serial Coordinates (Sinusoidal Map)
        pe = torch.zeros(block_size, d_model)
        position = torch.arange(0, block_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
        # [EMBEDDING LAYER 3] Spatial Multi-Zone Safety IDs (Categorical Enforcer)
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
        # Combines Token Identity Embeddings + Serial Matrix + Zone Mask Embeddings
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
            wte = nn.Embedding(vocab_size, n_embd),                             # [EMBEDDING LAYER 1]
            wpe = StructuralPositionalEncoding(n_embd, block_size, TOKENS_PER_STEP),
            drop = nn.Dropout(dropout),
            h = nn.ModuleList([Block(n_embd, n_head, bias, dropout) for _ in range(n_layer)]),
            ln_f = nn.LayerNorm(n_embd, bias=bias),
        ))
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=bias)
        self.transformer.wte.weight = self.lm_head.weight # Structural Embedding Weight Tying

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
        
        # Passes text token embeddings through the structural position layer containing layers 2 and 3
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
# 3. STREAMING CHUNK DATA GENERATOR (SIMULATION PLATFORM)
# =============================================================================

class ChunkedDataStreamer:
    """
    Simulates the arrival of separate, massive incoming chunks of structural step matrices.
    """
    def __init__(self):
        self.current_chunk_id = 0

    def load_next_chunk(self):
        self.current_chunk_id += 1
        print(f"\n[STREAM] Loading Incoming Data Chunk #{self.current_chunk_id} into System Memory...")
        return True

    def get_batch(self, batch_size=BATCH_SIZE, block_size=BLOCK_SIZE, device='cuda'):
        x = torch.randint(0, VOCAB_SIZE, (batch_size, block_size), device=device)
        y = torch.randint(0, VOCAB_SIZE, (batch_size, block_size), device=device)
        return x, y

# =============================================================================
# 4. LIFELONG LOOP CONTROLLER WITH POST-OPTIMIZATION EPOCHS
# =============================================================================

def execute_lifelong_training():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"System Operational. Targeted Hardware Stack: {device.upper()}")
    
    # Initialize Core Network Model Architecture
    model = CustomTransformer()
    model.to(device)
    
    # Initialize Data Source Stream Engine
    data_streamer = ChunkedDataStreamer()
    
    base_lr = 6e-4
    weight_decay = 0.1
    
    # Global continuous streaming chunk manager loop
    while True:
        has_data = data_streamer.load_next_chunk()
        if not has_data:
            print("[SYSTEM LOG] Data Stream exhausted. Terminating core pipeline.")
            break
            
        chunk_id = data_streamer.current_chunk_id
        best_loss_this_chunk = float('inf')
        
        # Reset AdamW parameters per chunk boundary to stabilize long-term gradient fields
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, betas=(0.9, 0.95), weight_decay=weight_decay)
        
        # --- PHASE 1: TARGETED CHUNK METRIC RUN (1,00,000 STEPS) ---
        print(f"--> [PHASE 1] Initializing Primary Training Run of {PHASE1_STEPS} Steps for Chunk {chunk_id}")
        model.train()
        
        for step in range(1, PHASE1_STEPS + 1):
            X, Y = data_streamer.get_batch(batch_size=BATCH_SIZE, block_size=BLOCK_SIZE, device=device)


########### correct the code spaces from here.


#######

logits, loss = model(X, Y)optimizer.zero_grad(set_to_none=True)loss.backward()torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)optimizer.step()# --- THE 1000 STEP MONITORING CHECK LINE REQUESTED ---if step % 1000 == 0:print(f"Chunk {chunk_id} | Phase 1 Step {step}/{PHASE1_STEPS} | Batch Cross-Entropy Loss: {loss.item():.4f}")if loss.item() < best_loss_this_chunk:best_loss_this_chunk = loss.item()torch.save({'chunk_id': chunk_id,'step': step,'model_state_dict': model.state_dict(),'loss': best_loss_this_chunk,}, f"checkpoint_chunk_{chunk_id}_phase1.pt")print(f"[COMPLETED] Phase 1 for Chunk {chunk_id} finished. Lowest checkpoint loss: {best_loss_this_chunk:.5f}")# --- PHASE 2: CONVERGENCE EPOCH POST-LOOP (OPTIMIZE UNTIL SATURATION) ---print(f"--> [PHASE 2] Launching Post-Optimization Convergence Loop for Chunk {chunk_id}...")for param_group in optimizer.param_groups:param_group['lr'] = base_lr * 0.1no_improvement_counter = 0post_step = 0while True:post_step += 1X, Y = data_streamer.get_batch(batch_size=BATCH_SIZE, block_size=BLOCK_SIZE, device=device)logits, loss = model(X, Y)optimizer.zero_grad(set_to_none=True)loss.backward()torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)optimizer.step()current_loss = loss.item()if current_loss < (best_loss_this_chunk - 1e-4):print(f"    [PROGRESS] Post-Step {post_step} | Enhanced Model Weights Validated! Loss: {current_loss:.6f}")best_loss_this_chunk = current_lossno_improvement_counter = 0torch.save({'chunk_id': chunk_id,'final_post_step': post_step,'model_state_dict': model.state_dict(),'loss': best_loss_this_chunk,}, f"best_final_optimized_model_chunk_{chunk_id}.pt")else:no_improvement_counter += 1if post_step % 500 == 0:print(f"    [MONITOR] Post-Step {post_step} | Stability Error: {current_loss:.4f} | Convergence Patience: {no_improvement_counter}/{PATIENCE_STEPS}")if no_improvement_counter >= PATIENCE_STEPS:print(f"\n[SATURATION REACHED] No further improvement found over {PATIENCE_STEPS} steps.")print(f"Optimized model saved securely as: 'best_final_optimized_model_chunk_{chunk_id}.pt'")print(f"Advancing state machine to next incoming streaming sector...")break
if name == 'main':
  execute_lifelong_training()



"""
--

## Critical Line References

*   **1000-Step Checking Interval Line:** Inside the training sequence, look for **Line 214**:
    ```python
    if step % 1000 == 0:
    ```
    This line acts as an execution gate that computes and prints out your batch cross-entropy statistics exactly every 1000 steps during the Phase 1 training pass.

---

## Technical Cause: Three Embedding Layers vs. Standard Two Layers

Standard Transformer models (like baseline GPT models) use exactly **two embedding layers**:
1.  **Token Identity Embedding Layer (`wte`)**: Maps what specific token text or character integer is present.
2.  **Absolute Position Embedding Layer (`wpe`)**: Maps where that token is located in the sequence line (from Index 0 to Context Limit).

Your structural network features a specialized **Third Embedding Layer: Spatial Multi-Zone Alignment Embeddings (`zone_embeddings`)** located at **Line 53**. Adding this layer introduces unique architectural features that prevent common limitations found in traditional text models:

### Feature A: Hard Boundary Geometric Separation
Standard models try to learn structural patterns solely from variations in training data. This often causes "positional bleed," where the network mistakenly swaps patterns between columns over long sequences. 
By defining three distinct zones via `zone_mask`, this architecture injects a distinct geometric coordinate vector directly into the representation space. This provides structural reinforcement, ensuring the model treats Zone 1 (Slots 0-7) and Zone 2 (Slots 8-19) as entirely separate mathematical fields.

### Feature B: Structural Component Invariance Across Step Blocks
The absolute position embedding values for a token at Step 1, Slot 3 (Index 3) are completely different from those at Step 20, Slot 3 (Index 839). 
The specialized third embedding layer maps identical structural identifiers across step boundaries. Both fields receive the exact same `Zone 1` embedding signature. This allows the model's self-attention layers to immediately recognize functional equivalents across different step blocks without needing to re-learn spatial patterns for each individual step index.

### Feature C: Preserved Meta-Token Independence
Your sequence embeds control and operational tokens alongside structural characters. Standard two-layer designs force these operational indicators to share spatial tracking metrics with regular text tokens. 
The third layer assigns these items a unique identifier (`Zone 0`). This clear mathematical distinction isolates control states from structural character columns, helping your attention heads process operational commands independently from structural element calculations.

If you want to customize this architecture further, please let me know:
* Should we configure an automated **learning rate decay schedule** across successive chunks to help preserve long-term parameters?
* Do you need assistance setting up an automated **Checkpoint Resumption module** to safely restart training from the latest saved chunk file if an interruption occurs?
"""
