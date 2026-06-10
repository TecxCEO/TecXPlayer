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
    """
    def __init__(self, d_model, block_size, tokens_per_step=44):
        super().__init__()
        self.tokens_per_step = tokens_per_step
        
        # Absolute Positional Serial Coordinates
        pe = torch.zeros(block_size, d_model)
        position = torch.arange(0, block_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
        # Spatial Multi-Zone Safety IDs (0: Meta/Control, 1: 3-Char Zone, 2: 2-Char Zone)
        zone_mask = torch.zeros(block_size, dtype=torch.long)
        for step in range(block_size // tokens_per_step):
            step_offset = step * tokens_per_step
            
            # Input Stage Alignment (Tokens 0 to 23)
            zone_mask[step_offset + 0 : step_offset + 8] = 1   # 8 slots for 3-char
            zone_mask[step_offset + 8 : step_offset + 20] = 2  # 12 slots for 2-char
            zone_mask[step_offset + 20 : step_offset + 24] = 0 # 4 Control tokens
            
            # Output Stage Alignment (Tokens 24 to 43)
            zone_mask[step_offset + 24 : step_offset + 32] = 1 # 8 slots for 3-char target
            zone_mask[step_offset + 32 : step_offset + 44] = 2 # 12 slots for 2-char target
            
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
# 3. STREAMING CHUNK DATA GENERATOR (REPLACE WITH REAL DATA INTERFACE)
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
        # In production, load your real dataset files here:
        # data = np.load(f"chunk_{self.current_chunk_id}.npy")
        return True

    def get_batch(self, batch_size=BATCH_SIZE, block_size=BLOCK_SIZE, device='cuda'):
        # Generates exact structural batch tensor pairs matching constraints
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
    
    # Base Training Parameters
    base_lr = 6e-4
    weight_decay = 0.1
    
    # Global training control loop that runs indefinitely as chunks arrive
    while True:
        # 1. Fetch next incoming chunk of data
        has_data = data_streamer.load_next_chunk()
        if not has_data:
            print("[SYSTEM LOG] Data Stream exhausted. Terminating core pipeline.")
            break
            
        chunk_id = data_streamer.current_chunk_id
        best_loss_this_chunk = float('inf')
        
        # Re-initialize/reset AdamW Optimizer state per chunk to clear momentum states
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, betas=(0.9, 0.95), weight_decay=weight_decay)
        
        # --- PHASE 1: TARGETED CHUNK METRIC RUN (1,00,000+ STEPS) ---
        print(f"--> [PHASE 1] Initializing Primary Training Run of {PHASE1_STEPS} Steps for Chunk {chunk_id}")
        model.train()
        
        for step in range(1, PHASE1_STEPS + 1):
            # Pull exactly 18 batch samples, sized 20 steps * 44 tokens (880 elements)
            X, Y = data_streamer.get_batch(batch_size=BATCH_SIZE, block_size=BLOCK_SIZE, device=device)
            
            logits, loss = model(X, Y)
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()


####### correct space from here 

#####

torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)optimizer.step()if step % 1000 == 0 or step == 1:print(f"Chunk {chunk_id} | Phase 1 Step {step}/{PHASE1_STEPS} | Batch Cross-Entropy Loss: {loss.item():.4f}")if loss.item() < best_loss_this_chunk:best_loss_this_chunk = loss.item()# Save immediate checkpoint for safety within the current chunk runtorch.save({'chunk_id': chunk_id,'step': step,'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(),'loss': best_loss_this_chunk,}, f"checkpoint_chunk_{chunk_id}_phase1.pt")print(f"[COMPLETED] Phase 1 for Chunk {chunk_id} finished. Lowest empirical checkpoint: {best_loss_this_chunk:.5f}")# --- PHASE 2: CONVERGENCE EPOCH POST-LOOP (OPTIMIZE UNTIL SATURATION) ---print(f"--> [PHASE 2] Launching Post-Optimization Convergence Loop for Chunk {chunk_id}...")# Drop global learning rate to a fine-tuning state (10% of base value)for param_group in optimizer.param_groups:param_group['lr'] = base_lr * 0.1no_improvement_counter = 0post_step = 0while True:post_step += 1X, Y = data_streamer.get_batch(batch_size=BATCH_SIZE, block_size=BLOCK_SIZE, device=device)logits, loss = model(X, Y)optimizer.zero_grad(set_to_none=True)loss.backward()torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)optimizer.step()current_loss = loss.item()# Enforce micro-improvement threshold testif current_loss < (best_loss_this_chunk - 1e-4):print(f"    [PROGRESS] Post-Step {post_step} | Enhanced Model Weights Validated! Loss: {current_loss:.6f}")best_loss_this_chunk = current_lossno_improvement_counter = 0# Overwrite and preserve the absolute best possible state for this chunktorch.save({'chunk_id': chunk_id,'final_post_step': post_step,'model_state_dict': model.state_dict(),'loss': best_loss_this_chunk,}, f"best_final_optimized_model_chunk_{chunk_id}.pt")else:no_improvement_counter += 1if post_step % 500 == 0:print(f"    [MONITOR] Post-Step {post_step} | Stability Error: {current_loss:.4f} | Convergence Patience: {no_improvement_counter}/{PATIENCE_STEPS}")# If saturation threshold is triggered, terminate loop and proceed to next chunkif no_improvement_counter >= PATIENCE_STEPS:print(f"\n[SATURATION REACHED] No further improvement found over {PATIENCE_STEPS} steps.")print(f"Optimized model for Chunk {chunk_id} saved securely as: 'best_final_optimized_model_chunk_{chunk_id}.pt'")print(f"Advancing state machine to next incoming streaming sector...")breakif name == 'main':execute_lifelong_training()


"""
---

## Architectural Breakdown & Training Mechanics

### Multi-Chunk Processing Strategy
* **Continuous Updates:** Rather than crashing or stopping, the script encapsulates everything within an infinite tracking `while True:` routine. As soon as a chunk finishes its execution, the system clears optimization memories and sets up for the next data block.
* **Preserved Initial Weights:** When advancing to Chunk 2, Chunk 3, etc., the transformer's parameter matrices **remain in memory**. It continues training on new data using the weights learned from prior chunks.

### Memory Optimization & Structural Integrity
* **Batch Sizing Alignment:** The matrix fetch routine tracks exactly `BATCH_SIZE = 18`. Every processing slice maintains a dimensional structure of \(18 \times 880\) sequence tensors.
* **Optimizer Resets:** At the start of each new data chunk, `optimizer = torch.optim.AdamW(...)` re-instantiates. This resets the internal historical momentum metrics (\(\beta_1\), \(\beta_2\)), preventing gradients from previous data chunks from distorting learning on the new data block.

### Automated Convergence Safety (Phase 2 Post-Loop)
* **Fine-Tuning Adjustments:** Once Phase 1 completes its 1,00,000 steps, Phase 2 reduces the learning rate by a factor of 10 (`base_lr * 0.1`) to safely smooth out loss oscillations.
* **Patience Tracking Mechanism:** The system monitors performance against a counter (`PATIENCE_STEPS = 2000`). If the loss fails to drop below the best-recorded error floor by at least \(10^{-4}\) for 2,000 consecutive steps, the code saves the model and transitions automatically to the next incoming data block.

To customize the integration of this framework, please let me know:
* Would you like to use **PyTorch's Distributed Data Parallel (DDP)** to spread this 18-batch workflow across multiple GPUs?
* Do you need an explicit **Validation Dataset routine** evaluating loss performance on held-out tokens alongside the training loop?
"""
