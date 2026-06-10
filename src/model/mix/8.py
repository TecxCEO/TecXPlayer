"""

Here is the complete, high-efficiency PyTorch script that integrates your explicit token layout mapping, structural dual-zone geometric enforcement, step positional tracking, and the automated lifelong multi-chunk training pipeline.Unified Production Script

"""

# Save this entire script as: geometry_constrained_transformer.py

import os
import csv
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

# =============================================================================
# 1. PARAMETERS & EXACT EXPLICIT VOCABULARY PROFILE
# =============================================================================
BASE_VOCAB_SIZE = 90      # Base structural tokens (0 to 89)

# Register Explicit Task and Structural Meta-Tokens
SOS_TOKEN = BASE_VOCAB_SIZE      # 90
EOS_TOKEN = BASE_VOCAB_SIZE + 1  # 91 (Used as US/End token)
TASK_FWD  = BASE_VOCAB_SIZE + 2  # 92
TASK_REV  = BASE_VOCAB_SIZE + 3  # 93
TASK_SOLV = BASE_VOCAB_SIZE + 4  # 94

ACTUAL_VOCAB_SIZE = BASE_VOCAB_SIZE + 5  # Exactly 95 total unique tokens

# Sequence Window Constraints
STEPS_PER_SEQ = 20       # 20 stages per matrix sequence
TOKENS_PER_STEP = 44     # Each step maps exactly 44 internal elements
BLOCK_SIZE = 880         # Context window matrix size: 20 * 44 = 880

# Sub-Step Structural Segment Boundaries
INPUT_STAGE_TOKENS = 20  # First 20 slots: 3-character & 2-character source elements
OUTPUT_STAGE_TOKENS = 20 # Next 20 slots: Target stage transformations
CONTROL_TOKENS = 4       # Final 4 slots: Meta commands (SOS, EOS, Task, Move)

# Lifelong Training Hyperparameters
BATCH_SIZE = 18          # Process exactly 18 parallel sequence streams per batch
PHASE1_STEPS = 100000    # Initial primary run limits per incoming chunk (1 Lakh steps)
PATIENCE_STEPS = 2000    # Target patience window for post-optimization convergence

CSV_LOG_FILE = "phase1_training_metrics.csv"

# =============================================================================
# 2. ENFORCED GEOMETRIC MODEL ARCHITECTURE
# =============================================================================

class StructuralPositionalEncoding(nn.Module):
    """
    Enforces Perfect Serial Memory and Step-Zone Position Invariance.
    Coordinates indices and maps boundaries to prevent slot pollution.
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
    """
    Multi-head attention mechanism leveraging a 3x embedding projection mapping.
    """
    def __init__(self, n_embd, n_head, bias, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias) # Projects Q, K, V simultaneously
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
        self.transformer.wte.weight = self.lm_head.weight # Structural Weight Tying

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
        assert t <= self.block_size, f"Sequence segment length {t} exceeds window context limit {self.block_size}"
        
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
                writer.writerow(['chunk_id', 'step', 'phase', 'loss', 'input_zone_acc', 'output_zone_acc'])

    def log_step(self, chunk_id, step, phase, loss, input_acc=0.0, output_acc=0.0):
        with open(self.filepath, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([chunk_id, step, phase, f"{loss:.6f}", f"{input_acc:.4f}", f"{output_acc:.4f}"])


@torch.no_grad()
def compute_structural_zone_accuracy(model, data_streamer, device, sample_batches=4):
    model.eval()
    total_in_correct, total_in_tokens = 0, 0
    total_out_correct, total_out_tokens = 0, 0
    
    zone_mask = model.transformer.wpe.zone_mask
    
    for _ in range(sample_batches):
        X, Y = data_streamer.get_batch(batch_size=BATCH_SIZE, block_size=BLOCK_SIZE, device=device)
        logits, _ = model(X)
        predictions = torch.argmax(logits, dim=-1)
        
        T = X.size(1)
        current_zone_mask = zone_mask[:T].unsqueeze(0).expand(BATCH_SIZE, -1)
        
        # Verify prediction profiles across Zone 1 (Input Segments)
        in_indices = (current_zone_mask == 1)
        total_in_correct += (predictions[in_indices] == Y[in_indices]).sum().item()
        total_in_tokens += in_indices.sum().item()
        
        # Verify prediction profiles across Zone 2 (Output Segments)
        out_indices = (current_zone_mask == 2)
        total_out_correct += (predictions[out_indices] == Y[out_indices]).sum().item()
        total_out_tokens += out_indices.sum().item()
        
    model.train()
    
    input_accuracy = (total_in_correct / total_in_tokens) if total_in_tokens > 0 else 0.0
    output_accuracy = (total_out_correct / total_out_tokens) if total_out_tokens > 0 else 0.0
    return input_accuracy, output_accuracy

# =============================================================================
# 4. DATA ENGINE (STREAMING GENERATOR INTERFACE)


######

######


}, f"checkpoint_chunk_{chunk_id}_phase1.pt")print(f"[COMPLETED] Chunk {chunk_id} Phase 1 run completed. Minimum loss floor achieved: {best_loss_this_chunk:.5f}")# --- PHASE 2: CONVERGENCE EPOCH LOOP (OPTIMIZE UNTIL SATURATION) ---print(f"--> [PHASE 2] Launching Fine-Tuning Convergence Loop for Chunk {chunk_id}...")# Scale back the global learning rate by 10x to smoothly optimize parametersfor param_group in optimizer.param_groups:param_group['lr'] = base_lr * 0.1no_improvement_counter = 0post_step = 0while True:post_step += 1X, Y = data_streamer.get_batch(batch_size=BATCH_SIZE, block_size=BLOCK_SIZE, device=device)logits, loss = model(X, Y)optimizer.zero_grad(set_to_none=True)loss.backward()torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)optimizer.step()current_loss = loss.item()# Enforce validation check against the tracking thresholdif current_loss < (best_loss_this_chunk - 1e-4):print(f"    [PROGRESS] Post-Step {post_step} | Enhanced Weights Saved! Target Loss: {current_loss:.6f}")best_loss_this_chunk = current_lossno_improvement_counter = 0# Overwrite and preserve the absolute best possible state for this chunktorch.save({'chunk_id': chunk_id,'final_post_step': post_step,'model_state_dict': model.state_dict(),'loss': best_loss_this_chunk,}, f"best_final_optimized_model_chunk_{chunk_id}.pt")else:no_improvement_counter += 1if post_step % 500 == 0:in_acc, out_acc = compute_structural_zone_accuracy(model, data_streamer, device)print(f"    [POST-MONITOR] Step {post_step} | Stability Loss: {current_loss:.4f} | Input Zone Acc: {in_acc:.2%} | Output Zone Acc: {out_acc:.2%}")logger.log_step(chunk_id, post_step, "Phase2", current_loss, in_acc, out_acc)# If saturation patience is exhausted, break out of loop and load the next data chunkif no_improvement_counter >= PATIENCE_STEPS:print(f"\n[SATURATION REACHED] Saturated at floor {best_loss_this_chunk:.6f} with no improvements for {PATIENCE_STEPS} steps.")print(f"Chunk {chunk_id} finalized. State preserved. Moving to next segment space...")print("================================================================================")breakif name == 'main':execute_lifelong_training()


"""
--

### Core Functional Features Addressed

1. **Explicit Token Index Architecture (Lines 14–23)**: Your specific token tracking offsets are hardcoded cleanly as core constants. The system allocates baseline structural components from indices `0 to 89` inside `BASE_VOCAB_SIZE`, mapping the meta configurations up to an absolute `ACTUAL_VOCAB_SIZE` of **95 total unique tokens**.
2. **Dynamic 44-Token Multi-Zone Setup (Lines 53–59)**: The positional mask inside `StructuralPositionalEncoding` is updated to implement your layout precisely. It allocates structural tags sequentially across the vector grid, keeping Input, Output, and Control commands isolated.
3. **Multi-Chunk Persistence Mechanism**: When a data chunk finishes both its initial 1 Lakh steps and its fine-tuning phase, **the weights remain in your GPU's memory**. The loop resets the optimizer variables to stabilize learning, then immediately begins training on the next data chunk, allowing the model to adapt continuously over long runs.

If you would like to customize this code further, please share:
* Do you need an automated **Checkpoint Resumption Module** added to `main` so the script can safely restart from the last saved chunk file after a sudden power loss?
* Would you like to introduce **Cosine Annealing Learning Rate Schedules** to systematically decay the step learning rate during Phase 1?
"""
s
