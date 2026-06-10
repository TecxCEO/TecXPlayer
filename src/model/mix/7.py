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
BLOCK_SIZE = 880         # Context Window: 20 steps * 44 tokens per step = 880
STEPS_PER_SEQ = 20       # Sequences contain exactly 20 internal processing steps
TOKENS_PER_STEP = 44     # Every individual processing step contains 44 tokens

# Upgraded Step Block Alignment Metrics (44 Tokens Total per Step)
INPUT_STAGE_TOKENS = 20  # 20 tokens reserved for Input Stage
OUTPUT_STAGE_TOKENS = 20 # 20 tokens reserved for Output Stage
CONTROL_TOKENS = 4       # 4 tokens for SOS, US, Move Code, Task Codes

# Training Hyperparameters
BATCH_SIZE = 18          # Explicitly set to 18 sequences per batch
PHASE1_STEPS = 100000    # Primary training run steps per data chunk (1 Lakh steps)
PATIENCE_STEPS = 2000    # Post-optimization saturation window limit

# File System Configurations
CSV_LOG_FILE = "phase1_training_metrics.csv"

# =============================================================================
# 2. ENFORCED GEOMETRIC MODEL ARCHITECTURE
# =============================================================================

class StructuralPositionalEncoding(nn.Module):
    """
    Enforces Perfect Serial Memory and New Tri-Zone Segment Alignment.
    """
    def __init__(self, d_model, block_size, tokens_per_step=44):
        super().__init__()
        self.tokens_per_step = tokens_per_step
        
        # Absolute Positional Serial Coordinates (Sinusoidal Map: 0 to 879)
        pe = torch.zeros(block_size, d_model)
        position = torch.arange(0, block_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
        # Spatial Multi-Zone Safety IDs (Categorical Enforcer)
        # Zone 0: Control Space, Zone 1: Input Stage Space, Zone 2: Output Stage Space
        zone_mask = torch.zeros(block_size, dtype=torch.long)
        for step in range(block_size // tokens_per_step):
            step_offset = step * tokens_per_step
            
            # 1. Input Stage Layer (First 20 Tokens)
            zone_mask[step_offset + 0 : step_offset + 20] = 1
            
            # 2. Output Stage Layer (Next 20 Tokens)
            zone_mask[step_offset + 20 : step_offset + 40] = 2
            
            # 3. Control Command Layer (Last 4 Tokens: SOS, US, Move, Task Codes)
            zone_mask[step_offset + 40 : step_offset + 44] = 0
            
        self.zone_embeddings = nn.Embedding(3, d_model) # Resolves into Zone 0, 1, or 2 vectors
        self.register_buffer('zone_mask', zone_mask)

    def forward(self, x):
        T = x.size(1)
        return x + self.pe[:, :T, :] + self.zone_embeddings(self.zone_mask[:T])


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, bias, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        # Expanded 3x projection matrix to process Q, K, and V values simultaneously
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
# 3. AUTOMATED ANALYTICS LOGGER & STRUCTURAL EVALUATOR
# =============================================================================

class MetricLogger:
    """
    Handles CSV logging for Phase 1 optimization tracking.
    """
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
    """
    Validates formatting accuracy across the Input and Output Stage token slots.
    """
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
        
        # Filter for Input Stage tokens (Zone 1)
        in_indices = (current_zone_mask == 1)
        total_in_correct += (predictions[in_indices] == Y[in_indices]).sum().item()
        total_in_tokens += in_indices.sum().item()
        
        # Filter for Output Stage tokens (Zone 2)
        out_indices = (current_zone_mask == 2)
        total_out_correct += (predictions[out_indices] == Y[out_indices]).sum().item()
        total_out_tokens += out_indices.sum().item()
        
    model.train()
    
    input_accuracy = (total_in_correct / total_in_tokens) if total_in_tokens > 0 else 0.0
    output_accuracy = (total_out_correct / total_out_tokens) if total_out_tokens > 0 else 0.0
    return input_accuracy, output_accuracy

# =============================================================================
# 4. DATA ENGINE (STREAMING GENERATOR INTERFACE)
# =============================================================================

class ChunkedDataStreamer:
    """
    Simulates loading consecutive data chunks with 18 batches of 20 steps * 44 tokens.
    """
    def __init__(self):
        self.current_chunk_id = 0



########

#######
print(f"[COMPLETED] Chunk {chunk_id} Phase 1 done. Minimum loss reached: {best_loss_this_chunk:.5f}")# --- PHASE 2: CONVERGENCE EPOCH POST-LOOP (OPTIMIZE UNTIL SATURATION) ---print(f"--> [PHASE 2] Launching Convergence Loop for Chunk {chunk_id} (Patience: {PATIENCE_STEPS} steps)")# Lower learning rate by 10x to optimize fine-grained parametersfor param_group in optimizer.param_groups:param_group['lr'] = base_lr * 0.1no_improvement_counter = 0post_step = 0while True:post_step += 1X, Y = data_streamer.get_batch(batch_size=BATCH_SIZE, block_size=BLOCK_SIZE, device=device)logits, loss = model(X, Y)optimizer.zero_grad(set_to_none=True)loss.backward()torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)optimizer.step()current_loss = loss.item()# Verify improvement against the micro-thresholdif current_loss < (best_loss_this_chunk - 1e-4):print(f"    [SAVING TARGET] Post-Step {post_step} | Lower Loss Validated: {current_loss:.6f}")best_loss_this_chunk = current_lossno_improvement_counter = 0# Overwrites file with the absolute best possible state for this chunktorch.save({'chunk_id': chunk_id,'final_post_step': post_step,'model_state_dict': model.state_dict(),'loss': best_loss_this_chunk,}, f"best_final_optimized_model_chunk_{chunk_id}.pt")else:no_improvement_counter += 1if post_step % 500 == 0:in_acc, out_acc = compute_structural_zone_accuracy(model, data_streamer, device)print(f"    [POST-MONITOR] Step {post_step} | Loss: {current_loss:.4f} | Input Acc: {in_acc:.2%} | Output Acc: {out_acc:.2%}")logger.log_step(chunk_id, post_step, "Phase2", current_loss, in_acc, out_acc)# Exit loop once performance saturatesif no_improvement_counter >= PATIENCE_STEPS:print(f"\n[SATURATION DETECTED] No improvements over {PATIENCE_STEPS} steps. Freezing state updates.")print(f"Chunk {chunk_id} optimization complete. Model state saved.")print("--------------------------------------------------------------------------------")breakif name == 'main':execute_lifelong_training()


"""
Model state saved.")print("--------------------------------------------------------------------------------")breakif name == 'main':execute_lifelong_training()
### Key Structural Adaptations
*   **Upgraded Spatial Encoding Rules (Lines 47-53)**: In the positional encoding initialization, the tokens are cleanly divided into your new structural layout. Indices 0–19 are designated as `Zone 1` (Input Stage), indices 20–39 are designated as `Zone 2` (Output Stage), and indices 40–43 are designated as `Zone 0` (Control Tokens).
*   **Targeted Validation Metrics (Lines 151-164)**: The validation accuracy calculations map cleanly onto this updated layout, tracking how well the transformer handles token distributions across your designated zones.

Feel free to ask if you would like to implement **gradient accumulation steps** to simulate larger effective batch sizes or configure an automated **email/webhook alert component** for remote step tracking.
"""
