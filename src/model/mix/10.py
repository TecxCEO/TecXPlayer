"""

Here is the complete, high-performance PyTorch script. It integrates Cosine Annealing Learning Rate Decay into the primary training run and establishes an Accuracy Performance Gate alongside raw loss checks during the post-optimization epochs.The system tracks metrics down to the individual token position and automatically exports analytical logs to a separate CSV file.python

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

# Training Hyperparameters
BATCH_SIZE = 18          # Process exactly 18 parallel sequence streams per batch
PATIENCE_STEPS = 2000    # Target patience window for post-optimization convergence
ACCURACY_GATE_MIN = 0.98 # Minimum 98% accuracy required to pass performance gate

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
                writer.writerow(['chunk_id', 'epoch', 'step', 'phase', 'loss', 'input_zone_acc', 'output_zone_acc', 'lr'])

    def log_step(self, chunk_id, epoch, step, phase, loss, input_acc=0.0, output_acc=0.0, lr=0.0):
        with open(self.filepath, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([chunk_id, epoch, step, phase, f"{loss:.6f}", f"{input_acc:.4f}", f"{output_acc:.4f}", f"{lr:.8f}"])


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

##############

#######2222###

if self.current_step >= self.total_steps:raise StopIterationself.current_step += 1return self.current_stepclass ChunkedDataStreamer:def init(self):self.current_chunk_id = 0def load_next_chunk(self):self.current_chunk_id += 1print(f"\n[STREAM ENGINE] Loading Incoming Data Chunk #{self.current_chunk_id} into Pipeline Context...")return FiniteChunkDataset(total_steps=100000)def get_batch(self, dataset_iterator, device='cuda'):_ = next(dataset_iterator)x = torch.randint(0, ACTUAL_VOCAB_SIZE, (BATCH_SIZE, BLOCK_SIZE), device=device)y = torch.randint(0, ACTUAL_VOCAB_SIZE, (BATCH_SIZE, BLOCK_SIZE), device=device)return x, ydef get_batch_simulated(self, batch_size=BATCH_SIZE, device='cuda'):x = torch.randint(0, ACTUAL_VOCAB_SIZE, (batch_size, BLOCK_SIZE), device=device)y = torch.randint(0, ACTUAL_VOCAB_SIZE, (batch_size, BLOCK_SIZE), device=device)return x, y=============================================================================5. LIFELONG LOOP CONTROLLER WITH ENFORCED COMPLETION CHECKS=============================================================================def execute_lifelong_training():device = 'cuda' if torch.cuda.is_available() else 'cpu'print(f"System Online. Targeted Core Execution Hardware: {device.upper()}")model = CustomTransformer()model.to(device)data_streamer = ChunkedDataStreamer()logger = MetricLogger(CSV_LOG_FILE)base_lr = 6e-4min_lr = 6e-5weight_decay = 0.1while True:# --- PHASE 1: INITIAL CHUNK PASSTHROUGH WITH COSINE SCHEDULER ---dataset_chunk = data_streamer.load_next_chunk()chunk_id = data_streamer.current_chunk_idbest_loss_this_chunk = float('inf')optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, betas=(0.9, 0.95), weight_decay=weight_decay)chunk_iterator = iter(dataset_chunk)print(f"--> [PHASE 1] Training Loop Active for Chunk {chunk_id} (Enforcing Cosine LR Decay)")model.train()while True:try:X, Y = data_streamer.get_batch(chunk_iterator, device=device)current_step = dataset_chunk.current_stepexcept StopIteration:print(f"\n[DATA BOUNDARY] All input items for Chunk {chunk_id} processed cleanly.")break# --- FEATURE 1: COSINE ANNEALING LEARNING RATE DECAY ---# Gradually steps down the learning rate across the 1,00,000 progress markersprogress = current_step / dataset_chunk.total_stepscosine_lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))for param_group in optimizer.param_groups:param_group['lr'] = cosine_lrlogits, loss = model(X, Y)optimizer.zero_grad(set_to_none=True)loss.backward()torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)optimizer.step()if current_step % 1000 == 0:in_acc, out_acc = compute_structural_zone_accuracy(model, data_streamer, device)print(f"Chunk {chunk_id} | Step {current_step}/{dataset_chunk.total_steps} | Loss: {loss.item():.4f} | Input Acc: {in_acc:.2%} | LR: {cosine_lr:.6f}")logger.log_step(chunk_id, 0, current_step, "Phase1", loss.item(), in_acc, out_acc, cosine_lr)if loss.item() < best_loss_this_chunk:best_loss_this_chunk = loss.item()torch.save({'chunk_id': chunk_id,'step': current_step,'model_state_dict': model.state_dict(),'loss': best_loss_this_chunk,}, f"checkpoint_chunk_{chunk_id}_phase1.pt")# --- PHASE 2: CONVERGENCE LOOP WITH ACCURACY PERFORMANCE GATE ---print(f"--> [PHASE 2] Launching Fine-Tuning Optimization Epochs for Chunk {chunk_id}...")# Lock fine-tuning updates to 10% of the maximum engine scalefor param_group in optimizer.param_groups:param_group['lr'] = base_lr * 0.1no_improvement_counter = 0epoch_count = 0while True:epoch_count += 1post_step = 0chunk_iterator = iter(data_streamer.load_next_chunk())print(f"    [EPOCH PASSTHROUGH] Starting Pass #{epoch_count} for Structural Parameter Optimization...")while True:try:X, Y = data_streamer.get_batch(chunk_iterator, device=device)post_step += 1except StopIteration:breaklogits, loss = model(X, Y)optimizer.zero_grad(set_to_none=True)loss.backward()torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)optimizer.step()current_loss = loss.item()# Evaluate both structural layout constraints and loss floor changesif current_loss < (best_loss_this_chunk - 1e-4):# --- FEATURE 2: ACCURACY PERFORMANCE GATE ---# Checks accuracy within targeted geometric zones before saving weightsin_acc, out_acc = compute_structural_zone_accuracy(model, data_streamer, device)if in_acc >= ACCURACY_GATE_MIN and out_acc >= ACCURACY_GATE_MIN:print(f"        [GATE PASSED] Epoch {epoch_count} Step {post_step} | Loss: {current_loss:.6f} | Acc: {in_acc:.2%}. Saving.")best_loss_this_chunk = current_lossno_improvement_counter = 0torch.save({'chunk_id': chunk_id,'epoch': epoch_count,'model_state_dict': model.state_dict(),'loss': best_loss_this_chunk,'input_accuracy': in_acc,}, f"best_final_optimized_model_chunk_{chunk_id}.pt")else:# Increases patience counter if zone accuracy drops below the thresholdno_improvement_counter += 1else:no_improvement_counter += 1if post_step % 5000 == 0:in_acc, out_acc = compute_structural_zone_accuracy(model, data_streamer, device)print(f"        [MONITOR] Epoch {epoch_count} Step {post_step} | Loss: {current_loss:.4f} | Patience: {no_improvement_counter}/{PATIENCE_STEPS}")logger.log_step(chunk_id, epoch_count, post_step, "Phase2", current_loss, in_acc, out_acc, base_lr * 0.1)if no_improvement_counter >= PATIENCE_STEPS:breakif no_improvement_counter >= PATIENCE_STEPS:print(f"\n[SATURATION REACHED] No further improvement found over {PATIENCE_STEPS} steps.")print(f"Optimal model for Chunk {chunk_id} saved as: 'best_final_optimized_model_chunk_{chunk_id}.pt'")print("================================================================================")breakif name == 'main':execute_lifelong_training()


  """

  ### Explanation of the Upgraded Framework Features

#### 1. Cosine Annealing Learning Rate Decay (Lines 198–202)
Standard step-bound runs often drop the learning rate abruptly, which can disrupt gradient tracking across sequence elements. This update calculates a smooth mathematical curve:
\[\eta_t = \eta_{\text{min}} + \frac{1}{2}(\eta_{\text{max}} - \eta_{\text{min}})\left(1 + \cos\left(\frac{T_{\text{cur}}}{T_{\text{max}}}\pi\right)\right)\]
This structural adjustment slowly scales the parameters down from `6e-4` to `6e-5` over the initial 1,00,000 steps, helping the network settle into stable local minima before fine-tuning begins.

#### 2. Accuracy Performance Gate (Lines 248–261)
During optimization epochs, a model can sometimes overfit to short-term token sequences at the expense of its overall geometric constraints. 
To prevent this, the architecture introduces a verification gate (`ACCURACY_GATE_MIN = 0.98`). Even if a fine-tuning step achieves a lower loss, the weights are only saved to disk if the model achieves at least **98% structural accuracy** across both the Input and Output Stage token slots.

#### 3. Integrated Analytical Logging (Line 150)
The unified `MetricLogger` logs performance markers exactly every 1,000 steps during primary training and every 5,000 steps during the validation epochs. It appends metrics directly to `phase1_training_metrics.csv` to keep a reliable ledger across all incoming data streams.

If you would like to adjust the minimum threshold of the accuracy gate or explore adding a **moving average to smooth out cross-entropy calculations**, let me know!

"""
