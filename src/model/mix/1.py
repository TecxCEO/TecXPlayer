import os
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

# =============================================================================
# 1. ARCHITECTURE DEFINITION
# =============================================================================

class CausalSelfAttention(nn.Module):
    """
    Multi-head masked self-attention mechanism with projection dropout.
    """
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
            q, k, v, 
            attn_mask=None, 
            dropout_p=self.attn_dropout.p if self.training else 0.0, 
            is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    """
    GELU multi-layer perceptron with projection dropout.
    """
    def __init__(self, n_embd, bias, dropout):
        super().__init__()
        self.c_fc    = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    """
    Standard Transformer block utilizing Pre-LayerNorm.
    """
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
    """
    GPT-style autoregressive model optimized for the structured 880-token block size.
    """
    def __init__(self, vocab_size=95, block_size=880, n_layer=12, n_head=12, n_embd=768, dropout=0.1, bias=False):
        super().__init__()
        self.block_size = block_size

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, n_embd),
            wpe = nn.Embedding(block_size, n_embd),
            drop = nn.Dropout(dropout),
            h = nn.ModuleList([Block(n_embd, n_head, bias, dropout) for _ in range(n_layer)]),
            ln_f = nn.LayerNorm(n_embd, bias=bias),
        ))
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=bias)
        self.transformer.wte.weight = self.lm_head.weight # Weight tying

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
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        
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
# 2. TRAINING LOOP & CONTINUOUS LOOP HOOKS
# =============================================================================

def get_batch(block_size=880, batch_size=16, device='cuda'):
    """
    Placeholder generator for structural data framework matching constraints.
    Replace this with your real dataset iterator.
    """
    x = torch.randint(0, 95, (batch_size, block_size), device=device)
    y = torch.randint(0, 95, (batch_size, block_size), device=device)
    return x, y

def train_model():
    # Structural Token System Settings
    VOCAB_SIZE = 95
    BLOCK_SIZE = 880 # 20 steps * 44 tokens per step
    MAX_STEPS = 50000 # Configurable up to 100,000 steps
    
    # Execution Hyperparameters
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 16
    learning_rate = 6e-4
    weight_decay = 1e-1
    beta1, beta2 = 0.9, 0.95
    best_loss = float('inf')
    
    print(f"Initializing model framework on: {device}")
    model = CustomTransformer(vocab_size=VOCAB_SIZE, block_size=BLOCK_SIZE)
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)
    
    # Phase 1: Fixed Step Warm-up & Training Run
    print(f"Starting Phase 1: Training targeted for {MAX_STEPS} steps...")
    model.train()
    for step in range(1, MAX_STEPS + 1):
        X, Y = get_batch(BLOCK_SIZE, batch_size, device)
        logits, loss = model(X, Y)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step % 100 == 0:
            print(f"Step {step}/{MAX_STEPS} | Loss: {loss.item():.4f}")
            
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), "best_model_phase1.pt")

    # Phase 2: Post-Optimization Loop (Runs until saturation convergence occurs)
    print("\nStarting Phase 2: Post-optimization convergence loop...")
    patience = 2000
    no_improvement_counter = 0
    post_step = 0
    
    # Scale down learning rate for fine adjustments during phase 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate * 0.1

    while True:
        post_step += 1
        X, Y = get_batch(BLOCK_SIZE, batch_size, device)
        logits, loss = model(X, Y)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        current_loss = loss.item()
        
        if current_loss < best_loss - 1e-4:
            print(f"Post-Loop Step {post_step} | Enhanced Loss Found: {current_loss:.6f}. Saving final configuration.")
            best_loss = current_loss
            torch.save(model.state_dict(), "best_final_optimized_model.pt")
            no_improvement_counter = 0
        else:
            no_improvement_counter += 1
            
        if post_step % 200 == 0:
            print(f"Post-Loop Step {post_step} | Stability evaluation score (Loss): {current_loss:.4f} | Idle count: {no_improvement_counter}/{patience}")
            
        if no_improvement_counter >= patience:
            print(f"\nConvergence reached. Loss has saturated at {best_loss:.6f} with no improvements for {patience} steps.")
            print("Terminating loop process. Optimal structural model saved successfully.")
            break

if __name__ == '__main__':
    train_model()
