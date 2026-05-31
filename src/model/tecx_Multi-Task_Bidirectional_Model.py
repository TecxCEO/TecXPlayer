# tecx_Multi-Task_Bidirectional_Model_Script.py
# Multi-Task Bidirectional Model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import random
import sys

# =====================================================================
# 1. SYSTEM HYPERPARAMETERS
# =====================================================================
batch_size = 18       
block_size = 688      # Strict timeline width constraint: 16 steps * 43 tokens
n_embed = 256         
n_head = 8            
n_layer = 6           
dropout = 0.1         
VOCAB_SIZE = 50       # Raw puzzle states and action values (IDs 0 to 49)

# Register Structural Tokens
SOS_TOKEN = VOCAB_SIZE      # 50
EOS_TOKEN = VOCAB_SIZE + 1  # 51
TASK_FWD  = VOCAB_SIZE + 2  # 52 (Forward Calculation)
TASK_REV  = VOCAB_SIZE + 3  # 53 (Reverse Calculation)
TASK_SOLV = VOCAB_SIZE + 4  # 54 (Solve Path Planning)

ACTUAL_VOCAB_SIZE = VOCAB_SIZE + 5 # Total vocabulary capacity = 55
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# =====================================================================
# 2. TRANSFORMER CORE INFRASTRUCTURE
# =====================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=block_size):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1)]

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        return self.dropout(F.softmax(wei, dim=-1)) @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embed)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x): return self.dropout(self.proj(torch.cat([h(x) for h in self.heads], dim=-1)))

class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embed, 4 * n_embed), nn.ReLU(), nn.Linear(4 * n_embed, n_embed), nn.Dropout(dropout))
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        self.attn = MultiHeadAttention(n_head, n_embed // n_head)
        self.ffwd = FeedForward(n_embed)
        self.ln1, self.ln2 = nn.LayerNorm(n_embed), nn.LayerNorm(n_embed)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ffwd(self.ln2(x))

# =====================================================================
# 3. BI-DIRECTIONAL MULTI-TASK PUZZLE MODEL
# =====================================================================

class BidirectionalPuzzleModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding = PositionalEncoding(n_embed, max_len=block_size)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        x = self.position_embedding(self.token_embedding_table(idx))
        logits = self.lm_head(self.ln_f(self.blocks(x)))
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1)) if targets is not None else None
        return logits, loss

    def stream_dynamic_tokens(self, idx, max_tokens_to_generate=20):
        """Streams out tokens one by one until it hits max ceiling or generates an EOS."""
        self.eval()
        for _ in range(max_tokens_to_generate):
            with torch.no_grad():
                logits, _ = self.forward(idx)
                next_token_logits = logits[:, -1, :]
                next_tokens = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                idx = torch.cat([idx, next_tokens], dim=-1)
                
                live_tokens = next_tokens.squeeze(-1).tolist()
                yield live_tokens, idx
                
                # Halt if all parallel channels safely emitted their closing boundaries
                if isinstance(live_tokens, list) and all(t == EOS_TOKEN for t in live_tokens):
                    break
                elif isinstance(live_tokens, int) and live_tokens == EOS_TOKEN:
                    break

# =====================================================================
# 4. MULTI-TASK DATASET FORMATTER
# =====================================================================

class MultiTaskPuzzleDataset(Dataset):
    def __init__(self, raw_data, pad_token_id=0):
        self.data = raw_data
        self.pad_token_id = pad_token_id

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        puzzle = self.data[idx]
        timeline = []
        
        # Randomly choose which task to train this specific sample on
        task_type = random.choice(['FORWARD', 'REVERSE', 'SOLVE'])
        
        if task_type == 'FORWARD':
            # Task 1, 2, 3: Forward sequence step mapping
            for i in range(len(puzzle['moves'])):
                curr_st = puzzle['initial_state'] if i == 0 else puzzle['states_history'][i-1]
                step_block = [SOS_TOKEN, TASK_FWD] + list(curr_st) + [puzzle['moves'][i]] + list(puzzle['states_history'][i]) + [EOS_TOKEN]
                timeline.extend(step_block)
                
        elif task_type == 'REVERSE':
            # Task 4: Reverse State lookup (Given result + move -> predict original)
            for i in range(len(puzzle['moves'])):
                curr_st = puzzle['initial_state'] if i == 0 else puzzle['states_history'][i-1]
                step_block = [SOS_TOKEN, TASK_REV] + list(puzzle['states_history'][i]) + [puzzle['moves'][i]] + list(curr_st) + [EOS_TOKEN]
                timeline.extend(step_block)
                
        elif task_type == 'SOLVE':
            # Task 5: Solve Path Planner (Given Current + Final Solve State -> predict moves)
            # Pick a random point in the game history as the "current state"
            random_start_idx = random.randint(0, len(puzzle['moves']) - 1)
            current_state = puzzle['initial_state'] if random_start_idx == 0 else puzzle['states_history'][random_start_idx - 1]
            solve_state = puzzle['states_history'][-1] # The absolute final target position
            
            # Extract the path of moves required to bridge the gap
            remaining_moves = puzzle['moves'][random_start_idx:]
            
            timeline = [SOS_TOKEN, TASK_SOLV] + list(current_state) + list(solve_state) + list(remaining_moves) + [EOS_TOKEN]

        # Standardize timelines length using padding
        if len(timeline) < block_size:
            timeline.extend([self.pad_token_id] * (block_size - len(timeline)))
            
        t_tensor = torch.tensor(timeline[:block_size], dtype=torch.long)
        return t_tensor[:-1], t_tensor[1:]

# =====================================================================
# 5. EXECUTION PIPELINE VERIFICATION
# =====================================================================

if __name__ == "__main__":
    # Create simple dummy model instance 
    model = BidirectionalPuzzleModel(vocab_size=ACTUAL_VOCAB_SIZE).to(device)
    model.eval()

    print("🚀 TESTING LIVE MULTI-TASK BIDIRECTIONAL ENGINE OPERATIONS...")

    # --- SIMULATION 1: RUNNING A REVERSE STATE LOOKUP ---
    print("\n🔄 OPERATION 1: Running Reverse Lookup (Find starting state from result state)...")
    mock_sos = torch.full((batch_size, 1), SOS_TOKEN, device=device)
    mock_task = torch.full((batch_size, 1), TASK_REV, device=device)
    mock_result_state = torch.randint(0, VOCAB_SIZE, (batch_size, 20), device=device)
    mock_action_taken = torch.randint(0, VOCAB_SIZE, (batch_size, 1), device=device)
    
    # Pack input array: [SOS] + [TASK_REV] + Result State (20) + Move (1) = 23 tokens
    rev_prompt = torch.cat([mock_sos, mock_task, mock_result_state, mock_action_taken], dim=-1)
    
    # Stream back the original 20 starting positions
    streamer = model.stream_dynamic_tokens(rev_prompt, max_tokens_to_generate=20)
    for sub_idx, (tokens_out, _) in enumerate(streamer):
        sys.stdout.write(f"  Reverse Reconstruction | Position [{sub_idx+1:02d}/20] -> Predicted Prior State: {tokens_out}\n")
        sys.stdout.flush()

    # --- SIMULATION 2: PATH SOLVING ROUTINE ---
    print("\n🧩 OPERATION 2: Running Puzzle Solver (Find move paths from current state to solve state)...")
    mock_task_solv = torch.full((batch_size, 1), TASK_SOLV, device=device)
    mock_current_state = torch.randint(0, VOCAB_SIZE, (batch_size, 20), device=device)
    mock_solved_state = torch.randint(0, VOCAB_SIZE, (batch_size, 20), device=device)
    
    # Pack input array: [SOS] + [TASK_SOLV] + Current (20) + Target Solve State (20) = 42 tokens
