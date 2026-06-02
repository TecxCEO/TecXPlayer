import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
import math
import random
import sys

# =====================================================================
# 1. OPTIMIZED HIGH-CAPACITY SYSTEM HYPERPARAMETERS
# =====================================================================
batch_size = 18       # Parallel independent game streams processed together
block_size = 688      # Max timeline context length: 16 steps * 43 tokens per step
n_embed = 512         # Dimension size of the internal vector mappings
n_head = 8            # 8 attention processing channels (512 / 8 = 64 dimensions per head)
n_layer = 8           # 8 deep logical blocks stacked sequentially
dropout = 0.0         # Set to 0.0 because overfitting is impossible on 4 quadrillion states

# Vocabulary Rules Map (72 Base Tokens + 5 Structural Flags = 77 Total)
# IDs 0 to 47   -> 48 Tokens (8 elements x 6 flipped variations each)
# IDs 48 to 71  -> 24 Tokens (12 elements x 2 flipped variations each)
SOS_TOKEN = 72      
EOS_TOKEN = 73      
TASK_FWD  = 74      
TASK_REV  = 75      
TASK_SOLV = 76      

ACTUAL_VOCAB_SIZE = 77 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# =====================================================================
# 2. TRANSFORMER INFRASTRUCTURE CLASSES
# =====================================================================

class PositionalEncoding(nn.Module):
    """CRITICAL: Injects unique order vectors so the model remembers strict element positioning."""
    def __init__(self, d_model, max_len=block_size):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) # Shape: (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)
        wei = (q @ k.transpose(-2, -1)) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # Causal forward mask
        return F.softmax(wei, dim=-1) @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embed)
    def forward(self, x):
        return self.proj(torch.cat([h(x) for h in self.heads], dim=-1))


class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed)
        )
    def forward(self, x): return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        self.attn = MultiHeadAttention(n_head, n_embed // n_head)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ffwd(self.ln2(x))

# =====================================================================
# 3. BIDIRECTIONAL MODEL CLASS WITH SINGLE-STEP STREAMING METHOD
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

    def stream_single_step(self, idx):
        """Streams out the 20 state tokens one by one, then appends a closing [EOS]."""
        self.eval()
        for _ in range(20):
            with torch.no_grad():
                logits, _ = self.forward(idx)
                next_tokens = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
                idx = torch.cat([idx, next_tokens], dim=-1)
                yield next_tokens.squeeze(-1).tolist(), idx
        with torch.no_grad():
            eos_tensor = torch.full((idx.size(0), 1), EOS_TOKEN, dtype=torch.long, device=idx.device)
            idx = torch.cat([idx, eos_tensor], dim=-1)
            yield eos_tensor.squeeze(-1).tolist(), idx

# =====================================================================
# 4. DATA GENERATOR ENFORCING EXCLUSIVITY & SLOT SEGMENTATION
# =====================================================================

def generate_state_with_strict_rules():
    """
    Builds a single state of 20 element IDs following your strict constraints:
    - First 8 positions: 3-character tokens (0-47), NO shared base element families (no flips)
    - Next 12 positions: 2-character tokens (48-71), NO shared base element families (no flips)
    """
    # Rule 1: Sample 8 unique element families out of 8, then choose 1 random flip variant (0-5) for each
    chosen_3char_tokens = []
    for element_idx in range(8):
        # Base ID starts at element_idx * 6. We add a random flip offset between 0 and 5.
        flip_variant = random.randint(0, 5)
        token_id = (element_idx * 6) + flip_variant
        chosen_3char_tokens.append(token_id)
    # Shuffle only the first 8 elements if you want, but your rule says they must come first:
    random.shuffle(chosen_3char_tokens) # Keeps them in the first 8 slots, but scrambles their order
        
    # Rule 2: Sample 12 unique element families out of 12, then choose 1 random flip variant (0-1) for each
    chosen_2char_tokens = []
    for element_idx in range(12):
        # Base ID starts at 48 + (element_idx * 2). We add a random flip offset between 0 and 1.
        flip_variant = random.randint(0, 1)
        token_id = 48 + (element_idx * 2) + flip_variant
        chosen_2char_tokens.append(token_id)
    random.shuffle(chosen_2char_tokens) # Keeps them in slots 8-19, but scrambles their order

    # Stitched state: First 8 are always 3-char, next 12 are always 2-char
    final_state = chosen_3char_tokens + chosen_2char_tokens
    return final_state


def puzzle_simulation_generator():
    """Endless simulator streaming game step logs across 4 quadrillion states."""
    while True:
        steps = 16
        initial_state = generate_state_with_strict_rules()
        states_history, moves_list = [], []
        
        for _ in range(steps):
            moves_list.append(random.randint(0, 71)) # A move is any valid token action ID
            states_history.append(generate_state_with_strict_rules())
            
        yield {
            'initial_state': initial_state,
            'moves': moves_list,
            'states_history': states_history
        }

# =====================================================================
# 5. DYNAMIC INFINITE MULTI-TASK STREAM DATASET
# =====================================================================

class InfinitePuzzleStream(IterableDataset):
    def __init__(self, generator_func, pad_token_id=0):
        self.generator_func = generator_func
        self.pad_token_id = pad_token_id

    def __iter__(self):
        for puzzle in self.generator_func():
            timeline = []
            task_type = random.choice(['FORWARD', 'REVERSE', 'SOLVE'])
            
            if task_type == 'FORWARD':
                for i in range(len(puzzle['moves'])):
                    curr_st = puzzle['initial_state'] if i == 0 else puzzle['states_history'][i-1]
                    timeline.extend([SOS_TOKEN, TASK_FWD] + list(curr_st) + [puzzle['moves'][i]] + list(puzzle['states_history'][i]) + [EOS_TOKEN])
            elif task_type == 'REVERSE':
                for i in range(len(puzzle['moves'])):
                    curr_st = puzzle['initial_state'] if i == 0 else puzzle['states_history'][i-1]
                    timeline.extend([SOS_TOKEN, TASK_REV] + list(puzzle['states_history'][i]) + [puzzle['moves'][i]] + list(curr_st) + [EOS_TOKEN])
            elif task_type == 'SOLVE':
                random_start_idx = random.randint(0, len(puzzle['moves']) - 1)
                current_state = puzzle['initial_state'] if random_start_idx == 0 else puzzle['states_history'][random_start_idx - 1]
                timeline = [SOS_TOKEN, TASK_SOLV] + list(current_state) + list(puzzle['states_history'][-1]) + list(puzzle['moves'][random_start_idx:]) + [EOS_TOKEN]

            if len(timeline) < block_size:
                timeline.extend([self.pad_token_id] * (block_size - len(timeline)))
            t_tensor = torch.tensor(timeline[:block_size], dtype=torch.long)
            yield t_tensor[:-1], t_tensor[1:]

# =====================================================================



# 6. PIPELINE REINFORCEMENT LOOP ENGINE=====================================================================

def train_on_live_stream_chunks(model, total_steps=100000):
  model.train()optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
  dataloader = DataLoader(InfinitePuzzleStream(puzzle_simulation_generator), batch_size=batch_size)
  print(f"📡 Infinite Chunk Stream Online. Optimizing over {total_steps} unique data matrices...")
  running_loss = 0.0for step_idx, (X, Y) in enumerate(dataloader):
    X, Y = X.to(device), Y.to(device)optimizer.zero_grad(), loss = model(X, targets=Y)loss.backward()torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=1.0)optimizer.step()running_loss += loss.item()if (step_idx + 1) % 100 == 0:print(f"⚙️ Step [{step_idx + 1}/{total_steps}] | Rolling Avg Loss: {running_loss / 100:.4f}")running_loss = 0.0if (step_idx + 1) % 10000 == 0:print("💾 Checkpoint interval reached. Saving secure brain weight snapshots...")torch.save(model.state_dict(), f"puzzle_engine_step_{step_idx+1}.pth")if step_idx + 1 >= total_steps:breakif name == "main":puzzle_model = BidirectionalPuzzleModel(vocab_size=ACTUAL_VOCAB_SIZE).to(device)
    # 1. Run the safe memory infinite chunk optimizer pass
train_on_live_stream_chunks(puzzle_model, total_steps=50000)# 2. Verify token-by-token live class generator streams safelyprint("\n🎬 VERIFYING LIVE MODEL STREAM PASS:")mock_sos = torch.full((batch_size, 1), SOS_TOKEN, device=device)mock_task = torch.full((batch_size, 1), TASK_FWD, device=device)mock_input_state = torch.tensor([generate_state_with_strict_rules() for _ in range(batch_size)], device=device)mock_move = torch.randint(0, 72, (batch_size, 1), device=device)streaming_prompt = torch.cat([mock_sos, mock_task, mock_input_state, mock_move], dim=-1) # Shape (18, 23)token_generator = puzzle_model.stream_single_step(streaming_prompt)for idx, (live_tokens, _) in enumerate(token_generator):if idx < 20:print(f"  Slot [Output Position {idx+1:02d}/20] -> Streaming Parallel Tokens: {live_tokens}")else:print(f"  Slot [Closing Boundary EOS ] -> Streaming Parallel Tokens: {live_tokens}")
