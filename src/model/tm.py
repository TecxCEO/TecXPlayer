import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
import math
import random
import sys

# =====================================================================
# 1. OPTIMIZED INFINITE SYSTEM HYPERPARAMETERS
# =====================================================================
batch_size = 18       
block_size = 688      # 16 steps * 43 tokens per step
n_embed = 512         
n_head = 8            
n_layer = 8           
dropout = 0.0         # Set to 0.0 since overfitting is impossible on 4 quadrillion states
VOCAB_SIZE = 50       # Base tokens (0 to 49)

# Register Task and Structural Tokens
SOS_TOKEN = VOCAB_SIZE      # 50
EOS_TOKEN = VOCAB_SIZE + 1  # 51
TASK_FWD  = VOCAB_SIZE + 2  # 52
TASK_REV  = VOCAB_SIZE + 3  # 53
TASK_SOLV = VOCAB_SIZE + 4  # 54

ACTUAL_VOCAB_SIZE = VOCAB_SIZE + 5 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# =====================================================================
# 2. TRANSFORMER LAYERS (Optimized for Rule-Learning)
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
        self.key, self.query, self.value = nn.Linear(n_embed, head_size, bias=False), nn.Linear(n_embed, head_size, bias=False), nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    def forward(self, x):
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)
        wei = (q @ k.transpose(-2, -1)) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        return F.softmax(wei, dim=-1) @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embed)
    def forward(self, x): return self.proj(torch.cat([h(x) for h in self.heads], dim=-1))

class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embed, 4 * n_embed), nn.ReLU(), nn.Linear(4 * n_embed, n_embed))
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
        self.eval()
        for _ in range(max_tokens_to_generate):
            with torch.no_grad():
                logits, _ = self.forward(idx)
                next_tokens = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
                idx = torch.cat([idx, next_tokens], dim=-1)
                yield next_tokens.squeeze(-1).tolist(), idx

# =====================================================================
# 3. ANTI-FLIP DICTIONARY MANAGEMENT ENGINE
# =====================================================================

# Split token IDs into 25 for 3-characters, 25 for 2-characters
VALID_3_CHAR_IDS = list(range(0, 25))  # 0 to 24
VALID_2_CHAR_IDS = list(range(25, 50)) # 25 to 49

# Map every Token ID to a unique set of letters to track flips
TOKEN_CHARACTER_MAP = {}

# Build 25 distinct letter groups for 3-character tokens
# We ensure no groups have identical letters to keep selections clean
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
for idx, token_id in enumerate(VALID_3_CHAR_IDS):
    # Grabs a unique triplet of letters (frozen set order-invariance tracker)
    chars = frozenset([alphabet[idx % 26], alphabet[(idx+1) % 26], alphabet[(idx+2) % 26]])
    TOKEN_CHARACTER_MAP[token_id] = chars

# Build 25 distinct letter groups for 2-character tokens
for idx, token_id in enumerate(VALID_2_CHAR_IDS):
    chars = frozenset([alphabet[idx % 26], alphabet[(idx+4) % 26]])
    TOKEN_CHARACTER_MAP[token_id] = chars

def generate_state_without_flips():
    """
    Generates a single state of 20 element IDs. 
    Guarantees no two tokens share the same character set (no flips) in this state.
    """
    while True:
        # Step 1: Draw sample candidates from both zones
        block_8 = random.sample(VALID_3_CHAR_IDS, 8)
        block_12 = random.sample(VALID_2_CHAR_IDS, 12)
        candidate_state = block_8 + block_12
        
        # Step 2: Validate that no item breaks your same-state flip constraint
        seen_character_sets = set()
        duplicate_flip_found = False
        
        for token_id in candidate_state:
            char_set = TOKEN_CHARACTER_MAP[token_id]
            if char_set in seen_character_sets:
                duplicate_flip_found = True
                break
            seen_character_sets.add(char_set)
            
        # If a flip combination coexisted, scrap the generation and try again
        if duplicate_flip_found:
            continue
            
        random.shuffle(candidate_state)
        return candidate_state

# =====================================================================
# 4. INFINITE STREAM GENERATOR & ITERABLE DATASET
# =====================================================================

def puzzle_simulation_generator():
    """Endless simulator streaming game sequences across steps."""
    while True:
        steps = 16
        initial_state = generate_state_without_flips()
        states_history, moves_list = [], []
        
        for _ in range(steps):
            moves_list.append(random.randint(0, 49))
            states_history.append(generate_state_without_flips())
            
        yield {
            'initial_state': initial_state,
            'moves': moves_list,
            'states_history': states_history
        }

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
# 5. STREAM PIPELINE TRAINING ENGINE
# =====================================================================

def train_infinitely(model, total_steps_to_train=100000):
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    dataloader = DataLoader(InfinitePuzzleStream(puzzle_simulation_generator), batch_size=batch_size)

    print(f"📡 Infinite pipeline running. Optimizing over {total_steps_to_train} unique data batches...")
    
    running_loss = 0.0
    for step_idx, (X, Y) in enumerate(dataloader):
        X, Y = X.to(device), Y.to(device)
        
        optimizer.zero_grad()
        _, loss = model(X, targets=Y)
        loss.backward()
