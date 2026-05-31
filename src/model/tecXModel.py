import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import os
import string
import sys

# =====================================================================
# 1. GLOBAL SYSTEM HYPERPARAMETERS
# =====================================================================
batch_size = 18       # Parallel independent game streams processed together
block_size = 356      # Max timeline context: 20 (init) + 16 * (1 move + 20 positions)
n_embed = 256         # Dimension size of the internal vector mappings
n_head = 8            # 8 attention mechanisms (n_embed 256 divides by 8 perfectly)
n_layer = 6           # Number of processing layers stacked sequentially
dropout = 0.1         # Regularisation dropping probability
VOCAB_SIZE = 50       # Number of unique puzzle elements / move IDs in your index
device = 'cuda' if torch.cuda.is_available() else 'cpu'


##№################################




# hyperparameters
#batch_size =16 # 32 #1 # 64 # how many independent sequences will we process in parallel?
#block_size =48 # 1 # for [state move state] #3 for state move state #64 #256 # what is the maximum context length for predictions?
epochs = 11
max_iters = 1# len(self.data)//3 #5000
eval_interval = 500
learning_rate = 3e-4
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
####n_embd = 384
#n_head = 8 #20 #6
#n_layer = 6 #20 # 6
#dropout = 0.2

torch.manual_seed(1337)
total_val_loss = 0.0  

class TecXModelTrain:
    
    def __init__(self, data, stoi, itos, valdata =[]):
        self.train_data = data
        self.val_data = valdata
        self.stoi = stoi
        self.itos = itos
        print(f" stoi in init fun = {self.stoi}")
        print(f" itos in init fun = {self.itos}")
        self.vocab_size = len(stoi)
        batch_size = len(self.train_data)
        
        max_iters = len(self.train_data)/batch_size
    def get_batch(self, split):
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - 1,  (batch_size,))
        x = torch.stack([torch.tensor(data[i], dtype=torch.long) for i in ix])
        # Convert shifted integer STOI target tokens to PyTorch Long Tensors
        y = torch.stack([torch.tensor(data[i+1], dtype=torch.long) for i in ix])
        
        x, y = x.to(device), y.to(device)
        return x, y
    
    @torch.no_grad()#
    
    def estimate_loss(self, model):
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = self.get_batch(split)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
                out[split] = losses.mean()
        model.train()
        return out
    def trainModel(self, tmodel= None, m_checkpoint = None):
        if m_checkpoint:
            checkpoint= m_checkpoint
        if tmodel:
            model = tmodel
            print(f" Model training start from last time trained model.")
        else:
            print(f" vocab_size = {self.vocab_size}")
            model = TecXModel(int(self.vocab_size))
            if locals().get("checkpoint") is not None:
                model_dict = checkpoint["model_state_dict"]
                optimizer_dict = checkpoint['optimizer_state_dict']
                self.stoi = checkpoint["stoi"]
                self.itos = checkpoint["itos"]
                model.load_state_dict(model_dict, strict=False)
                # Ensure your model is in evaluation mode
        print(f" stoi in tecxModel = {self.stoi}")
        print(f" itos  in tecxModel = {self.itos}")
        if locals().get("checkpoint"):
            print(f" checkpoint stoi in tecxModel = {checkpoint["stoi"]}")
            print(f" checkpoint itos in tecxModel = {checkpoint["itos"]}")
        model.eval()
        m = model.to(device)
        # print the number of parameters in the model
        print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
        # create a PyTorch optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        if locals().get("optimizer_dict") is not None:
            optimizer.load_state_dict(optimizer_dict, strict=False)#####
            optimizer.eval()
        best_val_loss = float('inf') # Start with infinity
        # Initialize a variable outside your training loop to track the last saved file
        prev_checkpoint_path = None
        for epoch in range(epochs):
            for iter in range(max_iters):
                print(f"In The Iteration no = {iter}")
                # every once in a while evaluate the loss on train and val sets
                if iter % eval_interval == 0 or iter == max_iters - 1:
                    losses = self.estimate_loss(model)
                    print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                # sample a batch of data
                xb, yb = self.get_batch('train')
                # evaluate the loss
                logits, loss = model(xb, yb)
            # 1. Initialize the counter BEFORE the loop
            total_val_loss = 0.0 
            # 2. Your validation loop
            print(f" loss = {loss}")
            total_val_loss += loss
            print(f" total_val_loss = {total_val_loss}")
            total_val_loss = loss
            print(f" total_val_loss after = {total_val_loss}")
            # 3. NOW you can calculate the average
            avg_val_loss = total_val_loss / batch_size
            print(f" avg_val_loss after = {avg_val_loss}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if locals().get("checkpoint") :
                checkpoint['epoch'] = epoch + 1
                checkpoint.update({'model_state_dict': model.state_dict()})
                checkpoint.update({'optimizer_state_dict': optimizer.state_dict()})
                checkpoint.update({'best_val_loss': best_val_loss})
                checkpoint.update({'stoi': self.stoi}) #edc.stoi , # Saving the vocabulary is critical!
                checkpoint.update({'itos' : self.itos}) #edc.itos
            elif locals().get("checkpoint") is None:
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'stoi': self.stoi, #edc.stoi , # Saving the vocabulary is critical!
                    'itos' : self.itos #edc.itos
                    }
            print(f" checkpoint before save = {checkpoint}")
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(checkpoint, 'models/tecx/tecx_best_model.pth')
                print(f"--> Saved new best model with Val Loss: {best_val_loss:.4f}")
            print(f" checkpoint[stoi] = {checkpoint['stoi']}")
            print(f" checkpoint[itos] = {checkpoint['itos']}")
            # Save progress
            # 1. Define the path for the current epoch
            current_checkpoint_path = f"models/tecx/tecx_model_epoch_{epoch}.pth"
            # 2. Save the new model checkpoint
            torch.save(checkpoint, current_checkpoint_path)
            print(f"Saved: {current_checkpoint_path}")
            # 3. Delete the previous epoch's file if it exists
            if prev_checkpoint_path and os.path.exists(prev_checkpoint_path):
                os.remove(prev_checkpoint_path)
            print(f"Deleted previous checkpoint: {prev_checkpoint_path}")
            # 4. Update the tracker to point to the current file for the next iteration
            prev_checkpoint_path = current_checkpoint_path
        return model, checkpoint
    
########################

# =====================================================================
# 2. TRANSFORMER INFRASTRUCTURE CLASSES
# =====================================================================

class PositionalEncoding(nn.Module):
    """Injects unique spatial ordering vectors so tokens keep strict sequence alignment."""
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
    """Calculates causal single-head self-attention relationships."""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5) # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # Look-ahead block mask
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        v = self.value(x) # (B, T, head_size)
        return wei @ v    # (B, T, head_size)


class MultiHeadAttention(nn.Module):
    """Combines calculations from independent attention views."""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """Standard non-linear multi-layer perceptron processing channel."""
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Structural layout containing Normalisation, Attention, and Feed-Forward processing."""
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.attn = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# =====================================================================
# 3. MAIN PUZZLE MODEL CONTROLLER (With internal stream method)
# =====================================================================
# class PuzzleModel(nn.Module):
class TecXModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding = PositionalEncoding(n_embed, max_len=block_size)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        x = self.position_embedding(self.token_embedding_table(idx))
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    # 🌊 CRITICAL STREAM METHOD ADDED HERE
    def stream(self, idx):
        """
        Takes an input tensor 'idx' of shape (Batch_Size, Current_Sequence_Length).
        Accepts: 20 input state tokens + 1 move token (21 tokens total).
        Yields: Tokens ONE BY ONE for exactly 20 positions in real-time.
        """
        self.eval() # Freeze active regularisation
        
        for _ in range(20):
            with torch.no_grad():
                # Compute predictions based on current sequence history size
                logits, _ = self.forward(idx)
                
                # Snatch logits mapping the final generated index position
                next_token_logits = logits[:, -1, :]
                next_tokens = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1) # (B, 1)
                
                # Append predicted token back to input matrix for next iteration tracking
                idx = torch.cat([idx, next_tokens], dim=-1)
                
                # Yield current predictions across all batch tracks as a simple list
                yield next_tokens.squeeze(-1).tolist()
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    def generate_stream(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            # 1. Get predictions
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            # 2. Apply Top-K filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # 3. Sample and Append
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            # 4. YIELD the newly generated token index
            yield idx_next.item() 

# =====================================================================
# 4. DATABASE PACKAGING INFRASTRUCTURE
# =====================================================================

class PuzzleDataset(Dataset):
    def __init__(self, raw_data, pad_token_id=0):
        self.data = raw_data
        self.pad_token_id = pad_token_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        puzzle = self.data[idx]
        timeline = list(puzzle['initial_state'])
        
        # Interleave moves and their resulting state updates
        for i in range(len(puzzle['moves'])):
            timeline.append(puzzle['moves'][i])
            timeline.extend(puzzle['states_history'][i])
            
        # Dynamically map shorter timelines up to the max budget constraint block 
        if len(timeline) < block_size:
            timeline.extend([self.pad_token_id] * (block_size - len(timeline)))
            
        timeline_tensor = torch.tensor(timeline[:block_size], dtype=torch.long)
        return timeline_tensor[:-1], timeline_tensor[1:]

# =====================================================================
# 5. TRAINING RUNNER & VALIDATION ROUTINES
# =====================================================================

def calculate_validation_accuracy(model, validation_raw_data):
    """Runs a complete test pass to find exact structural order correctness match."""
    model.eval()
    total_steps_evaluated = 0
    perfect_steps_completed = 0
    
    for i in range(0, len(validation_raw_data), batch_size):
        batch_puzzles = validation_raw_data[i:i+batch_size]
        if len(batch_puzzles) < batch_size:
            break
            
        # Compile start arrays for current evaluation slice
        batch_initial_states = [p['initial_state'] for p in batch_puzzles]
        active_indices = list(range(len(batch_puzzles)))
        
        timeline_tensor = torch.tensor(batch_initial_states, dtype=torch.long, device=device)
        first_moves = [batch_puzzles[idx]['moves'] for idx in active_indices]
        first_moves_tensor = torch.tensor(first_moves, dtype=torch.long, device=device).unsqueeze(-1)
        timeline_tensor = torch.cat([timeline_tensor, first_moves_tensor], dim=-1)

        # Loop through steps up to full timeline depth execution
        for step in range(16):
            current_batch_size = timeline_tensor.size(0)
            
            # Utilize the internal generation layer step block
            for _ in range(20):
                with torch.no_grad():
                    logits, _ = model(timeline_tensor)
                    next_tokens = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
                    timeline_tensor = torch.cat([timeline_tensor, next_tokens], dim=-1)
                    
            if step < 15:
                next_active_indices = []
                next_moves_list = []
                for local_idx, global_idx in enumerate(active_indices):
                    game_moves = batch_puzzles[global_idx]['moves']
                    if (step + 1) < len(game_moves):
                        next_active_indices.append(global_idx)
                        next_moves_list.append(game_moves[step + 1])
                        
                if len(next_active_indices) < current_batch_size:
                    local_rows_to_keep = [active_indices.index(g_idx) for g_idx in next_active_indices]
                    timeline_tensor = timeline_tensor[local_rows_to_keep, :]
