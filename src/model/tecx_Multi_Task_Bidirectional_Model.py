# tecx_Multi-Task_Bidirectional_Model_Script.py
# Multi-Task Bidirectional Model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import os
import random
import string
import sys




#####################
#####################

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

"""
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
"""

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
    # 1. DEFINE THE FUNCTION HERE (Inside the main block)
    def create_synthetic_data(count):
        pool = []
        for _ in range(count):
            steps = 16  # Test exactly 16 moves sequences depth
            pool.append({
                'initial_state': [i % VOCAB_SIZE for i in range(20)],
                'moves': [(i + 5) % VOCAB_SIZE for i in range(steps)],
                'states_history': [[(i + j) % VOCAB_SIZE for i in range(20)] for j in range(steps)]
            })
        return pool
    def train_bidirectional_model(model, raw_puzzle_logs, epochs=5, lr=3e-4):
        """
        Executes the actual mathematical training across all 5 tasks simultaneously.
        """
        print(f"🏋️‍♂️ Initializing Multi-Task Training Pipeline on: {device}")
        model.to(device)
        # 1. Setup multi-task datasets and packaging streams
        dataset = MultiTaskPuzzleDataset(raw_puzzle_logs)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        # Standard transformer weight optimisation configuration
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        # Activate internal training layers (Normalisation updates, Dropout filters)
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_idx, (X_batch, Y_batch) in enumerate(dataloader):
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                # -------------------------------------------------------------
                # THE CRITICAL LINES THAT TRAIN THE MODEL:
                # -------------------------------------------------------------
                optimizer.zero_grad()                 # Line A: Clear old memories
                logits, loss = model(X_b, targets=Y_b) # Line B: Calculate guess mistakes
                loss.backward()                       # Line C: Backpropagate gradient error
                optimizer.step()                      # Line D: Update neural weight nodes
                # -------------------------------------------------------------
                epoch_loss += loss.item()
            print(f"🔥 Epoch [{epoch+1}/{epochs}] Complete. Cross-Entropy Loss: {epoch_loss / len(dataloader):.4f}")
        print("🏆 Training complete! Saving bidirectional weights...")
        torch.save(model.state_dict(), "bidirectional_puzzle_engine.pth")
    
    def trainModel(self, tmodel= None, m_checkpoint = None):
        if m_checkpoint:
            checkpoint= m_checkpoint
        if tmodel:
            model = tmodel
            print(f" Model training start from last time trained model.")
        else:
            print(f" vocab_size = {self.vocab_size}")
            ##model = TecXBidirectionalPuzzleModel(int(self.vocab_size))
            ####model = TecXModel(int(self.vocab_size))
            model = TecXBidirectionalPuzzleModel(vocab_size=ACTUAL_VOCAB_SIZE).to(device)
            ####model.eval()
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
        """
        ###############
        ######
        # 1. Generate 180 random synthetic puzzle profiles (10 batches of 18 parallel items)
        my_raw_training_data = create_synthetic_data(180)
        # 2. Initialize your bidirectional model container
        model = BidirectionalPuzzleModel(vocab_size=ACTUAL_VOCAB_SIZE).to(device)
        # 3. RUN THIS COMMAND TO TRAIN THE ENTIRE MODEL FRAMEWORK LIVE
        train_bidirectional_model(model, my_raw_training_data, epochs=10)
        # Now you can safely call your inference streamer functions using trained parameters!
        ############
        ###############
        """
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

            """
            ###########
            ###########
            if __name__ == "__main__":
                # 1. Generate 180 random synthetic puzzle profiles (10 batches of 18 parallel items)
                my_raw_training_data = create_synthetic_data(180)
                # 2. Initialize your bidirectional model container
                model = BidirectionalPuzzleModel(vocab_size=ACTUAL_VOCAB_SIZE).to(device)
                # 3. RUN THIS COMMAND TO TRAIN THE ENTIRE MODEL FRAMEWORK LIVE
                train_bidirectional_model(model, my_raw_training_data, epochs=10)
                # Now you can safely call your inference streamer functions using trained parameters!
    
            ###########
            ###########
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
            """
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
    
#####################
#####################

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

class TecXBidirectionalPuzzleModel(nn.Module):
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
