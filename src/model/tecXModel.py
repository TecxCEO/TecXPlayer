import os
import string
import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size =16 # 32 #1 # 64 # how many independent sequences will we process in parallel?
block_size =48 # 1 # for [state move state] #3 for state move state #64 #256 # what is the maximum context length for predictions?
epochs = 11
max_iters = 1# len(self.data)//3 #5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 8 #20 #6
n_layer = 6 #20 # 6
dropout = 0.2

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
class Head(nn.Module):
    """ one head of self-attention """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    ##print(f" In the MultiHeadAttention Class") #
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """
    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

#class TecXLanguageModel(nn.Module):
class TecXModel(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original TecX video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        ##print(f" In the TecXLanguageModel Class's generate function.") #
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
        

if __name__ == "__main__":
    model = TecXModel()
    m = model.to(device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for iter in range(max_iters):
        print(f"In The Iteration no = {iter}")
        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        # sample a batch of data
        xb, yb = get_batch('train')
        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    #torch.save(m.state_dict(),"../TecXLM.pth")
    torch.save({'state_dict': model.state_dict(), 'chars': chars}, 'model/tecx/tecxmodel.pth')
    #torch.save({'state_dict': model.state_dict(), 'chars': char_list}, 'model.pth')

    
    # torch.save(m.state_dict(),"TecXLM.pt")
    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
    #open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))
    open('TecXLM_learned.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))
    
    """
    1. Set Up the Logger
    Add this at the top of your tecxlmgenerate.py script. It creates a file named generation_logs.txt and appends new conversations to the bottom.
    """
    def log_conversation(prompt, response):
        with open("generation_logs.txt", "a", encoding="utf-8") as f:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"\n{'='*50}\n")
            f.write(f"TIMESTAMP: {timestamp}\n")
            f.write(f"PROMPT: {prompt}\n")
            f.write(f"RESPONSE: {response}\n")
            f.write(f"{'='*50}\n")
    """
    Update the Interactive Loop
    Use sys.stdout.write and flush() to make the characters appear instantly on the same line.
    """
    # ... inside your 'while True' loop ...
    while True:
        # 1. Get custom text from the user
        user_prompt=input("\nEnter your starting text (or type 'exit' to quit): ")
        if user_prompt.lower() == 'exit':
            break
        # Encode and setup context
        context_list = [stoi[c] for c in user_prompt if c in stoi]
        context = torch.tensor([context_list], dtype=torch.long, device=device)
        print(f"\n[TECX LM]: ", end="")
        sys.stdout.flush()
        # Set the creativity (temperature) and focus (top_k)
        temp = 0.4 # 1.0 is standard; higher is more creative, lower is more focused
        top_k = None # 5   # Keeps the model focused on the top 5 most likely characters
        tokens = 500 # Number of characters to generate
        with torch.no_grad():
            # Use the generator function
            # 1. Initialize an empty string to hold the output
            full_response = "" 
            # 2. Start the streaming loop
            #for token_id in model.generate_stream(context, tokens, temp, top_k):
            for token_id in m.generate_stream(context, tokens, temp, top_k):
                char = decode([token_id])
                sys.stdout.write(char)
                sys.stdout.flush()
                full_response += char # Collect for logging
                # Optional: Add a tiny sleep to make it look like "typing"
                time.sleep(0.01) 
        print("\n" + "-"*30)
        # Automatically save the conversation
        log_conversation(user_prompt, full_response)
        print("\n\n(Conversation saved to generation_logs.txt)")
