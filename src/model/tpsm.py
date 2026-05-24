# TecX Puzzle Solving Model
# import encoding_decoding as ed #
## import importdataset as imd #
import torch
import torch.nn as nn
import math
from torch.nn import functional as F

#"" "
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 20 #6
n_layer = 20 #6
dropout = 0.2
# ------------
#" ""

class DictionaryTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, num_classes):
        super(DictionaryTransformer, self).__init__()
        # 1. Embedding Layer: Converts character indices to vectors
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 2. Positional Encoding: Helps the model understand the sequence order
        self.pos_encoder = nn.Parameter(torch.zeros(1, 1000, d_model)) 
        
        # 3. Transformer Encoder: Learns complex patterns in the nested data
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # 4. Output Layer: Predicts the final class/category
        self.fc_out = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        ######seq_len = x.size(1)
        seq_len = x.size(0)
        ######seq_len = x.size(1)
        x = self.embedding(x) + self.pos_encoder[:, :seq_len, :]
        
        # Pass through Transformer
        x = self.transformer_encoder(x)
        
        # Use the mean of the sequence for final classification
        x = x.mean(dim=1) 
        return self.fc_out(x)
    ########
    
    def generate_stream(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            # 1. Get predictions
            idx_cond = idx[:, -block_size:]
            # Change this:
            # logits, _ = self(idx_cond)

            # To this:
            logits = self(idx_cond)

            ## logits, _ = self(idx_cond)
            ## logits = logits[:, -1, :] / temperature
            # Change this:
            # logits = logits[:, -1, :] / temperature

            # To this:
            logits = logits / temperature

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
        
        #######
if __name__ == "__main__":
    import importdataset as imd #
    
    # Setup Example
    vocab_size = 80  # Size of your 'stoi' map
    model = DictionaryTransformer(vocab_size=vocab_size, d_model=128, nhead=8, num_layers=4, num_classes=3)
    # model = DictionaryTransformer(vocab_size=vocab_size, d_model=128, nhead=20, num_layers=12, num_classes=3)
    input = imd.ImportDataset() #
    model(input) #
