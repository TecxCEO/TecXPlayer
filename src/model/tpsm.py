# TecX Puzzle Solving Model
# import encoding_decoding as ed #
## import importdataset as imd #
import torch
import torch.nn as nn
import math

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
        seq_len = x.size(1)
        x = self.embedding(x) + self.pos_encoder[:, :seq_len, :]
        
        # Pass through Transformer
        x = self.transformer_encoder(x)
        
        # Use the mean of the sequence for final classification
        x = x.mean(dim=1) 
        return self.fc_out(x)
if __name__ == "__main__":
    import importdataset as imd #
    
    # Setup Example
    vocab_size = 80  # Size of your 'stoi' map
    model = DictionaryTransformer(vocab_size=vocab_size, d_model=128, nhead=8, num_layers=4, num_classes=3)
    # model = DictionaryTransformer(vocab_size=vocab_size, d_model=128, nhead=20, num_layers=12, num_classes=3)
    input = imd.ImportDataset() #
    model(input) #
