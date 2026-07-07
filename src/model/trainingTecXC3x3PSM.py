
"""

Here is the complete, high-performance PyTorch script. It integrates Cosine Annealing Learning Rate Decay into the primary training run and establishes an Accuracy Performance Gate alongside raw loss checks during the post-optimization epochs.The system tracks metrics down to the individual token position and automatically exports analytical logs to a separate CSV file.python
# Save this entire script as: geometry_constrained_transformer.py

"""


import os
import csv
import json
import math
import torch
import torch.nn as nn
from torch.nn import functional as F 
import encodingdecoding as ed
import cube3x3dataset20steps18perellel_batch as cdspb
import importDataset as imd

edacvr = ed.AdvancedCustomVocabularyRegistry()
def fresh_data_generator():
      state_given_to_solve={
            "rgy":"rgy",
            "rgw":"rgw",
            "rby":"rby",
            "rbw":"rbw",
            "ogy":"ogy",
            "ogw":"ogw",
            "oby":"oby",
            "obw":"obw",
            "rb":"rb",
            "rg":"rg",
            "rw":"rw",
            "ry":"ry",
            "ob":"ob",
            "og":"og",
            "ow":"ow",
            "oy":"oy",
            "by":"by",
            "bw":"bw",
            "gw":"gw",
            "gy":"gy"
      }
      s=cdspb.Solver()
      edc = ed.EncodeDecode()
      edcacvr = edc.acvr
      ####full_response = []
      for char in s.solve(state_given_to_solve):
            ###sys.stdout.write(str(char) + " ")
            ### sys.stdout.flush()
            ### full_response += (" " + str(char)) # Collect for logging
            ####time.sleep(7)
            #time.sleep(0.01) 
            token_list = edc.encoder(char)
      ## print(f" full_response = {full_response}")
      # Pass your fresh data string into your tokenization function
      """
      data_stream = s.solve(state_given_to_solve)
      # while len(x_list) < BATCH_SIZE:
      while True:
            try:
                # Grab the next sequence yielded from your data file
                full_sequence = next(data_stream)
            except StopIteration:
                # If the generator runs out of entries, restart the stream file
                data_stream = s.solve(state_given_to_solve)
                full_sequence = next(data_stream)
                
            # Ensure the stream sequence satisfies your structural window constraints
            if len(full_sequence) > BLOCK_SIZE:
                x_list.append(full_sequence[:BLOCK_SIZE])
                y_list.append(full_sequence[1:1 + BLOCK_SIZE])
      """
      token_list = edc.encoder(char)
      
      # Convert the structural sequence directly into a PyTorch tensor
      yield torch.tensor(token_list, dtype=torch.long)
# =============================================================================
# 1. PARAMETERS & EXACT EXPLICIT VOCABULARY PROFILE
# =============================================================================

# BASE_VOCAB_SIZE = len(edacvr.string_to_id)
# ACTUAL_VOCAB_SIZE = BASE_VOCAB_SIZE
BASE_VOCAB_SIZE = len(edacvr.string_to_id) - 5 # Base structural tokens (0 to 89)
print(f" BASE_VOCAB_SIZE = {BASE_VOCAB_SIZE}")

ACTUAL_VOCAB_SIZE = BASE_VOCAB_SIZE + 5  # Exactly 95 total unique tokens
print(f" ACTUAL_VOCAB_SIZE = {ACTUAL_VOCAB_SIZE}")
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

"""
#

def fresh_data_generator(file_path="data/dataset/cube3x3solvingdataset.json"):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Could not find fresh data at {file_path}")
        
    print(f"[DATA SHUTTLE] Streaming fresh external dataset from: {file_path}")
    
    with open(file_path, 'r') as f:
        # Assuming your JSON file contains a list of entry items
        raw_entries = json.load(f) 
        
    # Process and stream entries one by one using yield
    for entry in raw_entries:
        # Pass your fresh data string into your tokenization function
        edc = ed.EncodeDecode(entry['solution'])
        token_list = edc.createTokens(entry["solution"])
        
        # Convert the structural sequence directly into a PyTorch tensor
        yield torch.tensor(token_list, dtype=torch.long)
        
#
"""



"""
#
class ChunkedDataStreamer:
    def __init__(self):
        self.current_chunk_id = 0
        # Initialize your fresh data stream generator here
        self.data_stream = fresh_data_generator()

    def load_next_chunk(self):
        self.current_chunk_id += 1
        print(f"\n[STREAM ENGINE] Loading Data Chunk #{self.current_chunk_id}...")
        return FiniteChunkDataset(total_steps=99720)

    def get_batch(self, dataset_iterator, device='cuda'):
        x_list = []
        y_list = []
        
        # Build your training batch using the yield stream
        while len(x_list) < BATCH_SIZE:
            try:
                # Grab the next sequence yielded from your data file
                full_sequence = next(self.data_stream)
            except StopIteration:
                # If the generator runs out of entries, restart the stream file
                self.data_stream = fresh_data_generator()
                full_sequence = next(self.data_stream)
                
            # Ensure the stream sequence satisfies your structural window constraints
            if len(full_sequence) > BLOCK_SIZE:
                x_list.append(full_sequence[:BLOCK_SIZE])
                y_list.append(full_sequence[1:1 + BLOCK_SIZE])
                
        # Stack individual sequences into matrix blocks
        x = torch.stack(x_list).to(device)
        y = torch.stack(y_list).to(device)
        return x, y

    def get_batch_simulated(self, batch_size=BATCH_SIZE, device='cuda'):
        # Keep simulated training synchronized with your fresh yield data loop
        return self.get_batch(None, device=device)
        
#
"""
######
######
def get_nested_data(data, edc, idc):
    ####stoi, itos = 
    ##########edc.createTokens(data)
    #edc.stoi, edc.itos = edc.createTokens(data["solution"], edc.stoi, edc.itos)
    edc.stoi, edc.itos = edc.createTokens(data["solution"], edc.stoi, edc.itos) if data.get("solution") else  edc.createTokens(data, edc.stoi, edc.itos)
    """
    if not stmd and not smd and not smdl and not stmdl:
        stmd = None
        smd = None
        smdl = None
        stmdl = None
    """
    stmd = None
    smd = None
    smdl = None
    stmdl = None
    st_mv_data = []
    if data.get('state') and len(data) in (16, 19):
        st_mv_data += idc.createInputString(data)
        #print(f" st_mv_data len = {len(st_mv_data)}\n st_mv_data = {st_mv_data}")
        if st_mv_data:
            if stmd:
                stmd += st_mv_data
            else:
                stmd = st_mv_data
            print(f" st_mv_data len = {len(st_mv_data)}\n")
            print(f" stmd len = {len(stmd)}\n")
            print(f" stmd = {stmd}\n")
            for smdt in st_mv_data:
                #stoi, itos = 
                #########edc.createTokens(smdt[0]) #####
                edc.stoi, edc.itos = edc.createTokens(smdt[0], edc.stoi, edc.itos)
                #stoi, itos = 
                #####edc.createTokens(smdt[2]) ####
                edc.stoi, edc.itos = edc.createTokens(smdt[2], edc.stoi, edc.itos)
                #st_mv_data_list += idc.convertStateToList(smdt[0], smdt[1], smdt[2])
                st_mv_data_list = idc.convertStateToList(smdt[0], smdt[1], smdt[2])
                if stmdl:
                    stmdl += st_mv_data_list
                else:
                    stmdl = st_mv_data_list
    ####print(f" starting for loop \n stmd = {stmd}\n \n stmdl = {stmdl}\n " )
    #i=0
    for key, value in data.items():
        if key != 'state' and len(value) in (19, 16):
            edc, idc, smd, smdl =  get_nested_data(value, edc, idc)
            if smd and len(smd) > 1:
                stmd.extend(smd)
            elif smd and len(smd) == 1:
                stmd += smd
            if smdl and len(smdl) > 1:
                stmdl.extend(smdl)
            elif smdl and len(smdl) == 1:
                stmdl += smdl
        #i += 1
        ########print(f" Ending for loop, i = {i}, stmd = {stmd}, \n stmdl = {stmdl}\n " )
    return edc, idc, stmd, stmdl
#####
def createTVData(file, edctv = None, idctv = None):
    ########filepath = "./data/dataset/cube3x3solvingdataset.json"
    filepath = file
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"idctv \n")
    #if idctv not None: 
    if idctv:
        print(f"idctv in if statement  \n")
        idc = idctv
        print(f" idc idctv = {idc}")
        ##idc(filepath)#
        idc.load_data(filepath)
    else:
        print(f"imd in else statement.\n")
        idc = imd.ImportDataset(filepath) 
    print(f" data deep copy. \n")
    data=deepcopy(idc.data["solution"])
    if edctv:
        print(f"edctv in if statement \n")
        edc = edctv
        ###edc(data)
    else:
        print(f"ed in else statement \n")
        #ed.EncodeDecode(data)
        edc = ed.EncodeDecode(data)
    print(f"stoi before = {edc.stoi}")
    print(f"itos before = {edc.itos}")
    print(f"stoi len before = {len(edc.stoi)}")
    print(f"itos len before = {len(edc.itos)}")
    edc, idc, stmd, stmdl = get_nested_data(idc.data["solution"], edc, idc)
    print(f"itos after = {edc.itos}")
    print(f"stoi after = {edc.stoi}")
    print(f"itos len after = {len(edc.itos)}")
    print(f"stoi len after = {len(edc.stoi)}")
    print(f"stmd len = {len(stmd)}\n")
    print(f"stmdl len = {len(stmdl)}\n")
    stmdlenc = []
    stmdltensor = []
    #for epoch in range(max_epoch):
        ###print(epoch)
    for i in range(len(stmdl) // 3 if stmdl else 0):
        ####
        print(f" Loop no = {i}\n")
        stmdlin  = []
        ####stmdlin  = ['<SOS>']
        stmdlin += stmdl[3*i]
        #stmdlin += [stmdl[3*i]]
        # stmdlin += stmdl[3*i]
        stmdlin += ['<'+stmdl[3*i+1]+'>']
        stmdlin += stmdl[3*i+2]
        #stmdlin += [stmdl[3*i+2]]
        ####stmdlin += ['<EOS>']
        print(f" stmdlin = {stmdlin}")
        print(f" stmdlin len = {len(stmdlin)}")
        #stmdlenc = edc.encode(stmdlin)
        ####stmdlenc += edc.encode(stmdlin)
        stmdlenc += [edc.encode(stmdlin)] ####
        print(f"stmdlenc len= {len(stmdlenc[-1])}")
    return stmdlenc, edc, idc
######
######
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
    # def __init__(self, total_steps=100000):
    def __init__(self, total_steps=99720):
        self.total_steps = total_steps
        self.current_step = 0

    def __iter__(self):
        self.current_step = 0
        return self

    def __next__(self):
        if self.current_step >= self.total_steps:
            raise StopIteration
        self.current_step += 1
        return self.current_step
"""
#
class ChunkedDataStreamer:
    def __init__(self):
        self.current_chunk_id = 0
        # Initialize your fresh data stream generator here
        self.data_stream = fresh_data_generator()

    def load_next_chunk(self):
        self.current_chunk_id += 1
        print(f"\n[STREAM ENGINE] Loading Data Chunk #{self.current_chunk_id}...")
        return FiniteChunkDataset(total_steps=99720)

    def get_batch(self, dataset_iterator, device='cuda'):
        x_list = []
        y_list = []
        
        # Build your training batch using the yield stream
        while len(x_list) < BATCH_SIZE:
            try:
                # Grab the next sequence yielded from your data file
                full_sequence = next(self.data_stream)
            except StopIteration:
                # If the generator runs out of entries, restart the stream file
                self.data_stream = fresh_data_generator()
                full_sequence = next(self.data_stream)
                
            # Ensure the stream sequence satisfies your structural window constraints
            if len(full_sequence) > BLOCK_SIZE:
                x_list.append(full_sequence[:BLOCK_SIZE])
                y_list.append(full_sequence[1:1 + BLOCK_SIZE])
                
        # Stack individual sequences into matrix blocks
        x = torch.stack(x_list).to(device)
        y = torch.stack(y_list).to(device)
        return x, y

    def get_batch_simulated(self, batch_size=BATCH_SIZE, device='cuda'):
        # Keep simulated training synchronized with your fresh yield data loop
        return self.get_batch(None, device=device)
        
#
"""
class ChunkedDataStreamer:
    def __init__(self):
        self.current_chunk_id = 0
    def load_next_chunk(self):
        self.current_chunk_id += 1
        print(f"\n[STREAM ENGINE] Loading Incoming Data Chunk #{self.current_chunk_id} into Pipeline Context...")
        # return FiniteChunkDataset(total_steps=100000)
        return FiniteChunkDataset(total_steps=99720)
    def get_batch(self, dataset_iterator, device='cuda'):
        _ = next(dataset_iterator)
        x = torch.randint(0, ACTUAL_VOCAB_SIZE, (BATCH_SIZE, BLOCK_SIZE), device=device)
        y = torch.randint(0, ACTUAL_VOCAB_SIZE, (BATCH_SIZE, BLOCK_SIZE), device=device)
        return x, y
    #def get_batch_simulated(self, batch_size=BATCH_SIZE, device='cuda'):
        #x = torch.randint(0, ACTUAL_VOCAB_SIZE, (batch_size, BLOCK_SIZE), device=device)
        #y = torch.randint(0, ACTUAL_VOCAB_SIZE, (batch_size, BLOCK_SIZE), device=device)
        #return x, y
    def get_batch_simulated(self, batch_size=BATCH_SIZE, device='cuda'):
        x = torch.randint(0, ACTUAL_VOCAB_SIZE, (BATCH_SIZE, BLOCK_SIZE), device=device)
        y = torch.randint(0, ACTUAL_VOCAB_SIZE, (BATCH_SIZE, BLOCK_SIZE), device=device)
        # === ADD THESE LINES TO PRINT THE DATA ===
        print("\n--- CURRENT GENERATED TRAINING DATA BATCH ---")
        print("Input Data (X) Shape:", x.shape)
        print("Input Data (X) Sample Tensors:\n", x[:2])  # Prints the first 2 rows of the batch
        print("Target Data (Y) Sample Tensors:\n", y[:2]) # Prints the corresponding targets
        print("---------------------------------------------\n")
        return x, y
    # =========================================

    
    
#=============================================================================
# 5. LIFELONG LOOP CONTROLLER WITH ENFORCED COMPLETION CHECKS
# =============================================================================
def execute_lifelong_training():
    # Here there will many chunks, and each chunk has 277 sequences, and each sequence have 18 batch, and each batch have 20 steps
    # chunk = 99720 steps = 277*18*20 steps.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"System Online. Targeted Core Execution Hardware: {device.upper()}")
    model = CustomTransformer()
    model.to(device)
    data_streamer = ChunkedDataStreamer()
    logger = MetricLogger(CSV_LOG_FILE)
    print(f" logger {logger}")
    base_lr = 6e-4
    min_lr = 6e-5
    weight_decay = 0.1
    # For start training last time train model state
    # model_path = 'models/tecx/tecx_cube_solver_model_final.pth'
    model_path = 'models/tecx/tecx_cube_solver_last_model.pth'
    prev_checkpoint_path = None
    while True:
        checkpoint = {}
        ph1_cpc_id = 0
        ph2_cpc_id = 0
        #--- PHASE 1: INITIAL CHUNK PASSTHROUGH WITH COSINE SCHEDULER ---
        # dataset_chunk = data_streamer.load_next_chunk()
        # chunk_id = data_streamer.current_chunk_id
        # best_loss_this_chunk = float('inf')
        if os.path.exists(prev_checkpoint_path):
              # checkpoint2 = {}
              checkpoint2 = torch.load(prev_checkpoint_path)
              ph2_cpc_id = checkpoint2['chunk_id']
              data_streamer.current_chunk_id = ph2_cpc_id
              if locals().get("checkpoint2") is not None:
                    model_dict = checkpoint2["model_state_dict"]
                    ####optimizer_dict = checkpoint['optimizer_state_dict']
                    edacvr.string_to_id = checkpoint2["string_to_id"]
                    edacvr.vocab_map = checkpoint2["vocab_map"]
                    model.load_state_dict(model_dict, strict=False)
        if os.path.exists(model_path):
              # checkpoint1 = {}
              checkpoint1 = torch.load(model_path)
              ph1_cpc_id = checkpoint1['chunk_id']
              # model = tm.TecXModel(vocab_size=int(len(checkpoint["string_to_id"])))
              if ph1_cpc_id > ph2_cpc_id:
                    if locals().get("checkpoint1") is not None:
                          data_streamer.current_chunk_id = ph1_cpc_id
                          model_dict = checkpoint1["model_state_dict"]
                          ####optimizer_dict = checkpoint['optimizer_state_dict']
                          edacvr.string_to_id = checkpoint1["string_to_id"]
                          edacvr.vocab_map = checkpoint1["vocab_map"]
                          model.load_state_dict(model_dict, strict=False)
        #--- PHASE 1: INITIAL CHUNK PASSTHROUGH WITH COSINE SCHEDULER ---
        dataset_chunk = data_streamer.load_next_chunk() if ph1_cpc_id == ph2_cpc_id else None
        # dataset_chunk = data_streamer.load_next_chunk()
        chunk_id = data_streamer.current_chunk_id
        best_loss_this_chunk = float('inf')
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, betas=(0.9, 0.95), weight_decay=weight_decay)
        chunk_iterator = iter(dataset_chunk)
        print(f"--> [PHASE 1] Training Loop Active for Chunk {chunk_id} (Enforcing Cosine LR Decay)")
        model.train()
        while True:
            break if ph1_cpc_id > ph2_cpc_id and ph1_cpc_id >0 else None
            try:
                X, Y = data_streamer.get_batch(chunk_iterator, device=device)
                current_step = dataset_chunk.current_step
            except StopIteration:
                print(f"\n[DATA BOUNDARY] All input items for Chunk {chunk_id} processed cleanly.")
                break
            # --- FEATURE 1: COSINE ANNEALING LEARNING RATE DECAY --- 
            # Gradually steps down the learning rate across the 1,00,000 progress markers
            progress = current_step / dataset_chunk.total_steps
            cosine_lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
            for param_group in optimizer.param_groups:
                param_group['lr'] = cosine_lr
                logits, loss = model(X, Y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if current_step % 1000 == 0:
                    in_acc, out_acc = compute_structural_zone_accuracy(model, data_streamer, device)
                    print(f"Chunk {chunk_id} | Step {current_step}/{dataset_chunk.total_steps} | Loss: {loss.item():.4f} | Input Acc: {in_acc:.2%} | LR: {cosine_lr:.6f}")
                    logger.log_step(chunk_id, 0, current_step, "Phase1", loss.item(), in_acc, out_acc, cosine_lr)
                if loss.item() < best_loss_this_chunk:
                    best_loss_this_chunk = loss.item()
                    torch.save({'chunk_id': chunk_id,'step': current_step,'model_state_dict': model.state_dict(),'loss': best_loss_this_chunk,}, f"checkpoint_chunk_{chunk_id}_phase1.pt")
                if locals().get("checkpoint") :
                          # checkpoint['chunk_id'] = chunk_id
                          checkpoint.update({'chunk_id' : chunk_id })
                          checkpoint.update({'model_state_dict': model.state_dict()})
                          checkpoint.update({'optimizer_state_dict': optimizer.state_dict()})
                          checkpoint.update({'best_val_loss': best_val_loss})
                          checkpoint.update({'string_to_id': edacvr.string_to_id}) # Saving the vocabulary is critical!
                          checkpoint.update({'vocab_map' : edacvr.vocab_map})
                    elif locals().get("checkpoint") is None:
                          checkpoint = {
                                'chunk_id' : chunk_id,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'best_val_loss': best_val_loss,
                                'string_to_id': edacvr.string_to_id,
                                'vocab_map' : edacvr.vocab_map
                          }   
                torch.save(checkpoint, model_path) # 
        # --- PHASE 2: CONVERGENCE LOOP WITH ACCURACY PERFORMANCE GATE ---
        print(f"--> [PHASE 2] Launching Fine-Tuning Optimization Epochs for Chunk {chunk_id}...")
        # Lock fine-tuning updates to 10% of the maximum engine scale
        for param_group in optimizer.param_groups:
            param_group['lr'] = base_lr * 0.1
            no_improvement_counter = 0
            epoch_count = 0
            while True:
                epoch_count += 1
                post_step = 0
                chunk_iterator = iter(data_streamer.load_next_chunk())
                print(f"    [EPOCH PASSTHROUGH] Starting Pass #{epoch_count} for Structural Parameter Optimization...")
                while True:
                    try:
                        X, Y = data_streamer.get_batch(chunk_iterator, device=device)
                        post_step += 1
                    except StopIteration:
                        break
                    logits, loss = model(X, Y)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    current_loss = loss.item()
                    ######
                    if locals().get("checkpoint") :
                          checkpoint.update({'chunk_id' : chunk_id })
                          checkpoint.update({'epoch' = epoch_count })
                          checkpoint.update({'model_state_dict': model.state_dict()})
                          checkpoint.update({'optimizer_state_dict': optimizer.state_dict()})
                          checkpoint.update({'best_val_loss': best_val_loss})
                          checkpoint.update({'string_to_id': edacvr.string_to_id}) # Saving the vocabulary is critical!
                          checkpoint.update({'vocab_map' : edacvr.vocab_map})
                    elif locals().get("checkpoint") is None:
                          checkpoint = {
                                'chunk_id' : chunk_id,
                                'epoch': epoch_count,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'best_val_loss': best_val_loss,
                                'string_to_id': edacvr.string_to_id,
                                'vocab_map' : edacvr.vocab_map
                          }
                          print(f" checkpoint before save = {checkpoint}")
                    # torch.save(checkpoint, 'models/tecx/tecx_best_model.pth')
                    # Save progress
                    # 1. Define the path for the current epoch
                    current_checkpoint_path = f"models/tecx/tecx_model_checkpoint_chunk_{chunk_id}_phase2_epoch_{epoch}.pth"
                    # 2. Save the new model checkpoint
                    torch.save(checkpoint, current_checkpoint_path)
                    print(f"Saved: {current_checkpoint_path}")
                    # 3. Delete the previous epoch's file if it exists
                    if prev_checkpoint_path and os.path.exists(prev_checkpoint_path):
                          os.remove(prev_checkpoint_path)
                          print(f"Deleted previous checkpoint: {prev_checkpoint_path}")
                    # 4. Update the tracker to point to the current file for the next iteration
                    prev_checkpoint_path = current_checkpoint_path
                    #########
                    # Evaluate both structural layout constraints and loss floor changes
                    if current_loss < (best_loss_this_chunk - 1e-4):
                        # --- FEATURE 2: ACCURACY PERFORMANCE GATE ---
                        # Checks accuracy within targeted geometric zones before saving weights
                        in_acc, out_acc = compute_structural_zone_accuracy(model, data_streamer, device)
                        if in_acc >= ACCURACY_GATE_MIN and out_acc >= ACCURACY_GATE_MIN:
                            print(f"        [GATE PASSED] Epoch {epoch_count} Step {post_step} | Loss: {current_loss:.6f} | Acc: {in_acc:.2%}. Saving.")
                            best_loss_this_chunk = current_loss
                            no_improvement_counter = 0
                            torch.save({'chunk_id': chunk_id,'epoch': epoch_count,'model_state_dict': model.state_dict(),'loss': best_loss_this_chunk,'input_accuracy': in_acc,}, f"best_final_optimized_model_chunk_{chunk_id}.pt")
                        else:
                            # Increases patience counter if zone accuracy drops below the threshold
                            no_improvement_counter += 1
                    else:
                        no_improvement_counter += 1
                        if post_step % 5000 == 0:
                            in_acc, out_acc = compute_structural_zone_accuracy(model, data_streamer, device)
                            print(f"        [MONITOR] Epoch {epoch_count} Step {post_step} | Loss: {current_loss:.4f} | Patience: {no_improvement_counter}/{PATIENCE_STEPS}")
                            logger.log_step(chunk_id, epoch_count, post_step, "Phase2", current_loss, in_acc, out_acc, base_lr * 0.1)
                        if no_improvement_counter >= PATIENCE_STEPS:
                            break
                if no_improvement_counter >= PATIENCE_STEPS:
                      print(f"\n[SATURATION REACHED] No further improvement found over {PATIENCE_STEPS} steps.")
                      print(f"Optimal model for Chunk {chunk_id} saved as: 'best_final_optimized_model_chunk_{chunk_id}.pt'")
                      print("================================================================================")
                      break
                          
if __name__ == "__main__":
    print(f" Code is started")
    """
    edc = None
    idc = None
    model = None
    checkpoint = None
    datatraining = None
    dataval = None
    t = 0
    v = 0
    # For start training last time train model state
    model_path = 'models/tecx/tecx_cube_solver_model_final.pth'
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        model = tm.TecXModel(vocab_size=int(len(checkpoint["itos"])))
        if locals().get("checkpoint") is not None:
            model_dict = checkpoint["model_state_dict"]
            #optimizer_dict = checkpoint['optimizer_state_dict']
            #self.stoi = checkpoint["stoi"]
            #self.itos = checkpoint["itos"]
            model.load_state_dict(model_dict, strict=False)
            # Ensure your model is in evaluation mode
        tmt = tm.TecXModelTrain(datatraining, edc.stoi, edc.itos, dataval)
        model, checkpoint = tmt.trainModel(model, checkpoint)
        #####tmt = tmtbm.TecXModelTrain(datatraining, edc.stoi, edc.itos, dataval)
        ####model, checkpoint = tmt.trainModel(model, checkpoint)
    torch.save(checkpoint, model_path)
    print(f"--> Saved new final model with name as tecx_cube_solver_model_final.")
    # Save progress
    ##torch.save(checkpoint, f"models/tecx/checkpoint_epoch_{epoch}_{iter}.pth")
"""
    execute_lifelong_training()
