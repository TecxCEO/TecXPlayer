from tpsm import DictionaryTransformer
import datetime
# import tpsm as tm
#from encoding_decoding import EncodeDecode as edc
import encoding_decoding as ed
import importDataset as imd
import sys
import time
import torch




device = 'cuda' if torch.cuda.is_available() else 'cpu'
edc = ed.EncodeDecode()
model_path = "models/tecx/tecx_model_epoch_5.pth"
#####model_path ='models/tecx/tecx_best_model.pth'
#####model_path ='models/tecx/tecx_cube_solver_model_final.pth'
checkpoint = torch.load(model_path)
print(checkpoint)

# Load the file
# checkpoint = torch.load('your_checkpoint.pth')
# 1. Count top-level keys
print(f"Top-level keys: {len(checkpoint)}\n")
print(f"{checkpoint.keys()}\n")
# 2. Count model parameters (the actual weights)
# Replace 'model_state_dict' with the key used in your save file
if 'model_state_dict' in checkpoint:
    num_params = len(checkpoint['model_state_dict'])
    print(f"Total model parameter keys: {num_params}")
# checkpoint = torch.load("/data/data/com.termux/files/home/TecXPlayer/src/model/your_file.pth")

print("Keys in this checkpoint:")
for key in checkpoint.keys():
    print(f"key of checkpoint= {key}\n")
    
    
# model_dict = checkpoint["state_dict"]
model_dict = checkpoint["model_state_dict"]
print(f"checkpoint[stoi] = {checkpoint["stoi"]}\n")
edc.stoi = checkpoint["stoi"]
print(f"edc.stoi = {edc.stoi}\n")
edc.itos = checkpoint["itos"]
print(f"edc.itos= {edc.itos}\n")
#vocab_size =  edc.return_stoi_size() # Size of your 'stoi' map
model = DictionaryTransformer(vocab_size=int(len(edc.stoi)), d_model=128, nhead=8, num_layers=4, num_classes=3)
###model = DictionaryTransformer()

#model = TecXModel(vocab_size=71)
#model.load_state_dict(torch.load(model_path))

##model_dict = model.state_dict()
# Filter out layers with wrong shapes (like lm_head)
####pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and v.size() == model_dict[k].size()}
#####model_dict.update(pretrained_dict)
model.load_state_dict(model_dict, strict=False)
#model.to(device)
# Ensure your model is in evaluation mode
model.eval()
print(f"tecxmodelgen created")
m = model.to(device)
m.eval()
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
########
state_given_to_solve={
      "rgy":"ogw",
      "rgw":"ybo",
      "rby":"ryg",
      "rbw":"bwr",
      "ogy":"yrb",
      "ogw":"oyg",
      "oby":"owb",
      "obw":"wrg",
      "rb":"gy",
      "rg":"rw",
      "rw":"yr",
      "ry":"by",
      "ob":"gw",
      "og":"bw",
      "ow":"oy",
      "oy":"ow",
      "by":"go",
      "bw":"rb",
      "gw":"ob",
      "gy":"gr"
    }
#keys() in  state_given_to_solve:
print(f"Keys in state_given_to_solve = {state_given_to_solve.keys()}\n")
########
print(f"Value in state_given_to_solve = {state_given_to_solve.values()}\n")
# ... inside your 'while True' loop ...
while True:
    # 1. Get custom text from the user
    #user_input = input("\nEnter your starting text (or type 'exit' to quit): ")
    user_prompt=[input("\nEnter your starting text (or type 'exit' to quit): ")]
    print(f" user_prompt = {user_prompt}")
    print(f" user_prompt len = {len(user_prompt)}")
    if user_prompt.lower() == 'exit':
        break
    # Encode and setup context
    #context_list = [stoi[c] for c in user_prompt if c in stoi]
    #context = torch.tensor([context_list], dtype=torch.long, device=device)
    context = torch.tensor([edc.encoder(user_prompt)], dtype=torch.long).to(device)
    # context = torch.tensor([edc.encode(user_prompt)], dtype=torch.long).to(device)
    #data = torch.tensor(encode(user_prompt), dtype=torch.long)
    print(f"\nTecX Player Solving Puzzle: ", end="")
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
            ##char = decode(token_id)
            char = edc.decode([token_id])
            sys.stdout.write(char+" ")
            sys.stdout.flush()
            full_response += char # Collect for logging
            # Optional: Add a tiny sleep to make it look like "typing"
            time.sleep(0.01) 
    print("\n" + "-"*30)
    # Automatically save the conversation
    log_conversation(user_prompt, full_response)
    print("\n\n(Conversation saved to generation_logs.txt)")
