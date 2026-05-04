"""
To prevent losing your progress or saving a "worse" version of the model due to overfitting, you should use a Checkpoint System. This script saves the model weights to a file only when the Validation Loss hits a new record low.
1. The Checkpoint Logic
Add this logic inside your training loop, specifically right after the Validation Phase:

    
"""

import tpsm as tm
# from encoding_decoding import EncodeDecode as ed
import encoding_decoding as ed
import importDataset as imd
# import import_dataset as imd
import torch
def get_nested_data(data, edc, idc):
    stoi, itos = edc.createTokens(data)
    stmd = None
    smd = None
    smdl = None
    stmdl = None
    for key, value in data.items():
        #####
        st_mv_data = []
        
        st_mv_data += idc.createInputString(data["solution"])
        
        ##### stmdt = edc.encoder(st_mv_data) ####
        
        #### dict2.update(dict1) 
        if stmd:
            stmd += st_mv_data
        else:
            stmd = st_mv_data
        for smd in st_mv_data:
            st_mv_data_list = idc.convertStateToList(smd[0], smd[1], smd[2])
            if stmdl:
                stmdl += st_mv_data_list
            else:
                stmdl = st_mv_data_list
        if key != 'state' and len(value) == (19, 16):
           # edc, idc, smd, smdl = get_nested_data(value, edc, idc)
           return get_nested_data(value, edc, idc)
            
        # stmd.extend(smd) if smd and len(smd)> 1 else stmd.append(smd)
        if smd and len(smd) > 1:
            stmd.extend(smd)
        elif len(smd) == 1:
            stmd += smd
        # stmdl.extend(smdl) if smdl and len(smdl)> 1 else stmdl.append(smdl)
        if smdl and len(smdl) > 1:
            stmdl.extend(smdl)
        elif smdl and len(smdl) == 1:
            stmdl += smdl
    
    return edc, idc, stmd, stmdl
##
if __name__ == "__main__":
    filepath = "./data/dataset/cube3x3solvingdataset.json"
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    idc = imd.ImportDataset(filepath) #
    edc=ed.EncodeDecode(input)
    ## stoi, itos = edc.createTokens()
    print(f"stoi len before = {edc.stoi}")
    print(f"itos len before = {edc.itos}")
    edc, idc, stmd, stmdl = get_nested_data(idc.data, edc, idc) ####
    print(f"itos len after = {len(edc.itos)}")

    print(f"itos len after = {edc.itos}")
    print(f"stoi len after = {edc.stoi}")
    
    print(f"stoi len after = {len(edc.stoi)}")
    # print(f"stmd = {stmd[i]}\n") for i in range(len(stmd))
    # print(f"stmdl = {stmdl[i]}\n") for i in range(len(stmd))
    """
    for i in range(len(stmd)):
        print(f"stmd[{i}] = {stmd[i]}\n")
    for i in range(len(stmdl)):
        print(f"stmdl[{i}] = {stmdl[i]}\n")
    #"" " 
    #########
    def check_list_recursive(data, path="stmd"):
      for i, item in enumerate(data):
        current_path = f"{path}[{i}]"
        
        if isinstance(item, (list, tuple)):
            # If it's a list/tuple, dig deeper
            check_list_recursive(item, current_path)
        elif isinstance(item, dict):
            print(f"❌ ERROR at {current_path}: Found a DICT -> {item}")
        elif isinstance(item, str):
            print(f"❌ ERROR at {current_path}: Found a STRING -> '{item}'")
        else:
            print(f"✅ OK at {current_path}: {item}")

    # Call it like this:
    check_list_recursive(stmd)

    #####
    """
    # Setup Example
    # vocab_size = 80  # Size of your 'stoi' map
    #vocab_size =  len(checkpoint[ 'stoi']) # Size of your 'stoi' map
    vocab_size =  edc.return_stoi_size # Size of your 'stoi' map
    max_epoch = 11
    model = tm.DictionaryTransformer(vocab_size=int(vocab_size()), d_model=128, nhead=8, num_layers=4, num_classes=3)
    # model = DictionaryTransformer(vocab_size=vocab_size, d_model=128, nhead=20, num_layers=12, num_classes=3)
    """
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        ##'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
        'stoi': ed.createTokens() # Saving the vocabulary is critical!
        #'stoi': stoi # Saving the vocabulary is critical!
    }
    #"" "
    ########
    
    if isinstance(smdl, dict):
        # Take only the numbers from the dictionary
        stmdl.extend(smdl.values())
    elif isinstance(smdl, list) or isinstance(smdl, tuple):
        stmdl.extend(smdl)
    else:
        # If it's just a single number
        stmdl.append(smdl)
    
    from torch.utils.data import DataLoader


    ###########
    #"" "
    # Create a loader
    ############ train_loader = DataLoader(dataset=input, batch_size=32, shuffle=True)
    
    # Inside your training loop
    for batch in train_loader:
        # Assuming your dataset returns (features, labels)
        x_batch, y_batch = batch 
        # Pass the actual tensor to the model
        
        output = model(x_batch) 
    
    #######

    
    # Check if it's a dictionary first
    if isinstance(smdl, dict):
        # smdl.values() extracts just the numbers (e.g., 121, 122, 123)
        stmdl.extend(smdl.values())
    else:
        # Use your existing logic for non-dictionary items
        stmdl += smdl if (smdl and len(smdl) > 1) else [smdl]
    
    
    #######
    #" ""
    def describe_elements(data, level=0):
        for index, item in enumerate(data):
            indent = "  " * level  # Adds spacing for visual nesting
            
            if isinstance(item, list):
                print(f"{indent}Index {index}: This is a NESTED LIST. Digging deeper...")
                describe_elements(item, level + 1) # The "Magic": calls itself
            else:
                print(f"{indent}Index {index}: Element is '{item}'. Why? Because it is a flat value.")
    # Example with your 8-element concept
    # my_list = [1, 2, 3, [4, 5, 6], 7, 8]
    describe_elements(stmdl)
    """
    best_val_loss = float('inf') # Start with infinity
    ##
    model.to(device)
    model.eval()
    stmdl_in  = []
    stmdlin  = []
    stmdl_enc = []
    # stmdl_tensor = []
    """
    for i in range(len(stmdl) // 3):
        stmdlin += stmdl[3*i] 
        stmdlin += ('<'+stmdl[3*i+1]+'>')
        stmdlin += stmdl[3*i+2]
        stmdl_in += stmdl_in
        # stmdl_enc += edc.stoi(stmdl_in)
        # Change this:
        # stmdl_enc += edc.stoi(stmdl_in)
        # To this:
        ###### stmdl_enc.append(edc.stoi[stmdl_in])
        stmdl_enc.append(edc.encode(stmdl_in))
        # Convert your list to a tensor
        ########stmdl_tensor += torch.tensor(stmdl_enc)
        ##### stmdl_tensor = torch.tensor(stmdl_enc)
        #######stmdl_tensor = torch.tensor(stmdl_enc, dtype=torch.long)
        # Pass the tensor to your model
        #####
        # model(stmdl_tensor)
    ##
    
    stmdl_tensor = torch.tensor(stmdl_enc, dtype=torch.long)
    # Pass the tensor to your model
    ###model(stmdl_tensor)
    # model(stmd) #
    """
    for epoch in range(max_epoch):
        print(epoch)
        for i in range(len(stmdl) // 3):
            print(f" Epoch no = {epoch}\n, Loop no = {i}\n")
            stmdlin  = []
            stmdlin  = ['<SOS>']
            #stmdlin += stmdl[3*i]
            stmdlin += [stmdl[3*i]]
            # stmdlin += stmdl[3*i]
            stmdlin += ['<'+stmdl[3*i+1]+'>']
            #stmdlin += stmdl[3*i+2]
            stmdlin += [stmdl[3*i+2]]
            stmdlin  = ['<EOS>']
            # stmdlin += stmdl[3*i+2]
            # Prepend '<sos>' and append '<eos>' to every sentence
            ##tagged_data = ['<sos> ' + x + ' <eos>' for x in stmdlin]
            print(f" stmdlin = {stmdlin}")
            stmdl_in += stmdlin
            print(f" stmdl_in = {stmdl_in}")
            stmdl_enc.append(edc.encode(stmdlin))
            #stmdl_enc.append(edc.encode(stmdl_in))
            print(f" stmdl_enc = {stmdl_enc}")
            stmdl_tensor = torch.tensor(stmdl_enc, dtype=torch.long)
            print(f" stmdl_tensor = {stmdl_tensor}")
            model(stmdl_tensor)
        """
        # 1. Initialize the counter BEFORE the loop
        total_val_loss = 0.0  

        # 2. Your validation loop
        for batch in val_dataloader:
            outputs = model(batch)
            loss = criterion(outputs, targets)
            # Add the current loss to the total
            total_val_loss += loss.item()
        # 3. NOW you can calculate the average
        avg_val_loss = total_val_loss / len(val_dataloader)
        """
        # model(stmdl_tensor)
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            ##'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            #'stoi': ed.createTokens() # Saving the vocabulary is critical!
            'stoi': edc.stoi, # Saving the vocabulary is critical!
            'itos' : edc.itos
        }
    
        # Inside your epoch loop, after calculating avg_val_loss:
        # 1. Calculate the average loss (Total Loss / Number of Batches)
        ####avg_val_loss = total_val_loss / len(dataloader)
        #avg_val_loss = total_val_loss
        # 2. Now you can check if it's the best one
        ########if avg_val_loss < best_val_loss:
            ######best_val_loss = avg_val_loss
            # Save the model state
            ######torch.save(checkpoint, 'models/best_dictionary_model.pth')
            ######print(f"--> Saved new best model with Val Loss: {best_val_loss:.4f}")
        # Save progress
        torch.save(model.state_dict(), f"models/checkpoint4_epoch_{epoch}.pth")
    torch.save(checkpoint, 'models/best_dictionary_model2.pth')
    print(f"--> Saved new best model with Val Loss: {best_val_loss:.4f}")
