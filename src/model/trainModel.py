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
           edc, idc, smd, smdl = get_nested_data(value, edc, idc)
        stmd.extend(smd) if len(smd)> 1 and smd else stmd.append(smd)
        stmdl.extend(smdl) if len(smdl)> 1 and smdl else stmdl.append(smdl)
    return edc, imc, stmd, stmdl
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
    print(f"itos len after = {edc.itos}")
    print(f"stoi len after = {edc.stoi}")
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
    """
    from torch.utils.data import DataLoader
    
    # Create a loader
    train_loader = DataLoader(dataset=input, batch_size=32, shuffle=True)
    
    # Inside your training loop
    for batch in train_loader:
        # Assuming your dataset returns (features, labels)
        x_batch, y_batch = batch 
        # Pass the actual tensor to the model
        
        output = model(x_batch) 

    model(stmd) #
    best_val_loss = float('inf') # Start with infinity
    ##
    model.to(device)
    model.eval()
    for epoch in range(max_epoch):
        print(epoch)
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
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Save the model state
            torch.save(checkpoint, 'models/best_dictionary_model.pth')
            print(f"--> Saved new best model with Val Loss: {best_val_loss:.4f}")
        # Save progress
        torch.save(model.state_dict(), f"models/checkpoint_epoch_{epoch}.pth")
