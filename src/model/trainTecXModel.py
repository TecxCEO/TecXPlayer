"""
To prevent losing your progress or saving a "worse" version of the model due to overfitting, you should use a Checkpoint System. This script saves the model weights to a file only when the Validation Loss hits a new record low.
1. The Checkpoint Logic
Add this logic inside your training loop, specifically right after the Validation Phase:

    
"""
#import copy
from copy import deepcopy
########
import tecXModel as tm
import encoding_decoding as ed
import importDataset as imd
import torch
def get_nested_data(data, edc, idc):
    ####stoi, itos = 
    ##########edc.createTokens(data)
    edc.stoi, edc.itos = edc.createTokens(data["solution"], edc.stoi, edc.itos)
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
##
if __name__ == "__main__":
    filepath = "./data/dataset/cube3x3solvingdataset.json"
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    idc = imd.ImportDataset(filepath) #
    data=deepcopy(idc.data["solution"])
    edc=ed.EncodeDecode(data)
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
    ###########$$vocab_size =  edc.return_stoi_size # Size of your 'stoi' map
    ###########max_epoch = 1 # 11
    
    #####
    ####model = tm.DictionaryTransformer(vocab_size=int(vocab_size()), d_model=128, nhead=8, num_layers=4, num_classes=3)
    # model = DictionaryTransformer(vocab_size=vocab_size, d_model=128, nhead=20, num_layers=12, num_classes=3)

    
    ##############best_val_loss = float('inf') # Start with infinity
    ##
    ##model.to(device)
    ##model.eval()
    ##stmdl_in  = []
    #stmdlin  = []
    #stmdl_enc = []
    # stmdl_tensor = []
    stmdlenc = []
    stmdltensor = []

    for epoch in range(max_epoch):
        ####print(epoch)
        for i in range(len(stmdl) // 3 if stmdl else 0):
            ####print(f" Epoch no = {epoch}\n, Loop no = {i}\n")
            stmdlin  = []
            stmdlin  = ['<SOS>']
            stmdlin += stmdl[3*i]
            #stmdlin += [stmdl[3*i]]
            # stmdlin += stmdl[3*i]
            stmdlin += ['<'+stmdl[3*i+1]+'>']
            stmdlin += stmdl[3*i+2]
            #stmdlin += [stmdl[3*i+2]]
            stmdlin += ['<EOS>']
            #stmdlenc = edc.encode(stmdlin)
            stmdlenc += edc.encode(stmdlin)

    tm.TecXModelTrain(stmdlenc, databal, edc.stoi, edc.itos)
            ####stmdltensor = torch.tensor(stmdlenc, dtype=torch.long)
    
            """
            model(stmdltensor)
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
        torch.save(model.state_dict(), f"models/checkpoint47_epoch_{epoch}.pth")
    torch.save(checkpoint, 'models/best_dictionary_model3.pth')
    print(f"--> Saved new best model with Val Loss: {best_val_loss:.4f}")
    print(f" stmdl len = {len(stmdl)} stmdl len3 = {len(stmdl) // 3}")
    """
