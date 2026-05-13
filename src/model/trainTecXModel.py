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
#####if __name__ == "__main__":
def createTVData(file, edctv = None, idctv = None):
    ########filepath = "./data/dataset/cube3x3solvingdataset.json"
    filepath = file
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"idctv \n")
    #if idctv not None: 
    if idctv:
        print(f"idctv in if statement  \n")
        idc = idctv
        idc(filepath)#
    else:
        print(f"imd in else statement.\n")
        idc = imd.ImportDataset(filepath) 
    print(f" data deep copy. \n")
    data=deepcopy(idc.data["solution"])
    if edctv:
        print(f"edctv in if statement \n")
        edc = edctv
        edc(data)
    else:
        print(f"ed in else statement \n")
        ed.EncodeDecode(data)
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
    for epoch in range(max_epoch):
        ####print(epoch)
        for i in range(len(stmdl) // 3 if stmdl else 0):
            ####
            print(f" Epoch no = {epoch}\n, Loop no = {i}\n")
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
    return stmdlenc, edc, idc
def file_dir_nested(dir = "./data/dataset", d_list = []):
    #dir = "./data/dataset"
    dir_list = 
    t_filepath = f"{dir}/cube3x3trainingdataset.json"
    v_filepath = f"{dir}/cube3x3solvingdataset.json"
    dir_list += file_dir_nested(dir, dir_list)
    return dir_list
        print(f"Started \n")
        dir = "./data/dataset"
        t_filepath = f"{dir}/cube3x3trainingdataset.json"
        v_filepath = f"{dir}/cube3x3solvingdataset.json"
        #t_filepath = "./data/dataset/cube3x3trainingdataset.json"
        #v_filepath = "./data/dataset/cube3x3solvingdataset.json"
        dataval, edc, idc = createTVData(v_filepath)
        print(f"datatraining creating \n")
        datatraining, edc, idc = createTVData(t_filepath) ## 
        print(f"dataval creating \n")
        dataval, edc, idc = createTVData(v_filepath, edc, idc) ## []
        #######tm.TecXModelTrain(stmdlenc, edc.stoi, edc.itos, dataval) ####stmdltensor = torch.tensor(stmdlenc, dtype=torch.long)
        print(f"datatraining len = {len(datatraining)}")
        print(f"datatraining 0= {datatraining[0]}")
        print(f"datatraining 1 = {datatraining[1]}")
        print(f"datatraining 2 = {datatraining[2]}")
        print(f"dataval len = {len(dataval)}")
        print(f"dataval 0 = {dataval[0]}")
        print(f"dataval 1 = {dataval[1]}")
        print(f"dataval 2 = {dataval[2]}")
        tm.TecXModelTrain(datatraining, edc.stoi, edc.itos, dataval)
        #dir_name = ""
        #dir = dir + dir_name
if __name__ == "__main__":
    print(f"Started \n")
    t_data_dir_list = file_dir_nested("./data/dataset")
    v_data_dir_list = file_dir_nested("./data/dataset")
    edc = None
    idc = None
    model = None
    for i in range(max(len(t_data_dir_list),len(v_data_dir_list))):
        t_data_dir = t_data_dir_list[i]
        v_data_dir = v_data_dir_list[i]
        t_filepath = f"{t_data_dir}/cube3x3trainingdataset.json"
        v_filepath = f"{v_data_dir}/cube3x3solvingdataset.json"
        #t_filepath = "./data/dataset/cube3x3trainingdataset.json"
        #v_filepath = "./data/dataset/cube3x3solvingdataset.json"
        print(f"datatraining creating \n")
        datatraining, edc, idc = createTVData(t_filepath, edc, idc) ## 
        print(f"dataval creating \n")
        dataval, edc, idc = createTVData(v_filepath, edc, idc) ## []
        #######tm.TecXModelTrain(stmdlenc, edc.stoi, edc.itos, dataval) ####stmdltensor = torch.tensor(stmdlenc, dtype=torch.long)
        print(f"datatraining len = {len(datatraining)}")
        print(f"datatraining 0= {datatraining[0]}")
        print(f"datatraining 1 = {datatraining[1]}")
        print(f"datatraining 2 = {datatraining[2]}")
        print(f"dataval len = {len(dataval)}")
        print(f"dataval 0 = {dataval[0]}")
        print(f"dataval 1 = {dataval[1]}")
        print(f"dataval 2 = {dataval[2]}")
        tm.TecXModelTrain(datatraining, edc.stoi, edc.itos, dataval, model)
        model = 
        #dir_name = ""
        #dir = dir + dir_name
