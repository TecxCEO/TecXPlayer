import encoding_decoding as ed #
#from torch.utils.data import Dataset
import json
import copy

class ImportDataset():
    #def __init__(self, data, ...):
    def __init__(self, file_path):
        #super.__init__()
        self.file_path = file_path
        with open(self.file_path, 'r') as f:
            self.data = json.load(f)
            # data = json.load(f)
    def get_my_data(self):
        # Inside the class, use self
        return self.data["solution"]
    def importData(self):
            #cube_data = json.read(f)
            cube=self.data["solution"].copy
            # set default values: 
            cst, mv, amvst = None, None, None
            for result in self.get_nested_value(cube):
                ####cst, mv, amvst = result
                # ... rest of your processing and yield ...
                # Replace your current result check with this:
                ##result = next(self.get_nested_value(cube), None)
                if result is not None:
                    cst, mv, amvst = result
                else:
                    # Handle empty case
                    print(f"No data found for cube")
                    continue
                print(f"result={result}")
                # yield cst, mv, amvst
                yield result
    def get_nested_value(self,data):
        print(f"In the get_nested_value function.\n")
        """##result = self.get_nested_value(cube)
        Recursively searches for a target_key in a nested dictionary.
        """
        mv=[]
        cst = {}
        amst={}
        data_given=data.copy
        print(f" data = {data_given}")
        # If the current element is a dictionary, look inside
        if isinstance(data, dict):
            print(f"In the get_nested_value function, if statement.\n")
            for key, value in data.items():
                print(f"In the get_nested_value function for loop\n")
                if key == "state":
                    #yield value
                    cst=value
                elif len(key) == 3:
                    if len(value)==20 :
                        #yield value
                        mv=key
                        amvst=value
                    elif len(value)== (16, 19):
                        #yield value
                        mv=key
                        amvst=value["state"]
                        if cst and mv and amvst: # and key!="state":
                            print(f" cst ={cst}\n, mv = {mv}\n, amvst = { amvst}")
                            yield cst, mv, amvst
                        if isinstance(value, dict): # and len(value)==(16, 19) :
                            # If the value is another dictionary, dive deeper (recursion)
                            yield from get_nested_value(value)
            print(f"At the end of get_nested_value function.\n")
    def createInputString(self, data):
        #dat = importData()
        #stbm = data[state]
        mv = None
        stam = None
        for key, value in data.items():
            if key == 'state' :
                stbm = data[key]
            elif key != 'state' :
                mv = key
                stam = value['state']
                # stam = data[mv][state]
                #stam = data[key][state]
                yield (stbm, mv, stam)
    def convertStateToList(self, stbm, mv, stam):
        stbml = []
        staml = []
        for value in stbm.values():
            stbml.append(value)
            # stbml+=value
        # staml.append(value) for value in stbm.values()
        staml.extend(stbm.values())
        
        #(staml+=value) for value in stbm.values()
        return (stbml, mv, staml)
if __name__ == "__main__":
    idc = ImportDataset("data/dataset/cube3x3solvingdataset.json")
    # print()
    # st_data = idc.importData.copy()
    # Then you must use it like this:
    st_data = copy.copy(idc.importData)
    print(f" st_data = {vars(st_data)}")
    
    # OR print(st_data.__dict__)

    # This "exhausts" the generator and puts everything into a list
    st_data = list(idc.importData()) 
    # print(f"st_data = {st_data}")
    ##print(f" st_data = {next(st_data)}")
    # RIGHT
    iterator = iter(st_data)
    print(f" st_data = {next(iterator, None)}")
    idc.get_nested_value(idc.data["solution"])
    st_mv_data = []
    st_mv_data += idc.createInputString(idc.data["solution"])
    ###print(f"st_mv_data = {st_mv_data}")
    for l, smd in enumerate(st_mv_data, start = 1):
        print(f"st_mv_data {l} = {smd[0]}\n, {smd[1]}\n, {smd[2]}\n")
        # print(f"st_mv_data {l} = {smd[:3]}\n\n")
        st_mv_data_list = idc.convertStateToList(smd[0], smd[1], smd[2])
        print(f"st_mv_data_list {l} = {st_mv_data_list}\n\n")
