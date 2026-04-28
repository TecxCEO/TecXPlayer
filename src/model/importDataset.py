import encoding_decoding as ed #
#from torch.utils.data import Dataset
import json

class ImportDataset():
    #def __init__(self, data, ...):
    def __init__(self, file_path):
        #super.__init__()
        self.file_path = file_path
        with open(self.file_path, 'r') as f:
            data = json.load(f)
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
        """##result = self.get_nested_value(cube)
        Recursively searches for a target_key in a nested dictionary.
        """
        mv=[]
        cst = {}
        amst={}
        # If the current element is a dictionary, look inside
        if isinstance(data, dict):
            for key, value in data.items():
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
                if cst and mv and amvst and key!="state":
                    yield cst, mv, amvst
                if isinstance(value, dict) and len(value)==(16, 19) :
                    # If the value is another dictionary, dive deeper (recursion)
                    yield from get_nested_value(value)
    def createInputString(data):
        #dat = importData()
        #stbm = data[state]
        for key, value in data.items():
            if key == 'state' :
                stbm = data[state]
            elif key != 'state' :
                mv = key
                stam = value[state]
                # stam = data[mv][state]
                #stam = data[key][state]
            yield (stbm, mv, stam)
    def convertStateToList(stbm, mv, stam):
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
    st_data = idc.importData
    print(f" st_data = {st_data}")
    st_mv_data + = idc.createInputString(idc.data["solution"])
    print(f"st_mv_data = {st_mv_data}")
    st_mv_data_list = idc.convertStateToList(st_mv_data[0])
    print(f"st_mv_data_list = {st_mv_data_list}")
